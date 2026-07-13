"""Two-stage training loop for the supplementary VOLO transfer-learning experiment.

Not ``src/training/trainer.py::Trainer``: that class's Stage A/B/C
progressive-freezing plan and 2-group differential-LR optimizer are
specific to the from-scratch backbones (see that module's own docstring --
staged freezing is only "scientifically meaningful" there when the backbone
comes from this repo's own SimCLR pretraining) and don't fit what this
experiment needs: a 2-stage schedule (frozen backbone -> fine-tune) with 4
separate parameter groups (backbone / adapters / heads / loss-balancing),
gradient accumulation, and peak-GPU-memory tracking. Rather than stretch
``Trainer``'s stage/optimizer machinery to a shape it wasn't designed for,
this module reuses everything about it that genuinely is generic --
``compute_multitask_loss``, ``resolve_loss_balancing``, ``BestMetricTracker``,
``save_checkpoint``, ``EarlyStopping``, ``seed_worker``, and ``Trainer``'s own
warmup+cosine ``_build_scheduler`` -- by importing them directly, instead of
re-deriving any of that logic.

``src/training/trainer.py`` and ``src/training/stages.py`` are not modified
by this module or anything it imports.
"""

from __future__ import annotations

import contextlib
import copy
import json
import logging
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.losses.multitask_loss import compute_multitask_loss
from src.models.pretrained_volo import PretrainedVOLOFaceOnlyMultiTask
from src.training.callbacks import EarlyStopping
from src.training.checkpointing import BestMetricTracker, save_checkpoint
from src.training.persistent_artifacts import PersistentArtifactManager, capture_rng_state, restore_rng_state
from src.training.trainer import _build_scheduler, resolve_loss_balancing
from src.utils.seed import seed_worker

logger = logging.getLogger(__name__)


class InvalidStageTransitionError(RuntimeError):
    pass


STAGE_1_NAME = "Stage 1: frozen backbone"
STAGE_2_NAME = "Stage 2: fine-tune"


def _run_epoch(
    model: PretrainedVOLOFaceOnlyMultiTask,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    device: str,
    mixed_precision: bool,
    scaler: torch.amp.GradScaler,
    grad_clip_norm: float,
    grad_accumulation_steps: int,
    loss_cfg: dict,
    current_epoch: int,
    gender_class_weights: torch.Tensor | None,
    confidence_threshold: float,
    max_batches: int | None = None,
) -> dict[str, float]:
    """One training or validation epoch. Mirrors Trainer._run_batches's metric
    surface, plus gradient accumulation (Trainer's loop doesn't need this --
    VOLO-D1 at 224px needs a small per-step batch size)."""
    is_train = optimizer is not None
    model.train(is_train)

    total_loss, total_age_loss, total_gender_loss = 0.0, 0.0, 0.0
    n_age_batches, n_gender_batches, n_batches = 0, 0, 0
    age_abs_errors = []
    gender_correct, gender_total = 0, 0
    optimizer_steps = 0
    skipped_optimizer_steps = 0
    microbatches_processed = 0
    microbatches_in_group = 0

    mode, fixed = resolve_loss_balancing(loss_cfg, current_epoch)

    # The number of batches this epoch will actually process -- which is
    # min(len(loader), max_batches) when max_train_batches_per_epoch caps the
    # epoch early. The final *effective* batch (not necessarily len(loader)-1)
    # must trigger an optimizer step so its partial accumulation group is not
    # silently dropped.
    effective_num_batches = len(loader)
    if max_batches is not None:
        effective_num_batches = min(effective_num_batches, max_batches)

    if is_train:
        optimizer.zero_grad()

    for batch_idx, batch in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break
        images = batch["image"].to(device)
        age_target = batch["age"].to(device)
        age_mask = batch["age_mask"].to(device)
        gender_target = batch["gender_label"].to(device)
        gender_mask = batch["gender_mask"].to(device)

        autocast_ctx = torch.autocast(device_type="cuda") if mixed_precision else contextlib.nullcontext()
        with torch.set_grad_enabled(is_train):
            with autocast_ctx:
                outputs = model(images)
                loss_out = compute_multitask_loss(
                    outputs["age_output"], outputs["gender_logits"], age_target, age_mask,
                    gender_target, gender_mask, mode=mode,
                    fixed_age_weight=fixed.get("age_weight", 1.0),
                    fixed_gender_weight=fixed.get("gender_weight", 1.0),
                    log_var_age=model.log_var_age, log_var_gender=model.log_var_gender,
                    gender_class_weights=gender_class_weights,
                )

        if is_train:
            # Scale each microbatch loss by 1/grad_accumulation_steps so a
            # full accumulation window averages the group's gradients. A
            # partial final window (fewer than grad_accumulation_steps
            # microbatches) is corrected below by rescaling to divide by the
            # actual microbatch count, so it is never underweighted.
            scaler.scale(loss_out.total_loss / grad_accumulation_steps).backward()
            microbatches_processed += 1
            microbatches_in_group += 1
            is_accumulation_boundary = microbatches_in_group == grad_accumulation_steps
            is_last_effective_batch = (batch_idx + 1) == effective_num_batches
            if is_accumulation_boundary or is_last_effective_batch:
                # Unscale once (needed for both the partial-window correction
                # and gradient clipping); scaler.step() then won't re-unscale.
                scaler.unscale_(optimizer)
                if microbatches_in_group != grad_accumulation_steps:
                    correction = grad_accumulation_steps / microbatches_in_group
                    for group in optimizer.param_groups:
                        for p in group["params"]:
                            if p.grad is not None:
                                p.grad.mul_(correction)
                if grad_clip_norm:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
                scale_before_step = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                microbatches_in_group = 0
                # GradScaler skips optimizer.step() (only shrinking the scale)
                # on a non-finite gradient. Count a real step only when the
                # scale did not shrink, so global step counts / the scheduler
                # never advance on an AMP-skipped step.
                if scaler.get_scale() >= scale_before_step:
                    optimizer_steps += 1
                else:
                    skipped_optimizer_steps += 1

        total_loss += loss_out.total_loss.item()
        n_batches += 1
        if loss_out.age_loss is not None:
            total_age_loss += loss_out.age_loss.item()
            n_age_batches += 1
            with torch.no_grad():
                valid = age_mask.bool()
                if valid.any():
                    err = (outputs["age_output"]["q50"][valid] - age_target[valid]).abs()
                    age_abs_errors.append(err.detach().cpu())
        if loss_out.gender_loss is not None:
            total_gender_loss += loss_out.gender_loss.item()
            n_gender_batches += 1
            with torch.no_grad():
                valid = gender_mask.bool()
                if valid.any():
                    preds = torch.softmax(outputs["gender_logits"][valid], dim=-1).argmax(dim=-1)
                    correct = preds == gender_target[valid]
                    gender_correct += correct.sum().item()
                    gender_total += int(valid.sum().item())

    return {
        "loss": total_loss / max(1, n_batches),
        "age_loss": total_age_loss / max(1, n_age_batches),
        "gender_loss": total_gender_loss / max(1, n_gender_batches),
        "age_mae": float(torch.cat(age_abs_errors).mean()) if age_abs_errors else float("nan"),
        "gender_accuracy": gender_correct / max(1, gender_total) if gender_total else float("nan"),
        "_optimizer_steps": optimizer_steps,
        "_skipped_optimizer_steps": skipped_optimizer_steps,
        "_microbatches_processed": microbatches_processed,
        "_effective_batches": effective_num_batches if is_train else n_batches,
        "_any_optimizer_step": optimizer_steps > 0,
    }


def _balanced_score(age_mae: float, gender_acc: float, age_max: float) -> float:
    if age_mae != age_mae:
        return gender_acc if gender_acc == gender_acc else float("-inf")
    if gender_acc != gender_acc:
        return -age_mae
    return gender_acc - age_mae / max(age_max, 1e-6)


class TransferTrainer:
    """Runs Stage 1 (frozen backbone) then Stage 2 (fine-tune) for
    :class:`~src.models.pretrained_volo.PretrainedVOLOFaceOnlyMultiTask`."""

    def __init__(
        self,
        model: PretrainedVOLOFaceOnlyMultiTask,
        config: dict,
        train_dataset,
        val_dataset,
        device: str = "cpu",
        checkpoint_dir: str | Path = "./checkpoints/transfer_learning",
        experiment_name: str = "volo_d1_face_only_pretrained",
        gender_class_weights: torch.Tensor | None = None,
        output_dir: str | Path | None = None,
        artifact_manager: PersistentArtifactManager | None = None,
        resume_state: dict | None = None,
        git_commit_sha: str | None = None,
        split_sha256: str | None = None,
    ) -> None:
        self.model = model.to(device)
        self.config = config
        self.device = device
        self.training_cfg = config["training"]
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.experiment_name = experiment_name
        self.gender_class_weights = gender_class_weights.to(device) if gender_class_weights is not None else None
        self.confidence_threshold = config["model"]["gender_head"].get("confidence_threshold", 0.80)

        self.output_dir = Path(output_dir) if output_dir is not None else self.checkpoint_dir.parent
        self.metrics_dir = self.output_dir / "metrics"
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        self.history_path = self.metrics_dir / f"{experiment_name}_history.json"

        if device == "cuda" and self.training_cfg.get("mixed_precision", True) is False:
            logger.info("mixed_precision explicitly disabled on CUDA for '%s'.", experiment_name)
        if device != "cuda" and self.training_cfg.get("mixed_precision", True):
            logger.warning("mixed_precision requested but device='%s' (not cuda) -- AMP is CUDA-only, disabling.", device)
        self.mixed_precision = self.training_cfg.get("mixed_precision", True) and device == "cuda"
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.mixed_precision)
        self.grad_clip_norm = self.training_cfg.get("grad_clip_norm", 1.0)
        self.grad_accumulation_steps = max(1, self.training_cfg.get("grad_accumulation_steps", 1))

        batch_size = self.training_cfg.get("batch_size", 16)
        num_workers = self.training_cfg.get("num_workers", 2)
        pin_memory = device == "cuda"
        # Explicit seeded generator for reproducible shuffled-batch ordering
        # (see src/training/trainer.py for the same treatment in the core
        # trainer).
        self.dataloader_seed = int(self.training_cfg.get("seed", config.get("seed", 42)))
        self.train_generator = torch.Generator()
        self.train_generator.manual_seed(self.dataloader_seed)
        self.train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers,
            drop_last=len(train_dataset) > batch_size, pin_memory=pin_memory,
            worker_init_fn=seed_worker if num_workers > 0 else None,
            generator=self.train_generator,
        )
        self.val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
            pin_memory=pin_memory, worker_init_fn=seed_worker if num_workers > 0 else None,
        )
        self.max_train_batches = self.training_cfg.get("max_train_batches_per_epoch")
        self.max_val_batches = self.training_cfg.get("max_val_batches_per_epoch")

        self.trackers = {
            "age_mae": BestMetricTracker(mode="min"),
            "gender_accuracy": BestMetricTracker(mode="max"),
            "balanced_score": BestMetricTracker(mode="max"),
        }
        self.history: dict[str, list] = {
            "stage": [], "train_loss": [], "val_loss": [], "val_age_mae": [], "val_gender_accuracy": [],
            "lr_backbone": [], "lr_adapters": [], "lr_heads": [], "lr_balance": [], "epoch_time_seconds": [],
        }
        self._best_state_dict: dict | None = None
        self._best_balanced_score: float | None = None

        self.artifact_manager = artifact_manager
        self.git_commit_sha = git_commit_sha
        self.split_sha256 = split_sha256
        self._global_epoch = 0
        self._global_step = 0
        self._resume_stage: str | None = None
        self._resume_stage_epoch = 0
        self._resume_optimizer_state: dict | None = None
        self._resume_scheduler_state: dict | None = None
        self._resume_scaler_state: dict | None = None
        self._resume_early_stopping_state: dict | None = None
        self._current_early_stopping_state: dict = {"best_value": None, "num_bad_epochs": 0, "should_stop": False}
        if resume_state is not None:
            self._apply_resume_state(resume_state)

    def _apply_resume_state(self, state: dict) -> None:
        """Restore in-memory trainer state from a checkpoint payload produced
        by :meth:`_build_checkpoint_payload` (i.e. ``last.pt``/``previous_last.pt``
        as loaded by ``PersistentArtifactManager.find_latest_valid_checkpoint``).

        Distinguishes Stage 1 vs. Stage 2 resume via ``training_stage`` --
        restoring a Stage 1 checkpoint must resume Stage 1 (never skip to
        Stage 2), and restoring a Stage 2 checkpoint must never re-run
        Stage 1 (see ``run_stage_1``/``run_stage_2`` below, which consume
        ``self._resume_stage``)."""
        if state.get("model_state_dict") is not None:
            self.model.load_state_dict(state["model_state_dict"])
            self._best_state_dict = copy.deepcopy(state["model_state_dict"])
        self.history = state.get("training_history") or self.history
        self._global_epoch = state.get("epoch", 0) or 0
        self._global_step = state.get("global_step", 0) or 0
        self._resume_stage = state.get("training_stage")
        self._resume_stage_epoch = state.get("stage_epoch", 0) or 0
        self._best_balanced_score = state.get("best_validation_metric")
        if self._best_balanced_score is not None:
            self.trackers["balanced_score"].best_value = self._best_balanced_score
        early_stopping_state = state.get("early_stopping_state") or {}
        self._resume_early_stopping_state = early_stopping_state or None
        self._resume_optimizer_state = state.get("optimizer_state_dict")
        self._resume_scheduler_state = state.get("scheduler_state_dict")
        scaler_state = state.get("gradient_scaler_state_dict")
        self._resume_scaler_state = scaler_state
        if scaler_state is not None:
            self.scaler.load_state_dict(scaler_state)
        rng_state = state.get("rng_state")
        if rng_state:
            restore_rng_state(rng_state)
        logger.info(
            "[%s] Resuming from checkpoint: global_epoch=%d training_stage=%r stage_epoch=%d",
            self.experiment_name, self._global_epoch, self._resume_stage, self._resume_stage_epoch,
        )

    def _build_checkpoint_payload(
        self, stage_name: str, stage_epoch: int, optimizer: torch.optim.Optimizer | None,
        scheduler, metrics: dict,
    ) -> dict:
        """Everything needed to resume training byte-for-byte from this
        point -- see ``docs/transfer_learning.md`` "Persistent artifacts"
        for the full field list this must (and does) contain."""
        model_cfg = self.config.get("model", {})
        log_var_age = getattr(self.model, "log_var_age", None)
        log_var_gender = getattr(self.model, "log_var_gender", None)
        return {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
            "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
            "gradient_scaler_state_dict": self.scaler.state_dict(),
            "epoch": self._global_epoch,
            "stage_epoch": stage_epoch,
            "global_step": self._global_step,
            "training_stage": stage_name,
            "best_validation_metric": self._best_balanced_score,
            "early_stopping_state": self._current_early_stopping_state,
            "training_history": self.history,
            "seed": self.training_cfg.get("seed"),
            "rng_state": capture_rng_state(),
            "model_id": self.model.model_id,
            "pretrained_source": self.model.pretrained_source,
            "input_size": self.model.input_size,
            "transform_config": {
                "mean": list(self.model.data_config.get("mean", [])),
                "std": list(self.model.data_config.get("std", [])),
                "interpolation": self.model.interpolation_name,
                "input_size": self.model.input_size,
            },
            "split_sha256": self.split_sha256,
            "age_head_config": model_cfg.get("age_head", {}),
            "gender_head_config": model_cfg.get("gender_head", {}),
            "adapter_config": model_cfg.get("adapters", {}),
            "loss_balancing_params": {
                "log_var_age": float(log_var_age.detach().cpu()) if log_var_age is not None else None,
                "log_var_gender": float(log_var_gender.detach().cpu()) if log_var_gender is not None else None,
            },
            "optimizer_group_lrs": [g["lr"] for g in optimizer.param_groups] if optimizer is not None else None,
            "git_commit_sha": self.git_commit_sha,
            "config": self.config,
            "metrics": metrics,
            "extra": {"family": "pretrained_volo", "model_id": self.model.model_id,
                      "pretrained_source": self.model.pretrained_source},
        }

    def _build_optimizer(self, backbone_lr: float, adapter_lr: float, head_lr: float, balance_lr: float):
        weight_decay = self.training_cfg.get("weight_decay", 0.05)
        groups = self.model.get_parameter_groups(backbone_lr, adapter_lr, head_lr, balance_lr, weight_decay)
        return torch.optim.AdamW(groups)

    def _run_stage(
        self, stage_name: str, epochs: int, backbone_lr: float, adapter_lr: float, head_lr: float,
        balance_lr: float, warmup_epochs: int, early_stopping_patience: int, global_epoch: int,
        resume_epoch_in_stage: int = 0, resume_optimizer_state: dict | None = None,
        resume_scheduler_state: dict | None = None, resume_scaler_state: dict | None = None,
        resume_early_stopping_state: dict | None = None,
    ) -> int:
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.model.parameters())
        logger.info(
            "[%s] === %s (epochs=%d, resuming_from_stage_epoch=%d) === trainable_params=%s/%s",
            self.experiment_name, stage_name, epochs, resume_epoch_in_stage, f"{trainable:,}", f"{total:,}",
        )

        if self.device == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)

        optimizer = self._build_optimizer(backbone_lr, adapter_lr, head_lr, balance_lr)
        warmup_start_factor = self.training_cfg.get("scheduler", {}).get("warmup_start_factor", 0.1)
        scheduler = _build_scheduler(optimizer, epochs, warmup_epochs, warmup_start_factor)
        # Early stopping tracks the SAME balanced selection score (higher is
        # better) that checkpoint selection uses -- not validation total loss
        # -- so the best checkpoint and the stopping decision never diverge.
        early_stopping = EarlyStopping(patience=early_stopping_patience, mode="max")
        if resume_optimizer_state is not None:
            optimizer.load_state_dict(resume_optimizer_state)
        if resume_scheduler_state is not None:
            scheduler.load_state_dict(resume_scheduler_state)
        if resume_scaler_state is not None:
            self.scaler.load_state_dict(resume_scaler_state)
        if resume_early_stopping_state:
            early_stopping.best_value = resume_early_stopping_state.get("best_value")
            early_stopping.num_bad_epochs = resume_early_stopping_state.get("num_bad_epochs", 0)
            early_stopping.should_stop = resume_early_stopping_state.get("should_stop", False)
        loss_cfg = self.config["model"]["loss_balancing"]
        age_max = self.config["model"]["age_head"].get("age_max", 120)

        for epoch_in_stage in range(resume_epoch_in_stage, epochs):
            start = time.time()
            train_metrics = _run_epoch(
                self.model, self.train_loader, optimizer, self.device, self.mixed_precision, self.scaler,
                self.grad_clip_norm, self.grad_accumulation_steps, loss_cfg, global_epoch + 1,
                self.gender_class_weights, self.confidence_threshold, self.max_train_batches,
            )
            val_metrics = _run_epoch(
                self.model, self.val_loader, None, self.device, self.mixed_precision, self.scaler,
                self.grad_clip_norm, self.grad_accumulation_steps, loss_cfg, global_epoch + 1,
                self.gender_class_weights, self.confidence_threshold, self.max_val_batches,
            )
            # Only advance the (epoch-based) LR scheduler when the optimizer
            # actually stepped this epoch -- if AMP skipped every step on
            # non-finite gradients, stepping the scheduler would run it ahead
            # of the optimizer (torch raises "lr_scheduler.step() before
            # optimizer.step()"). Mirrors the core Trainer's behaviour.
            if train_metrics.get("_any_optimizer_step", True):
                scheduler.step()
            if train_metrics.get("_skipped_optimizer_steps"):
                logger.warning(
                    "[%s] %s epoch %d: %d optimizer step(s) skipped by GradScaler (non-finite gradients).",
                    self.experiment_name, stage_name, global_epoch + 1,
                    train_metrics["_skipped_optimizer_steps"],
                )
            elapsed = time.time() - start
            global_epoch += 1
            self._global_epoch = global_epoch
            self._global_step += train_metrics.get("_optimizer_steps", 0)

            # get_parameter_groups() omits a group entirely when it has no
            # trainable params (e.g. "backbone" while Stage 1 has it
            # frozen), so record the LR actually *requested* for each
            # component this epoch (None when that component wasn't
            # trainable at all this stage) rather than trying to
            # reverse-match against optimizer.param_groups' insertion order.
            backbone_has_trainable = any(p.requires_grad for p in self.model.backbone.parameters())
            self.history["stage"].append(stage_name)
            self.history["train_loss"].append(train_metrics["loss"])
            self.history["val_loss"].append(val_metrics["loss"])
            self.history["val_age_mae"].append(val_metrics["age_mae"])
            self.history["val_gender_accuracy"].append(val_metrics["gender_accuracy"])
            self.history["epoch_time_seconds"].append(elapsed)
            self.history["lr_backbone"].append(backbone_lr if backbone_has_trainable else None)
            self.history["lr_adapters"].append(adapter_lr)
            self.history["lr_heads"].append(head_lr)
            self.history["lr_balance"].append(balance_lr if self.model.log_var_age is not None else None)

            balanced = _balanced_score(val_metrics["age_mae"], val_metrics["gender_accuracy"], age_max)
            self._maybe_checkpoint("age_mae", val_metrics["age_mae"], global_epoch, val_metrics)
            self._maybe_checkpoint("gender_accuracy", val_metrics["gender_accuracy"], global_epoch, val_metrics)
            is_best = self._maybe_checkpoint("balanced_score", balanced, global_epoch, val_metrics)
            if is_best:
                self._best_state_dict = copy.deepcopy(self.model.state_dict())
                self._best_balanced_score = balanced

            logger.info(
                "[%s] %s epoch %d | %.1fs | train_loss=%.4f val_loss=%.4f val_age_mae=%.3f "
                "val_gender_acc=%.3f | balanced_score=%.4f | best=%s",
                self.experiment_name, stage_name, global_epoch, elapsed, train_metrics["loss"],
                val_metrics["loss"], val_metrics["age_mae"], val_metrics["gender_accuracy"], balanced,
                "yes" if is_best else "no",
            )

            should_stop = balanced == balanced and early_stopping.step(balanced)
            self._current_early_stopping_state = {
                "best_value": early_stopping.best_value, "num_bad_epochs": early_stopping.num_bad_epochs,
                "should_stop": early_stopping.should_stop,
            }

            if self.artifact_manager is not None:
                payload = self._build_checkpoint_payload(stage_name, epoch_in_stage + 1, optimizer, scheduler, val_metrics)
                trainer_state = {
                    "seed": self.training_cfg.get("seed"), "global_epoch": self._global_epoch,
                    "stage_epoch": epoch_in_stage + 1, "training_stage": stage_name,
                    "best_validation_metric": self._best_balanced_score,
                    "git_commit_sha": self.git_commit_sha, "model_id": self.model.model_id,
                    "pretrained_source": self.model.pretrained_source, "split_sha256": self.split_sha256,
                }
                self.artifact_manager.on_epoch_end(payload, self.history, trainer_state, is_best)
                if is_best:
                    self.artifact_manager.on_new_best(payload)

            if should_stop:
                logger.info("[%s] Early stopping triggered in %s at epoch %d", self.experiment_name, stage_name, global_epoch)
                break

        if self.device == "cuda":
            peak_bytes = torch.cuda.max_memory_allocated(self.device)
            logger.info("[%s] %s peak CUDA memory: %.1f MiB", self.experiment_name, stage_name, peak_bytes / 2**20)

        return global_epoch

    def _maybe_checkpoint(self, metric_name: str, value: float, epoch: int, metrics: dict) -> bool:
        if value != value:
            return False
        tracker = self.trackers[metric_name]
        improved = tracker.update(value)
        if improved:
            path = self.checkpoint_dir / f"{self.experiment_name}_best_{metric_name}.pt"
            save_checkpoint(
                path, self.model, None, epoch, metrics, self.config,
                extra={"family": "pretrained_volo", "model_id": self.model.model_id,
                       "pretrained_source": self.model.pretrained_source},
            )
        return improved

    def _stage_lrs(self) -> tuple[float, float, float, float, int, int]:
        cfg = self.training_cfg
        return (
            cfg.get("backbone_lr", 3.0e-5), cfg.get("adapter_lr", 3.0e-4), cfg.get("head_lr", 3.0e-4),
            cfg.get("loss_balance_lr", 3.0e-4), cfg.get("scheduler", {}).get("warmup_epochs", 1),
            cfg.get("early_stopping_patience", 12),
        )

    def _consume_resume_state_for(self, stage_name: str) -> tuple[int, dict | None, dict | None, dict | None, dict | None]:
        """Pop and clear any pending resume state, returning it only if it
        matches ``stage_name`` -- consumed exactly once so a later stage
        never accidentally re-applies an earlier stage's optimizer/scheduler
        state (which would have incompatible parameter-group shapes)."""
        stage_resume, stage_epoch_resume = self._resume_stage, self._resume_stage_epoch
        opt_state, sched_state = self._resume_optimizer_state, self._resume_scheduler_state
        scaler_state, es_state = self._resume_scaler_state, self._resume_early_stopping_state
        self._resume_stage = None
        self._resume_optimizer_state = self._resume_scheduler_state = None
        self._resume_scaler_state = self._resume_early_stopping_state = None

        if stage_resume != stage_name:
            return 0, None, None, None, None
        resume_epoch_in_stage = stage_epoch_resume
        if resume_epoch_in_stage <= 0:
            return 0, None, None, None, None
        return resume_epoch_in_stage, opt_state, sched_state, scaler_state, es_state

    def run_stage_1(self) -> int:
        """Stage 1: freeze the backbone, train adapters/heads/balancing only.
        Public (not just called from train()) so a notebook can show Stage 1
        and Stage 2 as separate cells. If resume state says training already
        reached Stage 2, this is a no-op (never re-runs Stage 1)."""
        if self._resume_stage is not None and self._resume_stage != STAGE_1_NAME:
            logger.info(
                "[%s] Resume state is already past Stage 1 (training_stage=%r) -- skipping Stage 1.",
                self.experiment_name, self._resume_stage,
            )
            self._resume_stage = None
            self._resume_optimizer_state = self._resume_scheduler_state = None
            self._resume_scaler_state = self._resume_early_stopping_state = None
            return self._global_epoch

        resume_epoch, opt_state, sched_state, scaler_state, es_state = self._consume_resume_state_for(STAGE_1_NAME)
        backbone_lr, adapter_lr, head_lr, balance_lr, warmup_epochs, patience = self._stage_lrs()
        head_only_epochs = self.training_cfg.get("head_only_epochs", 3)
        logger.info(
            "[%s] Starting Stage 1 | device=%s | train_samples=%d | val_samples=%d",
            self.experiment_name, self.device, len(self.train_loader.dataset), len(self.val_loader.dataset),
        )
        self.model.freeze_backbone()
        self._global_epoch = self._run_stage(
            STAGE_1_NAME, head_only_epochs, backbone_lr, adapter_lr, head_lr, balance_lr,
            warmup_epochs, patience, global_epoch=self._global_epoch,
            resume_epoch_in_stage=resume_epoch, resume_optimizer_state=opt_state,
            resume_scheduler_state=sched_state, resume_scaler_state=scaler_state,
            resume_early_stopping_state=es_state,
        )
        if self.artifact_manager is not None:
            transition_payload = self._build_checkpoint_payload(STAGE_2_NAME, 0, None, None, {})
            trainer_state = {
                "seed": self.training_cfg.get("seed"), "global_epoch": self._global_epoch, "stage_epoch": 0,
                "training_stage": STAGE_2_NAME, "best_validation_metric": self._best_balanced_score,
                "git_commit_sha": self.git_commit_sha, "model_id": self.model.model_id,
                "pretrained_source": self.model.pretrained_source, "split_sha256": self.split_sha256,
            }
            self.artifact_manager.on_stage_transition(transition_payload, trainer_state)
        return self._global_epoch

    def run_stage_2(self) -> int:
        """Stage 2: unfreeze (full or last-N-stages, per config) and fine-tune
        at a low backbone LR / higher adapter+head+balance LR. Must be
        called after run_stage_1()."""
        resume_epoch, opt_state, sched_state, scaler_state, es_state = self._consume_resume_state_for(STAGE_2_NAME)
        backbone_lr, adapter_lr, head_lr, balance_lr, warmup_epochs, patience = self._stage_lrs()
        finetune_epochs = self.training_cfg.get("finetune_epochs", 20)
        finetune_unfreeze = self.training_cfg.get("finetune_unfreeze", "full")

        if finetune_unfreeze == "full":
            self.model.unfreeze_backbone()
        elif isinstance(finetune_unfreeze, dict) and "last_n_stages" in finetune_unfreeze:
            self.model.unfreeze_last_stages(int(finetune_unfreeze["last_n_stages"]))
        else:
            raise InvalidStageTransitionError(
                f"training.finetune_unfreeze={finetune_unfreeze!r} must be 'full' or "
                "{'last_n_stages': N}."
            )
        self._global_epoch = self._run_stage(
            STAGE_2_NAME, finetune_epochs, backbone_lr, adapter_lr, head_lr, balance_lr,
            warmup_epochs, patience, global_epoch=self._global_epoch,
            resume_epoch_in_stage=resume_epoch, resume_optimizer_state=opt_state,
            resume_scheduler_state=sched_state, resume_scaler_state=scaler_state,
            resume_early_stopping_state=es_state,
        )
        return self._global_epoch

    def restore_best_checkpoint(self) -> None:
        """Reload the highest-balanced-score state dict seen across both
        stages, so the in-memory model matches what evaluate.py will load
        from {experiment_name}_best_balanced_score.pt. Also flushes history.json."""
        if self._best_state_dict is not None:
            self.model.load_state_dict(self._best_state_dict)
            logger.info(
                "[%s] Restored best checkpoint (balanced_score=%.4f)",
                self.experiment_name, self._best_balanced_score,
            )
        with open(self.history_path, "w", encoding="utf-8") as fh:
            json.dump(self.history, fh, indent=2)

    def train(self) -> dict:
        """Convenience wrapper: run_stage_1() -> run_stage_2() -> restore_best_checkpoint()."""
        self.run_stage_1()
        self.run_stage_2()
        self.restore_best_checkpoint()
        return {"history": self.history, "total_epochs": self._global_epoch}
