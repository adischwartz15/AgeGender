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
from src.training.trainer import _build_scheduler, resolve_loss_balancing
from src.utils.seed import seed_worker

logger = logging.getLogger(__name__)


class InvalidStageTransitionError(RuntimeError):
    pass


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

    mode, fixed = resolve_loss_balancing(loss_cfg, current_epoch)

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
            scaled_loss = loss_out.total_loss / grad_accumulation_steps
            scaler.scale(scaled_loss).backward()
            is_accumulation_boundary = (batch_idx + 1) % grad_accumulation_steps == 0
            is_last_batch = (batch_idx + 1) == len(loader)
            if is_accumulation_boundary or is_last_batch:
                if grad_clip_norm:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

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
        self.train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers,
            drop_last=len(train_dataset) > batch_size, pin_memory=pin_memory,
            worker_init_fn=seed_worker if num_workers > 0 else None,
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

    def _build_optimizer(self, backbone_lr: float, adapter_lr: float, head_lr: float, balance_lr: float):
        weight_decay = self.training_cfg.get("weight_decay", 0.05)
        groups = self.model.get_parameter_groups(backbone_lr, adapter_lr, head_lr, balance_lr, weight_decay)
        return torch.optim.AdamW(groups)

    def _run_stage(self, stage_name: str, epochs: int, backbone_lr: float, adapter_lr: float, head_lr: float,
                    balance_lr: float, warmup_epochs: int, early_stopping_patience: int, global_epoch: int) -> int:
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.model.parameters())
        logger.info(
            "[%s] === %s (epochs=%d) === trainable_params=%s/%s",
            self.experiment_name, stage_name, epochs, f"{trainable:,}", f"{total:,}",
        )

        if self.device == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)

        optimizer = self._build_optimizer(backbone_lr, adapter_lr, head_lr, balance_lr)
        scheduler = _build_scheduler(optimizer, epochs, warmup_epochs)
        early_stopping = EarlyStopping(patience=early_stopping_patience, mode="min")
        loss_cfg = self.config["model"]["loss_balancing"]
        age_max = self.config["model"]["age_head"].get("age_max", 120)

        for _ in range(epochs):
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
            scheduler.step()
            elapsed = time.time() - start
            global_epoch += 1

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

            if val_metrics["loss"] == val_metrics["loss"] and early_stopping.step(val_metrics["loss"]):
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

    def run_stage_1(self) -> int:
        """Stage 1: freeze the backbone, train adapters/heads/balancing only.
        Public (not just called from train()) so a notebook can show Stage 1
        and Stage 2 as separate cells."""
        backbone_lr, adapter_lr, head_lr, balance_lr, warmup_epochs, patience = self._stage_lrs()
        head_only_epochs = self.training_cfg.get("head_only_epochs", 3)
        logger.info(
            "[%s] Starting Stage 1 | device=%s | train_samples=%d | val_samples=%d",
            self.experiment_name, self.device, len(self.train_loader.dataset), len(self.val_loader.dataset),
        )
        self.model.freeze_backbone()
        self._global_epoch = self._run_stage(
            "Stage 1: frozen backbone", head_only_epochs, backbone_lr, adapter_lr, head_lr, balance_lr,
            warmup_epochs, patience, global_epoch=getattr(self, "_global_epoch", 0),
        )
        return self._global_epoch

    def run_stage_2(self) -> int:
        """Stage 2: unfreeze (full or last-N-stages, per config) and fine-tune
        at a low backbone LR / higher adapter+head+balance LR. Must be
        called after run_stage_1()."""
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
            "Stage 2: fine-tune", finetune_epochs, backbone_lr, adapter_lr, head_lr, balance_lr,
            warmup_epochs, patience, global_epoch=getattr(self, "_global_epoch", 0),
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
