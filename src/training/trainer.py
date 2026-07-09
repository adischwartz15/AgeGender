"""Progressive multi-stage trainer for the multi-task face model."""

from __future__ import annotations

import contextlib
import csv
import datetime
import json
import logging
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.losses.multitask_loss import compute_multitask_loss
from src.models.multitask_model import MultiTaskFaceModel
from src.training.callbacks import EarlyStopping
from src.training.checkpointing import BestMetricTracker, save_checkpoint
from src.training.stages import Stage, build_stage_plan
from src.utils.seed import seed_worker

logger = logging.getLogger(__name__)


def _build_optimizer(model: MultiTaskFaceModel, lr: float, weight_decay: float) -> torch.optim.Optimizer:
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    return torch.optim.AdamW(trainable_params, lr=lr, weight_decay=weight_decay)


def _build_scheduler(optimizer: torch.optim.Optimizer, total_epochs: int, warmup_epochs: int):
    def lr_lambda(epoch: int) -> float:
        if epoch < warmup_epochs:
            return (epoch + 1) / max(1, warmup_epochs)
        progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
        import math

        return 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


class Trainer:
    """Runs the Stage A/B/C (or warm-up) progressive training loop.

    Saves three separate "best" checkpoints during training: lowest
    validation age MAE, highest validation gender-label accuracy, and best
    balanced multi-task score (``gender_accuracy - normalized_age_mae``).
    """

    def __init__(
        self,
        model: MultiTaskFaceModel,
        config: dict,
        train_dataset,
        val_dataset,
        device: str = "cpu",
        checkpoint_dir: str | Path = "./checkpoints",
        experiment_name: str = "multitask",
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

        # Incremental per-epoch artifacts (history.csv/json, a live status
        # file) default to living alongside the checkpoint directory unless
        # an explicit output_dir is given -- this is what lets a notebook
        # recover training progress after a session interruption without
        # waiting for train() to return.
        self.output_dir = Path(output_dir) if output_dir is not None else self.checkpoint_dir.parent
        self.metrics_dir = self.output_dir / "metrics"
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir = self.output_dir / "logs"
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.history_csv_path = self.metrics_dir / f"{experiment_name}_history.csv"
        self.history_json_path = self.metrics_dir / f"{experiment_name}_history.json"
        self.status_path = self.log_dir / f"{experiment_name}_status.json"

        self.train_dataset_size = len(train_dataset)
        self.val_dataset_size = len(val_dataset)

        batch_size = self.training_cfg.get("batch_size", 64)
        num_workers = self.training_cfg.get("num_workers", 2)
        # pin_memory speeds up host->device transfer, but only actually helps
        # (and is only supported) when transferring to a CUDA device.
        # worker_init_fn=seed_worker gives each DataLoader worker process its
        # own deterministic-but-distinct RNG state, so augmentation
        # randomness is reproducible across runs even with num_workers > 0
        # (without it, workers can otherwise end up sharing correlated RNG
        # state inherited from the parent process).
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

        self.mixed_precision = self.training_cfg.get("mixed_precision", True) and device == "cuda"
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.mixed_precision)
        self.grad_clip_norm = self.training_cfg.get("grad_clip_norm", 1.0)

        # Optional hard caps on batches-per-epoch (config-driven, default
        # None = unlimited). Distinct from epoch count: a "smoke test" that
        # caps epochs to 1 still iterates the *entire* dataset once, which
        # can be slow on a large dataset -- these let a fast integration
        # check also cap batches-per-epoch, without affecting any real
        # training run that doesn't set them.
        self.max_train_batches = self.training_cfg.get("max_train_batches_per_epoch")
        self.max_val_batches = self.training_cfg.get("max_val_batches_per_epoch")

        self.history: dict[str, list[float]] = {
            "train_loss": [], "val_loss": [], "val_age_mae": [], "val_age_rmse": [], "val_gender_accuracy": [],
            "val_gender_selective_accuracy": [], "val_gender_coverage": [], "val_gender_abstention": [],
            "age_loss": [], "gender_loss": [], "effective_age_weight": [], "effective_gender_weight": [],
            "log_var_age": [], "log_var_gender": [], "lr": [], "epoch_time_seconds": [],
        }
        self.epoch_times: list[float] = []

        self.trackers = {
            "age_mae": BestMetricTracker(mode="min"),
            "gender_accuracy": BestMetricTracker(mode="max"),
            "balanced_score": BestMetricTracker(mode="max"),
        }

    def _loss_mode(self) -> str:
        return self.config["model"]["loss_balancing"]["mode"]

    def _run_batches(self, loader: DataLoader, optimizer: torch.optim.Optimizer | None) -> dict[str, float]:
        is_train = optimizer is not None
        self.model.train(is_train)

        total_loss, total_age_loss, total_gender_loss = 0.0, 0.0, 0.0
        n_age_batches, n_gender_batches, n_batches = 0, 0, 0
        eff_age_w, eff_gender_w, lv_age, lv_gender = 0.0, 0.0, 0.0, 0.0
        age_abs_errors = []
        gender_correct, gender_total = 0, 0
        gender_correct_accepted, gender_total_accepted = 0, 0
        gender_confidences = []

        loss_cfg = self.config["model"]["loss_balancing"]
        mode = loss_cfg["mode"]
        fixed = loss_cfg.get("fixed", {"age_weight": 1.0, "gender_weight": 1.0})
        max_batches = self.max_train_batches if is_train else self.max_val_batches

        for batch_idx, batch in enumerate(loader):
            if max_batches is not None and batch_idx >= max_batches:
                break
            images = batch["image"].to(self.device)
            age_target = batch["age"].to(self.device)
            age_mask = batch["age_mask"].to(self.device)
            gender_target = batch["gender_label"].to(self.device)
            gender_mask = batch["gender_mask"].to(self.device)

            autocast_ctx = (
                torch.autocast(device_type="cuda") if self.mixed_precision else contextlib.nullcontext()
            )
            with torch.set_grad_enabled(is_train):
                with autocast_ctx:
                    outputs = self.model(images)
                    loss_out = compute_multitask_loss(
                        outputs["age_output"], outputs["gender_logits"], age_target, age_mask,
                        gender_target, gender_mask, mode=mode,
                        fixed_age_weight=fixed.get("age_weight", 1.0),
                        fixed_gender_weight=fixed.get("gender_weight", 1.0),
                        log_var_age=self.model.log_var_age, log_var_gender=self.model.log_var_gender,
                        gender_class_weights=self.gender_class_weights,
                    )

            if is_train:
                optimizer.zero_grad()
                self.scaler.scale(loss_out.total_loss).backward()
                if self.grad_clip_norm:
                    self.scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
                self.scaler.step(optimizer)
                self.scaler.update()

            total_loss += loss_out.total_loss.item()
            n_batches += 1
            if loss_out.age_loss is not None:
                total_age_loss += loss_out.age_loss.item()
                n_age_batches += 1
                eff_age_w += loss_out.effective_age_weight
                if loss_out.log_var_age is not None:
                    lv_age += loss_out.log_var_age
                with torch.no_grad():
                    valid = age_mask.bool()
                    if valid.any():
                        err = (outputs["age_output"]["q50"][valid] - age_target[valid]).abs()
                        age_abs_errors.append(err.detach().cpu())
            if loss_out.gender_loss is not None:
                total_gender_loss += loss_out.gender_loss.item()
                n_gender_batches += 1
                eff_gender_w += loss_out.effective_gender_weight
                if loss_out.log_var_gender is not None:
                    lv_gender += loss_out.log_var_gender
                with torch.no_grad():
                    valid = gender_mask.bool()
                    if valid.any():
                        probs = torch.softmax(outputs["gender_logits"][valid], dim=-1)
                        preds = probs.argmax(dim=-1)
                        confidence = probs.max(dim=-1).values
                        correct = preds == gender_target[valid]
                        gender_correct += correct.sum().item()
                        gender_total += int(valid.sum().item())
                        # Selective accuracy/coverage/abstention (confidence-threshold aware) are
                        # tracked purely for the live console line / history.csv -- checkpoint
                        # selection (_balanced_score, BestMetricTracker) always uses the raw,
                        # non-abstention-aware "gender_accuracy" below, unchanged from before.
                        accepted = confidence >= self.confidence_threshold
                        gender_correct_accepted += (correct & accepted).sum().item()
                        gender_total_accepted += int(accepted.sum().item())
                        gender_confidences.append(confidence.detach().cpu())

        gender_abstention_value = float("nan")
        if gender_confidences:
            all_confidence = torch.cat(gender_confidences)
            gender_abstention_value = float((all_confidence < self.confidence_threshold).float().mean().item())

        metrics = {
            "loss": total_loss / max(1, n_batches),
            "age_loss": total_age_loss / max(1, n_age_batches),
            "gender_loss": total_gender_loss / max(1, n_gender_batches),
            "effective_age_weight": eff_age_w / max(1, n_age_batches),
            "effective_gender_weight": eff_gender_w / max(1, n_gender_batches),
            "log_var_age": lv_age / max(1, n_age_batches),
            "log_var_gender": lv_gender / max(1, n_gender_batches),
            "age_mae": float(torch.cat(age_abs_errors).mean()) if age_abs_errors else float("nan"),
            "age_rmse": float(torch.sqrt((torch.cat(age_abs_errors) ** 2).mean())) if age_abs_errors else float("nan"),
            "gender_accuracy": gender_correct / max(1, gender_total) if gender_total else float("nan"),
            "gender_selective_accuracy": (
                gender_correct_accepted / max(1, gender_total_accepted) if gender_total_accepted else float("nan")
            ),
            "gender_abstention": gender_abstention_value,
            "gender_coverage": (
                1.0 - gender_abstention_value if gender_abstention_value == gender_abstention_value else float("nan")
            ),
        }
        return metrics

    def _balanced_score(self, age_mae: float, gender_acc: float, age_max: float) -> float:
        if age_mae != age_mae:  # NaN check
            return gender_acc if gender_acc == gender_acc else float("-inf")
        if gender_acc != gender_acc:
            return -age_mae
        normalized_mae = age_mae / max(age_max, 1e-6)
        return gender_acc - normalized_mae

    def train(self) -> dict:
        has_pretrained = bool(self.config["model"].get("pretrained_checkpoint"))
        stages = build_stage_plan(self.training_cfg, has_pretrained)
        age_max = self.config["model"]["age_head"].get("age_max", 120)
        total_epochs_planned = sum(stage.epochs for stage in stages)
        seed_display = self.training_cfg.get("seed", self.config.get("seed"))

        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.model.parameters())
        start_line = (
            f"[{self.experiment_name} | seed={seed_display}] Starting training | device={self.device} | "
            f"train_samples={self.train_dataset_size} | val_samples={self.val_dataset_size} | "
            f"trainable_params={trainable_params:,}/{total_params:,} | "
            "checkpoint_selection=balanced_score (also tracked: age_mae, gender_accuracy)"
        )
        print(start_line, flush=True)
        logger.info(start_line)

        global_epoch = 0
        for stage in stages:
            stage_line = f"=== {stage.name} (epochs={stage.epochs}, lr={stage.lr:.2e}) ==="
            print(stage_line, flush=True)
            logger.info(stage_line)
            self.model.set_stage_trainable(stage.freeze_backbone, stage.unfreeze_layers)
            optimizer = _build_optimizer(self.model, stage.lr, self.training_cfg.get("weight_decay", 0.05))
            scheduler = _build_scheduler(optimizer, stage.epochs, self.training_cfg["scheduler"].get("warmup_epochs", 1))
            early_stopping = EarlyStopping(patience=self.training_cfg.get("early_stopping_patience", 8), mode="min")

            for _ in range(stage.epochs):
                start = time.time()
                train_metrics = self._run_batches(self.train_loader, optimizer)
                val_metrics = self._run_batches(self.val_loader, None)
                current_lr = optimizer.param_groups[0]["lr"]
                scheduler.step()
                elapsed = time.time() - start
                self.epoch_times.append(elapsed)
                global_epoch += 1

                self.history["train_loss"].append(train_metrics["loss"])
                self.history["val_loss"].append(val_metrics["loss"])
                self.history["val_age_mae"].append(val_metrics["age_mae"])
                self.history["val_age_rmse"].append(val_metrics["age_rmse"])
                self.history["val_gender_accuracy"].append(val_metrics["gender_accuracy"])
                self.history["val_gender_selective_accuracy"].append(val_metrics["gender_selective_accuracy"])
                self.history["val_gender_coverage"].append(val_metrics["gender_coverage"])
                self.history["val_gender_abstention"].append(val_metrics["gender_abstention"])
                self.history["age_loss"].append(train_metrics["age_loss"])
                self.history["gender_loss"].append(train_metrics["gender_loss"])
                self.history["effective_age_weight"].append(train_metrics["effective_age_weight"])
                self.history["effective_gender_weight"].append(train_metrics["effective_gender_weight"])
                self.history["log_var_age"].append(train_metrics["log_var_age"])
                self.history["log_var_gender"].append(train_metrics["log_var_gender"])
                self.history["lr"].append(current_lr)
                self.history["epoch_time_seconds"].append(elapsed)

                balanced = self._balanced_score(val_metrics["age_mae"], val_metrics["gender_accuracy"], age_max)
                self._maybe_checkpoint("age_mae", val_metrics["age_mae"], global_epoch, val_metrics)
                self._maybe_checkpoint("gender_accuracy", val_metrics["gender_accuracy"], global_epoch, val_metrics)
                is_best_balanced = self._maybe_checkpoint("balanced_score", balanced, global_epoch, val_metrics)

                epoch_line = (
                    f"[{self.experiment_name} | seed={seed_display}] "
                    f"Epoch {global_epoch:02d}/{total_epochs_planned} | {elapsed:.1f}s | lr={current_lr:.5f} | "
                    f"train_total={train_metrics['loss']:.4f} train_age={train_metrics['age_loss']:.4f} "
                    f"train_gender={train_metrics['gender_loss']:.4f} | "
                    f"val_total={val_metrics['loss']:.4f} val_age_mae={val_metrics['age_mae']:.3f} "
                    f"val_age_rmse={val_metrics['age_rmse']:.3f} | "
                    f"val_gender_selective_acc={val_metrics['gender_selective_accuracy']:.3f} "
                    f"val_coverage={val_metrics['gender_coverage']:.3f} val_abstention={val_metrics['gender_abstention']:.3f} | "
                    f"selection_score={balanced:.4f} | best={'yes' if is_best_balanced else 'no'} | "
                    f"early_stop={early_stopping.num_bad_epochs}/{early_stopping.patience}"
                )
                print(epoch_line, flush=True)
                logger.info(epoch_line)

                self._write_incremental_history()
                self._write_status_atomic(stage.name, global_epoch, total_epochs_planned, early_stopping)

                if not (val_metrics["loss"] == val_metrics["loss"]):
                    continue
                if early_stopping.step(val_metrics["loss"]):
                    stop_line = f"[{self.experiment_name} | seed={seed_display}] Early stopping triggered at epoch {global_epoch}"
                    print(stop_line, flush=True)
                    logger.info(stop_line)
                    break

        best_line = (
            f"[{self.experiment_name} | seed={seed_display}] Training complete | "
            f"best scores: {{'age_mae': {self.trackers['age_mae'].best_value}, "
            f"'gender_accuracy': {self.trackers['gender_accuracy'].best_value}, "
            f"'balanced_score': {self.trackers['balanced_score'].best_value}}}"
        )
        print(best_line, flush=True)
        logger.info(best_line)

        return {"history": self.history, "epoch_times": self.epoch_times}

    def _maybe_checkpoint(self, metric_name: str, value: float, epoch: int, metrics: dict) -> bool:
        if value != value:  # NaN, task absent this run
            return False
        tracker = self.trackers[metric_name]
        improved = tracker.update(value)
        if improved:
            path = self.checkpoint_dir / f"{self.experiment_name}_best_{metric_name}.pt"
            save_checkpoint(path, self.model, None, epoch, metrics, self.config)
        return improved

    def _write_incremental_history(self) -> None:
        """Rewrite history.csv/json after every epoch (not just at the end of
        train()), so a notebook can inspect progress -- or recover a
        partial run's history -- even if the process is interrupted
        mid-training. Rewriting the whole file each time (rather than
        appending) is simplest and cheap at these epoch counts, and avoids
        any header/row mismatch risk."""
        keys = list(self.history.keys())
        n_rows = len(self.history[keys[0]]) if keys else 0
        with open(self.history_csv_path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(keys)
            for i in range(n_rows):
                writer.writerow([self.history[key][i] for key in keys])

        with open(self.history_json_path, "w", encoding="utf-8") as fh:
            json.dump(self.history, fh, indent=2)

    def _write_status_atomic(self, stage_name: str, epoch: int, total_epochs_planned: int, early_stopping: EarlyStopping) -> None:
        """Write a live status file via write-temp-then-rename, so a reader
        never observes a half-written file (``Path.replace`` is atomic on
        both POSIX and Windows when source/destination are on the same
        filesystem, which they always are here)."""
        status = {
            "experiment_name": self.experiment_name,
            "stage": stage_name,
            "epoch": epoch,
            "total_epochs_planned": total_epochs_planned,
            "best_scores": {name: tracker.best_value for name, tracker in self.trackers.items()},
            "early_stopping_bad_epochs": early_stopping.num_bad_epochs,
            "early_stopping_patience": early_stopping.patience,
            "updated_at_utc": datetime.datetime.utcnow().isoformat() + "Z",
        }
        tmp_path = self.status_path.with_suffix(".json.tmp")
        with open(tmp_path, "w", encoding="utf-8") as fh:
            json.dump(status, fh, indent=2)
        tmp_path.replace(self.status_path)
