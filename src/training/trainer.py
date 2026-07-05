"""Progressive multi-stage trainer for the multi-task face model."""

from __future__ import annotations

import contextlib
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
    ) -> None:
        self.model = model.to(device)
        self.config = config
        self.device = device
        self.training_cfg = config["training"]
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.experiment_name = experiment_name
        self.gender_class_weights = gender_class_weights.to(device) if gender_class_weights is not None else None

        batch_size = self.training_cfg.get("batch_size", 64)
        num_workers = self.training_cfg.get("num_workers", 2)
        self.train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=len(train_dataset) > batch_size
        )
        self.val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

        self.mixed_precision = self.training_cfg.get("mixed_precision", True) and device == "cuda"
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.mixed_precision)
        self.grad_clip_norm = self.training_cfg.get("grad_clip_norm", 1.0)

        self.history: dict[str, list[float]] = {
            "train_loss": [], "val_loss": [], "val_age_mae": [], "val_gender_accuracy": [],
            "age_loss": [], "gender_loss": [], "effective_age_weight": [], "effective_gender_weight": [],
            "log_var_age": [], "log_var_gender": [],
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

        loss_cfg = self.config["model"]["loss_balancing"]
        mode = loss_cfg["mode"]
        fixed = loss_cfg.get("fixed", {"age_weight": 1.0, "gender_weight": 1.0})

        for batch in loader:
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
                        preds = outputs["gender_logits"][valid].argmax(dim=-1)
                        gender_correct += (preds == gender_target[valid]).sum().item()
                        gender_total += int(valid.sum().item())

        metrics = {
            "loss": total_loss / max(1, n_batches),
            "age_loss": total_age_loss / max(1, n_age_batches),
            "gender_loss": total_gender_loss / max(1, n_gender_batches),
            "effective_age_weight": eff_age_w / max(1, n_age_batches),
            "effective_gender_weight": eff_gender_w / max(1, n_gender_batches),
            "log_var_age": lv_age / max(1, n_age_batches),
            "log_var_gender": lv_gender / max(1, n_gender_batches),
            "age_mae": float(torch.cat(age_abs_errors).mean()) if age_abs_errors else float("nan"),
            "gender_accuracy": gender_correct / max(1, gender_total) if gender_total else float("nan"),
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

        global_epoch = 0
        for stage in stages:
            logger.info("=== %s (epochs=%d, lr=%.2e) ===", stage.name, stage.epochs, stage.lr)
            self.model.set_stage_trainable(stage.freeze_backbone, stage.unfreeze_layers)
            optimizer = _build_optimizer(self.model, stage.lr, self.training_cfg.get("weight_decay", 0.05))
            scheduler = _build_scheduler(optimizer, stage.epochs, self.training_cfg["scheduler"].get("warmup_epochs", 1))
            early_stopping = EarlyStopping(patience=self.training_cfg.get("early_stopping_patience", 8), mode="min")

            for _ in range(stage.epochs):
                start = time.time()
                train_metrics = self._run_batches(self.train_loader, optimizer)
                val_metrics = self._run_batches(self.val_loader, None)
                scheduler.step()
                elapsed = time.time() - start
                self.epoch_times.append(elapsed)
                global_epoch += 1

                self.history["train_loss"].append(train_metrics["loss"])
                self.history["val_loss"].append(val_metrics["loss"])
                self.history["val_age_mae"].append(val_metrics["age_mae"])
                self.history["val_gender_accuracy"].append(val_metrics["gender_accuracy"])
                self.history["age_loss"].append(train_metrics["age_loss"])
                self.history["gender_loss"].append(train_metrics["gender_loss"])
                self.history["effective_age_weight"].append(train_metrics["effective_age_weight"])
                self.history["effective_gender_weight"].append(train_metrics["effective_gender_weight"])
                self.history["log_var_age"].append(train_metrics["log_var_age"])
                self.history["log_var_gender"].append(train_metrics["log_var_gender"])

                logger.info(
                    "epoch %d | train_loss=%.4f val_loss=%.4f val_age_mae=%.3f val_gender_acc=%.3f (%.1fs)",
                    global_epoch, train_metrics["loss"], val_metrics["loss"],
                    val_metrics["age_mae"], val_metrics["gender_accuracy"], elapsed,
                )

                balanced = self._balanced_score(val_metrics["age_mae"], val_metrics["gender_accuracy"], age_max)
                self._maybe_checkpoint("age_mae", val_metrics["age_mae"], global_epoch, val_metrics)
                self._maybe_checkpoint("gender_accuracy", val_metrics["gender_accuracy"], global_epoch, val_metrics)
                self._maybe_checkpoint("balanced_score", balanced, global_epoch, val_metrics)

                if not (val_metrics["loss"] == val_metrics["loss"]):
                    continue
                if early_stopping.step(val_metrics["loss"]):
                    logger.info("Early stopping triggered at epoch %d", global_epoch)
                    break

        return {"history": self.history, "epoch_times": self.epoch_times}

    def _maybe_checkpoint(self, metric_name: str, value: float, epoch: int, metrics: dict) -> None:
        if value != value:  # NaN, task absent this run
            return
        tracker = self.trackers[metric_name]
        if tracker.update(value):
            path = self.checkpoint_dir / f"{self.experiment_name}_best_{metric_name}.pt"
            save_checkpoint(path, self.model, None, epoch, metrics, self.config)
