"""Tests for the transfer trainer's gradient-accumulation and AMP-skip
correctness (src/training/transfer_trainer.py::_run_epoch), plus the
unified early-stopping / checkpoint-selection metric.

These exercise _run_epoch directly with a tiny synthetic multi-task model
and hand-built dict batches -- no timm, no real data, no GPU, no pretrained
downloads.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from src.models.heads import AgeQuantileHead, GenderClassificationHead
from src.training.transfer_trainer import _run_epoch


class _TinyMT(nn.Module):
    def __init__(self, dim: int = 8):
        super().__init__()
        self.backbone = nn.Linear(3, dim)
        self.age_head = AgeQuantileHead(dim, 8)
        self.gender_head = GenderClassificationHead(dim, 8)
        self.log_var_age = nn.Parameter(torch.zeros(()))
        self.log_var_gender = nn.Parameter(torch.zeros(()))

    def forward(self, images):
        z = self.backbone(images.mean(dim=(2, 3)))
        return {"age_output": self.age_head(z), "gender_logits": self.gender_head(z)}


def _batches(n_batches: int, batch_size: int = 2):
    torch.manual_seed(0)
    out = []
    for _ in range(n_batches):
        out.append({
            "image": torch.randn(batch_size, 3, 4, 4),
            "age": torch.randint(1, 90, (batch_size,)).float(),
            "age_mask": torch.ones(batch_size),
            "gender_label": torch.randint(0, 2, (batch_size,)),
            "gender_mask": torch.ones(batch_size),
        })
    return out


def _run(model, batches, grad_accum, max_batches=None, scaler=None):
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    if scaler is None:
        scaler = torch.amp.GradScaler("cuda", enabled=False)
    loss_cfg = {"mode": "fixed", "fixed": {"age_weight": 1.0, "gender_weight": 1.0}}
    return _run_epoch(
        model, batches, optimizer, device="cpu", mixed_precision=False, scaler=scaler,
        grad_clip_norm=1.0, grad_accumulation_steps=grad_accum, loss_cfg=loss_cfg, current_epoch=1,
        gender_class_weights=None, confidence_threshold=0.8, max_batches=max_batches,
    )


def test_four_batches_accum_two_gives_two_steps():
    metrics = _run(_TinyMT(), _batches(4), grad_accum=2)
    assert metrics["_optimizer_steps"] == 2
    assert metrics["_microbatches_processed"] == 4


def test_five_batches_accum_two_final_partial_group_not_dropped():
    """5 microbatches, accumulation 2 -> groups [0,1], [2,3], [4 (partial)]:
    the final partial group must still trigger a 3rd optimizer step."""
    metrics = _run(_TinyMT(), _batches(5), grad_accum=2)
    assert metrics["_optimizer_steps"] == 3
    assert metrics["_microbatches_processed"] == 5


def test_three_effective_batches_accum_two_with_max_batches():
    """8 batches available but max_batches=3 -> effective 3 microbatches,
    accumulation 2 -> groups [0,1], [2 (partial)] -> 2 steps. The old
    is_last_batch == len(loader) check would have missed the partial group."""
    metrics = _run(_TinyMT(), _batches(8), grad_accum=2, max_batches=3)
    assert metrics["_microbatches_processed"] == 3
    assert metrics["_effective_batches"] == 3
    assert metrics["_optimizer_steps"] == 2


def test_single_batch_accum_larger_than_one_still_steps():
    metrics = _run(_TinyMT(), _batches(1), grad_accum=4)
    assert metrics["_optimizer_steps"] == 1
    assert metrics["_microbatches_processed"] == 1


def test_final_partial_group_actually_updates_parameters():
    """A partial final group must move the weights (a real optimizer step),
    not silently no-op."""
    model = _TinyMT()
    before = model.backbone.weight.detach().clone()
    _run(model, _batches(3), grad_accum=2)  # groups [0,1], [2 partial]
    assert not torch.allclose(before, model.backbone.weight.detach())


class _FakeSkippingScaler:
    """Simulates torch.amp.GradScaler skipping optimizer.step() on non-finite
    gradients: get_scale() shrinks after update(), which the trainer detects
    as 'the step was skipped'."""

    def __init__(self, skip: bool):
        self._scale = 65536.0
        self._skip = skip

    def scale(self, loss):
        return loss

    def unscale_(self, optimizer):
        pass

    def get_scale(self):
        return self._scale

    def step(self, optimizer):
        if not self._skip:
            optimizer.step()

    def update(self):
        if self._skip:
            self._scale /= 2.0  # signal a skipped step


def test_amp_skipped_step_not_counted_and_no_scheduler_advance_signal():
    metrics = _run(_TinyMT(), _batches(4), grad_accum=2, scaler=_FakeSkippingScaler(skip=True))
    assert metrics["_optimizer_steps"] == 0
    assert metrics["_skipped_optimizer_steps"] == 2
    assert metrics["_any_optimizer_step"] is False


def test_amp_non_skipped_step_counted():
    metrics = _run(_TinyMT(), _batches(4), grad_accum=2, scaler=_FakeSkippingScaler(skip=False))
    assert metrics["_optimizer_steps"] == 2
    assert metrics["_skipped_optimizer_steps"] == 0
    assert metrics["_any_optimizer_step"] is True


def test_core_trainer_unifies_early_stopping_and_checkpoint_selection():
    """Early stopping and the main 'best' checkpoint must use the same metric
    and mode (higher-is-better balanced score), never validation loss for one
    and balanced score for the other."""
    from src.training.trainer import Trainer

    assert Trainer.SELECTION_METRIC == "balanced_score"
    assert Trainer.SELECTION_MODE == "max"
