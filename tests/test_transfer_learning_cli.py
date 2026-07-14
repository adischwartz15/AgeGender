"""Tests for scripts/run_transfer_learning.py's CLI-level resume/skip/evaluate-only
behavior. Deliberately does not exercise a real training run (that's
tests/test_transfer_trainer.py's job) -- these tests monkeypatch the heavy
model-construction entry points to confirm they are (or are not) called at
all, per the branch under test.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pytest

pytest.importorskip("timm")

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = str(REPO_ROOT / "scripts")
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

import run_transfer_learning as rtl  # noqa: E402

from src.training.persistent_artifacts import (  # noqa: E402
    CorruptedCheckpointError, PersistentArtifactManager, sha256_file,
)


def _args(**overrides) -> argparse.Namespace:
    defaults = dict(
        storage_root=None, persistent_root=None, sync_after_epoch=False,
        resume=False, skip_completed=False, evaluate_only=False, model_family="volo",
    )
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def test_default_seeds_are_canonical():
    import inspect

    source = inspect.getsource(rtl.main)
    assert '"--seeds", default="42,123,2026"' in source
    assert "42,43,44" not in source


def test_skip_completed_seed_never_calls_training(tmp_path, monkeypatch):
    monkeypatch.setattr(rtl, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(rtl, "DEFAULT_LOCAL_ROOT", tmp_path / "checkpoints" / "transfer_learning")

    manager = PersistentArtifactManager(
        rtl.VOLO_EXPERIMENT_NAME, seed=42, local_root=tmp_path / "checkpoints" / "transfer_learning",
    )
    checkpoint_path = manager.save_best_checkpoint({"model_state_dict": {}, "config": {}})
    manager.mark_seed_complete({
        "seed": 42, "status": "complete", "best_checkpoint": str(checkpoint_path),
        "test_metrics": {"age_mae": 5.0}, "completed_at": "now",
        "checkpoint_sha256": sha256_file(checkpoint_path),
    })

    def _boom(*a, **kw):
        raise AssertionError("training must not be entered for an already-complete seed")

    monkeypatch.setattr("src.models.pretrained_volo.build_pretrained_volo_model", _boom)

    metrics = rtl.run_volo_seed(42, _args(skip_completed=True))
    assert metrics == {"age_mae": 5.0}


def test_evaluate_only_never_trains_even_without_existing_checkpoint(tmp_path, monkeypatch):
    monkeypatch.setattr(rtl, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(rtl, "DEFAULT_LOCAL_ROOT", tmp_path / "checkpoints" / "transfer_learning")

    def _boom(*a, **kw):
        raise AssertionError("--evaluate-only must never train")

    monkeypatch.setattr("src.models.pretrained_volo.build_pretrained_volo_model", _boom)

    metrics = rtl.run_volo_seed(42, _args(evaluate_only=True))
    assert metrics is None  # no checkpoint exists anywhere -- reported as unavailable, not fabricated


def test_evaluate_only_evaluates_existing_checkpoint_without_training(tmp_path, monkeypatch):
    monkeypatch.setattr(rtl, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(rtl, "DEFAULT_LOCAL_ROOT", tmp_path / "checkpoints" / "transfer_learning")

    manager = PersistentArtifactManager(
        rtl.VOLO_EXPERIMENT_NAME, seed=42, local_root=tmp_path / "checkpoints" / "transfer_learning",
    )
    manager.save_best_checkpoint({"model_state_dict": {}, "config": {}})

    def _boom(*a, **kw):
        raise AssertionError("--evaluate-only must never train")

    monkeypatch.setattr("src.models.pretrained_volo.build_pretrained_volo_model", _boom)
    calls = []
    monkeypatch.setattr(rtl, "evaluate_checkpoint", lambda path, output_name: calls.append(path) or {"age_mae": 1.0})

    metrics = rtl.run_volo_seed(42, _args(evaluate_only=True))
    assert metrics == {"age_mae": 1.0}
    assert calls and calls[0].endswith("best.pt")


def test_missing_prepared_split_returns_none_without_training(tmp_path, monkeypatch):
    monkeypatch.setattr(rtl, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(rtl, "DEFAULT_LOCAL_ROOT", tmp_path / "checkpoints" / "transfer_learning")

    def _boom(*a, **kw):
        raise AssertionError("must not attempt training with no prepared split")

    monkeypatch.setattr("src.models.pretrained_volo.build_pretrained_volo_model", _boom)

    metrics = rtl.run_volo_seed(42, _args())
    assert metrics is None


def test_resume_propagates_corrupted_checkpoint_error(tmp_path, monkeypatch):
    """A corrupted checkpoint must raise (never silently retrain from
    scratch) -- see PersistentArtifactManager.find_latest_valid_checkpoint."""
    monkeypatch.setattr(rtl, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(rtl, "DEFAULT_LOCAL_ROOT", tmp_path / "checkpoints" / "transfer_learning")

    # A prepared split must exist to reach the resume check.
    splits_dir = tmp_path / "data" / "splits"
    splits_dir.mkdir(parents=True)
    (splits_dir / "full_metadata_with_splits.csv").write_text("id\n1\n", encoding="utf-8")

    def _raise_corrupted(self):
        raise CorruptedCheckpointError("simulated corruption")

    monkeypatch.setattr(PersistentArtifactManager, "find_latest_valid_checkpoint", _raise_corrupted)

    with pytest.raises(CorruptedCheckpointError):
        rtl.run_volo_seed(42, _args(resume=True))


def test_no_stale_seed_literal_in_transfer_learning_cli_source():
    import inspect

    source = inspect.getsource(rtl)
    assert "42,43,44" not in source
    assert "42, 43, 44" not in source
