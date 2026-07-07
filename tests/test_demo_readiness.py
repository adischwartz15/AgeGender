"""Tests for scripts/check_demo_readiness.py's pass/fail logic.

The demo launcher (scripts/run_demo.py) refuses to start unless this
check reports ready, so its checkpoint/calibration detection needs to be
correct in both directions: missing-required-artifact fails, and missing
optional-artifact (kNN index, demo images) warns without blocking.
"""

from __future__ import annotations

import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from check_demo_readiness import check_demo_readiness  # noqa: E402

from src.utils.io import save_json  # noqa: E402


def _api_config(tmp_path, with_checkpoint=False, with_calibration=False):
    checkpoint_path = tmp_path / "checkpoints" / "multitask_best_balanced_score.pt"
    calibration_dir = tmp_path / "outputs" / "calibration"
    knn_dir = tmp_path / "outputs" / "knn"

    if with_checkpoint:
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint_path.write_bytes(b"fake-checkpoint")
    if with_calibration:
        save_json({"method": "split_conformal_cqr", "offset": 3.0}, calibration_dir / "conformal_calibration.json")

    return {
        "active_checkpoint": str(checkpoint_path),
        "calibration_dir": str(calibration_dir),
        "knn_index_dir": str(knn_dir),
    }


def test_not_ready_when_checkpoint_missing(tmp_path):
    ready, messages = check_demo_readiness(_api_config(tmp_path, with_checkpoint=False, with_calibration=True))
    assert ready is False
    assert any("[FAIL]" in m and "checkpoint" in m.lower() for m in messages)


def test_not_ready_when_calibration_missing(tmp_path):
    ready, messages = check_demo_readiness(_api_config(tmp_path, with_checkpoint=True, with_calibration=False))
    assert ready is False
    assert any("[FAIL]" in m and "calibration" in m.lower() for m in messages)


def test_ready_when_checkpoint_and_calibration_present(tmp_path):
    ready, messages = check_demo_readiness(_api_config(tmp_path, with_checkpoint=True, with_calibration=True))
    assert ready is True
    assert not any("[FAIL]" in m for m in messages)


def test_missing_knn_index_warns_but_does_not_block(tmp_path):
    ready, messages = check_demo_readiness(_api_config(tmp_path, with_checkpoint=True, with_calibration=True))
    assert ready is True
    assert any("[WARN]" in m and "k-nn" in m.lower() for m in messages)
