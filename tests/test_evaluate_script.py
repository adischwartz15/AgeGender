"""Tests for scripts/evaluate.py's compute_parametric_metrics wiring.

Guards against a regression where the module docstring described
per-bucket uncertainty metrics but the function body still called a
now-removed helper (age_error_by_bucket) that wasn't imported -- the kind
of inconsistency that only surfaces at runtime, not at import time.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from evaluate import compute_parametric_metrics  # noqa: E402


def _synthetic_preds(n=40, seed=0):
    rng = np.random.default_rng(seed)
    age = rng.uniform(0, 80, size=n)
    q50 = age + rng.normal(0, 2, size=n)
    q10 = q50 - rng.uniform(5, 10, size=n)
    q90 = q50 + rng.uniform(5, 10, size=n)
    probs = rng.dirichlet([1, 1], size=n)
    return {
        "q10": q10, "q50": q50, "q90": q90, "probs": probs,
        "age": age, "age_mask": np.ones(n, dtype=bool),
        "gender": rng.integers(0, 2, size=n), "gender_mask": np.ones(n, dtype=bool),
        "latency_ms_per_image": 1.23,
    }


def test_compute_parametric_metrics_includes_age_metrics_by_bucket():
    preds = _synthetic_preds()
    metrics = compute_parametric_metrics(preds, confidence_threshold=0.80, calibration=None)
    assert "age_metrics_by_bucket" in metrics
    assert "age_error_by_bucket" not in metrics
    bucket = metrics["age_metrics_by_bucket"]
    assert any(v["count"] > 0 for v in bucket.values())
    for label, stats in bucket.items():
        if stats["count"] > 0:
            assert stats["mae"] is not None
            assert stats["coverage"] is not None
            assert stats["mean_width"] is not None


def test_compute_parametric_metrics_adds_calibrated_bucket_metrics_when_calibration_present():
    preds = _synthetic_preds()
    calibration = {"offset": 1.5}
    metrics = compute_parametric_metrics(preds, confidence_threshold=0.80, calibration=calibration)
    assert "age_metrics_by_bucket_calibrated" in metrics
    assert "interval_coverage_calibrated" in metrics
    assert "mean_interval_width_calibrated" in metrics
    # A positive offset widens intervals, which can only raise or hold coverage.
    assert metrics["mean_interval_width_calibrated"] > metrics["mean_interval_width"]


def test_compute_parametric_metrics_omits_calibrated_keys_when_calibration_absent():
    preds = _synthetic_preds()
    metrics = compute_parametric_metrics(preds, confidence_threshold=0.80, calibration=None)
    assert "age_metrics_by_bucket_calibrated" not in metrics
    assert "interval_coverage_calibrated" not in metrics
