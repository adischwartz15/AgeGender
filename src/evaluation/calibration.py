"""Split conformal calibration for age q10/q90 prediction intervals.

Uses the validation set only (never the test set) to learn a single
scalar interval-expansion offset via conformalized quantile regression
(CQR, Romano et al. 2019):

    score_i = max(q10_i - y_i, y_i - q90_i)
    offset  = the ceil((n+1)(1-alpha))/n empirical quantile of {score_i}
    calibrated interval = [q10 - offset, q90 + offset]

This guarantees (marginally, under exchangeability) that the calibrated
interval covers the true value with probability >= 1 - alpha on held-out
data drawn from the same distribution. Intervals are only ever described
as "calibrated" in the API/frontend when a calibration artifact produced
by this procedure actually exists and loaded successfully.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from src.utils.io import load_json, save_json


def compute_nonconformity_scores(y_true: np.ndarray, q10: np.ndarray, q90: np.ndarray) -> np.ndarray:
    return np.maximum(q10 - y_true, y_true - q90)


def fit_conformal_offset(scores: np.ndarray, alpha: float = 0.10) -> float:
    """Compute the split-conformal offset for miscoverage level ``alpha``."""
    n = len(scores)
    if n == 0:
        raise ValueError("Cannot fit conformal calibration on an empty validation set")
    level = min(1.0, math.ceil((n + 1) * (1 - alpha)) / n)
    return float(np.quantile(scores, level))


def apply_conformal_offset(q10: np.ndarray, q90: np.ndarray, offset: float) -> tuple[np.ndarray, np.ndarray]:
    return q10 - offset, q90 + offset


def fit_and_save_calibration(
    y_true_val: np.ndarray, q10_val: np.ndarray, q90_val: np.ndarray, alpha: float, output_dir: str | Path
) -> dict:
    """Fit conformal calibration on the validation set and save the artifact."""
    scores = compute_nonconformity_scores(y_true_val, q10_val, q90_val)
    offset = fit_conformal_offset(scores, alpha)
    artifact = {
        "method": "split_conformal_cqr",
        "alpha": alpha,
        "target_coverage": 1 - alpha,
        "offset": offset,
        "n_calibration_samples": int(len(y_true_val)),
    }
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_json(artifact, output_dir / "conformal_calibration.json")
    return artifact


def load_calibration(output_dir: str | Path) -> dict | None:
    """Load a previously saved calibration artifact, or None if it doesn't exist."""
    path = Path(output_dir) / "conformal_calibration.json"
    if not path.exists():
        return None
    return load_json(path)


def evaluate_calibration_effect(
    y_true_test: np.ndarray, q10_test: np.ndarray, q90_test: np.ndarray, offset: float
) -> dict:
    """Report coverage/width before and after applying the conformal offset."""
    from src.evaluation.metrics import interval_coverage, mean_interval_width

    coverage_before = interval_coverage(y_true_test, q10_test, q90_test)
    width_before = mean_interval_width(q10_test, q90_test)

    q10_cal, q90_cal = apply_conformal_offset(q10_test, q90_test, offset)
    coverage_after = interval_coverage(y_true_test, q10_cal, q90_cal)
    width_after = mean_interval_width(q10_cal, q90_cal)

    return {
        "coverage_before_calibration": coverage_before,
        "coverage_after_calibration": coverage_after,
        "mean_width_before_calibration": width_before,
        "mean_width_after_calibration": width_after,
    }
