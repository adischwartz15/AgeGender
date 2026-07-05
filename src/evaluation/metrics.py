"""Core age and dataset gender-label evaluation metrics."""

from __future__ import annotations

import numpy as np


def age_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def age_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def age_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot == 0:
        return float("nan")
    return float(1 - ss_res / ss_tot)


def interval_coverage(y_true: np.ndarray, q_low: np.ndarray, q_high: np.ndarray) -> float:
    """Fraction of samples where ``q_low <= y_true <= q_high``."""
    return float(np.mean((y_true >= q_low) & (y_true <= q_high)))


def mean_interval_width(q_low: np.ndarray, q_high: np.ndarray) -> float:
    return float(np.mean(q_high - q_low))


def median_interval_width(q_low: np.ndarray, q_high: np.ndarray) -> float:
    return float(np.median(q_high - q_low))


def expected_calibration_error_intervals(y_true: np.ndarray, q_low: np.ndarray, q_high: np.ndarray, target_coverage: float = 0.80) -> float:
    """|empirical coverage - target coverage| for the q10-q90 interval."""
    empirical = interval_coverage(y_true, q_low, q_high)
    return float(abs(empirical - target_coverage))


def age_error_by_bucket(y_true: np.ndarray, y_pred: np.ndarray, bucket_edges: list[int] | None = None) -> dict[str, dict]:
    """Mean absolute error grouped into age buckets, e.g. 0-10, 10-20, ..."""
    if bucket_edges is None:
        bucket_edges = [0, 10, 20, 30, 40, 50, 60, 70, 80, 200]
    result = {}
    for lo, hi in zip(bucket_edges[:-1], bucket_edges[1:]):
        mask = (y_true >= lo) & (y_true < hi)
        label = f"{lo}-{hi if hi < 200 else '120+'}"
        if mask.sum() == 0:
            result[label] = {"count": 0, "mae": None, "coverage": None}
        else:
            result[label] = {
                "count": int(mask.sum()),
                "mae": age_mae(y_true[mask], y_pred[mask]),
            }
    return result


def gender_accuracy(y_true: np.ndarray, y_pred: np.ndarray, abstain_mask: np.ndarray | None = None) -> float:
    """Accuracy computed only over non-abstained predictions."""
    if abstain_mask is not None:
        keep = ~abstain_mask
        if keep.sum() == 0:
            return float("nan")
        y_true, y_pred = y_true[keep], y_pred[keep]
    return float(np.mean(y_true == y_pred))


def abstention_rate(abstain_mask: np.ndarray) -> float:
    return float(np.mean(abstain_mask))


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int = 2) -> np.ndarray:
    matrix = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        matrix[int(t), int(p)] += 1
    return matrix


def confidence_statistics(confidences: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(np.mean(confidences)),
        "std": float(np.std(confidences)),
        "min": float(np.min(confidences)),
        "max": float(np.max(confidences)),
        "median": float(np.median(confidences)),
    }
