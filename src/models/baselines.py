"""Trivial sanity-check baselines (not architecture ablation experiments).

These are intentionally dumb reference points -- a constant-quantile age
predictor and a majority-class gender predictor -- used only to confirm
that learned models actually beat trivial statistics. The real ablation
experiments (separate backbones, shared+adapters, learned balancing,
kNN) live in ``configs/experiments.yaml`` and
``src/evaluation/knn_baseline.py``.
"""

from __future__ import annotations

import numpy as np


class ConstantQuantileAgeBaseline:
    """Predicts the same (q10, q50, q90), estimated from training-set age quantiles."""

    def __init__(self) -> None:
        self.q10_: float | None = None
        self.q50_: float | None = None
        self.q90_: float | None = None

    def fit(self, train_ages: np.ndarray) -> "ConstantQuantileAgeBaseline":
        ages = np.asarray(train_ages, dtype=np.float64)
        self.q10_, self.q50_, self.q90_ = np.quantile(ages, [0.10, 0.50, 0.90])
        return self

    def predict(self, n: int) -> dict[str, np.ndarray]:
        if self.q50_ is None:
            raise RuntimeError("Call fit() before predict().")
        return {
            "q10": np.full(n, self.q10_),
            "q50": np.full(n, self.q50_),
            "q90": np.full(n, self.q90_),
        }


class MajorityClassGenderBaseline:
    """Always predicts the most frequent dataset gender label from training data."""

    def __init__(self) -> None:
        self.majority_class_: int | None = None
        self.class_prior_: np.ndarray | None = None

    def fit(self, train_labels: np.ndarray, num_classes: int = 2) -> "MajorityClassGenderBaseline":
        labels = np.asarray(train_labels)
        counts = np.bincount(labels, minlength=num_classes).astype(np.float64)
        self.class_prior_ = counts / counts.sum()
        self.majority_class_ = int(np.argmax(counts))
        return self

    def predict_proba(self, n: int) -> np.ndarray:
        if self.class_prior_ is None:
            raise RuntimeError("Call fit() before predict_proba().")
        return np.tile(self.class_prior_, (n, 1))
