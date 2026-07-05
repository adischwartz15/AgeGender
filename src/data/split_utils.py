"""Deterministic train/val/test splitting with subject-level leakage prevention.

When a ``subject_id`` column is available and ``subject_level_if_available``
is True, splitting is done at the subject (group) level so the same
person's images never appear in more than one split. Otherwise, splitting
falls back to a per-row stratified-by-nothing random split. All splitting
is seeded for reproducibility and is saved to ``data/splits/`` so every
experiment in the ablation suite can reuse the identical split.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _normalize_fractions(train_fraction: float, val_fraction: float, test_fraction: float) -> tuple[float, float, float]:
    total = train_fraction + val_fraction + test_fraction
    if abs(total - 1.0) > 1e-6:
        logger.warning("Split fractions sum to %.4f, renormalizing to 1.0", total)
        train_fraction, val_fraction, test_fraction = (
            train_fraction / total,
            val_fraction / total,
            test_fraction / total,
        )
    return train_fraction, val_fraction, test_fraction


def split_dataframe(
    df: pd.DataFrame,
    train_fraction: float = 0.7,
    val_fraction: float = 0.15,
    test_fraction: float = 0.15,
    seed: int = 42,
    subject_level_if_available: bool = True,
) -> pd.DataFrame:
    """Return ``df`` with an added ``split`` column in {"train", "val", "test"}.

    If ``df`` already has a non-null ``split`` column (e.g. supplied by the
    dataset itself via a CSV split column), it is respected and returned
    unchanged.
    """
    if "split" in df.columns and df["split"].notna().all():
        logger.info("Using pre-existing split column from dataset metadata")
        return df

    train_fraction, val_fraction, test_fraction = _normalize_fractions(
        train_fraction, val_fraction, test_fraction
    )
    rng = np.random.default_rng(seed)

    has_subjects = subject_level_if_available and "subject_id" in df.columns and df["subject_id"].notna().any()

    df = df.copy()
    if has_subjects:
        subjects = df["subject_id"].dropna().unique()
        rng.shuffle(subjects)
        n = len(subjects)
        n_train = int(round(n * train_fraction))
        n_val = int(round(n * val_fraction))
        train_subjects = set(subjects[:n_train])
        val_subjects = set(subjects[n_train : n_train + n_val])
        test_subjects = set(subjects[n_train + n_val :])

        def _assign(subject_id):
            if subject_id in train_subjects:
                return "train"
            if subject_id in val_subjects:
                return "val"
            return "test"

        # Rows without a subject_id fall back to independent random assignment.
        no_subject_mask = df["subject_id"].isna()
        df["split"] = df["subject_id"].map(_assign)
        if no_subject_mask.any():
            n_no_subject = int(no_subject_mask.sum())
            assignments = rng.choice(
                ["train", "val", "test"], size=n_no_subject, p=[train_fraction, val_fraction, test_fraction]
            )
            df.loc[no_subject_mask, "split"] = assignments
        logger.info("Subject-level split across %d unique subjects", n)
    else:
        n = len(df)
        indices = rng.permutation(n)
        n_train = int(round(n * train_fraction))
        n_val = int(round(n * val_fraction))
        split_labels = np.empty(n, dtype=object)
        split_labels[indices[:n_train]] = "train"
        split_labels[indices[n_train : n_train + n_val]] = "val"
        split_labels[indices[n_train + n_val :]] = "test"
        df["split"] = split_labels
        logger.info("Row-level random split (no usable subject_id column found)")

    return df


def assert_no_leakage(df: pd.DataFrame) -> None:
    """Raise if any image path or (when available) subject_id spans multiple splits."""
    dup_paths = df.groupby("image_path")["split"].nunique()
    leaking_paths = dup_paths[dup_paths > 1]
    if len(leaking_paths) > 0:
        raise ValueError(f"Data leakage: {len(leaking_paths)} image paths appear in multiple splits")

    if "subject_id" in df.columns and df["subject_id"].notna().any():
        subj_df = df.dropna(subset=["subject_id"])
        dup_subjects = subj_df.groupby("subject_id")["split"].nunique()
        leaking_subjects = dup_subjects[dup_subjects > 1]
        if len(leaking_subjects) > 0:
            raise ValueError(f"Data leakage: {len(leaking_subjects)} subjects appear in multiple splits")
