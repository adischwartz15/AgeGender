"""Shared Matplotlib plotting helpers used by evaluation/reporting scripts.

All functions save a PNG to ``out_path`` and return the path. Kept
dependency-light (Matplotlib only) so plotting works in headless/CI
environments (the ``Agg`` backend is forced).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _save(fig: plt.Figure, out_path: str | Path) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_training_curves(history: dict[str, list[float]], out_path: str | Path) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    epochs = range(1, len(history.get("train_loss", [])) + 1)
    axes[0].plot(epochs, history.get("train_loss", []), label="train")
    axes[0].plot(epochs, history.get("val_loss", []), label="val")
    axes[0].set_title("Total loss")
    axes[0].set_xlabel("epoch")
    axes[0].legend()

    if "val_age_mae" in history:
        axes[1].plot(epochs, history["val_age_mae"], label="val age MAE", color="tab:orange")
    if "val_gender_accuracy" in history:
        ax2 = axes[1].twinx()
        ax2.plot(epochs, history["val_gender_accuracy"], label="val gender acc", color="tab:green")
        ax2.set_ylabel("gender accuracy")
    axes[1].set_title("Validation metrics")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("age MAE")
    return _save(fig, out_path)


def plot_age_scatter(y_true: np.ndarray, y_pred: np.ndarray, out_path: str | Path) -> Path:
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(y_true, y_pred, s=8, alpha=0.4)
    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    ax.plot(lims, lims, color="red", linestyle="--", linewidth=1)
    ax.set_xlabel("True age")
    ax.set_ylabel("Predicted age (q50)")
    ax.set_title("Predicted vs true age")
    return _save(fig, out_path)


def plot_error_histogram(errors: np.ndarray, out_path: str | Path) -> Path:
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.hist(errors, bins=30, color="tab:blue", alpha=0.8)
    ax.set_xlabel("Prediction error (years)")
    ax.set_ylabel("Count")
    ax.set_title("Age error distribution")
    return _save(fig, out_path)


def plot_interval_coverage(bucket_labels: list[str], coverage: np.ndarray, target: float, out_path: str | Path) -> Path:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(bucket_labels, coverage, color="tab:purple", alpha=0.8)
    ax.axhline(target, color="red", linestyle="--", label=f"target={target:.2f}")
    ax.set_ylabel("Empirical coverage")
    ax.set_title("q10-q90 interval coverage by age bucket")
    ax.legend()
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    return _save(fig, out_path)


def plot_confusion_matrix(matrix: np.ndarray, class_names: list[str], out_path: str | Path) -> Path:
    fig, ax = plt.subplots(figsize=(4.5, 4))
    im = ax.imshow(matrix, cmap="Blues")
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=30, ha="right")
    ax.set_yticklabels(class_names)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, str(int(matrix[i, j])), ha="center", va="center")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Dataset gender-label confusion matrix")
    fig.colorbar(im, ax=ax, fraction=0.046)
    return _save(fig, out_path)


def plot_loss_balancing(history: dict[str, list[float]], out_path: str | Path) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    epochs = range(1, len(history.get("age_loss", [])) + 1)
    axes[0].plot(epochs, history.get("age_loss", []), label="age loss")
    axes[0].plot(epochs, history.get("gender_loss", []), label="gender loss")
    axes[0].set_title("Per-task loss")
    axes[0].legend()
    if "effective_age_weight" in history:
        axes[1].plot(epochs, history["effective_age_weight"], label="effective age weight")
        axes[1].plot(epochs, history["effective_gender_weight"], label="effective gender weight")
        axes[1].set_title("Effective task weights")
        axes[1].legend()
    return _save(fig, out_path)


def plot_gradient_cosine_similarity(values: np.ndarray, out_path: str | Path, title: str) -> Path:
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.hist(values, bins=30, color="tab:red", alpha=0.7)
    ax.axvline(float(np.mean(values)), color="black", linestyle="--", label=f"mean={np.mean(values):.3f}")
    ax.set_xlabel("cosine similarity(grad_age, grad_gender)")
    ax.set_ylabel("count")
    ax.set_title(title)
    ax.legend()
    return _save(fig, out_path)


def plot_embedding_scatter(
    coords: np.ndarray,
    labels: np.ndarray | None,
    label_names: dict[int, str] | None,
    out_path: str | Path,
    title: str,
) -> Path:
    fig, ax = plt.subplots(figsize=(5.5, 5))
    if labels is None:
        ax.scatter(coords[:, 0], coords[:, 1], s=8, alpha=0.6)
    else:
        unique = np.unique(labels)
        cmap = plt.get_cmap("tab10")
        for i, value in enumerate(unique):
            mask = labels == value
            name = label_names.get(int(value), str(value)) if label_names else str(value)
            ax.scatter(coords[mask, 0], coords[mask, 1], s=8, alpha=0.6, label=name, color=cmap(i % 10))
        ax.legend(markerscale=2, fontsize=8)
    ax.set_title(title)
    ax.set_xlabel("component 1")
    ax.set_ylabel("component 2")
    return _save(fig, out_path)


def plot_robustness_curves(df, metric: str, out_path: str | Path) -> Path:
    """``df`` is a pandas DataFrame with columns corruption, severity, and ``metric``."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for corruption, group in df.groupby("corruption"):
        group = group.sort_values("severity")
        ax.plot(group["severity"], group[metric], marker="o", label=corruption)
    ax.set_xlabel("severity")
    ax.set_ylabel(metric)
    ax.set_title(f"Robustness: {metric} vs corruption severity")
    ax.legend(fontsize=7, ncol=2)
    return _save(fig, out_path)


def save_gradcam_overlay(image_rgb: np.ndarray, heatmap: np.ndarray, out_path: str | Path, title: str) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(7, 3.5))
    axes[0].imshow(image_rgb)
    axes[0].set_title("Input")
    axes[0].axis("off")
    axes[1].imshow(image_rgb)
    axes[1].imshow(heatmap, cmap="jet", alpha=0.45)
    axes[1].set_title(title)
    axes[1].axis("off")
    return _save(fig, out_path)
