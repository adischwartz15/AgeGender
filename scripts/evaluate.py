#!/usr/bin/env python
"""CLI: evaluate a trained checkpoint on the held-out test split.

Computes age MAE/RMSE/R2, interval coverage/width, calibration error,
per-age-bucket error, and dataset gender-label accuracy/confusion
matrix/abstention rate. Optionally compares against a k-NN baseline.

Usage:
    python scripts/evaluate.py --checkpoint checkpoints/multitask_best_balanced_score.pt [--compare-knn]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.dataset import FaceMultiTaskDataset
from src.data.transforms import EvalTransform
from src.evaluation.calibration import apply_conformal_offset, load_calibration
from src.evaluation.comparison import build_parametric_vs_knn_table
from src.evaluation.knn_baseline import KNNEmbeddingBaseline
from src.evaluation.metrics import (
    abstention_rate, age_error_by_bucket, age_mae, age_r2, age_rmse, confidence_statistics,
    confusion_matrix, expected_calibration_error_intervals, gender_accuracy, interval_coverage,
    mean_interval_width, median_interval_width,
)
from src.inference.artifacts import load_model_checkpoint
from src.utils.config import REPO_ROOT
from src.utils.io import save_json
from src.utils.logging import get_logger
from src.utils.visualization import plot_age_scatter, plot_confusion_matrix, plot_error_histogram, plot_interval_coverage

logger = get_logger("scripts.evaluate")


@torch.no_grad()
def run_inference(model, dataset, device, batch_size=64):
    from torch.utils.data import DataLoader

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    q10s, q50s, q90s, probs_all = [], [], [], []
    ages, age_masks, genders, gender_masks = [], [], [], []
    start = time.time()
    n_images = 0
    for batch in loader:
        images = batch["image"].to(device)
        outputs = model(images)
        q10s.append(outputs["age_output"]["q10"].cpu().numpy())
        q50s.append(outputs["age_output"]["q50"].cpu().numpy())
        q90s.append(outputs["age_output"]["q90"].cpu().numpy())
        probs_all.append(torch.softmax(outputs["gender_logits"], dim=-1).cpu().numpy())
        ages.append(batch["age"].numpy())
        age_masks.append(batch["age_mask"].numpy())
        genders.append(batch["gender_label"].numpy())
        gender_masks.append(batch["gender_mask"].numpy())
        n_images += len(images)
    elapsed = time.time() - start
    latency_ms_per_image = (elapsed / max(1, n_images)) * 1000.0
    return {
        "q10": np.concatenate(q10s), "q50": np.concatenate(q50s), "q90": np.concatenate(q90s),
        "probs": np.concatenate(probs_all), "age": np.concatenate(ages), "age_mask": np.concatenate(age_masks),
        "gender": np.concatenate(genders), "gender_mask": np.concatenate(gender_masks),
        "latency_ms_per_image": latency_ms_per_image,
    }


def compute_parametric_metrics(preds: dict, confidence_threshold: float, calibration: dict | None) -> dict:
    age_mask = preds["age_mask"].astype(bool)
    gender_mask = preds["gender_mask"].astype(bool)

    metrics = {"latency_ms_per_image": preds["latency_ms_per_image"]}

    if age_mask.any():
        y_true = preds["age"][age_mask]
        q10, q50, q90 = preds["q10"][age_mask], preds["q50"][age_mask], preds["q90"][age_mask]
        metrics.update({
            "age_mae": age_mae(y_true, q50), "age_rmse": age_rmse(y_true, q50), "age_r2": age_r2(y_true, q50),
            "interval_coverage": interval_coverage(y_true, q10, q90),
            "mean_interval_width": mean_interval_width(q10, q90),
            "median_interval_width": median_interval_width(q10, q90),
            "calibration_error": expected_calibration_error_intervals(y_true, q10, q90, target_coverage=0.80),
            "age_error_by_bucket": age_error_by_bucket(y_true, q50),
        })
        if calibration is not None:
            q10_cal, q90_cal = apply_conformal_offset(q10, q90, calibration["offset"])
            metrics["interval_coverage_calibrated"] = interval_coverage(y_true, q10_cal, q90_cal)
            metrics["mean_interval_width_calibrated"] = mean_interval_width(q10_cal, q90_cal)

    if gender_mask.any():
        probs = preds["probs"][gender_mask]
        y_true_gender = preds["gender"][gender_mask].astype(int)
        predicted = probs.argmax(axis=1)
        confidence = probs.max(axis=1)
        abstain = confidence < confidence_threshold
        metrics.update({
            "gender_accuracy": gender_accuracy(y_true_gender, predicted, abstain),
            "abstention_rate": abstention_rate(abstain),
            "confidence_stats": confidence_statistics(confidence),
        })

    return metrics


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--compare-knn", action="store_true")
    parser.add_argument("--knn-path", default=str(REPO_ROOT / "outputs" / "knn" / "knn_baseline.pkl"))
    parser.add_argument("--calibration-dir", default=str(REPO_ROOT / "outputs" / "calibration"))
    parser.add_argument("--output-name", default="test_evaluation")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, config, _ = load_model_checkpoint(args.checkpoint, device)

    splits_path = REPO_ROOT / config["paths"]["splits_dir"] / "full_metadata_with_splits.csv"
    if not splits_path.exists():
        logger.error("No prepared split found at %s.", splits_path)
        return 1
    df = pd.read_csv(splits_path)
    test_df = df[df["split"] == "test"]
    dataset = FaceMultiTaskDataset(test_df, EvalTransform(config["dataset"]["image_size"]))

    preds = run_inference(model, dataset, device)
    calibration = load_calibration(args.calibration_dir)
    confidence_threshold = config["model"]["gender_head"].get("confidence_threshold", 0.80)
    metrics = compute_parametric_metrics(preds, confidence_threshold, calibration)

    output_dir = REPO_ROOT / config["paths"]["output_dir"]
    metrics_dir, plots_dir = output_dir / "metrics", output_dir / "plots"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    save_json(metrics, metrics_dir / f"{args.output_name}.json")

    age_mask = preds["age_mask"].astype(bool)
    if age_mask.any():
        y_true, q50 = preds["age"][age_mask], preds["q50"][age_mask]
        plot_age_scatter(y_true, q50, plots_dir / f"{args.output_name}_age_scatter.png")
        plot_error_histogram(y_true - q50, plots_dir / f"{args.output_name}_age_error_hist.png")
        bucket_report = metrics["age_error_by_bucket"]
        labels = [k for k, v in bucket_report.items() if v["count"] > 0]
        if labels:
            coverage_by_bucket = np.array([
                interval_coverage(
                    y_true[(y_true >= _bucket_lo(k)) & (y_true < _bucket_hi(k))],
                    preds["q10"][age_mask][(y_true >= _bucket_lo(k)) & (y_true < _bucket_hi(k))],
                    preds["q90"][age_mask][(y_true >= _bucket_lo(k)) & (y_true < _bucket_hi(k))],
                )
                if ((y_true >= _bucket_lo(k)) & (y_true < _bucket_hi(k))).sum() > 0 else np.nan
                for k in labels
            ])
            plot_interval_coverage(labels, coverage_by_bucket, 0.80, plots_dir / f"{args.output_name}_interval_coverage.png")

    gender_mask = preds["gender_mask"].astype(bool)
    if gender_mask.any():
        y_true_gender = preds["gender"][gender_mask].astype(int)
        predicted = preds["probs"][gender_mask].argmax(axis=1)
        cm = confusion_matrix(y_true_gender, predicted, num_classes=config["model"]["gender_head"]["num_classes"])
        plot_confusion_matrix(cm, config["model"]["gender_head"]["class_names"], plots_dir / f"{args.output_name}_confusion_matrix.png")

    if args.compare_knn:
        knn_path = Path(args.knn_path)
        if not knn_path.exists():
            logger.warning("No k-NN index at %s; run 'make build-knn' first.", knn_path)
        else:
            knn = KNNEmbeddingBaseline.load(knn_path)
            knn_metrics = _evaluate_knn(model, dataset, device, knn, confidence_threshold)
            table = build_parametric_vs_knn_table(metrics, knn_metrics)
            table.to_csv(REPO_ROOT / "outputs" / "knn" / "parametric_vs_knn.csv", index=False)
            logger.info("Saved parametric-vs-kNN comparison table")

    logger.info("Evaluation metrics: %s", {k: v for k, v in metrics.items() if not isinstance(v, dict)})
    return 0


def _bucket_lo(label: str) -> float:
    return float(label.split("-")[0])


def _bucket_hi(label: str) -> float:
    part = label.split("-")[1]
    return 1000.0 if part == "120+" else float(part)


@torch.no_grad()
def _evaluate_knn(model, dataset, device, knn: KNNEmbeddingBaseline, confidence_threshold: float) -> dict:
    from torch.utils.data import DataLoader

    loader = DataLoader(dataset, batch_size=64, shuffle=False)
    age_embeds, gender_embeds = [], []
    ages, age_masks, genders, gender_masks = [], [], [], []
    start = time.time()
    n = 0
    for batch in loader:
        images = batch["image"].to(device)
        emb = model.encode(images)
        age_embeds.append(emb["age_embedding"].cpu().numpy())
        gender_embeds.append(emb["gender_embedding"].cpu().numpy())
        ages.append(batch["age"].numpy())
        age_masks.append(batch["age_mask"].numpy())
        genders.append(batch["gender_label"].numpy())
        gender_masks.append(batch["gender_mask"].numpy())
        n += len(images)
    latency_ms_per_image = (time.time() - start) / max(1, n) * 1000.0

    age_embeds = np.concatenate(age_embeds)
    gender_embeds = np.concatenate(gender_embeds)
    ages, age_masks = np.concatenate(ages), np.concatenate(age_masks).astype(bool)
    genders, gender_masks = np.concatenate(genders), np.concatenate(gender_masks).astype(bool)

    metrics = {"latency_ms_per_image": latency_ms_per_image}
    if age_masks.any():
        result = knn.predict_age(age_embeds[age_masks])
        y_true = ages[age_masks]
        metrics.update({
            "age_mae": age_mae(y_true, result.q50), "age_rmse": age_rmse(y_true, result.q50),
            "interval_coverage": interval_coverage(y_true, result.q10, result.q90),
            "mean_interval_width": mean_interval_width(result.q10, result.q90),
        })
    if gender_masks.any():
        result = knn.predict_gender(gender_embeds[gender_masks], confidence_threshold)
        y_true_gender = genders[gender_masks].astype(int)
        metrics.update({
            "gender_accuracy": gender_accuracy(y_true_gender, result.predicted_class, result.abstain),
            "abstention_rate": abstention_rate(result.abstain),
            "mean_confidence": float(result.confidence.mean()),
        })
    return metrics


if __name__ == "__main__":
    raise SystemExit(main())
