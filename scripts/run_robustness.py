#!/usr/bin/env python
"""CLI: deterministic robustness evaluation across corruption types and severities.

Usage:
    python scripts/run_robustness.py --checkpoint checkpoints/multitask_best_balanced_score.pt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.transforms import EvalTransform
from src.evaluation.robustness import apply_corruption, evaluate_condition, iter_corruption_configs
from src.inference.artifacts import load_model_checkpoint
from src.utils.config import REPO_ROOT, load_config
from src.utils.logging import get_logger
from src.utils.visualization import plot_robustness_curves

logger = get_logger("scripts.run_robustness")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--max-samples", type=int, default=500, help="Cap test-set samples for speed")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, config, _ = load_model_checkpoint(args.checkpoint, device)
    robustness_cfg = load_config(REPO_ROOT / "configs" / "robustness.yaml")["robustness"]

    splits_path = REPO_ROOT / config["paths"]["splits_dir"] / "full_metadata_with_splits.csv"
    if not splits_path.exists():
        logger.error("No prepared split found at %s.", splits_path)
        return 1
    df = pd.read_csv(splits_path)
    test_df = df[df["split"] == "test"].head(args.max_samples)

    transform = EvalTransform(config["dataset"]["image_size"])
    confidence_threshold = config["model"]["gender_head"].get("confidence_threshold", 0.80)
    seed = robustness_cfg.get("seed", 42)

    output_dir = REPO_ROOT / robustness_cfg.get("output_dir", "./outputs/robustness")
    output_dir.mkdir(parents=True, exist_ok=True)
    samples_dir = output_dir / "sample_corrupted_images"
    samples_dir.mkdir(parents=True, exist_ok=True)

    results = [evaluate_condition(model, test_df, transform, device, confidence_threshold, None, 0, None, seed)]
    for corruption_name, severity, param in iter_corruption_configs(robustness_cfg):
        logger.info("Evaluating %s severity=%d (param=%s)", corruption_name, severity, param)
        metrics = evaluate_condition(
            model, test_df, transform, device, confidence_threshold, corruption_name, severity, param, seed
        )
        results.append(metrics)

    from PIL import Image

    n_samples = robustness_cfg.get("num_samples_per_corruption_plot", 6)
    for corruption_name, severity, param in iter_corruption_configs(robustness_cfg):
        if severity != 1:
            continue
        for i, row in enumerate(test_df.head(n_samples).to_dict("records")):
            with Image.open(row["image_path"]) as img:
                corrupted = apply_corruption(img.convert("RGB"), corruption_name, param, seed=seed + i)
                corrupted.save(samples_dir / f"{corruption_name}_sample{i}.png")

    results_df = pd.DataFrame(results)
    results_df.to_csv(output_dir / "robustness_results.csv", index=False)

    for metric in ("age_mae", "gender_accuracy", "abstention_rate"):
        if metric in results_df.columns:
            corrupted_only = results_df[results_df["corruption"] != "clean"]
            if not corrupted_only.empty:
                plot_robustness_curves(corrupted_only, metric, output_dir / f"robustness_{metric}.png")

    summary_lines = ["# Robustness Evaluation Summary\n"]
    clean_row = results_df[results_df["corruption"] == "clean"].iloc[0].to_dict()
    summary_lines.append(f"**Clean baseline:** {clean_row}\n")
    for corruption_name in results_df["corruption"].unique():
        if corruption_name == "clean":
            continue
        subset = results_df[results_df["corruption"] == corruption_name]
        summary_lines.append(f"## {corruption_name}\n")
        summary_lines.append(subset.to_string(index=False) + "\n")
    (output_dir / "robustness_summary.md").write_text("\n".join(summary_lines), encoding="utf-8")

    logger.info("Saved robustness results to %s", output_dir)
    print(f"Saved robustness CSV/plots/summary to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
