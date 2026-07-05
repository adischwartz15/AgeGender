#!/usr/bin/env python
"""CLI: run the deep architecture analysis (gradient interference, representation
similarity, embedding visualizations) and assemble the final Markdown report.

Usage:
    python scripts/generate_architecture_report.py --checkpoint checkpoints/exp_c_shared_adapters_best_balanced_score.pt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.dataset import FaceMultiTaskDataset
from src.data.transforms import EvalTransform
from src.evaluation.architecture_analysis import (
    compute_gradient_cosine_similarity, compute_representation_similarity, extract_embeddings, reduce_embeddings,
)
from src.evaluation.comparison import build_architecture_ablation_table
from src.evaluation.reports import save_report
from src.inference.artifacts import load_model_checkpoint
from src.utils.config import REPO_ROOT
from src.utils.io import save_json
from src.utils.logging import get_logger
from src.utils.visualization import plot_embedding_scatter, plot_gradient_cosine_similarity

logger = get_logger("scripts.generate_architecture_report")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="A shared-backbone checkpoint (Experiment B/C/D)")
    parser.add_argument("--reduction", default="pca", choices=["pca", "tsne"])
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, config, _ = load_model_checkpoint(args.checkpoint, device)

    splits_path = REPO_ROOT / config["paths"]["splits_dir"] / "full_metadata_with_splits.csv"
    if not splits_path.exists():
        logger.error("No prepared split found at %s.", splits_path)
        return 1
    df = pd.read_csv(splits_path)
    val_dataset = FaceMultiTaskDataset(df[df["split"] == "val"], EvalTransform(config["dataset"]["image_size"]))

    from torch.utils.data import DataLoader

    loader = DataLoader(val_dataset, batch_size=32, shuffle=True)

    output_dir = REPO_ROOT / "outputs" / "architecture_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    if model.architecture != "separate":
        similarities = compute_gradient_cosine_similarity(model, loader, device)
        if len(similarities) > 0:
            grad_summary = {
                "mean": float(np.mean(similarities)), "std": float(np.std(similarities)),
                "n_batches": int(len(similarities)), "architecture": model.architecture,
            }
            save_json(grad_summary, output_dir / "gradient_cosine_similarity.json")
            np.save(output_dir / "gradient_cosine_similarity_samples.npy", similarities)
            plot_gradient_cosine_similarity(
                similarities, output_dir / "gradient_cosine_similarity.png",
                f"Gradient cosine similarity (age vs gender), architecture={model.architecture}",
            )
            logger.info("Gradient cosine similarity: %s", grad_summary)
        else:
            logger.warning("No batches had both age and gender labels; gradient interference not computed.")
    else:
        logger.info("Architecture is 'separate' (Experiment A); gradient interference is not defined.")

    embeddings = extract_embeddings(model, loader, device)
    if embeddings["shared_embedding"] is not None:
        cka = compute_representation_similarity(
            embeddings["shared_embedding"], embeddings["age_embedding"], embeddings["gender_embedding"]
        )
        save_json(cka, output_dir / "representation_similarity.json")
        logger.info("Representation similarity (linear CKA): %s", cka)

        age_valid = embeddings["age_mask"]
        if age_valid.sum() > 10:
            coords = reduce_embeddings(embeddings["shared_embedding"][age_valid], method=args.reduction)
            age_values = embeddings["age"][age_valid]
            buckets = np.digitize(age_values, [10, 20, 30, 40, 50, 60, 70, 80])
            plot_embedding_scatter(
                coords, buckets, {i: label for i, label in enumerate([
                    "0-10", "10-20", "20-30", "30-40", "40-50", "50-60", "60-70", "70-80", "80+"
                ])}, output_dir / f"embedding_{args.reduction}_age_buckets.png",
                f"Shared embedding ({args.reduction.upper()}), colored by age bucket",
            )
        gender_valid = embeddings["gender_mask"]
        if gender_valid.sum() > 10:
            coords = reduce_embeddings(embeddings["shared_embedding"][gender_valid], method=args.reduction)
            gender_values = embeddings["gender_label"][gender_valid].astype(int)
            class_names = config["model"]["gender_head"]["class_names"]
            plot_embedding_scatter(
                coords, gender_values, dict(enumerate(class_names)),
                output_dir / f"embedding_{args.reduction}_gender_labels.png",
                f"Shared embedding ({args.reduction.upper()}), colored by dataset gender label",
            )
    else:
        logger.info("Architecture 'separate' has no single shared embedding to visualize.")

    # Assemble ablation table from any per-experiment metrics already on disk.
    metrics_dir = REPO_ROOT / "outputs" / "metrics"
    experiment_results = {}
    for param_file in metrics_dir.glob("*_parameter_breakdown.json"):
        exp_name = param_file.name.replace("_parameter_breakdown.json", "")
        import json

        with open(param_file) as fh:
            breakdown = json.load(fh)
        timing_file = metrics_dir / f"{exp_name}_timing.json"
        timing = json.load(open(timing_file)) if timing_file.exists() else {}
        experiment_results[exp_name] = {"parameter_breakdown": breakdown, **timing}
    if experiment_results:
        table = build_architecture_ablation_table(experiment_results)
        table.to_csv(output_dir / "ablation_table.csv", index=False)

    report_path = save_report(REPO_ROOT / "outputs", REPO_ROOT / "docs")
    logger.info("Saved architecture analysis report to %s", report_path)
    print(f"Saved architecture analysis artifacts to {output_dir} and report to {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
