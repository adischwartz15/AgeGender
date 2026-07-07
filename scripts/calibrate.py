#!/usr/bin/env python
"""CLI: fit split conformal calibration for age intervals using the dedicated calibration split.

Per the 4-way split protocol (train/validation/calibration/test), this
intentionally does NOT reuse the validation split (which is reserved for
early stopping / checkpoint selection) -- fitting conformal intervals on
the same data used for model/checkpoint selection would let that data
influence both decisions, muddying the calibration guarantee. The test
split is only ever used afterward, once, to report the calibration's
effect (coverage/width before vs. after).

Usage:
    python scripts/calibrate.py --checkpoint checkpoints/multitask_best_balanced_score.pt
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
from src.evaluation.calibration import evaluate_calibration_effect, fit_and_save_calibration
from src.inference.artifacts import load_model_checkpoint
from src.utils.config import REPO_ROOT
from src.utils.io import save_json
from src.utils.logging import get_logger

logger = get_logger("scripts.calibrate")


@torch.no_grad()
def _predict_age(model, dataset, device, batch_size=64):
    from torch.utils.data import DataLoader

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    q10s, q90s, ages, masks = [], [], [], []
    for batch in loader:
        images = batch["image"].to(device)
        outputs = model(images)
        q10s.append(outputs["age_output"]["q10"].cpu().numpy())
        q90s.append(outputs["age_output"]["q90"].cpu().numpy())
        ages.append(batch["age"].numpy())
        masks.append(batch["age_mask"].numpy())
    return np.concatenate(q10s), np.concatenate(q90s), np.concatenate(ages), np.concatenate(masks).astype(bool)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--alpha", type=float, default=None, help="Target miscoverage (default from configs/training.yaml)")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, config, _ = load_model_checkpoint(args.checkpoint, device)
    alpha = args.alpha if args.alpha is not None else config["calibration"]["alpha"]

    splits_path = REPO_ROOT / config["paths"]["splits_dir"] / "full_metadata_with_splits.csv"
    if not splits_path.exists():
        logger.error("No prepared split found at %s.", splits_path)
        return 1
    df = pd.read_csv(splits_path)

    calibration_dataset = FaceMultiTaskDataset(
        df[df["split"] == "calibration"], EvalTransform(config["dataset"]["image_size"])
    )
    if len(calibration_dataset) == 0:
        logger.error(
            "Calibration split is empty. Re-run 'make prepare-data' with the current "
            "4-way split config (configs/data.yaml: split.calibration_fraction) if this "
            "split was prepared before the calibration split existed."
        )
        return 1
    q10_cal, q90_cal, ages_cal, mask_cal = _predict_age(model, calibration_dataset, device)
    if not mask_cal.any():
        logger.error("Calibration split has no age labels; cannot calibrate.")
        return 1

    calibration_dir = REPO_ROOT / config["calibration"]["output_dir"]
    artifact = fit_and_save_calibration(ages_cal[mask_cal], q10_cal[mask_cal], q90_cal[mask_cal], alpha, calibration_dir)
    logger.info("Calibration artifact: %s", artifact)

    test_dataset = FaceMultiTaskDataset(df[df["split"] == "test"], EvalTransform(config["dataset"]["image_size"]))
    q10_test, q90_test, ages_test, mask_test = _predict_age(model, test_dataset, device)
    if mask_test.any():
        effect = evaluate_calibration_effect(ages_test[mask_test], q10_test[mask_test], q90_test[mask_test], artifact["offset"])
        logger.info("Calibration effect on test set: %s", effect)
        save_json(effect, calibration_dir / "calibration_test_effect.json")

    print(f"Saved calibration artifact to {calibration_dir}/conformal_calibration.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
