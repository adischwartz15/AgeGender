#!/usr/bin/env python
"""CLI: train and evaluate one experiment across multiple seeds, for mean +/- std reporting.

A single training run cannot distinguish "this architecture is better"
from "this particular random initialization got lucky". This script
reuses configs/experiments.yaml's overrides for a named experiment,
trains it once per seed (checkpoint/metric names suffixed
``_seed{N}``), evaluates each resulting checkpoint on the test split, and
leaves per-seed artifacts on disk for scripts/generate_final_report.py to
aggregate into a mean +/- std table. Nothing here computes or renders the
aggregate itself -- see src/evaluation/comparison.py:aggregate_seed_metrics.

Usage:
    python scripts/run_seeds.py --experiment exp_c_shared_adapters --seeds 42,43,44
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluate import evaluate_checkpoint  # noqa: E402
from train import run_training  # noqa: E402

from src.utils.config import REPO_ROOT, load_config, load_full_config  # noqa: E402
from src.utils.logging import get_logger  # noqa: E402

logger = get_logger("scripts.run_seeds")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", required=True, help="Experiment name from configs/experiments.yaml")
    parser.add_argument("--seeds", required=True, help="Comma-separated seeds, e.g. 42,43,44 (>=2 needed for a real std)")
    args = parser.parse_args()

    seeds = [int(s.strip()) for s in args.seeds.split(",")]
    if len(seeds) < 2:
        logger.warning(
            "Only %d seed(s) requested; mean +/- std across seeds needs at least 2 to be meaningful.", len(seeds)
        )

    experiments_cfg = load_config(REPO_ROOT / "configs" / "experiments.yaml")["experiments"]
    if args.experiment not in experiments_cfg:
        logger.error("Unknown experiment '%s'. See configs/experiments.yaml.", args.experiment)
        return 1
    base_overrides = experiments_cfg[args.experiment].get("overrides", {})

    for seed in seeds:
        run_name = f"{args.experiment}_seed{seed}"
        overrides = {**base_overrides, "seed": seed, "training": {**base_overrides.get("training", {}), "seed": seed}}
        logger.info("=== Training %s (seed=%d) ===", args.experiment, seed)
        config = load_full_config(overrides=overrides)
        try:
            run_training(config, experiment_name=run_name)
        except FileNotFoundError as exc:
            logger.error(str(exc))
            return 1

        checkpoint_path = REPO_ROOT / config["paths"]["checkpoint_dir"] / f"{run_name}_best_balanced_score.pt"
        if checkpoint_path.exists():
            evaluate_checkpoint(str(checkpoint_path), output_name=f"{run_name}_test_metrics")
        else:
            logger.warning("Checkpoint '%s' not found after training; skipping its evaluation.", checkpoint_path)

    logger.info(
        "Finished %d seed run(s) for '%s'. Run scripts/generate_final_report.py to aggregate mean +/- std.",
        len(seeds), args.experiment,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
