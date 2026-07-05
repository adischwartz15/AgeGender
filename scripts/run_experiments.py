#!/usr/bin/env python
"""CLI: run the full config-driven architecture ablation suite (Experiments A-F).

See configs/experiments.yaml for what each experiment tests. Experiment E
(parametric vs kNN) does not train a new model -- run scripts/build_knn_index.py
and scripts/evaluate.py --compare-knn against Experiment D's checkpoint instead.
Experiment F (pretrained vs scratch) is skipped automatically with a clear
message if no self-supervised checkpoint exists yet (run scripts/pretrain.py first).

Usage:
    python scripts/run_experiments.py [--only exp_a_separate,exp_c_shared_adapters]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from train import run_training  # noqa: E402

from src.utils.config import REPO_ROOT, load_config, load_full_config  # noqa: E402
from src.utils.io import save_json  # noqa: E402
from src.utils.logging import get_logger  # noqa: E402

logger = get_logger("scripts.run_experiments")

NO_TRAINING_EXPERIMENTS = {"exp_e_parametric_vs_knn"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", default=None, help="Comma-separated experiment names to run (default: all)")
    args = parser.parse_args()

    experiments_cfg = load_config(REPO_ROOT / "configs" / "experiments.yaml")["experiments"]
    run_order = load_config(REPO_ROOT / "configs" / "experiments.yaml")["run_order"]
    if args.only:
        run_order = [name for name in run_order if name in set(args.only.split(","))]

    results = {}
    for name in run_order:
        spec = experiments_cfg[name]
        if name in NO_TRAINING_EXPERIMENTS:
            logger.info("Skipping '%s' (no training step: %s)", name, spec["description"].strip())
            continue

        base_experiment = spec.get("base_experiment")
        if base_experiment:
            logger.info("Experiment '%s' reuses '%s' as its base checkpoint; skipping separate training.", name, base_experiment)
            continue

        overrides = spec.get("overrides", {})
        if name == "exp_f_pretrained_vs_scratch":
            checkpoint_path = REPO_ROOT / overrides.get("model", {}).get("pretrained_checkpoint", "")
            if not checkpoint_path.exists():
                logger.warning(
                    "Skipping '%s': pretrained checkpoint '%s' not found. Run 'make pretrain' first.",
                    name, checkpoint_path,
                )
                continue

        logger.info("=== Running %s ===\n%s", name, spec["description"].strip())
        config = load_full_config(overrides=overrides)
        try:
            result = run_training(config, experiment_name=name)
            results[name] = result
        except FileNotFoundError as exc:
            logger.error(str(exc))
            return 1

    output_dir = REPO_ROOT / "outputs" / "architecture_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)
    save_json(
        {name: r["parameter_breakdown"] for name, r in results.items()}, output_dir / "parameter_comparison.json"
    )
    logger.info("Finished %d experiment(s).", len(results))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
