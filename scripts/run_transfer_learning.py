#!/usr/bin/env python
"""CLI: run the supplementary ImageNet-pretrained VOLO-D1 (face-only) transfer-learning
extension and build Table B (best from-scratch model vs. VOLO-D1).

This is entirely separate from ``scripts/run_experiments.py`` (the core,
from-scratch ablation suite) -- it is never called by that script, never
reads/writes ``configs/experiments.yaml``, and its outputs live under
``results/transfer_learning/`` and ``checkpoints/transfer_learning/``, never
under ``experiments/`` or the flat ``outputs/``/``checkpoints/`` directories
the core suite uses. Requires ``timm`` (see requirements-transfer.txt);
importing this script itself does not require timm (the import only
happens inside ``src/models/pretrained_volo.py``, lazily, when a VOLO model
is actually constructed).

The from-scratch baseline (default: exp_d_shared_adapters_learned_balance)
is never retrained here -- this script only evaluates its already-trained
checkpoint(s) (or reuses an already-computed test-metrics JSON, without
even re-running inference) through the exact same ``evaluate_checkpoint()``
function used for the VOLO checkpoint, so Table B's two rows come from a
byte-identical evaluation path.

Usage:
    python scripts/run_transfer_learning.py --smoke
    python scripts/run_transfer_learning.py --seeds 42,43,44
    python scripts/run_transfer_learning.py --seeds 42,43,44 --evaluate-only
"""

from __future__ import annotations

import argparse
import platform
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluate import evaluate_checkpoint  # noqa: E402

from src.evaluation.comparison import aggregate_seed_metrics, build_transfer_learning_table  # noqa: E402
from src.utils.config import CONFIG_DIR, REPO_ROOT, _deep_merge, _load_yaml, load_config  # noqa: E402
from src.utils.experiment_paths import experiment_paths  # noqa: E402
from src.utils.io import file_sha256, save_json  # noqa: E402
from src.utils.logging import get_logger  # noqa: E402
from src.utils.seed import set_global_seed

logger = get_logger("scripts.run_transfer_learning")

VOLO_EXPERIMENT_NAME = "volo_d1_face_only_pretrained"


def load_transfer_learning_config(seed: int, smoke: bool = False, overrides: dict | None = None) -> dict:
    """Merge default -> data -> model -> training -> transfer_learning.yaml, then
    (if ``smoke``) the ``transfer_learning_smoke`` block on top, then ``overrides``.

    ``load_config()`` applies ``.env``-derived overrides (e.g. a
    ``CHECKPOINT_DIR``/``OUTPUT_DIR`` set for the core from-scratch profile)
    with higher priority than any merged YAML file -- see
    ``src/utils/config.py``'s documented precedence. Left alone, that would
    silently redirect this experiment's isolated
    ``checkpoints/transfer_learning``/``results/transfer_learning`` paths
    back into the shared core ``checkpoints/``/``outputs/`` directories.
    This function re-asserts ``configs/transfer_learning.yaml``'s own
    ``paths`` block *after* ``load_config()`` returns, so those two paths
    are never influenced by an unrelated ``.env`` file.
    """
    transfer_learning_yaml = _load_yaml(CONFIG_DIR / "transfer_learning.yaml")
    merged = load_config(
        CONFIG_DIR / "data.yaml", CONFIG_DIR / "model.yaml", CONFIG_DIR / "training.yaml",
        CONFIG_DIR / "transfer_learning.yaml",
    )
    if smoke:
        merged = _deep_merge(merged, merged.get("transfer_learning_smoke", {}))
        merged = _deep_merge(merged, {"paths": transfer_learning_yaml.get("transfer_learning_smoke", {}).get("paths", {})})
    else:
        merged = _deep_merge(merged, {"paths": transfer_learning_yaml.get("paths", {})})
    merged = _deep_merge(merged, {"seed": seed, "training": {"seed": seed}})
    if overrides:
        merged = _deep_merge(merged, overrides)
    return merged


def _dependency_versions() -> dict:
    import torch

    versions = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
    }
    try:
        import timm

        versions["timm"] = timm.__version__
    except ImportError:
        versions["timm"] = None
    return versions


def _git_commit_sha() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, stderr=subprocess.DEVNULL,
        ).decode().strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def train_and_evaluate_volo(seed: int, smoke: bool = False) -> dict | None:
    """Train (Stage 1 + Stage 2) and evaluate one VOLO seed run. Returns the
    real test-set metrics dict (never fabricated), or None if no prepared
    split exists yet."""
    import pandas as pd

    from src.data.dataset import build_datasets
    from src.models.pretrained_volo import build_pretrained_volo_model
    from src.training.transfer_trainer import TransferTrainer

    run_name = f"{VOLO_EXPERIMENT_NAME}_smoke" if smoke else f"{VOLO_EXPERIMENT_NAME}_seed{seed}"
    checkpoint_dir = REPO_ROOT / "checkpoints" / "transfer_learning" / (
        "_smoke" if smoke else f"seed_{seed}"
    )
    output_dir = REPO_ROOT / "results" / "transfer_learning" / VOLO_EXPERIMENT_NAME / (
        "_smoke" if smoke else f"seed_{seed}"
    )

    config = load_transfer_learning_config(
        seed, smoke=smoke,
        overrides={"paths": {"checkpoint_dir": str(checkpoint_dir), "output_dir": str(output_dir)}},
    )
    set_global_seed(seed)

    splits_path = REPO_ROOT / config["paths"]["splits_dir"] / "full_metadata_with_splits.csv"
    if not splits_path.exists():
        logger.error("No prepared split found at %s. Run 'make prepare-data' first.", splits_path)
        return None

    df = pd.read_csv(splits_path)
    model = build_pretrained_volo_model(config)
    train_transform, eval_transform = model.build_transforms()
    datasets = build_datasets(df, train_transform, eval_transform)

    trainer = TransferTrainer(
        model, config, datasets["train"], datasets["validation"],
        device=_resolve_device(), checkpoint_dir=checkpoint_dir, experiment_name=run_name, output_dir=output_dir,
    )
    trainer.train()

    checkpoint_path = checkpoint_dir / f"{run_name}_best_balanced_score.pt"
    if not checkpoint_path.exists():
        logger.warning("Checkpoint '%s' not found after training; skipping evaluation.", checkpoint_path)
        return None

    return evaluate_checkpoint(str(checkpoint_path), output_name=f"{run_name}_test_metrics")


def _resolve_device() -> str:
    from src.utils.config import resolve_device

    return resolve_device("auto")


def collect_baseline_metrics(baseline_experiment: str, seed: int) -> dict | None:
    """Reuse (never retrain) the from-scratch baseline's already-trained
    checkpoint for one seed: reuse an already-computed test-metrics JSON if
    present, else run (real, not fabricated) evaluation on the existing
    checkpoint. Returns None if no checkpoint exists for this seed at all."""
    seed_paths = experiment_paths(baseline_experiment, seed)
    seed_checkpoint = seed_paths["checkpoint_dir"] / f"{baseline_experiment}_best_balanced_score.pt"
    flat_checkpoint = REPO_ROOT / "checkpoints" / f"{baseline_experiment}_best_balanced_score.pt"

    if seed_checkpoint.exists():
        checkpoint_path, output_name = seed_checkpoint, f"{baseline_experiment}_seed{seed}_test_metrics"
        existing_metrics = seed_paths["metrics_dir"] / f"{output_name}.json"
    elif seed == 42 and flat_checkpoint.exists():
        # The single default (non-multi-seed) training run's checkpoint,
        # produced by `make train`/`make experiments` at configs/default.yaml's
        # top-level seed (42) -- reused here as the seed=42 baseline row
        # rather than retraining.
        checkpoint_path, output_name = flat_checkpoint, f"{baseline_experiment}_test_metrics"
        existing_metrics = REPO_ROOT / "outputs" / "metrics" / f"{output_name}.json"
    else:
        logger.warning("No '%s' checkpoint found for seed=%d; skipping this seed.", baseline_experiment, seed)
        return None

    if existing_metrics.exists():
        from src.utils.io import load_json

        logger.info("Reusing existing test metrics at %s (not re-evaluating).", existing_metrics)
        return load_json(existing_metrics)

    logger.info("Evaluating existing '%s' checkpoint (seed=%d) -- not retraining.", baseline_experiment, seed)
    return evaluate_checkpoint(str(checkpoint_path), output_name=output_name)


def _assemble_row(
    model_label: str, category: str, initialization: str, backbone: str, adapters: str, loss_balancing: str,
    input_size: int, param_breakdown: dict, metrics_per_seed: list[dict],
) -> dict:
    keys = ("age_mae", "age_rmse", "age_cs5", "gender_accuracy", "gender_f1")
    aggregate = aggregate_seed_metrics(metrics_per_seed, keys=keys)
    row = {
        "model": model_label, "experiment_category": category, "initialization": initialization,
        "backbone": backbone, "adapters": adapters, "loss_balancing": loss_balancing, "input_size": input_size,
        "total_params": param_breakdown.get("total_parameters"),
        "trainable_params": param_breakdown.get("trainable_parameters", param_breakdown.get("total_parameters")),
        "n_seeds": aggregate.get("_n_seed_runs", 0),
    }
    for key in keys:
        stats = aggregate.get(key)
        row[key] = stats["mean"] if stats else None
        row[f"{key}_std"] = stats["std"] if stats else None
    return row


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", default="42,43,44", help="Comma-separated seeds for VOLO + baseline")
    parser.add_argument("--baseline-experiment", default="exp_d_shared_adapters_learned_balance")
    parser.add_argument("--only", choices=["volo", "baseline", "both"], default="both")
    parser.add_argument("--evaluate-only", action="store_true", help="Skip VOLO training; evaluate existing checkpoints only")
    parser.add_argument("--smoke", action="store_true", help="Run the tiny non-scientific transfer_learning_smoke profile (single seed)")
    args = parser.parse_args()

    if args.smoke:
        seeds = [42]
    else:
        seeds = [int(s.strip()) for s in args.seeds.split(",")]

    volo_metrics_per_seed, baseline_metrics_per_seed = [], []

    if args.only in ("volo", "both"):
        for seed in seeds:
            if args.evaluate_only:
                run_name = f"{VOLO_EXPERIMENT_NAME}_seed{seed}"
                checkpoint_dir = REPO_ROOT / "checkpoints" / "transfer_learning" / f"seed_{seed}"
                checkpoint_path = checkpoint_dir / f"{run_name}_best_balanced_score.pt"
                metrics = evaluate_checkpoint(str(checkpoint_path), output_name=f"{run_name}_test_metrics") \
                    if checkpoint_path.exists() else None
            else:
                metrics = train_and_evaluate_volo(seed, smoke=args.smoke)
            if metrics is not None:
                volo_metrics_per_seed.append(metrics)

    if args.only in ("baseline", "both") and not args.smoke:
        for seed in seeds:
            metrics = collect_baseline_metrics(args.baseline_experiment, seed)
            if metrics is not None:
                baseline_metrics_per_seed.append(metrics)

    if len(seeds) < 3 or len(volo_metrics_per_seed) < 2 or len(baseline_metrics_per_seed) < 2:
        logger.warning(
            "Table B will be reported as single-seed (no variance estimate): "
            "volo_seeds=%d baseline_seeds=%d of %d requested.",
            len(volo_metrics_per_seed), len(baseline_metrics_per_seed), len(seeds),
        )

    rows = []
    if baseline_metrics_per_seed:
        # Parameter breakdown does not vary across seeds for the same
        # architecture -- read it from the existing per-experiment JSON if
        # present, otherwise omit (never fabricate a parameter count).
        from src.utils.io import load_json

        breakdown_path = REPO_ROOT / "outputs" / "metrics" / f"{args.baseline_experiment}_parameter_breakdown.json"
        param_breakdown = load_json(breakdown_path) if breakdown_path.exists() else {}
        rows.append(_assemble_row(
            model_label=args.baseline_experiment, category="core (from-scratch)", initialization="Random",
            backbone="custom_resnet18", adapters="shared_adapters", loss_balancing="learned_uncertainty",
            input_size=128, param_breakdown=param_breakdown, metrics_per_seed=baseline_metrics_per_seed,
        ))

    if volo_metrics_per_seed:
        volo_breakdown_path = (
            REPO_ROOT / "checkpoints" / "transfer_learning" / f"seed_{seeds[0]}"
            / f"{VOLO_EXPERIMENT_NAME}_seed{seeds[0]}_parameter_breakdown.json"
        )
        from src.utils.io import load_json

        param_breakdown = load_json(volo_breakdown_path) if volo_breakdown_path.exists() else {}
        rows.append(_assemble_row(
            model_label=VOLO_EXPERIMENT_NAME, category="supplementary (transfer learning)",
            initialization="ImageNet pretrained", backbone="volo_d1_224 (timm)", adapters="shared_adapters (reused)",
            loss_balancing="learned_uncertainty (reused)", input_size=224,
            param_breakdown=param_breakdown, metrics_per_seed=volo_metrics_per_seed,
        ))

    table_b = build_transfer_learning_table(rows)
    output_dir = REPO_ROOT / "results" / "transfer_learning"
    output_dir.mkdir(parents=True, exist_ok=True)
    table_b.to_csv(output_dir / "table_b.csv", index=False)

    split_path = REPO_ROOT / "data" / "splits" / "full_metadata_with_splits.csv"
    manifest = {
        "seeds_requested": seeds,
        "volo_seeds_completed": len(volo_metrics_per_seed),
        "baseline_seeds_completed": len(baseline_metrics_per_seed),
        "single_seed_no_variance_estimate": len(volo_metrics_per_seed) < 2 or len(baseline_metrics_per_seed) < 2,
        "git_commit_sha": _git_commit_sha(),
        "dependency_versions": _dependency_versions(),
        "split_sha256": file_sha256(split_path) if split_path.exists() else None,
        "smoke": args.smoke,
    }
    save_json(manifest, output_dir / "table_b_manifest.json")
    logger.info("Wrote Table B (%d row(s)) to %s", len(rows), output_dir / "table_b.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
