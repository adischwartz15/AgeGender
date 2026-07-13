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
    python scripts/run_transfer_learning.py --seeds 42,123,2026
    python scripts/run_transfer_learning.py --seeds 42,123,2026 --resume --skip-completed --sync-after-epoch
    python scripts/run_transfer_learning.py --seeds 42,123,2026 --evaluate-only

Persistence / resume (see docs/transfer_learning.md "Persistent artifacts"
and ``src/training/persistent_artifacts.py::PersistentArtifactManager``):

    --resume             Resume an incomplete seed from its latest valid
                          checkpoint instead of restarting from scratch.
    --skip-completed      Reuse an already-complete seed's saved metrics
                          instead of retraining it.
    --sync-after-epoch    Mirror checkpoints/metrics to --persistent-root
                          after every epoch (not just at seed completion) --
                          minimizes work lost to a mid-epoch disconnect.
    --storage-root PATH   Local working root for checkpoints/metrics
                          (default: checkpoints/transfer_learning).
    --persistent-root PATH  Persistent mirror root (a mounted Google Drive
                          folder, /kaggle/working, or any durable path) --
                          restored from at startup, synced to during/after
                          training.
    --evaluate-only        Never trains -- evaluates/reuses existing
                          checkpoints only, then rebuilds Table B.
"""

from __future__ import annotations

import argparse
import datetime
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluate import evaluate_checkpoint  # noqa: E402

from src.evaluation.comparison import aggregate_seed_metrics, build_transfer_learning_table  # noqa: E402
from src.training.persistent_artifacts import (  # noqa: E402
    CorruptedCheckpointError, PersistentArtifactManager, SeedCompletionInfo, build_summary_archive, sha256_file,
)
from src.utils.config import CONFIG_DIR, REPO_ROOT, _deep_merge, _load_yaml, load_config  # noqa: E402
from src.utils.experiment_paths import experiment_paths  # noqa: E402
from src.utils.provenance import dependency_versions, git_commit_sha  # noqa: E402
from src.utils.io import file_sha256, save_json  # noqa: E402
from src.utils.logging import get_logger  # noqa: E402
from src.utils.seed import set_global_seed

logger = get_logger("scripts.run_transfer_learning")

VOLO_EXPERIMENT_NAME = "volo_d1_face_only_pretrained"
DEFAULT_LOCAL_ROOT = REPO_ROOT / "checkpoints" / "transfer_learning"


def _local_root(storage_root: str | None) -> Path:
    return Path(storage_root) if storage_root else DEFAULT_LOCAL_ROOT


def _persistent_root(persistent_root: str | None) -> Path | None:
    return Path(persistent_root) if persistent_root else None


def _split_sha256() -> str | None:
    split_path = REPO_ROOT / "data" / "splits" / "full_metadata_with_splits.csv"
    return file_sha256(split_path) if split_path.exists() else None


def refresh_summary_archive(args: argparse.Namespace, extra_files: list[Path] | None = None) -> Path:
    """Rebuild the lightweight ``transfer_learning_summary.zip`` (manifests/
    metrics/plots/configs/completion markers -- never checkpoints, see
    ``build_summary_archive``) and mirror it to ``--persistent-root`` if
    configured. Called at seed-completion and full-run-completion
    boundaries only (never per-epoch)."""
    import os
    import shutil

    experiment_root = _local_root(args.storage_root) / VOLO_EXPERIMENT_NAME
    archive_path = build_summary_archive(
        experiment_root, experiment_root / "transfer_learning_summary.zip", extra_files=extra_files,
    )
    persistent_root = _persistent_root(args.persistent_root)
    if persistent_root is not None:
        dest = persistent_root / VOLO_EXPERIMENT_NAME / "transfer_learning_summary.zip"
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp_dest = dest.with_suffix(dest.suffix + ".tmp")
        shutil.copy2(archive_path, tmp_dest)
        os.replace(tmp_dest, dest)
    return archive_path


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




def train_and_evaluate_volo_smoke() -> dict | None:
    """The tiny, non-scientific ``--smoke`` profile: single seed, no
    persistence layer, no resume -- a fast CPU pipeline check, not a
    resumable long-running experiment (see ``transfer_learning_smoke`` in
    ``configs/transfer_learning.yaml``)."""
    import pandas as pd

    from src.data.dataset import build_datasets
    from src.models.pretrained_volo import build_pretrained_volo_model
    from src.training.transfer_trainer import TransferTrainer

    seed = 42
    run_name = f"{VOLO_EXPERIMENT_NAME}_smoke"
    checkpoint_dir = REPO_ROOT / "checkpoints" / "transfer_learning" / "_smoke"
    output_dir = REPO_ROOT / "results" / "transfer_learning" / VOLO_EXPERIMENT_NAME / "_smoke"

    config = load_transfer_learning_config(
        seed, smoke=True,
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


def run_volo_seed(seed: int, args: argparse.Namespace) -> dict | None:
    """Train (if needed) and evaluate one VOLO seed, honoring
    ``--resume``/``--skip-completed``/``--evaluate-only``/``--persistent-root``.

    Returns the real test-set metrics dict (never fabricated), or ``None``
    if no prepared split exists, no checkpoint is available for
    ``--evaluate-only``, or training could not produce a checkpoint.
    """
    import pandas as pd

    from src.data.dataset import build_datasets
    from src.models.pretrained_volo import build_pretrained_volo_model
    from src.training.transfer_trainer import TransferTrainer

    run_name = f"{VOLO_EXPERIMENT_NAME}_seed{seed}"
    # Legacy flat paths -- TransferTrainer still writes its own
    # {run_name}_best_{metric}.pt files here (unchanged, for
    # evaluate.py/inference backward compatibility); the
    # PersistentArtifactManager below additionally maintains the isolated,
    # resumable seed_<seed>/ tree these paths do not.
    legacy_checkpoint_dir = REPO_ROOT / "checkpoints" / "transfer_learning" / f"seed_{seed}"
    legacy_output_dir = REPO_ROOT / "results" / "transfer_learning" / VOLO_EXPERIMENT_NAME / f"seed_{seed}"

    config = load_transfer_learning_config(
        seed, smoke=False,
        overrides={"paths": {"checkpoint_dir": str(legacy_checkpoint_dir), "output_dir": str(legacy_output_dir)}},
    )
    model_id = config["model"]["volo"]["model_id"]
    pretrained_source = config["model"]["volo"]["pretrained_source"]
    split_sha256 = _split_sha256()
    git_sha = git_commit_sha()

    manager = PersistentArtifactManager(
        VOLO_EXPERIMENT_NAME, seed, local_root=_local_root(args.storage_root),
        persistent_root=_persistent_root(args.persistent_root), sync_after_epoch=args.sync_after_epoch,
    )
    if _persistent_root(args.persistent_root) is not None:
        manager.restore_seed()

    if args.skip_completed and manager.is_seed_complete(
        expected_split_sha256=split_sha256, expected_model_id=model_id, expected_pretrained_source=pretrained_source,
    ):
        completion = manager.load_completion()
        logger.info("seed=%d: COMPLETE -- reusing existing metrics (not retraining).", seed)
        return completion.get("test_metrics")

    if args.evaluate_only:
        best_checkpoint = manager.checkpoints_dir / "best.pt"
        if not best_checkpoint.exists():
            legacy = legacy_checkpoint_dir / f"{run_name}_best_balanced_score.pt"
            best_checkpoint = legacy if legacy.exists() else None
        if best_checkpoint is None:
            logger.warning("seed=%d: --evaluate-only but no existing checkpoint found -- skipping.", seed)
            return None
        logger.info("seed=%d: --evaluate-only -- evaluating %s (never trains).", seed, best_checkpoint)
        return evaluate_checkpoint(str(best_checkpoint), output_name=f"{run_name}_test_metrics")

    splits_path = REPO_ROOT / config["paths"]["splits_dir"] / "full_metadata_with_splits.csv"
    if not splits_path.exists():
        logger.error("No prepared split found at %s. Run 'make prepare-data' first.", splits_path)
        return None

    resume_state = None
    if args.resume:
        try:
            resume_state = manager.find_latest_valid_checkpoint()
        except CorruptedCheckpointError:
            logger.exception(
                "seed=%d: both last.pt and previous_last.pt are corrupted -- refusing to silently "
                "restart from scratch. Inspect the checkpoint directory manually.", seed,
            )
            raise
        if resume_state is not None:
            logger.info(
                "seed=%d: INCOMPLETE -- resuming from stage=%r epoch=%s.",
                seed, resume_state.get("training_stage"), resume_state.get("epoch"),
            )
        else:
            logger.info("seed=%d: NOT STARTED -- training from scratch.", seed)

    set_global_seed(seed)
    df = pd.read_csv(splits_path)
    model = build_pretrained_volo_model(config)
    train_transform, eval_transform = model.build_transforms()
    datasets = build_datasets(df, train_transform, eval_transform)

    trainer = TransferTrainer(
        model, config, datasets["train"], datasets["validation"],
        device=_resolve_device(), checkpoint_dir=legacy_checkpoint_dir, experiment_name=run_name,
        output_dir=legacy_output_dir, artifact_manager=manager, resume_state=resume_state,
        git_commit_sha=git_sha, split_sha256=split_sha256,
    )
    trainer.train()

    best_checkpoint_path = manager.checkpoints_dir / "best.pt"
    if not best_checkpoint_path.exists():
        legacy = legacy_checkpoint_dir / f"{run_name}_best_balanced_score.pt"
        if legacy.exists():
            best_checkpoint_path = legacy
        else:
            logger.warning("Checkpoint not found after training; skipping evaluation.")
            return None

    metrics = evaluate_checkpoint(str(best_checkpoint_path), output_name=f"{run_name}_test_metrics")
    if metrics is not None:
        manager.save_metrics("test_metrics", metrics)
        completion = SeedCompletionInfo(
            seed=seed, status="complete", best_checkpoint=str(best_checkpoint_path), test_metrics=metrics,
            completed_at=datetime.datetime.now(datetime.timezone.utc).isoformat(), split_sha256=split_sha256,
            git_commit_sha=git_sha, checkpoint_sha256=sha256_file(best_checkpoint_path),
            model_id=model_id, pretrained_source=pretrained_source,
        )
        manager.on_seed_complete(completion)
        logger.info("seed=%d: marked COMPLETE.", seed)
        refresh_summary_archive(args)
    return metrics


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


def _atomic_save_json(data: dict, path: Path) -> None:
    import os

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    save_json(data, tmp_path)
    os.replace(tmp_path, path)


def _atomic_write_csv(df, path: Path) -> None:
    import os

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp_path, index=False)
    os.replace(tmp_path, path)


def print_seed_status(seeds: list[int], args: argparse.Namespace) -> None:
    """Startup status display: 'Seed 42: COMPLETE -- reused' / 'Seed 123:
    INCOMPLETE -- resuming from epoch 11, Stage 2' / 'Seed 2026: NOT STARTED'.
    Mirrors the status cells in both notebooks' persistence sections."""
    from src.training.persistent_artifacts import format_status_line, seed_status_report

    split_sha256 = _split_sha256()
    for seed in seeds:
        status = seed_status_report(
            VOLO_EXPERIMENT_NAME, seed, _local_root(args.storage_root), _persistent_root(args.persistent_root),
            expected_split_sha256=split_sha256,
        )
        logger.info(format_status_line(status))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seeds", default="42,123,2026", help="Comma-separated seeds for VOLO + baseline")
    parser.add_argument("--baseline-experiment", default="exp_d_shared_adapters_learned_balance")
    parser.add_argument("--only", choices=["volo", "baseline", "both"], default="both")
    parser.add_argument(
        "--evaluate-only", action="store_true",
        help="Never train -- evaluate/reuse existing checkpoints only, then rebuild Table B",
    )
    parser.add_argument("--smoke", action="store_true", help="Run the tiny non-scientific transfer_learning_smoke profile (single seed)")
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume an incomplete seed from its latest valid checkpoint instead of restarting",
    )
    parser.add_argument(
        "--skip-completed", action="store_true",
        help="Reuse an already-complete seed's saved metrics instead of retraining it",
    )
    parser.add_argument(
        "--sync-after-epoch", action="store_true",
        help="Mirror checkpoints/metrics to --persistent-root after every epoch, not just at seed completion",
    )
    parser.add_argument("--storage-root", default=None, help="Local working root (default: checkpoints/transfer_learning)")
    parser.add_argument(
        "--persistent-root", default=None,
        help="Persistent mirror root (e.g. a mounted Google Drive folder or /kaggle/working)",
    )
    args = parser.parse_args()

    if args.smoke:
        if args.only in ("volo", "both"):
            train_and_evaluate_volo_smoke()
        logger.info("--smoke run complete (non-scientific; not written to Table B/table_b_manifest.json).")
        return 0

    seeds = [int(s.strip()) for s in args.seeds.split(",")]
    print_seed_status(seeds, args)

    volo_metrics_by_seed: dict[int, dict] = {}
    baseline_metrics_by_seed: dict[int, dict] = {}

    if args.only in ("volo", "both"):
        for seed in seeds:
            metrics = run_volo_seed(seed, args)
            if metrics is not None:
                volo_metrics_by_seed[seed] = metrics

    if args.only in ("baseline", "both"):
        for seed in seeds:
            metrics = collect_baseline_metrics(args.baseline_experiment, seed)
            if metrics is not None:
                baseline_metrics_by_seed[seed] = metrics

    volo_metrics_per_seed = list(volo_metrics_by_seed.values())
    baseline_metrics_per_seed = list(baseline_metrics_by_seed.values())
    missing_volo_seeds = [s for s in seeds if s not in volo_metrics_by_seed]
    missing_baseline_seeds = [s for s in seeds if s not in baseline_metrics_by_seed]

    if len(seeds) < 3 or len(volo_metrics_per_seed) < 2 or len(baseline_metrics_per_seed) < 2:
        logger.warning(
            "Table B will be reported as single-seed (no variance estimate): "
            "volo_seeds=%d baseline_seeds=%d of %d requested. missing_volo_seeds=%s missing_baseline_seeds=%s "
            "-- a missing baseline/VOLO seed is reported explicitly, never silently substituted with another seed's result.",
            len(volo_metrics_per_seed), len(baseline_metrics_per_seed), len(seeds),
            missing_volo_seeds, missing_baseline_seeds,
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
        volo_seeds_with_metrics = list(volo_metrics_by_seed)
        volo_breakdown_path = (
            REPO_ROOT / "checkpoints" / "transfer_learning" / f"seed_{volo_seeds_with_metrics[0]}"
            / f"{VOLO_EXPERIMENT_NAME}_seed{volo_seeds_with_metrics[0]}_parameter_breakdown.json"
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
    _atomic_write_csv(table_b, output_dir / "table_b.csv")

    # Incremental, recoverable per-seed index -- lets Table B be rebuilt
    # (--evaluate-only --skip-completed) from saved metrics alone, without
    # retraining anything, and records exactly which seed contributed which
    # row so a missing seed is never silently blended into another's slot.
    seed_metrics_index = {
        str(seed): {"volo": volo_metrics_by_seed.get(seed), "baseline": baseline_metrics_by_seed.get(seed)}
        for seed in seeds
    }
    _atomic_save_json(seed_metrics_index, output_dir / "seed_metrics_index.json")

    split_path = REPO_ROOT / "data" / "splits" / "full_metadata_with_splits.csv"
    manifest = {
        "seeds_requested": seeds,
        "volo_seeds_completed": sorted(volo_metrics_by_seed),
        "baseline_seeds_completed": sorted(baseline_metrics_by_seed),
        "missing_volo_seeds": missing_volo_seeds,
        "missing_baseline_seeds": missing_baseline_seeds,
        "single_seed_no_variance_estimate": len(volo_metrics_per_seed) < 2 or len(baseline_metrics_per_seed) < 2,
        "git_commit_sha": git_commit_sha(),
        "dependency_versions": dependency_versions(),
        "split_sha256": file_sha256(split_path) if split_path.exists() else None,
        "smoke": args.smoke,
        "artifact_root": str(_local_root(args.storage_root)),
        "persistence_backend": "persistent_root" if _persistent_root(args.persistent_root) else "local_only",
    }
    # Atomic overwrite: when the last missing seed completes, this replaces
    # the previous (partial) manifest in one os.replace(), never leaving a
    # reader observing a half-written file.
    _atomic_save_json(manifest, output_dir / "table_b_manifest.json")
    logger.info("Wrote Table B (%d row(s)) to %s", len(rows), output_dir / "table_b.csv")

    if args.only in ("volo", "both"):
        refresh_summary_archive(args, extra_files=[
            output_dir / "table_b.csv", output_dir / "table_b_manifest.json", output_dir / "seed_metrics_index.json",
        ])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
