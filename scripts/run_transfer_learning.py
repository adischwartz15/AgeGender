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
from src.training.progress import emit, format_multi_seed_preflight  # noqa: E402
from src.utils.config import CONFIG_DIR, REPO_ROOT, _deep_merge, _load_yaml, load_config  # noqa: E402
from src.utils.experiment_paths import experiment_paths  # noqa: E402
from src.utils.provenance import dependency_versions, git_commit_sha  # noqa: E402
from src.utils.io import file_sha256, save_json  # noqa: E402
from src.utils.logging import get_logger  # noqa: E402
from src.utils.seed import set_global_seed

logger = get_logger("scripts.run_transfer_learning")

VOLO_EXPERIMENT_NAME = "volo_d1_face_only_pretrained"
DEFAULT_LOCAL_ROOT = REPO_ROOT / "checkpoints" / "transfer_learning"

# Model-family registry: generalizes this script across the VOLO-D1
# extension and the pretrained-ResNet bridge baselines (T3, final-run
# hardening) without duplicating the persistence/resume/Table-B machinery
# below -- only the small model-specific bits (which config file, which
# config sub-key, which builder function, how to label the row in Table B)
# vary by family. "volo" is the default everywhere a --model-family isn't
# passed, so existing behavior/tests are unaffected by this generalization.
_MODEL_FAMILIES: dict[str, dict] = {
    "volo": {
        "experiment_name": VOLO_EXPERIMENT_NAME,
        "config_file": "transfer_learning.yaml",
        "model_config_key": "volo",
        "backbone_label": "volo_d1_224 (timm)",
        "input_size_label": 224,
        "category_label": "supplementary (transfer learning)",
    },
    "pretrained_resnet18": {
        "experiment_name": "pretrained_resnet18_face_only",
        "config_file": "pretrained_resnet18.yaml",
        "model_config_key": "pretrained_resnet",
        "backbone_label": "resnet18 (torchvision, ImageNet-pretrained)",
        "input_size_label": 224,
        "category_label": "supplementary (pretraining bridge baseline)",
    },
    "pretrained_resnet50": {
        "experiment_name": "pretrained_resnet50_face_only",
        "config_file": "pretrained_resnet50.yaml",
        "model_config_key": "pretrained_resnet",
        "backbone_label": "resnet50 (torchvision, ImageNet-pretrained)",
        "input_size_label": 224,
        "category_label": "supplementary (pretraining + architecture + capacity, NOT pretraining-isolated)",
    },
}


def _build_model_for_family(family: str, config: dict):
    if family == "volo":
        from src.models.pretrained_volo import build_pretrained_volo_model

        return build_pretrained_volo_model(config)
    from src.models.pretrained_resnet import build_pretrained_resnet_model

    return build_pretrained_resnet_model(config)


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

    experiment_name = _MODEL_FAMILIES[args.model_family]["experiment_name"]
    experiment_root = _local_root(args.storage_root) / experiment_name
    archive_path = build_summary_archive(
        experiment_root, experiment_root / "transfer_learning_summary.zip", extra_files=extra_files,
    )
    persistent_root = _persistent_root(args.persistent_root)
    if persistent_root is not None:
        dest = persistent_root / experiment_name / "transfer_learning_summary.zip"
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp_dest = dest.with_suffix(dest.suffix + ".tmp")
        shutil.copy2(archive_path, tmp_dest)
        os.replace(tmp_dest, dest)
    return archive_path


def load_transfer_learning_config(
    seed: int, smoke: bool = False, overrides: dict | None = None, family: str = "volo",
) -> dict:
    """Merge default -> data -> model -> training -> <family's config file>, then
    (if ``smoke``) the ``transfer_learning_smoke`` block on top, then ``overrides``.

    ``load_config()`` applies ``.env``-derived overrides (e.g. a
    ``CHECKPOINT_DIR``/``OUTPUT_DIR`` set for the core from-scratch profile)
    with higher priority than any merged YAML file -- see
    ``src/utils/config.py``'s documented precedence. Left alone, that would
    silently redirect this experiment's isolated
    ``checkpoints/transfer_learning``/``results/transfer_learning`` paths
    back into the shared core ``checkpoints/``/``outputs/`` directories.
    This function re-asserts the family config file's own ``paths`` block
    *after* ``load_config()`` returns, so those two paths are never
    influenced by an unrelated ``.env`` file.
    """
    config_filename = _MODEL_FAMILIES[family]["config_file"]
    family_yaml = _load_yaml(CONFIG_DIR / config_filename)
    merged = load_config(
        CONFIG_DIR / "data.yaml", CONFIG_DIR / "model.yaml", CONFIG_DIR / "training.yaml",
        CONFIG_DIR / config_filename,
    )
    if smoke:
        merged = _deep_merge(merged, merged.get("transfer_learning_smoke", {}))
        merged = _deep_merge(merged, {"paths": family_yaml.get("transfer_learning_smoke", {}).get("paths", {})})
    else:
        merged = _deep_merge(merged, {"paths": family_yaml.get("paths", {})})
    merged = _deep_merge(merged, {"seed": seed, "training": {"seed": seed}})
    if overrides:
        merged = _deep_merge(merged, overrides)
    return merged




def train_and_evaluate_volo_smoke(family: str = "volo") -> dict | None:
    """The tiny, non-scientific ``--smoke`` profile: single seed, no
    persistence layer, no resume -- a fast CPU pipeline check, not a
    resumable long-running experiment (see ``transfer_learning_smoke`` in
    the family's config file)."""
    import pandas as pd

    from src.data.dataset import build_datasets
    from src.training.transfer_trainer import TransferTrainer

    experiment_name = _MODEL_FAMILIES[family]["experiment_name"]
    seed = 42
    run_name = f"{experiment_name}_smoke"
    checkpoint_dir = REPO_ROOT / "checkpoints" / "transfer_learning" / "_smoke"
    output_dir = REPO_ROOT / "results" / "transfer_learning" / experiment_name / "_smoke"

    config = load_transfer_learning_config(
        seed, smoke=True, family=family,
        overrides={"paths": {"checkpoint_dir": str(checkpoint_dir), "output_dir": str(output_dir)}},
    )
    set_global_seed(seed)

    splits_path = REPO_ROOT / config["paths"]["splits_dir"] / "full_metadata_with_splits.csv"
    if not splits_path.exists():
        logger.error("No prepared split found at %s. Run 'make prepare-data' first.", splits_path)
        return None

    df = pd.read_csv(splits_path)
    model = _build_model_for_family(family, config)
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
    """Train (if needed) and evaluate one seed of ``args.model_family``
    (VOLO-D1, or a pretrained-ResNet bridge baseline -- see
    ``_MODEL_FAMILIES``), honoring
    ``--resume``/``--skip-completed``/``--evaluate-only``/``--persistent-root``.

    Returns the real test-set metrics dict (never fabricated), or ``None``
    if no prepared split exists, no checkpoint is available for
    ``--evaluate-only``, or training could not produce a checkpoint.
    """
    import pandas as pd

    from src.data.dataset import build_datasets
    from src.training.transfer_trainer import TransferTrainer

    family = args.model_family
    family_info = _MODEL_FAMILIES[family]
    experiment_name = family_info["experiment_name"]
    run_name = f"{experiment_name}_seed{seed}"
    # Legacy flat paths -- TransferTrainer still writes its own
    # {run_name}_best_{metric}.pt files here (unchanged, for
    # evaluate.py/inference backward compatibility); the
    # PersistentArtifactManager below additionally maintains the isolated,
    # resumable seed_<seed>/ tree these paths do not.
    legacy_checkpoint_dir = REPO_ROOT / "checkpoints" / "transfer_learning" / f"seed_{seed}"
    legacy_output_dir = REPO_ROOT / "results" / "transfer_learning" / experiment_name / f"seed_{seed}"

    config = load_transfer_learning_config(
        seed, smoke=False, family=family,
        overrides={"paths": {"checkpoint_dir": str(legacy_checkpoint_dir), "output_dir": str(legacy_output_dir)}},
    )
    model_config = config["model"][family_info["model_config_key"]]
    model_id = model_config["model_id"]
    pretrained_source = model_config["pretrained_source"]
    split_sha256 = _split_sha256()
    git_sha = git_commit_sha()

    manager = PersistentArtifactManager(
        experiment_name, seed, local_root=_local_root(args.storage_root),
        persistent_root=_persistent_root(args.persistent_root), sync_after_epoch=args.sync_after_epoch,
    )
    restored_files: list = []
    if _persistent_root(args.persistent_root) is not None:
        restored_files = manager.restore_seed()
    resume_source = "persistent" if restored_files else "local"

    manager.save_run_manifest(
        {
            "experiment_name": experiment_name, "seed": seed, "model_family": family,
            "model_id": model_id, "pretrained_source": pretrained_source,
            "split_sha256": split_sha256, "git_commit_sha": git_sha,
            "dependency_versions": dependency_versions(),
            "storage_root": str(_local_root(args.storage_root)),
            "persistent_root": str(_persistent_root(args.persistent_root)) if _persistent_root(args.persistent_root) else None,
            "started_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        }
    )

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
    model = _build_model_for_family(family, config)
    train_transform, eval_transform = model.build_transforms()
    datasets = build_datasets(df, train_transform, eval_transform)

    trainer = TransferTrainer(
        model, config, datasets["train"], datasets["validation"],
        device=_resolve_device(), checkpoint_dir=legacy_checkpoint_dir, experiment_name=run_name,
        output_dir=legacy_output_dir, artifact_manager=manager, resume_state=resume_state,
        resume_source=resume_source if resume_state is not None else None,
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

    # Also evaluate the best Stage-1-only (frozen-backbone) checkpoint
    # snapshot, if one was saved (see TransferTrainer._save_stage1_checkpoint_snapshot)
    # -- answers "how much of the performance comes from the pretrained
    # representation alone, before any fine-tuning?" without re-running
    # Stage 1 in isolation. Best-effort: never fails the main run if this
    # secondary evaluation has a problem.
    stage1_checkpoint_path = legacy_checkpoint_dir / f"{run_name}_best_stage1_frozen.pt"
    if stage1_checkpoint_path.exists():
        try:
            stage1_metrics = evaluate_checkpoint(str(stage1_checkpoint_path), output_name=f"{run_name}_stage1_test_metrics")
            if stage1_metrics is not None:
                manager.save_metrics("stage1_frozen_test_metrics", stage1_metrics)
                logger.info("seed=%d: saved Stage-1-only (frozen backbone) test metrics separately.", seed)
        except Exception:
            logger.exception("seed=%d: Stage-1-only checkpoint evaluation failed (non-fatal).", seed)

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
    keys = ("age_mae", "age_rmse", "age_median_ae", "age_cs5", "gender_accuracy", "gender_balanced_accuracy", "gender_f1")
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
    """Startup status display: per-seed detail lines ('Seed 42: COMPLETE --
    reused' / 'Seed 123: INCOMPLETE -- resuming from epoch 11, Stage 2' /
    'Seed 2026: NOT STARTED'), preceded by a categorized multi-seed plan
    summary (requested/completed/incomplete-resumable/missing/will-run-now)
    -- printed before any training starts, so it's immediately clear from
    the top of a run's output what this invocation is and isn't about to
    do. Mirrors the status cells in both notebooks' persistence sections."""
    from src.training.persistent_artifacts import format_status_line, seed_status_report

    experiment_name = _MODEL_FAMILIES[args.model_family]["experiment_name"]
    split_sha256 = _split_sha256()
    statuses = [
        seed_status_report(
            experiment_name, seed, _local_root(args.storage_root), _persistent_root(args.persistent_root),
            expected_split_sha256=split_sha256,
        )
        for seed in seeds
    ]
    completed, incomplete_resumable, missing, corrupted = [], [], [], []
    for seed, status in zip(seeds, statuses):
        {
            "COMPLETE": completed, "INCOMPLETE": incomplete_resumable,
            "NOT STARTED": missing, "CORRUPTED": corrupted,
        }[status["status"]].append(seed)

    if args.evaluate_only:
        will_run_now: list[int] = []
    elif args.skip_completed:
        will_run_now = [s for s in seeds if s not in completed]
    else:
        will_run_now = list(seeds)

    emit(
        format_multi_seed_preflight(
            experiment_name, requested_seeds=seeds, completed_seeds=completed,
            incomplete_resumable_seeds=incomplete_resumable, missing_seeds=missing,
            will_run_now_seeds=will_run_now,
        )
    )
    if corrupted:
        logger.warning(
            "seed(s) %s have a CORRUPTED checkpoint (both last.pt and previous_last.pt failed to "
            "load) -- these will raise CorruptedCheckpointError rather than silently restarting.",
            corrupted,
        )
    for status in statuses:
        logger.info(format_status_line(status))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seeds", default="42,123,2026", help="Comma-separated seeds for the pretrained model + baseline")
    parser.add_argument(
        "--model-family", choices=sorted(_MODEL_FAMILIES), default="volo",
        help="Which supplementary pretrained model to run: 'volo' (VOLO-D1, default), "
             "'pretrained_resnet18' (required bridge baseline), or 'pretrained_resnet50' (optional, "
             "combined pretraining+architecture+capacity comparison -- never described as isolating pretraining)",
    )
    parser.add_argument("--baseline-experiment", default="exp_d_shared_adapters_learned_balance")
    parser.add_argument("--only", choices=["volo", "baseline", "both"], default="both", help="'volo' here means the selected --model-family, kept for CLI backward compatibility")
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
    family_info = _MODEL_FAMILIES[args.model_family]

    if args.smoke:
        if args.only in ("volo", "both"):
            train_and_evaluate_volo_smoke(family=args.model_family)
        logger.info("--smoke run complete (non-scientific; not written to Table B/table_b_manifest.json).")
        return 0

    seeds = [int(s.strip()) for s in args.seeds.split(",")]
    print_seed_status(seeds, args)

    pretrained_metrics_by_seed: dict[int, dict] = {}
    baseline_metrics_by_seed: dict[int, dict] = {}

    if args.only in ("volo", "both"):
        for seed in seeds:
            metrics = run_volo_seed(seed, args)
            if metrics is not None:
                pretrained_metrics_by_seed[seed] = metrics

    if args.only in ("baseline", "both"):
        for seed in seeds:
            metrics = collect_baseline_metrics(args.baseline_experiment, seed)
            if metrics is not None:
                baseline_metrics_by_seed[seed] = metrics

    pretrained_metrics_per_seed = list(pretrained_metrics_by_seed.values())
    baseline_metrics_per_seed = list(baseline_metrics_by_seed.values())
    missing_pretrained_seeds = [s for s in seeds if s not in pretrained_metrics_by_seed]
    missing_baseline_seeds = [s for s in seeds if s not in baseline_metrics_by_seed]

    if len(seeds) < 3 or len(pretrained_metrics_per_seed) < 2 or len(baseline_metrics_per_seed) < 2:
        logger.warning(
            "Table B will be reported as single-seed (no variance estimate): "
            "%s_seeds=%d baseline_seeds=%d of %d requested. missing_%s_seeds=%s missing_baseline_seeds=%s "
            "-- a missing baseline/pretrained-model seed is reported explicitly, never silently substituted "
            "with another seed's result.",
            args.model_family, len(pretrained_metrics_per_seed), len(baseline_metrics_per_seed), len(seeds),
            args.model_family, missing_pretrained_seeds, missing_baseline_seeds,
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

    if pretrained_metrics_per_seed:
        experiment_name = family_info["experiment_name"]
        pretrained_seeds_with_metrics = list(pretrained_metrics_by_seed)
        pretrained_breakdown_path = (
            REPO_ROOT / "checkpoints" / "transfer_learning" / f"seed_{pretrained_seeds_with_metrics[0]}"
            / f"{experiment_name}_seed{pretrained_seeds_with_metrics[0]}_parameter_breakdown.json"
        )
        from src.utils.io import load_json

        param_breakdown = load_json(pretrained_breakdown_path) if pretrained_breakdown_path.exists() else {}
        rows.append(_assemble_row(
            model_label=experiment_name, category=family_info["category_label"],
            initialization="ImageNet pretrained", backbone=family_info["backbone_label"],
            adapters="shared_adapters (reused)", loss_balancing="learned_uncertainty (reused)",
            input_size=family_info["input_size_label"],
            param_breakdown=param_breakdown, metrics_per_seed=pretrained_metrics_per_seed,
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
        str(seed): {"volo": pretrained_metrics_by_seed.get(seed), "baseline": baseline_metrics_by_seed.get(seed)}
        for seed in seeds
    }
    _atomic_save_json(seed_metrics_index, output_dir / "seed_metrics_index.json")

    split_path = REPO_ROOT / "data" / "splits" / "full_metadata_with_splits.csv"
    manifest = {
        "seeds_requested": seeds,
        "model_family": args.model_family,
        # "volo_*" key names are kept for on-disk/notebook backward
        # compatibility regardless of --model-family (the default and by
        # far the most common case); "model_family" above disambiguates
        # which pretrained model these counts actually refer to.
        "volo_seeds_completed": sorted(pretrained_metrics_by_seed),
        "baseline_seeds_completed": sorted(baseline_metrics_by_seed),
        "missing_volo_seeds": missing_pretrained_seeds,
        "missing_baseline_seeds": missing_baseline_seeds,
        "single_seed_no_variance_estimate": len(pretrained_metrics_per_seed) < 2 or len(baseline_metrics_per_seed) < 2,
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
