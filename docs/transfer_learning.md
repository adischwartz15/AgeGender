# Supplementary Experiment: ImageNet-Pretrained VOLO-D1 (Face-Only)

Practical guide to the transfer-learning extension. This originally covered
only VOLO-D1; it now also covers the pretrained-ResNet-18/50 bridge
baselines (see "Model families" below) -- all three model families share
this same trainer, persistence layer, seeds, and Table B, so one guide
covers all of them. For the research question the *core* ablation suite
answers, see
[docs/experiment_plan.md](experiment_plan.md); for why this extension is
kept structurally separate from that suite, see the README's
[Core vs. supplementary experiments](../README.md#core-vs-supplementary-experiments)
section, which this document assumes as context.

## Motivation

The core suite (Experiments 0/0b/0c, A-F) is a controlled, from-scratch
comparison: every model in it shares the same random initialization
philosophy, so differences in outcome can be attributed to the one design
choice each experiment isolates. That is a different question from a
practical one: *if the goal were simply the best possible numbers on this
task, how much would grabbing a modern, externally pretrained backbone
actually buy over the from-scratch research system?* Answering that needs
an experiment that is explicitly **not** controlled the same way, so it
gets its own model, its own trainer, its own config, its own outputs, and
its own results table (Table B) -- never merged into the core suite or its
Table A.

## Core-vs-supplementary distinction, mechanically enforced

- **Model**: `PretrainedVOLOFaceOnlyMultiTask`
  (`src/models/pretrained_volo.py`) is a standalone `nn.Module`, not a
  `backbone_factory.py` entry and not a `MultiTaskFaceModel` subclass --
  no core config value can accidentally select it (see
  `tests/test_transfer_learning_config.py::test_pretrained_volo_not_registered_in_backbone_factory`).
- **Reuse, not forking**: it directly imports and reuses
  `AgeAdapter`/`GenderAdapter` (`src/models/adapters.py`),
  `AgeQuantileHead`/`GenderClassificationHead` (`src/models/heads.py`), and
  `compute_multitask_loss`/the learned-uncertainty balancing
  (`src/losses/multitask_loss.py`) completely unmodified -- both adapter
  and head classes already accepted an arbitrary `input_dim`, so no change
  was needed there at all.
- **Trainer**: `TransferTrainer` (`src/training/transfer_trainer.py`) is a
  separate class from `Trainer` (`src/training/trainer.py`). The core
  `Trainer`'s Stage A/B/C progressive-freezing plan is only meaningful when
  the backbone comes from this repo's own SimCLR pretraining
  (`scripts/pretrain.py`) -- it doesn't fit an externally pretrained
  backbone's needs (a 2-stage schedule, 4 parameter groups, gradient
  accumulation, peak-memory tracking). `src/training/trainer.py` and
  `src/training/stages.py` are not modified by, or imported into, any of
  this (only two small generic helpers, `_build_scheduler` and
  `resolve_loss_balancing`, are reused by direct import).
- **Config**: `configs/transfer_learning.yaml` is a separate file, merged
  the same way any experiment override is, but never referenced by
  `configs/experiments.yaml` or `scripts/run_experiments.py`.
- **Outputs**: `checkpoints/transfer_learning/` and
  `results/transfer_learning/` -- never the core suite's `checkpoints/`,
  `outputs/`, or `experiments/` directories. This isolation is structural,
  not just a naming convention: `src/evaluation/reports.py`'s
  `discover_experiment_results` (which feeds Table A) only globs under
  `outputs/` and `experiments/`, so it cannot pick up VOLO's results even
  if someone tried.
- **Evaluation**: the exact same `scripts/evaluate.py::evaluate_checkpoint`
  function is used for both the from-scratch baseline row and the VOLO row
  in Table B -- a model that declares its own `build_transforms()` (only
  VOLO does) is evaluated with that transform instead of the project's
  128px/IMAGENET-constant default; every core model has no such method, so
  this is a no-op for them.

## Optional dependency

`timm` is never a core dependency. Every core experiment imports and runs
with `timm` completely absent (`tests/test_pretrained_volo.py` enforces
this both statically, via an AST scan for module-scope `import timm`
outside `src/models/pretrained_volo.py`, and dynamically, via a subprocess
that blocks the import and imports every core entry point anyway).

```bash
pip install -r requirements-transfer.txt
```

Selecting the extension without `timm` installed raises immediately with:
"The VOLO transfer-learning extension requires timm. Install it with
`pip install -r requirements-transfer.txt`." -- never a silent fallback to
random initialization.

## Model families

`scripts/run_transfer_learning.py --model-family {volo,pretrained_resnet18,pretrained_resnet50}`
(default `volo`) selects which pretrained backbone is trained/evaluated;
all three reuse the exact same `TransferTrainer`, adapters, heads, learned
loss-balancing, two-stage schedule, canonical seeds, locked split, and
persistence layer above -- only the backbone class and its own official
preprocessing differ.

- **`pretrained_resnet18`** (`src/models/pretrained_resnet.py`,
  `configs/pretrained_resnet18.yaml`) -- **required**: an ImageNet-pretrained
  `torchvision.models.resnet18`, the most interpretable pretraining-bridge
  comparison against this project's from-scratch Custom ResNet-18, since
  both share the ResNet architecture family. It is **not** claimed to be
  byte-identical to the from-scratch Custom ResNet-18 -- see
  `PretrainedResNetFaceOnlyMultiTask`'s module docstring for the exact,
  documented implementation differences (stem, downsampling, block design).
- **`pretrained_resnet50`** (`configs/pretrained_resnet50.yaml`) --
  **optional**: same bridge role plus a capacity/architecture-depth
  comparison. Its role is always described as "pretraining + architecture +
  capacity comparison," **never** as isolating pretraining alone (it
  differs from ResNet-18 in depth and parameter count too).
- Preprocessing for both is `weights.transforms()` from `torchvision`'s
  official model weight metadata (resolved via `resolve_eval_transform`,
  `src/data/transforms.py`) -- never this project's own 128px/ImageNet-constant
  default, and never VOLO's `crop_pct`.

```bash
python scripts/run_transfer_learning.py --model-family pretrained_resnet18 --seeds 42,123,2026
python scripts/run_transfer_learning.py --model-family pretrained_resnet50 --seeds 42,123,2026 \
    --resume --skip-completed --sync-after-epoch
```

Requires `torchvision` (`requirements-transfer.txt`, same optional-dependency
posture as `timm` -- never a core dependency).

## Canonical seeds

**42, 123, 2026** -- identical to the core suite's pre-registered protocol
(`docs/final_evaluation_protocol.md`). Table B's VOLO row and baseline row
are always drawn from the same requested seed set; if a baseline checkpoint
is unavailable for a requested seed, that is reported explicitly
(`missing_baseline_seeds` in `table_b_manifest.json`), never silently
backfilled from a different seed.

## How to run it

```bash
# Fresh full run (needs a GPU in practice)
python scripts/run_transfer_learning.py --seeds 42,123,2026

# Resume after a Colab/Kaggle disconnect or manual interruption: completed
# seeds are reused, an interrupted seed resumes from its latest valid
# checkpoint, a seed that never started trains from scratch.
python scripts/run_transfer_learning.py --seeds 42,123,2026 \
    --resume --skip-completed --sync-after-epoch \
    --persistent-root /content/drive/MyDrive/AgeGender/transfer_learning

# Evaluate only -- never trains; rebuilds Table B from existing checkpoints.
python scripts/run_transfer_learning.py --seeds 42,123,2026 --evaluate-only

# Single seed only (e.g. to finish just the one still-incomplete seed).
python scripts/run_transfer_learning.py --seeds 123 --resume --skip-completed

# tiny, non-scientific CPU pipeline check (no persistence layer, single seed)
python scripts/run_transfer_learning.py --smoke

python scripts/run_transfer_learning.py --only volo                      # skip the baseline row
python scripts/run_transfer_learning.py --only baseline                  # skip VOLO, reuse the baseline row only

# Other model families (see "Model families" above) -- identical flags, just --model-family:
python scripts/run_transfer_learning.py --model-family pretrained_resnet18 --seeds 42,123,2026
python scripts/run_transfer_learning.py --model-family pretrained_resnet50 --seeds 42,123,2026 --smoke
```

The from-scratch baseline (`exp_d_shared_adapters_learned_balance` by
default, `--baseline-experiment` to change it) is **never retrained** here
-- its already-computed test-metrics JSON is reused if present, or its
existing checkpoint is evaluated (never retrained) if not.

## Output paths

```
checkpoints/transfer_learning/volo_d1_face_only_pretrained/seed_<seed>/...   # legacy flat per-metric checkpoints (evaluate.py/inference compatibility)
checkpoints/transfer_learning/volo_d1_face_only_pretrained/                  # PersistentArtifactManager's isolated, resumable tree (see below)
  seed_42/{checkpoints,state,metrics,predictions,plots,logs}/...
  seed_123/...
  seed_2026/...
  transfer_learning_summary.zip           # lightweight archive (no checkpoints) -- rebuilt at seed/full-run completion
results/transfer_learning/volo_d1_face_only_pretrained/seed_<seed>/...       # evaluate_checkpoint() output (plots, metrics)
results/transfer_learning/table_b.csv
results/transfer_learning/table_b_manifest.json   # git SHA, dependency versions, split fingerprint, seed count, missing seeds
results/transfer_learning/seed_metrics_index.json # per-seed VOLO/baseline metrics -- lets Table B rebuild without retraining
```

## Persistent artifacts

Implemented by `src/training/persistent_artifacts.py::PersistentArtifactManager`
-- a reusable, platform-agnostic persistence layer (two plain filesystem
paths: a fast local working root and an optional persistent mirror root;
Colab passes a mounted Drive folder, Kaggle passes `/kaggle/working`, unit
tests pass two temp directories). Nothing platform-specific is scattered
through the model or trainer code; `TransferTrainer` only calls
`artifact_manager.on_epoch_end/on_new_best/on_stage_transition/on_seed_complete`.

**Per-seed directory layout** (under either root):

```
seed_<seed>/
|-- checkpoints/{last.pt, previous_last.pt, best.pt}
|-- state/{trainer_state.json, run_manifest.json, completion.json, checkpoint_checksums.json}
|-- metrics/{validation_history.json, test_metrics.json}
|-- predictions/
|-- plots/
`-- logs/
```

**Checkpoint frequency.** Saved and (if `--sync-after-epoch`) mirrored:
after every completed epoch, whenever a new best validation score is
reached, immediately after the Stage 1 -> Stage 2 transition, and after
each seed completes (final test evaluation + completion marker). Never
per-batch. Losing the currently-running incomplete epoch on disconnect is
acceptable; losing a previously completed epoch or seed is not.

**Best vs. last.** `best.pt` is the highest-balanced-score checkpoint seen
so far (what `evaluate.py`/inference load); `last.pt` is the most recent
epoch's checkpoint, used to resume training. Before `last.pt` is replaced,
the previous valid one is atomically rotated to `previous_last.pt` -- if
`last.pt` is later found corrupted (bad checksum or a failed `torch.load`),
resume automatically falls back to `previous_last.pt` with a logged
warning, and raises `CorruptedCheckpointError` (never silently restarts
from scratch) only if both are corrupted.

**Atomic writes.** Every checkpoint/JSON write goes to a `.tmp` path first,
is flushed/fsynced, then `os.replace()`d into place -- a crash mid-write
can never leave a half-written file at the real path. `table_b.csv`/
`table_b_manifest.json` use the same pattern, so a partial Table B is never
observed mid-overwrite.

**Checksums.** SHA-256 is computed immediately after every checkpoint write
and recorded in `state/checkpoint_checksums.json`; resume validates the
checksum before trusting a checkpoint.

**Completion markers.** `state/completion.json` is written only after a
seed's Stage 1 + Stage 2 + final test evaluation all succeed. A seed counts
as complete only if *all* of: the marker's `status == "complete"`, its
referenced best checkpoint exists, its checksum matches, test metrics are
present, and (when checked) the split fingerprint and model
identifier/pretrained source match the current run -- never from directory
existence alone (`PersistentArtifactManager.is_seed_complete`).

**Resumable checkpoint contents.** `last.pt`/`previous_last.pt` carry
everything needed to resume byte-for-byte: model/optimizer/scheduler/AMP-
scaler state, epoch/global step/training stage (Stage 1 vs. Stage 2 --
restoring one never re-runs or skips the wrong stage), best validation
metric, early-stopping state, full training history, seed, Python/NumPy/
PyTorch(CPU+CUDA) RNG state, model identifier, pretrained source, resolved
input size/transform config, split fingerprint, age/gender-head and adapter
config, learned loss-balancing parameters, per-group learning rates, git
commit SHA, and the full config snapshot.

**Table B regeneration.** Rebuilt after every run (partial or complete)
from `seed_metrics_index.json` -- `--evaluate-only --skip-completed`
rebuilds it from saved metrics alone, without retraining anything. With
fewer than 2 completed seeds per row, values render as `(n=1, no std)`;
`table_b_manifest.json`'s `missing_volo_seeds`/`missing_baseline_seeds`
record exactly which requested seeds are still outstanding.

**Archive contents.** `transfer_learning_summary.zip` (rebuilt at
seed-completion and full-run-completion, never per-epoch) bundles
manifests, metrics, plots, configs, completion markers, and Table B --
never checkpoints, dataset images, cache/temp files, or anything
credential-shaped (`src/training/persistent_artifacts.py::build_summary_archive`).
The Kaggle notebook additionally builds a fuller
`agegender_transfer_learning_artifacts.zip` that also includes `best.pt`/
`last.pt` (never `previous_last.pt`, a duplicate).

**Secret handling.** Optional Kaggle -> Google Drive backup
(`src/utils/kaggle_drive_backup.py`) reads `GOOGLE_DRIVE_SERVICE_ACCOUNT_JSON`
and `GOOGLE_DRIVE_FOLDER_ID` only from Kaggle Secrets, holds the credential
JSON only in memory (never writes it to disk), never logs or prints it, and
fails soft (a warning, `/kaggle/working` still gets everything) on any
missing secret, missing optional dependency, or network/API error.

```bash
pip install -r requirements-kaggle-drive.txt   # only if ENABLE_KAGGLE_DRIVE_BACKUP = True
```

To configure it: create a Google Cloud service account, share the target
Drive folder with that service account's email address (Editor access), add
the service account's JSON key as the Kaggle Secret
`GOOGLE_DRIVE_SERVICE_ACCOUNT_JSON` and the target folder's ID as
`GOOGLE_DRIVE_FOLDER_ID` (Kaggle notebook editor -> Add-ons -> Secrets), then
set `ENABLE_KAGGLE_DRIVE_BACKUP = True` in the bootstrap cell.

## Colab recovery

After a runtime disconnect or restart:

1. Reconnect the runtime.
2. Rerun environment setup, repository checkout, and dependency
   installation (sections 3-5 of `notebooks/train_evaluate_colab.ipynb`).
3. Rerun the Drive-mount cell (section 6) and the "Persistent
   Transfer-Learning Storage and Resume" bootstrap cell -- this restores
   the newest valid checkpoint per seed from Drive into the local working
   directory and prints each seed's status (`COMPLETE`/`INCOMPLETE`/`NOT
   STARTED`).
4. Rerun the Table B run cell. With `AUTO_RESUME = SKIP_COMPLETED =
   SYNC_AFTER_EPOCH = True` (the defaults), this is equivalent to
   `--resume --skip-completed --sync-after-epoch`.

No other cell needs to be rerun -- the bootstrap cell reconstructs every
path/config itself rather than depending on a variable from the dead kernel.

## Kaggle recovery

1. Restore from Drive backup (if `ENABLE_KAGGLE_DRIVE_BACKUP=True`) or, the
   recommended path, attach a previous run's saved notebook output as an
   input dataset: save this notebook's output as a version, then in a new
   session, Add Data > Your Datasets/Notebooks > that version's output, and
   set `KAGGLE_TRANSFER_LEARNING_INPUT_DATASET_DIR` in the bootstrap cell to
   its mounted path (e.g. `/kaggle/input/<your-dataset-slug>`).
2. Rerun the bootstrap cell -- this restores from the configured
   `KAGGLE_RESTORE_SOURCE` (`"attached_dataset"`, `"drive"`,
   `"working_directory"`, or `"auto"`), verifies checksums, and prints each
   seed's status.
3. Rerun the Table B run cell to resume the incomplete seed(s) and skip
   completed ones.
4. Save a new notebook version when done, so its `/kaggle/working` output
   (including `agegender_transfer_learning_artifacts.zip`) becomes the next
   session's attachable dataset.

## Fresh clone / from-scratch commands

```bash
# fresh full run
python scripts/run_transfer_learning.py --seeds 42,123,2026

# resume run
python scripts/run_transfer_learning.py --seeds 42,123,2026 --resume --skip-completed --sync-after-epoch

# evaluate only (never trains)
python scripts/run_transfer_learning.py --seeds 42,123,2026 --evaluate-only

# single-seed run
python scripts/run_transfer_learning.py --seeds 123 --resume --skip-completed

# rebuild Table B without retraining anything
python scripts/run_transfer_learning.py --seeds 42,123,2026 --evaluate-only --skip-completed
```

## Resource requirements

VOLO-D1 at 224px is far heavier than the core suite's 128px models --
default batch size is a conservative 16 (with 2-step gradient accumulation
for an effective batch of 32), specifically to avoid OOM on a free-tier
Colab GPU. A CPU run is possible (the `transfer_learning_smoke` profile is
CPU-only and runs in well under a minute) but real training needs a GPU in
practice; on CPU, one VOLO-D1 forward+backward pass at 224px with batch
size 2 took ~1.2s in this project's own dev-environment measurement --
extrapolate accordingly before attempting a full run without a GPU.

## Two-stage training schedule

| | Stage 1 | Stage 2 |
|---|---|---|
| Backbone | frozen | unfrozen (full, or last N stages via `training.finetune_unfreeze`) |
| Trains | adapters, heads, loss-balancing params | + backbone |
| Epochs (default) | `head_only_epochs: 3` | `finetune_epochs: 20` (ceiling; early stopping still applies) |
| LR | `head_lr`/`adapter_lr`/`loss_balance_lr`: 3e-4 | `backbone_lr`: 3e-5, others unchanged |

AdamW, weight decay 0.05, gradient clipping 1.0, warmup+cosine schedule,
early stopping on **validation only**, best-checkpoint restoration
(`TransferTrainer.restore_best_checkpoint`), AMP on CUDA only (never
requested/enabled on CPU). Four separate optimizer parameter groups
(backbone / adapters / heads / loss-balancing) are rebuilt at the Stage
1->2 transition; a frozen group is omitted from the optimizer entirely
rather than included with zero gradient.

## Reproducibility

Same UTKFace split as the core suite (`data/splits/full_metadata_with_splits.csv`,
never regenerated for this experiment; see
[docs/reproducibility.md](reproducibility.md#stratified-locked-split) for
how it is generated and locked), same age loss/gender loss/label
mapping/age clipping, same evaluation functions and postprocessing. Every
run saves: metrics JSON, predictions, config snapshot, split fingerprint
(SHA-256), training history, plots, seed, dependency versions
(Python/PyTorch/`timm`/CUDA), and the git commit SHA.

**Preprocessing is model-aware, not a shared hardcoded default.** Every
evaluation/inference/calibration entry point resolves each model's own
preprocessing via `resolve_eval_transform` (`src/data/transforms.py`) --
`model.build_transforms()` when the model declares one (VOLO, both
pretrained-ResNet families), the project's 128px/ImageNet-constant default
otherwise. This includes VOLO's `crop_pct` (`0.96` for `volo_d1_224`,
verified against a real `timm` install, not assumed to be `1.0`) and each
pretrained-ResNet's official `torchvision` weight transform -- silently
using the wrong crop ratio or resize/crop protocol would bias every metric
downstream without an explicit error.

## Limitations

- **Confounded comparison, by design.** VOLO-D1 differs from the
  from-scratch baseline in initialization, pretraining data, parameter
  count, input resolution (224px vs. 128px), **and** its own optimizer/
  training schedule (the 2-stage trainer above, not the core suite's
  progressive Stage A/B/C trainer) all at once. Any difference in Table B
  reflects the combined effect of all of these; it is never attributed to
  "the Transformer-style backbone" alone.
- **Small dataset, large model.** A VOLO-D1-scale model can overfit a
  dataset of a few thousand UTKFace images. If it performs worse than the
  from-scratch baseline, that is reported honestly, not tuned away against
  the test set.
- **Single-seed risk.** A full multi-seed run needs real GPU time. With
  fewer than 2 completed seeds per row, `build_transfer_learning_table`
  renders that row's metrics as `"... (n=1, no std)"` rather than a
  variance-estimated result, and `table_b_manifest.json`'s
  `single_seed_no_variance_estimate` flag records this explicitly.
- **Verification gap.** `src/models/pretrained_volo.py`'s assumptions
  about `timm`'s VOLO internals (e.g. `self.backbone.network` as the
  stage-list attribute `unfreeze_last_stages` indexes into) were verified
  against a real `timm==1.0.28` install during development; a materially
  different `timm` version should be re-verified against
  `tests/test_pretrained_volo_with_timm.py` before trusting a new run.

## Citations and licenses

- **VOLO**: Yuan et al., "VOLO: Vision Outlooker for Visual Recognition,"
  [arXiv:2106.13112](https://arxiv.org/abs/2106.13112).
- **MiVOLO** (motivation for face-only usage, not implemented here):
  Kuprashevich & Tolstykh, "MiVOLO: Multi-input Transformer for Age and
  Gender Estimation," [arXiv:2307.04616](https://arxiv.org/abs/2307.04616).
- **timm**: Ross Wightman et al., [pytorch-image-models](https://github.com/huggingface/pytorch-image-models), Apache-2.0.
- **ImageNet**: Deng et al., "ImageNet: A Large-Scale Hierarchical Image
  Database," CVPR 2009.

`timm`'s code license and the license of the specific pretrained weights it
downloads are **not automatically the same** -- check the weight card for
the exact `model_id` configured in `configs/transfer_learning.yaml` before
any redistribution or commercial use.
