# Codebase Audit (2026-07-09)

Full architectural review, file-by-file inventory, dead-code scan, and
targeted refactoring pass over the entire repository. This document is the
durable record of that audit; see the end of this file for exact changes
made vs. items proposed for the maintainer's confirmation.

---

## 1. Architectural Overview

### What the project does

This is a **multi-task face-image model** that jointly predicts:

1. **Age** as an uncertainty-aware quantile regression: (q10, q50, q90),
   trained with a **pinball (quantile) loss** at quantiles **(0.10, 0.50,
   0.90)** -- not (0.05, 0.50, 0.95); see the "Deviations from the prompt's
   description" note below.
2. **Dataset gender label** as a softmax classification with a
   **confidence-threshold abstention** mechanism ("Not sure" below
   `confidence_threshold`, default 0.80).

A single **shared backbone** (a manually implemented ResNet-18 -- no
`torchvision.models`/`timm`/pretrained weights anywhere) feeds two
**task-specific residual bottleneck adapters**, one per task, which in turn
feed the two task heads. Two controlled baseline backbones also exist
(`SimpleCNNBackbone`, `PlainDeep18NoSkip`) purely to isolate what the
residual design contributes -- neither is the project's real backbone.

### Data flow, end to end

```
raw images (UTKFace filenames or a Kaggle CSV)
  -> src/data/metadata.py            (parse filenames/CSV -> unified DataFrame)
  -> src/data/validation.py          (drop corrupt/duplicate images, quality report)
  -> src/data/split_utils.py         (train/validation/calibration/test, subject-level safe)
  -> data/splits/full_metadata_with_splits.csv   (single source of truth, reused by every experiment)
  -> src/data/dataset.py             (FaceMultiTaskDataset: image + masked age/gender labels)
  -> src/data/transforms.py          (TrainTransform / EvalTransform / SimCLRTransform -- PIL/NumPy/torch only)
  -> src/models/backbone_factory.py  (custom_resnet18 | simple_cnn | plain_deep18_no_skip)
  -> src/models/adapters.py          (per-task residual bottleneck adapter)
  -> src/models/heads.py             (AgeQuantileHead, GenderClassificationHead)
  -> src/losses/{quantile_loss,multitask_loss}.py  (pinball loss + CE, combined via fixed or learned-uncertainty weights)
  -> src/training/trainer.py         (progressive Stage A/B/C or single warm-up, checkpointing, history)
  -> src/evaluation/*                (metrics, calibration, robustness, backbone comparison, reports)
  -> src/inference/predictor.py + src/api/*   (FastAPI serving)
  -> frontend/                       (React/Vite/Tailwind dashboard consuming the API)
```

The **four-way data split** (train / validation / calibration / test) is the
architectural backbone of the whole evaluation methodology: each split is
used for exactly one purpose (model fitting / early stopping+checkpoint
selection / split-conformal calibration / final evaluation), enforced by
`src/data/split_utils.py:assert_no_leakage` and never crossed anywhere else
in the codebase.

### The three architecture modes

Selected via `model.architecture` in `configs/model.yaml`, all built by
`src/models/multitask_model.py:MultiTaskFaceModel`:

| Mode | Backbones | Adapters | Maps to |
|---|---|---|---|
| `separate` | 2 independent | n/a | Experiment A -- no-sharing baseline |
| `shared_no_adapters` | 1 shared | none (heads read the raw shared embedding) | Experiment B -- naive sharing |
| `shared_adapters` | 1 shared | per-task bottleneck adapter | Experiments C/D and the backbone-comparison controls (0/0b/0c) |

### Loss balancing -- what actually exists vs. the prompt's description

`src/losses/multitask_loss.py` implements exactly **two** balancing modes
(`model.loss_balancing.mode`):

- **`fixed`**: `total = age_weight * age_loss + gender_weight * gender_loss`.
- **`learned_uncertainty`**: homoscedastic uncertainty weighting (Kendall,
  Gal & Cipolla 2018) via two trainable log-variance parameters owned by
  `MultiTaskFaceModel` (`log_var_age`, `log_var_gender`):
  `total = exp(-s_age)*age_loss + s_age + exp(-s_gender)*gender_loss + s_gender`.

**GradNorm is not implemented anywhere in this codebase.** The task
description mentions it as an available strategy; a full-repository grep
confirms it doesn't exist. This is flagged here rather than silently
assumed -- if GradNorm balancing is actually wanted, it would be a new
addition (a third `loss_balancing.mode`), not something this audit found
already present and refactored.

A task's loss term is fully **omitted** (not just zero-weighted) from the
total whenever a batch has zero labeled samples for that task -- this
matters specifically for `learned_uncertainty` mode, where a "loss of 0"
combined with the `+ s_task` bias term would otherwise still contribute a
spurious regularization signal with no real supervision behind it.

### Uncertainty and calibration pipeline

- **Pinball loss** (`src/losses/quantile_loss.py`) trains the raw
  (q10, q50, q90) head, at quantiles **(0.10, 0.50, 0.90)**.
- `src/models/heads.py:AgeQuantileHead` parameterizes q50 via a sigmoid
  (bounded to [age_min, age_max]) and q10/q90 as `q50 -/+ softplus(delta)`,
  which **guarantees q10 <= q50 <= q90 by construction** for any network
  output -- ordering can never invert, a real numerical-safety property,
  not an incidental one.
- **Split Conformal Prediction (CQR)** (`src/evaluation/calibration.py`)
  fits a single scalar offset on the dedicated **calibration** split
  (never validation or test), giving a marginal coverage guarantee under
  exchangeability. The offset is applied identically to clean and
  corrupted (robustness) predictions -- never refit per corruption.
- Every calibration artifact records **provenance** (checkpoint SHA-256,
  split-CSV SHA-256, ordered test-sample-ID hash, experiment, seed, alpha,
  target coverage), and `scripts/evaluate.py` / `scripts/run_robustness.py`
  validate that provenance before applying it, raising
  `CalibrationMismatchError` loudly on any mismatch (cross-seed,
  cross-checkpoint, or reordered-split contamination).

### Selective classification / abstention

`src/inference/predictor.py` and `src/training/trainer.py` implement the
gender-label abstention mechanism identically: predict via softmax, abstain
("Not sure") when `max_prob < confidence_threshold` (**tau**, default
0.80). Four distinct, always-jointly-reported metrics
(`src/evaluation/metrics.py`) prevent this from hiding a real trade-off:
selective accuracy, coverage, abstention rate, and *effective* accuracy
(denominator includes abstentions).

### Evaluation phases

1. **`scripts/evaluate.py`** -- single-checkpoint test-set evaluation: age
   MAE/RMSE/R2, raw + calibrated interval coverage/width, per-age-bucket
   uncertainty, gender confusion matrix/abstention, optional k-NN
   comparison. Every trained checkpoint gets its own isolated
   `experiments/<experiment>/seed_<seed>/{checkpoints,calibration,metrics,plots,robustness,knn}`
   artifact tree (`src/utils/experiment_paths.py`) -- never a shared global
   `outputs/` directory two runs could silently collide in.
2. **`scripts/run_robustness.py`** -- deterministic degradation evaluation
   under 11 corruption types x 3 severities (blur, noise, JPEG, resolution,
   brightness, contrast, grayscale, occlusion, crop), both raw and
   calibrated, full test split by default (or deterministic
   age-bucket x gender-label stratified sampling via `--max-samples`).
3. **`scripts/compare_backbones.py`** -- cross-model comparison suite:
   clean-test summary, gender/age selective-risk-coverage curves + AURC +
   paired-bootstrap CIs (both at fixed coverage levels *and*, separately,
   on the AURC statistic itself -- only the latter is ever cited as
   evidence for an "AURC is lower" claim), tail-error analysis, robustness
   diff table (all pairwise model comparisons), and a final, mechanically
   generated, explicitly conditional "is the added residual complexity
   justified?" interpretation that is equally capable of concluding
   *against* the more complex architecture.
4. **`src/evaluation/knn_baseline.py`** -- the non-parametric methodological
   comparison: distance-weighted k-NN over L2-normalized embeddings from
   the trained encoder, compared against the parametric heads on identical
   metrics.

### Design patterns used

- **Config-driven, not hardcoded**: every architecture/training/evaluation
  choice lives in `configs/*.yaml`; `--set key.path=value` overrides let ad
  hoc runs skip editing YAML. Every checkpoint embeds a full snapshot of
  the config that produced it.
- **Factory pattern**: `src/models/backbone_factory.py` dispatches on
  `model.backbone.name`; all three backbones expose an identical interface
  (`forward`, `forward_features`, `embedding_dim`, `num_parameters()`) so
  callers (the model, Grad-CAM, stage-freezing logic) never need to know
  which one is active.
- **Dataclasses for structured results**: `ParameterBreakdown`,
  `MultiTaskLossOutput`, `AgePrediction`/`GenderPrediction`/`PredictionResult`,
  `QualityDiagnostics`, `Stage` -- typed, self-documenting boundaries
  instead of loose dicts at the module seams that matter most.
- **No fabricated results**: `src/evaluation/reports.py` /
  `final_report.py` render an explicit "not yet generated" placeholder
  (with the exact command to produce it) for any section whose backing
  artifact doesn't exist -- never a hardcoded example number.
- **Dependency-light by design**: no `torchvision` transforms/models, no
  Grad-CAM library, no pretrained-weight downloads anywhere -- every one of
  these is a deliberate, documented choice (see individual file docstrings),
  not an oversight.
- **Isolated per-run artifact trees**: `src/utils/experiment_paths.py` +
  the calibration-provenance system above exist specifically to make
  cross-experiment/cross-seed contamination structurally hard, not just
  discouraged by convention.

---

## 2. File-by-File Report

Grouped by directory. "Resp." = primary responsibility.

### `src/data/` -- dataset construction and integrity

| File | Responsibility | Key symbols |
|---|---|---|
| `metadata.py` | Parse UTKFace filenames or a Kaggle CSV into a unified DataFrame | `parse_utkface_directory`, `parse_csv_metadata`, `load_metadata` |
| `validation.py` | Corrupt/duplicate-image filtering, data-quality report, split orchestration | `validate_dataset`, `validate_and_split` |
| `split_utils.py` | Deterministic 4-way split with subject-level leakage prevention | `split_dataframe`, `assert_no_leakage` |
| `dataset.py` | PyTorch `Dataset`s: masked-label multi-task dataset + SimCLR pretrain dataset | `FaceMultiTaskDataset`, `SimCLRPretrainDataset`, `build_datasets` |
| `transforms.py` | Manual (no-torchvision) image transforms | `EvalTransform`, `TrainTransform`, `SimCLRTransform` |
| `kaggle_download.py` | Kaggle API dataset download (env-var credentials only) | `download_dataset`, `validate_credentials` |

### `src/models/` -- architecture

| File | Responsibility | Key symbols |
|---|---|---|
| `custom_resnet.py` | Manually implemented ResNet-18 (main backbone); `zero_init_residual` control | `CustomResNet18`, `BasicBlock` |
| `simple_cnn.py` | Non-residual controlled baseline (Experiment 0) | `SimpleCNNBackbone` |
| `plain_deep18_no_skip.py` | Depth/width-matched no-skip counterpart (Experiment 0b, the real residual ablation) | `PlainDeep18NoSkip`, `PlainBlock` |
| `backbone_factory.py` | Dispatches on `model.backbone.name` | `build_backbone` |
| `adapters.py` | Residual bottleneck task adapters | `BottleneckAdapter`, `AgeAdapter`, `GenderAdapter`, `IdentityAdapter` |
| `heads.py` | Age quantile head (order-safe by construction) + gender classification head | `AgeQuantileHead`, `GenderClassificationHead` |
| `multitask_model.py` | Assembles backbone+adapters+heads per architecture mode; owns learned-uncertainty log-variances | `MultiTaskFaceModel`, `ParameterBreakdown`, `build_multitask_model` |
| `baselines.py` | **Unused** trivial sanity baselines (constant-quantile age, majority-class gender) -- see Section 3 | `ConstantQuantileAgeBaseline`, `MajorityClassGenderBaseline` |

### `src/losses/`

| File | Responsibility |
|---|---|
| `quantile_loss.py` | Pinball loss at (0.10, 0.50, 0.90), masked mean |
| `multitask_loss.py` | Combines per-task losses via fixed weights or learned homoscedastic uncertainty |

### `src/training/`

| File | Responsibility | Key symbols |
|---|---|---|
| `trainer.py` | Progressive Stage A/B/C training loop, checkpointing, incremental history/status | `Trainer` |
| `stages.py` | Builds the Stage A/B/C plan (or single warm-up when no pretrained backbone) | `build_stage_plan`, `Stage` |
| `checkpointing.py` | Checkpoint save/load, best-metric tracking | `save_checkpoint`, `load_checkpoint`, `BestMetricTracker` |
| `callbacks.py` | Early stopping | `EarlyStopping` |
| `pretrain.py` | Optional SimCLR self-supervised pretraining | `pretrain_simclr`, `nt_xent_loss`, `ProjectionHead` |

### `src/evaluation/` -- metrics, calibration, robustness, comparisons, reports

| File | Responsibility |
|---|---|
| `metrics.py` | Core age/gender metrics (MAE/RMSE/R2, interval coverage/width, selective/effective accuracy, per-bucket uncertainty) |
| `calibration.py` | Split-conformal fit/apply + provenance recording/validation (`CalibrationMismatchError`) |
| `robustness.py` | 11 deterministic corruptions, stratified sampling, degradation tables, all-pairs diff table |
| `selective.py` | Generic risk-coverage curve, AURC, paired bootstrap CIs (fixed-coverage and AURC-itself) |
| `backbone_comparison.py` | Cross-model clean-test/selective-risk/tail-error analysis, final conditional interpretation |
| `knn_baseline.py` | Non-parametric k-NN baseline in the learned embedding space |
| `comparison.py` | Table builders (parametric-vs-kNN, ablation table, backbone comparison, seed aggregation) |
| `gradcam.py` | Manual Grad-CAM ("model attention visualization") |
| `architecture_analysis.py` | Gradient-interference cosine similarity + linear CKA representation similarity |
| `reports.py` | Architecture-analysis Markdown report (never fabricates missing sections) |
| `final_report.py` | Cross-cutting final results report (ablation, seeds, uncertainty, robustness, AURC CIs) |

### `src/inference/` and `src/api/`

| File | Responsibility |
|---|---|
| `inference/artifacts.py` | Loads checkpoint + calibration + kNN index, with honest warnings on gaps |
| `inference/predictor.py` | End-to-end single-image inference (quality, face crop, prediction, Grad-CAM, kNN) |
| `inference/face_detection.py` | Classical Haar-cascade face cropping (not neural, not biometric ID) |
| `inference/quality.py` | Non-biometric image-quality diagnostics (blur/brightness/contrast/resolution) |
| `api/main.py` | FastAPI app + routes (`/predict`, `/health`, `/models`, `/admin/reload-models`, ...) |
| `api/dependencies.py` | In-process `AppState`, device resolution, artifact (re)loading |
| `api/schemas.py` | Pydantic request/response models, shared `DISCLAIMER` |

### `src/utils/`

| File | Responsibility |
|---|---|
| `config.py` | YAML load/deep-merge, env-var overrides, `--set` CLI overrides, `resolve_device` |
| `experiment_paths.py` | Isolated per-experiment/seed artifact directory layout |
| `io.py` | JSON/YAML I/O, SHA-256 hashing, checkpoint-name parsing |
| `seed.py` | Global + per-DataLoader-worker deterministic seeding |
| `logging.py` | Shared structured logger factory |
| `visualization.py` | All Matplotlib plot builders (training curves, calibration, robustness, Grad-CAM overlays, embeddings) |

### `scripts/` -- CLI entry points (one per pipeline stage)

| Script | Responsibility |
|---|---|
| `prepare_data.py` | Parse + validate + split raw metadata |
| `download_kaggle_data.py` | Download the configured Kaggle dataset |
| `train.py` | Single-configuration training entry point (`run_training`, reused by the scripts below) |
| `run_experiments.py` | Full ablation suite (Experiments 0/0b/0c, A-D); isolated per-experiment artifacts, calibrate-then-evaluate |
| `run_seeds.py` | One experiment across multiple seeds for mean +/- std |
| `pretrain.py` | CLI wrapper for SimCLR pretraining |
| `calibrate.py` | Fit + save a conformal calibration artifact (with provenance) for one checkpoint |
| `build_knn_index.py` | Build the k-NN baseline index from a checkpoint's training-split embeddings |
| `evaluate.py` | Single-checkpoint test evaluation (+ optional k-NN comparison) |
| `run_robustness.py` | Deterministic robustness/corruption evaluation for one checkpoint |
| `compare_backbones.py` | Cross-model comparison suite (never retrains) |
| `generate_architecture_report.py` | Gradient interference + CKA + assembles `docs/architecture_analysis_generated.md` |
| `generate_final_report.py` | Assembles the cross-cutting `docs/final_results_report.md` |
| `export_report.py` | Thin, undocumented convenience wrapper re-running just `save_report` (see Section 3) |
| `generate_gradcam.py` | Grad-CAM overlays for correct/incorrect/low-confidence/widest-interval/blurry samples |
| `generate_demo_images.py` | Procedurally draws synthetic (non-photographic) placeholder faces for demo mode |
| `check_demo_readiness.py` | Verifies a checkpoint/calibration artifact exists before a live demo |
| `run_demo.py` | Launches backend (uvicorn) + frontend (Vite) for a live demo, after the readiness check |

### `frontend/` -- React + TypeScript + Vite + Tailwind

Standard SPA structure: `App.tsx` composes per-feature panels
(`AgePredictionCard`, `GenderPredictionCard`, `GradCamPanel`,
`ModelComparisonPanel`, `QualityPanel`, `UncertaintyPanel`, `Disclaimer`,
`ImageUploader`, `LoadingState`, `PredictionPanel`), `api.ts` wraps the
backend HTTP calls, `types.ts` mirrors the Pydantic response schemas. Not
deep-audited line-by-line here (out of scope for a PyTorch-focused pass);
no obvious dead files -- every component is imported from `App.tsx`.

### `tests/` -- 262+ tests, one file per module/feature under test

Broad, deliberate coverage: unit tests for every loss/metric/model
component, integration tests that train tiny real checkpoints end-to-end
(`test_smoke_training.py`, `test_evaluate_script.py`,
`test_compare_backbones.py`), and regression tests named after the specific
bug they guard against (e.g. `test_notebook_resume_logic.py`,
`test_trainer_observability.py`). No stale/orphaned test files found.

### `configs/`, `docs/` -- all actively referenced

Every YAML file is loaded by `src/utils/config.py:load_full_config` or a
script's explicit merge; every `docs/*.md` is either hand-written
methodology (kept in sync with code, e.g. `experiment_plan.md`,
`final_evaluation_protocol.md`) or auto-generated
(`architecture_analysis_generated.md`, `final_results_report.md`, both
correctly gitignored -- see `.gitignore`).

---

## 3. Cleanup Findings

**Nothing here was deleted.** Per instructions, everything below is listed
for the maintainer's explicit confirmation.

### Confirmed dead code (defined, never imported/used/tested anywhere)

- **`src/models/baselines.py`** (`ConstantQuantileAgeBaseline`,
  `MajorityClassGenderBaseline`) -- a repo-wide grep for these class names
  or `models.baselines`/`from src.models import baselines` found zero
  references outside the file's own docstring comment in
  `backbone_factory.py`. Either wire these into an actual "beats trivial
  baselines" check (e.g. in `scripts/evaluate.py` or a dedicated test), or
  remove the file.

### Likely-stale local artifacts (not git-tracked -- verified via `git ls-files`; `.gitignore` already correctly excludes all of these, so no repository cleanup is actually required)

- `checkpoints/multitask_best_{age_mae,balanced_score,gender_accuracy}.pt`
  -- three checkpoints under the generic `multitask` experiment name (the
  default of `scripts/train.py --experiment-name`), alongside the properly
  named `exp_a`/`exp_b`/`exp_c`/`exp_d_*` checkpoints from the real ablation
  suite. These look like an early ad hoc run predating the experiment
  naming convention. **Not deleted** -- flag for the maintainer to decide
  whether to keep (e.g. as a "default single-config" reference run) or
  remove locally.
- `__pycache__/`, `.pytest_cache/`, `face_multitask_research.egg-info/`,
  `frontend_stdout.log`, `frontend_stderr.log` -- all confirmed untracked
  and gitignored; purely local build/test artifacts with no effect on the
  repository. No action needed, listed only for completeness.

### Under-documented (not dead, just missing a Makefile/README mention)

- **`scripts/export_report.py`** -- a real, working, distinct convenience
  script (regenerates the architecture-analysis report without recomputing
  gradient-interference/CKA), but absent from `Makefile` and `README.md`
  and has no test. Recommend adding a `make export-report` target and a
  one-line README mention rather than removing it.

---

## 4. Refactoring Changes Made

All changes below were validated against the full test suite (262 tests
passing before this audit; see the final count after these changes in the
commit this document ships with).

1. **Device resolution: added Apple Silicon (MPS) support and centralized
   it.** `src/utils/config.py:resolve_device` previously only checked
   `torch.cuda.is_available()`; it now also checks
   `torch.backends.mps.is_available()` before falling back to CPU. Seven
   scripts (`evaluate.py`, `calibrate.py`, `run_robustness.py`,
   `compare_backbones.py`, `build_knn_index.py`,
   `generate_architecture_report.py`, `generate_gradcam.py`) each
   duplicated an inline `"cuda" if torch.cuda.is_available() else "cpu"`
   that never checked MPS; all now call `resolve_device("auto")`, so every
   entry point picks up CUDA/MPS/CPU consistently from one place.

2. **Reproducible multi-worker data loading.** `src/utils/seed.py:seed_worker`
   was defined (and `docs/reproducibility.md` claimed it was used) but was
   **never actually passed to any `DataLoader`** -- a real
   documentation/implementation mismatch. Now wired as `worker_init_fn` in
   both `src/training/trainer.py` (train/val loaders) and
   `src/training/pretrain.py`, so `num_workers > 0` (the default, 2) is
   actually reproducible per the seed, matching what the docs already claimed.

3. **`pin_memory` on CUDA.** Both DataLoaders above now pass
   `pin_memory=(device == "cuda")`, speeding up host->device transfer on
   GPU runs at zero cost on CPU/MPS.

4. **Smoke-mode speed cap: batches-per-epoch, not just epochs.**
   `RUN_PROFILE="smoke"` already capped `MAX_EPOCHS`/`EARLY_STOPPING_PATIENCE`
   to 1, but a "1 epoch" smoke test still iterated the *entire* dataset
   once -- slow on a large dataset and not what a fast integration check
   needs. Added `training.max_train_batches_per_epoch` /
   `max_val_batches_per_epoch` (optional, default `None` = unlimited,
   zero behavior change for any run that doesn't set them) to
   `Trainer._run_batches`, and wired `RUN_PROFILE="smoke"` in both
   notebooks to set `MAX_BATCHES_PER_EPOCH = 3`. See
   `tests/test_trainer_observability.py` for the new regression tests.

5. **FastAPI: modernized `@app.on_event("startup")` to the `lifespan`
   context-manager pattern** (the pre-0.109-compatible, non-deprecated
   API). This also **fixed a real, silent bug**: `_configure_cors()` used
   to run at import time (required, since middleware can't be added after
   the app starts) but read `app_state.config`, which was still the empty
   `{}` from `AppState.__init__` at that point (startup, which populates
   it, hadn't run yet) -- so the CORS middleware was **always** falling
   back to the wildcard `["*"]` default, silently ignoring
   `configs/api.yaml`'s configured
   `cors_origins: ["http://localhost:5173", "http://127.0.0.1:5173"]`.
   `_configure_cors()` now loads `configs/api.yaml` directly and
   independently of `app_state`, so the configured origins actually apply.

6. **Code-quality cleanups**: replaced two `__import__("numpy")` calls in
   `scripts/generate_gradcam.py` with a normal top-level `import numpy as np`;
   removed a stray extra blank-line gap in `src/evaluation/metrics.py`.

### What was deliberately left alone

- **No migration to PyTorch Lightning or any new training framework.**
  This codebase's hand-rolled `Trainer` is mature, fully covered by tests,
  and its design (progressive stage freezing, dual-checkpoint-metric
  tracking, incremental crash-safe history) is intentional and documented.
  Rewriting it onto a new framework would be a large, high-risk change
  disproportionate to an audit pass, not a "fix" -- flagged in Section 6
  as a future option, not attempted here.
- **Numerical stability**: reviewed the pinball loss, homoscedastic
  uncertainty weighting, softplus-based quantile ordering, and the
  gender-confidence-threshold path specifically for NaN/Inf risk. No bugs
  found -- masked losses already correctly early-out to a zero-valued
  (not NaN) tensor when a batch has no labels for a task, `softplus`
  guarantees non-negative quantile deltas, and `_balanced_score` /
  `_maybe_checkpoint` already explicitly guard every NaN comparison.

---

## 5. Execution Modes and Colab/Kaggle Notebook Configuration

See Section 6 of the audit request. Full detail lives in
`docs/execution_modes.md` (new).

---

## 6. Recommendations for Future Work

- **Wire or remove `src/models/baselines.py`** (see Section 3) -- a
  trivial-baseline sanity check ("do we beat a constant-quantile
  predictor?") is cheap to add to `scripts/evaluate.py`'s output and would
  give the ablation suite one more honesty check for free.
- **Consider `pin_memory=True` on the inference-side DataLoaders** too
  (`scripts/evaluate.py`, `calibrate.py`, `build_knn_index.py`,
  `generate_architecture_report.py`) -- left out of this pass since they're
  single-pass, not the training-loop bottleneck the task emphasized, but a
  trivial follow-up.
- **`GradNorm` loss balancing**: not implemented; if wanted, it would slot
  in as a third `model.loss_balancing.mode` alongside `fixed` and
  `learned_uncertainty` in `src/losses/multitask_loss.py` -- a genuinely
  new feature, not a refactor of something already there.
- **Colab resume-across-session-restart**: `RESUME_RUN_ID` currently
  requires `WORKSPACE_DIR/<RUN_ID>` to still exist on local Colab disk
  (`/content`, which is wiped on a fresh Colab VM). Cross-session resume
  after `/content` is wiped would need an explicit "restore this run from
  Drive into `/content/agegender_runs/<RUN_ID>` first" step before the
  existing resume logic can find it locally. Not attempted here since it's
  a real feature addition, not a bug fix, and the existing safeguards
  (never overwrite an existing run dir, `safe_copy2` same-file guard) are
  already correct for same-session resume and cross-session Drive sync.
- **`export_report.py`**: add a `make export-report` Makefile target and a
  short README mention (see Section 3) rather than leaving it
  undiscoverable.
