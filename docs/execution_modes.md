# Execution Modes and Notebook Configuration

Covers the `RUN_PROFILE` execution modes shared by both notebooks
(`notebooks/train_evaluate_colab.ipynb`, `notebooks/train_evaluate_kaggle.ipynb`)
and every hyperparameter/path/flag in their "USER CONFIGURATION" cell.

## 1. Execution modes (`RUN_PROFILE`)

Defined in the notebooks' "Controlled CNN-vs-ResNet experiment setup" cell
(`PROFILE_EXPERIMENTS`), which maps each profile to the experiments it runs:

| `RUN_PROFILE` | Experiments run | Epochs/patience | When to use |
|---|---|---|---|
| `"smoke"` | SimpleCNN (Experiment 0) + Custom ResNet-18 (Experiment D) | Capped to 1 epoch, patience 1, and (as of this audit) `MAX_BATCHES_PER_EPOCH` capped to 3 | Fast end-to-end pipeline validation only -- confirms every stage (train -> calibrate -> evaluate -> optional robustness/Grad-CAM/kNN) runs without crashing, on real code with a real (tiny) forward/backward pass. **Results are explicitly never treated as scientific findings** -- the notebook prints a warning and skips the CNN-vs-ResNet comparison table and the detailed results report entirely for this profile. Use before a real run, after any code change, or to validate a new environment/dataset path. |
| `"core"` | SimpleCNN (Experiment 0) + Custom ResNet-18 (Experiment D) | Full `MAX_EPOCHS`/`EARLY_STOPPING_PATIENCE` (default 40/12) | The default, "normal" run: the two models needed for the headline SimpleCNN-vs-ResNet efficiency/accuracy comparison, at full training length. Use for a real (non-smoke) single-seed result without the full A-D ablation suite or the residual-connection-specific control. |
| `"backbone_comparison"` | SimpleCNN + **PlainDeep18NoSkip** (Experiment 0b) + Custom ResNet-18 | Full | Adds the depth/width-matched no-skip-connection backbone, which `"core"` does not train. This is the profile that unlocks the *real* residual-connection ablation (PlainDeep18NoSkip vs. ResNet) and triggers `scripts/compare_backbones.py`'s full suite (AURC, paired bootstrap CIs, tail-error analysis, final conditional interpretation) in a dedicated notebook cell. Use when the actual research question ("do residual connections help, holding depth/width fixed?") is what you want answered, not just the efficiency/accuracy trade-off. |
| `"full"` | SimpleCNN, Experiment A (separate), Experiment B (shared, no adapters), Experiment C (shared + adapters), Experiment D (+ learned balancing) | Full | The complete architecture ablation suite (A -> B -> C -> D), isolating sharing, adapters, and learned loss balancing individually, plus the SimpleCNN baseline. This is the profile for a genuine "official" full results run (see `FORCE_RERUN=True` + `RUN_PROFILE="full"` in the config cell, which is the pre-commented "first official experiment run" combination). Takes the longest by a wide margin (5 architectures trained at full length). |

All four profiles are validated against `configs/experiments.yaml` at
notebook run time (`PROFILE_EXPERIMENTS` + explicit `RuntimeError`s if a
required experiment config is missing) -- an unknown or misconfigured
profile fails loudly before any training starts, never silently
substitutes a different model.

Independently of `RUN_PROFILE`, `RUN_MULTI_SEED=True` (with `SEEDS`, default
`[42, 123]`) additionally re-runs the profile's core experiments
(SimpleCNN, PlainDeep18NoSkip if present, ResNet) at every seed via
`scripts/run_seeds.py`, for a real mean +/- std rather than a single-run
point estimate -- see `docs/final_evaluation_protocol.md` for why these
three specific seeds are pre-registered.

## 2. The "USER CONFIGURATION" cell, flag by flag

This is the single cell users are expected to edit before running either
notebook top to bottom.

| Flag | Controls | Notes |
|---|---|---|
| `RUN_PROFILE` | Which experiments run and at what length | See table above. |
| `FORCE_RERUN` | Whether to retrain/recalibrate/re-evaluate a stage even if its artifact already exists | Default `False` (restart-safe: an already-complete stage is skipped). Set `True` only for an "official clean rerun" (paired with `RUN_PROFILE="full"` for the canonical combination, which the notebook prints a banner for). |
| `ALLOW_TEST_FAILURES` | Whether the notebook continues past a failing `pytest` run | Default `False` -- a real test failure should stop the notebook, not be silently ignored. |
| `RUN_ROBUSTNESS` / `RUN_GRADCAM` / `RUN_KNN` | Whether each optional analysis stage runs | All independently toggleable; each writes into that experiment/seed's own isolated subdirectory. |
| `RUN_MULTI_SEED` | Whether the multi-seed aggregation stage runs (see above) | Off by default since it multiplies training time by `len(SEEDS)`. |
| `SEEDS` | The pre-registered seed list for multi-seed runs | Default `[42, 123, 2026]`, matching `docs/final_evaluation_protocol.md`. `SEEDS[0]` (`PRIMARY_SEED`) is always used for the profile's single-seed run regardless of `RUN_MULTI_SEED`. |
| `MAX_EPOCHS` | Cap passed as `training.warm_up_from_scratch.epochs` (no pretrained backbone is used by default, so staged Stage A/B/C freezing is skipped -- see `src/training/stages.py`) | Default 30. Hard-capped to 1 automatically under `RUN_PROFILE="smoke"`. |
| `EARLY_STOPPING_PATIENCE` | Epochs of no `val_loss` improvement before stopping a stage early | Default 8. Hard-capped to 1 under `"smoke"`. |
| `MAX_BATCHES_PER_EPOCH` (new) | Optional hard cap on batches per epoch, independent of epoch count | Default `None` (unlimited). Auto-set to 3 under `"smoke"` so a "fast" smoke test is actually fast even on a large dataset, not just short in epoch count. |
| `REPO_URL` / `REPO_BRANCH` | Where the notebook clones/pulls this repository from | Must match the real remote (verified: `https://github.com/adischwartz15/AgeGender.git`, branch `main`). |
| `RESUME_RUN_ID` | Continue a previous run instead of starting fresh | Must be a `RUN_ID` still present under `WORKSPACE_DIR` (`/content/agegender_runs` on Colab) -- see the Pre-Flight section below for the one caveat here. |
| `USE_GOOGLE_DRIVE` (Colab only) | Whether the run directory is mirrored to Google Drive after every major phase and archived there at the end | Requires Drive to actually be mounted (handled automatically when this is `True`). |
| `KAGGLE_DATASET_SLUG` | Kaggle API dataset slug to download (e.g. `jangedoo/utkface-new`) | Requires `KAGGLE_USERNAME`/`KAGGLE_KEY` (Colab Secrets or a hidden prompt; never logged). |
| `DRIVE_DATASET_DIR` (Colab) / `KAGGLE_INPUT_DATASET_DIR` (Kaggle) | Use an already-available local/Drive/attached-input dataset instead of downloading | Exactly one dataset source (this or `KAGGLE_DATASET_SLUG`) must be set, or the dataset-setup cell raises. |