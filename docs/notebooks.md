# Colab and Kaggle Notebooks

Two polished, self-contained notebooks run the entire research pipeline
(setup, dependency install, data validation, tests, the controlled
plain-CNN-vs-Custom-ResNet-18 comparison, calibration, evaluation, optional
robustness/Grad-CAM/k-NN analyses, multi-seed aggregation, and a final
report + archive) end to end on a hosted GPU runtime, without needing a
local machine:

- **`notebooks/train_evaluate_colab.ipynb`** -- for Google Colab. Clones the
  repo under `/content/AgeGender`, trains under `/content` for speed, and
  synchronizes checkpoints/metrics/plots/reports to Google Drive after every
  major phase (never raw dataset images, unless you explicitly enable that).
  Use this if you want a persistent Drive record across Colab session
  recycling, or plan to run the notebook over several sessions.
- **`notebooks/train_evaluate_kaggle.ipynb`** -- for Kaggle Notebooks. Clones
  the repo under `/kaggle/working/AgeGender`, uses an attached Kaggle input
  dataset (or the Kaggle API) for data, never mounts Google Drive, and
  produces a downloadable `AgeGender_<RUN_ID>.zip` under Kaggle's Output tab.
  Use this if your dataset is already a Kaggle Dataset, or you prefer
  Kaggle's free GPU quota over Colab's.

Both notebooks are pure orchestration around this repository's real
`scripts/*.py` -- they never reimplement model, dataset, training, or
evaluation logic. Each run gets its own timestamped, non-overwriting run
directory (`config/`, `logs/`, `tests/`, `checkpoints/`, `metrics/`,
`plots/`, `calibration/`, `reports/`, `manifests/`, `archives/`,
`experiments/<name>/seed_<seed>/`, ...), and defaults to `RUN_PROFILE="core"`
(the two-experiment SimpleCNN-vs-ResNet comparison, 30 max epochs, patience
8, seed 42). Set `RUN_PROFILE="backbone_comparison"` to also train
PlainDeep18NoSkip (Experiment 0b) and run the full comparison suite
(`scripts/compare_backbones.py`) automatically. See
`docs/execution_modes.md` for the full `RUN_PROFILE` table and every
"USER CONFIGURATION" cell flag, and each notebook's first two cells for
the live configuration options (`RUN_PROFILE`, `SEEDS`, `RUN_ROBUSTNESS`,
`RUN_GRADCAM`, `RUN_KNN`, `RUN_MULTI_SEED`, etc.).

## Stage-level restart-safety

With `FORCE_RERUN=False` and `RESUME_RUN_ID` set to a previous run's
printed ID, each of training, calibration, k-NN index building, and
evaluation is independently skipped-or-rerun based only on whether *that
stage's own* artifact already exists -- printed up front as a
per-experiment stage plan (e.g. "training: skipped, checkpoint found" /
"evaluation: rerunning, metrics missing"). A failure in a later stage
(e.g. evaluation) never triggers retraining a checkpoint that already
completed successfully. This is not resumption of an interrupted training
run mid-epoch -- checkpoints in this repository never save optimizer
state (`src/training/checkpointing.py`), so a stage that was interrupted
partway through re-runs from its own beginning, not from a mid-epoch
point.

## Rerunning post-hoc analysis without retraining

Since training, calibration, and evaluation are all skipped once their
artifacts exist, simply re-running the notebook with the same
`RESUME_RUN_ID` and `FORCE_RERUN=False` re-executes only whatever hasn't
been produced yet (typically the report/analysis cells) -- nothing
upstream is retrained. To do the same thing outside a notebook against
existing checkpoints:

```bash
python scripts/compare_backbones.py \
    --checkpoint simple_cnn=checkpoints/exp_0_..._best_balanced_score.pt \
    --checkpoint custom_resnet18=checkpoints/exp_d_..._best_balanced_score.pt \
    --resnet-name custom_resnet18 --output-dir outputs/backbone_comparison
python scripts/generate_final_report.py
```

See `docs/backbone_comparison.md` for the full output-file list.

## Running only the new three-seed backbone comparison

Set `RUN_PROFILE="backbone_comparison"` and `RUN_MULTI_SEED=True` (with
`SEEDS=[42, 123, 2026]`, the default) in either notebook's configuration
cell -- section 16 will then train Experiments 0/0b/D at all three seeds
and save `reports/multiseed_summary.{csv,md}`. Outside a notebook:

```bash
python scripts/run_seeds.py --experiment exp_0_simple_cnn_shared_adapters_learned_balance --seeds 42,123,2026
python scripts/run_seeds.py --experiment exp_0b_plain_deep18_no_skip_shared_adapters_learned_balance --seeds 42,123,2026
python scripts/run_seeds.py --experiment exp_d_shared_adapters_learned_balance --seeds 42,123,2026
```

See `docs/final_evaluation_protocol.md` for why these three specific seeds
are pre-registered.

## Running only robustness evaluation

With the isolated layout above, `--output-dir`/`--calibration-dir` default
to that same checkpoint's own `robustness/`/`calibration/` subdirectories,
so they don't need to be passed explicitly:

```bash
python scripts/run_robustness.py --checkpoint experiments/exp_0_simple_cnn_shared_adapters_learned_balance/seed_42/checkpoints/exp_0_simple_cnn_shared_adapters_learned_balance_best_balanced_score.pt
python scripts/run_robustness.py --checkpoint experiments/exp_d_shared_adapters_learned_balance/seed_42/checkpoints/exp_d_shared_adapters_learned_balance_best_balanced_score.pt
```

Or in a notebook: set `RUN_ROBUSTNESS=True` and the other `RUN_*` optional
flags to `False`, then re-run with `RESUME_RUN_ID` set and
`FORCE_RERUN=False` -- training/calibration/evaluation are skipped
(already complete) and only the robustness cell executes. See
`docs/robustness.md` for what this produces.

## Platform limits

Either platform will hit free-tier session/GPU time limits before a large
multi-experiment sweep finishes; both notebooks are restart-safe for
exactly this reason -- re-run the notebook (optionally setting
`RESUME_RUN_ID` to the previous run's printed ID) and already-complete
experiments are skipped rather than retrained. The training/evaluation
pipeline (`scripts/*.py`) is plain Python and has no dependency on the
FastAPI backend or the React frontend, so it runs fine in a hosted
notebook with a free GPU. The frontend build is checked
(`RUN_FRONTEND_CHECKS=True`) but its dev server is never launched by
either notebook -- that's not needed to reproduce any research result.
See `docs/reproducibility.md` for the environment/compute expectations
that apply whether you run locally or in a notebook.
