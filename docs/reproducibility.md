# Reproducibility

## Repository layout

```
configs/       YAML configuration (data, model, training, experiments, robustness, api)
src/           Library code (data, models, losses, training, evaluation, inference, api, utils)
scripts/       CLI entry points (one per pipeline stage)
tests/         Pytest suite, including a synthetic-data smoke training test
frontend/      React + TypeScript + Vite + Tailwind dashboard
docs/          Architecture analysis, experiment plan, model/data cards, reproducibility
data/          Local dataset (never committed); splits; not tracked by git
checkpoints/   Trained model checkpoints (never committed)
experiments/   Isolated per-experiment/seed run trees (checkpoints, calibration,
               metrics, plots, robustness, knn) from run_seeds.py/run_experiments.py
outputs/       Cross-run artifacts: architecture_analysis, backbone_comparison,
               gradcam, reports, data_quality (single global outputs/calibration
               or outputs/robustness are never used by the isolated pipeline above)
```

`experiments/<experiment>/seed_<seed>/{checkpoints,calibration,metrics,plots,robustness,knn}`
(`src/utils/experiment_paths.py`) is the isolated artifact tree used by
`scripts/run_experiments.py` and `scripts/run_seeds.py` -- every
checkpoint gets calibrated and evaluated inside its own tree so two
experiments or seeds can never silently collide or contaminate each
other's calibration/robustness artifacts (see `docs/calibration.md`).

## Seeds

Every script that trains, splits, evaluates, calibrates, or runs
robustness corruptions accepts/uses a seed (`configs/default.yaml: seed`,
`configs/data.yaml: split.seed`, `configs/training.yaml: training.seed`,
`configs/robustness.yaml: robustness.seed`). `src/utils/seed.py:set_global_seed`
seeds Python's `random`, NumPy, and PyTorch (including CUDA) and enables
deterministic cuDNN algorithms.

Determinism caveats:
- cuDNN deterministic mode can slow down GPU training measurably.
- Some CUDA reduction ops are not bit-exact deterministic across GPU
  models/driver versions even with deterministic mode enabled; expect
  reproducibility "up to noise" across different hardware, exact
  reproducibility on the same machine/driver/PyTorch version.
- Multi-worker `DataLoader` workers are seeded via `seed_worker`, but OS
  thread scheduling can still introduce minor nondeterminism in data order
  timing (not in the seeded augmentation RNG itself).

## Splits are fixed once, reused everywhere

`scripts/prepare_data.py` writes a single `data/splits/full_metadata_with_splits.csv`
with four splits -- `train` / `validation` / `calibration` / `test`, each
used for exactly one purpose (see `docs/data_card.md`). Every experiment
in `configs/experiments.yaml` reads this same file, so Experiments 0, A-F
are comparable: differences in results reflect the architecture/training
change under test, not a different data split.

## Config-driven, not hardcoded

All architecture, training, and evaluation choices live in `configs/*.yaml`.
Scripts accept `--set key.path=value` overrides so ad hoc experiments don't
require editing YAML in place. Every checkpoint saved by
`src/training/checkpointing.py` embeds a full snapshot of the config used to
produce it, so any checkpoint can be inspected later to see exactly which
settings produced it.

## No fabricated results

`src/evaluation/reports.py` reads real artifacts from `outputs/` and renders
an explicit "not yet generated" placeholder (with the command that would
produce it) for any section whose backing file doesn't exist yet. Nothing
in this repository hardcodes example metrics as if they were real results.

## Compute expectations

These are rough CPU/GPU-agnostic expectations for the *default* configs on
a mid-range single GPU (e.g. a laptop RTX-class GPU); CPU-only training is
possible but much slower and is really only practical for the tests'
tiny synthetic smoke run, not real experiments.

| Stage | Rough cost |
|---|---|
| `scripts/prepare_data.py` | Seconds to ~1 minute per 10k images (I/O + hashing bound) |
| `scripts/train.py` (one architecture, Stage A+B+C or warm-up) | Minutes to ~1 hour per experiment on a few thousand images at 128px, GPU |
| `scripts/run_experiments.py` (Experiments A-D) | Roughly 4x a single `train.py` run |
| `scripts/pretrain.py` (SimCLR) | Meaningfully more expensive than supervised training for the same epoch count -- contrastive learning typically needs larger batch sizes and more epochs to show benefit; treat pretraining as optional and budget accordingly |
| `scripts/build_knn_index.py` | Fast: one forward pass per training image plus an in-memory k-NN fit |
| `scripts/run_robustness.py` | Roughly (1 + num corruption/severity combinations) x one evaluation pass |
| `scripts/generate_gradcam.py` | One forward+backward pass per sample per task, negligible for typical sample counts |

None of these numbers are measured on real hardware in this repository --
they are order-of-magnitude planning guidance. Actual `epoch_time_seconds`
values for your run are recorded by the trainer and reported in
`outputs/metrics/*_timing.json` / the architecture analysis report.

## Environment

- Python 3.11+ (developed/tested primarily against CPython 3.10-3.12; no
  3.11-only language features are used, so 3.10 works too).
- PyTorch (CPU or CUDA build; see `requirements.txt` -- this repository
  does not pin a specific CUDA version, install the PyTorch build matching
  your system per pytorch.org's instructions if the default pip package
  doesn't match your GPU/driver).
- Node.js 20+ / npm for the frontend.

## Running on Kaggle Notebooks / Google Colab

See `docs/notebooks.md` for the full guide (execution profiles, restart
safety, multi-seed runs, rerunning analysis without retraining). Summary:
two ready-to-run notebooks implement the full pipeline described in this
document -- environment/GPU checks, repository setup, dependency
installation, data validation and split preparation, automated tests, the
controlled plain-CNN-vs-Custom-ResNet-18 comparison, calibration,
evaluation, optional robustness/Grad-CAM/k-NN analyses, multi-seed
aggregation, and a final report + archive:

- `notebooks/train_evaluate_colab.ipynb` -- Google Colab. Trains under
  `/content` for speed and synchronizes checkpoints/metrics/plots/reports
  to Google Drive after every major phase.
- `notebooks/train_evaluate_kaggle.ipynb` -- Kaggle Notebooks. Uses an
  attached Kaggle input dataset (or the Kaggle API), never mounts Google
  Drive, and produces a downloadable zip archive under Kaggle's Output tab.

See `docs/notebooks.md` for which one to use. Both are pure orchestration around this repository's real
`scripts/*.py` -- they never reimplement model, dataset, training, or
evaluation logic, use a readable generated run ID
(`<date>_<time>_<profile>_seed<seed>`), never overwrite an existing run
directory, and are restart-safe (an already-complete experiment/seed is
reused unless `FORCE_RERUN=True`).

The training/evaluation pipeline (`scripts/*.py`) is plain Python and has
no dependency on the FastAPI backend or the React frontend, so it runs
fine in a hosted notebook with a free GPU. The frontend build is checked
(`RUN_FRONTEND_CHECKS=True`) but its dev server is never launched by
either notebook -- that's not needed to reproduce any research result.

Either platform will hit free-tier session/GPU time limits before a large
multi-experiment sweep finishes; both notebooks are restart-safe for
exactly this reason -- re-run the notebook (optionally setting
`RESUME_RUN_ID` to the previous run's printed ID) and already-complete
experiments are skipped rather than retrained.
