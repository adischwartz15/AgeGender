# Reproducibility

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

`scripts/prepare_data.py` writes a single `data/splits/full_metadata_with_splits.csv`.
Every experiment in `configs/experiments.yaml` reads this same file, so
Experiments A-F are comparable: differences in results reflect the
architecture/training change under test, not a different data split.

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

A ready-to-run Colab notebook implementing everything in this section --
Drive mounting, credential prompts, and a Drive-sync helper called after
every pipeline stage so checkpoints/outputs survive session recycling --
is provided at `notebooks/face_multitask_research_colab.ipynb`. Upload it
to https://colab.research.google.com (File -> Upload notebook) or open it
directly from GitHub once you've pushed this repo there.

The training/evaluation pipeline (`scripts/*.py`) is plain Python and has
no dependency on the FastAPI backend or the React frontend, so it runs
fine in a hosted notebook with a free GPU. The frontend is not needed to
reproduce any research result -- skip it entirely on Kaggle/Colab and just
inspect `outputs/plots/`, `outputs/gradcam/`, etc. as images in the
notebook.

### Getting the code onto the notebook

Neither platform can see your local disk directly. Easiest options,
in order of preference:
1. Push this repo to a GitHub repo (public or private), then
   `!git clone <your-repo-url>` in a notebook cell.
2. Zip the repo locally and use Colab's `files.upload()` (Colab) or
   "Add Data" -> upload a dataset from the zip (Kaggle), then unzip.

### Google Colab

```python
# Cell 1: get the code (pick one)
!git clone https://github.com/<you>/face-multitask-research.git
%cd face-multitask-research

# Cell 2: install dependencies (torch is already preinstalled on Colab
# GPU runtimes, so this mostly adds project-specific packages)
!pip install -r requirements.txt

# Cell 3: Kaggle credentials for scripts/download_kaggle_data.py -- set as
# environment variables directly rather than writing a kaggle.json file
# that could end up committed by accident.
import os
os.environ["KAGGLE_USERNAME"] = "<your-username>"
os.environ["KAGGLE_KEY"] = "<your-key>"
os.environ["KAGGLE_DATASET_SLUG"] = "jangedoo/utkface-new"  # or your dataset

!python scripts/download_kaggle_data.py
!python scripts/prepare_data.py
!python scripts/train.py           # Runtime -> Change runtime type -> GPU first
!python scripts/calibrate.py --checkpoint checkpoints/multitask_best_balanced_score.pt
!python scripts/build_knn_index.py --checkpoint checkpoints/multitask_best_balanced_score.pt
!python scripts/evaluate.py --checkpoint checkpoints/multitask_best_balanced_score.pt --compare-knn
!python scripts/run_robustness.py --checkpoint checkpoints/multitask_best_balanced_score.pt
!python scripts/generate_gradcam.py --checkpoint checkpoints/multitask_best_balanced_score.pt
!python scripts/generate_architecture_report.py --checkpoint checkpoints/multitask_best_balanced_score.pt
```

To view a generated plot inline: `from IPython.display import Image; Image("outputs/plots/multitask_training_curves.png")`.

If you want to expose the FastAPI backend from Colab for a quick demo
(optional, not needed for research results), tunnel it with a tool such as
`ngrok` or Colab's own port-forwarding rather than running the frontend
dev server there.

### Kaggle Notebooks

Kaggle Notebooks are more locked down (limited/no outbound internet by
default) but if you're already working with a Kaggle-hosted dataset, you
don't need the Kaggle API at all -- attach the dataset via **Add Data** in
the notebook UI, which mounts it read-only at `/kaggle/input/<dataset-slug>/`.

1. Add this code as a Kaggle Dataset (upload the zipped repo) or a Kaggle
   Notebook "Utility Script", then add it as an input to a new notebook.
2. Add your target dataset (e.g. UTKFace) via **Add Data** as well.
3. Enable a GPU accelerator in the notebook's Settings panel.
4. Point the config at the mounted dataset instead of downloading via the
   Kaggle API:
   ```python
   !pip install -r /kaggle/input/face-multitask-research/requirements.txt
   %cd /kaggle/working
   !cp -r /kaggle/input/face-multitask-research/* .
   !python scripts/prepare_data.py --set dataset.image_root=/kaggle/input/utkface-new/UTKFace
   !python scripts/train.py --set paths.checkpoint_dir=/kaggle/working/checkpoints --set paths.output_dir=/kaggle/working/outputs
   ```
   (adjust the input path to match whatever folder name Kaggle mounts your
   attached dataset under -- check the notebook's Data pane).
5. `scripts/download_kaggle_data.py` and the `KAGGLE_*` environment
   variables are only needed if you instead want to fetch a dataset via
   the Kaggle API from within the notebook (requires internet access
   enabled in notebook settings, plus your own API token set as Colab-style
   environment variables or Kaggle "Secrets").

Either platform will hit the free-tier session/GPU time limits before a
large multi-experiment `run_experiments.py` sweep finishes -- checkpoint
after each stage (already done automatically by the trainer) and resume
by re-running individual experiments rather than the whole sweep in one
session.
