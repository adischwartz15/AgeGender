# face-multitask-research

Multi-task face age and dataset gender-label prediction with shared
representations, task-specific adapters, learned loss balancing,
uncertainty estimation, non-parametric comparison, and Grad-CAM
explainability -- built from scratch on a manually implemented ResNet-18.

> **Research and demonstration only.** Predictions may be inaccurate,
> biased, or unreliable. Gender-related output reflects labels in the
> training dataset and is **not** a determination of identity. This
> project must not be used for employment, policing, surveillance,
> identity verification, medical diagnosis, admissions, insurance, or any
> other high-impact decision. See [Ethical limitations](#ethical-limitations).

## Table of contents

- [Research objective](#research-objective)
- [Ethical limitations](#ethical-limitations)
- [Architecture](#architecture)
- [Results](#results)
- [Repository layout](#repository-layout)
- [Setup](#setup)
- [Kaggle API setup](#kaggle-api-setup)
- [Dataset format](#dataset-format)
- [Data validation workflow](#data-validation-workflow)
- [Training](#training)
- [Self-supervised pretraining (optional)](#self-supervised-pretraining-optional)
- [Architecture ablation experiments](#architecture-ablation-experiments)
- [Conformal calibration](#conformal-calibration)
- [k-NN non-parametric baseline](#k-nn-non-parametric-baseline)
- [Robustness evaluation](#robustness-evaluation)
- [Grad-CAM explainability](#grad-cam-explainability)
- [Backend (FastAPI)](#backend-fastapi)
- [Frontend (React)](#frontend-react)
- [Example API requests](#example-api-requests)
- [Evaluation metric definitions](#evaluation-metric-definitions)
- [Gradient interference and representation similarity](#gradient-interference-and-representation-similarity)
- [Troubleshooting](#troubleshooting)
- [Compute requirements](#compute-requirements)
- [Results depend on your data](#results-depend-on-your-data)

## Research objective

Given a face image, predict:

- **Estimated age**, as a central estimate (q50) plus a q10-q90 prediction interval.
- **Age uncertainty**, both raw (pinball/quantile) and conformal-calibrated.
- **Dataset gender-label prediction**: softmax probabilities over dataset-defined labels, or **"Not sure"** below a configurable confidence threshold.
- **Separate Grad-CAM heatmaps** ("model attention visualization") for the age and gender-label decisions.
- **A parametric-vs-non-parametric comparison**: the trained model's heads vs. a k-NN classifier/regressor in its own embedding space.

The central research question: does a **shared** visual backbone learn
useful common features for both tasks, and do **task-specific adapters**
plus **learned loss balancing** reduce negative transfer relative to
independent backbones and fixed loss weights? See
`docs/experiment_plan.md` and `docs/architecture_analysis.md`.

## Ethical limitations

- **"Dataset gender-label prediction"**, not "gender prediction" -- the
  output reflects a label defined by whichever dataset you train on, not a
  determination of a person's gender identity. Class names default to the
  neutral `gender_label_0` / `gender_label_1` and are only ever displayed
  differently if you explicitly configure alternative names based on your
  dataset's own documentation (`GENDER_LABEL_0`/`GENDER_LABEL_1` in `.env`,
  or `model.gender_head.class_names` in `configs/model.yaml`).
- Dataset labels may be binary, incomplete, inaccurate, self-reported,
  annotator-assigned, or culturally limited.
- Race/ethnicity metadata (when present, e.g. in UTKFace) is **never** used
  as a feature, prediction target, or split criterion.
- Uploaded images are processed in memory and **not persisted to disk** by
  the API by default.
- This system has not been validated for, and must not be used for:
  employment, policing, surveillance, identity verification, medical
  diagnosis, admissions, insurance, or any other high-impact decision.
- Grad-CAM output is a gradient-weighted visualization, **not proof of
  causality** and not an explanation of the model's reasoning.

## Architecture

```
Input face image
      |
      v
Custom ResNet-18 backbone (manually implemented, block layout [2,2,2,2])
      |
      v
Shared feature vector z (512-d)
      |
      +----------------------+----------------------+
      |                                             |
      v                                             v
Age Adapter (residual bottleneck)          Gender Adapter (residual bottleneck)
      |                                             |
      v                                             v
Age Quantile Head                          Gender Classification Head
  -> q10, q50, q90                            -> probabilities, or "Not sure"
```

- **Backbone**: `src/models/custom_resnet.py` -- hand-written `BasicBlock`
  residual blocks, manual downsampling (strided 1x1 conv + BN shortcuts),
  stem conv + BN + ReLU + max-pool, adaptive average pooling, 512-d
  embedding. **No `torchvision.models`, `timm`, Hugging Face vision
  models, or downloaded pretrained checkpoints anywhere in this repo.**
  The only way to initialize non-random weights is a checkpoint produced
  by this repository (supervised training or the optional SimCLR-style
  pretraining) or a compatible local file you explicitly point at.
- **Adapters**: `src/models/adapters.py` -- `adapter_output = z + up(dropout(gelu(down(z))))`,
  configurable bottleneck dimension (default 128), near-identity at
  initialization (zero-initialized up-projection).
- **Heads**: `src/models/heads.py` -- age quantile head (safe
  `q50, q50 - softplus(.), q50 + softplus(.)` parameterization guaranteeing
  `q10 <= q50 <= q90`) and a softmax gender classification head.
- **Loss balancing**: `src/losses/multitask_loss.py` -- fixed weights or
  learned homoscedastic-uncertainty weighting, with masked losses so a
  task with no labels in a batch contributes nothing.

## Results

Real numbers from an actual training run on UTKFace (via the Kaggle API,
default configs unless noted) -- not fabricated placeholders. Reproduce
with `make experiments`, `make build-knn`, `make evaluate --compare-knn`,
`make robustness`, and `make architecture-report`; see
`docs/architecture_analysis_generated.md` for the full, regenerated
report after you run the pipeline yourself.

### Architecture parameter comparison (Experiments A-D)

| Experiment | Backbone params | Adapter params | Total params |
|---|---|---|---|
| A -- separate backbones | 22,353,024 | 0 | 22,484,997 |
| B -- shared, no adapters | 11,176,512 | 0 | 11,308,485 |
| C -- shared + adapters | 11,176,512 | 263,424 | 11,571,909 |
| D -- shared + adapters + learned balancing | 11,176,512 | 263,424 | 11,571,911 |

Sharing the backbone (B/C/D) roughly halves parameter count versus
independent backbones (A); adapters add back only ~2.4% of the shared
backbone's parameters per task. *Per-experiment accuracy/MAE comparison
(does sharing + adapters actually help, not just cost fewer parameters)
requires re-running `scripts/evaluate.py` against each experiment's
checkpoint and isn't included here yet -- the sections below reflect one
specific (shared-backbone + adapters) checkpoint, not a cross-experiment
comparison.*

### Parametric model vs. k-NN baseline (shared-backbone + adapters model)

| Metric | Parametric | k-NN (k=15) |
|---|---|---|
| Age MAE | 5.71 | 5.79 |
| Age RMSE | 8.32 | 8.53 |
| q10-q90 interval coverage | 0.79 | 0.91 |
| Mean interval width | 16.79 | 26.88 |
| Gender-label accuracy | 0.970 | 0.966 |
| Abstention rate | 0.192 | 0.179 |
| Latency per image (ms) | 1.8 | 2.0 |

The k-NN baseline is competitive on gender-label accuracy and reaches
*higher* interval coverage than the (uncalibrated) parametric model, at
the cost of much wider intervals -- consistent with a non-parametric
method being more conservative rather than more precise here.

### Gradient interference and representation similarity

Measured on the shared-backbone + adapters model (30 sampled batches;
see [Gradient interference and representation similarity](#gradient-interference-and-representation-similarity)
below for methodology):

- Mean task-gradient cosine similarity: **+0.08** (std 0.33) -- weakly
  positive, i.e. the age and gender-label gradients are not strongly in
  conflict on this dataset/split, with meaningful batch-to-batch variance.
- Linear CKA: shared-vs-age-adapter **0.79**, shared-vs-gender-adapter
  **0.90**, age-vs-gender-adapter **0.59** -- the gender adapter moves
  the shared representation less than the age adapter does, and the two
  adapters diverge from each other more than either diverges from the
  shared embedding.

### Robustness (deterministic corruptions, severity 1 of 3)

| Condition | Age MAE | Gender accuracy |
|---|---|---|
| Clean (no corruption) | 5.52 | 0.975 |
| Gaussian blur | 5.72 | 0.953 |
| Low resolution | 5.82 | 0.934 |
| Low brightness | 6.21 | 0.962 |
| JPEG compression | 6.60 | 0.960 |
| High brightness | 6.80 | 0.947 |
| Partial crop | 8.50 | 0.868 |
| **Partial occlusion** | **13.35** | 0.765 |
| **Gaussian noise** | **14.82** | 0.960 |

Gaussian noise and partial occlusion are, by a wide margin, the most
damaging conditions for age estimation in this run; gender-label
accuracy degrades more gracefully except under occlusion.

*(All numbers above are from one real training run and one dataset
split -- see [Results depend on your data](#results-depend-on-your-data);
they are not a claim about performance on any other dataset, population,
or camera.)*

## Repository layout

See the full tree in `docs/` if needed; top level:

```
configs/       YAML configuration (data, model, training, experiments, robustness, api)
src/           Library code (data, models, losses, training, evaluation, inference, api, utils)
scripts/       CLI entry points (one per pipeline stage)
tests/         Pytest suite, including a synthetic-data smoke training test
frontend/      React + TypeScript + Vite + Tailwind dashboard
docs/          Architecture analysis, experiment plan, model/data cards, reproducibility
data/          Local dataset (never committed); splits; not tracked by git
checkpoints/   Trained model checkpoints (never committed)
outputs/       Metrics, plots, reports, gradcam, robustness, calibration, kNN artifacts
```

## Setup

Requirements: Python 3.11+ (3.10+ also works), Node.js 20+/npm for the
frontend.

```bash
git clone <this-repo>
cd face-multitask-research
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env              # fill in Kaggle credentials if using Kaggle data
cd frontend && npm install && cd ..
```

Or simply: `make install`

## Kaggle API setup

1. Create a Kaggle account, then create an API token at
   <https://www.kaggle.com/settings> ("Create New Token"). This downloads
   `kaggle.json` -- do not commit it.
2. Set environment variables (in `.env`, sourced by your shell, or
   directly in your environment):
   ```
   KAGGLE_USERNAME=<your-username>
   KAGGLE_KEY=<your-key>
   KAGGLE_DATASET_SLUG=<owner>/<dataset-name>   # e.g. jangedoo/utkface-new
   ```
3. Run `make download-data` (wraps `scripts/download_kaggle_data.py`). It
   validates credentials, downloads via the official `kaggle` package
   (no scraping), extracts into `data/raw/`, skips re-downloading unless
   `--force` is passed, and writes `data/raw/manifest.json` with the
   slug, timestamp, and file/image counts.

If credentials or the dataset slug are missing, the script prints setup
instructions and exits non-zero instead of failing silently.

## Dataset format

**UTKFace-style (default, `DATASET_SOURCE=utkface`)**: filenames
`age_gender_race_date.jpg`, e.g. `25_0_2_20170116174525125.jpg` (age=25,
gender_label=0, race=2 -- race is metadata only, never a model input).

**Generic CSV (`DATASET_SOURCE=csv`)**: configure `configs/data.yaml`'s
`dataset.csv` block with your dataset's image root, metadata CSV path, and
column names for image path / age / gender-label / (optional) split /
(optional) subject ID / (optional) raw-value-to-{0,1} label mapping.
Missing age or gender-label values are allowed (masked out of the loss);
rows missing both are dropped.

See `docs/data_card.md` for full details.

## Data validation workflow

```bash
make prepare-data
```

Runs `scripts/prepare_data.py`: parses raw metadata, drops
corrupt/unreadable images and duplicate paths/content (SHA-256 hash),
reports age/gender-label distributions and image-size stats to
`outputs/data_quality/data_quality_report.json`, and writes a deterministic,
leakage-checked (subject-level when possible) train/val/test split to
`data/splits/full_metadata_with_splits.csv`. This split is reused by every
experiment so results are comparable.

## Training

```bash
make train                                   # single default configuration
make train ARGS="--set model.architecture=shared_no_adapters"   # override any config key
```

Runs `scripts/train.py`, which uses `src/training/trainer.py`'s
progressive Stage A -> B -> C fine-tuning (or a single supervised warm-up
stage with a logged warning if no pretrained backbone checkpoint is
configured -- freezing a randomly initialized backbone isn't scientifically
meaningful). Saves three "best" checkpoints (lowest val age MAE, highest
val gender accuracy, best balanced score), training curves, and a
parameter breakdown to `outputs/` and `checkpoints/`.

## Self-supervised pretraining (optional)

```bash
make pretrain
```

Runs a lightweight SimCLR-style contrastive pretraining pass
(`scripts/pretrain.py` / `src/training/pretrain.py`) on the same Custom
ResNet-18 backbone with a separate projection head (discarded after
pretraining). Saves `checkpoints/simclr_encoder.pt`; point
`model.pretrained_checkpoint` at it to enable staged fine-tuning and
Experiment F. This is optional and more compute-hungry than supervised
training -- see `docs/reproducibility.md`.

## Architecture ablation experiments

```bash
make experiments
```

Runs `scripts/run_experiments.py` against `configs/experiments.yaml`
(Experiments A-F, see `docs/experiment_plan.md`), reusing the same split
for all of them. Experiment E (parametric vs. kNN) and Experiment F
(pretrained vs. scratch, only if a pretrained checkpoint exists) are
handled via the dedicated commands below rather than a fresh training run.

## Conformal calibration

```bash
make calibrate CHECKPOINT=checkpoints/<your_checkpoint>.pt
```

Fits split-conformal calibration (`src/evaluation/calibration.py`) on the
**validation set only**, saving the offset to
`outputs/calibration/conformal_calibration.json`. Reports coverage/width
before and after calibration on the test set
(`outputs/calibration/calibration_test_effect.json`). The API and
`Predictor` only ever describe an interval as "calibrated" when this
artifact exists and loaded successfully.

## k-NN non-parametric baseline

```bash
make build-knn CHECKPOINT=checkpoints/<your_checkpoint>.pt
```

Extracts L2-normalized embeddings from the training split
(`src/evaluation/knn_baseline.py`, backed by `sklearn.neighbors.NearestNeighbors`),
fits a distance-weighted k-NN index (default k=15) separately for the age
and gender-label embedding spaces, and saves it to
`outputs/knn/knn_baseline.pkl`. Compare against the parametric model with:

```bash
make evaluate CHECKPOINT=checkpoints/<your_checkpoint>.pt
```

(`scripts/evaluate.py --compare-knn`, producing `outputs/knn/parametric_vs_knn.csv`).

## Robustness evaluation

```bash
make robustness CHECKPOINT=checkpoints/<your_checkpoint>.pt
```

Evaluates the test set under 8 deterministic corruption types x 3
severities (`configs/robustness.yaml`, `src/evaluation/robustness.py`):
Gaussian blur, Gaussian noise, low resolution, JPEG compression, low/high
brightness, partial occlusion, partial crop. Saves
`outputs/robustness/robustness_results.csv`, plots, sample corrupted
images, and a Markdown summary.

## Grad-CAM explainability

```bash
make gradcam CHECKPOINT=checkpoints/<your_checkpoint>.pt
```

Manually implemented Grad-CAM (`src/evaluation/gradcam.py`, no external
Grad-CAM library) at the last residual stage (`layer4` by default).
Generates **separate** heatmaps for the age (q50) and gender-label
(selected class logit) decisions across correct, incorrect,
low-confidence, and blurred examples, saved to `outputs/gradcam/`. Always
labeled "Model attention visualization" -- never described as proof of
causality or an explanation of reasoning.

## Backend (FastAPI)

```bash
make api
```

Starts Uvicorn on `:8000`. Endpoints:

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | Liveness + whether a model is loaded |
| GET | `/models` | Active model/version/checkpoint info |
| POST | `/quality-check` | Image-quality diagnostics only (no model needed) |
| POST | `/predict` | Full prediction (`?include_gradcam=true&include_knn=true` optional) |
| POST | `/predict/compare` | Prediction with k-NN comparison always included |
| POST | `/predict/gradcam` | Prediction with Grad-CAM always included |
| POST | `/admin/reload-models` | Reload checkpoint/calibration/kNN index from disk |

Uploaded images are processed in memory and are not persisted to disk by
default.

**Face-region preprocessing.** Since the model is trained on tightly
face-cropped images (e.g. UTKFace), `/predict` and friends first try to
crop to the largest detected face using a classical Haar cascade
(`src/inference/face_detection.py` -- OpenCV's bundled Viola-Jones
detector, not a neural network, no pretrained weights downloaded), so an
arbitrary uploaded photo (with background, clothing, hair styling, etc.)
is closer to what the model actually learned from. **If no face is
found, the API declines to generate an age or dataset gender-label
prediction at all** (`age`/`gender`/`gradcam` are returned as `null`,
with a warning explaining why) rather than running the model on a
non-face image and returning a confident-looking but meaningless
result -- e.g. a photo of an object or an animal should not receive an
age or gender-label guess. Toggle via `api.enable_face_detection` /
`api.face_margin_ratio` in `configs/api.yaml`. This is a real but
classical/moderate-accuracy detector -- it can miss faces at extreme
angles, in poor lighting, or when occluded, and does not perform
identity verification or any other biometric function.

## Frontend (React)

```bash
make frontend
```

Starts the Vite dev server on `:5173` (proxies `/api` to `:8000`, see
`frontend/vite.config.ts`). Drag-and-drop upload, image preview, privacy
notice, age/gender/uncertainty/quality/Grad-CAM/kNN-comparison panels, and
a visible research disclaimer, built with Tailwind CSS.

## Example API requests

```bash
curl http://localhost:8000/health

curl -X POST http://localhost:8000/quality-check \
  -F "file=@/path/to/face.jpg"

curl -X POST "http://localhost:8000/predict?include_gradcam=true&include_knn=true" \
  -F "file=@/path/to/face.jpg"
```

Example (abridged) `/predict` response shape, when a face was detected:

```json
{
  "age": {"q10": 24.1, "q50": 29.4, "q90": 35.2, "q10_calibrated": 22.8, "q90_calibrated": 36.9, "is_calibrated": true},
  "gender": {"probabilities": {"gender_label_0": 0.91, "gender_label_1": 0.09}, "predicted_label": "gender_label_0", "confidence": 0.91, "abstained": false, "display_label": "gender_label_0"},
  "quality": {"width": 512, "height": 512, "brightness": 0.52, "contrast": 0.21, "blur_score": 143.2, "warnings": []},
  "gradcam": null,
  "knn_comparison": null,
  "model_version": "v1",
  "checkpoint_name": "multitask_best_balanced_score.pt",
  "face_detected": true,
  "warnings": [],
  "latency_ms": 42.3,
  "disclaimer": "This tool is for research and demonstration only. ..."
}
```

...and when no face was detected (`age`/`gender`/`gradcam`/`knn_comparison`
are all `null` -- no prediction is generated):

```json
{
  "age": null,
  "gender": null,
  "quality": {"width": 800, "height": 600, "brightness": 0.61, "contrast": 0.30, "blur_score": 210.5, "warnings": []},
  "gradcam": null,
  "knn_comparison": null,
  "model_version": "v1",
  "checkpoint_name": "multitask_best_balanced_score.pt",
  "face_detected": false,
  "warnings": ["No face detected via classical Haar-cascade detection; declining to generate age or dataset gender-label predictions, since the model is only meaningful on face images similar to its training data."],
  "latency_ms": 8.1,
  "disclaimer": "This tool is for research and demonstration only. ..."
}
```

## Evaluation metric definitions

- **Age MAE / RMSE / R2**: standard regression metrics on q50 vs. true age.
- **q10-q90 coverage**: fraction of samples where the true age falls inside `[q10, q90]`.
- **Mean/median interval width**: `q90 - q10`, averaged (or median) across samples.
- **Calibration error**: `|empirical coverage - target coverage|` for the (nominal 80%) interval.
- **Age error by bucket**: MAE computed within fixed age ranges (0-10, 10-20, ..., 80+).
- **Gender-label accuracy**: computed only over non-abstained predictions.
- **Abstention rate**: fraction of samples where confidence fell below the threshold and "Not sure" was returned.

## Gradient interference and representation similarity

See `docs/architecture_analysis.md` for the full methodology behind
gradient cosine similarity (age-loss vs. gender-loss gradients w.r.t. the
shared backbone) and linear CKA (shared embedding vs. each adapter's
output), and `docs/architecture_analysis_generated.md` (produced by
`make architecture-report`) for your actual run's numbers.

## Troubleshooting

- **"No trained checkpoint found"** from the API: train a model first
  (`make train` or `make experiments`) and confirm
  `configs/api.yaml: api.active_checkpoint` points at a real file, then hit
  `POST /admin/reload-models`.
- **Kaggle download fails with a credentials error**: confirm
  `KAGGLE_USERNAME`/`KAGGLE_KEY`/`KAGGLE_DATASET_SLUG` are set, either in a
  local `.env` file (copied from `.env.example` -- loaded automatically by
  `src/utils/config.py:load_env_file()`) or exported directly in your shell
  (shell exports always take priority over `.env`).
- **CUDA out of memory**: lower `training.batch_size` in
  `configs/training.yaml`, or run on CPU (`device: cpu` in
  `configs/default.yaml`) for small-scale experimentation.
- **Predictions look wildly wrong / overconfident on a photo that looks
  fine to you**: check the response's `face_detected` field and
  `warnings`. The model is trained on tightly face-cropped images (e.g.
  UTKFace); a photo with a lot of background, heavy styling/makeup,
  jewelry, or a watermark/overlay across the face is a different visual
  distribution even when a human would call it "a clear photo of a
  person," and face-crop preprocessing (see "Face-region preprocessing"
  above) only partially compensates for that gap. Try a plain,
  front-facing, tightly-cropped photo to check whether the issue is
  input distribution rather than a bug.
- **Frontend can't reach the API**: confirm the backend is running on
  `:8000` (`curl http://localhost:8000/health`) and that the Vite dev
  server's proxy config (`frontend/vite.config.ts`) points at it.
- **Tests fail with `ModuleNotFoundError: src`**: run pytest from the
  repository root (`pyproject.toml` sets `pythonpath = ["."]`), or
  `pip install -e .`-equivalent isn't required since tests add the repo
  root to `sys.path` implicitly via `pytest`'s rootdir behavior.

## Compute requirements

See `docs/reproducibility.md` for a stage-by-stage breakdown. Summary:
supervised training on a few thousand 128px images is feasible on a single
consumer GPU in well under an hour per experiment; CPU-only training works
for small-scale testing/smoke runs but is not recommended for full
experiments; self-supervised pretraining is markedly more compute-hungry
than supervised training for a comparable epoch count.

## Results depend on your data

Every number this repository produces -- age MAE, gender-label accuracy,
interval coverage, robustness curves, gradient interference, CKA -- is a
property of **the specific dataset, labels, split, and evaluation design
you use**, not a universal statement about the underlying task. Different
datasets have different demographic coverage, label quality, and image
conditions; do not extrapolate results here to populations, cameras, or
use cases outside the evaluation data. See `docs/data_card.md` and
`docs/model_card.md`.
