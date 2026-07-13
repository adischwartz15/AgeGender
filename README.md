# Face Multi-Task Research

An uncertainty-aware multi-task vision system for age quantile estimation
and dataset gender-label classification, built around a manually
implemented ResNet-18, shared representations, task-specific adapters,
conformal calibration, and selective prediction.

> **Research and demonstration only.** Predictions may be inaccurate,
> biased, or unreliable. Gender-related output reflects labels in the
> training dataset and is **not** a determination of identity. This
> project must not be used for employment, policing, surveillance,
> identity verification, medical diagnosis, admissions, insurance, or any
> other high-impact decision. See [Ethical limitations](#ethical-limitations).

## Highlights

- **Manually implemented ResNet-18** backbone for every core, controlled
  experiment -- no `torchvision.models`, `timm` backbones, or externally
  pretrained checkpoints anywhere in the core ablation suite. A separate,
  clearly labelled **supplementary transfer-learning experiment** (optional,
  off by default) uses an ImageNet-pretrained VOLO-D1 backbone via `timm` --
  see [Supplementary experiment: ImageNet-pretrained VOLO-D1](#supplementary-experiment-imagenet-pretrained-volo-d1-face-only)
  below for the exact scope of that exception.
- **Shared backbone + task-specific residual bottleneck adapters** for
  age and dataset gender-label prediction, with an optional learned
  homoscedastic-uncertainty loss balancer instead of fixed weights.
- **Quantile-based age estimation** (q10/q50/q90) with **split-conformal
  calibration** for a marginal coverage guarantee, not just a point
  estimate.
- **Confidence-based abstention** ("Not sure") for the gender-label head,
  with selective/effective accuracy always reported together.
- **Parametric vs. k-NN comparison** in the model's own learned embedding
  space, deterministic **robustness testing** under 11 corruption types,
  and manually implemented **Grad-CAM** ("model attention visualization").
- **Controlled architecture ablations** (shared vs. separate backbones,
  adapters vs. none, fixed vs. learned loss balancing, and a true
  residual-connections ablation) with gradient-interference and
  representation-similarity (CKA) analysis.
- Reproducible, leakage-checked data splitting and fully isolated
  per-experiment/seed artifacts -- see [Reproducibility and scope](#reproducibility-and-scope).

## Research question

Does a **shared** visual backbone learn useful common features for both
age and dataset gender-label prediction, and do **task-specific
adapters** plus **learned loss balancing** reduce negative transfer
relative to independent backbones and fixed loss weights? Separately: is
the added complexity of a residual (skip-connection) architecture
actually justified, once measured against a depth/width-matched
non-residual baseline rather than an unrelated compact CNN?

These questions are answered by a config-driven ablation suite (Experiments
0/0b/0c, A-F) against one fixed, reused data split, following a protocol
pre-registered before results are observed -- see
[docs/experiment_plan.md](docs/experiment_plan.md) and
[docs/final_evaluation_protocol.md](docs/final_evaluation_protocol.md).
Results below are not a claim that every question was answered
conclusively; see [docs/results.md](docs/results.md) for what one real
run actually found.

## Architecture

```
Input face image
    |
    v
Custom ResNet-18 backbone (manually implemented)
    |
    v
Shared 512-dimensional embedding
    |
    +-- Age Adapter -------- Age Quantile Head -------- q10, q50, q90
    |
    +-- Gender Adapter ----- Classification Head ------- probabilities / "Not sure"
```

A single hand-written ResNet-18 backbone feeds two residual bottleneck
adapters, one per task, which in turn feed the age quantile head and the
gender classification head. Two controlled baseline backbones
(`simple_cnn`, `plain_deep18_no_skip`) also exist purely to isolate what
the residual design contributes -- neither is used by the deployed model.
See [docs/architecture_analysis.md](docs/architecture_analysis.md) for
the full module-by-module design and analysis methodology.

## Core vs. supplementary experiments

> All core controlled experiments are implemented from scratch and use no
> `torchvision.models`, `timm` backbones, or externally pretrained
> checkpoints. A separate supplementary transfer-learning experiment uses
> an ImageNet-pretrained VOLO-D1 backbone via `timm`. It is reported
> independently and is excluded from the controlled architectural
> ablations and from any conclusion about residual connections,
> task-specific adapters, or negative transfer.

Concretely: Experiments 0/0b/0c and A-F (the config-driven ablation suite in
`configs/experiments.yaml`, Table A) are the project's from-scratch
research contribution and are entirely unaffected by the supplementary
experiment below -- same code paths, same defaults, same results. The
supplementary VOLO-D1 experiment lives in its own files
(`src/models/pretrained_volo.py`, `src/training/transfer_trainer.py`,
`configs/transfer_learning.yaml`, `scripts/run_transfer_learning.py`),
writes to its own output paths, is never added to the default/`core`/
`backbone_comparison` profiles, and is reported in its own Table B, never
merged with Table A.

## Headline results

From one real training run on UTKFace (checkpoint
`exp_d_shared_adapters_learned_balance`, one seed, one dataset split --
see [docs/results.md](docs/results.md) for the full numbers, robustness
table, and gradient-interference/CKA analysis).

| Metric | Parametric | k-NN (k=15) |
|---|---|---|
| Age MAE | 5.71 | 5.79 |
| q10-q90 interval coverage (raw, uncalibrated) | 0.79 | 0.91 |
| Gender-label selective accuracy | 0.970 | 0.966 |
| Abstention rate (confidence threshold 0.80) | 0.192 | 0.179 |
| Latency per image (ms) | 1.8 | 2.0 |

| Experiment | Backbone params | Adapter params | Total params |
|---|---|---|---|
| A -- separate backbones | 22,353,024 | 0 | 22,484,997 |
| D -- shared + adapters + learned balancing | 11,176,512 | 263,424 | 11,571,911 |

"Gender-label selective accuracy" is computed only over non-abstained
predictions (see [docs/evaluation.md](docs/evaluation.md) for the
distinction from coverage and effective accuracy). The q10-q90 interval
is a nominal 80% interval before conformal calibration; the row above is
raw, not calibrated. These numbers describe one checkpoint on one
dataset split -- see [Reproducibility and scope](#reproducibility-and-scope).

## Quick start

```bash
git clone https://github.com/adischwartz15/AgeGender.git
cd AgeGender
make install
cp .env.example .env              # fill in Kaggle credentials (see docs/data_card.md)
make download-data
make prepare-data
make train
make calibrate CHECKPOINT=checkpoints/multitask_best_balanced_score.pt
make demo
```

Requirements: Python 3.11+ (3.10+ also works), Node.js 20+/npm for the
frontend. `make demo` checks readiness (a trained checkpoint + calibration
artifact) and launches both the API and the frontend together.

## Main workflow

```bash
make prepare-data                          # validate + split raw metadata
make train                                  # single default configuration
make experiments                            # full ablation suite (0/0b/0c, A-F)
make calibrate CHECKPOINT=<checkpoint>.pt   # split-conformal age intervals
make build-knn CHECKPOINT=<checkpoint>.pt   # k-NN baseline index
make evaluate CHECKPOINT=<checkpoint>.pt    # test-set metrics + k-NN comparison
make robustness CHECKPOINT=<checkpoint>.pt  # corruption robustness sweep
make gradcam CHECKPOINT=<checkpoint>.pt     # Grad-CAM heatmaps
make demo                                   # launch API + frontend together
```

`prepare-data`, `pretrain`, `train`, and `experiments` accept
`--set key.path=value` config overrides via `ARGS` (e.g.
`make train ARGS="--set model.architecture=shared_no_adapters"`) instead
of editing YAML in place. The evaluation-side commands (`calibrate`,
`build-knn`, `evaluate`, `robustness`, `gradcam`, `compare-backbones`,
`run-seeds`) take explicit flags instead (`CHECKPOINT=`, `EXPERIMENT=`,
`SEEDS=`, etc. -- see each script's `--help`), not `--set`. See
[Documentation](#documentation) below for the guide covering each stage.

## Supplementary experiment: ImageNet-pretrained VOLO-D1 (face-only)

**Motivation.** The core suite above answers a controlled question about
architecture design choices, all trained from scratch. A different,
practical question -- "how much does an externally pretrained modern visual
backbone actually buy over our best from-scratch system?" -- needs a
different kind of experiment, so it lives entirely separately: one
additional model, an ImageNet-pretrained VOLO-D1 (`timm`, `volo_d1_224`)
backbone feeding the *same, unmodified* task-specific adapters
(`src/models/adapters.py`) and the *same* learned homoscedastic-uncertainty
loss balancing (`src/losses/multitask_loss.py`) the core suite already uses.
Face-only -- not [MiVOLO](https://arxiv.org/abs/2307.04616), which
additionally requires a body crop, a person detector, and face+body
cross-attention; MiVOLO is cited below only as the motivation for trying a
VOLO-family backbone on faces, not as the architecture used here. Backbone
weights are ImageNet-only, validated against a closed allow-list before any
weight is loaded (`src/models/pretrained_volo.py::ALLOWED_PRETRAINED_SOURCES`)
-- never a MiVOLO or UTKFace-trained checkpoint, which would leak the test
split.

**Structure.**

```
configs/transfer_learning.yaml       transfer_learning_extension profile + transfer_learning_smoke
src/models/pretrained_volo.py        PretrainedVOLOFaceOnlyMultiTask (the VOLO wrapper)
src/training/transfer_trainer.py     2-stage trainer (frozen backbone -> fine-tune)
src/training/persistent_artifacts.py PersistentArtifactManager -- atomic, resumable, checksummed checkpoints
src/utils/kaggle_drive_backup.py     optional, secure Kaggle -> Google Drive backup (Kaggle Secrets only)
scripts/run_transfer_learning.py     orchestrates training/evaluation + builds Table B (--resume/--skip-completed)
requirements-transfer.txt            optional `timm` dependency
```

**Optional dependency.** `timm` is never a core dependency -- every core
experiment imports and runs with `timm` completely absent. Install it only
to run this extension:

```bash
pip install -r requirements-transfer.txt
```

Selecting the extension without `timm` installed fails immediately with an
actionable message ("The VOLO transfer-learning extension requires timm.
Install it with `pip install -r requirements-transfer.txt`."), never a
silent fallback.

**Two-stage schedule.** Stage 1 trains only the adapters/heads/loss-balancing
params (backbone frozen, a few epochs, larger LR); Stage 2 unfreezes the
backbone (fully, or its last N stages, per `training.finetune_unfreeze`) and
fine-tunes at a much lower backbone LR alongside a higher adapter/head/
balancing LR, with its own 4 separate optimizer parameter groups. AdamW,
gradient clipping, warmup+cosine, early stopping on validation only,
best-checkpoint restoration, gradient accumulation, and AMP on CUDA (never
on CPU). See `configs/transfer_learning.yaml` for the exact defaults.

**How to run it.** Canonical seeds are `42, 123, 2026` -- the same three
seeds the core suite's pre-registered protocol uses
(`docs/final_evaluation_protocol.md`), so Table B's VOLO row and baseline
row are always drawn from the same requested seed set.

```bash
python scripts/run_transfer_learning.py --smoke                            # tiny, non-scientific CPU integration check
python scripts/run_transfer_learning.py --seeds 42,123,2026                # fresh full run (needs a CUDA GPU in practice)
python scripts/run_transfer_learning.py --seeds 42,123,2026 \
    --resume --skip-completed --sync-after-epoch                          # resume after a disconnect, skip finished seeds
python scripts/run_transfer_learning.py --seeds 42,123,2026 --evaluate-only  # never trains -- rebuilds Table B from existing checkpoints only
```

**Output paths.** Never the core suite's `checkpoints/` / `outputs/` /
`experiments/` directories -- always
`checkpoints/transfer_learning/volo_d1_face_only_pretrained/` and
`results/transfer_learning/volo_d1_face_only_pretrained/`, plus
`results/transfer_learning/table_b.csv`, `table_b_manifest.json`, and
`seed_metrics_index.json` (git commit SHA, dependency versions, split
fingerprint, per-seed metrics).

**Persistence and resume.** A Colab/Kaggle runtime can disconnect mid-run
without losing progress: every epoch's checkpoint is written atomically
(temp file + `os.replace`), checksummed, and mirrored to a persistent root
(a mounted Google Drive folder on Colab, `/kaggle/working` on Kaggle) via
`src/training/persistent_artifacts.py::PersistentArtifactManager`. A seed
is only ever treated as complete once its completion marker, checksum, test
metrics, and split/model fingerprint all validate -- never from directory
existence alone -- so `--skip-completed` never re-trains a finished seed
and `--resume` never silently restarts an interrupted one from scratch. See
[docs/transfer_learning.md](docs/transfer_learning.md) "Persistent
artifacts", "Colab recovery", and "Kaggle recovery" for the full directory
layout and exact recovery commands.

**Resource requirements.** VOLO-D1 at 224px is far heavier than the core
suite's 128px models. Default batch size is conservative (16, with gradient
accumulation to an effective batch of 32) specifically to avoid OOM on a
free-tier Colab GPU; raise it only if your GPU has headroom. A CPU run is
possible (the smoke profile is CPU-only) but real training realistically
needs a GPU.

**Table B** (`results/transfer_learning/table_b.csv`) is a structurally
separate table from Table A (the core ablation table) -- never merged,
never compared cell-for-cell with Table A's rows. Columns: Model, Experiment
category, Initialization, Backbone, Adapters, Loss balancing, Input size,
Age MAE, Age RMSE, CS@5, Gender acc, Gender F1, Params, Trainable params.
With fewer than 2 completed seeds per row, values are explicitly labelled
"(n=1, no std)" rather than presented as a variance-estimated result.

**Limitations.**
- **Confounded comparison, by design.** VOLO-D1 differs from the
  from-scratch baseline in initialization, pretraining data, parameter
  count, input resolution (224px vs. 128px), **and** its own optimizer/
  training schedule (the 2-stage frozen-then-fine-tune trainer above, not
  the core suite's progressive Stage A/B/C trainer) all at once. Any
  difference in Table B reflects all of these simultaneously and cannot be
  attributed to "the Transformer-style backbone" alone.
- **Small dataset, large model.** A VOLO-D1-scale model can overfit a
  dataset of a few thousand UTKFace images; if it performs worse than the
  from-scratch baseline, that is reported as-is, not tuned away against the
  test set.
- **Single-seed risk.** Full multi-seed runs require real GPU time this
  project's authors may not always have budgeted; when fewer than 2 seeds
  complete per row, Table B is explicitly labelled single-seed and any
  interpretation is weakened accordingly (see the notebook's own
  interpretation cell for the exact wording used).

**Reproducibility.** Same split, same evaluation functions
(`scripts/evaluate.py::evaluate_checkpoint`, used unmodified for both
Table B rows), same postprocessing as the core suite. Config snapshot,
seed, dependency versions (Python/PyTorch/`timm`/CUDA), and the git commit
SHA are saved alongside every metrics file.

**Citations and licenses.** VOLO-D1 is an existing architecture
([Yuan et al., "VOLO: Vision Outlooker for Visual Recognition,"
arXiv:2106.13112](https://arxiv.org/abs/2106.13112)) with externally
sourced ImageNet weights, made available through
[`timm`](https://github.com/huggingface/pytorch-image-models) (Ross
Wightman et al., Apache-2.0). This project's contribution here is solely
the integration into the existing multi-task framework (adapters, heads,
loss balancing, evaluation) -- not the backbone architecture or its
weights. Face-only usage is motivated by MiVOLO ([Kuprashevich & Tolstykh,
"MiVOLO: Multi-input Transformer for Age and Gender Estimation,"
arXiv:2307.04616](https://arxiv.org/abs/2307.04616)), which this project
does **not** implement (no body crop, no person detector, no face+body
cross-attention). ImageNet: [Deng et al., 2009](https://ieeexplore.ieee.org/document/5206848).
`timm`'s code license (Apache-2.0) and the specific pretrained weights'
license are **not automatically the same** -- check the weight card for the
exact `model_id` in use (`configs/transfer_learning.yaml`) before any
redistribution or commercial use.

## Demo and API

```bash
make api        # FastAPI backend on :8000
make frontend   # Vite dev server on :5173
make demo       # both together, after a readiness check
```

`POST /predict` returns age (q10/q50/q90, raw and calibrated), gender-label
probabilities or "Not sure", and optional Grad-CAM/k-NN comparison.
Uploaded images are processed in memory and not persisted by default; if
no face is detected, the API declines to predict rather than guessing.
See [docs/api.md](docs/api.md) for the full endpoint table and example
requests/responses.

## Repository structure

```
configs/     YAML configuration (data, model, training, experiments, robustness, api)
src/         Library code (data, models, losses, training, evaluation, inference, api, utils)
scripts/     CLI entry points, one per pipeline stage
tests/       Pytest suite, including a synthetic-data smoke training test
frontend/    React + TypeScript + Vite + Tailwind dashboard
notebooks/   Self-contained Colab and Kaggle notebooks running the full pipeline
docs/        Architecture, experiments, data/model cards, API, reproducibility
```

`data/`, `checkpoints/`, `experiments/`, `outputs/`, and `results/` (the
supplementary transfer-learning extension's own output root) hold generated
artifacts and are never committed -- see
[docs/reproducibility.md](docs/reproducibility.md) for the full layout.

## Documentation

- [Architecture and model design](docs/architecture_analysis.md)
- [Experiment plan (Experiments 0/0b/0c, A-F)](docs/experiment_plan.md)
- [Final evaluation protocol (pre-registered)](docs/final_evaluation_protocol.md)
- [Headline results (full numbers)](docs/results.md)
- [Backbone comparison suite](docs/backbone_comparison.md)
- [Conformal calibration](docs/calibration.md)
- [Robustness evaluation](docs/robustness.md)
- [Evaluation metric definitions](docs/evaluation.md)
- [API usage](docs/api.md)
- [Supplementary experiment: ImageNet-pretrained VOLO-D1](docs/transfer_learning.md)
- [Colab and Kaggle notebooks](docs/notebooks.md)
- [Execution modes and notebook flags](docs/execution_modes.md)
- [Reproducibility](docs/reproducibility.md)
- [Data card](docs/data_card.md)
- [Model card](docs/model_card.md)
- [Troubleshooting](docs/troubleshooting.md)

## Ethical limitations

- **"Dataset gender-label prediction"**, not "gender prediction" -- the
  output reflects a label defined by whichever dataset you train on, not
  a determination of a person's gender identity. Class names default to
  the neutral `gender_label_0` / `gender_label_1`.
- Dataset labels may be binary, incomplete, inaccurate, self-reported,
  annotator-assigned, or culturally limited.
- Race/ethnicity metadata (when present, e.g. in UTKFace) is **never**
  used as a feature, prediction target, or split criterion.
- Uploaded images are processed in memory and **not persisted to disk**
  by the API by default.
- This system has not been validated for, and must not be used for:
  employment, policing, surveillance, identity verification, medical
  diagnosis, admissions, insurance, or any other high-impact decision.
- Grad-CAM output is a gradient-weighted visualization, **not proof of
  causality** and not an explanation of the model's reasoning.

See [docs/model_card.md](docs/model_card.md) and
[docs/data_card.md](docs/data_card.md) for the full discussion.

## Reproducibility and scope

Every reported number is a property of one specific dataset, split, seed,
and evaluation design -- not a universal statement about the underlying
task. All splits are fixed once and reused by every experiment; every
checkpoint/seed gets its own isolated artifact tree; calibration artifacts
record and verify provenance (checkpoint/split hashes) before being
applied; and nothing in this repository hardcodes example metrics as if
they were real results. Supervised training on a few thousand 128px
images is feasible on a single consumer GPU in well under an hour per
experiment. See [docs/reproducibility.md](docs/reproducibility.md) for
seeds, splits, compute expectations, and notebook details, and
[docs/data_card.md](docs/data_card.md) for demographic-coverage caveats.

## License / authors

MIT License -- see [LICENSE](LICENSE).
