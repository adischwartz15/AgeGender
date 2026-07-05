# Experiment Plan

This document describes the config-driven ablation suite in
`configs/experiments.yaml` and the specific hypotheses each experiment is
designed to test. All experiments train/evaluate against the same
`data/splits/full_metadata_with_splits.csv` (see `docs/reproducibility.md`),
so differences in outcome are attributable to the architecture/training
change under test.

## Research question

Does a single shared Custom ResNet-18 backbone learn visual features
useful for *both* age estimation and dataset gender-label classification,
and do task-specific bottleneck adapters plus learned uncertainty-based
loss balancing reduce negative transfer relative to naive sharing or fully
independent backbones? Separately: how does a non-parametric k-NN
classifier/regressor in the learned embedding space compare to the
parametric heads?

## Experiment A -- Separate models (`exp_a_separate`)

Two independent Custom ResNet-18 backbones, one trained only for age, one
trained only for dataset gender-label classification. This is the
"no sharing" baseline: it isolates what each task can achieve with a
dedicated backbone and establishes the parameter-cost ceiling (2x backbone
parameters) against which sharing is judged.

## Experiment B -- Shared backbone, no adapters (`exp_b_shared_no_adapters`)

One shared backbone feeds both heads directly, fixed loss weights. Tests
whether naive parameter sharing causes **negative transfer** (worse
per-task performance than Experiment A) or **positive transfer** (better,
due to shared low-level visual features like edges/texture/illumination
invariance).

## Experiment C -- Shared backbone + adapters (`exp_c_shared_adapters`)

Adds task-specific residual bottleneck adapters on top of Experiment B's
shared backbone. Tests whether adapters recover per-task specialization
lost in Experiment B while keeping most parameters shared (adapters are
configured to be a small fraction of backbone size, see
`docs/architecture_analysis.md`).

## Experiment D -- Shared + adapters + learned loss balancing (`exp_d_shared_adapters_learned_balance`)

Same architecture as C, but replaces fixed loss weights with learned
homoscedastic-uncertainty weighting (trainable log-variances per task).
Tests whether automatic loss balancing improves on manually fixed weights
once adapters already address representational conflict.

## Experiment E -- Parametric vs. k-NN (`exp_e_parametric_vs_knn`)

Not a separate training run: reuses Experiment D's (or the best-performing
experiment's) checkpoint. `scripts/build_knn_index.py` extracts embeddings
from the training split and fits a distance-weighted k-NN index;
`scripts/evaluate.py --compare-knn` compares the parametric heads against
k-NN prediction in the same embedding space on age MAE/RMSE/coverage/width,
gender accuracy/abstention, and inference latency.

## Experiment F -- Pretrained vs. scratch (`exp_f_pretrained_vs_scratch`, optional)

Compares a backbone initialized from this repository's own SimCLR-style
self-supervised pretraining (`scripts/pretrain.py`) against the same
architecture trained from scratch. Skipped automatically (with a logged
message) if no pretrained checkpoint exists -- this experiment is opt-in
because self-supervised pretraining is comparatively compute-hungry (see
`docs/reproducibility.md`).

## What "success" would look like (to be judged only from real results)

- **B vs. A**: shared backbone should not be meaningfully worse than
  separate backbones at a fraction of the parameters, ideally similar or
  better on at least one task (evidence of positive transfer).
- **C vs. B**: adapters should reduce or eliminate any negative-transfer
  gap seen in B, at a small parameter cost (see the adapter-vs-backbone
  parameter ratio in the architecture report).
- **D vs. C**: learned loss balancing should match or exceed C's fixed
  weights without manual tuning, and its effective weights (logged per
  epoch) should evolve to sensible task-difficulty-reflecting values.
- **E**: the k-NN baseline is expected to underperform the parametric
  model on average but may be competitive for in-distribution queries and
  provides interpretable "nearest examples" the parametric model cannot.
- **F**: pretraining may help most when labeled data is scarce; with a
  large labeled training set the gap over from-scratch training may be
  small.

This template intentionally does not claim any of the above outcomes
occurred -- see `docs/architecture_analysis.md` and the auto-generated
`docs/architecture_analysis_generated.md` for whatever your actual run
produced.
