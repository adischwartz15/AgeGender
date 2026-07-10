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

**Does the residual design itself provide measurable value?** This
question splits into two *distinct* claims, deliberately answered by two
separate controlled experiments (see `scripts/compare_backbones.py` and
`docs/architecture_analysis.md`):

1. **SimpleCNN vs. Custom ResNet-18 (Experiment 0 vs. D)** -- "is the
   larger residual architecture justified relative to a compact CNN?"
   SimpleCNN differs from Custom ResNet-18 in depth, stage widths, *and*
   the presence of skip connections all at once, so this is an
   **efficiency/accuracy trade-off** comparison, not a clean ablation of
   residual connections specifically. A ResNet win here could be explained
   by depth/width alone, with skip connections contributing nothing.
2. **PlainDeep18NoSkip vs. Custom ResNet-18 (Experiment 0b vs. D)** --
   "what is the contribution of residual skip connections when depth and
   width are held fixed?" PlainDeep18NoSkip
   (`src/models/plain_deep18_no_skip.py`) copies Custom ResNet-18's stem,
   stage widths, block layout, embedding size, and training recipe exactly,
   removing only the residual additions (and, unavoidably, the
   downsample-shortcut parameters that only exist to support them -- see
   Experiment 0b below for the exact count). This *is* a clean ablation of
   residual connections specifically.

**Do not treat a SimpleCNN-vs-ResNet result alone as evidence that residual
connections help** -- only Experiment 0b vs. D isolates that variable.

## Experiment 0 -- Plain CNN backbone baseline (`exp_0_simple_cnn_shared_adapters_learned_balance`)

A **controlled baseline, not a general CNN benchmark**: a conventional,
non-residual CNN (`src/models/simple_cnn.py` -- stacked Conv+BN+ReLU+MaxPool
blocks, no skip connections) substituted for the Custom ResNet-18 backbone,
with everything else held identical to Experiment D -- the same shared
multi-task structure, task-specific adapters, learned uncertainty loss
balancing, training setup, data split, and evaluation pipeline. The plain
CNN uses the same 512-d embedding output, so the adapters/heads/losses are
byte-for-byte the same code path regardless of which backbone feeds them.

This isolates one variable: **residual connections, present or absent**.
It deliberately does not compare a weak CNN with fixed losses against a
ResNet with adapters and learned balancing -- that would change too many
variables at once to attribute any difference to the backbone. This is
not intended to be tuned into a competitive standalone architecture, and
the plain CNN must never be described as this project's main backbone;
`CustomResNet18` remains that throughout.

```bash
# Run only the plain CNN baseline
python scripts/run_experiments.py --only exp_0_simple_cnn_shared_adapters_learned_balance

# Run the controlled CNN-vs-ResNet comparison (Experiment 0 + Experiment D)
python scripts/run_experiments.py --only exp_0_simple_cnn_shared_adapters_learned_balance,exp_d_shared_adapters_learned_balance

# Regenerate the research report, including the
# "Plain CNN vs Custom ResNet-18 Backbone Comparison" section
python scripts/generate_architecture_report.py --checkpoint checkpoints/exp_d_shared_adapters_learned_balance_best_balanced_score.pt
```

All three backbones (`custom_resnet18 | simple_cnn | plain_deep18_no_skip`)
expose the same `forward` / `forward_features` (`layer1`-`layer4`, for
Grad-CAM compatibility) / `num_parameters` interface, so the rest of the
pipeline (adapters, heads, trainer, evaluation, inference, Grad-CAM) is
unmodified by which one is active.

## Experiment 0b -- Plain, depth/width-matched, no-skip-connection backbone (`exp_0b_plain_deep18_no_skip_shared_adapters_learned_balance`)

The controlled residual-connections ablation Experiment 0 cannot provide
(see "Research question" above): `src/models/plain_deep18_no_skip.py`'s
`PlainDeep18NoSkip` uses the **same** stem, stage widths (64/128/256/512),
block layout `[2, 2, 2, 2]` (two 3x3 convolutions per block), BatchNorm,
ReLU placement, embedding size, adapters, heads, learned-uncertainty loss
balancing, and training recipe as Custom ResNet-18 (Experiment D) -- the
only change is that `PlainBlock.forward` never adds an identity/projection
shortcut.

**Unavoidable parameter difference.** Because there is no residual addition,
there is also no need for the three downsample shortcuts (1x1 conv +
BatchNorm) Custom ResNet-18 uses at the layer2/layer3/layer4 channel/stride
transitions -- `PlainDeep18NoSkip` has exactly **173,824 fewer parameters**
than `CustomResNet18` (11,002,688 vs. 11,176,512 with default
`stem_channels=64`), matching those three shortcuts' parameter count
exactly (verified in `tests/test_plain_deep18_no_skip.py`). This is not a
design choice that favors either architecture; it is what "remove the skip
connections and nothing else" necessarily implies.

Run via `RUN_PROFILE="backbone_comparison"` in either notebook, or:
```
python scripts/run_experiments.py --only exp_0_simple_cnn_shared_adapters_learned_balance,exp_0b_plain_deep18_no_skip_shared_adapters_learned_balance,exp_d_shared_adapters_learned_balance
python scripts/compare_backbones.py \
    --checkpoint simple_cnn=checkpoints/exp_0_..._best_balanced_score.pt \
    --checkpoint plain_deep18_no_skip=checkpoints/exp_0b_..._best_balanced_score.pt \
    --checkpoint custom_resnet18=checkpoints/exp_d_..._best_balanced_score.pt \
    --resnet-name custom_resnet18 --output-dir outputs/backbone_comparison
```

## Experiment 0c -- Custom ResNet-18, no zero-init residual (`exp_0c_custom_resnet18_no_zero_init_shared_adapters_learned_balance`)

The recommended architecture control: identical to Experiment D in every
respect (architecture, adapters, learned loss balancing, seeds, training
setup) except `model.backbone.zero_init_residual=false` -- each residual
branch's final BatchNorm keeps its default init (weight=1) instead of
being zeroed. This is orthogonal to "does the residual connection exist at
all" (Experiment 0b's question) and isolates a specific, common ResNet
training trick instead:

- **PlainDeep18NoSkip vs. Experiment 0c** tests residual shortcuts more
  cleanly than PlainDeep18NoSkip vs. Experiment D, because PlainDeep18NoSkip
  and Experiment 0c both use non-zero-init residual-branch normalization --
  Experiment D additionally differs by zero-initializing its residual
  branches, which is a second, confounding variable if PlainDeep18NoSkip is
  only ever compared against Experiment D.
- **Experiment D vs. Experiment 0c** isolates the effect of zero-initialized
  residual branches on their own, holding the presence of the residual
  connections themselves fixed.

Experiment D (the project's actual reported ResNet configuration, with
`zero_init_residual: true`) is never changed by adding this control.

```
python scripts/run_experiments.py --only exp_0c_custom_resnet18_no_zero_init_shared_adapters_learned_balance
```

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
experiment's) checkpoint.

```bash
make build-knn CHECKPOINT=checkpoints/<your_checkpoint>.pt
make evaluate CHECKPOINT=checkpoints/<your_checkpoint>.pt
```

`scripts/build_knn_index.py` extracts L2-normalized embeddings from the
training split (`src/evaluation/knn_baseline.py`, backed by
`sklearn.neighbors.NearestNeighbors`), fits a distance-weighted k-NN index
(default k=15) separately for the age and gender-label embedding spaces,
and saves it to `outputs/knn/knn_baseline.pkl`. `make evaluate` always
passes `--compare-knn` (`scripts/evaluate.py --checkpoint ... --compare-knn`),
which compares the parametric heads against k-NN prediction in the same
embedding space on age MAE/RMSE/coverage/width, gender-label selective
accuracy/abstention, and inference latency, writing
`{output_dir}/knn/{output_name}_parametric_vs_knn.csv` -- isolated per
checkpoint/experiment under that checkpoint's own configured output
directory, never a single shared global path, so evaluating multiple
checkpoints with `--compare-knn` never overwrites another experiment's
comparison table. The saved metrics JSON also records this exact path
under `knn_comparison_table_path`, so downstream code never has to guess
it. See `docs/evaluation.md` for the metric definitions used in that
comparison.

## Experiment F -- Pretrained vs. scratch (`exp_f_pretrained_vs_scratch`, optional)

Compares a backbone initialized from this repository's own SimCLR-style
self-supervised pretraining (`scripts/pretrain.py`) against the same
architecture trained from scratch. Skipped automatically (with a logged
message) if no pretrained checkpoint exists -- this experiment is opt-in
because self-supervised pretraining is comparatively compute-hungry (see
`docs/reproducibility.md`).

## What "success" would look like (to be judged only from real results)

- **D vs. 0 (efficiency/accuracy trade-off, not a residual-connections
  ablation)**: if the larger architecture provides real value, Experiment D
  (Custom ResNet-18) should show a meaningfully lower age MAE and/or
  higher gender-label accuracy than Experiment 0 (plain CNN) at a
  comparable or modestly higher parameter/latency cost -- not just a
  cheaper model that happens to be worse. See the auto-generated "Backbone
  Comparison (SimpleCNN / PlainDeep18NoSkip / Custom ResNet-18)" section of
  `docs/architecture_analysis_generated.md` for the actual numbers and a
  factual (non-causal) one-sentence summary. This comparison alone does
  **not** establish that residual connections specifically are what helps
  (see D vs. 0b below).
- **D vs. 0b (the actual residual-connections ablation)**: `scripts/compare_backbones.py`'s
  "Is Additional Residual Complexity Justified?" section reports this
  honestly and conditionally -- it credits ResNet only when a paired
  bootstrap confidence interval for the AURC (area under the risk-coverage
  curve, gender or age selective prediction) excludes zero in ResNet's
  favor, and explicitly states the compact/plain alternative is preferred
  when results are tied or favor it. A single-seed difference, even if
  numerically in ResNet's favor, is never reported as decisive; see the
  mean +/- std table across >= 3 seeds for stability evidence first.
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
