# Supplementary Experiment: ImageNet-Pretrained VOLO-D1 (Face-Only)

Practical guide to the transfer-learning extension. For the research
question the *core* ablation suite answers, see
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

## How to run it

```bash
python scripts/run_transfer_learning.py --smoke                          # tiny, non-scientific CPU check
python scripts/run_transfer_learning.py --seeds 42,43,44                 # full run (needs a GPU in practice)
python scripts/run_transfer_learning.py --seeds 42,43,44 --evaluate-only # evaluate existing checkpoints only
python scripts/run_transfer_learning.py --only volo                      # skip the baseline row
python scripts/run_transfer_learning.py --only baseline                  # skip VOLO, reuse the baseline row only
```

The from-scratch baseline (`exp_d_shared_adapters_learned_balance` by
default, `--baseline-experiment` to change it) is **never retrained** here
-- its already-computed test-metrics JSON is reused if present, or its
existing checkpoint is evaluated (never retrained) if not.

## Output paths

```
checkpoints/transfer_learning/volo_d1_face_only_pretrained/seed_<seed>/...
results/transfer_learning/volo_d1_face_only_pretrained/seed_<seed>/...
results/transfer_learning/table_b.csv
results/transfer_learning/table_b_manifest.json   # git SHA, dependency versions, split fingerprint, seed count
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
never regenerated for this experiment), same age loss/gender loss/label
mapping/age clipping, same evaluation functions and postprocessing. Every
run saves: metrics JSON, predictions, config snapshot, split fingerprint
(SHA-256), training history, plots, seed, dependency versions
(Python/PyTorch/`timm`/CUDA), and the git commit SHA.

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
