# Architecture Analysis (Methodology)

This document describes *how* this repository analyzes the shared-backbone
/ adapter / loss-balancing architecture questions, and how to read the
generated numbers. It is a fixed methods reference; the actual numbers for
your run live in `docs/architecture_analysis_generated.md` (produced by
`make architecture-report`, backed by `src/evaluation/reports.py`) and in
`outputs/architecture_analysis/`. See `docs/experiment_plan.md` for the
per-experiment hypotheses this analysis is meant to test.

## 1. Parameter counts

`MultiTaskFaceModel.parameter_breakdown()` (`src/models/multitask_model.py`)
splits parameters into `backbone`, `adapters`, `heads`, and (if learned
loss balancing is enabled) `log_variance`. For Experiment A (separate
backbones) `backbone` is the sum of both independent backbones' parameters.
This lets you directly compare, e.g., Experiment A's 2x backbone cost
against Experiment C/D's single backbone + two small adapters.

## 2. Training time / inference latency / GPU memory

`src/training/trainer.py` records wall-clock epoch time per stage
(`outputs/metrics/<experiment>_timing.json`). `scripts/evaluate.py` and
`scripts/build_knn_index.py` record per-image inference latency for both
the parametric model and the k-NN baseline. GPU memory, when running on
CUDA, can be read from `torch.cuda.max_memory_allocated()` around a
training or inference call if you need it for your own reporting -- it is
not persisted by default since this repository's default target hardware
is not assumed to have a GPU.

## 3. Task performance

Standard metrics (`src/evaluation/metrics.py`): age MAE/RMSE/R2, q10-q90
interval coverage and width (before/after conformal calibration, see
`docs/reproducibility.md` and `src/evaluation/calibration.py`), gender-label
accuracy (computed only over non-abstained predictions), abstention rate,
and confidence statistics.

## 4. Gradient interference (task-gradient cosine similarity)

For shared-backbone architectures (Experiments B/C/D; not defined for
Experiment A's independent backbones), `src/evaluation/architecture_analysis.py:compute_gradient_cosine_similarity`
does, per sampled batch:

1. Forward pass once.
2. Backward the age (pinball) loss with `retain_graph=True`; snapshot the
   gradient of every shared-backbone parameter.
3. Zero gradients; backward the gender (cross-entropy) loss; snapshot
   again.
4. Cosine similarity between the two flattened gradient vectors.

**Interpretation:**
- **Positive** mean cosine similarity: the two tasks pull shared weights
  in aligned directions -- evidence the shared representation is not (on
  average) fighting itself.
- **Negative**: the tasks pull in conflicting directions for at least part
  of training -- a mechanistic signal for "negative transfer", motivating
  adapters or learned loss balancing.
- **Near zero**: a weak/inconsistent relationship; not strong evidence
  either way.

This is measured for shared-backbone runs with and without adapters so the
report can state whether adapters change the *effective* conflict at the
backbone (adapters change what reaches the backbone via backprop, since the
adapter sits between the shared feature and the loss).

## 5. Representation similarity (linear CKA)

`src/evaluation/architecture_analysis.py:linear_cka` implements linear
Centered Kernel Alignment (Kornblith et al., 2019) between the shared
embedding `z` and each adapter's output. CKA is invariant to orthogonal
transformation and isotropic scaling, so it measures representational
similarity independent of arbitrary rotations a network might learn.

**Interpretation:**
- CKA close to 1: the adapter barely changes the representation for that
  task (little specialization, or the task doesn't need much).
- Lower CKA: the adapter meaningfully transforms the representation
  (more specialization). This is descriptive -- it does not by itself
  indicate whether that specialization is *helpful*; read it alongside the
  task performance tables.

## 6. Representation visualization (PCA / t-SNE)

`reduce_embeddings` projects shared embeddings to 2D via PCA or t-SNE for
visualization only, colored by age bucket (where age labels exist) and,
separately, by dataset gender label (where labels exist). These plots are
descriptive/exploratory -- proximity or separation in a 2D projection does
not establish a causal claim about what the network is "using" to make a
prediction.

## 7. Robustness

See `configs/robustness.yaml` and `src/evaluation/robustness.py`: eight
deterministic corruption types (blur, noise, low-resolution, JPEG
compression, brightness shifts, partial occlusion, partial crop) at
multiple severities, evaluated with a fixed seed. Reported per
corruption/severity: age MAE, interval coverage/width, gender accuracy,
abstention rate, and (if a k-NN index exists) the same metrics for the
non-parametric baseline.

## 8. Grad-CAM ("model attention visualization")

Manually implemented (`src/evaluation/gradcam.py`) -- no external Grad-CAM
library. Separate heatmaps for the age decision (backprop from q50) and
the gender-label decision (backprop from the selected class logit) at
the last residual stage (`layer4` by default). **This is a gradient-weighted
activation visualization. It is not proof of causality, and it does not
explain the model's "reasoning" in any human sense** -- treat it as a
diagnostic aid, not an explanation.

## Reading the generated report

`docs/architecture_analysis_generated.md` fills in each section above with
real numbers from your `outputs/` directory, or an explicit "not yet
generated" placeholder (with the command to produce it) if you haven't run
that stage yet. It never contains fabricated numbers.
