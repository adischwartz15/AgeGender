# Results (One Real Training Run)

Real numbers from an actual training run on UTKFace (via the Kaggle API,
default configs unless noted) -- not fabricated placeholders. This is a
point-in-time snapshot committed to the repository; it is **not**
regenerated automatically (unlike `docs/architecture_analysis_generated.md`,
which is gitignored and produced fresh by `make architecture-report`).

Reproduce with `make experiments`, `make build-knn CHECKPOINT=...`,
`make evaluate CHECKPOINT=...` (always includes the k-NN comparison),
`make robustness CHECKPOINT=...`, and `make architecture-report
CHECKPOINT=...`; see `docs/architecture_analysis_generated.md` for the
full, regenerated report after you run the pipeline yourself, and
`docs/architecture_analysis.md` for the methodology behind every number
below.

**Scope.** All numbers below come from one checkpoint
(`exp_d_shared_adapters_learned_balance`, the shared-backbone + adapters +
learned-loss-balancing architecture -- see `docs/experiment_plan.md`),
one dataset (UTKFace), and one train/validation/calibration/test split.
They are not a claim about performance on any other dataset, population,
camera, or checkpoint, and none of them is a multi-seed mean (see
`docs/final_evaluation_protocol.md` for the pre-registered three-seed
protocol used for the project's actual reported comparisons). See
`docs/data_card.md` and `docs/model_card.md` for the demographic-coverage
and generalization caveats that apply to every table here.

## Architecture parameter comparison (Experiments A-D)

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

## Parametric model vs. k-NN baseline (shared-backbone + adapters model)

| Metric | Parametric | k-NN (k=15) |
|---|---|---|
| Age MAE | 5.71 | 5.79 |
| Age RMSE | 8.32 | 8.53 |
| q10-q90 interval coverage (raw, uncalibrated) | 0.79 | 0.91 |
| Mean interval width | 16.79 | 26.88 |
| Gender-label selective accuracy | 0.970 | 0.966 |
| Abstention rate | 0.192 | 0.179 |
| Latency per image (ms) | 1.8 | 2.0 |

The q10-q90 interval is nominally an 80% interval (`calibration.alpha:
0.10` in `configs/training.yaml`) before conformal calibration is applied
-- neither row above is calibrated, so 0.79/0.91 are raw empirical
coverage, not a calibration guarantee. Gender-label accuracy is
**selective accuracy**: computed only over non-abstained predictions (see
`docs/evaluation.md` for the distinction from coverage and effective
accuracy). The k-NN baseline is competitive on gender-label accuracy and
reaches *higher* raw interval coverage than the (uncalibrated) parametric
model, at the cost of much wider intervals -- consistent with a
non-parametric method being more conservative rather than more precise
here.

## Gradient interference and representation similarity

Measured on the shared-backbone + adapters model (30 sampled batches; see
`docs/architecture_analysis.md`, sections 4-5, for full methodology):

- Mean task-gradient cosine similarity: **+0.08** (std 0.33) -- weakly
  positive, i.e. the age and gender-label gradients are not strongly in
  conflict on this dataset/split, with meaningful batch-to-batch variance.
- Linear CKA: shared-vs-age-adapter **0.79**, shared-vs-gender-adapter
  **0.90**, age-vs-gender-adapter **0.59** -- the gender adapter moves the
  shared representation less than the age adapter does, and the two
  adapters diverge from each other more than either diverges from the
  shared embedding.

## Robustness (deterministic corruptions, severity 1 of 3)

See `docs/robustness.md` for how to run this evaluation and
`docs/architecture_analysis.md` (section 7) / `docs/final_evaluation_protocol.md`
for the full corruption/severity definitions.

| Condition | Age MAE | Gender-label selective accuracy |
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
damaging conditions for age estimation in this run; gender-label selective
accuracy degrades more gracefully except under occlusion. This table is
severity 1 of 3 only -- see `docs/robustness.md` for the full
severity range and `docs/final_evaluation_protocol.md` for the exact
per-severity corruption parameters.

## Results depend on your data

Every number on this page -- age MAE, gender-label accuracy, interval
coverage, robustness curves, gradient interference, CKA -- is a property
of **the specific dataset, labels, split, and evaluation design used for
this run**, not a universal statement about the underlying task.
Different datasets have different demographic coverage, label quality,
and image conditions; do not extrapolate these results to populations,
cameras, or use cases outside the evaluation data. See
`docs/data_card.md` and `docs/model_card.md`.
