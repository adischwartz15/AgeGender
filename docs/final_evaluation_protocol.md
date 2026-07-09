# Final Evaluation Protocol (Pre-Registered)

This document locks the protocol for the project's one full, reported run
**before** that run's results are observed. Its purpose is to prevent
post-hoc "researcher degrees of freedom" -- picking a favorable metric,
corruption, threshold, or seed subset only after seeing which choice makes
a preferred architecture look best. Everything below is decided in
advance and must not change once full results exist (see "No post-hoc
changes" at the end).

This protocol governs the **final, reported** run only. Exploratory work
during development (trying configs, debugging, tuning hyperparameters) is
unrestricted and expected -- it just isn't what gets reported as "the"
result.

## Models

All seven configured experiments in `configs/experiments.yaml`, run in
`run_order`:

| Experiment | Backbone | Role |
|---|---|---|
| `exp_0_simple_cnn_shared_adapters_learned_balance` | SimpleCNN | Efficiency/accuracy trade-off baseline vs. ResNet (**not** a residual-connection ablation -- differs in depth/width too) |
| `exp_0b_plain_deep18_no_skip_shared_adapters_learned_balance` | PlainDeep18NoSkip | The residual-connection ablation vs. ResNet (depth/width held fixed) |
| `exp_0c_custom_resnet18_no_zero_init_shared_adapters_learned_balance` | Custom ResNet-18, `zero_init_residual=false` | Zero-init-residual ablation; also lets PlainDeep18NoSkip be compared against a same-init-convention ResNet variant |
| `exp_a_separate` | Custom ResNet-18 x2 (independent) | No-sharing baseline |
| `exp_b_shared_no_adapters` | Custom ResNet-18 (shared) | Naive sharing, fixed loss weights |
| `exp_c_shared_adapters` | Custom ResNet-18 (shared) | + task adapters, fixed loss weights |
| `exp_d_shared_adapters_learned_balance` | Custom ResNet-18 (shared) | + adapters + learned loss balancing -- **the project's main/reported backbone** |

`exp_e_parametric_vs_knn` (k-NN over Experiment D's embeddings) and
`exp_f_pretrained_vs_scratch` (optional, requires a pretrained checkpoint)
run as documented in `docs/experiment_plan.md`; they do not train a new
architecture and are not part of the backbone-comparison decision below.

## Seeds

**42, 123, 2026** -- three seeds for every trained experiment
(`python scripts/run_seeds.py --experiment <name> --seeds 42,123,2026`),
giving a real mean +/- std per experiment
(`src/evaluation/comparison.py:aggregate_seed_metrics`) rather than a
single-run point estimate. Seed 42 is also each experiment's primary seed
for the isolated `experiments/<experiment>/seed_42/` artifacts used
directly in the backbone-comparison and robustness sections.

## Primary metrics

- **Age:** MAE (`age_mae`) on the test split, point estimate q50 vs. true age.
- **Gender label:** selective accuracy (`gender_accuracy`, denominator excludes abstentions) at the fixed confidence threshold below.

These two are what "is Experiment X better than Experiment Y" is judged on
first. Both are reported per experiment as mean +/- std across the three
seeds above (`scripts/generate_final_report.py`).

## Secondary metrics

- Age RMSE (`age_rmse`), raw and conformal-calibrated interval coverage/width (`interval_coverage`, `mean_interval_width`, and the `_calibrated` variants), per-age-bucket MAE/coverage/width (`age_metrics_by_bucket[_calibrated]`).
- Gender label: coverage (`gender_coverage`), abstention rate (`abstention_rate`), effective accuracy (`gender_effective_accuracy`), confidence statistics.
- Selective-risk AURC for both tasks (`src/evaluation/selective.py:compute_aurc`) and its paired bootstrap CI (`paired_bootstrap_aurc_diff_ci`) -- this is what the residual-complexity decision rule below is actually gated on, not the primary metrics' raw point difference.
- Parameter count, backbone parameter count, mean epoch time, inference latency per image (`latency_ms_per_image`) -- the cost side of every trade-off claim.

## Robustness conditions and severities

Exactly the corruptions and severities in `configs/robustness.yaml`, evaluated identically for every model (same seed, same deterministic per-sample corruption parameters):

| Corruption | Severities (param values) |
|---|---|
| `gaussian_blur` (sigma) | 1: 0.8, 2: 1.6, 3: 2.6 |
| `gaussian_noise` (std) | 1: 0.03, 2: 0.08, 3: 0.15 |
| `low_resolution` (scale_factor) | 1: 0.5, 2: 0.3, 3: 0.15 |
| `jpeg_compression` (quality) | 1: 40, 2: 20, 3: 10 |
| `low_brightness` / `high_brightness` (factor) | 1/2/3 per `configs/robustness.yaml` |
| `low_contrast` / `high_contrast` (factor) | 1/2/3 per `configs/robustness.yaml` |
| `grayscale` (blend_factor) | 1: 0.4, 2: 0.7, 3: 1.0 |
| `partial_occlusion` (occlusion_fraction) | 1: 0.1, 2: 0.2, 3: 0.35 |
| `partial_crop` (crop_fraction) | 1: 0.1, 2: 0.2, 3: 0.35 |

Evaluated on the **full test split** (`scripts/run_robustness.py`, no
`--max-samples`) for the final reported run; `--max-samples` with
deterministic (age-bucket x gender-label) stratified sampling is for
faster iteration during development only, not the final numbers. The
fixed conformal offset from that checkpoint/seed's own calibration
artifact (never refit per corruption) is applied to every condition, so
both raw and calibrated coverage/width are reported at every severity.

## Fixed confidence thresholds

- **Gender-label abstention threshold:** 0.80 (`configs/model.yaml: model.gender_head.confidence_threshold`), identical across every model and every corruption condition.
- **Conformal miscoverage / target coverage:** `alpha=0.10` -> 90% target coverage (`configs/training.yaml: calibration.alpha`).
- **Selective-risk-coverage evaluation levels:** 0.80, 0.90, 0.95, 0.98 (`src/evaluation/backbone_comparison.py: COMMON_COVERAGE_LEVELS`).
- **Statistical significance:** two-sided 95% paired bootstrap CI (`alpha=0.05`, 1000 resamples, `src/evaluation/selective.py`), fixed seed for reproducibility of the bootstrap itself.

None of these thresholds are tuned per model or chosen after seeing results.

## Decision rule: is added residual complexity justified?

Computed mechanically by `src/evaluation/backbone_comparison.py:build_final_interpretation` (via `scripts/compare_backbones.py`), never hand-edited:

1. For the residual-connection ablation (PlainDeep18NoSkip vs. Custom ResNet-18) and, if `exp_0c` was run, the zero-init ablation (Custom ResNet-18 vs. Custom ResNet-18 no-zero-init): compute the paired bootstrap CI on the **AURC statistic itself** (`paired_bootstrap_aurc_diff_ci`) for both gender and age selective-risk curves.
2. ResNet's added complexity is credited **only** if at least one of those AURC CIs excludes zero in ResNet's favor (i.e., ResNet's AURC is significantly lower).
3. A CI computed only at one fixed coverage level (`paired_bootstrap_risk_diff_ci`) is **never** sufficient evidence for an AURC-level claim -- only `pairwise_bootstrap_aurc` counts.
4. A single-seed numeric advantage that doesn't clear step 2 is reported as "not statistically supported," and the compact/plain alternative is stated as preferred for this dataset and training setup.
5. The efficiency/accuracy trade-off comparison (SimpleCNN vs. ResNet) is reported separately and is **never** used as evidence about residual connections specifically (see `docs/experiment_plan.md`).

## No post-hoc changes

Once the full run's results exist, the following are frozen and must not be changed retroactively to make any particular model look better:

- Which metrics are "primary" vs. "secondary" (see above).
- Which corruptions/severities are evaluated, and the max-samples policy (full split for final numbers).
- The confidence threshold, calibration alpha, coverage levels, and bootstrap settings above.
- The decision rule for crediting residual complexity.

If a real methodological bug is found after results exist (e.g. the kind of contamination/misalignment bugs this protocol's supporting code changes were written to catch), the fix and its effect on the numbers must be disclosed explicitly rather than silently re-run and replaced. Genuinely new analyses are welcome, but must be presented as exploratory/follow-up, not substituted for the pre-registered primary result.
