# Comprehensive Backbone Comparison Suite

Practical guide to `scripts/compare_backbones.py`. For the methodology
behind each analysis (selective prediction, AURC, paired bootstrap,
tail-error analysis, the conditional "is residual complexity justified"
interpretation), see `docs/architecture_analysis.md` (section 9). For the
experiments being compared and why each pairing matters, see
`docs/experiment_plan.md`.

## What it does

Post-hoc analysis across two or more already-trained checkpoints --
**never retrains**, only re-runs inference (a single forward pass per
test-set image) against each checkpoint's own test split.

```bash
python scripts/compare_backbones.py \
    --checkpoint simple_cnn=experiments/exp_0_simple_cnn_shared_adapters_learned_balance/seed_42/checkpoints/exp_0_simple_cnn_shared_adapters_learned_balance_best_balanced_score.pt \
    --checkpoint plain_deep18_no_skip=experiments/exp_0b_plain_deep18_no_skip_shared_adapters_learned_balance/seed_42/checkpoints/exp_0b_plain_deep18_no_skip_shared_adapters_learned_balance_best_balanced_score.pt \
    --checkpoint custom_resnet18=experiments/exp_d_shared_adapters_learned_balance/seed_42/checkpoints/exp_d_shared_adapters_learned_balance_best_balanced_score.pt \
    --calibration-dir simple_cnn=experiments/exp_0_simple_cnn_shared_adapters_learned_balance/seed_42/calibration \
    --calibration-dir plain_deep18_no_skip=experiments/exp_0b_plain_deep18_no_skip_shared_adapters_learned_balance/seed_42/calibration \
    --calibration-dir custom_resnet18=experiments/exp_d_shared_adapters_learned_balance/seed_42/calibration \
    --robustness-csv simple_cnn=experiments/exp_0_simple_cnn_shared_adapters_learned_balance/seed_42/robustness/robustness_results.csv \
    --robustness-csv custom_resnet18=experiments/exp_d_shared_adapters_learned_balance/seed_42/robustness/robustness_results.csv \
    --resnet-name custom_resnet18 \
    --output-dir outputs/backbone_comparison
```

(Optionally add
`--checkpoint custom_resnet18_no_zero_init=experiments/exp_0c_.../seed_42/checkpoints/..._best_balanced_score.pt`
for the zero-init-residual ablation -- see `docs/final_evaluation_protocol.md`.)

A shorter, Makefile-driven form covers the common two-checkpoint case
(`CHECKPOINTS`/`RESNET_NAME` override the defaults shown by
`make -n compare-backbones`):

```bash
make compare-backbones CHECKPOINTS="simple_cnn=checkpoints/exp_0_..._best_balanced_score.pt custom_resnet18=checkpoints/exp_d_..._best_balanced_score.pt" RESNET_NAME=custom_resnet18
```

## Output files (`--output-dir`)

- `clean_test_summary.csv` -- age MAE/RMSE, median/p90/p95 absolute age
  error, error-tail rates (>5/>10/>15/>20 years), raw and calibrated
  interval coverage/width, gender-label **selective accuracy** (correct /
  accepted only), **coverage** (fraction answered, `1 - abstention_rate`),
  and **effective accuracy** (correct-and-accepted / *all* samples,
  denominator includes abstentions) -- plus parameter counts and latency.
  Selective accuracy can look excellent while effective accuracy is
  mediocre if a model abstains constantly; both numbers are reported so
  this can't be hidden. `plots/pareto_params_vs_mae.png` and
  `pareto_latency_vs_mae.png` visualize the trade-off.
- `gender_risk_at_coverage.csv`, `gender_aurc.json`,
  `gender_pairwise_bootstrap.json`, `gender_aurc_bootstrap.json`,
  `plots/gender_risk_coverage.png` -- gender selective-risk-vs-coverage
  curves (confidence = max class probability), AURC (lower is better),
  risk at 80/90/95/98% coverage with paired-bootstrap CIs for the
  ResNet-vs-other difference *at each coverage level*
  (`gender_pairwise_bootstrap.json`), and a **separate** paired-bootstrap
  CI on the **AURC summary statistic itself**
  (`gender_aurc_bootstrap.json`) -- only the latter is treated as
  sufficient evidence for a "ResNet has lower AURC" claim; a CI at one
  fixed coverage level is not. Models are always compared at the *same*
  coverage, never at independent arbitrary thresholds, and only after
  verifying both models share the identical, index-aligned test-sample
  IDs (not just equal sample counts).
- `age_selective_mae_at_coverage.csv`, `age_selective_aurc.json`,
  `age_pairwise_bootstrap.json`, `age_aurc_bootstrap.json`,
  `plots/age_risk_coverage_{mae,rmse}.png` -- the same analysis (including
  the AURC-vs-fixed-coverage CI distinction above) for age, using
  q90-q10 interval width as the confidence score (narrower = more
  confident).
- `age_bucket_mae.csv`, `age_error_percentiles.json`,
  `plots/age_error_cdf.png`, `plots/age_tail_error_rates.png` --
  per-age-bucket MAE (0-12/13-19/20-34/35-49/50-64/65+), the empirical
  CDF of absolute age error per model, and error-tail-rate bars --
  answers whether ResNet reduces catastrophic errors even when average
  MAE is similar.
- `robustness_degradation_<model>.csv`, `robustness_diff_table.csv`
  (when `--robustness-csv` is given per model) -- delta and relative (%)
  degradation from the clean baseline per corruption/severity, plus
  **every pairwise** model-vs-model difference (not just the first two by
  argument order) -- with three models this includes SimpleCNN-vs-ResNet,
  PlainDeep18NoSkip-vs-ResNet, and SimpleCNN-vs-PlainDeep18NoSkip all at
  once.
- `final_interpretation.md` -- "Is Additional Residual Complexity
  Justified?": credits ResNet only when a paired-bootstrap CI excludes
  zero in its favor, and explicitly states the compact/plain alternative
  is preferred otherwise. Never treats a single-seed difference as
  decisive.

Both notebooks (`notebooks/train_evaluate_*.ipynb`) run this automatically
when `RUN_PROFILE = "backbone_comparison"` -- see `docs/notebooks.md`.
