# Scientific Validity Audit

**Date**: 2026-07-10  
**Source of truth**: Committed code + `docs/results.md` (one real run)

## Research Claim Traceability

| # | Research Claim | Supporting Experiment | Metric | Artifact | Seeds | Strength of Evidence | Verdict |
|---|---|---|---|---|---|---|---|
| 1 | Sharing a backbone reduces parameter count | A vs. B/C/D | Parameter count | `results.md` table | N/A (structural) | Deterministic; verified from code | **Supported** |
| 2 | Sharing a backbone improves predictive performance | A vs. D | MAE, selective acc | Not committed (cross-experiment eval not in `results.md`) | 1 | No committed comparison | **Cannot verify from committed artifacts** |
| 3 | Task-specific adapters improve over naive sharing | B vs. C | MAE, selective acc | Not committed | 1 | No committed comparison | **Cannot verify from committed artifacts** |
| 4 | Learned loss balancing improves over fixed weights | C vs. D | MAE, selective acc | Not committed | 1 | No committed comparison | **Cannot verify from committed artifacts** |
| 5 | Residual architecture improves over compact CNN | Exp 0 vs. D | MAE, selective acc | Not committed | 1 | No committed comparison | **Cannot verify from committed artifacts** |
| 6 | Residual connections help when depth/width controlled | Exp 0b vs. D | MAE, selective acc | Not committed | 1 | No committed comparison | **Cannot verify from committed artifacts** |
| 7 | Parametric model is better than k-NN | Exp E | MAE, selective acc | `results.md` table | 1 | Single run, both models on same test set | **Partially supported** |
| 8 | Raw age intervals are well calibrated | Exp D | Raw coverage | `results.md`: 0.79 | 1 | 0.79 vs 0.80 nominal = slight undercoverage | **Partially supported** |
| 9 | Conformal calibration improves interval coverage | Calibrate script | Calibrated coverage | Not in `results.md` | 1 | Pipeline exists but calibrated numbers not committed | **Cannot verify from committed artifacts** |
| 10 | Confidence-based abstention improves selective accuracy | Exp D | Selective acc vs. raw acc | `results.md`: sel_acc=0.970, abstention=0.192 | 1 | Single run, trade-off shown | **Partially supported** |
| 11 | Robustness differs across corruption types | Exp D | MAE, selective acc | `results.md` corruption table | 1 | Clear ordering of corruption impact | **Supported** (for this checkpoint) |
| 12 | Task gradients are aligned or conflicting | Exp D | Cosine similarity | `results.md`: mean=+0.08, std=0.33 | 1 | Weakly positive, high variance | **Partially supported** |
| 13 | Adapters learn distinct representations | Exp D | Linear CKA | `results.md`: age-gender CKA=0.59 | 1 | Moderate divergence observed | **Partially supported** |
| 14 | Results are stable across seeds | 3-seed protocol | Seed variation | Not committed | 0 committed | No multi-seed results in repo | **Not yet supported** |
| 15 | Added architectural complexity is statistically justified | Cross-exp bootstrap CI | AURC CI | Not committed | 1 | No statistical test results committed | **Not yet supported** |

## Detailed Notes

### Claim 1: Parameter count reduction

This is a structural/mathematical claim verified directly from code:
- **Separate (A)**: 2 × 11,176,512 backbone params = 22,353,024
- **Shared (D)**: 1 × 11,176,512 backbone params = 11,176,512
- Adapters add 263,424 (2.4% of shared backbone)
- **Verdict**: Definitively supported. Sharing halves backbone parameters by construction.
- **Important caveat**: Parameter count reduction does not prove performance improvement.

### Claim 7: Parametric vs. k-NN

From `results.md`:
- Parametric MAE 5.71 vs. k-NN MAE 5.79 (slight parametric advantage)
- Parametric selective accuracy 0.970 vs. k-NN 0.966
- k-NN has wider intervals (26.88 vs 16.79) but higher raw coverage (0.91 vs 0.79)
- This is one run, one seed — the difference may not be statistically significant.

### Claim 8: Raw interval calibration

- Nominal target: 80% (alpha=0.10)
- Actual raw coverage: 79% — close but slightly under
- This is expected: raw quantile regression does not guarantee coverage
- Conformal calibration exists to fix this, but calibrated results not committed

### Claims 2-6: Cross-experiment comparisons

The experiment framework (Experiments 0/0b/0c/A-F) is well-designed and the
code to run all experiments exists (`scripts/run_experiments.py`), but
`docs/results.md` only commits results for **Experiment D** and the
k-NN comparison. Cross-experiment accuracy comparisons are not committed.

**Recommendation for defense**: Be transparent that the repository provides
the tooling for all ablations but committed results focus on the main
architecture. Running `make experiments` + `make evaluate` per checkpoint
would produce the full comparison.

### Claim 14: Multi-seed stability

The 3-seed protocol is documented in `docs/final_evaluation_protocol.md`
and implemented in `scripts/run_seeds.py`, but no multi-seed results are
committed. The README correctly states "one seed, one dataset split."

### Metric Terminology Verification

| Term used in docs | Definition verified in code | Correct |
|---|---|---|
| Selective accuracy | `gender_accuracy()` with `abstain_mask` in `metrics.py` | ✅ |
| Effective accuracy | `gender_effective_accuracy()` — correct/(all including abstained) | ✅ |
| Coverage | `gender_coverage()` = 1 - abstention_rate | ✅ |
| Abstention rate | `abstention_rate()` = mean(abstain_mask) | ✅ |
| Interval coverage | `interval_coverage()` = mean(q_low ≤ y ≤ q_high) | ✅ |
| MAE | `age_mae()` = mean(|y_true - y_pred|) | ✅ |
| RMSE | `age_rmse()` = sqrt(mean((y_true - y_pred)²)) | ✅ |

### Causal vs. Observational Claims

All claims in `docs/results.md` and the README are correctly observational:
- "These numbers describe one checkpoint" ✅
- "not a claim about performance on any other dataset" ✅
- "not a multi-seed mean" ✅
- Grad-CAM called "model attention visualization," not causal ✅
- PlainDeep18NoSkip described as "isolating" residual connections, which is
  appropriately controlled language for a single-variable ablation ✅

### Statistical Evidence

No statistical tests (bootstrap CIs, significance tests) are committed in
`docs/results.md`. The infrastructure exists in `src/evaluation/selective.py`
(paired bootstrap CI) and `src/evaluation/backbone_comparison.py`, but
results would require running the full pipeline. This is honestly
acknowledged in the repository.
