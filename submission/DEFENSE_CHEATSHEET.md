# Defense Cheatsheet (1 page)

## Architecture in 30 seconds
- **Backbone**: Custom ResNet-18 (hand-written, no torchvision, ~11M params)
- **Multi-task**: Shared backbone → task-specific bottleneck adapters (+263K params) → separate heads
- **Age**: Quantile regression (q10/q50/q90) via pinball loss, softplus guarantees ordering
- **Gender**: Binary classification with confidence-based abstention at 0.80 threshold
- **Loss**: Fixed or learned uncertainty weighting (Kendall et al. 2018)

## Key Numbers (one checkpoint, one dataset, one split)
| Metric | Value |
|---|---|
| Age MAE | 5.71 |
| Age RMSE | 8.32 |
| Raw q10-q90 coverage | 0.79 (nominal 0.80) |
| Gender selective accuracy | 0.970 |
| Abstention rate | 0.192 |
| Total parameters | 11.57M |
| Gradient cosine similarity | +0.08 (std 0.33) |
| Worst corruption (age MAE) | Gaussian noise: 14.82 |

## 9 Experiments at a Glance
| Exp | What it tests | Architecture |
|---|---|---|
| 0 | Backbone comparison (compact CNN) | SimpleCNN + adapters |
| 0b | Residual connection ablation | PlainDeep18NoSkip + adapters |
| 0c | Zero-init residual ablation | ResNet-18 (no zero-init) + adapters |
| A | Two separate models | 2× ResNet-18 |
| B | Naive sharing | 1× ResNet-18, no adapters |
| C | Adapters help? | 1× ResNet-18 + adapters, fixed weights |
| D | Learned weighting? | 1× ResNet-18 + adapters, learned weights |
| E | Parametric vs. kNN | D's embeddings + kNN |
| F | Pretraining helps? | SimCLR pretrained backbone |

## Critical Terminology
- "Dataset gender-label prediction" — NOT "gender identity detection"
- "Selective accuracy" — accuracy over non-abstained predictions only
- "Effective accuracy" — correct/all (includes abstentions in denominator)
- "Nominal 80% interval" — NOT "90% interval"
- "Model attention visualization" — NOT "explanation" (for Grad-CAM)
- "Raw" vs. "calibrated" — always specify which

## Top 5 Gotcha Questions
1. **"Why not just use torchvision ResNet?"** → Demonstrates understanding; no external pretrained weights.
2. **"How do you prevent data leakage?"** → 4-way split, SHA-256 provenance, calibration ≠ validation.
3. **"Is sharing always better?"** → Parameter reduction ≠ performance improvement. Cross-experiment eval needed.
4. **"Are your results statistically significant?"** → One seed. 3-seed protocol exists but not committed. Be honest.
5. **"What about bias?"** → Binary labels, limited demographics, no fairness audit. This is a limitation, not a claim.

## Quick File Reference
| Purpose | File |
|---|---|
| Model definition | `src/models/multitask_model.py` |
| Backbone | `src/models/custom_resnet.py` |
| Adapters | `src/models/adapters.py` |
| Heads | `src/models/heads.py` |
| Pinball loss | `src/losses/quantile_loss.py` |
| Multi-task loss | `src/losses/multitask_loss.py` |
| Training loop | `src/training/trainer.py` |
| Calibration | `src/evaluation/calibration.py` |
| Metrics | `src/evaluation/metrics.py` |
| Face detection | `src/inference/face_detection.py` |
| Results | `docs/results.md` |
| Experiment plan | `docs/experiment_plan.md` |
