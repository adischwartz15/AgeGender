# Submission README

## Project Overview

**MT-AGNet**: Multi-Task Age and Gender Network — a research project
exploring multi-task learning for joint face age estimation and dataset
gender-label prediction.

**Repository**: https://github.com/adischwartz15/AgeGender

## What This Project Demonstrates

1. **Custom Neural Architecture**: Hand-written ResNet-18 backbone with no
   external pretrained weights or torchvision dependency
2. **Multi-Task Learning**: Shared backbone with task-specific bottleneck
   adapters and learned uncertainty loss balancing
3. **Uncertainty Quantification**: Quantile regression (q10/q50/q90) with
   split conformal calibration for coverage guarantees
4. **Controlled Ablation**: 9 experiments isolating specific architectural
   choices (residual connections, adapters, loss balancing, pretraining)
5. **Ethical Awareness**: Confidence-based abstention, identity disclaimers,
   demographic coverage documentation
6. **CLI-Driven Pipeline**: Config-driven CLI scripts + Makefile automation
   for the full train/calibrate/evaluate/robustness/Grad-CAM workflow

## Repository Structure

```
MT-AGNet/
├── src/                     # Source code
│   ├── models/              # ResNet-18, adapters, heads, backbone factory
│   ├── losses/              # Pinball loss, multi-task loss balancing
│   ├── training/            # Training loop, stages, checkpointing
│   ├── evaluation/          # Metrics, calibration, Grad-CAM, kNN, selective
│   ├── inference/           # Face detection, quality checks
│   ├── data/                # Dataset, transforms, splits, metadata
│   └── utils/               # Config, logging, seeds, visualization
├── configs/                 # YAML configuration files
├── scripts/                 # CLI entry points (train, evaluate, etc.)
├── tests/                   # pytest tests
├── docs/                    # Technical documentation (15+ files)
├── notebooks/               # Kaggle/Colab training notebooks
├── submission/              # Defense materials (this directory)
├── Makefile                 # Automation targets
├── requirements.txt         # Python dependencies
└── pyproject.toml           # Package configuration
```

## Key Documents

| Document | Purpose |
|---|---|
| `README.md` | Project overview and quick start |
| `docs/architecture_analysis.md` | Module-by-module analysis |
| `docs/experiment_plan.md` | Pre-registered ablation protocol |
| `docs/results.md` | Committed results from one real run |
| `docs/evaluation.md` | Metric definitions |
| `docs/model_card.md` | Model capabilities and limitations |
| `docs/data_card.md` | Dataset documentation |
| `docs/calibration.md` | Conformal calibration methodology |
| `docs/code_walkthrough.md` | End-to-end flow traces |
| `docs/code_learning_priority.md` | File reading order for reviewers |
| `docs/submission_readiness_audit.md` | Full readiness audit |
| `docs/scientific_validity_audit.md` | Claim-by-claim evidence |
| `docs/fresh_clone_validation.md` | Fresh clone reproduction steps |

## Submission Materials

| File | Purpose |
|---|---|
| `submission/SUBMISSION_README.md` | This file |
| `submission/DEFENSE_CHEATSHEET.md` | 1-page defense reference |
| `submission/DEFENSE_QUESTION_BANK.md` | 50+ questions with answers |

## Verification

```bash
git clone https://github.com/adischwartz15/AgeGender.git
cd AgeGender
make install
make test      # all tests pass
make lint      # Clean
```
