# Code Learning Priority Guide

Read order for a new reviewer to understand the project fastest, with the
most critical files first.

## Priority 1: Architecture (read first — takes ~30 min)

These four files are the heart of the project:

1. **`src/models/custom_resnet.py`** — The main backbone. Read this to
   understand BasicBlock, skip connections, Kaiming init, zero-init residual.
2. **`src/models/multitask_model.py`** — How the backbone, adapters, and
   heads connect. `build_multitask_model()` is the model factory.
3. **`src/models/adapters.py`** — 43 lines. Residual bottleneck adapter
   with zero-initialized up projection.
4. **`src/models/heads.py`** — Age quantile head (sigmoid + softplus
   guarantees), gender classification head.

## Priority 2: Loss Functions (read next — takes ~15 min)

5. **`src/losses/quantile_loss.py`** — Pinball loss for the three quantiles.
6. **`src/losses/multitask_loss.py`** — Fixed vs. learned uncertainty
   weighting, mask-aware aggregation.

## Priority 3: Training Loop (takes ~20 min)

7. **`src/training/trainer.py`** — The training loop. Focus on `_run_batches`,
   `_train_epoch`, mixed precision handling, gradient clipping.
8. **`src/training/stages.py`** — 75 lines. Progressive freeze/unfreeze
   stage planning.
9. **`src/training/checkpointing.py`** — Best-checkpoint selection logic.

## Priority 4: Evaluation & Calibration (takes ~20 min)

10. **`src/evaluation/metrics.py`** — All metric definitions. Read alongside
    `docs/evaluation.md` which defines the terminology.
11. **`src/evaluation/calibration.py`** — Split conformal calibration with
    provenance validation.
12. **`src/evaluation/gradcam.py`** — Manual Grad-CAM implementation.

## Priority 5: Inference Pipeline (takes ~15 min)

13. **`src/inference/predictor.py`** — Full inference pipeline: quality →
    face detection → transform → model → calibration → response.
14. **`src/inference/face_detection.py`** — Multi-pass Haar cascade with
    eye validation.

## Priority 6: Configuration & Data (takes ~15 min)

15. **`configs/model.yaml`** — Backbone, adapter, head configs.
16. **`configs/experiments.yaml`** — Ablation suite definitions.
17. **`src/data/dataset.py`** — 81 lines. Dataset class with mask support.
18. **`src/data/transforms.py`** — Manual transforms (no torchvision).

## Priority 7: Documentation (skim — takes ~15 min)

19. **`docs/architecture_analysis.md`** — What each experiment tests.
20. **`docs/experiment_plan.md`** — Pre-registered protocol.
21. **`docs/results.md`** — Committed results from one real run.
22. **`docs/evaluation.md`** — Metric definitions (selective acc vs. effective acc).

## Key Design Decisions to Understand

| Decision | Why | Where |
|---|---|---|
| No torchvision | Demonstrate understanding | `transforms.py`, `custom_resnet.py` |
| Masks, not dropping | Maximize training data | `dataset.py`, `multitask_loss.py` |
| Zero-init adapter up | Identity at init | `adapters.py` L34-35 |
| softplus for intervals | Guarantee q10 ≤ q50 ≤ q90 | `heads.py` L44-48 |
| Raw quantiles for loss | Prevent gradient death at clamp | `heads.py` L50-53 |
| Separate calibration split | No data leakage | `data.yaml`, `calibration.py` |
| SHA-256 provenance | Cross-model contamination guard | `calibration.py` L100-105 |
| Identity disclaimer | Ethical: labels ≠ identity | `schemas.py` L7-11 |
| Decline on no face | Domain mismatch prevention | `predictor.py` L131-146 |
| PlainDeep18 as control | Clean residual ablation | `plain_deep18_no_skip.py` |
