# Model Card

## Overview

**Name:** face-multitask-research multi-task age / dataset gender-label model
**Type:** Convolutional multi-task regression + classification model
**Backbone:** Manually implemented ResNet-18 (`src/models/custom_resnet.py`), no pretrained ImageNet weights, no `torchvision.models`/`timm`/model-hub dependency anywhere in the codebase.
**Intended use:** Research and educational demonstration of multi-task learning, shared representations, adapters, uncertainty estimation, and explainability techniques on face-image datasets.

## Intended use and out-of-scope use

This model is intended **only** for:

- Studying multi-task learning dynamics (shared backbone vs. adapters vs. independent backbones).
- Studying loss-balancing strategies (fixed weights vs. learned homoscedastic uncertainty).
- Studying uncertainty quantification (quantile regression + conformal calibration) for a regression task.
- Studying non-parametric vs. parametric prediction in a learned embedding space.
- Educational demonstrations of Grad-CAM-style explainability.

This model **must not** be used for:

- Employment screening or hiring decisions.
- Policing, surveillance, or law enforcement.
- Identity verification or authentication.
- Medical diagnosis or health-related decisions.
- Admissions decisions (school, university, program).
- Insurance underwriting or pricing.
- Any other decision with material consequences for a real person.

## Outputs

- **Estimated age**: a point estimate (q50) plus a q10-q90 prediction interval, optionally conformal-calibrated.
- **Dataset gender-label prediction**: softmax probabilities over dataset-defined classes (default names `gender_label_0` / `gender_label_1`), or "Not sure" when the top probability is below the configured confidence threshold (default 0.80).
- **Model attention visualization** (Grad-CAM): separate heatmaps for the age and gender-label decisions. This is a gradient-weighted activation visualization, not proof of causality or an explanation of the model's "reasoning".

## Training data

Trained on a Kaggle-hosted, user-supplied dataset (default target: a UTKFace-style dataset). See `docs/data_card.md` for details, and note that **no dataset is bundled with this repository** -- you must supply your own via `scripts/download_kaggle_data.py` or a local copy, subject to that dataset's own license and terms.

## Ethical considerations

- "Gender-label" is a categorical field defined by the source dataset's authors/annotators, not a determination of a person's gender identity.
- Dataset labels may be binary, incomplete, self-reported, annotator-assigned, or otherwise limited; they do not necessarily reflect how any individual identifies.
- Age and gender-label distributions in any given dataset are rarely demographically balanced or globally representative; performance figures in `outputs/` describe behavior on the specific dataset/split used, not general human populations.
- Race/ethnicity metadata, where present in a dataset (e.g. UTKFace), is never used as a model input, target, or split criterion in this codebase.

## Known limitations

- Small custom ResNet-18 trained on a single dataset; expect a real generalization gap to other cameras, populations, lighting, and capture conditions.
- Conformal-calibrated intervals provide **marginal** (population-average) coverage guarantees, not per-individual or per-subgroup guarantees.
- Robustness figures in `outputs/robustness/` are limited to the corruption types/severities defined in `configs/robustness.yaml`; real-world degradations may differ.

## How to reproduce reported numbers

See `docs/reproducibility.md`. All numbers in `outputs/` and any generated report are produced by the scripts in `scripts/` against a specific checkpoint and split; no numbers in this repository are fabricated or hand-edited.
