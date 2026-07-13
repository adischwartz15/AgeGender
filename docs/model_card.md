# Model Card

## Overview

**Name:** face-multitask-research multi-task age / dataset gender-label model
**Type:** Convolutional multi-task regression + classification model
**Backbone:** Manually implemented ResNet-18 (`src/models/custom_resnet.py`), no pretrained ImageNet weights, no `torchvision.models`/`timm`/model-hub dependency in the deployed API or any core training/evaluation path (Experiments 0/0b/0c, A-F, Table A). Three controlled ablation baselines also exist, none ever the project's main backbone nor used by the deployed API: a non-residual `SimpleCNNBackbone` (`src/models/simple_cnn.py`, Experiment 0, differs in depth/width *and* residual connections -- an efficiency/accuracy trade-off comparison, not a residual-connection ablation), `PlainDeep18NoSkip` (`src/models/plain_deep18_no_skip.py`, Experiment 0b, depth/width-matched to Custom ResNet-18 with only the residual additions removed -- the actual residual-connections ablation), and Custom ResNet-18 with `zero_init_residual=false` (Experiment 0c, isolates the effect of zero-initialized residual branches specifically, and lets Experiment 0b be compared against a same-init-convention ResNet variant). See `docs/experiment_plan.md`, `docs/final_evaluation_protocol.md`, and `scripts/compare_backbones.py`.

**Not the deployed model, but present in the codebase as optional supplementary experiments (Table B, never Table A, never the deployed API):** an ImageNet-pretrained VOLO-D1 (`timm`) and ImageNet-pretrained ResNet-18/50 (`torchvision`) -- both require their own optional extra dependency (`requirements-transfer.txt`), never installed or imported by the core project. See [docs/transfer_learning.md](transfer_learning.md).
**Intended use:** Research and educational demonstration of multi-task learning, shared representations, adapters, uncertainty estimation, and explainability techniques on face-image datasets.

## Intended use and non-intended use

This model is intended **only** for:

- Studying multi-task learning dynamics (shared backbone vs. adapters vs. independent backbones).
- Studying loss-balancing strategies (fixed weights vs. learned homoscedastic uncertainty).
- Studying uncertainty quantification (quantile regression + conformal calibration) for a regression task.
- Studying non-parametric vs. parametric prediction in a learned embedding space.
- Educational demonstrations of Grad-CAM-style explainability.
- Classroom/portfolio demonstration via the synthetic images in `data/demo_images/` or a presenter's own consented photo (see "Demo mode" in the root `README.md`).

This model **must not** be used for:

- Employment screening or hiring decisions.
- Policing, surveillance, or law enforcement.
- Identity verification or authentication, or any other biometric identification purpose.
- Medical diagnosis or health-related decisions.
- Admissions decisions (school, university, program).
- Insurance underwriting or pricing.
- Any other decision with material consequences for a real person.

Nothing about this model's design, training data, or evaluation supports these uses; using it for them would be a misuse of a research/education artifact, not a supported deployment.

## Outputs and label definitions

- **Estimated age**: a point estimate (q50) plus a q10-q90 prediction interval, optionally conformal-calibrated. This is a regression estimate against a training dataset's recorded age labels, not a biological or legal age determination.
- **Dataset gender-label prediction**: softmax probabilities over dataset-defined classes (default names `gender_label_0` / `gender_label_1`, configurable via `GENDER_LABEL_0`/`GENDER_LABEL_1` or `configs/api.yaml`'s `gender_label_overrides`), or **"Not sure"** when the top probability is below the configured confidence threshold (default 0.80; see "Abstention behavior" below). "Gender-label" is a categorical field defined by the source dataset's authors/annotators (e.g. UTKFace's binary 0/1 convention) -- it is **not** a determination of a person's gender identity, and the model has no concept of gender identity at all.
- **Model attention visualization** (Grad-CAM): separate heatmaps for the age and gender-label decisions. This is a gradient-weighted activation visualization, not proof of causality or an explanation of the model's "reasoning".

## Uncertainty interpretation

Age predictions are quantile estimates (q10/q50/q90) from a single quantile-regression head, not an ensemble or a Bayesian posterior. Two levels of intervals exist:

- **Raw q10-q90 interval**: whatever the trained quantile head outputs directly. Its empirical coverage on held-out data is not guaranteed to match any particular target (e.g. it may cover the true age less than 80% of the time even though it's a "q10-q90" interval).
- **Conformal-calibrated interval** (`src/evaluation/calibration.py`, split-conformal/CQR): a single scalar offset, fit once on the dedicated **calibration** split (never the validation or test split -- see `docs/data_card.md` and `docs/reproducibility.md` for the train/validation/calibration/test protocol), is added/subtracted from the raw interval. This gives a **marginal** coverage guarantee under exchangeability: averaged across the whole test set, the calibrated interval should cover the true age roughly at the target rate (e.g. 80%).

**Marginal coverage is not conditional coverage.** A calibrated interval that achieves 80% coverage overall can still systematically under-cover or over-cover any particular subgroup -- a specific age range, a specific gender label, a specific image-quality bucket -- while the population-average number looks fine. `scripts/evaluate.py` and `scripts/generate_final_report.py` report per-age-bucket coverage and width (both raw and calibrated) and narrow/wide interval examples specifically so this can be checked empirically for the dataset/split actually used, rather than assumed away. Always inspect the per-bucket table before trusting a single headline coverage number for any subgroup that matters to your use case.

A wide interval is not a defect -- it is the model expressing genuine uncertainty (e.g. for image conditions poorly represented in training data), and should be read as such rather than discarded in favor of the point estimate.

## Abstention behavior

The gender-label head returns **"Not sure"** instead of a class label whenever its top softmax probability is below `confidence_threshold` (default 0.80, in `configs/model.yaml`'s `gender_head` and mirrored in `configs/api.yaml`). This is a deliberate design choice: a low-confidence guess presented as if it were a normal prediction is more misleading than an explicit "the model doesn't know." Abstention rate is tracked as a first-class evaluation metric (`abstention_rate` in `src/evaluation/metrics.py`) alongside accuracy, including under image corruption (`scripts/run_robustness.py`) -- a model that abstains more often under blur/noise/occlusion is behaving correctly, not failing. Abstention applies **only** to the gender-label head; the age head always returns a point estimate and interval (there is no "Not sure" state for age), so the interval width itself is the age head's analogous signal of low confidence.

**Four related-but-distinct numbers, all reported wherever gender-label performance is measured (`src/evaluation/metrics.py`, `src/evaluation/backbone_comparison.py`):**

- **Selective accuracy** (`gender_accuracy`): accuracy computed only over samples the model actually answered (denominator excludes abstentions). This is "how good is the model when it commits to an answer."
- **Coverage** (`gender_coverage`): the fraction of samples the model actually answers, `1 - abstention_rate`.
- **Abstention rate** (`abstention_rate`): `1 - coverage` -- the fraction returned as "Not sure."
- **Effective accuracy** (`gender_effective_accuracy`): correct-and-accepted predictions divided by *all* samples (denominator includes abstentions). This is "how often does a user actually get a correct answer out of everything they asked."

A model can have excellent selective accuracy while abstaining on every difficult case, which would look poor on effective accuracy -- reporting only selective accuracy would hide that trade-off, so both numbers (plus coverage/abstention) are always reported together rather than either alone.

## Face-detection limitations

Because the model is trained on tightly face-cropped images (e.g. UTKFace), `/predict` and related endpoints first try to crop to the largest detected face using a classical Haar-cascade detector (`src/inference/face_detection.py` -- OpenCV's bundled Viola-Jones detector; not a neural network, no pretrained weights downloaded, and not a biometric identity system). **If no face is found, the API declines to generate an age or dataset gender-label prediction at all** (`age`/`gender`/`gradcam` are returned as `null`, with an explanatory warning) rather than running the model on a non-face image and returning a confident-looking but meaningless result -- e.g. a photo of an object, an animal, or a heavily obscured face should not receive an age or gender-label guess.

This detector is real but classical and moderate-accuracy:

- It can miss faces at extreme angles, in poor lighting, when heavily occluded, or when a face is very small/large relative to the frame.
- It can occasionally produce a false-positive detection on a non-face region with face-like contrast patterns.
- It performs no landmark localization, liveness check, or identity matching -- it only draws a bounding box, and that box is discarded after cropping (never stored or compared against other images).
- The synthetic cartoon placeholder images in `data/demo_images/` may or may not be detected as faces, since they are drawings rather than photographs; this is expected, not a bug.

Toggle via `api.enable_face_detection` / `api.face_margin_ratio` in `configs/api.yaml`.

## Privacy considerations

- Uploaded images are processed **in memory** for the duration of a single request and are **not persisted to disk by default** (`api.persist_uploaded_images: false` in `configs/api.yaml`).
- The face-detection bounding box is used only to crop the in-memory image before inference; it is never logged, stored, or used to match against any other image or database.
- No third-party service, model-hub download, or external API call is made at inference time -- everything runs locally against the loaded checkpoint.
- This system performs no biometric identification or re-identification: it has no notion of "who" is in an image, only age/gender-label estimates for whatever single image it is given, statistically independent of any other request.
- If you enable `persist_uploaded_images` or otherwise log uploaded images, you take on responsibility for the privacy/consent implications of storing photos of real people -- this is off by default specifically to avoid that by default.

## Training data

Trained on a Kaggle-hosted, user-supplied dataset (default target: a UTKFace-style dataset). See `docs/data_card.md` for full provenance, licensing, and the train/validation/calibration/test split protocol. **No dataset is bundled with this repository** -- you must supply your own via `scripts/download_kaggle_data.py` or a local copy, subject to that dataset's own license and terms. The only images committed to this repository are the five synthetic, procedurally-drawn placeholders in `data/demo_images/` (see that directory's README), which depict no real person.

## Ethical considerations

- "Gender-label" is a categorical field defined by the source dataset's authors/annotators, not a determination of a person's gender identity.
- Dataset labels may be binary, incomplete, self-reported, annotator-assigned, or otherwise limited; they do not necessarily reflect how any individual identifies.
- Age and gender-label distributions in any given dataset are rarely demographically balanced or globally representative; performance figures in `outputs/` describe behavior on the specific dataset/split used, not general human populations.
- Race/ethnicity metadata, where present in a dataset (e.g. UTKFace), is never used as a model input, target, or split criterion in this codebase.

## Known limitations

- Small custom ResNet-18 trained on a single dataset; expect a real generalization gap to other cameras, populations, lighting, and capture conditions.
- Conformal-calibrated intervals provide **marginal** (population-average) coverage guarantees, not per-individual or per-subgroup guarantees -- see "Uncertainty interpretation" above. Every calibration artifact records which exact checkpoint, split file, and ordered test-sample set it was fit against (`src/evaluation/calibration.py`); `scripts/evaluate.py` and `scripts/run_robustness.py` refuse (loudly) to apply a calibration artifact to a mismatched checkpoint or split, so a stale or cross-experiment calibration offset can never be silently applied.
- Every trained checkpoint (`scripts/run_seeds.py` / `scripts/run_experiments.py`) gets its own isolated `experiments/<experiment>/seed_<seed>/{checkpoints,calibration,metrics,plots,robustness,knn}` artifact tree -- never a single shared `outputs/` directory a second experiment or seed could silently overwrite or get calibrated against.
- Robustness figures (`scripts/run_robustness.py`, one isolated directory per checkpoint) are limited to the corruption types/severities defined in `configs/robustness.yaml` (blur, brightness, contrast, Gaussian noise, JPEG compression, partial occlusion, resize/low-resolution, grayscale); real-world degradations may differ or compound in ways not captured here. By default the full test split is evaluated; `--max-samples` deterministically stratified-samples by age bucket and gender label rather than truncating to whichever rows happen to sort first.
- Face detection is classical/Haar-cascade based (see "Face-detection limitations" above), not a modern neural detector; it can miss or misfire on faces outside typical frontal, well-lit conditions.
- Any mean +/- std figures across seeds (`scripts/run_seeds.py`) reflect only the seeds actually run in a given environment; a small seed count (e.g. 1-2) gives a weak estimate of run-to-run variance, and this repository never fabricates a standard deviation from a single run.
- A claim that one backbone has a statistically lower AURC (area under the selective-risk-coverage curve) than another is only ever made when a paired bootstrap confidence interval computed on the **AURC statistic itself** excludes zero (`src/evaluation/selective.py:paired_bootstrap_aurc_diff_ci`) -- a CI at one fixed coverage level is not treated as sufficient evidence for an AURC-level claim.

## How to reproduce reported numbers

See `docs/reproducibility.md`. All numbers in `outputs/` and any generated report (`docs/architecture_analysis_generated.md`, `docs/final_results_report.md`) are produced by the scripts in `scripts/` against a specific checkpoint and split; no numbers in this repository are fabricated or hand-edited. Where an artifact hasn't been generated yet in a given environment, the generated reports say so explicitly instead of showing a placeholder number.
