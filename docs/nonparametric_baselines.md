# Non-Parametric Baselines (Raw/PCA and Frozen-Backbone)

Two CPU-only, "unlearned" baselines that never touch this project's own
trained multi-task embeddings -- a sanity floor to compare the trained
models (Table A, Table B) against. Neither is a neural network that gets
trained on this task; both are classical k-NN / kernel methods over a
fixed feature representation.

## The two feature pipelines

- **`raw_pca`** -- flattened raw pixels (resized to a small, fixed size) ->
  train-only `StandardScaler` -> train-only PCA -> optional L2-normalization.
  The "no learning at all" floor.
- **`frozen_backbone`** -- pooled features from a frozen, ImageNet-pretrained
  backbone (`src/models/pretrained_resnet.py`, adapters/heads never
  attached, weights never fine-tuned) -> the same scaler/PCA/L2-norm
  protocol. Tests whether generic pretrained visual features alone (with no
  task-specific training at all) already carry most of the signal.

Implemented in `src/evaluation/nonparametric/`:

```
features.py    extract_raw_pixel_features, extract_frozen_backbone_features
pipeline.py    fit_feature_pipeline (train-only scaler/PCA), tune_knn_age,
               tune_knn_gender, tune_kernel_regression_age, tune_kde_gender
kernels.py     NadarayaWatsonRegressor, ClassConditionalKDEClassifier
               (numerically safe: NN fallback on kernel-weight underflow,
               diagonal-covariance fallback for tiny/singular classes)
```

## Protocol: which split does what

| Split | Used for |
|---|---|
| **train** | Fitting the scaler, PCA, and the reference set for k-NN/kernel methods -- never validation, calibration, or test. |
| **validation** | Selecting every hyperparameter (k, distance metric, PCA dimensionality, L2-normalization, kernel bandwidth) -- never test. |
| **calibration** | Fitting split-conformal calibration for the k-NN age-interval baseline only -- never validation or test. |
| **test** | Final, one-shot reported numbers -- never used for any selection. |

This is the same 4-way split discipline every other evaluation path in this
project follows (see [docs/reproducibility.md](reproducibility.md#stratified-locked-split)).
PCA is intentionally restricted to a small number of components
(`safe_n_components`, capped by both a fixed grid and by `min(n_samples,
n_features)`) -- the kernel/KDE methods run in this reduced space, never on
full-dimensional raw pixels or full-width backbone features, since
Nadaraya-Watson/KDE degrade badly in high dimensions.

## Running it

```bash
# 1. Validation-only hyperparameter search (writes outputs/nonparametric/best_params.json,
#    outputs/nonparametric/all_candidates.csv, and the fitted pipeline .pkl files)
python scripts/tune_nonparametric_baselines.py
python scripts/tune_nonparametric_baselines.py --feature-sources raw_pca   # just one pipeline
python scripts/tune_nonparametric_baselines.py --max-train-samples 2000    # faster grid search

# 2. Test-set evaluation using the already-selected hyperparameters (never re-tunes)
python scripts/evaluate_nonparametric_baselines.py
```

`evaluate_nonparametric_baselines.py` rejects the saved pipeline/params if
their recorded split SHA-256 doesn't match the currently locked split
(`--force-resplit` on `scripts/lock_split.py` invalidates all previously
tuned non-parametric baselines -- rerun step 1 first).

## Outputs

```
outputs/nonparametric/best_params.json         winning hyperparameters per feature source/task, with split hash + provenance
outputs/nonparametric/all_candidates.csv       every candidate config tried, not just the winner
outputs/nonparametric/{feature_source}_{age,gender}_pipeline.pkl   fitted scaler + PCA (train-only)
outputs/nonparametric/test_results.json        final test-set metrics, split hash
outputs/nonparametric/results.csv              same, as a flat table
```

## Notebooks

Both `notebooks/train_evaluate_colab.ipynb` and
`notebooks/train_evaluate_kaggle.ipynb` have an optional "Non-Parametric
Baselines" cell (`RUN_NONPARAMETRIC_BASELINES`, default `False`) in the
Supplementary Experiment section that runs both scripts in sequence and
copies `results.csv` into the run's report directory. It requires only the
locked split, not a GPU or a trained checkpoint.

## Limitations

- **Not a controlled ablation.** These are reference floors, not part of
  Table A or Table B -- never compared cell-for-cell against them as if on
  equal footing.
- **`frozen_backbone` reuses `pretrained_resnet.py` but trains nothing.**
  The backbone's ImageNet weights are used purely as a fixed feature
  extractor (`freeze_backbone()`, `model.backbone(images)`, `fc=Identity`)
  -- this is not the pretrained-ResNet **bridge baseline** described in
  [docs/transfer_learning.md](transfer_learning.md#model-families), which
  fine-tunes the same backbone end-to-end; the two must not be confused.
