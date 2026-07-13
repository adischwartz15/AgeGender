# Data Card

## Kaggle API setup

1. Create a Kaggle account, then create an API token at
   <https://www.kaggle.com/settings> ("Create New Token"). This downloads
   `kaggle.json` -- do not commit it.
2. Set environment variables (in `.env`, sourced by your shell, or
   directly in your environment):
   ```
   KAGGLE_USERNAME=<your-username>
   KAGGLE_KEY=<your-key>
   KAGGLE_DATASET_SLUG=<owner>/<dataset-name>   # e.g. jangedoo/utkface-new
   ```
3. Run `make download-data` (wraps `scripts/download_kaggle_data.py`). It
   validates credentials, downloads via the official `kaggle` package
   (no scraping), extracts into `data/raw/`, skips re-downloading unless
   `--force` is passed, and writes `data/raw/manifest.json` with the
   slug, timestamp, and file/image counts.

If credentials or the dataset slug are missing, the script prints setup
instructions and exits non-zero instead of failing silently.

## Default target dataset

This repository defaults to a UTKFace-style dataset (`DATASET_SOURCE=utkface`
in `.env.example`), downloaded through the official Kaggle API using
credentials and a dataset slug you supply yourself (see README.md). No
dataset is bundled with this repository.

**UTKFace filename convention:** `age_gender_race_date.jpg`, e.g.
`25_0_2_20170116174525125.jpg` decodes to age=25, gender_label=0, race=2
(race metadata is carried through for documentation only and is never used
as a model feature, target, or split criterion).

## Generic CSV adapter

For other Kaggle face datasets that ship a metadata CSV instead of encoding
labels in filenames, set `DATASET_SOURCE=csv` and configure
`configs/data.yaml`'s `dataset.csv` block:

- `metadata_csv`: path to the CSV
- `image_path_column`, `age_column`, `gender_label_column`: your dataset's actual column names
- `split_column` (optional): use a pre-existing split instead of re-splitting
- `subject_id_column` (optional): enables subject-level (not just row-level) leakage prevention
- `label_mapping` (optional): maps raw CSV values to `{0, 1}` for the gender-label column

## Label semantics and limitations

- **Gender-label is not gender identity.** It is whatever binary (or, if you
  extend the head, multi-class) field the source dataset's authors defined,
  which may be self-reported, annotator-assigned, inferred, or otherwise
  imperfect. Display names are configurable
  (`GENDER_LABEL_0`/`GENDER_LABEL_1` in `.env`, or `model.gender_head.class_names`
  in `configs/model.yaml`) and default to the neutral `gender_label_0`/`gender_label_1`
  until you explicitly set alternatives based on your dataset's own documentation.
- **Age labels** are whatever the dataset provides (often self-reported or
  estimated at collection time) and may contain noise, especially near the
  extremes of the range.
- **Missing labels are supported.** A row may have age only, gender-label
  only, both, or (rare, and dropped during validation) neither. Training
  uses masked losses so missing labels never contribute gradient signal for
  that sample/task.
- **Race/ethnicity** metadata, when present, is retained purely as
  descriptive metadata for transparency about dataset composition. It is
  never used as a feature, prediction target, or split key anywhere in this
  codebase.

## Validation and leakage prevention

```bash
make prepare-data
```

`scripts/prepare_data.py` (backed by `src/data/validation.py`) runs before
any training:

1. Verifies every image path is readable and above a minimum resolution; drops corrupt/unreadable files.
2. Detects and removes duplicate file paths and duplicate image content (via SHA-256 hash).
3. Reports age distribution, gender-label distribution, and image-size
   statistics to `outputs/data_quality/data_quality_report.json`.
4. Splits into **train / validation / calibration / test** (default fractions 60/15/10/15, `configs/data.yaml: split.*_fraction`) with a fixed seed; splits at the **subject level** (not just image level) when a `subject_id` column is available, so the same person never appears in more than one split. **UTKFace itself provides no `subject_id`/identity field**, so this guarantee does not currently apply to this project's actual dataset -- see [docs/reproducibility.md](reproducibility.md#stratified-locked-split).
5. Asserts no image path or subject_id spans multiple splits before writing `data/splits/full_metadata_with_splits.csv`.
6. For the final, locked split used by every reported experiment,
   `scripts/lock_split.py` additionally stratifies by age bin x gender
   label and records a manifest (split hash, per-stratum counts,
   near-duplicate audit) -- see
   [docs/reproducibility.md](reproducibility.md#stratified-locked-split).

Each of the four splits has exactly one job: `train` fits the model;
`validation` is read only for early stopping and checkpoint selection
(`scripts/train.py`); `calibration` is read only when fitting conformal
prediction intervals (`scripts/calibrate.py`); `test` is read only for
final, one-time evaluation (`scripts/evaluate.py`,
`scripts/run_robustness.py`, etc.). No split is ever used for more than
one of these roles -- in particular, calibration is **not** fit on the
validation split, since that would let the same data influence both
checkpoint selection and the calibration guarantee.

## Demographic coverage caveat

Whatever dataset you point this pipeline at, its demographic coverage
(age range, ethnic/racial composition, lighting conditions, camera types,
image quality, collection era/region) determines what the model can
possibly learn and how its error rates are distributed across groups.
Reported metrics describe performance **on that dataset's test split
only** and should not be extrapolated to populations or conditions not
represented in it.
