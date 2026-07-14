# Reproducibility

## Stratified, Locked Split

`scripts/prepare_data.py` creates one file,
`data/splits/full_metadata_with_splits.csv`, with four splits: `train`,
`validation`, `calibration`, and `test`. Every experiment in
`configs/experiments.yaml` reads this same file, so results are always
comparable — any difference comes from the model/training change being
tested, not from a different data split.

- **Stratified**: samples are split so that each age group x gender label
  keeps roughly the same train/val/calibration/test proportions, not just
  a random shuffle. See `src/data/split_utils.py`.
- **Locked**: once a split is created, its SHA-256 hash is saved in
  `data/splits/split_manifest.json`. Every script that reads the split
  checks this hash first. If it doesn't match, something changed the
  split file — so it's either an error or a deliberate re-split. You can
  force a new split with `--force-resplit`. Either way, the old split is
  always backed up (never deleted) to `data/splits/.backup/`.
- **Safe to interrupt**: the split file and manifest are written to a
  temporary path first, then swapped in. A crash mid-write can't corrupt
  them.
- **The manifest records**: split method/seed, the split file's SHA-256,
  per-split sample counts, a near-duplicate check summary, the git commit
  that created it, and a timestamp.

### Near-duplicate check

`src/data/near_duplicate_audit.py` flags images that are probably
near-duplicates of each other (e.g. the same photo resized or
re-compressed) using perceptual hashing. It only reports candidates in
the manifest — it never removes anything automatically.

Every experiment records the split's SHA-256 in its own output, and
downstream steps like calibration refuse to run if that hash doesn't
match the currently locked split.

## Config-driven, not hardcoded

All architecture, training, and evaluation settings live in
`configs/*.yaml`. 
Every saved checkpoint embeds a full copy of the config that produced it.


## Environment

- Python 3.10+ 
- PyTorch, CPU or CUDA — see `requirements.txt`. 

## Running on Kaggle or Google Colab

See `docs/notebooks.md` for the full guide. In short: two ready-to-run
notebooks cover the entire pipeline — setup, data prep, tests, training,
calibration, evaluation, optional robustness/Grad-CAM/k-NN, multi-seed
runs, and a final report.
