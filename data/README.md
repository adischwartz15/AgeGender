# data/

This directory holds local dataset files. Nothing under `raw/`, `processed/`,
`splits/`, or any dataset content is committed to git (see `.gitignore`) --
only this README and `.gitkeep` placeholders are tracked.

- `raw/` -- exactly what `scripts/download_kaggle_data.py` extracts from
  Kaggle (or what you copy in manually for a non-Kaggle local dataset).
  Never edit files here by hand; re-run the download script instead.
- `processed/` -- reserved for any intermediate cached artifacts a future
  adapter might produce (currently unused by the default UTKFace/CSV
  adapters, which read directly from `raw/`).
- `splits/` -- `full_metadata_with_splits.csv`, produced by
  `scripts/prepare_data.py`. This is the single source of truth for the
  train/val/test split used by every experiment in `configs/experiments.yaml`,
  so all ablations are comparable.

See the root `README.md` for full setup, Kaggle credential, and dataset
format instructions.
