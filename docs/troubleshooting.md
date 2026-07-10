# Troubleshooting

- **"No trained checkpoint found"** from the API: train a model first
  (`make train` or `make experiments`) and confirm
  `configs/api.yaml: api.active_checkpoint` points at a real file, then hit
  `POST /admin/reload-models`.
- **Kaggle download fails with a credentials error**: confirm
  `KAGGLE_USERNAME`/`KAGGLE_KEY`/`KAGGLE_DATASET_SLUG` are set, either in a
  local `.env` file (copied from `.env.example` -- loaded automatically by
  `src/utils/config.py:load_env_file()`) or exported directly in your shell
  (shell exports always take priority over `.env`). See `docs/data_card.md`
  ("Kaggle API setup") for the full setup steps.
- **CUDA out of memory**: lower `training.batch_size` in
  `configs/training.yaml`, or run on CPU (`device: cpu` in
  `configs/default.yaml`) for small-scale experimentation.
- **Predictions look wildly wrong / overconfident on a photo that looks
  fine to you**: check the response's `face_detected` field and
  `warnings`. The model is trained on tightly face-cropped images (e.g.
  UTKFace); a photo with a lot of background, heavy styling/makeup,
  jewelry, or a watermark/overlay across the face is a different visual
  distribution even when a human would call it "a clear photo of a
  person," and face-crop preprocessing (see `docs/api.md`) only partially
  compensates for that gap. Try a plain, front-facing, tightly-cropped
  photo to check whether the issue is input distribution rather than a
  bug.
- **Frontend can't reach the API**: confirm the backend is running on
  `:8000` (`curl http://localhost:8000/health`) and that the Vite dev
  server's proxy config (`frontend/vite.config.ts`) points at it.
- **Tests fail with `ModuleNotFoundError: src`**: run pytest from the
  repository root (`pyproject.toml` sets `pythonpath = ["."]`), or
  `pip install -e .`-equivalent isn't required since tests add the repo
  root to `sys.path` implicitly via `pytest`'s rootdir behavior.
