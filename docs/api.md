# API Usage (FastAPI backend)

```bash
make api
```

Starts Uvicorn on `:8000`. Configuration lives in `configs/api.yaml`
(active checkpoint, CORS origins, upload limits, face-detection toggle,
gender-label display overrides, etc.).

## Endpoints

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | Liveness + whether a model is loaded |
| GET | `/models` | Active model/version/checkpoint info |
| POST | `/quality-check` | Image-quality diagnostics only (no model needed) |
| POST | `/predict` | Full prediction (`?include_gradcam=true&include_knn=true` optional) |
| POST | `/predict/compare` | Prediction with k-NN comparison always included |
| POST | `/predict/gradcam` | Prediction with Grad-CAM always included |
| POST | `/admin/reload-models` | Reload checkpoint/calibration/kNN index from disk |

Uploaded images are processed in memory and are not persisted to disk by
default (`api.persist_uploaded_images: false`). See `docs/model_card.md`
("Privacy considerations") for the full privacy discussion.

## Face-region preprocessing

Since the model is trained on tightly face-cropped images (e.g.
UTKFace), `/predict` and friends first try to crop to the largest
detected face using a classical Haar cascade
(`src/inference/face_detection.py` -- OpenCV's bundled Viola-Jones
detector, not a neural network, no pretrained weights downloaded), so an
arbitrary uploaded photo (with background, clothing, hair styling, etc.)
is closer to what the model actually learned from.

**If no face is found, the API declines to generate an age or dataset
gender-label prediction at all** (`age`/`gender`/`gradcam` are returned as
`null`, with a warning explaining why) rather than running the model on a
non-face image and returning a confident-looking but meaningless result
-- e.g. a photo of an object or an animal should not receive an age or
gender-label guess. Toggle via `api.enable_face_detection` /
`api.face_margin_ratio` in `configs/api.yaml`.

This is a real but classical/moderate-accuracy detector -- it can miss
faces at extreme angles, in poor lighting, or when occluded, and does not
perform identity verification or any other biometric function. See
`docs/model_card.md` ("Face-detection limitations") for the full
discussion.

## Example requests

```bash
curl http://localhost:8000/health

curl -X POST http://localhost:8000/quality-check \
  -F "file=@/path/to/face.jpg"

curl -X POST "http://localhost:8000/predict?include_gradcam=true&include_knn=true" \
  -F "file=@/path/to/face.jpg"
```

Example (abridged) `/predict` response shape, when a face was detected:

```json
{
  "age": {"q10": 24.1, "q50": 29.4, "q90": 35.2, "q10_calibrated": 22.8, "q90_calibrated": 36.9, "is_calibrated": true},
  "gender": {"probabilities": {"gender_label_0": 0.91, "gender_label_1": 0.09}, "predicted_label": "gender_label_0", "confidence": 0.91, "abstained": false, "display_label": "gender_label_0"},
  "quality": {"width": 512, "height": 512, "brightness": 0.52, "contrast": 0.21, "blur_score": 143.2, "warnings": []},
  "gradcam": null,
  "knn_comparison": null,
  "model_version": "v1",
  "checkpoint_name": "multitask_best_balanced_score.pt",
  "face_detected": true,
  "warnings": [],
  "latency_ms": 42.3,
  "disclaimer": "This tool is for research and demonstration only. ..."
}
```

...and when no face was detected (`age`/`gender`/`gradcam`/`knn_comparison`
are all `null` -- no prediction is generated):

```json
{
  "age": null,
  "gender": null,
  "quality": {"width": 800, "height": 600, "brightness": 0.61, "contrast": 0.30, "blur_score": 210.5, "warnings": []},
  "gradcam": null,
  "knn_comparison": null,
  "model_version": "v1",
  "checkpoint_name": "multitask_best_balanced_score.pt",
  "face_detected": false,
  "warnings": ["No face detected via classical Haar-cascade detection; declining to generate age or dataset gender-label predictions, since the model is only meaningful on face images similar to its training data."],
  "latency_ms": 8.1,
  "disclaimer": "This tool is for research and demonstration only. ..."
}
```

`gender.predicted_label` is `null` (not a class name) and `gender.abstained`
is `true` when the top class probability falls below
`model.gender_head.confidence_threshold` (default 0.80) -- see
`docs/model_card.md` ("Abstention behavior").

## Demo mode

```bash
make demo
```

A single command for a live demo instead of starting the backend and
frontend in two separate terminals. Runs a readiness check
(`scripts/check_demo_readiness.py`) confirming a trained checkpoint and a
conformal calibration artifact exist (warning, but not blocking, if the
optional k-NN index is missing), then launches Uvicorn and the Vite dev
server together as subprocesses, printing both URLs. Ctrl+C stops both.
If the readiness check fails, fix the reported item (e.g. `make train`
then `make calibrate CHECKPOINT=...`) or pass `--skip-readiness-check` to
`python scripts/run_demo.py` to launch anyway.

Five synthetic placeholder "face" images (procedurally drawn with PIL, no
real person, no dataset content) live in `data/demo_images/` for quick
upload during a demo -- see `data/demo_images/README.md`. Regenerate them
with `make demo-images`. Because they're cartoon shapes rather than
photographs, the classical face detector may decline to predict on some
of them; that's expected, and itself demonstrates the system's
decline-rather-than-guess safety behavior described above. For a demo
that reliably shows a full prediction, upload your own consented photo
through the frontend instead.
