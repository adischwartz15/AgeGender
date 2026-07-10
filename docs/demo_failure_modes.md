# Demo Failure Modes and Recovery

What can go wrong during a live demo, how each failure presents,
and what to say/do.

## Pre-Demo Checklist

Run before any demo:
```bash
make check-demo
```

This executes `scripts/check_demo_readiness.py` which verifies:
- [ ] Checkpoint file exists and loads
- [ ] Calibration artifact exists and matches checkpoint
- [ ] kNN index exists (warning if missing, not blocking)
- [ ] OpenCV Haar cascades load
- [ ] Config files parse without errors

---

## Failure Scenarios

### 1. No Checkpoint Found

**Symptom**: API returns HTTP 503: "No trained model checkpoint is loaded."

**Recovery**:
1. Check `configs/api.yaml` → `api.active_checkpoint` path
2. Verify the `.pt` file exists at that path
3. If missing: "The checkpoint file is a trained model artifact (~45 MB);
   it's gitignored by design. Run `make train` or download from the
   training output."
4. Call `POST /admin/reload-models` after fixing

**What to say**: "Checkpoints are excluded from version control by design
because they're large binary files. The training pipeline generates them."

---

### 2. Face Not Detected

**Symptom**: Response has `face_detected: false`, `age: null`, `gender: null`

**Recovery**:
- Try a different photo with a clear, frontal face
- Avoid: side profiles, heavy sunglasses, masks, animal faces
- The classical Haar cascade has limited recall

**What to say**: "This is by design — the model was trained on face crops,
so running it on non-face images would give meaningless predictions. The
detector uses a classical Haar cascade, which trades recall for
simplicity and no external model dependency."

---

### 3. Model Returns "Not Sure"

**Symptom**: `gender.abstained: true`, `display_label: "Not sure"`

**Recovery**:
- This is correct behavior, not a failure
- The confidence threshold is 0.80 (configurable)
- Try a clearer face image

**What to say**: "The model abstains when its confidence is below 80%.
This is a deliberate safety feature — we report selective accuracy
alongside coverage to quantify this trade-off."

---

### 4. Uncalibrated Intervals Warning

**Symptom**: `is_calibrated: false`, warning about missing calibration

**Recovery**:
1. Run `make calibrate CHECKPOINT=<path>`
2. Check `configs/api.yaml` → `api.calibration_dir`
3. Call `POST /admin/reload-models`

**What to say**: "Conformal calibration requires a separate post-training
step using a dedicated calibration split. Without it, the raw q10-q90
interval has no coverage guarantee."

---

### 5. Grad-CAM Returns Blank/Uniform Heatmap

**Symptom**: Heatmap is all one color (no spatial structure)

**Recovery**:
- This can happen when the model's prediction is very confident
  (gradients are small)
- Try a different image
- Not a bug — it means the model's output is not strongly sensitive
  to any particular region

**What to say**: "Grad-CAM gradients can be small when the model is very
confident, producing a near-uniform heatmap. This is a known limitation
of gradient-based attribution methods."

---

### 6. API Startup Fails (Module Not Found)

**Symptom**: `ModuleNotFoundError` when running `make api`

**Recovery**:
```bash
make install  # reinstalls all dependencies
```

**What to say**: "The project uses an editable pip install for correct
import resolution. Running `make install` sets up the correct package
paths."

---

### 7. Frontend Cannot Connect to API

**Symptom**: Frontend shows "Connection refused" or CORS errors

**Recovery**:
1. Verify API is running: `curl http://localhost:8000/health`
2. Check CORS origins in `configs/api.yaml`
3. Frontend default port is 5173; API expects it in `cors_origins`

**What to say**: "The frontend and API are separate processes. CORS is
configured to allow the development frontend origin."

---

### 8. Large Image Upload Rejected

**Symptom**: HTTP 413: "Uploaded file exceeds the maximum allowed size"

**Recovery**:
- Maximum is 10 MB (configurable in `configs/api.yaml`)
- Resize the image before uploading

**What to say**: "We enforce a 10 MB upload limit as a basic safety
measure for a research demo."

---

### 9. Slow Prediction on CPU

**Symptom**: Latency > 2 seconds per image

**Recovery**:
- This is expected on CPU (no GPU)
- Prediction should be ~2ms on GPU, ~100ms on CPU

**What to say**: "Inference latency depends on hardware. The model is
~11M parameters, which is lightweight for a ResNet but still benefits
from GPU acceleration."

---

### 10. kNN Comparison Not Available

**Symptom**: `knn_comparison: null`, warning about missing kNN index

**Recovery**:
1. Run `make build-knn CHECKPOINT=<path>`
2. Check `configs/api.yaml` → `api.knn_index_dir`
3. Call `POST /admin/reload-models`

**What to say**: "The kNN baseline is a post-hoc evaluation tool, not
part of the core model. It requires building an index from training
set embeddings."

---

## Things That Look Like Bugs But Aren't

| Symptom | Explanation |
|---|---|
| Age prediction outside [0, 120] | Calibration offset can push q10 below 0 or q90 above 120; this is correct conformal behavior |
| Gender labels show "gender_label_0" | No `.env` configured; set `GENDER_LABEL_0=male` and `GENDER_LABEL_1=female` for UTKFace |
| Different predictions on same image | If using training transforms (not eval transforms), augmentation adds randomness |
| `balanced_score` can be negative | If `age_mae / age_max > gender_accuracy`, the balanced metric goes negative |
| Empty plots directory | Plots are generated during training/evaluation runs, not at repo clone time |

---

## Emergency Recovery Script

If everything breaks during demo:
```bash
# Full reset (re-install + reload)
make install
make api &
sleep 3
curl http://localhost:8000/health
```
