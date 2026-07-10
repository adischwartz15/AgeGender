# Demo Script

Step-by-step walkthrough for a live project demonstration.

## Before the Demo

### Setup (5 minutes before)
```bash
cd AgeGender
make check-demo                    # verify checkpoint + calibration
make api &                         # start API on port 8000
cd frontend && npm run dev &       # start frontend on port 5173
```

### Verify
```bash
curl http://localhost:8000/health
# Should return: {"status":"ok","model_loaded":true,...}
```

### Have Ready
- 3-4 test face images (frontal, clear lighting, varied ages)
- 1 non-face image (to show decline-to-predict behavior)
- 1 partially occluded face (to show abstention)
- Browser open to http://localhost:5173

---

## Part 1: Architecture Overview (3 min)

### Talking Points
1. "This is a multi-task learning research project for age estimation and
   dataset gender-label prediction from face images."
2. "The backbone is a custom ResNet-18 — hand-written, no torchvision, no
   pretrained weights."
3. "Both tasks share the backbone, with task-specific bottleneck adapters
   that start as identity and learn task-specific adjustments."

### Show Code (optional)
- `src/models/custom_resnet.py` — BasicBlock with skip connections
- `src/models/adapters.py` — 43 lines, zero-initialized up projection
- `src/models/heads.py` — softplus guarantees q10 ≤ q50 ≤ q90

---

## Part 2: Live Prediction (5 min)

### Action: Upload a clear face image
1. Go to the frontend (http://localhost:5173)
2. Upload a clear, frontal face image
3. Show the prediction results

### Point Out
- **Age**: q10, q50 (central estimate), q90 forming a prediction interval
- **Gender label**: Predicted label with confidence percentage
- **Disclaimer**: Visible in the response — "research only"
- **Latency**: Shown in milliseconds

### Action: Upload a non-face image
1. Upload a landscape, object, or animal image
2. Show: `face_detected: false`, `age: null`, `gender: null`

### Talking Point
"The model declines to predict when no face is detected. This prevents
meaningless predictions on images outside its training distribution."

### Action: Show abstention
1. Upload an ambiguous or partially occluded face
2. If model abstains: Show `abstained: true`, `display_label: "Not sure"`

### Talking Point
"When the model's confidence is below 80%, it abstains rather than
guessing. We report selective accuracy (accuracy on answered questions)
AND effective accuracy (accounting for abstentions) to be transparent
about this trade-off."

---

## Part 3: Grad-CAM Visualization (2 min)

### Action
1. Upload a face image with Grad-CAM enabled
2. Show the age and gender attention heatmaps

### Talking Points
- "This shows which image regions influence the model's output."
- "It's a gradient-weighted visualization, NOT a causal explanation."
- "For age, the model tends to focus on facial texture and shape."
- "For gender labels, it often focuses on hair and facial structure."

---

## Part 4: Experiment Framework (3 min)

### Show
- `configs/experiments.yaml` — 9 experiments defined
- `docs/results.md` — Committed results from one real run
- Parameter comparison table: sharing halves backbone parameters

### Talking Points
1. "We have 9 controlled experiments, each isolating one variable."
2. "Experiment 0b (PlainDeep18NoSkip) is a depth/width-matched control
   for residual connections — unlike SimpleCNN which also changes shape."
3. "Results are from one run, one seed. The 3-seed protocol exists but
   hasn't been committed yet."

---

## Part 5: Calibration (2 min)

### Show
- `src/evaluation/calibration.py` — conformal offset computation
- Results: raw coverage 0.79 vs nominal 0.80

### Talking Points
1. "Raw quantile regression doesn't guarantee coverage."
2. "Split conformal calibration adds a scalar offset to widen intervals
   until marginal coverage reaches the target."
3. "It uses a separate calibration split — never the test set."

---

## Part 6: Testing & Quality (2 min)

### Action
```bash
make test  # run in terminal
```

### Talking Points
- "279 tests, all passing."
- "Tests cover models, losses, calibration, robustness, API, face detection."
- "Smoke training tests use synthetic data — no dataset dependency."

---

## Part 7: API Endpoints (1 min)

### Show
Open http://localhost:8000/docs (FastAPI auto-docs)

### Point Out
- `POST /predict` — Main prediction endpoint
- `POST /predict/compare` — Parametric vs. kNN comparison
- `POST /predict/gradcam` — With attention maps
- `GET /health` — Readiness check
- `GET /models` — Model info
- `POST /quality-check` — Image quality only

---

## Wrap-Up (1 min)

### Key Takeaways
1. "Multi-task sharing halves parameters with competitive performance."
2. "Quantile regression gives prediction intervals, not just point estimates."
3. "Confidence-based abstention is more honest than forcing a prediction."
4. "The model knows its limitations — it declines on non-faces and uncertain cases."
5. "All results are scoped to one dataset and honestly qualified."

---

## Recovery Procedures

If something goes wrong during the demo, see `docs/demo_failure_modes.md`
for specific failure scenarios and recovery steps.
