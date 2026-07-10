# Fresh Clone Validation Protocol

Step-by-step procedure to verify the repository works from a fresh clone.

## Prerequisites

- Python 3.10+ installed
- pip and virtualenv available
- Git installed
- (Optional) Kaggle API credentials for data download

## Validation Steps

### Step 1: Clone and Install

```bash
git clone https://github.com/adischwartz15/AgeGender.git
cd AgeGender
make install
```

**Expected**: No errors. All dependencies install from `requirements.txt`.

**Verify**:
```bash
.venv/Scripts/python -c "import torch; import fastapi; import cv2; print('OK')"
```

### Step 2: Run Test Suite

```bash
make test
```

**Expected**: 279 tests pass, 4 warnings (all benign: Starlette deprecation,
PyTorch LR scheduler order in test-only code).

**Verify**: Output ends with `279 passed, 4 warnings`.

### Step 3: Lint/Type Check

```bash
make lint
```

**Expected**: No errors. Uses ruff for linting.

### Step 4: Config Validation

```bash
.venv/Scripts/python -c "
from src.utils.config import load_full_config
cfg = load_full_config()
print('Config loads:', list(cfg.keys()))
print('Backbone:', cfg['model']['backbone']['name'])
print('Image size:', cfg['dataset']['image_size'])
print('Split fractions sum:', sum([
    cfg['split']['train_fraction'],
    cfg['split']['validation_fraction'],
    cfg['split']['calibration_fraction'],
    cfg['split']['test_fraction'],
]))
"
```

**Expected**: Config loads without error, backbone is `custom_resnet18`,
image size is 128, fractions sum to 1.0.

### Step 5: Model Instantiation (No Data Needed)

```bash
.venv/Scripts/python -c "
import torch
from src.models.multitask_model import build_multitask_model
from src.utils.config import load_full_config
cfg = load_full_config()
model = build_multitask_model(cfg)
x = torch.randn(1, 3, 128, 128)
out = model(x)
print('Age q50:', out['age_output']['q50'][0].item())
print('Gender logits:', out['gender_logits'][0].tolist())
breakdown = model.parameter_breakdown()
print('Total params:', breakdown.total)
print('Backbone params:', breakdown.backbone_params)
print('Adapter params:', breakdown.adapter_params)
"
```

**Expected**: Model runs a forward pass on random data. Total params ~11.57M.

### Step 6: Data Download (Requires Kaggle Credentials)

```bash
cp .env.example .env
# Edit .env: fill in KAGGLE_USERNAME and KAGGLE_KEY
make download-data
make prepare-data
```

**Expected**: Data downloaded to `data/raw/`, split written to `data/splits/`.

### Step 7: Smoke Training (With Data)

```bash
make train-smoke
# or:
.venv/Scripts/python scripts/train.py --experiment-name smoke --set training.stages.stage_a.epochs=1 training.stages.stage_b.epochs=0 training.stages.stage_c.epochs=0
```

**Expected**: 1-epoch training run completes, checkpoint saved.

### Step 8: API Startup (With or Without Checkpoint)

```bash
make api
```

**Expected without checkpoint**: API starts, `/health` returns `model_loaded: false`.

**Expected with checkpoint**: API starts, `/health` returns `model_loaded: true`.

```bash
curl http://localhost:8000/health
```

### Step 9: Demo Readiness (With Checkpoint)

```bash
make check-demo
```

**Expected**: Reports checkpoint status, calibration status, kNN status.

### Step 10: Documentation Build

All documentation is plain Markdown — no build step needed. Verify key files exist:

```bash
ls docs/architecture_analysis.md
ls docs/experiment_plan.md
ls docs/results.md
ls docs/model_card.md
ls docs/data_card.md
ls docs/evaluation.md
ls docs/calibration.md
```

## What Cannot Be Validated Without Data

| Step | Requires |
|---|---|
| Full training run | Dataset (Kaggle download) |
| Evaluation metrics | Trained checkpoint + test set |
| Calibration | Trained checkpoint + calibration split |
| kNN baseline | Trained checkpoint + training embeddings |
| Robustness evaluation | Trained checkpoint + test set |
| Architecture comparison report | Multiple trained checkpoints |

## What Can Be Validated Immediately

| Step | Status |
|---|---|
| Install from requirements.txt | ✅ Verified |
| All 279 tests pass | ✅ Verified |
| Config loads and validates | ✅ Verified |
| Model instantiates and runs forward pass | ✅ Verified |
| API starts | ✅ Verified |
| Documentation files present | ✅ Verified |
| `.gitignore` excludes data/checkpoints | ✅ Verified |
| No secrets in repository | ✅ Verified |
