"""FastAPI backend for face multi-task age + dataset gender-label prediction.

Research and demonstration only. Uploaded images are processed in memory
and are never written to disk or persisted by default (see
``api.persist_uploaded_images`` in configs/api.yaml, which defaults to
False and is not exposed for change via the API itself).
"""

from __future__ import annotations

import base64
import io
import logging
from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

from src.api.dependencies import app_state, get_app_state, get_predictor
from src.api.schemas import (
    DISCLAIMER,
    AgePredictionResponse,
    AgeUncertaintyMetadata,
    GenderLabelPredictionResponse,
    GradCamResponse,
    HealthResponse,
    KNNComparisonResponse,
    ModelInfoResponse,
    PredictionResponse,
    QualityDiagnosticsResponse,
    ReloadModelsResponse,
)
from src.inference.quality import compute_quality_diagnostics
from src.utils.config import CONFIG_DIR, load_config, load_env_file
from src.utils.logging import get_logger

load_env_file()  # populate os.environ from .env (e.g. GENDER_LABEL_0/1) before anything reads it

logger = get_logger("api")


@asynccontextmanager
async def _lifespan(app: FastAPI):
    """Modern replacement for the deprecated ``@app.on_event("startup")`` hook.

    Everything after ``yield`` would run on shutdown; there is nothing to
    clean up here (no open connections/background tasks), so the
    generator just yields once.
    """
    try:
        app_state.load()
    except Exception:  # pragma: no cover - defensive: API should still boot
        logger.exception("Failed to load model artifacts at startup; /health will report not-ready")
    yield


app = FastAPI(
    title="face-multitask-research API",
    description=(
        "Research/education-only API for multi-task face age estimation and "
        "dataset gender-label prediction. " + DISCLAIMER
    ),
    version="0.1.0",
    lifespan=_lifespan,
)


def _configure_cors() -> None:
    """Add CORS middleware using ``configs/api.yaml`` directly (not ``app_state.config``).

    Middleware must be registered before the app starts handling requests
    (Starlette raises if ``add_middleware`` is called afterward), which is
    before ``_lifespan``'s startup code runs -- so this reads
    ``configs/api.yaml`` itself rather than depending on ``app_state.load()``,
    which hadn't populated ``app_state.config`` yet at the point this used
    to be called, silently always falling back to the wildcard default
    regardless of the configured ``cors_origins``.
    """
    api_config = load_config(CONFIG_DIR / "api.yaml").get("api", {})
    origins = api_config.get("cors_origins", ["*"])
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins or ["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )


_configure_cors()


MAX_UPLOAD_BYTES = 10 * 1024 * 1024


async def _read_image(file: UploadFile) -> Image.Image:
    contents = await file.read()
    if len(contents) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Uploaded file exceeds the maximum allowed size")
    try:
        image = Image.open(io.BytesIO(contents))
        image.load()
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not read uploaded image: {exc}") from exc
    return image


def _encode_heatmap_overlay(image: Image.Image, heatmap: np.ndarray) -> str:
    """Blend a Grad-CAM heatmap onto the image and return a base64 PNG string."""
    import matplotlib.cm as cm

    rgb = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    colored = cm.jet(heatmap)[:, :, :3]
    overlay = 0.55 * rgb + 0.45 * colored
    overlay = np.clip(overlay, 0, 1)
    overlay_img = Image.fromarray((overlay * 255).astype(np.uint8))
    buffer = io.BytesIO()
    overlay_img.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    state = get_app_state()
    predictor = state.predictor
    return HealthResponse(
        status="ok",
        model_loaded=predictor is not None and predictor.is_ready(),
        model_version=state.config.get("api", {}).get("model_version", "v1"),
        checkpoint_name=predictor.artifacts.checkpoint_name if predictor else None,
    )


@app.get("/models", response_model=ModelInfoResponse)
def models() -> ModelInfoResponse:
    predictor = get_predictor()
    config = predictor.config or {}
    model_cfg = config.get("model", {})
    return ModelInfoResponse(
        model_version=predictor.api_config.get("model_version", "v1"),
        checkpoint_name=predictor.artifacts.checkpoint_name,
        architecture=model_cfg.get("architecture"),
        gender_label_names=predictor.class_names,
        gender_confidence_threshold=predictor.confidence_threshold,
        age_min=model_cfg.get("age_head", {}).get("age_min", 0),
        age_max=model_cfg.get("age_head", {}).get("age_max", 120),
        calibration_available=predictor.artifacts.calibration is not None,
        knn_available=predictor.artifacts.knn_baseline is not None,
    )


@app.post("/quality-check", response_model=QualityDiagnosticsResponse)
async def quality_check(file: UploadFile = File(...)) -> QualityDiagnosticsResponse:
    contents = await file.read()
    if len(contents) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Uploaded file exceeds the maximum allowed size")
    try:
        image = Image.open(io.BytesIO(contents))
        image.load()
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not read uploaded image: {exc}") from exc
    diagnostics = compute_quality_diagnostics(image, image.format or "unknown", len(contents))
    return QualityDiagnosticsResponse(**diagnostics.as_dict())


def _build_prediction_response(result) -> PredictionResponse:
    gradcam_response = None
    if result.gradcam_age is not None or result.gradcam_gender is not None:
        # Overlay onto the face crop actually fed to the model (model_input_image),
        # not the raw upload -- see predictor.py for why.
        base_image = result.model_input_image
        age_b64 = _encode_heatmap_overlay(base_image, result.gradcam_age) if result.gradcam_age is not None else None
        gender_b64 = _encode_heatmap_overlay(base_image, result.gradcam_gender) if result.gradcam_gender is not None else None
        gradcam_response = GradCamResponse(age_attention_map_base64=age_b64, gender_attention_map_base64=gender_b64)

    knn_response = None
    if result.knn is not None:
        knn_response = KNNComparisonResponse(
            age_q10=result.knn.age_q10, age_q50=result.knn.age_q50, age_q90=result.knn.age_q90,
            gender_probabilities=result.knn.gender_probabilities,
            gender_predicted_label=result.knn.gender_predicted_label,
            gender_display_label=result.knn.gender_predicted_label or "Not sure",
            gender_abstained=result.knn.gender_abstained,
            mean_neighbor_distance=result.knn.mean_neighbor_distance,
        )

    age_response = None
    if result.age is not None:
        age_response = AgePredictionResponse(
            q10=result.age.q10, q50=result.age.q50, q90=result.age.q90,
            q10_calibrated=result.age.q10_calibrated, q90_calibrated=result.age.q90_calibrated,
            is_calibrated=result.age.is_calibrated,
            uncertainty=AgeUncertaintyMetadata(
                method="split_conformal_cqr" if result.age.is_calibrated else "uncalibrated_pinball_quantiles",
                calibrated=result.age.is_calibrated,
                note=(
                    "q10/q90 form a prediction interval, not a guarantee; calibrated "
                    "intervals target marginal coverage under split-conformal calibration."
                ),
            ),
        )

    gender_response = None
    if result.gender is not None:
        gender_response = GenderLabelPredictionResponse(
            probabilities=result.gender.probabilities,
            predicted_label=result.gender.predicted_label,
            confidence=result.gender.confidence,
            abstained=result.gender.abstained,
            display_label=result.gender.predicted_label or "Not sure",
        )

    return PredictionResponse(
        age=age_response,
        gender=gender_response,
        quality=QualityDiagnosticsResponse(**result.quality.as_dict()),
        gradcam=gradcam_response,
        knn_comparison=knn_response,
        model_version=result.model_version,
        checkpoint_name=result.checkpoint_name,
        face_detected=result.face_detected,
        warnings=result.warnings,
        latency_ms=result.latency_ms,
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict(
    file: UploadFile = File(...), include_gradcam: bool = False, include_knn: bool = False
) -> PredictionResponse:
    predictor = get_predictor()
    if not predictor.is_ready():
        raise HTTPException(
            status_code=503,
            detail="No trained model checkpoint is loaded. Train a model (make train) and set "
            "api.active_checkpoint, then call /admin/reload-models.",
        )
    image = await _read_image(file)
    result = predictor.predict(image, include_gradcam=include_gradcam, include_knn=include_knn)
    return _build_prediction_response(result)


@app.post("/predict/compare", response_model=PredictionResponse)
async def predict_compare(file: UploadFile = File(...)) -> PredictionResponse:
    """Convenience endpoint: always runs the parametric-vs-kNN comparison."""
    predictor = get_predictor()
    if not predictor.is_ready():
        raise HTTPException(status_code=503, detail="No trained model checkpoint is loaded.")
    image = await _read_image(file)
    result = predictor.predict(image, include_gradcam=False, include_knn=True)
    return _build_prediction_response(result)


@app.post("/predict/gradcam", response_model=PredictionResponse)
async def predict_gradcam(file: UploadFile = File(...)) -> PredictionResponse:
    """Convenience endpoint: always runs Grad-CAM (age + gender attention maps)."""
    predictor = get_predictor()
    if not predictor.is_ready():
        raise HTTPException(status_code=503, detail="No trained model checkpoint is loaded.")
    image = await _read_image(file)
    result = predictor.predict(image, include_gradcam=True, include_knn=False)
    return _build_prediction_response(result)


@app.post("/admin/reload-models", response_model=ReloadModelsResponse)
def reload_models() -> ReloadModelsResponse:
    """Reload the active checkpoint, calibration artifact, and kNN index from disk."""
    app_state.load()
    predictor = app_state.predictor
    return ReloadModelsResponse(
        reloaded=True,
        model_loaded=predictor is not None and predictor.is_ready(),
        checkpoint_name=predictor.artifacts.checkpoint_name if predictor else None,
        warnings=predictor.artifacts.warnings if predictor else [],
    )
