"""Pydantic request/response schemas for the FastAPI backend."""

from __future__ import annotations

from pydantic import BaseModel, Field

DISCLAIMER = (
    "This tool is for research and demonstration only. Predictions may be "
    "inaccurate, biased, or unreliable. Gender-related output reflects labels "
    "in the training dataset and is not a determination of identity."
)


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_version: str
    checkpoint_name: str | None
    disclaimer: str = DISCLAIMER


class ModelInfoResponse(BaseModel):
    model_version: str
    checkpoint_name: str | None
    architecture: str | None
    gender_label_names: list[str]
    gender_confidence_threshold: float
    age_min: float
    age_max: float
    calibration_available: bool
    knn_available: bool
    disclaimer: str = DISCLAIMER


class QualityDiagnosticsResponse(BaseModel):
    width: int
    height: int
    brightness: float
    contrast: float
    blur_score: float
    file_type: str
    file_size_bytes: int
    warnings: list[str]


class AgeUncertaintyMetadata(BaseModel):
    method: str
    calibrated: bool
    note: str


class AgePredictionResponse(BaseModel):
    q10: float
    q50: float = Field(..., description="Central age estimate")
    q90: float
    q10_calibrated: float | None
    q90_calibrated: float | None
    is_calibrated: bool
    uncertainty: AgeUncertaintyMetadata


class GenderLabelPredictionResponse(BaseModel):
    probabilities: dict[str, float]
    predicted_label: str | None = Field(None, description="Null when abstained ('Not sure')")
    confidence: float
    abstained: bool
    display_label: str = Field(..., description='"Not sure" or the predicted dataset gender-label name')


class GradCamResponse(BaseModel):
    age_attention_map_base64: str | None
    gender_attention_map_base64: str | None
    label: str = "Model attention visualization"
    caveat: str = (
        "Grad-CAM highlights regions that influenced the model's output. It is a "
        "gradient-weighted visualization, not proof of causality or an explanation "
        "of human reasoning."
    )


class KNNComparisonResponse(BaseModel):
    age_q10: float
    age_q50: float
    age_q90: float
    gender_probabilities: dict[str, float]
    gender_predicted_label: str | None
    gender_display_label: str
    gender_abstained: bool
    mean_neighbor_distance: float


class PredictionResponse(BaseModel):
    age: AgePredictionResponse
    gender: GenderLabelPredictionResponse
    quality: QualityDiagnosticsResponse
    gradcam: GradCamResponse | None
    knn_comparison: KNNComparisonResponse | None
    model_version: str
    checkpoint_name: str | None
    warnings: list[str]
    latency_ms: float
    disclaimer: str = DISCLAIMER


class ReloadModelsResponse(BaseModel):
    reloaded: bool
    model_loaded: bool
    checkpoint_name: str | None
    warnings: list[str]
