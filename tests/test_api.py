"""API tests: health endpoint, and prediction schema using a mocked (untrained) model.

No real checkpoint file is required -- a tiny in-memory model is injected
directly into the app state so these tests run fast and without GPU/data.
"""

from __future__ import annotations

import io

import numpy as np
from fastapi.testclient import TestClient
from PIL import Image

from src.api import dependencies
from src.api.main import app
from src.inference.artifacts import LoadedArtifacts
from src.inference.predictor import Predictor
from src.models.multitask_model import MultiTaskFaceModel


def _tiny_model_config():
    return {
        "dataset": {"image_size": 32},
        "gradcam": {"target_layer": "layer4"},
        "model": {
            "architecture": "shared_adapters",
            "backbone": {"block_layout": [1, 1, 1, 1], "embedding_dim": 32, "stem_channels": 8},
            "adapters": {"enabled": True, "bottleneck_dim": 8, "dropout": 0.0},
            "age_head": {"hidden_dim": 8, "dropout": 0.0, "age_min": 0, "age_max": 120},
            "gender_head": {
                "hidden_dim": 8, "dropout": 0.0, "num_classes": 2,
                "class_names": ["gender_label_0", "gender_label_1"], "confidence_threshold": 0.80,
            },
            "loss_balancing": {"mode": "fixed", "fixed": {"age_weight": 1.0, "gender_weight": 1.0}},
        },
    }


def _make_fake_predictor(api_config: dict | None = None) -> Predictor:
    config = _tiny_model_config()
    model = MultiTaskFaceModel(config)
    model.eval()
    artifacts = LoadedArtifacts(
        model=model, model_config=config, checkpoint_name="fake_test_checkpoint.pt",
        checkpoint_epoch=1, calibration=None, knn_baseline=None, warnings=[],
    )
    resolved_api_config = {"model_version": "v1-test", **(api_config or {})}
    return Predictor(artifacts, resolved_api_config, device="cpu")


def _sample_image_bytes() -> bytes:
    rng = np.random.default_rng(0)
    array = rng.integers(0, 255, size=(64, 64, 3), dtype=np.uint8)
    buffer = io.BytesIO()
    Image.fromarray(array).save(buffer, format="PNG")
    return buffer.getvalue()


def test_health_endpoint_returns_ok():
    with TestClient(app) as client:
        response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert "disclaimer" in body
    assert "research" in body["disclaimer"].lower()


def test_predict_endpoint_schema_with_mocked_model():
    with TestClient(app) as client:
        dependencies.app_state.predictor = _make_fake_predictor()
        dependencies.app_state.config = {"api": {"model_version": "v1-test"}}

        response = client.post(
            "/predict", files={"file": ("test.png", _sample_image_bytes(), "image/png")}
        )
    assert response.status_code == 200
    body = response.json()

    assert set(body["age"].keys()) >= {"q10", "q50", "q90", "is_calibrated"}
    assert body["age"]["q10"] <= body["age"]["q50"] <= body["age"]["q90"]

    assert "probabilities" in body["gender"]
    assert body["gender"]["display_label"] in list(body["gender"]["probabilities"].keys()) + ["Not sure"]

    assert "warnings" in body["quality"]
    assert body["model_version"] == "v1-test"
    assert "research and demonstration only" in body["disclaimer"]


def test_predict_reports_no_face_detected_on_synthetic_noise_image():
    """Face detection is enabled by default; a random-noise upload has no real face in it."""
    with TestClient(app) as client:
        dependencies.app_state.predictor = _make_fake_predictor()
        dependencies.app_state.config = {"api": {"model_version": "v1-test"}}

        response = client.post(
            "/predict", files={"file": ("test.png", _sample_image_bytes(), "image/png")}
        )
    assert response.status_code == 200
    body = response.json()
    assert body["face_detected"] is False
    assert any("No face detected" in w for w in body["warnings"])


def test_predict_face_detected_is_null_when_disabled():
    with TestClient(app) as client:
        dependencies.app_state.predictor = _make_fake_predictor({"enable_face_detection": False})
        dependencies.app_state.config = {"api": {"model_version": "v1-test"}}

        response = client.post(
            "/predict", files={"file": ("test.png", _sample_image_bytes(), "image/png")}
        )
    assert response.status_code == 200
    body = response.json()
    assert body["face_detected"] is None
    assert not any("No face detected" in w for w in body["warnings"])


def test_predict_endpoint_returns_503_when_no_model_loaded():
    with TestClient(app) as client:
        empty_artifacts = LoadedArtifacts(model=None, warnings=["no checkpoint"])
        dependencies.app_state.predictor = Predictor(empty_artifacts, {"model_version": "v1"}, device="cpu")

        response = client.post(
            "/predict", files={"file": ("test.png", _sample_image_bytes(), "image/png")}
        )
    assert response.status_code == 503


def test_quality_check_endpoint():
    with TestClient(app) as client:
        response = client.post(
            "/quality-check", files={"file": ("test.png", _sample_image_bytes(), "image/png")}
        )
    assert response.status_code == 200
    body = response.json()
    assert body["width"] == 64
    assert body["height"] == 64
    assert isinstance(body["warnings"], list)


def test_gender_label_override_replaces_both_names():
    """api.gender_label_overrides lets a deployer rename classes without retraining."""
    predictor = _make_fake_predictor({"gender_label_overrides": ["male", "female"]})
    assert predictor.class_names == ["male", "female"]


def test_gender_label_override_is_partial_by_index():
    """A null/missing entry falls back to the checkpoint's own trained class name."""
    predictor = _make_fake_predictor({"gender_label_overrides": ["male", None]})
    assert predictor.class_names == ["male", "gender_label_1"]


def test_no_gender_label_override_keeps_checkpoint_names():
    predictor = _make_fake_predictor()
    assert predictor.class_names == ["gender_label_0", "gender_label_1"]


def test_predict_endpoint_reflects_gender_label_override():
    with TestClient(app) as client:
        dependencies.app_state.predictor = _make_fake_predictor({"gender_label_overrides": ["male", "female"]})
        dependencies.app_state.config = {"api": {"model_version": "v1-test"}}

        response = client.post(
            "/predict", files={"file": ("test.png", _sample_image_bytes(), "image/png")}
        )
    assert response.status_code == 200
    body = response.json()
    assert set(body["gender"]["probabilities"].keys()) == {"male", "female"}
