"""End-to-end inference: image -> quality diagnostics + predictions + explanations.

This is the single place that turns a raw uploaded image into everything
the API response needs: age quantiles (raw and conformal-calibrated),
dataset gender-label probabilities with "Not sure" abstention, image
quality diagnostics, optional Grad-CAM attention maps, and an optional
non-parametric k-NN comparison.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import torch
from PIL import Image

from src.data.transforms import EvalTransform, resolve_eval_transform
from src.evaluation.calibration import apply_conformal_offset
from src.evaluation.gradcam import GradCAM, resize_heatmap
from src.inference.artifacts import LoadedArtifacts
from src.inference.face_detection import crop_to_face
from src.inference.quality import QualityDiagnostics, compute_quality_diagnostics


@dataclass
class AgePrediction:
    q10: float
    q50: float
    q90: float
    q10_calibrated: float | None
    q90_calibrated: float | None
    is_calibrated: bool


@dataclass
class GenderPrediction:
    probabilities: dict[str, float]
    predicted_label: str | None
    confidence: float
    abstained: bool


@dataclass
class KNNComparison:
    age_q10: float
    age_q50: float
    age_q90: float
    gender_probabilities: dict[str, float]
    gender_predicted_label: str | None
    gender_abstained: bool
    mean_neighbor_distance: float


@dataclass
class PredictionResult:
    age: AgePrediction | None
    gender: GenderPrediction | None
    quality: QualityDiagnostics
    gradcam_age: np.ndarray | None
    gradcam_gender: np.ndarray | None
    knn: KNNComparison | None
    model_version: str
    checkpoint_name: str | None
    face_detected: bool | None = None
    model_input_image: Image.Image | None = field(default=None, repr=False)
    warnings: list[str] = field(default_factory=list)
    latency_ms: float = 0.0


class Predictor:
    """Wraps a loaded model + calibration + kNN index for single-image inference."""

    def __init__(self, artifacts: LoadedArtifacts, api_config: dict, device: str = "cpu") -> None:
        self.artifacts = artifacts
        self.api_config = api_config
        self.device = device
        self.model = artifacts.model
        self.config = artifacts.model_config
        self.enable_face_detection: bool = api_config.get("enable_face_detection", True)
        self.face_margin_ratio: float = api_config.get("face_margin_ratio", 0.35)

        if self.model is not None:
            gender_head_cfg = self.config["model"]["gender_head"]
            self.class_names: list[str] = gender_head_cfg.get(
                "class_names", ["gender_label_0", "gender_label_1"]
            )
            # Display-name override, independent of whatever was baked into
            # the checkpoint's training-time config -- lets a deployer
            # rename the two classes (e.g. to a dataset's own documented
            # convention) without retraining. Never changes which learned
            # class index is which; only the label shown for it. See
            # GENDER_LABEL_0 / GENDER_LABEL_1 in .env.example.
            overrides = api_config.get("gender_label_overrides")
            if overrides:
                self.class_names = [
                    overrides[i] if i < len(overrides) and overrides[i] else name
                    for i, name in enumerate(self.class_names)
                ]
            self.confidence_threshold: float = gender_head_cfg.get("confidence_threshold", 0.80)
            # Model-aware preprocessing (see
            # src/data/transforms.py::resolve_eval_transform) -- a VOLO/
            # pretrained-ResNet checkpoint's own resolved transform, never
            # this project's 128px/IMAGENET-constant default for such a model.
            self.transform = resolve_eval_transform(self.model, self.config)
            self.gradcam = GradCAM(self.model, self.config["gradcam"]["target_layer"]) if "gradcam" in self.config else GradCAM(self.model)
        else:
            self.class_names = ["gender_label_0", "gender_label_1"]
            self.confidence_threshold = 0.80
            self.transform = EvalTransform(128)
            self.gradcam = None

    def is_ready(self) -> bool:
        return self.model is not None

    def predict(
        self, image: Image.Image, include_gradcam: bool = False, include_knn: bool = False
    ) -> PredictionResult:
        start = time.time()
        warnings = list(self.artifacts.warnings)

        quality = compute_quality_diagnostics(image, image.format or "unknown", _estimate_size_bytes(image))

        if self.model is None:
            raise RuntimeError(
                "No trained model checkpoint is loaded; cannot run predictions. "
                "See the warnings field for how to produce one."
            )

        image_rgb = image.convert("RGB")
        face_detected: bool | None = None
        model_input_image = image_rgb
        if self.enable_face_detection:
            model_input_image, face_detected = crop_to_face(image_rgb, self.face_margin_ratio)
            if not face_detected:
                warnings.append(
                    "No face detected via classical Haar-cascade detection; declining to "
                    "generate age or dataset gender-label predictions, since the model is "
                    "only meaningful on face images similar to its training data."
                )
                return PredictionResult(
                    age=None, gender=None, quality=quality,
                    gradcam_age=None, gradcam_gender=None, knn=None,
                    model_version=self.api_config.get("model_version", "v1"),
                    checkpoint_name=self.artifacts.checkpoint_name,
                    face_detected=False, model_input_image=None,
                    warnings=warnings, latency_ms=(time.time() - start) * 1000.0,
                )

        image_tensor = self.transform(model_input_image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = self.model(image_tensor)
            probs = torch.softmax(outputs["gender_logits"], dim=-1)[0].cpu().numpy()

        q10 = float(outputs["age_output"]["q10"][0].item())
        q50 = float(outputs["age_output"]["q50"][0].item())
        q90 = float(outputs["age_output"]["q90"][0].item())

        calibration = self.artifacts.calibration
        q10_cal, q90_cal, is_calibrated = None, None, False
        if calibration is not None:
            offset = calibration["offset"]
            q10_cal, q90_cal = apply_conformal_offset(np.array([q10]), np.array([q90]), offset)
            q10_cal, q90_cal = float(q10_cal[0]), float(q90_cal[0])
            is_calibrated = True
        else:
            warnings.append("No calibration artifact available; interval is uncalibrated.")

        gender_pred = self._build_gender_prediction(probs)

        gradcam_age, gradcam_gender = None, None
        if include_gradcam and self.gradcam is not None:
            age_result = self.gradcam.generate(image_tensor.clone(), task="age")
            gender_result = self.gradcam.generate(image_tensor.clone(), task="gender")
            # Overlay onto model_input_image (the face crop actually fed to the
            # model), not the raw upload -- resizing the heatmap to the raw
            # upload's dimensions would misalign it whenever a face crop
            # changed the aspect ratio/region relative to the original.
            size = model_input_image.size
            gradcam_age = resize_heatmap(age_result["heatmap"], size)
            gradcam_gender = resize_heatmap(gender_result["heatmap"], size)

        knn_result = None
        if include_knn:
            if self.artifacts.knn_baseline is not None:
                knn_result = self._build_knn_comparison(image_tensor)
            else:
                warnings.append("k-NN index not available; comparison skipped.")

        latency_ms = (time.time() - start) * 1000.0

        return PredictionResult(
            age=AgePrediction(
                q10=q10, q50=q50, q90=q90,
                q10_calibrated=q10_cal, q90_calibrated=q90_cal, is_calibrated=is_calibrated,
            ),
            gender=gender_pred,
            quality=quality,
            gradcam_age=gradcam_age,
            gradcam_gender=gradcam_gender,
            knn=knn_result,
            model_version=self.api_config.get("model_version", "v1"),
            checkpoint_name=self.artifacts.checkpoint_name,
            face_detected=face_detected,
            model_input_image=model_input_image,
            warnings=warnings,
            latency_ms=latency_ms,
        )

    def _build_gender_prediction(self, probs: np.ndarray) -> GenderPrediction:
        prob_dict = {name: float(p) for name, p in zip(self.class_names, probs)}
        confidence = float(probs.max())
        predicted_idx = int(probs.argmax())
        abstained = confidence < self.confidence_threshold
        predicted_label = None if abstained else self.class_names[predicted_idx]
        return GenderPrediction(
            probabilities=prob_dict, predicted_label=predicted_label,
            confidence=confidence, abstained=abstained,
        )

    def _build_knn_comparison(self, image_tensor: torch.Tensor) -> KNNComparison:
        with torch.no_grad():
            embeddings = self.model.encode(image_tensor)
        age_embedding = embeddings["age_embedding"].cpu().numpy()
        gender_embedding = embeddings["gender_embedding"].cpu().numpy()

        knn = self.artifacts.knn_baseline
        age_result = knn.predict_age(age_embedding)
        gender_result = knn.predict_gender(gender_embedding, self.confidence_threshold)

        prob_dict = {name: float(p) for name, p in zip(self.class_names, gender_result.probabilities[0])}
        predicted_label = None if gender_result.abstain[0] else self.class_names[gender_result.predicted_class[0]]

        return KNNComparison(
            age_q10=float(age_result.q10[0]), age_q50=float(age_result.q50[0]), age_q90=float(age_result.q90[0]),
            gender_probabilities=prob_dict, gender_predicted_label=predicted_label,
            gender_abstained=bool(gender_result.abstain[0]), mean_neighbor_distance=float(age_result.mean_distance[0]),
        )


def _estimate_size_bytes(image: Image.Image) -> int:
    import io

    buffer = io.BytesIO()
    fmt = image.format or "PNG"
    try:
        image.save(buffer, format=fmt)
    except (ValueError, OSError, KeyError):
        image.save(buffer, format="PNG")
    return buffer.tell()
