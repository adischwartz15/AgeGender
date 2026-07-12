"""Locates and loads trained model artifacts (checkpoint, calibration, kNN index).

Centralizes "does this artifact exist, and if not what should the caller
tell the user" logic so the API and evaluation scripts report consistent,
honest warnings instead of silently pretending an artifact is available.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import torch.nn as nn

from src.evaluation.calibration import load_calibration
from src.evaluation.knn_baseline import KNNEmbeddingBaseline
from src.models.multitask_model import MultiTaskFaceModel
from src.training.checkpointing import load_checkpoint

logger = logging.getLogger(__name__)


@dataclass
class LoadedArtifacts:
    model: MultiTaskFaceModel | None = None
    model_config: dict | None = None
    checkpoint_name: str | None = None
    checkpoint_epoch: int | None = None
    calibration: dict | None = None
    knn_baseline: KNNEmbeddingBaseline | None = None
    warnings: list[str] = field(default_factory=list)


def load_model_checkpoint(checkpoint_path: str | Path, device: str = "cpu") -> tuple[nn.Module, dict, dict]:
    """Load a model from a checkpoint produced by this repository's trainer.

    ``config["model"]["family"]`` selects the model class to reconstruct:
    ``"core"`` (the default -- every existing checkpoint/config lacks this
    key, so it always resolves to the original ``MultiTaskFaceModel`` path
    unchanged) or ``"pretrained_volo"`` (the supplementary transfer-learning
    extension's ``PretrainedVOLOFaceOnlyMultiTask``, see
    ``src/models/pretrained_volo.py``).
    """
    checkpoint = load_checkpoint(checkpoint_path, map_location=device)
    config = checkpoint["config"]
    family = config["model"].get("family", "core")
    if family == "pretrained_volo":
        from src.models.pretrained_volo import build_pretrained_volo_model

        model = build_pretrained_volo_model(config)
    elif family == "core":
        model = MultiTaskFaceModel(config)
    else:
        raise ValueError(f"Unknown model.family '{family}', expected 'core' or 'pretrained_volo'")
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model, config, checkpoint


def load_all_artifacts(api_config: dict, device: str = "cpu") -> LoadedArtifacts:
    """Load the active checkpoint, calibration artifact, and kNN index, warning on gaps."""
    result = LoadedArtifacts()

    checkpoint_path = Path(api_config["active_checkpoint"])
    if not checkpoint_path.exists():
        msg = (
            f"No trained checkpoint found at '{checkpoint_path}'. Predictions are unavailable "
            "until you run training (see 'make train' or 'make experiments') and point "
            "api.active_checkpoint at a produced checkpoint."
        )
        logger.warning(msg)
        result.warnings.append(msg)
        return result

    model, config, checkpoint = load_model_checkpoint(checkpoint_path, device)
    result.model = model
    result.model_config = config
    result.checkpoint_name = checkpoint_path.name
    result.checkpoint_epoch = checkpoint.get("epoch")

    calibration_dir = Path(api_config["calibration_dir"])
    calibration = load_calibration(calibration_dir)
    if calibration is None:
        result.warnings.append(
            "No conformal calibration artifact found; age intervals are uncalibrated. "
            "Run 'make calibrate' to generate one."
        )
    result.calibration = calibration

    knn_dir = Path(api_config["knn_index_dir"])
    knn_path = knn_dir / "knn_baseline.pkl"
    if knn_path.exists():
        result.knn_baseline = KNNEmbeddingBaseline.load(knn_path)
    else:
        result.warnings.append(
            "No k-NN index found; parametric-vs-kNN comparison is unavailable. "
            "Run 'make build-knn' to generate one."
        )

    return result
