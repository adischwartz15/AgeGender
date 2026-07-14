"""Locates and loads trained model artifacts (checkpoint, calibration, kNN index).

Centralizes "does this artifact exist, and if not what should the caller
tell the user" logic so the API and evaluation scripts report consistent,
honest warnings instead of silently pretending an artifact is available.
"""

from __future__ import annotations

import copy
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


def _construct_model_offline(family: str, saved_config: dict) -> nn.Module:
    """Reconstruct a model architecture for loading a **local** checkpoint's
    weights -- never triggers a pretrained-weight download, regardless of
    what the saved config's own ``pretrained: true`` flag says, since the
    real (fine-tuned) weights are loaded from ``checkpoint["model_state_dict"]``
    immediately after this returns; downloading fresh ImageNet weights first
    would be wasted work at best and an offline failure at worst.

    Deep-copies ``saved_config`` before forcing any reconstruction-only
    ``pretrained`` flag to ``False``, so the caller's returned provenance
    config (the original, unmutated ``saved_config``) still faithfully
    records what the checkpoint was actually *trained* with.
    """
    if family == "pretrained_volo":
        from src.models.pretrained_volo import build_pretrained_volo_model

        construction_config = copy.deepcopy(saved_config)
        construction_config["model"]["volo"]["pretrained"] = False
        return build_pretrained_volo_model(construction_config)
    if family == "pretrained_resnet":
        from src.models.pretrained_resnet import build_pretrained_resnet_model

        construction_config = copy.deepcopy(saved_config)
        construction_config["model"]["pretrained_resnet"]["pretrained"] = False
        return build_pretrained_resnet_model(construction_config)
    if family == "core":
        return MultiTaskFaceModel(saved_config)
    raise ValueError(
        f"Unknown model.family '{family}', expected 'core', 'pretrained_volo', or 'pretrained_resnet'"
    )


def load_model_checkpoint(checkpoint_path: str | Path, device: str = "cpu") -> tuple[nn.Module, dict, dict]:
    """Load a model from a checkpoint produced by this repository's trainer.

    ``config["model"]["family"]`` selects the model class to reconstruct:
    ``"core"`` (the default -- every existing checkpoint/config lacks this
    key, so it always resolves to the original ``MultiTaskFaceModel`` path
    unchanged), ``"pretrained_volo"`` (the supplementary transfer-learning
    extension's ``PretrainedVOLOFaceOnlyMultiTask``, see
    ``src/models/pretrained_volo.py``), or ``"pretrained_resnet"`` (the
    pretrained-torchvision-ResNet bridge baseline, see
    ``src/models/pretrained_resnet.py``).

    Reconstruction never re-downloads pretrained weights (see
    :func:`_construct_model_offline`) -- the returned ``config`` is the
    checkpoint's original, unmutated saved config (i.e. still records
    ``pretrained: true`` if that's what training actually used), not the
    ``pretrained: false`` reconstruction-only config used internally to
    build the architecture before loading its real weights.
    """
    checkpoint = load_checkpoint(checkpoint_path, map_location=device)
    saved_config = checkpoint["config"]
    family = saved_config["model"].get("family", "core")
    model = _construct_model_offline(family, saved_config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model, saved_config, checkpoint


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
