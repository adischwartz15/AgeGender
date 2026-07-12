"""Regression tests: adding the transfer-learning extension must not change
any existing config, run profile, or Table A behaviour."""

from __future__ import annotations

import pytest

from src.utils.config import CONFIG_DIR, load_config, load_full_config


def test_default_profile_unchanged_by_transfer_learning_extension():
    """load_full_config() (the core/default profile) must not reference
    volo/timm/pretrained_volo anywhere -- VOLO is never added to the
    default profile."""
    config = load_full_config()
    assert config["model"].get("family", "core") == "core"
    assert "volo" not in config["model"]
    assert config["model"]["architecture"] == "shared_adapters"
    assert config["model"]["backbone"]["name"] == "custom_resnet18"
    assert config["model"]["adapters"]["bottleneck_dim"] == 256  # unchanged from configs/model.yaml


def test_experiments_yaml_run_order_unchanged():
    """configs/experiments.yaml (the core ablation suite) must not reference
    the transfer-learning extension at all."""
    experiments_cfg = load_config(CONFIG_DIR / "experiments.yaml")
    assert "transfer_learning" not in experiments_cfg
    assert "volo_d1_face_only_pretrained" not in experiments_cfg["run_order"]
    for name in experiments_cfg["run_order"]:
        assert "volo" not in name


def test_backbone_comparison_backbones_unchanged():
    """The 3 controlled backbones (custom_resnet18/simple_cnn/plain_deep18_no_skip)
    remain the only entries in the backbone factory registry."""
    from src.models.backbone_factory import _BACKBONE_BUILDERS

    assert set(_BACKBONE_BUILDERS) == {"custom_resnet18", "simple_cnn", "plain_deep18_no_skip"}


def test_transfer_learning_yaml_does_not_alter_experiments_yaml_defaults():
    """Merging configs/transfer_learning.yaml on top of the standard stack
    must not change any core model/training default -- only add the new
    model.family/model.volo keys and override paths/training for this
    profile specifically."""
    core_config = load_full_config()
    tl_config = load_config(
        CONFIG_DIR / "data.yaml", CONFIG_DIR / "model.yaml", CONFIG_DIR / "training.yaml",
        CONFIG_DIR / "transfer_learning.yaml",
    )
    # Core age/gender range and adapter dropout are untouched by the
    # transfer_learning.yaml override (only bottleneck_ratio/dropout for
    # THIS profile's own model.adapters block change, not the core one --
    # this asserts the *core* config loaded independently is unaffected).
    assert core_config["model"]["age_head"]["age_max"] == 120
    assert core_config["dataset"]["image_size"] == 128
    # And confirm the transfer-learning profile really does declare its own family/volo block.
    assert tl_config["model"]["family"] == "pretrained_volo"
    assert "volo" in tl_config["model"]


def test_pretrained_volo_not_registered_in_backbone_factory():
    """VOLO must not be reachable via model.backbone.name -- it's a
    separate model class (PretrainedVOLOFaceOnlyMultiTask), not a
    backbone_factory entry, so no core config value can accidentally
    select it."""
    from src.models.backbone_factory import build_backbone

    with pytest.raises(ValueError):
        build_backbone({"name": "volo_d1_224"})
