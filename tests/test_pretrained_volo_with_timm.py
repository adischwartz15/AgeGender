"""Tests for the VOLO transfer-learning extension that require a real,
offline (``pretrained=False``, no network access) ``timm`` install.

Skipped entirely (not failed) wherever the optional
``requirements-transfer.txt`` extra isn't installed -- see
``tests/test_pretrained_volo.py`` for the tests that must pass regardless.
"""

from __future__ import annotations

import pytest
import torch

pytest.importorskip("timm")

from src.models.pretrained_volo import InvalidStageTransitionError, PretrainedVOLOFaceOnlyMultiTask  # noqa: E402


def _tiny_volo_config(pretrained: bool = False) -> dict:
    return {
        "model": {
            "family": "pretrained_volo",
            "volo": {"model_id": "volo_d1_224", "pretrained": pretrained, "pretrained_source": "imagenet1k"},
            "adapters": {"enabled": True, "bottleneck_ratio": 4, "dropout": 0.1},
            "age_head": {"hidden_dim": 16, "dropout": 0.1, "age_min": 0, "age_max": 120},
            "gender_head": {"hidden_dim": 16, "dropout": 0.1, "num_classes": 2},
            "loss_balancing": {
                "mode": "learned_uncertainty",
                "learned_uncertainty": {"init_log_var_age": 0.0, "init_log_var_gender": 0.0},
            },
        }
    }


@pytest.fixture(scope="module")
def volo_model():
    return PretrainedVOLOFaceOnlyMultiTask(_tiny_volo_config(pretrained=False))


def test_volo_wrapper_builds_with_pretrained_false(volo_model):
    assert volo_model.pretrained is False
    assert volo_model.embedding_dim > 0
    assert volo_model.embedding_dim != 512  # must not silently inherit the ResNet-18 assumption


def test_backbone_feature_dim_discovered_dynamically_and_pooled_shape_is_2d(volo_model):
    dummy = torch.zeros(2, 3, volo_model.input_size, volo_model.input_size)
    with torch.no_grad():
        pooled = volo_model.backbone(dummy)
    assert pooled.shape == (2, volo_model.embedding_dim)


def test_dummy_batch_passes_through_full_model(volo_model):
    dummy = torch.randn(2, 3, volo_model.input_size, volo_model.input_size)
    out = volo_model(dummy)
    assert out["age_output"]["q50"].shape == (2,)
    assert out["gender_logits"].shape == (2, 2)


def test_output_shapes_match_existing_trainer_contract(volo_model):
    dummy = torch.randn(2, 3, volo_model.input_size, volo_model.input_size)
    out = volo_model(dummy)
    for key in ("shared_embedding", "age_embedding", "gender_embedding", "age_output", "gender_logits"):
        assert key in out
    for key in ("q10", "q50", "q90", "q10_raw", "q50_raw", "q90_raw"):
        assert key in out["age_output"]


def test_adapters_receive_the_discovered_embedding_dim(volo_model):
    assert volo_model.age_adapter.input_dim == volo_model.embedding_dim
    assert volo_model.gender_adapter.input_dim == volo_model.embedding_dim
    expected_bottleneck = round(volo_model.embedding_dim / volo_model.bottleneck_ratio)
    assert volo_model.age_adapter.bottleneck_dim == expected_bottleneck
    # Adapters stay a small fraction of the backbone -- same invariant
    # tests/test_adapters.py already checks for the from-scratch adapters.
    assert volo_model.age_adapter.num_parameters() < sum(p.numel() for p in volo_model.backbone.parameters()) * 0.05


def test_freeze_backbone_keeps_adapters_and_heads_trainable(volo_model):
    volo_model.freeze_backbone()
    assert all(not p.requires_grad for p in volo_model.backbone.parameters())
    assert all(p.requires_grad for p in volo_model.age_adapter.parameters())
    assert all(p.requires_grad for p in volo_model.gender_adapter.parameters())
    assert all(p.requires_grad for p in volo_model.age_head.parameters())
    assert all(p.requires_grad for p in volo_model.gender_head.parameters())
    volo_model.unfreeze_backbone()  # restore for other tests in this module


def test_unfreeze_backbone_makes_all_backbone_params_trainable(volo_model):
    volo_model.freeze_backbone()
    volo_model.unfreeze_backbone()
    assert all(p.requires_grad for p in volo_model.backbone.parameters())


def test_unfreeze_last_stages_rejects_invalid_n(volo_model):
    with pytest.raises(InvalidStageTransitionError):
        volo_model.unfreeze_last_stages(0)
    with pytest.raises(InvalidStageTransitionError):
        volo_model.unfreeze_last_stages(10_000)
    volo_model.unfreeze_backbone()  # restore


def test_unfreeze_last_stages_only_unfreezes_requested_stages(volo_model):
    volo_model.unfreeze_last_stages(1)
    network = volo_model.backbone.network
    # Every stage except the last must still be frozen.
    for stage in list(network)[:-1]:
        assert all(not p.requires_grad for p in stage.parameters())
    assert all(p.requires_grad for p in list(network)[-1].parameters())
    volo_model.unfreeze_backbone()  # restore


def test_get_parameter_groups_carries_intended_lrs_and_omits_frozen_backbone(volo_model):
    volo_model.freeze_backbone()
    groups = volo_model.get_parameter_groups(
        backbone_lr=3e-5, adapter_lr=3e-4, head_lr=3e-4, balance_lr=3e-4, weight_decay=0.05,
    )
    lrs = {g["lr"] for g in groups}
    assert 3e-5 not in lrs  # backbone group must be omitted entirely while frozen
    assert 3e-4 in lrs

    volo_model.unfreeze_backbone()
    groups_unfrozen = volo_model.get_parameter_groups(
        backbone_lr=3e-5, adapter_lr=3e-4, head_lr=3e-4, balance_lr=3e-4, weight_decay=0.05,
    )
    assert 3e-5 in {g["lr"] for g in groups_unfrozen}


def test_parameter_breakdown_reports_trainable_vs_total(volo_model):
    volo_model.freeze_backbone()
    breakdown = volo_model.parameter_breakdown()
    assert breakdown.trainable_total < breakdown.total
    volo_model.unfreeze_backbone()


def test_build_transforms_uses_backbones_own_resolved_config(volo_model):
    _, eval_transform = volo_model.build_transforms()
    assert eval_transform.image_size == volo_model.input_size
    assert tuple(eval_transform.mean) == tuple(volo_model.data_config["mean"])
    assert tuple(eval_transform.std) == tuple(volo_model.data_config["std"])
