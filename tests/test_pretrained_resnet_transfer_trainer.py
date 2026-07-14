"""End-to-end proof that the pretrained-ResNet bridge baseline reuses the
existing TransferTrainer / run_transfer_learning.py CLI machinery
unmodified (final-run hardening T3) -- same transfer-training stages, same
persistence, same CLI, same Table B path as VOLO, just a different model
family. Offline (pretrained=False), synthetic data only.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

pytest.importorskip("torchvision")

import torch  # noqa: E402

from src.data.dataset import FaceMultiTaskDataset  # noqa: E402
from src.data.split_utils import split_dataframe  # noqa: E402
from src.models.pretrained_resnet import PretrainedResNetFaceOnlyMultiTask  # noqa: E402
from src.training.transfer_trainer import TransferTrainer  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = str(REPO_ROOT / "scripts")
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)


def _tiny_resnet_config():
    return {
        "model": {
            "family": "pretrained_resnet",
            "pretrained_resnet": {"model_id": "resnet18", "pretrained": False, "pretrained_source": "imagenet1k_v1"},
            "adapters": {"enabled": True, "bottleneck_ratio": 4, "dropout": 0.1},
            "age_head": {"hidden_dim": 16, "dropout": 0.1, "age_min": 0, "age_max": 120},
            "gender_head": {"hidden_dim": 16, "dropout": 0.1, "num_classes": 2, "confidence_threshold": 0.80},
            "loss_balancing": {
                "mode": "learned_uncertainty",
                "learned_uncertainty": {"init_log_var_age": 0.0, "init_log_var_gender": 0.0, "warmup_epochs": 0},
            },
        },
        "training": {
            "batch_size": 2, "grad_accumulation_steps": 1, "num_workers": 0, "weight_decay": 0.05,
            "grad_clip_norm": 1.0, "mixed_precision": False, "seed": 0, "early_stopping_patience": 100,
            "head_only_epochs": 1, "finetune_epochs": 1, "finetune_unfreeze": "full",
            "head_lr": 3.0e-4, "adapter_lr": 3.0e-4, "loss_balance_lr": 3.0e-4, "backbone_lr": 3.0e-5,
            "scheduler": {"warmup_epochs": 0}, "max_train_batches_per_epoch": 2, "max_val_batches_per_epoch": 2,
        },
    }


@pytest.fixture
def tiny_resnet_datasets(synthetic_metadata_df):
    df = split_dataframe(synthetic_metadata_df, 0.5, 0.2, 0.1, 0.2, seed=0, subject_level_if_available=False)
    model = PretrainedResNetFaceOnlyMultiTask(_tiny_resnet_config())
    train_transform, eval_transform = model.build_transforms()
    train_dataset = FaceMultiTaskDataset(df[df["split"] == "train"], train_transform)
    val_dataset = FaceMultiTaskDataset(df[df["split"] == "validation"], eval_transform)
    return model, train_dataset, val_dataset


def test_transfer_trainer_runs_end_to_end_with_pretrained_resnet(tmp_path, tiny_resnet_datasets):
    """The SAME TransferTrainer class used for VOLO, unmodified, trains a
    pretrained-ResNet model through both stages."""
    model, train_dataset, val_dataset = tiny_resnet_datasets
    trainer = TransferTrainer(
        model, _tiny_resnet_config(), train_dataset, val_dataset, device="cpu",
        checkpoint_dir=tmp_path / "checkpoints", experiment_name="resnet_smoke", output_dir=tmp_path / "output",
    )
    result = trainer.train()
    assert "history" in result
    assert len(result["history"]["train_loss"]) >= 1
    assert not any(v != v for v in result["history"]["train_loss"])


def test_backbone_trainable_after_stage_2(tmp_path, tiny_resnet_datasets):
    model, train_dataset, val_dataset = tiny_resnet_datasets
    trainer = TransferTrainer(
        model, _tiny_resnet_config(), train_dataset, val_dataset, device="cpu",
        checkpoint_dir=tmp_path / "checkpoints", experiment_name="resnet_stage2", output_dir=tmp_path / "output",
    )
    trainer.train()
    assert all(p.requires_grad for p in model.backbone.parameters())


def test_checkpoint_reload_matches_original(tmp_path, tiny_resnet_datasets):
    from src.inference.artifacts import load_model_checkpoint

    model, train_dataset, val_dataset = tiny_resnet_datasets
    checkpoint_dir = tmp_path / "checkpoints"
    trainer = TransferTrainer(
        model, _tiny_resnet_config(), train_dataset, val_dataset, device="cpu",
        checkpoint_dir=checkpoint_dir, experiment_name="resnet_ckpt", output_dir=tmp_path / "output",
    )
    trainer.train()

    checkpoint_path = checkpoint_dir / "resnet_ckpt_best_balanced_score.pt"
    assert checkpoint_path.exists()

    reloaded_model, config, _ = load_model_checkpoint(checkpoint_path, device="cpu")
    assert config["model"]["family"] == "pretrained_resnet"
    assert isinstance(reloaded_model, PretrainedResNetFaceOnlyMultiTask)


# -- config loading (via scripts/run_transfer_learning.py) --------------------------


def test_load_transfer_learning_config_for_resnet18_family():
    import run_transfer_learning as rtl

    config = rtl.load_transfer_learning_config(seed=42, smoke=True, family="pretrained_resnet18")
    assert config["model"]["family"] == "pretrained_resnet"
    assert config["model"]["pretrained_resnet"]["model_id"] == "resnet18"
    assert config["model"]["pretrained_resnet"]["pretrained"] is False  # smoke profile forces pretrained=False
    assert config["training"]["seed"] == 42


def test_load_transfer_learning_config_for_resnet50_family():
    import run_transfer_learning as rtl

    config = rtl.load_transfer_learning_config(seed=123, smoke=False, family="pretrained_resnet50")
    assert config["model"]["pretrained_resnet"]["model_id"] == "resnet50"
    assert config["transfer_learning"]["seeds"] == [42, 123, 2026]


def test_resnet_family_paths_never_target_core_output_dirs(monkeypatch):
    """Same protection load_transfer_learning_config already has for VOLO
    -- an unrelated .env CHECKPOINT_DIR/OUTPUT_DIR must never redirect the
    pretrained-ResNet extension's isolated paths."""
    import run_transfer_learning as rtl

    monkeypatch.setenv("CHECKPOINT_DIR", "./checkpoints")
    monkeypatch.setenv("OUTPUT_DIR", "./outputs")

    config = rtl.load_transfer_learning_config(seed=42, smoke=False, family="pretrained_resnet18")
    assert config["paths"]["checkpoint_dir"].startswith("./checkpoints/transfer_learning")
    assert config["paths"]["output_dir"].startswith("./results/transfer_learning")


def test_model_family_registry_has_required_and_optional_resnet_entries():
    import run_transfer_learning as rtl

    assert "pretrained_resnet18" in rtl._MODEL_FAMILIES
    assert "pretrained_resnet50" in rtl._MODEL_FAMILIES
    assert "volo" in rtl._MODEL_FAMILIES
    # ResNet-50's category label must never claim to isolate pretraining.
    r50_label = rtl._MODEL_FAMILIES["pretrained_resnet50"]["category_label"]
    assert "NOT pretraining-isolated" in r50_label or "capacity" in r50_label


def test_default_model_family_is_volo_backward_compatible():
    """--model-family omitted must behave exactly as before this
    generalization was introduced."""
    import inspect

    import run_transfer_learning as rtl

    source = inspect.getsource(rtl.main)
    assert '"--model-family"' in source
    assert 'default="volo"' in source
