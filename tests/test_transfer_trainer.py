"""Tests for the two-stage VOLO transfer-learning trainer.

Requires a real (offline, ``pretrained=False``) ``timm`` install -- skipped
entirely where the optional ``requirements-transfer.txt`` extra isn't
installed. Uses the same synthetic (non-real) image fixtures as
``tests/test_smoke_training.py``, never real Kaggle data.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("timm")

import torch  # noqa: E402

from src.data.dataset import FaceMultiTaskDataset  # noqa: E402
from src.data.split_utils import split_dataframe  # noqa: E402
from src.models.pretrained_volo import PretrainedVOLOFaceOnlyMultiTask  # noqa: E402
from src.training.transfer_trainer import TransferTrainer  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]


def _tiny_transfer_config() -> dict:
    return {
        "model": {
            "family": "pretrained_volo",
            "volo": {"model_id": "volo_d1_224", "pretrained": False, "pretrained_source": "imagenet1k"},
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
def tiny_transfer_datasets(synthetic_metadata_df):
    df = split_dataframe(synthetic_metadata_df, 0.5, 0.2, 0.1, 0.2, seed=0, subject_level_if_available=False)
    model = PretrainedVOLOFaceOnlyMultiTask(_tiny_transfer_config())
    train_transform, eval_transform = model.build_transforms()
    train_dataset = FaceMultiTaskDataset(df[df["split"] == "train"], train_transform)
    val_dataset = FaceMultiTaskDataset(df[df["split"] == "validation"], eval_transform)
    return model, train_dataset, val_dataset


def test_one_training_step_completes(tmp_path, tiny_transfer_datasets):
    model, train_dataset, val_dataset = tiny_transfer_datasets
    trainer = TransferTrainer(
        model, _tiny_transfer_config(), train_dataset, val_dataset, device="cpu",
        checkpoint_dir=tmp_path / "checkpoints", experiment_name="volo_smoke", output_dir=tmp_path / "output",
    )
    result = trainer.train()
    assert "history" in result
    assert len(result["history"]["train_loss"]) >= 1
    assert not any(v != v for v in result["history"]["train_loss"])  # no NaNs


def test_loss_balancing_params_receive_gradients(tiny_transfer_datasets):
    model, train_dataset, _ = tiny_transfer_datasets
    from torch.utils.data import DataLoader

    from src.losses.multitask_loss import compute_multitask_loss

    loader = DataLoader(train_dataset, batch_size=2, shuffle=False)
    batch = next(iter(loader))
    outputs = model(batch["image"])
    loss_out = compute_multitask_loss(
        outputs["age_output"], outputs["gender_logits"], batch["age"], batch["age_mask"],
        batch["gender_label"], batch["gender_mask"], mode="learned_uncertainty",
        log_var_age=model.log_var_age, log_var_gender=model.log_var_gender,
    )
    loss_out.total_loss.backward()
    assert model.log_var_age.grad is not None
    assert model.log_var_gender.grad is not None


def test_backbone_params_trainable_after_stage_2(tmp_path, tiny_transfer_datasets):
    model, train_dataset, val_dataset = tiny_transfer_datasets
    trainer = TransferTrainer(
        model, _tiny_transfer_config(), train_dataset, val_dataset, device="cpu",
        checkpoint_dir=tmp_path / "checkpoints", experiment_name="volo_smoke_stage2", output_dir=tmp_path / "output",
    )
    trainer.train()
    # After Stage 2 (finetune_unfreeze="full"), the backbone must have been
    # unfrozen at some point -- verified via requires_grad state right now
    # (Stage 2 is the last stage TransferTrainer.train() runs).
    assert all(p.requires_grad for p in model.backbone.parameters())


def test_optimizer_groups_carry_intended_lrs(tmp_path, tiny_transfer_datasets):
    model, train_dataset, val_dataset = tiny_transfer_datasets
    trainer = TransferTrainer(
        model, _tiny_transfer_config(), train_dataset, val_dataset, device="cpu",
        checkpoint_dir=tmp_path / "checkpoints", experiment_name="volo_lr_groups", output_dir=tmp_path / "output",
    )
    model.freeze_backbone()
    optimizer = trainer._build_optimizer(backbone_lr=3e-5, adapter_lr=3e-4, head_lr=3e-4, balance_lr=3e-4)
    lrs = {g["lr"] for g in optimizer.param_groups}
    assert 3e-5 not in lrs  # backbone frozen -> no backbone group
    assert 3e-4 in lrs

    model.unfreeze_backbone()
    optimizer2 = trainer._build_optimizer(backbone_lr=3e-5, adapter_lr=3e-4, head_lr=3e-4, balance_lr=3e-4)
    assert 3e-5 in {g["lr"] for g in optimizer2.param_groups}


def test_checkpoint_save_load_reproduces_outputs_within_tolerance(tmp_path, tiny_transfer_datasets):
    from src.inference.artifacts import load_model_checkpoint

    model, train_dataset, val_dataset = tiny_transfer_datasets
    checkpoint_dir = tmp_path / "checkpoints"
    trainer = TransferTrainer(
        model, _tiny_transfer_config(), train_dataset, val_dataset, device="cpu",
        checkpoint_dir=checkpoint_dir, experiment_name="volo_ckpt", output_dir=tmp_path / "output",
    )
    trainer.train()

    checkpoint_path = checkpoint_dir / "volo_ckpt_best_balanced_score.pt"
    assert checkpoint_path.exists()

    reloaded_model, config, _ = load_model_checkpoint(checkpoint_path, device="cpu")
    assert config["model"]["family"] == "pretrained_volo"

    model.eval()
    reloaded_model.eval()
    dummy = torch.zeros(2, 3, model.input_size, model.input_size)
    with torch.no_grad():
        out_original = model(dummy)
        out_reloaded = reloaded_model(dummy)
    assert torch.allclose(out_original["age_output"]["q50"], out_reloaded["age_output"]["q50"], atol=1e-5)
    assert torch.allclose(out_original["gender_logits"], out_reloaded["gender_logits"], atol=1e-5)


def test_transfer_learning_smoke_config_runs_end_to_end(tmp_path, synthetic_metadata_df):
    """Exercises the *real* configs/transfer_learning.yaml
    transfer_learning_smoke profile (not a hand-rolled test dict), so a
    typo in that YAML block would fail this test -- builds the model,
    datasets, and TransferTrainer straight from it, trains both stages,
    and verifies a checkpoint round-trips through save/load. Explicitly
    non-scientific (synthetic data, pretrained=False, 2 batches/epoch)."""
    import sys as _sys

    scripts_dir = str(REPO_ROOT / "scripts")
    if scripts_dir not in _sys.path:
        _sys.path.insert(0, scripts_dir)
    from run_transfer_learning import load_transfer_learning_config

    from src.inference.artifacts import load_model_checkpoint

    checkpoint_dir = tmp_path / "checkpoints"
    output_dir = tmp_path / "output"
    config = load_transfer_learning_config(
        seed=0, smoke=True,
        overrides={"paths": {"checkpoint_dir": str(checkpoint_dir), "output_dir": str(output_dir)}},
    )
    assert config["training"]["max_train_batches_per_epoch"] == 2  # sanity: really the smoke profile
    assert config["model"]["volo"]["pretrained"] is False

    df = split_dataframe(synthetic_metadata_df, 0.5, 0.2, 0.1, 0.2, seed=0, subject_level_if_available=False)
    model = PretrainedVOLOFaceOnlyMultiTask(config)
    train_transform, eval_transform = model.build_transforms()
    train_dataset = FaceMultiTaskDataset(df[df["split"] == "train"], train_transform)
    val_dataset = FaceMultiTaskDataset(df[df["split"] == "validation"], eval_transform)

    trainer = TransferTrainer(
        model, config, train_dataset, val_dataset, device="cpu",
        checkpoint_dir=checkpoint_dir, experiment_name="volo_d1_face_only_pretrained_smoke", output_dir=output_dir,
    )
    result = trainer.train()
    assert not any(v != v for v in result["history"]["train_loss"])

    checkpoint_path = checkpoint_dir / "volo_d1_face_only_pretrained_smoke_best_balanced_score.pt"
    assert checkpoint_path.exists()
    reloaded_model, reloaded_config, _ = load_model_checkpoint(checkpoint_path, device="cpu")
    assert reloaded_config["model"]["family"] == "pretrained_volo"
    assert isinstance(reloaded_model, PretrainedVOLOFaceOnlyMultiTask)


def test_transfer_learning_paths_never_target_core_output_dirs(monkeypatch):
    """load_transfer_learning_config()'s checkpoint_dir/output_dir must
    always resolve under the isolated transfer_learning subdirectories,
    never the bare checkpoints//outputs/ dirs the core from-scratch
    experiments write to -- even when a real .env sets CHECKPOINT_DIR/
    OUTPUT_DIR for the core profile (see run_transfer_learning.py's
    load_transfer_learning_config docstring for why this can otherwise
    leak through load_config()'s documented .env precedence)."""
    monkeypatch.setenv("CHECKPOINT_DIR", "./checkpoints")
    monkeypatch.setenv("OUTPUT_DIR", "./outputs")

    import sys as _sys

    scripts_dir = str(REPO_ROOT / "scripts")
    if scripts_dir not in _sys.path:
        _sys.path.insert(0, scripts_dir)
    from run_transfer_learning import load_transfer_learning_config

    config = load_transfer_learning_config(seed=42, smoke=False)
    assert config["paths"]["checkpoint_dir"].startswith("./checkpoints/transfer_learning")
    assert config["paths"]["output_dir"].startswith("./results/transfer_learning")

    smoke_config = load_transfer_learning_config(seed=42, smoke=True)
    assert smoke_config["paths"]["checkpoint_dir"].startswith("./checkpoints/transfer_learning")
    assert smoke_config["paths"]["output_dir"].startswith("./results/transfer_learning")
