"""Tests for the two-stage VOLO transfer-learning trainer.

Requires a real (offline, ``pretrained=False``) ``timm`` install -- skipped
entirely where the optional ``requirements-transfer.txt`` extra isn't
installed. Uses the same synthetic (non-real) image fixtures as
``tests/test_smoke_training.py``, never real Kaggle data.
"""

from __future__ import annotations

import copy
from pathlib import Path

import pytest

pytest.importorskip("timm")

import torch  # noqa: E402

from src.data.dataset import FaceMultiTaskDataset  # noqa: E402
from src.data.split_utils import split_dataframe  # noqa: E402
from src.models.pretrained_volo import PretrainedVOLOFaceOnlyMultiTask  # noqa: E402
from src.training.persistent_artifacts import PersistentArtifactManager  # noqa: E402
from src.training.transfer_trainer import STAGE_1_NAME, STAGE_2_NAME, TransferTrainer  # noqa: E402

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


def test_stage1_checkpoint_snapshot_saved_separately_from_overall_best(tmp_path, tiny_transfer_datasets):
    """T7: the best Stage-1-only (frozen-backbone) checkpoint must be saved
    as its own file, distinct from the overall best -- answers "how much
    comes from the pretrained representation alone" without a second Stage-1
    training run."""
    from src.inference.artifacts import load_model_checkpoint

    model, train_dataset, val_dataset = tiny_transfer_datasets
    config = _tiny_transfer_config()
    config["training"]["head_only_epochs"] = 2
    checkpoint_dir = tmp_path / "checkpoints"
    trainer = TransferTrainer(
        model, config, train_dataset, val_dataset, device="cpu",
        checkpoint_dir=checkpoint_dir, experiment_name="volo_stage1_snapshot", output_dir=tmp_path / "output",
    )
    trainer.train()

    stage1_path = checkpoint_dir / "volo_stage1_snapshot_best_stage1_frozen.pt"
    assert stage1_path.exists()

    reloaded_model, config, checkpoint = load_model_checkpoint(stage1_path, device="cpu")
    assert checkpoint["extra"]["training_stage"] == STAGE_1_NAME
    assert isinstance(reloaded_model, PretrainedVOLOFaceOnlyMultiTask)

    # Distinct file from the overall best (which may reflect Stage 2).
    overall_best_path = checkpoint_dir / "volo_stage1_snapshot_best_balanced_score.pt"
    assert overall_best_path.exists()
    assert stage1_path != overall_best_path


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


# -- Persistence / resume ---------------------------------------------------------


def test_checkpoint_payload_contains_full_resumable_state(tmp_path, tiny_transfer_datasets):
    """Every epoch's checkpoint (via the artifact manager's last.pt) must
    carry everything needed to resume byte-for-byte -- optimizer, scheduler,
    AMP scaler, RNG state, stage, config snapshot, etc."""
    model, train_dataset, val_dataset = tiny_transfer_datasets
    manager = PersistentArtifactManager("volo_payload_test", seed=0, local_root=tmp_path / "artifacts")
    trainer = TransferTrainer(
        model, _tiny_transfer_config(), train_dataset, val_dataset, device="cpu",
        checkpoint_dir=tmp_path / "checkpoints", experiment_name="volo_payload", output_dir=tmp_path / "output",
        artifact_manager=manager, split_sha256="deadbeef", git_commit_sha="abc123",
    )
    trainer.run_stage_1()

    payload = manager.find_latest_valid_checkpoint()
    assert payload is not None
    for key in (
        "model_state_dict", "optimizer_state_dict", "scheduler_state_dict", "gradient_scaler_state_dict",
        "epoch", "global_step", "training_stage", "best_validation_metric", "early_stopping_state",
        "training_history", "seed", "rng_state", "model_id", "pretrained_source", "input_size",
        "transform_config", "split_sha256", "age_head_config", "gender_head_config", "adapter_config",
        "loss_balancing_params", "optimizer_group_lrs", "git_commit_sha", "config",
    ):
        assert key in payload, f"missing {key!r} in resumable checkpoint payload"
    assert payload["split_sha256"] == "deadbeef"
    assert payload["git_commit_sha"] == "abc123"
    assert payload["model_id"] == model.model_id
    assert payload["config"]["model"]["family"] == "pretrained_volo"


def test_resume_mid_stage1_continues_from_saved_epoch_and_restores_optimizer_state(tmp_path, tiny_transfer_datasets):
    model, train_dataset, val_dataset = tiny_transfer_datasets
    full_config = _tiny_transfer_config()
    full_config["training"]["head_only_epochs"] = 3

    manager = PersistentArtifactManager("volo_resume_mid", seed=0, local_root=tmp_path / "artifacts")
    trainer1 = TransferTrainer(
        model, full_config, train_dataset, val_dataset, device="cpu",
        checkpoint_dir=tmp_path / "checkpoints1", experiment_name="volo_resume_mid", output_dir=tmp_path / "output1",
        artifact_manager=manager,
    )
    # Simulate a runtime disconnect after 2 of 3 Stage-1 epochs: call the
    # private _run_stage directly with epochs=2 instead of run_stage_1(),
    # so the Stage-1 -> Stage-2 transition hook (which only fires when
    # run_stage_1() completes normally) never runs -- last.pt is left as a
    # genuine mid-stage checkpoint, exactly like a killed process would.
    model.freeze_backbone()
    backbone_lr, adapter_lr, head_lr, balance_lr, warmup_epochs, patience = trainer1._stage_lrs()
    trainer1._global_epoch = trainer1._run_stage(
        STAGE_1_NAME, 2, backbone_lr, adapter_lr, head_lr, balance_lr, warmup_epochs, patience, global_epoch=0,
    )
    assert len(trainer1.history["train_loss"]) == 2

    resume_payload = manager.find_latest_valid_checkpoint()
    assert resume_payload["training_stage"] == STAGE_1_NAME
    assert resume_payload["stage_epoch"] == 2
    assert resume_payload["optimizer_state_dict"] is not None
    assert resume_payload["scheduler_state_dict"] is not None

    model2 = PretrainedVOLOFaceOnlyMultiTask(_tiny_transfer_config())
    trainer2 = TransferTrainer(
        model2, full_config, train_dataset, val_dataset, device="cpu",
        checkpoint_dir=tmp_path / "checkpoints2", experiment_name="volo_resume_mid", output_dir=tmp_path / "output2",
        artifact_manager=manager, resume_state=resume_payload,
    )
    # Resumed history already has the 2 completed epochs before any new training runs.
    assert len(trainer2.history["train_loss"]) == 2
    assert torch.allclose(model2.log_var_age, model.log_var_age)
    assert trainer2.scaler.state_dict() == resume_payload["gradient_scaler_state_dict"]

    trainer2.run_stage_1()
    # Only the 1 remaining stage-1 epoch (3 - 2) should have run.
    assert len(trainer2.history["train_loss"]) == 3
    assert all(stage == STAGE_1_NAME for stage in trainer2.history["stage"])


def test_resume_from_stage2_checkpoint_skips_stage1(tmp_path, tiny_transfer_datasets):
    model, train_dataset, val_dataset = tiny_transfer_datasets
    config = _tiny_transfer_config()  # head_only_epochs=1 -> Stage 1 completes in a single epoch
    manager = PersistentArtifactManager("volo_resume_stage2", seed=0, local_root=tmp_path / "artifacts")
    trainer1 = TransferTrainer(
        model, config, train_dataset, val_dataset, device="cpu",
        checkpoint_dir=tmp_path / "checkpoints1", experiment_name="volo_resume_stage2", output_dir=tmp_path / "output1",
        artifact_manager=manager,
    )
    trainer1.run_stage_1()

    resume_payload = manager.find_latest_valid_checkpoint()
    assert resume_payload["training_stage"] == STAGE_2_NAME  # stage-transition marker
    assert resume_payload["stage_epoch"] == 0

    model2 = PretrainedVOLOFaceOnlyMultiTask(_tiny_transfer_config())
    trainer2 = TransferTrainer(
        model2, config, train_dataset, val_dataset, device="cpu",
        checkpoint_dir=tmp_path / "checkpoints2", experiment_name="volo_resume_stage2", output_dir=tmp_path / "output2",
        resume_state=resume_payload,
    )
    history_len_before = len(trainer2.history["train_loss"])
    trainer2.run_stage_1()
    # Stage 1 must be a complete no-op: no new epochs, no re-freezing/retraining.
    assert len(trainer2.history["train_loss"]) == history_len_before
    assert trainer2._resume_stage is None


# -- Live progress reporting -----------------------------------------------------------


def test_stage_announcement_printed_with_trainable_backbone_info(tmp_path, tiny_transfer_datasets, capsys):
    """Regression guard for a real bug: TransferTrainer used to report
    progress only via logger.info(...), which reaches no visible output at
    all unless some unrelated code happens to configure the root logger
    (see src/training/progress.py's module docstring) -- everything must
    also go through print(..., flush=True) via emit()."""
    model, train_dataset, val_dataset = tiny_transfer_datasets
    trainer = TransferTrainer(
        model, _tiny_transfer_config(), train_dataset, val_dataset, device="cpu",
        checkpoint_dir=tmp_path / "checkpoints", experiment_name="volo_stage_announce", output_dir=tmp_path / "output",
    )
    trainer.run_stage_1()
    captured = capsys.readouterr()
    assert STAGE_1_NAME in captured.out
    assert "trainable_params=" in captured.out
    assert "backbone parts:" in captured.out


def test_epoch_report_printed_with_full_metric_surface(tmp_path, tiny_transfer_datasets, capsys):
    model, train_dataset, val_dataset = tiny_transfer_datasets
    trainer = TransferTrainer(
        model, _tiny_transfer_config(), train_dataset, val_dataset, device="cpu",
        checkpoint_dir=tmp_path / "checkpoints", experiment_name="volo_epoch_report", output_dir=tmp_path / "output",
    )
    trainer.run_stage_1()
    captured = capsys.readouterr()
    for expected in (
        "balanced_acc=", "f1=", "selective_acc=", "coverage=", "abstention=",
        "log_var:", "loss_weights:", "backbone=", "adapters=", "heads=", "balance=",
        "selection_score=", "checkpoint:",
    ):
        assert expected in captured.out, f"missing {expected!r} in epoch report output"


def test_resume_announcement_printed_with_source_stage_and_checksums(tmp_path, tiny_transfer_datasets, capsys):
    model, train_dataset, val_dataset = tiny_transfer_datasets
    full_config = _tiny_transfer_config()
    full_config["training"]["head_only_epochs"] = 3

    manager = PersistentArtifactManager("volo_resume_announce", seed=0, local_root=tmp_path / "artifacts")
    trainer1 = TransferTrainer(
        model, full_config, train_dataset, val_dataset, device="cpu",
        checkpoint_dir=tmp_path / "checkpoints1", experiment_name="volo_resume_announce", output_dir=tmp_path / "output1",
        artifact_manager=manager, split_sha256="split-hash-abc123",
    )
    model.freeze_backbone()
    backbone_lr, adapter_lr, head_lr, balance_lr, warmup_epochs, patience = trainer1._stage_lrs()
    trainer1._run_stage(STAGE_1_NAME, 2, backbone_lr, adapter_lr, head_lr, balance_lr, warmup_epochs, patience, global_epoch=0)

    resume_payload = manager.find_latest_valid_checkpoint()
    capsys.readouterr()  # discard trainer1's own output

    model2 = PretrainedVOLOFaceOnlyMultiTask(_tiny_transfer_config())
    TransferTrainer(
        model2, full_config, train_dataset, val_dataset, device="cpu",
        checkpoint_dir=tmp_path / "checkpoints2", experiment_name="volo_resume_announce", output_dir=tmp_path / "output2",
        artifact_manager=manager, resume_state=resume_payload, resume_source="persistent",
        split_sha256="split-hash-abc123",
    )
    captured = capsys.readouterr()
    assert "Resuming training:" in captured.out
    assert "resume source:     persistent" in captured.out
    assert f"stage:             {STAGE_1_NAME}" in captured.out
    assert "checkpoint sha256:" in captured.out
    assert "n/a" not in captured.out.split("checkpoint sha256:")[1].split("\n")[0]  # a real hash, not "n/a"
    assert "split sha256:      split-hash-abc123" in captured.out
