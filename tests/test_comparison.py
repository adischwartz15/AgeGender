"""Regression tests for architecture-ablation table assembly.

Guards against a real bug found in a live run: the ablation table was
always showing NaN for age_mae / gender_accuracy / interval_coverage
because per-experiment test metrics were never merged into the dict
passed to build_architecture_ablation_table (only parameter counts and
epoch timing were). See scripts/run_experiments.py and
scripts/generate_architecture_report.py for the fix.
"""

from __future__ import annotations

import sys
from pathlib import Path

from src.evaluation.comparison import (
    aggregate_seed_metrics, build_architecture_ablation_table, build_seed_aggregate_table,
    build_transfer_learning_table,
)

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from evaluate import _default_output_name  # noqa: E402


def test_ablation_table_picks_up_test_metrics_when_present():
    experiment_results = {
        "exp_c_shared_adapters": {
            "parameter_breakdown": {
                "backbone_name": "custom_resnet18", "backbone_parameters": 11176512,
                "adapter_parameters": 263424, "total_parameters": 11571909,
            },
            "test_metrics": {"age_mae": 5.71, "gender_accuracy": 0.97, "interval_coverage": 0.79},
            "mean_epoch_time_seconds": 41.5,
        }
    }
    table = build_architecture_ablation_table(experiment_results)
    row = table.iloc[0]
    assert row["age_mae"] == 5.71
    assert row["gender_accuracy"] == 0.97
    assert row["interval_coverage"] == 0.79
    assert row["backbone_params"] == 11176512
    assert row["backbone_name"] == "custom_resnet18"


def test_ablation_table_is_nan_only_when_test_metrics_truly_absent():
    experiment_results = {
        "exp_a_separate": {
            "parameter_breakdown": {
                "backbone_name": "custom_resnet18", "backbone_parameters": 22353024,
                "adapter_parameters": 0, "total_parameters": 22484997,
            },
            "test_metrics": {},
            "mean_epoch_time_seconds": 44.2,
        }
    }
    table = build_architecture_ablation_table(experiment_results)
    row = table.iloc[0]
    assert row["age_mae"] is None
    assert row["backbone_params"] == 22353024


def test_ablation_table_includes_simple_cnn_experiment():
    experiment_results = {
        "exp_0_simple_cnn_shared_adapters_learned_balance": {
            "parameter_breakdown": {
                "backbone_name": "simple_cnn", "backbone_parameters": 4_000_000,
                "adapter_parameters": 263424, "total_parameters": 4_400_000,
            },
            "test_metrics": {"age_mae": 6.5, "gender_accuracy": 0.94, "interval_coverage": 0.75},
            "mean_epoch_time_seconds": 30.0,
        }
    }
    table = build_architecture_ablation_table(experiment_results)
    row = table.iloc[0]
    assert row["backbone_name"] == "simple_cnn"
    assert row["backbone_params"] == 4_000_000


def test_default_output_name_strips_best_checkpoint_suffix():
    assert _default_output_name("checkpoints/exp_c_shared_adapters_best_balanced_score.pt") == "exp_c_shared_adapters_test_metrics"
    assert _default_output_name("checkpoints/multitask_best_age_mae.pt") == "multitask_test_metrics"
    assert _default_output_name("checkpoints/multitask_best_gender_accuracy.pt") == "multitask_test_metrics"


def test_default_output_name_falls_back_when_no_known_suffix():
    assert _default_output_name("checkpoints/some_custom_checkpoint.pt") == "some_custom_checkpoint_test_metrics"


def test_aggregate_seed_metrics_computes_mean_and_std():
    per_seed = [
        {"age_mae": 5.0, "gender_accuracy": 0.90},
        {"age_mae": 6.0, "gender_accuracy": 0.92},
        {"age_mae": 7.0, "gender_accuracy": 0.94},
    ]
    agg = aggregate_seed_metrics(per_seed)
    assert agg["age_mae"]["mean"] == 6.0
    assert agg["age_mae"]["n_seeds"] == 3
    assert agg["age_mae"]["std"] is not None and agg["age_mae"]["std"] > 0
    assert agg["_n_seed_runs"] == 3


def test_aggregate_seed_metrics_uses_sample_std_ddof1():
    """The reported std must be the sample std (ddof=1), not the population
    std (ddof=0), for the final multi-seed table -- for [5,6,7] that is 1.0,
    not sqrt(2/3) ~= 0.816."""
    import numpy as np

    per_seed = [{"age_mae": 5.0}, {"age_mae": 6.0}, {"age_mae": 7.0}]
    agg = aggregate_seed_metrics(per_seed)
    assert abs(agg["age_mae"]["std"] - float(np.std([5.0, 6.0, 7.0], ddof=1))) < 1e-9
    assert abs(agg["age_mae"]["std"] - 1.0) < 1e-9  # sample std, not 0.8165 (population)


def test_aggregate_seed_metrics_std_is_none_with_single_seed():
    agg = aggregate_seed_metrics([{"age_mae": 5.0}])
    assert agg["age_mae"]["mean"] == 5.0
    assert agg["age_mae"]["std"] is None
    assert agg["age_mae"]["n_seeds"] == 1


def test_aggregate_seed_metrics_skips_missing_values_per_key():
    per_seed = [{"age_mae": 5.0, "gender_accuracy": 0.9}, {"age_mae": 6.0}]  # gender_accuracy missing in 2nd seed
    agg = aggregate_seed_metrics(per_seed)
    assert agg["age_mae"]["n_seeds"] == 2
    assert agg["gender_accuracy"]["n_seeds"] == 1
    assert agg["gender_accuracy"]["std"] is None


def test_build_seed_aggregate_table_formats_mean_and_std():
    aggregates = {
        "exp_c_shared_adapters": aggregate_seed_metrics([{"age_mae": 5.0}, {"age_mae": 7.0}]),
        "exp_d_shared_adapters_learned_balance": aggregate_seed_metrics([{"age_mae": 4.0}]),
    }
    table = build_seed_aggregate_table(aggregates)
    row_c = table[table["experiment"] == "exp_c_shared_adapters"].iloc[0]
    row_d = table[table["experiment"] == "exp_d_shared_adapters_learned_balance"].iloc[0]
    assert "+/-" in row_c["age_mae"]
    assert "no std" in row_d["age_mae"]


def _table_b_row(model_label, n_seeds, age_mae=5.0, age_mae_std=None):
    return {
        "model": model_label, "experiment_category": "core (from-scratch)", "initialization": "Random",
        "backbone": "custom_resnet18", "adapters": "shared_adapters", "loss_balancing": "learned_uncertainty",
        "input_size": 128, "age_mae": age_mae, "age_mae_std": age_mae_std, "age_rmse": 6.0, "age_rmse_std": None,
        "age_cs5": 0.5, "age_cs5_std": None, "gender_accuracy": 0.9, "gender_accuracy_std": None,
        "gender_f1": 0.88, "gender_f1_std": None, "total_params": 1000, "trainable_params": 1000,
        "n_seeds": n_seeds,
    }


def test_build_transfer_learning_table_labels_single_seed_no_variance():
    table = build_transfer_learning_table([_table_b_row("exp_d_shared_adapters_learned_balance", n_seeds=1)])
    row = table.iloc[0]
    assert "n=1, no std" in row["Age MAE"]
    assert "+/-" not in row["Age MAE"]


def test_build_transfer_learning_table_formats_multi_seed_mean_and_std():
    row_dict = _table_b_row("volo_d1_face_only_pretrained", n_seeds=3, age_mae=5.2, age_mae_std=0.3)
    table = build_transfer_learning_table([row_dict])
    row = table.iloc[0]
    assert "+/-" in row["Age MAE"]
    assert "n=3" in row["Age MAE"]


def test_build_transfer_learning_table_never_merges_with_table_a_columns():
    """Table B's columns are a structurally distinct set from Table A's
    (build_architecture_ablation_table) -- never the same schema, so the
    two tables cannot accidentally be concatenated into one."""
    table_a = build_architecture_ablation_table(
        {"exp_d_shared_adapters_learned_balance": {"parameter_breakdown": {}, "test_metrics": {}}}
    )
    table_b = build_transfer_learning_table([_table_b_row("exp_d_shared_adapters_learned_balance", n_seeds=1)])
    assert set(table_a.columns) != set(table_b.columns)
    assert "Model" in table_b.columns and "Model" not in table_a.columns


def test_build_transfer_learning_table_missing_keys_render_as_none_not_fabricated():
    row_dict = _table_b_row("volo_d1_face_only_pretrained", n_seeds=1)
    del row_dict["gender_f1"]
    table = build_transfer_learning_table([row_dict])
    assert table.iloc[0]["Gender F1"] is None


def test_table_b_labels_selective_accuracy_distinctly_from_balanced_accuracy():
    """T7: gender selective accuracy (confidence-thresholded) must be
    labelled distinctly from balanced accuracy (raw argmax, full coverage)
    -- never presented as interchangeable "gender accuracy" columns."""
    row_dict = _table_b_row("volo_d1_face_only_pretrained", n_seeds=1)
    row_dict["gender_balanced_accuracy"] = 0.87
    row_dict["gender_balanced_accuracy_std"] = None
    row_dict["age_median_ae"] = 3.1
    row_dict["age_median_ae_std"] = None
    table = build_transfer_learning_table([row_dict])
    columns = list(table.columns)
    assert "Gender selective acc (tau=0.80)" in columns
    assert "Gender balanced acc (raw, full coverage)" in columns
    assert "Age median AE" in columns
    assert columns.index("Gender selective acc (tau=0.80)") != columns.index("Gender balanced acc (raw, full coverage)")
