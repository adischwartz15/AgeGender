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

from src.evaluation.comparison import build_architecture_ablation_table

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from evaluate import _default_output_name  # noqa: E402


def test_ablation_table_picks_up_test_metrics_when_present():
    experiment_results = {
        "exp_c_shared_adapters": {
            "parameter_breakdown": {"backbone": 11176512, "adapters": 263424, "total": 11571909},
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


def test_ablation_table_is_nan_only_when_test_metrics_truly_absent():
    experiment_results = {
        "exp_a_separate": {
            "parameter_breakdown": {"backbone": 22353024, "adapters": 0, "total": 22484997},
            "test_metrics": {},
            "mean_epoch_time_seconds": 44.2,
        }
    }
    table = build_architecture_ablation_table(experiment_results)
    row = table.iloc[0]
    assert row["age_mae"] is None
    assert row["backbone_params"] == 22353024


def test_default_output_name_strips_best_checkpoint_suffix():
    assert _default_output_name("checkpoints/exp_c_shared_adapters_best_balanced_score.pt") == "exp_c_shared_adapters_test_metrics"
    assert _default_output_name("checkpoints/multitask_best_age_mae.pt") == "multitask_test_metrics"
    assert _default_output_name("checkpoints/multitask_best_gender_accuracy.pt") == "multitask_test_metrics"


def test_default_output_name_falls_back_when_no_known_suffix():
    assert _default_output_name("checkpoints/some_custom_checkpoint.pt") == "some_custom_checkpoint_test_metrics"
