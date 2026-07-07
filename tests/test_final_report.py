"""Tests for the final cross-cutting results report generator.

Two things matter most here: (1) with no artifacts on disk, every section
renders an honest "not yet generated" message instead of fabricating
numbers; (2) once real (synthetic-but-saved) artifacts exist, the report
picks them up correctly -- ablation table, seed mean+/-std, per-bucket
uncertainty metrics, robustness summary, and parameter/latency comparison.
"""

from __future__ import annotations

import json

from src.evaluation.final_report import generate_final_results_report


def _write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


def test_report_is_honest_when_no_artifacts_exist(tmp_path):
    outputs_dir = tmp_path / "outputs"
    outputs_dir.mkdir()
    repo_root = tmp_path

    report = generate_final_results_report(outputs_dir, repo_root)

    assert "Final Results Report" in report
    assert "Not yet generated" in report
    assert "marginal coverage" in report.lower() or "conditional coverage" in report.lower()
    assert "No findings are stated yet" in report


def test_report_picks_up_seed_runs_and_bucket_metrics(tmp_path):
    outputs_dir = tmp_path / "outputs"
    metrics_dir = outputs_dir / "metrics"
    repo_root = tmp_path

    for seed in (42, 43):
        _write_json(
            metrics_dir / f"exp_c_shared_adapters_seed{seed}_test_metrics.json",
            {"age_mae": 5.0 + seed * 0.01, "gender_accuracy": 0.9},
        )

    _write_json(
        metrics_dir / "exp_d_shared_adapters_learned_balance_test_metrics.json",
        {
            "age_mae": 5.2,
            "interval_coverage": 0.79,
            "mean_interval_width": 12.0,
            "age_metrics_by_bucket": {
                "0-10": {"count": 5, "mae": 2.1, "coverage": 0.8, "mean_width": 8.0, "median_width": 7.5},
                "10-20": {"count": 0, "mae": None, "coverage": None, "mean_width": None, "median_width": None},
            },
            "interval_examples": {
                "narrowest": [{"image_path": "a.jpg", "true_age": 20.0, "q10": 15.0, "q50": 20.0, "q90": 25.0, "width": 10.0}],
                "widest": [{"image_path": "b.jpg", "true_age": 40.0, "q10": 20.0, "q50": 40.0, "q90": 60.0, "width": 40.0}],
            },
        },
    )

    report = generate_final_results_report(outputs_dir, repo_root)

    assert "exp_c_shared_adapters" in report
    assert "exp_d_shared_adapters_learned_balance" in report
    assert "0-10" in report
    assert "a.jpg" in report and "b.jpg" in report
    # exp_c has 2 real seed runs on disk -- a real mean +/- std should be rendered.
    assert "+/-" in report
