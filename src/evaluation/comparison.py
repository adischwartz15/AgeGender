"""Builds the parametric-vs-kNN and architecture-ablation comparison tables.

All functions here operate on already-computed metric dicts (produced by
``scripts/evaluate.py`` and ``scripts/build_knn_index.py``) -- nothing in
this module invents numbers; it only aggregates and tabulates real results.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

_SEED_METRIC_KEYS = (
    "age_mae", "age_rmse", "gender_accuracy", "abstention_rate",
    "interval_coverage", "mean_interval_width", "latency_ms_per_image",
)


def build_parametric_vs_knn_table(parametric_metrics: dict, knn_metrics: dict) -> pd.DataFrame:
    """Side-by-side comparison table for the metrics listed in the spec.

    Expects both metric dicts to share the same keys (age_mae, age_rmse,
    interval_coverage, mean_interval_width, gender_accuracy,
    abstention_rate, mean_confidence, latency_ms_per_image).
    """
    keys = [
        "age_mae", "age_rmse", "interval_coverage", "mean_interval_width",
        "gender_accuracy", "abstention_rate", "mean_confidence", "latency_ms_per_image",
    ]
    rows = []
    for key in keys:
        rows.append(
            {
                "metric": key,
                "parametric": parametric_metrics.get(key),
                "knn": knn_metrics.get(key),
            }
        )
    return pd.DataFrame(rows)


def build_architecture_ablation_table(experiment_results: dict[str, dict]) -> pd.DataFrame:
    """One row per experiment (0, A-D), columns = parameter counts + key metrics.

    ``experiment_results`` maps experiment name -> a dict with keys
    ``parameter_breakdown`` (from ``ParameterBreakdown.as_dict()``),
    ``val_metrics`` / ``test_metrics``, and ``mean_epoch_time_seconds``.
    """
    rows = []
    for name, result in experiment_results.items():
        params = result.get("parameter_breakdown", {})
        metrics = result.get("test_metrics", result.get("val_metrics", {}))
        rows.append(
            {
                "experiment": name,
                "backbone_name": params.get("backbone_name"),
                "backbone_params": params.get("backbone_parameters"),
                "adapter_params": params.get("adapter_parameters"),
                "total_params": params.get("total_parameters"),
                "age_mae": metrics.get("age_mae"),
                "gender_accuracy": metrics.get("gender_accuracy"),
                "interval_coverage": metrics.get("interval_coverage"),
                "mean_epoch_time_seconds": result.get("mean_epoch_time_seconds"),
            }
        )
    return pd.DataFrame(rows)


_BACKBONE_COMPARISON_KEYS = [
    ("backbone_name", "Backbone"),
    ("total_parameters", "Total parameters"),
    ("backbone_parameters", "Backbone parameters"),
    ("mean_epoch_time_seconds", "Mean epoch time (s)"),
    ("latency_ms_per_image", "Inference latency per image (ms)"),
    ("age_mae", "Age MAE"),
    ("age_rmse", "Age RMSE"),
    ("gender_accuracy", "Gender-label accuracy"),
    ("abstention_rate", "Abstention rate"),
    ("interval_coverage", "Raw interval coverage"),
    ("interval_coverage_calibrated", "Calibrated interval coverage"),
    ("mean_interval_width", "Mean interval width"),
]


def build_backbone_comparison_table(cnn_metrics: dict, resnet_metrics: dict) -> pd.DataFrame:
    """Side-by-side table for the plain-CNN-vs-Custom-ResNet-18 backbone comparison.

    Each of ``cnn_metrics`` / ``resnet_metrics`` is expected to already
    merge that experiment's parameter breakdown, timing, and test metrics
    into one flat dict (see ``scripts/generate_architecture_report.py``).
    Missing keys render as ``None`` rather than being fabricated. Kept for
    backward compatibility -- see :func:`build_backbone_comparison_table_multi`
    for the 3-model (SimpleCNN / PlainDeep18NoSkip / Custom ResNet-18) table.
    """
    rows = []
    for key, label in _BACKBONE_COMPARISON_KEYS:
        rows.append({"metric": label, "simple_cnn": cnn_metrics.get(key), "custom_resnet18": resnet_metrics.get(key)})
    return pd.DataFrame(rows)


def build_backbone_comparison_table_multi(metrics_by_name: dict[str, dict]) -> pd.DataFrame:
    """Side-by-side table across an arbitrary number of backbone experiments.

    ``metrics_by_name`` maps a display column name (e.g. "simple_cnn",
    "plain_deep18_no_skip", "custom_resnet18") to that experiment's merged
    parameter-breakdown/timing/test-metrics dict. Missing keys render as
    ``None`` rather than being fabricated.
    """
    rows = []
    for key, label in _BACKBONE_COMPARISON_KEYS:
        row = {"metric": label}
        for name, metrics in metrics_by_name.items():
            row[name] = metrics.get(key)
        rows.append(row)
    return pd.DataFrame(rows)


def aggregate_seed_metrics(per_seed_metrics: list[dict], keys: tuple[str, ...] = _SEED_METRIC_KEYS) -> dict:
    """Compute mean +/- **sample** std across N seed runs' test-metric dicts.

    Returns ``{key: {"mean": ..., "std": ..., "n_seeds": N}}`` for each
    key present (and numeric) in at least one provided dict; missing
    values for a given seed are simply excluded from that key's mean/std
    rather than treated as zero (so each metric carries its own
    ``n_seeds``, not a single row-level count -- see the final-run reporting
    protocol). With fewer than 2 seed runs, ``std`` is reported as ``None``
    (not 0.0) so callers can render an honest "insufficient runs" message
    instead of a misleadingly precise number.

    The reported std is the **sample** standard deviation (``ddof=1``), the
    correct estimator for the population std from a small number of seed
    runs -- the population std (``ddof=0``) would systematically
    understate run-to-run variability for the final 3-seed table.
    """
    result: dict[str, dict] = {}
    n_seeds = len(per_seed_metrics)
    for key in keys:
        values = [m[key] for m in per_seed_metrics if m.get(key) is not None]
        if not values:
            continue
        result[key] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values, ddof=1)) if len(values) >= 2 else None,
            "n_seeds": len(values),
        }
    result["_n_seed_runs"] = n_seeds
    return result


def build_seed_aggregate_table(aggregates: dict[str, dict]) -> pd.DataFrame:
    """One row per experiment, columns = mean +/- std for each metric.

    ``aggregates`` maps experiment name -> the dict returned by
    :func:`aggregate_seed_metrics`.
    """
    rows = []
    for exp_name, agg in aggregates.items():
        row = {"experiment": exp_name, "n_seeds": agg.get("_n_seed_runs")}
        for key in _SEED_METRIC_KEYS:
            stats = agg.get(key)
            if stats is None:
                row[key] = None
            elif stats["std"] is None:
                row[key] = f"{stats['mean']:.3f} (n=1, no std)"
            else:
                row[key] = f"{stats['mean']:.3f} +/- {stats['std']:.3f}"
        rows.append(row)
    return pd.DataFrame(rows)


_TABLE_B_COLUMNS = [
    ("model", "Model"),
    ("experiment_category", "Experiment category"),
    ("initialization", "Initialization"),
    ("backbone", "Backbone"),
    ("adapters", "Adapters"),
    ("loss_balancing", "Loss balancing"),
    ("input_size", "Input size"),
    ("age_mae", "Age MAE"),
    ("age_rmse", "Age RMSE"),
    ("age_cs5", "CS@5"),
    ("gender_accuracy", "Gender acc"),
    ("gender_f1", "Gender F1"),
    ("total_params", "Params"),
    ("trainable_params", "Trainable params"),
]
_TABLE_B_SEED_SENSITIVE_KEYS = ("age_mae", "age_rmse", "age_cs5", "gender_accuracy", "gender_f1")


def build_transfer_learning_table(rows: list[dict]) -> pd.DataFrame:
    """Table B: best from-scratch model vs. the supplementary ImageNet-pretrained
    VOLO-D1 transfer-learning experiment.

    Structurally separate from :func:`build_architecture_ablation_table`
    (Table A) -- this function is never called with, or merged into,
    Table A's rows, since Table B compares systems that differ in
    initialization, capacity, input resolution, and optimizer/schedule
    simultaneously, not a controlled ablation.

    Each element of ``rows`` is one already-computed model's flat metric
    dict (see ``scripts/run_transfer_learning.py`` for how a row is
    assembled from real ``evaluate_checkpoint()`` / ``parameter_breakdown()``
    output, optionally seed-aggregated via :func:`aggregate_seed_metrics`).
    Missing keys render as ``None``, never fabricated.

    When a row's ``n_seeds`` is 1 (or absent), its seed-sensitive metric
    values are annotated "(n=1, no std)" -- the same convention
    :func:`build_seed_aggregate_table` already uses -- so a reader cannot
    mistake a single run for a variance-estimated result. With >=2 seeds
    and a matching ``{key}_std`` present, the value is rendered as
    ``mean +/- std (n=N)``.
    """
    table_rows = []
    for row in rows:
        n_seeds = row.get("n_seeds", 1)
        table_row = {}
        for key, label in _TABLE_B_COLUMNS:
            value = row.get(key)
            if key in _TABLE_B_SEED_SENSITIVE_KEYS and isinstance(value, (int, float)):
                std = row.get(f"{key}_std")
                if n_seeds and n_seeds >= 2 and std is not None:
                    value = f"{value:.3f} +/- {std:.3f} (n={n_seeds})"
                else:
                    value = f"{value:.3f} (n=1, no std)"
            table_row[label] = value
        table_rows.append(table_row)
    return pd.DataFrame(table_rows)
