"""Builds the final, cross-cutting results report for the whole project.

Unlike ``src/evaluation/reports.py`` (which focuses on the architecture
ablation study), this module assembles *all* of the course-facing results
in one document: the ablation table, the plain-CNN-vs-ResNet comparison,
a mean +/- std table across seeds, per-age-bucket uncertainty metrics
(raw and calibrated), robustness degradation, and parameter/latency
comparison plots. Every section reads only real artifacts already saved
under ``outputs/`` by other scripts (``run_experiments.py``,
``run_seeds.py``, ``evaluate.py``, ``run_robustness.py``) -- any section
whose backing artifact is missing renders an explicit "not yet generated"
message with the command that would produce it, never a fabricated number.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from src.evaluation.comparison import (
    aggregate_seed_metrics, build_architecture_ablation_table, build_seed_aggregate_table,
)
from src.evaluation.reports import (
    _CNN_EXPERIMENT, _RESNET_EXPERIMENT, _MISSING, _backbone_comparison_interpretation, _df_to_md_table,
    _load_merged_experiment_metrics, _read_csv, _read_json, build_backbone_comparison_section,
    discover_experiment_results,
)
from src.utils.visualization import plot_mean_std_bar, plot_parameter_latency_comparison

_SEED_GROUP_RE = re.compile(r"^(?P<experiment>.+)_seed\d+_test_metrics\.json$")

# Preferred order for picking the "primary" model shown in the uncertainty
# section: the main research backbone (learned-balance shared adapters)
# first, falling back to earlier ablation stages, and finally to whatever
# checkpoint was evaluated under the generic "multitask" name.
_PRIMARY_EXPERIMENT_CANDIDATES = (
    "exp_d_shared_adapters_learned_balance", "exp_c_shared_adapters", "multitask",
)


def _md_image(path: Path, repo_root: Path, label: str) -> str:
    rel = path.relative_to(repo_root).as_posix()
    return f"![{label}](../{rel})\n"


def _discover_seed_metrics(metrics_dir: Path) -> dict[str, list[dict]]:
    groups: dict[str, list[dict]] = {}
    for f in sorted(metrics_dir.glob("*_seed*_test_metrics.json")):
        match = _SEED_GROUP_RE.match(f.name)
        if not match:
            continue
        data = _read_json(f)
        if data is not None:
            groups.setdefault(match.group("experiment"), []).append(data)
    return groups


def _find_primary_test_metrics(metrics_dir: Path) -> tuple[str | None, dict | None]:
    for name in _PRIMARY_EXPERIMENT_CANDIDATES:
        data = _read_json(metrics_dir / f"{name}_test_metrics.json")
        if data is not None:
            return name, data
    return None, None


def _build_ablation_section(outputs_dir: Path) -> str:
    lines = ["## Architecture Ablation Table\n"]
    experiment_results = discover_experiment_results(outputs_dir / "metrics")
    if not experiment_results:
        lines.append(_MISSING.format(cmd="python scripts/run_experiments.py") + "\n")
        return "\n".join(lines)
    table = build_architecture_ablation_table(experiment_results)
    lines.append(_df_to_md_table(table) + "\n")
    return "\n".join(lines)


def _build_seed_aggregate_section(outputs_dir: Path) -> str:
    lines = ["## Mean +/- Std Across Seeds\n"]
    groups = _discover_seed_metrics(outputs_dir / "metrics")
    if not groups:
        lines.append(
            _MISSING.format(cmd="python scripts/run_seeds.py --experiment <name> --seeds 42,43,44") + "\n"
        )
        return "\n".join(lines)

    aggregates = {name: aggregate_seed_metrics(seed_metrics) for name, seed_metrics in groups.items()}
    table = build_seed_aggregate_table(aggregates)
    lines.append(_df_to_md_table(table) + "\n")

    single_seed = [name for name, agg in aggregates.items() if agg.get("_n_seed_runs", 0) < 2]
    if single_seed:
        lines.append(
            f"_Note: {', '.join(single_seed)} has fewer than 2 seed runs on disk; std is reported "
            "as unavailable rather than a misleadingly precise 0.000._\n"
        )
    return "\n".join(lines)


def _build_seed_plots(outputs_dir: Path, repo_root: Path) -> str:
    groups = _discover_seed_metrics(outputs_dir / "metrics")
    if not groups:
        return ""
    aggregates = {name: aggregate_seed_metrics(seed_metrics) for name, seed_metrics in groups.items()}
    plots_dir = outputs_dir / "plots" / "final_report"
    plots_dir.mkdir(parents=True, exist_ok=True)

    lines = []
    for metric in ("age_mae", "gender_accuracy"):
        names, means, stds = [], [], []
        for name, agg in aggregates.items():
            stats = agg.get(metric)
            if stats is None:
                continue
            names.append(name)
            means.append(stats["mean"])
            stds.append(stats["std"] or 0.0)
        if names:
            out = plot_mean_std_bar(names, np.array(means), np.array(stds), metric, plots_dir / f"seed_mean_std_{metric}.png")
            lines.append(_md_image(out, repo_root, f"{metric} mean +/- std across seeds"))
    return "\n".join(lines)


def _build_uncertainty_section(outputs_dir: Path, repo_root: Path) -> str:
    lines = ["## Uncertainty Evaluation\n"]
    lines.append(
        "**Important caveat: marginal coverage is not conditional coverage.** "
        "Conformal calibration (when used) targets *marginal* coverage -- "
        "averaged across the entire test set -- not coverage conditioned on "
        "age bucket, gender-label subgroup, or any other subpopulation. A "
        "bucket can be systematically under- or over-covered even while the "
        "overall test-set coverage exactly matches the target. The per-bucket "
        "tables and plots below exist specifically so this can be checked, "
        "not assumed away.\n"
    )

    metrics_dir = outputs_dir / "metrics"
    primary_name, primary_metrics = _find_primary_test_metrics(metrics_dir)
    if primary_metrics is None:
        lines.append(_MISSING.format(cmd="python scripts/evaluate.py --checkpoint <primary checkpoint>") + "\n")
        return "\n".join(lines)
    lines.append(f"Primary model shown below: `{primary_name}`.\n")

    bucket_report = primary_metrics.get("age_metrics_by_bucket")
    lines.append("### Age MAE / Coverage / Width by Age Bucket (raw)\n")
    if bucket_report:
        table = pd.DataFrame([{"age_bucket": label, **stats} for label, stats in bucket_report.items()])
        lines.append(_df_to_md_table(table) + "\n")
    else:
        lines.append(_MISSING.format(cmd="python scripts/evaluate.py --checkpoint <primary checkpoint>") + "\n")

    calibrated_report = primary_metrics.get("age_metrics_by_bucket_calibrated")
    lines.append("### Age MAE / Coverage / Width by Age Bucket (after conformal calibration)\n")
    if calibrated_report:
        table = pd.DataFrame([{"age_bucket": label, **stats} for label, stats in calibrated_report.items()])
        lines.append(_df_to_md_table(table) + "\n")
    else:
        lines.append(
            "_Calibrated per-bucket metrics unavailable -- run `python scripts/calibrate.py` "
            "then re-run `python scripts/evaluate.py` against the same checkpoint._\n"
        )

    plots_dir = outputs_dir / "plots"
    plot_specs = (
        (f"{primary_name}_test_metrics_interval_coverage.png", "Empirical interval coverage by age bucket"),
        (f"{primary_name}_test_metrics_interval_width_by_bucket.png", "Interval width by age bucket"),
        (f"{primary_name}_test_metrics_coverage_width_tradeoff.png", "Coverage-width trade-off before/after conformal calibration"),
    )
    for filename, label in plot_specs:
        path = plots_dir / filename
        if path.exists():
            lines.append(_md_image(path, repo_root, label))
        else:
            lines.append(f"_{label} plot not yet generated (`{filename}` not found)._\n")

    lines.append("### Narrowest and Widest Prediction Intervals\n")
    examples = primary_metrics.get("interval_examples")
    if examples:
        for kind in ("narrowest", "widest"):
            lines.append(f"**{kind.capitalize()}**\n")
            lines.append(_df_to_md_table(pd.DataFrame(examples[kind])) + "\n")
    else:
        lines.append(_MISSING.format(cmd="python scripts/evaluate.py --checkpoint <primary checkpoint>") + "\n")

    return "\n".join(lines)


def _build_robustness_section(outputs_dir: Path, repo_root: Path) -> str:
    lines = ["## Robustness Degradation\n"]
    df = _read_csv(outputs_dir / "robustness" / "robustness_results.csv")
    if df is None:
        lines.append(_MISSING.format(cmd="python scripts/run_robustness.py --checkpoint <checkpoint>") + "\n")
        return "\n".join(lines)

    clean = df[df["corruption"] == "clean"]
    lines.append("**Clean baseline**\n\n" + _df_to_md_table(clean) + "\n")

    corrupted = df[df["corruption"] != "clean"]
    summary_cols = [
        c for c in ("age_mae", "gender_accuracy", "abstention_rate", "mean_confidence", "mean_interval_width")
        if c in df.columns
    ]
    if not corrupted.empty and summary_cols:
        summary = corrupted.groupby("corruption")[summary_cols].mean().reset_index()
        lines.append("**Mean metrics by corruption type (across severities)**\n\n" + _df_to_md_table(summary) + "\n")

    robustness_dir = outputs_dir / "robustness"
    for metric in ("age_mae", "gender_accuracy", "abstention_rate"):
        plot_path = robustness_dir / f"robustness_{metric}.png"
        if plot_path.exists():
            lines.append(_md_image(plot_path, repo_root, f"Robustness degradation curve: {metric}"))
    return "\n".join(lines)


def _build_parameter_latency_section(outputs_dir: Path, repo_root: Path) -> str:
    lines = ["## Parameter Count and Inference Latency Comparison\n"]
    experiment_results = discover_experiment_results(outputs_dir / "metrics")
    labels, params, latencies = [], [], []
    for name, result in experiment_results.items():
        total_params = result.get("parameter_breakdown", {}).get("total_parameters")
        latency = result.get("test_metrics", {}).get("latency_ms_per_image")
        if total_params is not None and latency is not None:
            labels.append(name)
            params.append(total_params)
            latencies.append(latency)

    if not labels:
        lines.append(
            _MISSING.format(cmd="python scripts/run_experiments.py (trains + evaluates each experiment)") + "\n"
        )
        return "\n".join(lines)

    plots_dir = outputs_dir / "plots" / "final_report"
    plots_dir.mkdir(parents=True, exist_ok=True)
    out_path = plot_parameter_latency_comparison(labels, params, latencies, plots_dir / "parameter_latency_comparison.png")
    lines.append(_md_image(out_path, repo_root, "Parameter count vs inference latency per experiment"))
    table = pd.DataFrame({"experiment": labels, "total_parameters": params, "latency_ms_per_image": latencies})
    lines.append(_df_to_md_table(table) + "\n")
    return "\n".join(lines)


def _build_findings_section(outputs_dir: Path) -> str:
    findings = []

    cnn_metrics = _load_merged_experiment_metrics(outputs_dir, _CNN_EXPERIMENT)
    resnet_metrics = _load_merged_experiment_metrics(outputs_dir, _RESNET_EXPERIMENT)
    if (
        cnn_metrics and resnet_metrics
        and cnn_metrics.get("age_mae") is not None and resnet_metrics.get("age_mae") is not None
    ):
        findings.append(_backbone_comparison_interpretation(cnn_metrics, resnet_metrics).strip())

    df = _read_csv(outputs_dir / "robustness" / "robustness_results.csv")
    if df is not None and "age_mae" in df.columns:
        clean_row = df[df["corruption"] == "clean"]
        corrupted = df[df["corruption"] != "clean"]
        if not clean_row.empty and not corrupted.empty:
            clean_mae = float(clean_row.iloc[0]["age_mae"])
            worst = corrupted.loc[corrupted["age_mae"].idxmax()]
            findings.append(
                f"Under the measured corruptions, age MAE degraded from {clean_mae:.2f} years (clean) to "
                f"as much as {float(worst['age_mae']):.2f} years under '{worst['corruption']}' at severity "
                f"{int(worst['severity'])}, an increase of {float(worst['age_mae']) - clean_mae:.2f} years."
            )

    lines = ["## Evidence-Based Findings\n"]
    if not findings:
        lines.append(
            "_No findings are stated yet because the underlying experiments/evaluations have not been run "
            "in this environment. Run `python scripts/run_experiments.py`, `python scripts/run_seeds.py`, "
            "and `python scripts/run_robustness.py`, then re-run this report to populate this section with "
            "real, measured results._\n"
        )
    else:
        for finding in findings:
            lines.append(f"- {finding}\n")
    return "\n".join(lines)


def generate_final_results_report(outputs_dir: str | Path, repo_root: str | Path) -> str:
    outputs_dir, repo_root = Path(outputs_dir), Path(repo_root)
    sections = [
        "# Final Results Report\n",
        (
            "Auto-generated from real saved artifacts under `outputs/` only. Any "
            "section whose backing artifact does not exist yet renders an explicit "
            "\"not yet generated\" message with the command that would produce it, "
            "rather than a fabricated number. Regenerate with "
            "`python scripts/generate_final_report.py` after (re-)running the "
            "relevant experiment/evaluation/robustness scripts.\n"
        ),
        (
            "**Scope note.** This is a research/education artifact. Dataset "
            "gender-label predictions reflect labels defined by the source "
            "dataset's documentation, not a determination of gender identity, and "
            "this system must not be used for employment, policing, surveillance, "
            "identity verification, medical diagnosis, admissions, insurance, or "
            "other high-impact decisions.\n"
        ),
        _build_ablation_section(outputs_dir),
        build_backbone_comparison_section(outputs_dir),
        _build_seed_aggregate_section(outputs_dir),
        _build_seed_plots(outputs_dir, repo_root),
        _build_uncertainty_section(outputs_dir, repo_root),
        _build_robustness_section(outputs_dir, repo_root),
        _build_parameter_latency_section(outputs_dir, repo_root),
        _build_findings_section(outputs_dir),
    ]
    return "\n".join(s for s in sections if s)


def save_final_results_report(outputs_dir: str | Path, docs_dir: str | Path, repo_root: str | Path) -> Path:
    report = generate_final_results_report(outputs_dir, repo_root)
    out_path = Path(docs_dir) / "final_results_report.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report, encoding="utf-8")
    return out_path
