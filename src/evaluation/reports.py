"""Builds the automated architecture-analysis Markdown report.

Reads whatever JSON/CSV artifacts already exist under ``outputs/`` and
composes a Markdown report from them. Never invents numbers: any section
whose backing artifact is missing is rendered as an explicit
"not yet generated" placeholder with the command that would produce it,
rather than fabricated content.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

_MISSING = "_Not yet generated. Run `{cmd}` to produce this section._"


def _read_json(path: Path) -> dict | None:
    if path.exists():
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    return None


def _read_csv(path: Path) -> pd.DataFrame | None:
    if path.exists():
        return pd.read_csv(path)
    return None


def _dict_to_md_table(d: dict) -> str:
    lines = ["| key | value |", "|---|---|"]
    for k, v in d.items():
        lines.append(f"| {k} | {v} |")
    return "\n".join(lines)


def _df_to_md_table(df: pd.DataFrame) -> str:
    columns = list(df.columns)
    lines = ["| " + " | ".join(columns) + " |", "|" + "---|" * len(columns)]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(row[c]) for c in columns) + " |")
    return "\n".join(lines)


def generate_markdown_report(outputs_dir: str | Path) -> str:
    outputs_dir = Path(outputs_dir)
    sections: list[str] = []

    sections.append("# Architecture Analysis Report\n")
    sections.append(
        "**Research question.** Does a shared Custom ResNet-18 backbone learn "
        "useful common visual features for both age estimation and dataset "
        "gender-label classification, and do task-specific bottleneck adapters "
        "and learned uncertainty-based loss balancing reduce negative transfer "
        "relative to independent per-task backbones and fixed loss weights? "
        "This report also compares the parametric multi-task model to a "
        "non-parametric k-NN baseline in the learned embedding space.\n"
    )
    sections.append(
        "**Scope note.** This is a research/education artifact only. "
        "Results depend entirely on the dataset used, label quality, "
        "demographic coverage, and the evaluation design below; no claim "
        "here should be read as evidence the underlying task (dataset "
        "gender-label prediction) generalizes beyond the specific dataset "
        "and labels used to train and evaluate the model.\n"
    )

    # Architecture summary
    sections.append("## Architecture Summary\n")
    sections.append(
        "- Backbone: manually implemented ResNet-18 (`src/models/custom_resnet.py`), "
        "block layout [2, 2, 2, 2], 512-d embedding.\n"
        "- Task adapters: residual bottleneck adapters "
        "(`z + up(dropout(gelu(down(z))))`), configurable bottleneck dim (default 128).\n"
        "- Heads: age quantile head (q10/q50/q90) and dataset gender-label softmax head.\n"
        "- Loss balancing: fixed weights or learned homoscedastic-uncertainty weighting.\n"
    )

    # Parameter comparison
    sections.append("## Parameter Comparison\n")
    param_path = outputs_dir / "architecture_analysis" / "parameter_comparison.json"
    param_data = _read_json(param_path)
    if param_data:
        for exp_name, breakdown in param_data.items():
            sections.append(f"**{exp_name}**\n\n{_dict_to_md_table(breakdown)}\n")
    else:
        sections.append(_MISSING.format(cmd="make architecture-report") + "\n")

    # Performance tables
    sections.append("## Performance Tables\n")
    ablation_path = outputs_dir / "architecture_analysis" / "ablation_table.csv"
    ablation_df = _read_csv(ablation_path)
    if ablation_df is not None:
        sections.append(_df_to_md_table(ablation_df) + "\n")
    else:
        sections.append(_MISSING.format(cmd="make experiments && make architecture-report") + "\n")

    knn_path = outputs_dir / "knn" / "parametric_vs_knn.csv"
    knn_df = _read_csv(knn_path)
    sections.append("### Parametric vs k-NN\n")
    if knn_df is not None:
        sections.append(_df_to_md_table(knn_df) + "\n")
    else:
        sections.append(_MISSING.format(cmd="make build-knn && make evaluate") + "\n")

    # Gradient interference
    sections.append("## Gradient Interference (Task-Gradient Cosine Similarity)\n")
    grad_path = outputs_dir / "architecture_analysis" / "gradient_cosine_similarity.json"
    grad_data = _read_json(grad_path)
    if grad_data:
        sections.append(_dict_to_md_table(grad_data) + "\n")
        sections.append(
            "Interpretation: positive mean cosine similarity suggests the age and "
            "gender-label gradients pull shared backbone weights in aligned "
            "directions; negative suggests conflict (negative transfer risk); "
            "near-zero suggests a weak relationship.\n"
        )
    else:
        sections.append(_MISSING.format(cmd="make architecture-report") + "\n")

    # Representation similarity
    sections.append("## Representation Similarity (Linear CKA)\n")
    cka_path = outputs_dir / "architecture_analysis" / "representation_similarity.json"
    cka_data = _read_json(cka_path)
    if cka_data:
        sections.append(_dict_to_md_table(cka_data) + "\n")
        sections.append(
            "Interpretation: CKA close to 1 means an adapter barely changes the "
            "shared representation; lower values indicate the adapter specializes "
            "the representation for its task. This is descriptive only and does "
            "not, by itself, establish which behavior yields better generalization.\n"
        )
    else:
        sections.append(_MISSING.format(cmd="make architecture-report") + "\n")

    # Robustness
    sections.append("## Robustness Results\n")
    robustness_path = outputs_dir / "robustness" / "robustness_results.csv"
    robustness_df = _read_csv(robustness_path)
    if robustness_df is not None:
        summary = robustness_df.groupby("corruption").agg(
            {"age_mae": "mean", "gender_accuracy": "mean", "abstention_rate": "mean"}
        ).reset_index()
        sections.append(_df_to_md_table(summary) + "\n")
    else:
        sections.append(_MISSING.format(cmd="make robustness") + "\n")

    # Grad-CAM
    sections.append("## Grad-CAM Observations\n")
    gradcam_dir = outputs_dir / "gradcam"
    gradcam_images = list(gradcam_dir.glob("*.png")) if gradcam_dir.exists() else []
    if gradcam_images:
        sections.append(
            f"{len(gradcam_images)} model-attention-visualization overlays generated in "
            f"`outputs/gradcam/`. Grad-CAM highlights which spatial regions most "
            "influenced a given prediction; it is a gradient-weighted activation "
            "visualization, not proof of causality or an explanation of reasoning.\n"
        )
    else:
        sections.append(_MISSING.format(cmd="make gradcam") + "\n")

    # Limitations
    sections.append("## Limitations\n")
    sections.append(
        "- Dataset gender-label predictions reflect labels defined by the source "
        "dataset's documentation, not a determination of gender identity.\n"
        "- Results depend on data quality, label noise, demographic coverage, "
        "image quality, and the specific train/val/test split used.\n"
        "- Age labels beyond the dataset's observed range are extrapolation and "
        "should be treated with reduced confidence.\n"
        "- Conformal calibration provides marginal (not per-group) coverage "
        "guarantees under the exchangeability assumption.\n"
        "- This system is not validated for, and must not be used for, "
        "employment, policing, surveillance, identity verification, medical "
        "diagnosis, admissions, insurance, or other high-impact decisions.\n"
    )

    # Conclusions
    sections.append("## Conclusions\n")
    if param_data and ablation_df is not None:
        sections.append(
            "Conclusions are drawn strictly from the tables above once experiments "
            "have been run; see `docs/architecture_analysis.md` for the fixed "
            "narrative template this section fills in once real results exist.\n"
        )
    else:
        sections.append(
            "No experiments have been run yet in this environment, so no "
            "empirical conclusion is stated here. Run `make experiments` "
            "followed by `make architecture-report` to populate this section "
            "with real results.\n"
        )

    return "\n".join(sections)


def save_report(outputs_dir: str | Path, docs_dir: str | Path) -> Path:
    report = generate_markdown_report(outputs_dir)
    out_path = Path(docs_dir) / "architecture_analysis_generated.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report, encoding="utf-8")
    return out_path
