#!/usr/bin/env python
"""CLI: generate every diagram referenced by docs/model_visualizations.md and README.md.

Produces, from the repository's actual configuration and model code (never
invented numbers):

1. ``multitask_architecture.svg`` -- hand-drawn high-level architecture diagram
   (backbone -> shared embedding -> per-task adapters -> heads -> abstention).
2. ``backbone_comparison.svg`` -- hand-drawn diagram of the controlled
   SimpleCNN / PlainDeep18NoSkip / CustomResNet18 backbone comparison.
3. ``model_graph_<backbone>.svg`` (one per backbone in ``_BACKBONE_NAMES``) --
   automatically generated, layer-level computational graphs of the full
   ``MultiTaskFaceModel`` (backbone + adapters + heads), produced with
   ``torchview`` from a synthetic input tensor (no dataset or checkpoint
   required).
4. ``model_output_example.png`` -- one polished example of the system's
   real behavior on a synthetic demo image: a real forward pass through a
   trained checkpoint if one is available locally, honestly reported
   including a decline-to-predict outcome if the face detector finds no
   face (it never bypasses that safety behavior to fabricate a result).

Every diagram is derived from ``configs/*.yaml`` and ``src/models/*`` at run
time, so it stays in sync with the real implementation as long as this
script is re-run after an architecture/config change.

Usage:
    python scripts/generate_model_visualizations.py
    python scripts/generate_model_visualizations.py --output-dir docs/assets --device cpu
    python scripts/generate_model_visualizations.py --checkpoint checkpoints/exp_d_shared_adapters_learned_balance_best_balanced_score.pt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.config import CONFIG_DIR, REPO_ROOT, load_config
from src.utils.logging import get_logger

logger = get_logger("scripts.generate_model_visualizations")

_BACKBONE_NAMES = ("custom_resnet18", "simple_cnn", "plain_deep18_no_skip")

_GRAPHVIZ_INSTALL_MESSAGE = (
    "The 'graphviz' Python package (and the Graphviz system executable, 'dot') "
    "are required to generate diagrams. Install the Python package with:\n"
    "    pip install -r requirements-visualization.txt\n"
    "and make sure the Graphviz system package is installed and on PATH "
    "(e.g. 'apt install graphviz', 'brew install graphviz', or "
    "https://graphviz.org/download/ on Windows)."
)

_TORCHVIEW_INSTALL_MESSAGE = (
    "The 'torchview' Python package is required to generate the detailed "
    "computational graphs. Install it with:\n"
    "    pip install -r requirements-visualization.txt\n"
    "(this also requires the Graphviz Python package and system executable; "
    "see the error above/below if those are missing too)."
)


def _require_graphviz():
    try:
        import graphviz
    except ImportError as exc:  # pragma: no cover - exercised only when dependency missing
        raise SystemExit(f"Cannot generate diagram: {_GRAPHVIZ_INSTALL_MESSAGE}") from exc
    return graphviz


# --------------------------------------------------------------------------
# Visualization 1: high-level multi-task architecture diagram
# --------------------------------------------------------------------------

def build_multitask_architecture_diagram(config: dict, output_path: Path) -> Path:
    """Render the high-level architecture diagram to ``output_path`` (SVG)."""
    graphviz = _require_graphviz()

    model_cfg = config["model"]
    image_size = config["dataset"]["image_size"]
    embedding_dim = model_cfg["backbone"]["embedding_dim"]
    bottleneck_dim = model_cfg["adapters"]["bottleneck_dim"]
    age_min = model_cfg["age_head"]["age_min"]
    age_max = model_cfg["age_head"]["age_max"]
    num_classes = model_cfg["gender_head"]["num_classes"]
    threshold = model_cfg["gender_head"]["confidence_threshold"]

    g = graphviz.Digraph("multitask_architecture", format="svg")
    g.attr(
        rankdir="TB", bgcolor="white", splines="spline", nodesep="0.45", ranksep="0.55",
        fontname="Helvetica,Arial,sans-serif",
    )
    g.attr("node", fontname="Helvetica,Arial,sans-serif", fontsize="13", shape="box",
           style="rounded,filled", color="#2b2d42", fontcolor="#111111", penwidth="1.3")
    g.attr("edge", fontname="Helvetica,Arial,sans-serif", fontsize="11", color="#2b2d42", penwidth="1.2")

    shared_fill = "#dbe9f6"
    age_fill = "#fdebd3"
    gender_fill = "#e2f0d9"
    decision_fill = "#f2f2f2"
    accept_fill = "#d7ecd1"
    abstain_fill = "#f8d7da"

    g.node("input", f"Input face image\n(RGB, resized to {image_size}x{image_size})",
           fillcolor="#eef2f7")
    g.node("backbone",
           "Custom ResNet-18 backbone\n(manually implemented, no torchvision/timm,\n"
           "no pretrained weights)\nBasicBlock residual blocks, layout [2, 2, 2, 2]",
           fillcolor=shared_fill)
    g.node("shared_z", f"Shared embedding z\n({embedding_dim}-dim)",
           fillcolor="#c9d6e3", penwidth="2.0")

    g.edge("input", "backbone")
    g.edge("backbone", "shared_z")

    g.edge("shared_z", "age_adapter")
    g.edge("shared_z", "gender_adapter")

    with g.subgraph(name="cluster_gender") as c:
        c.attr(label="Dataset gender-label task (task-specific)", style="dashed", color="#2e6f40",
               fontname="Helvetica,Arial,sans-serif", fontsize="12", labeljust="l")
        c.node("gender_adapter",
               "Gender-Label Adapter -- residual bottleneck\n"
               f"z + Up(Dropout(GELU(Down(z))))\n{embedding_dim} -> {bottleneck_dim} -> {embedding_dim}",
               fillcolor=gender_fill)
        c.node("gender_head", f"Classification Head\n(hidden 128, dropout 0.1, {num_classes} classes)",
               fillcolor=gender_fill)
        c.node("gender_logits", f"Logits ({num_classes})\n(raw, used for cross-entropy at train time)",
               fillcolor=gender_fill)
        c.node("softmax", "Softmax\n(inference/evaluation only)", fillcolor="#eef7ea")
        c.node("threshold", f"max probability\n>= {threshold:.2f} ?", shape="diamond", fillcolor=decision_fill,
               fontsize="11")
        c.node("accepted", "Accepted dataset\ngender label", fillcolor=accept_fill)
        c.node("notsure", "\"Not sure\"\n(abstain)", fillcolor=abstain_fill)
        c.edge("gender_adapter", "gender_head")
        c.edge("gender_head", "gender_logits")
        c.edge("gender_logits", "softmax")
        c.edge("softmax", "threshold")
        c.edge("threshold", "accepted", label="yes")
        c.edge("threshold", "notsure", label="no")

    with g.subgraph(name="cluster_age") as c:
        c.attr(label="Age task (task-specific)", style="dashed", color="#b5651d",
               fontname="Helvetica,Arial,sans-serif", fontsize="12", labeljust="l")
        c.node("age_adapter",
               "Age Adapter -- residual bottleneck\n"
               f"z + Up(Dropout(GELU(Down(z))))\n{embedding_dim} -> {bottleneck_dim} -> {embedding_dim}",
               fillcolor=age_fill)
        c.node("age_head", "Age Quantile Head\n(hidden 128, dropout 0.1)", fillcolor=age_fill)
        c.node("age_out",
               f"q10, q50, q90\n(safe parameterization: q10 <= q50 <= q90;\n"
               f"clamped to [{age_min}, {age_max}] for display)",
               fillcolor=age_fill)
        c.edge("age_adapter", "age_head")
        c.edge("age_head", "age_out")

    g.node(
        "note",
        "Softmax, the confidence threshold, and split-conformal calibration of the age\n"
        "interval are all applied at inference/evaluation time -- not learned inside the network.\n"
        "Grad-CAM, k-NN comparison, and face detection are separate diagnostic/preprocessing\n"
        "components, not part of this forward pass.",
        shape="note", fillcolor="#fffbe6", fontsize="10", style="filled", color="#8a8a8a",
    )
    g.edge("age_out", "note", style="invis")
    g.edge("notsure", "note", style="invis")
    g.edge("accepted", "note", style="invis")

    return _render(g, output_path)


# --------------------------------------------------------------------------
# Visualization 2: controlled backbone comparison diagram
# --------------------------------------------------------------------------

def build_backbone_comparison_diagram(output_path: Path) -> Path:
    graphviz = _require_graphviz()

    g = graphviz.Digraph("backbone_comparison", format="svg")
    g.attr(rankdir="TB", bgcolor="white", splines="spline", nodesep="0.5", ranksep="0.55",
           fontname="Helvetica,Arial,sans-serif")
    g.attr("node", fontname="Helvetica,Arial,sans-serif", fontsize="13", shape="box",
           style="rounded,filled", color="#2b2d42", fontcolor="#111111", penwidth="1.3")
    g.attr("edge", fontname="Helvetica,Arial,sans-serif", fontsize="11", color="#2b2d42", penwidth="1.2")

    g.node("top", "Same input data and same fixed data split", fillcolor="#eef2f7")

    with g.subgraph() as s:
        s.attr(rank="same")
        s.node("simple_cnn", "SimpleCNN\nCompact non-residual baseline\n"
                              "(efficiency/accuracy trade-off vs. ResNet)", fillcolor="#f6e8d9")
        s.node("plain_deep18", "PlainDeep18NoSkip\nResNet-matched depth and stage widths,\n"
                                "residual additions removed\n"
                                "(controlled skip-connection ablation)", fillcolor="#e8dff5")
        s.node("custom_resnet18", "Custom ResNet-18\nMain residual backbone\n"
                                   "(the project's deployed backbone)", fillcolor="#dbe9f6")

    g.node(
        "bottom",
        "Same adapters\nSame task heads\nSame losses\n"
        "Same data split\nSame training protocol\nSame evaluation protocol",
        fillcolor="#eef2f7",
    )

    for backbone_node in ("simple_cnn", "plain_deep18", "custom_resnet18"):
        g.edge("top", backbone_node)
        g.edge(backbone_node, "bottom")

    g.node(
        "note",
        "SimpleCNN differs from Custom ResNet-18 in BOTH depth and channel width, so that pairing\n"
        "alone does not isolate residual (skip) connections -- it is a compact-vs-full efficiency\n"
        "comparison. PlainDeep18NoSkip copies Custom ResNet-18's stem, stage widths, and block layout\n"
        "exactly, with only the residual additions removed -- that pairing is the controlled ablation\n"
        "of skip connections specifically. See docs/backbone_comparison.md.",
        shape="note", fillcolor="#fffbe6", fontsize="10", style="filled", color="#8a8a8a",
    )
    g.edge("bottom", "note", style="invis")

    return _render(g, output_path)


def _render(graph, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    stem = output_path.with_suffix("")
    rendered = graph.render(filename=str(stem), format="svg", cleanup=True)
    rendered_path = Path(rendered)
    if rendered_path != output_path:
        rendered_path.replace(output_path)
    return output_path


# --------------------------------------------------------------------------
# Visualization 3: detailed automatic computational graphs (torchview)
# --------------------------------------------------------------------------

def build_model_graph(backbone_name: str, output_path: Path, device: str) -> Path:
    """Build the full ``MultiTaskFaceModel`` (given ``backbone_name``) and render its
    torchview computational graph -- backbone, adapters, and both task heads --
    from a synthetic input tensor. No dataset or checkpoint is required or loaded.
    """
    try:
        import torch
        from torchview import draw_graph
    except ImportError as exc:
        raise SystemExit(f"Cannot generate model graph: {_TORCHVIEW_INSTALL_MESSAGE}") from exc
    _require_graphviz()

    from src.models.multitask_model import MultiTaskFaceModel

    model_config = load_config(
        CONFIG_DIR / "data.yaml", CONFIG_DIR / "model.yaml",
        overrides={"model": {"backbone": {"name": backbone_name}}},
    )
    model = MultiTaskFaceModel(model_config)
    model.eval()

    image_size = model_config["dataset"]["image_size"]
    in_channels = model_config["model"]["backbone"].get("in_channels", 3)
    synthetic_input = torch.zeros(1, in_channels, image_size, image_size, device=device)

    graph = draw_graph(
        model,
        input_data=synthetic_input,
        graph_name=f"model_graph_{backbone_name}",
        expand_nested=True,
        depth=3,
        device=device,
        save_graph=False,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    stem = output_path.with_suffix("")
    rendered = graph.visual_graph.render(filename=str(stem), format="svg", cleanup=True)
    rendered_path = Path(rendered)
    if rendered_path != output_path:
        rendered_path.replace(output_path)
    return output_path


# --------------------------------------------------------------------------
# Visualization 4: polished prediction-output example
# --------------------------------------------------------------------------

def build_prediction_example(
    output_path: Path, checkpoint: Path | None, device: str, demo_image_path: Path | None = None,
) -> Path:
    """Render one polished example of the deployed system's real behavior.

    Uses the repository's own ``Predictor`` end-to-end. Prefers an already
    committed synthetic demo image. If a compatible checkpoint is available
    locally, this is a genuine forward pass (never fabricated numbers); if
    the classical face detector declines to find a face in the synthetic
    demo image (a real, documented possibility for these cartoon-style
    images -- see data/demo_images/README.md), that decline is rendered
    honestly rather than bypassed. If no checkpoint is available at all, an
    explicitly labeled illustrative layout is rendered instead, with no
    numeric values that could be mistaken for a real result.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    demo_dir = REPO_ROOT / "data" / "demo_images"
    if demo_image_path is None:
        candidates = sorted(demo_dir.glob("*.png"))
        if not candidates:
            raise SystemExit(
                f"No demo images found under {demo_dir}. Run "
                "'python scripts/generate_demo_images.py' first."
            )
        demo_image_path = candidates[0]

    image = Image.open(demo_image_path).convert("RGB")

    checkpoint_path = checkpoint
    if checkpoint_path is None:
        api_config = load_config(CONFIG_DIR / "api.yaml")["api"]
        candidate = REPO_ROOT / api_config["active_checkpoint"]
        checkpoint_path = candidate if candidate.exists() else None

    if checkpoint_path is None or not Path(checkpoint_path).exists():
        _render_illustrative_layout(image, demo_image_path, output_path)
        return output_path

    from src.inference.artifacts import load_all_artifacts
    from src.inference.predictor import Predictor

    api_config = load_config(CONFIG_DIR / "api.yaml")["api"]
    api_config = dict(api_config)
    api_config["active_checkpoint"] = str(checkpoint_path)
    artifacts = load_all_artifacts(api_config, device=device)
    if artifacts.model is None:
        _render_illustrative_layout(image, demo_image_path, output_path)
        return output_path

    predictor = Predictor(artifacts, api_config, device=device)
    result = predictor.predict(image, include_gradcam=True, include_knn=False)

    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.6), gridspec_kw={"width_ratios": [1, 1.35]})
    axes[0].imshow(image)
    axes[0].set_title("Input (synthetic demo image)", fontsize=10)
    axes[0].axis("off")
    axes[1].axis("off")

    lines = [
        f"Checkpoint: {artifacts.checkpoint_name} (epoch {artifacts.checkpoint_epoch})",
        f"Face detected: {result.face_detected}",
        "",
    ]
    if result.age is not None and result.gender is not None:
        lines += [
            f"Estimated age (q50): {result.age.q50:.1f}",
            f"Raw interval [q10, q90]: [{result.age.q10:.1f}, {result.age.q90:.1f}]",
            (
                f"Calibrated interval: [{result.age.q10_calibrated:.1f}, {result.age.q90_calibrated:.1f}]"
                if result.age.is_calibrated
                else "Calibrated interval: not available (no calibration artifact)"
            ),
            f"Dataset gender label: {result.gender.predicted_label or 'Not sure'}",
            f"Confidence: {result.gender.confidence:.2f}",
            f"Abstained: {result.gender.abstained}",
        ]
    else:
        lines += [
            "Model declined to predict.",
            "No face was found by the classical Haar-cascade face detector,",
            "so age and dataset gender-label predictions were not generated",
            "(this is the system's real 'decline rather than guess' behavior,",
            "not an error) -- see src/inference/face_detection.py.",
        ]
    lines += [
        "",
        f"Image-quality diagnostics: {result.quality.width}x{result.quality.height}px, "
        f"brightness={result.quality.brightness:.2f}, contrast={result.quality.contrast:.2f}, "
        f"blur_score={result.quality.blur_score:.0f}",
    ]
    if result.warnings:
        lines.append("")
        lines.append("Warnings:")
        for warning in result.warnings:
            lines.append(f"- {warning}")

    axes[1].text(0.0, 1.0, "\n".join(_wrap(line, 62) for line in lines), fontsize=8.5,
                 va="top", ha="left", family="monospace", transform=axes[1].transAxes)

    import textwrap

    caption = textwrap.fill(
        "Real run of the repository's Predictor (checkpoint "
        f"{artifacts.checkpoint_name}) on a committed synthetic demo image -- "
        "not a photograph of a real person. This is a genuine model/decline "
        "outcome, not a fabricated result.",
        width=110,
    )
    fig.tight_layout(rect=(0, 0.09, 1, 1))
    fig.text(0.02, 0.02, caption, fontsize=8, va="bottom", ha="left")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=170)
    plt.close(fig)
    return output_path


def _wrap(line: str, width: int) -> str:
    if len(line) <= width:
        return line
    import textwrap

    return "\n".join(textwrap.wrap(line, width=width, subsequent_indent="  ")) or line


def _render_illustrative_layout(image, demo_image_path: Path, output_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.6), gridspec_kw={"width_ratios": [1, 1.35]})
    axes[0].imshow(image)
    axes[0].set_title("Input (synthetic demo image)", fontsize=10)
    axes[0].axis("off")
    axes[1].axis("off")

    lines = [
        "ILLUSTRATIVE OUTPUT LAYOUT -- NOT A MODEL RESULT",
        "",
        "No trained checkpoint was available when this figure was generated,",
        "so every field below is a placeholder for where real output would",
        "appear, not an actual prediction.",
        "",
        "Estimated age (q50): <q50>",
        "Raw interval [q10, q90]: [<q10>, <q90>]",
        "Calibrated interval: [<q10_cal>, <q90_cal>]",
        "Dataset gender label: <label> | \"Not sure\"",
        "Confidence: <confidence>",
        "Abstained: <yes/no>",
        "Image-quality diagnostics: <brightness, contrast, blur, resolution>",
    ]
    axes[1].text(0.0, 1.0, "\n".join(lines), fontsize=9, va="top", ha="left",
                 family="monospace", transform=axes[1].transAxes, color="#7a1f1f")

    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.text(
        0.02, 0.02,
        "Illustrative output layout -- not a model result (no checkpoint was available locally).",
        fontsize=8, va="bottom", ha="left", color="#7a1f1f",
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=170)
    plt.close(fig)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", default="docs/assets", help="Directory to write generated assets into")
    parser.add_argument("--device", default="cpu", help="Device for the (CPU-friendly) structural forward passes")
    parser.add_argument(
        "--checkpoint", default=None,
        help="Checkpoint to use for the prediction-example figure (default: configs/api.yaml's active_checkpoint, "
             "if it exists locally; otherwise an illustrative layout is generated instead)",
    )
    parser.add_argument(
        "--demo-image", default=None,
        help="Path to the demo image used for the prediction-example figure "
             "(default: the first PNG under data/demo_images/)",
    )
    parser.add_argument(
        "--skip-graphs", action="store_true",
        help="Skip the torchview computational graphs (diagrams 1/2/4 only; useful if torchview isn't installed)",
    )
    args = parser.parse_args()

    output_dir = (REPO_ROOT / args.output_dir).resolve() if not Path(args.output_dir).is_absolute() else Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = load_config(CONFIG_DIR / "data.yaml", CONFIG_DIR / "model.yaml")

    generated: list[Path] = []

    logger.info("Generating high-level multi-task architecture diagram...")
    generated.append(build_multitask_architecture_diagram(config, output_dir / "multitask_architecture.svg"))

    logger.info("Generating controlled backbone comparison diagram...")
    generated.append(build_backbone_comparison_diagram(output_dir / "backbone_comparison.svg"))

    if not args.skip_graphs:
        for backbone_name in _BACKBONE_NAMES:
            logger.info("Generating detailed computational graph for '%s'...", backbone_name)
            generated.append(
                build_model_graph(backbone_name, output_dir / f"model_graph_{backbone_name}.svg", args.device)
            )

    logger.info("Generating prediction-output example figure...")
    checkpoint = Path(args.checkpoint) if args.checkpoint else None
    demo_image = Path(args.demo_image) if args.demo_image else None
    generated.append(
        build_prediction_example(output_dir / "model_output_example.png", checkpoint, args.device, demo_image)
    )

    print("Generated visualization assets:")
    for path in generated:
        print(f"  - {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
