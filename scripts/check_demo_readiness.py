#!/usr/bin/env python
"""CLI: verify the API has everything a live demo needs before launching it.

Checks, against the same `configs/api.yaml` the FastAPI backend itself
reads: a trained checkpoint at `api.active_checkpoint`, a conformal
calibration artifact under `api.calibration_dir`, and (informationally,
not required) a k-NN index and the synthetic demo images. Exits non-zero
with an actionable message if anything required is missing, so
`scripts/run_demo.py` never launches a demo that will silently fail on
the first upload.

Usage:
    python scripts/check_demo_readiness.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluation.calibration import load_calibration
from src.utils.config import CONFIG_DIR, REPO_ROOT, load_config
from src.utils.logging import get_logger

logger = get_logger("scripts.check_demo_readiness")


def check_demo_readiness(api_config: dict) -> tuple[bool, list[str]]:
    """Return (ready, messages). ``ready`` requires a checkpoint and calibration artifact."""
    messages = []
    ready = True

    checkpoint_path = REPO_ROOT / api_config["active_checkpoint"]
    if checkpoint_path.exists():
        messages.append(f"[OK]   Checkpoint found: {checkpoint_path}")
    else:
        ready = False
        messages.append(
            f"[FAIL] No checkpoint at {checkpoint_path}. Run 'make train' or 'make experiments', "
            "then point configs/api.yaml's active_checkpoint at the resulting file."
        )

    calibration_dir = REPO_ROOT / api_config["calibration_dir"]
    if load_calibration(calibration_dir) is not None:
        messages.append(f"[OK]   Calibration artifact found under: {calibration_dir}")
    else:
        ready = False
        messages.append(
            f"[FAIL] No conformal calibration artifact under {calibration_dir}. "
            "Run 'make calibrate CHECKPOINT=<your checkpoint>' first."
        )

    # Informational only -- neither blocks the demo from launching.
    knn_path = REPO_ROOT / api_config["knn_index_dir"] / "knn_baseline.pkl"
    if knn_path.exists():
        messages.append(f"[OK]   k-NN index found: {knn_path}")
    else:
        messages.append(f"[WARN] No k-NN index at {knn_path} (optional). Run 'make build-knn' to add it.")

    demo_images_dir = REPO_ROOT / "data" / "demo_images"
    demo_images = list(demo_images_dir.glob("*.png")) if demo_images_dir.exists() else []
    if demo_images:
        messages.append(f"[OK]   {len(demo_images)} synthetic demo image(s) found in {demo_images_dir}")
    else:
        messages.append(
            f"[WARN] No demo images in {demo_images_dir} (optional). "
            "Run 'python scripts/generate_demo_images.py' to add some."
        )

    return ready, messages


def main() -> int:
    api_config = load_config(CONFIG_DIR / "api.yaml")["api"]
    ready, messages = check_demo_readiness(api_config)

    print("Demo readiness check")
    print("=" * 40)
    for message in messages:
        print(message)
    print("=" * 40)
    print("READY" if ready else "NOT READY")
    return 0 if ready else 1


if __name__ == "__main__":
    raise SystemExit(main())
