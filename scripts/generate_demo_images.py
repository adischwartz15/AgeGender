#!/usr/bin/env python
"""CLI: procedurally generate a small set of synthetic placeholder "face" images for demo mode.

These are drawn from scratch with PIL primitives (ovals, arcs, lines) --
no photograph, no real person, and no dataset content is used or copied.
They exist so `scripts/run_demo.py` has something repository-safe to show
during a live course demonstration without ever committing UTKFace (or any
other license-restricted dataset) images to git.

Because these are cartoon-style shapes rather than photographic human
faces, the classical Haar-cascade face detector (src/inference/face_detection.py)
may or may not recognize a "face" in any given one -- that is expected, and
is itself a valid demonstration of the system's "decline to predict when no
face is detected" safety behavior. For a demo that reliably shows real
predictions, supply your own consented photo at demo time instead.

Usage:
    python scripts/generate_demo_images.py
"""

from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.config import REPO_ROOT
from src.utils.logging import get_logger

logger = get_logger("scripts.generate_demo_images")

# (skin_tone, hair_color, mouth_curve) per synthetic placeholder -- purely
# cosmetic variety, not modeled on any real individual or demographic claim.
_FACE_SPECS = [
    {"skin": (240, 200, 170), "hair": (60, 40, 30), "mouth_curve": 18, "name": "demo_face_1"},
    {"skin": (200, 150, 110), "hair": (20, 20, 20), "mouth_curve": -6, "name": "demo_face_2"},
    {"skin": (255, 224, 189), "hair": (200, 180, 60), "mouth_curve": 24, "name": "demo_face_3"},
    {"skin": (140, 100, 75), "hair": (10, 10, 10), "mouth_curve": 10, "name": "demo_face_4"},
    {"skin": (225, 190, 150), "hair": (120, 80, 50), "mouth_curve": 0, "name": "demo_face_5"},
]

_SIZE = 256


def _draw_synthetic_face(skin: tuple, hair: tuple, mouth_curve: int) -> Image.Image:
    img = Image.new("RGB", (_SIZE, _SIZE), (235, 235, 235))
    draw = ImageDraw.Draw(img)

    cx, cy = _SIZE // 2, _SIZE // 2 + 10
    face_w, face_h = 90, 110

    # Hair (simple arc behind the head).
    draw.ellipse([cx - face_w - 6, cy - face_h - 20, cx + face_w + 6, cy - 10], fill=hair)
    # Face oval.
    draw.ellipse([cx - face_w, cy - face_h, cx + face_w, cy + face_h], fill=skin, outline=(90, 60, 45))
    # Eyes.
    for dx in (-38, 38):
        draw.ellipse([cx + dx - 14, cy - 20 - 10, cx + dx + 14, cy - 20 + 10], fill=(255, 255, 255), outline=(60, 60, 60))
        draw.ellipse([cx + dx - 5, cy - 20 - 5, cx + dx + 5, cy - 20 + 5], fill=(30, 30, 30))
    # Eyebrows.
    for dx in (-38, 38):
        draw.line([cx + dx - 16, cy - 42, cx + dx + 16, cy - 40], fill=(60, 40, 30), width=4)
    # Nose.
    draw.line([cx, cy - 5, cx - 6, cy + 20], fill=(150, 110, 85), width=3)
    # Mouth (arc bowed up or down depending on mouth_curve).
    draw.arc([cx - 30, cy + 30 - mouth_curve, cx + 30, cy + 60 - mouth_curve], start=0, end=180, fill=(120, 40, 40), width=4)

    return img


def main() -> int:
    out_dir = REPO_ROOT / "data" / "demo_images"
    out_dir.mkdir(parents=True, exist_ok=True)

    for spec in _FACE_SPECS:
        img = _draw_synthetic_face(spec["skin"], spec["hair"], spec["mouth_curve"])
        out_path = out_dir / f"{spec['name']}.png"
        img.save(out_path)
        logger.info("Wrote %s", out_path)

    print(f"Generated {len(_FACE_SPECS)} synthetic demo images in {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
