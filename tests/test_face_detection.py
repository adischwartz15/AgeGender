"""Tests for classical Haar-cascade face-region cropping.

Real face detection needs an actual photographic face -- Haar cascades
won't fire on synthetic noise -- so the "no face found" path is tested
against synthetic images, and the margin/crop arithmetic is tested by
monkeypatching detect_largest_face with a fixed bounding box.
"""

from __future__ import annotations

import numpy as np
from PIL import Image

import src.inference.face_detection as face_detection
from src.inference.face_detection import crop_to_face, detect_largest_face


def _solid_color_image(size=(200, 200), color=(120, 120, 120)) -> Image.Image:
    return Image.new("RGB", size, color)


def _noise_image(size=(200, 200), seed=0) -> Image.Image:
    rng = np.random.default_rng(seed)
    array = rng.integers(0, 255, size=(size[1], size[0], 3), dtype=np.uint8)
    return Image.fromarray(array)


def test_detect_largest_face_returns_none_on_blank_image():
    assert detect_largest_face(_solid_color_image()) is None


def test_detect_largest_face_returns_none_on_noise_image():
    assert detect_largest_face(_noise_image()) is None


def test_crop_to_face_falls_back_to_full_image_when_no_face_found():
    image = _solid_color_image(size=(150, 100))
    cropped, found = crop_to_face(image)
    assert found is False
    assert cropped.size == (150, 100)


def test_crop_to_face_applies_margin_around_detected_box(monkeypatch):
    monkeypatch.setattr(face_detection, "detect_largest_face", lambda image: (50, 50, 40, 40))
    image = _solid_color_image(size=(200, 200))

    cropped, found = crop_to_face(image, margin_ratio=0.5)
    assert found is True
    # box=(50,50,40,40), margin=0.5*40=20 each side -> crop (30,30)-(110,110)
    assert cropped.size == (80, 80)


def test_crop_to_face_clamps_margin_to_image_bounds(monkeypatch):
    """A face near the edge should not push the crop box outside the image."""
    monkeypatch.setattr(face_detection, "detect_largest_face", lambda image: (0, 0, 40, 40))
    image = _solid_color_image(size=(200, 200))

    cropped, found = crop_to_face(image, margin_ratio=1.0)
    assert found is True
    # margin=40 each side but left/top clamp to 0 -> crop (0,0)-(80,80)
    assert cropped.size == (80, 80)


def test_crop_to_face_converts_to_rgb():
    grayscale = Image.new("L", (100, 100), 128)
    cropped, _ = crop_to_face(grayscale)
    assert cropped.mode == "RGB"
