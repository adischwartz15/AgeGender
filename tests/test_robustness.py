"""Tests for deterministic robustness corruption functions.

Covers the full required set: blur, brightness, contrast, Gaussian
noise, JPEG compression, partial occlusion, resize degradation
(low_resolution), and grayscale conversion.
"""

from __future__ import annotations

import numpy as np
from PIL import Image

from src.evaluation.robustness import (
    CORRUPTION_NAMES, apply_corruption, gaussian_blur, gaussian_noise, grayscale, high_brightness,
    high_contrast, iter_corruption_configs, jpeg_compression, low_brightness, low_contrast, low_resolution,
    partial_crop, partial_occlusion,
)


def _sample_image(size=(64, 64)) -> Image.Image:
    rng = np.random.default_rng(0)
    array = rng.integers(0, 255, size=(size[1], size[0], 3), dtype=np.uint8)
    return Image.fromarray(array)


def test_all_required_corruption_types_are_registered():
    required = {
        "gaussian_blur", "gaussian_noise", "low_resolution", "jpeg_compression",
        "low_brightness", "high_brightness", "low_contrast", "high_contrast",
        "grayscale", "partial_occlusion",
    }
    assert required <= set(CORRUPTION_NAMES)


def test_each_corruption_preserves_image_size():
    image = _sample_image((80, 60))
    corruptions = [
        (gaussian_blur, 1.5), (gaussian_noise, 0.1), (low_resolution, 0.3), (jpeg_compression, 20),
        (low_brightness, 0.5), (high_brightness, 1.6), (low_contrast, 0.5), (high_contrast, 1.8),
        (grayscale, 0.7), (partial_occlusion, 0.2), (partial_crop, 0.2),
    ]
    for fn, param in corruptions:
        result = fn(image, param, seed=1)
        assert result.size == (80, 60), f"{fn.__name__} changed image size"


def test_grayscale_blend_factor_one_removes_all_color_variation():
    image = _sample_image()
    result = grayscale(image, blend_factor=1.0)
    array = np.asarray(result)
    # Fully desaturated: R, G, B channels should be identical per pixel.
    assert np.allclose(array[..., 0], array[..., 1])
    assert np.allclose(array[..., 1], array[..., 2])


def test_grayscale_blend_factor_zero_is_original_image():
    image = _sample_image()
    result = grayscale(image, blend_factor=0.0)
    assert np.array_equal(np.asarray(result), np.asarray(image.convert("RGB")))


def test_grayscale_clamps_out_of_range_blend_factor():
    image = _sample_image()
    over = grayscale(image, blend_factor=1.5)
    under = grayscale(image, blend_factor=-0.5)
    fully_gray = grayscale(image, blend_factor=1.0)
    original = np.asarray(image.convert("RGB"))
    assert np.array_equal(np.asarray(over), np.asarray(fully_gray))
    assert np.array_equal(np.asarray(under), original)


def test_low_contrast_and_high_contrast_move_in_opposite_directions():
    image = _sample_image()
    baseline_std = np.asarray(image.convert("L"), dtype=np.float64).std()
    low = np.asarray(low_contrast(image, 0.3).convert("L"), dtype=np.float64).std()
    high = np.asarray(high_contrast(image, 2.0).convert("L"), dtype=np.float64).std()
    assert low < baseline_std
    assert high > baseline_std


def test_apply_corruption_dispatches_new_corruption_types():
    image = _sample_image()
    for name, param in (("low_contrast", 0.5), ("high_contrast", 1.5), ("grayscale", 0.5)):
        result = apply_corruption(image, name, param, seed=0)
        assert result.size == image.size


def test_apply_corruption_rejects_unknown_name():
    import pytest

    with pytest.raises(ValueError):
        apply_corruption(_sample_image(), "not_a_real_corruption", 1.0)


def test_iter_corruption_configs_yields_new_corruption_types():
    robustness_cfg = {
        "corruptions": {
            "low_contrast": {"severities": [1, 2], "params": [0.7, 0.5]},
            "grayscale": {"severities": [1], "params": [0.4]},
        }
    }
    configs = list(iter_corruption_configs(robustness_cfg))
    names = {name for name, _, _ in configs}
    assert names == {"low_contrast", "grayscale"}
    assert len(configs) == 3
