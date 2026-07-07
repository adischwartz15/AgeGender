"""Tests for the new uncertainty/comparison plotting helpers.

These only check that each function runs without error and produces a
real, non-empty image file -- exact pixel content isn't asserted (that's
what would make these tests brittle for no real benefit).
"""

from __future__ import annotations

import numpy as np

from src.utils.visualization import (
    plot_coverage_width_tradeoff, plot_interval_width_by_bucket, plot_mean_std_bar,
    plot_parameter_latency_comparison,
)


def test_plot_interval_width_by_bucket_creates_file(tmp_path):
    out = plot_interval_width_by_bucket(["0-10", "10-20"], np.array([5.0, 8.0]), tmp_path / "width.png")
    assert out.exists()
    assert out.stat().st_size > 0


def test_plot_coverage_width_tradeoff_creates_file(tmp_path):
    out = plot_coverage_width_tradeoff(
        coverage_before=0.72, width_before=12.0, coverage_after=0.90, width_after=18.0,
        target_coverage=0.90, out_path=tmp_path / "tradeoff.png",
    )
    assert out.exists()
    assert out.stat().st_size > 0


def test_plot_parameter_latency_comparison_creates_file(tmp_path):
    out = plot_parameter_latency_comparison(
        ["simple_cnn", "custom_resnet18"], [4_000_000, 11_500_000], [1.5, 1.8], tmp_path / "param_latency.png",
    )
    assert out.exists()
    assert out.stat().st_size > 0


def test_plot_mean_std_bar_creates_file(tmp_path):
    out = plot_mean_std_bar(
        ["exp_c", "exp_d"], np.array([5.7, 5.5]), np.array([0.2, 0.15]), "Age MAE", tmp_path / "mean_std.png",
    )
    assert out.exists()
    assert out.stat().st_size > 0
