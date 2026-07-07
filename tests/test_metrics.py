"""Tests for the per-bucket uncertainty metrics and interval-example selection
used by the uncertainty evaluation report section."""

from __future__ import annotations

import numpy as np

from src.evaluation.metrics import age_uncertainty_by_bucket, select_interval_examples


def test_age_uncertainty_by_bucket_computes_coverage_and_width_per_bucket():
    y_true = np.array([5.0, 5.0, 25.0, 25.0])
    q10 = np.array([0.0, 0.0, 20.0, 30.0])   # bucket 0-10: both covered; bucket 20-30: one covered, one not
    q50 = np.array([5.0, 5.0, 25.0, 25.0])
    q90 = np.array([10.0, 10.0, 30.0, 35.0])

    result = age_uncertainty_by_bucket(y_true, q10, q50, q90, bucket_edges=[0, 10, 20, 30, 200])
    assert result["0-10"]["count"] == 2
    assert result["0-10"]["coverage"] == 1.0
    assert result["20-30"]["count"] == 2
    assert result["20-30"]["coverage"] == 0.5  # second sample: q10=30 > y_true=25 -> not covered
    assert result["10-20"]["count"] == 0
    assert result["10-20"]["mae"] is None
    assert result["10-20"]["coverage"] is None


def test_age_uncertainty_by_bucket_mean_width_matches_manual_computation():
    y_true = np.array([5.0, 6.0])
    q10 = np.array([2.0, 3.0])
    q50 = np.array([5.0, 6.0])
    q90 = np.array([8.0, 10.0])
    result = age_uncertainty_by_bucket(y_true, q10, q50, q90, bucket_edges=[0, 10, 200])
    expected_mean_width = np.mean([8.0 - 2.0, 10.0 - 3.0])
    assert abs(result["0-10"]["mean_width"] - expected_mean_width) < 1e-9


def test_select_interval_examples_picks_narrowest_and_widest():
    image_paths = np.array([f"img_{i}.jpg" for i in range(5)])
    y_true = np.array([20.0, 25.0, 30.0, 35.0, 40.0])
    q10 = np.array([19.0, 20.0, 10.0, 34.0, 5.0])
    q50 = y_true.copy()
    q90 = np.array([21.0, 30.0, 50.0, 36.0, 75.0])
    # widths: [2, 10, 40, 2, 70] -- two ties at width=2 (img_0, img_3), widest is img_4 (70)

    result = select_interval_examples(image_paths, y_true, q10, q50, q90, k=2)
    narrow_paths = {r["image_path"] for r in result["narrowest"]}
    wide_paths = {r["image_path"] for r in result["widest"]}
    assert narrow_paths == {"img_0.jpg", "img_3.jpg"}
    assert "img_4.jpg" in wide_paths
    assert result["widest"][0]["image_path"] == "img_4.jpg"
    assert result["widest"][0]["width"] == 70.0


def test_select_interval_examples_handles_fewer_samples_than_k():
    image_paths = np.array(["a.jpg", "b.jpg"])
    y_true = np.array([10.0, 20.0])
    q10 = np.array([8.0, 15.0])
    q50 = y_true.copy()
    q90 = np.array([12.0, 25.0])
    result = select_interval_examples(image_paths, y_true, q10, q50, q90, k=5)
    assert len(result["narrowest"]) == 2
    assert len(result["widest"]) == 2


def test_select_interval_examples_record_fields():
    image_paths = np.array(["only.jpg"])
    y_true = np.array([42.0])
    q10 = np.array([40.0])
    q50 = np.array([42.0])
    q90 = np.array([45.0])
    result = select_interval_examples(image_paths, y_true, q10, q50, q90, k=1)
    record = result["narrowest"][0]
    assert record == {
        "image_path": "only.jpg", "true_age": 42.0, "q10": 40.0, "q50": 42.0, "q90": 45.0, "width": 5.0,
    }
