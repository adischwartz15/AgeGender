"""Shared pytest fixtures: synthetic (non-real) images and configs for fast tests.

Synthetic data is used only for tests/smoke tests and must never be mixed
with real Kaggle experiment results.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from src.utils.config import load_full_config


@pytest.fixture
def synthetic_image_dir(tmp_path):
    """Create a small UTKFace-style synthetic image directory."""
    image_dir = tmp_path / "raw"
    image_dir.mkdir()
    rng = np.random.default_rng(0)
    records = []
    for i in range(40):
        age = int(rng.integers(1, 90))
        gender = int(rng.integers(0, 2))
        filename = f"{age}_{gender}_0_2017011617452{i:04d}.jpg"
        array = rng.integers(0, 255, size=(64, 64, 3), dtype=np.uint8)
        Image.fromarray(array).save(image_dir / filename)
        records.append({"age": age, "gender_label": gender, "path": str(image_dir / filename)})
    return image_dir, pd.DataFrame(records)


@pytest.fixture
def synthetic_metadata_df(synthetic_image_dir):
    _, records_df = synthetic_image_dir
    n = len(records_df)
    df = pd.DataFrame(
        {
            "image_path": records_df["path"],
            "age": records_df["age"].astype(float),
            "gender_label": records_df["gender_label"].astype(float),
            "race": 0,
            "subject_id": None,
            "split": None,
        }
    )
    return df


@pytest.fixture
def tiny_config():
    """A full config with a tiny model/backbone-friendly image size for fast CPU tests."""
    config = load_full_config()
    config["dataset"]["image_size"] = 32
    config["model"]["adapters"]["bottleneck_dim"] = 16
    config["model"]["age_head"]["hidden_dim"] = 16
    config["model"]["gender_head"]["hidden_dim"] = 16
    config["training"]["batch_size"] = 4
    config["training"]["num_workers"] = 0
    config["training"]["mixed_precision"] = False
    config["training"]["stages"]["stage_a"]["epochs"] = 1
    config["training"]["stages"]["stage_b"]["epochs"] = 1
    config["training"]["stages"]["stage_c"]["epochs"] = 1
    config["training"]["warm_up_from_scratch"]["epochs"] = 1
    config["training"]["early_stopping_patience"] = 100
    config["seed"] = 0
    return config
