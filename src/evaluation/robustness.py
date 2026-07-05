"""Deterministic robustness/corruption evaluation on the held-out test set.

Corruptions are applied to PIL images *before* the standard eval
transform, so severities are comparable across corruption types. All
randomness (noise, occlusion location, crop side) is seeded per-sample
for reproducibility.
"""

from __future__ import annotations

import io
import random

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

CORRUPTION_NAMES = (
    "gaussian_blur", "gaussian_noise", "low_resolution", "jpeg_compression",
    "low_brightness", "high_brightness", "partial_occlusion", "partial_crop",
)


def gaussian_blur(image: Image.Image, sigma: float, seed: int = 0) -> Image.Image:
    return image.filter(ImageFilter.GaussianBlur(radius=sigma))


def gaussian_noise(image: Image.Image, std: float, seed: int = 0) -> Image.Image:
    rng = np.random.default_rng(seed)
    array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    noisy = array + rng.normal(0, std, array.shape)
    noisy = np.clip(noisy, 0.0, 1.0)
    return Image.fromarray((noisy * 255).astype(np.uint8))


def low_resolution(image: Image.Image, scale_factor: float, seed: int = 0) -> Image.Image:
    w, h = image.size
    small = image.resize((max(1, int(w * scale_factor)), max(1, int(h * scale_factor))), Image.BILINEAR)
    return small.resize((w, h), Image.BILINEAR)


def jpeg_compression(image: Image.Image, quality: int, seed: int = 0) -> Image.Image:
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="JPEG", quality=int(quality))
    buffer.seek(0)
    return Image.open(buffer).convert("RGB")


def low_brightness(image: Image.Image, factor: float, seed: int = 0) -> Image.Image:
    return ImageEnhance.Brightness(image).enhance(factor)


def high_brightness(image: Image.Image, factor: float, seed: int = 0) -> Image.Image:
    return ImageEnhance.Brightness(image).enhance(factor)


def partial_occlusion(image: Image.Image, occlusion_fraction: float, seed: int = 0) -> Image.Image:
    rng = random.Random(seed)
    img = image.convert("RGB").copy()
    w, h = img.size
    box_w, box_h = int(w * occlusion_fraction ** 0.5), int(h * occlusion_fraction ** 0.5)
    x = rng.randint(0, max(0, w - box_w))
    y = rng.randint(0, max(0, h - box_h))
    array = np.asarray(img).copy()
    array[y : y + box_h, x : x + box_w, :] = 0
    return Image.fromarray(array)


def partial_crop(image: Image.Image, crop_fraction: float, seed: int = 0) -> Image.Image:
    rng = random.Random(seed)
    w, h = image.size
    side = rng.choice(["left", "right", "top", "bottom"])
    if side in ("left", "right"):
        cut = int(w * crop_fraction)
        box = (cut, 0, w, h) if side == "left" else (0, 0, w - cut, h)
    else:
        cut = int(h * crop_fraction)
        box = (0, cut, w, h) if side == "top" else (0, 0, w, h - cut)
    cropped = image.crop(box)
    return cropped.resize((w, h), Image.BILINEAR)


_CORRUPTION_FUNCS = {
    "gaussian_blur": gaussian_blur,
    "gaussian_noise": gaussian_noise,
    "low_resolution": low_resolution,
    "jpeg_compression": jpeg_compression,
    "low_brightness": low_brightness,
    "high_brightness": high_brightness,
    "partial_occlusion": partial_occlusion,
    "partial_crop": partial_crop,
}


def apply_corruption(image: Image.Image, name: str, param: float, seed: int = 0) -> Image.Image:
    if name not in _CORRUPTION_FUNCS:
        raise ValueError(f"Unknown corruption '{name}', expected one of {CORRUPTION_NAMES}")
    return _CORRUPTION_FUNCS[name](image, param, seed=seed)


def iter_corruption_configs(robustness_cfg: dict):
    """Yield (corruption_name, severity_level, param_value) tuples from the config."""
    for name, spec in robustness_cfg["corruptions"].items():
        for severity, param in zip(spec["severities"], spec["params"]):
            yield name, severity, param


def _predict_batch(model, images_tensor, device, gender_confidence_threshold: float):
    """Run one forward pass and return numpy prediction arrays for a batch of images."""
    import torch

    model.eval()
    with torch.no_grad():
        images_tensor = images_tensor.to(device)
        outputs = model(images_tensor)
        probs = torch.softmax(outputs["gender_logits"], dim=-1).cpu().numpy()
    q10 = outputs["age_output"]["q10"].cpu().numpy()
    q50 = outputs["age_output"]["q50"].cpu().numpy()
    q90 = outputs["age_output"]["q90"].cpu().numpy()
    predicted_class = probs.argmax(axis=1)
    confidence = probs.max(axis=1)
    abstain = confidence < gender_confidence_threshold
    return {
        "q10": q10, "q50": q50, "q90": q90,
        "predicted_class": predicted_class, "confidence": confidence, "abstain": abstain,
    }


def evaluate_condition(
    model,
    df,
    transform,
    device: str,
    gender_confidence_threshold: float,
    corruption_name: str | None,
    severity: int | None,
    param: float | None,
    seed: int,
    batch_size: int = 32,
):
    """Run the model over ``df`` with an optional corruption applied, returning a metrics dict.

    ``corruption_name=None`` evaluates the clean (uncorrupted) baseline.
    """
    import torch
    from PIL import Image

    from src.evaluation.metrics import (
        abstention_rate, age_mae, age_rmse, gender_accuracy, interval_coverage, mean_interval_width,
    )

    all_q10, all_q50, all_q90 = [], [], []
    all_pred_class, all_confidence, all_abstain = [], [], []
    ages, genders, age_valid, gender_valid = [], [], [], []

    rows = df.to_dict("records")
    for start in range(0, len(rows), batch_size):
        batch_rows = rows[start : start + batch_size]
        tensors = []
        for i, row in enumerate(batch_rows):
            with Image.open(row["image_path"]) as img:
                img = img.convert("RGB")
                if corruption_name is not None:
                    img = apply_corruption(img, corruption_name, param, seed=seed + start + i)
                tensors.append(transform(img))
            ages.append(row["age"])
            genders.append(row["gender_label"])
            age_valid.append(row["age"] == row["age"])  # not NaN
            gender_valid.append(row["gender_label"] == row["gender_label"])
        batch_tensor = torch.stack(tensors)
        preds = _predict_batch(model, batch_tensor, device, gender_confidence_threshold)
        all_q10.append(preds["q10"]); all_q50.append(preds["q50"]); all_q90.append(preds["q90"])
        all_pred_class.append(preds["predicted_class"])
        all_confidence.append(preds["confidence"])
        all_abstain.append(preds["abstain"])

    q10 = np.concatenate(all_q10); q50 = np.concatenate(all_q50); q90 = np.concatenate(all_q90)
    pred_class = np.concatenate(all_pred_class)
    confidence = np.concatenate(all_confidence)
    abstain = np.concatenate(all_abstain)
    ages = np.array(ages, dtype=np.float64)
    genders = np.array(genders, dtype=np.float64)
    age_valid = np.array(age_valid)
    gender_valid = np.array(gender_valid)

    metrics = {
        "corruption": corruption_name or "clean",
        "severity": severity or 0,
        "param": param,
        "n_samples": len(rows),
    }
    if age_valid.any():
        metrics["age_mae"] = age_mae(ages[age_valid], q50[age_valid])
        metrics["age_rmse"] = age_rmse(ages[age_valid], q50[age_valid])
        metrics["interval_coverage"] = interval_coverage(ages[age_valid], q10[age_valid], q90[age_valid])
        metrics["mean_interval_width"] = mean_interval_width(q10[age_valid], q90[age_valid])
    if gender_valid.any():
        combined_abstain = abstain[gender_valid]
        metrics["gender_accuracy"] = gender_accuracy(
            genders[gender_valid], pred_class[gender_valid], combined_abstain
        )
        metrics["abstention_rate"] = abstention_rate(abstain[gender_valid])
        metrics["mean_confidence"] = float(confidence[gender_valid].mean())
    return metrics
