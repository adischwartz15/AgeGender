"""Manual image transforms (PIL/NumPy/PyTorch only -- no torchvision).

Kept dependency-light and dependency-explicit: everything here is a plain
function or small class operating on PIL images / NumPy arrays, so there is
no reliance on any prebuilt vision-transform library.
"""

from __future__ import annotations

import random

import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageFilter

IMAGENET_MEAN = (0.485, 0.456, 0.406)  # standard RGB normalization constants, not pretrained weights
IMAGENET_STD = (0.229, 0.224, 0.225)


def to_tensor(image: Image.Image) -> torch.Tensor:
    """Convert a PIL RGB image to a CHW float tensor in [0, 1]."""
    array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).contiguous()


def normalize(tensor: torch.Tensor, mean=IMAGENET_MEAN, std=IMAGENET_STD) -> torch.Tensor:
    mean_t = torch.tensor(mean).view(-1, 1, 1)
    std_t = torch.tensor(std).view(-1, 1, 1)
    return (tensor - mean_t) / std_t


def resize(image: Image.Image, size: int) -> Image.Image:
    return image.resize((size, size), Image.BILINEAR)


def random_horizontal_flip(image: Image.Image, p: float = 0.5) -> Image.Image:
    if random.random() < p:
        return image.transpose(Image.FLIP_LEFT_RIGHT)
    return image


def random_crop_resize(image: Image.Image, size: int, scale: tuple[float, float] = (0.8, 1.0)) -> Image.Image:
    width, height = image.size
    area = width * height
    for _ in range(10):
        target_area = random.uniform(*scale) * area
        aspect = random.uniform(0.9, 1.1)
        w = int(round((target_area * aspect) ** 0.5))
        h = int(round((target_area / aspect) ** 0.5))
        if w <= width and h <= height:
            x = random.randint(0, width - w)
            y = random.randint(0, height - h)
            return image.crop((x, y, x + w, y + h)).resize((size, size), Image.BILINEAR)
    return resize(image, size)


def color_jitter(image: Image.Image, brightness: float = 0.2, contrast: float = 0.2, saturation: float = 0.2) -> Image.Image:
    if brightness:
        image = ImageEnhance.Brightness(image).enhance(1.0 + random.uniform(-brightness, brightness))
    if contrast:
        image = ImageEnhance.Contrast(image).enhance(1.0 + random.uniform(-contrast, contrast))
    if saturation:
        image = ImageEnhance.Color(image).enhance(1.0 + random.uniform(-saturation, saturation))
    return image


def random_gaussian_blur(image: Image.Image, p: float = 0.2, radius_range: tuple[float, float] = (0.1, 1.5)) -> Image.Image:
    if random.random() < p:
        return image.filter(ImageFilter.GaussianBlur(radius=random.uniform(*radius_range)))
    return image


class EvalTransform:
    """Deterministic resize + normalize, used for validation/test/inference."""

    def __init__(self, image_size: int = 128) -> None:
        self.image_size = image_size

    def __call__(self, image: Image.Image) -> torch.Tensor:
        image = resize(image, self.image_size)
        tensor = to_tensor(image)
        return normalize(tensor)


class TrainTransform:
    """Moderate augmentation pipeline used for supervised multi-task training."""

    def __init__(self, image_size: int = 128) -> None:
        self.image_size = image_size

    def __call__(self, image: Image.Image) -> torch.Tensor:
        image = random_crop_resize(image, self.image_size)
        image = random_horizontal_flip(image)
        image = color_jitter(image)
        tensor = to_tensor(image)
        return normalize(tensor)


class SimCLRTransform:
    """Strong augmentation pipeline for SimCLR-style self-supervised pretraining.

    Produces two independently augmented views of the same image.
    """

    def __init__(self, image_size: int = 128) -> None:
        self.image_size = image_size

    def _view(self, image: Image.Image) -> torch.Tensor:
        image = random_crop_resize(image, self.image_size, scale=(0.5, 1.0))
        image = random_horizontal_flip(image)
        image = color_jitter(image, brightness=0.4, contrast=0.4, saturation=0.4)
        image = random_gaussian_blur(image, p=0.5)
        tensor = to_tensor(image)
        return normalize(tensor)

    def __call__(self, image: Image.Image) -> tuple[torch.Tensor, torch.Tensor]:
        return self._view(image), self._view(image)
