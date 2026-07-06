"""Classical face-region cropping for inference preprocessing.

Uses OpenCV's bundled Haar cascade frontal-face classifier (Viola-Jones:
a classical, non-neural computer-vision technique trained via boosting on
Haar-like features, not a downloaded pretrained neural network) to find
the most prominent face in an uploaded image and crop to it with a
margin before the image is handed to the model.

Why this exists: the model is trained on UTKFace, whose images are
already tight face crops. An arbitrary user photo (with background,
clothing, jewelry, hair styling, etc.) is a very different visual
distribution -- cropping to the detected face at inference time brings
the input closer to what the model actually learned from.

This is a real, functioning face detector, not a placeholder -- but it is
a classical, moderate-accuracy method: it can miss faces at extreme
angles, in poor lighting, or when partially occluded, and picks the
single largest detected face if more than one is present. It performs no
identity verification, liveness check, landmark localization, or any
other biometric function beyond locating a face-shaped region to crop.
"""

from __future__ import annotations

import logging

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

_CASCADE_FILENAME = "haarcascade_frontalface_default.xml"
_face_cascade: "cv2.CascadeClassifier | None" = None


def _get_cascade() -> cv2.CascadeClassifier:
    global _face_cascade
    if _face_cascade is None:
        cascade_path = cv2.data.haarcascades + _CASCADE_FILENAME
        classifier = cv2.CascadeClassifier(cascade_path)
        if classifier.empty():
            raise RuntimeError(
                f"Failed to load Haar cascade from '{cascade_path}'. This ships with "
                "opencv-python-headless<5.0 (see requirements.txt); a newer opencv-python "
                "build may not bundle it."
            )
        _face_cascade = classifier
    return _face_cascade


def detect_largest_face(image: Image.Image) -> tuple[int, int, int, int] | None:
    """Return ``(x, y, w, h)`` of the largest detected face, or None if none found."""
    gray = np.asarray(image.convert("L"))
    gray = cv2.equalizeHist(gray)  # improves detection robustness under uneven lighting
    cascade = _get_cascade()
    faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))
    if len(faces) == 0:
        return None
    x, y, w, h = max(faces, key=lambda box: box[2] * box[3])
    return int(x), int(y), int(w), int(h)


def crop_to_face(image: Image.Image, margin_ratio: float = 0.35) -> tuple[Image.Image, bool]:
    """Crop ``image`` to its largest detected face plus a margin.

    Returns ``(possibly_cropped_image, face_found)``. When no face is
    detected, returns the original (RGB-converted) image unchanged with
    ``face_found=False`` -- callers should surface that as a warning
    rather than silently proceeding as if a face-centered crop was used.
    """
    rgb = image.convert("RGB")
    box = detect_largest_face(rgb)
    if box is None:
        return rgb, False

    x, y, w, h = box
    margin_x, margin_y = int(w * margin_ratio), int(h * margin_ratio)
    width, height = rgb.size
    left = max(0, x - margin_x)
    top = max(0, y - margin_y)
    right = min(width, x + w + margin_x)
    bottom = min(height, y + h + margin_y)
    return rgb.crop((left, top, right, bottom)), True
