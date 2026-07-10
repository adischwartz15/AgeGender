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

Detection tries a small, fixed sequence of cascades/parameters from
strictest to most lenient (see ``_DETECTION_ATTEMPTS``) before giving up,
since any single Haar cascade + parameter set misses real faces often
enough in practice (watermarks/overlays, scale, lighting) that a single
strict pass is not a good user experience. This only ever *increases*
recall (finds faces a stricter pass would miss); it never invents a face
where a human would clearly see none, since even the most lenient
attempt still requires the classifier's own boosted-feature cascade to
fire on real structure.
"""

from __future__ import annotations

import logging

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

# (cascade filename, scaleFactor, minNeighbors, minSize) -- strictest first.
# alt2 is commonly reported as more accurate than the classic "default"
# cascade; the final pass loosens minNeighbors/minSize/scaleFactor to
# catch smaller or less textbook-frontal faces before giving up.
_DETECTION_ATTEMPTS = (
    ("haarcascade_frontalface_default.xml", 1.1, 5, (40, 40)),
    ("haarcascade_frontalface_alt2.xml", 1.1, 5, (40, 40)),
    ("haarcascade_frontalface_alt2.xml", 1.05, 3, (30, 30)),
)

_cascade_cache: dict[str, cv2.CascadeClassifier] = {}


def _get_cascade(filename: str) -> cv2.CascadeClassifier:
    if filename not in _cascade_cache:
        cascade_path = cv2.data.haarcascades + filename
        classifier = cv2.CascadeClassifier(cascade_path)
        if classifier.empty():
            raise RuntimeError(
                f"Failed to load Haar cascade from '{cascade_path}'. This ships with "
                "opencv-python-headless<5.0 (see requirements.txt); a newer opencv-python "
                "build may not bundle it."
            )
        _cascade_cache[filename] = classifier
    return _cascade_cache[filename]


def _has_eyes(face_region: np.ndarray) -> bool:
    """Return True if at least one eye is detected inside a candidate face region.

    The frontal-face cascade alone false-positives on non-human textures
    that happen to have face-like structure (most commonly animal faces --
    e.g. a dog or cat portrait). Requiring a corroborating eye detection
    inside the candidate box is the standard classical mitigation: eyes
    are a much more specific pattern that these false positives generally
    lack, so this rejects most of them without needing a neural detector.
    """
    if face_region.size == 0:
        return False
    eye_cascade = _get_cascade("haarcascade_eye.xml")
    min_size = max(10, int(min(face_region.shape[:2]) * 0.12))
    eyes = eye_cascade.detectMultiScale(
        face_region, scaleFactor=1.1, minNeighbors=5, minSize=(min_size, min_size)
    )
    return len(eyes) > 0


def detect_largest_face(image: Image.Image) -> tuple[int, int, int, int] | None:
    """Return ``(x, y, w, h)`` of the largest detected face, or None if none found.

    Tries each entry in ``_DETECTION_ATTEMPTS`` in order. A candidate box
    must also pass ``_has_eyes`` before being accepted; if it doesn't,
    detection falls through to the next (more lenient) attempt rather
    than accepting the false positive.
    """
    gray = np.asarray(image.convert("L"))
    gray = cv2.equalizeHist(gray)  # improves detection robustness under uneven lighting

    for filename, scale_factor, min_neighbors, min_size in _DETECTION_ATTEMPTS:
        cascade = _get_cascade(filename)
        faces = cascade.detectMultiScale(
            gray, scaleFactor=scale_factor, minNeighbors=min_neighbors, minSize=min_size
        )
        if len(faces) > 0:
            x, y, w, h = max(faces, key=lambda box: box[2] * box[3])
            if _has_eyes(gray[y : y + h, x : x + w]):
                return int(x), int(y), int(w), int(h)
    return None


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
