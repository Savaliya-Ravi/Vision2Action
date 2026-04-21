from __future__ import annotations

from typing import Optional, Tuple

import numpy as np


class Detector:
    def detect(self, image: np.ndarray, target: str):
        raise NotImplementedError


def _color_mask(rgb_image: np.ndarray, target: str) -> np.ndarray:
    """Return boolean mask for target object color."""
    r = rgb_image[:, :, 0]
    g = rgb_image[:, :, 1]
    b = rgb_image[:, :, 2]

    target = target.lower().strip()
    total = r.astype(float) + g.astype(float) + b.astype(float) + 1.0
    r_ratio = r.astype(float) / total
    g_ratio = g.astype(float) / total
    b_ratio = b.astype(float) / total

    if target == "blue":
        return (b > 70) & (b_ratio > 0.42)
    if target == "red":
        return (r > 70) & (r_ratio > 0.42)
    if target == "green":
        return (g > 70) & (g_ratio > 0.42)
    if target == "brown":
        return (r > 80) & (g > 35) & (b < 90) & (r > g) & (g > b)

    raise ValueError(f"Unsupported target: {target}")


def detect_target_centroid(rgb_image: np.ndarray, target: str) -> Tuple[np.ndarray, Optional[Tuple[int, int]]]:
    """Detect target from RGB image and return (mask_uint8, centroid)."""
    mask_bool = _color_mask(rgb_image, target)
    ys, xs = np.where(mask_bool)

    mask_uint8 = mask_bool.astype(np.uint8) * 255
    if xs.size == 0:
        return mask_uint8, None

    cx = int(np.mean(xs))
    cy = int(np.mean(ys))
    return mask_uint8, (cx, cy)


class ColorDetector(Detector):
    """Current deterministic baseline detector implementation."""

    def detect(self, image: np.ndarray, target: str):
        return detect_target_centroid(image, target)
