"""Classical-CV colour estimation for attribute handling ("red can").

Two jobs, both deliberately simple and dependency-free (numpy only):

* :func:`estimate_dominant_color` - given an image crop (a detected bounding
  box), name its dominant colour.  Used to disambiguate same-class detections,
  e.g. pick the *red* can among several cans, or the *blue* bottle.
* :func:`find_color_blob` - scan a whole image for the largest region of a
  colour and return its box. Used as a fallback for target categories that a
  detector cannot label, such as a can.

Supported colour words are intentionally few but robust: red, green, blue,
yellow, white.  Everything works on ratios/relative brightness so it is
insensitive to the overall lighting level.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

SUPPORTED_COLORS: tuple[str, ...] = ("red", "green", "blue", "yellow", "white")


def _color_masks(rgb: np.ndarray) -> dict[str, np.ndarray]:
    """Boolean per-colour masks for an RGB uint8 image/crop."""
    r = rgb[..., 0].astype(np.float32)
    g = rgb[..., 1].astype(np.float32)
    b = rgb[..., 2].astype(np.float32)
    mx = np.maximum(np.maximum(r, g), b)
    mn = np.minimum(np.minimum(r, g), b)
    bright = mx > 60.0  # ignore near-black pixels (shadows / background)

    return {
        # ``b >= 0.7 * g`` rejects orange (green >> blue, e.g. a mustard-bottle
        # cap) while keeping true red, where green and blue are both low and
        # roughly balanced (red can b/g~0.99, red mug b/g~1.11, orange cap ~0.66).
        "red": bright & (r > 1.35 * g) & (r > 1.35 * b) & (r > 80) & (b >= 0.7 * g),
        "green": bright & (g > 1.25 * r) & (g > 1.15 * b) & (g > 70),
        "blue": bright & (b > 1.2 * r) & (b > 1.05 * g) & (b > 75),
        "yellow": bright & (r > 100) & (g > 90) & (b < 0.75 * np.minimum(r, g)),
        "white": (mn > 145) & ((mx - mn) < 42),
    }


def color_fractions(crop_rgb: np.ndarray) -> dict[str, float]:
    """Fraction of pixels in ``crop_rgb`` matching each supported colour."""
    crop = np.asarray(crop_rgb)
    if crop.ndim != 3 or crop.size == 0:
        return {c: 0.0 for c in SUPPORTED_COLORS}
    n = float(crop.shape[0] * crop.shape[1])
    masks = _color_masks(crop)
    return {c: float(masks[c].sum()) / n for c in SUPPORTED_COLORS}


def estimate_dominant_color(crop_rgb: np.ndarray, min_fraction: float = 0.12) -> Optional[str]:
    """Return the dominant supported colour of a crop, or ``None`` if unclear."""
    fr = color_fractions(crop_rgb)
    best = max(fr, key=fr.get)
    return best if fr[best] >= min_fraction else None


def matches_color(crop_rgb: np.ndarray, color: str, min_fraction: float = 0.10) -> bool:
    """True if ``color`` is present and the strongest colour in the crop."""
    color = color.lower().strip()
    if color not in SUPPORTED_COLORS:
        return False
    fr = color_fractions(crop_rgb)
    return fr[color] >= min_fraction and color == max(fr, key=fr.get)


def find_color_blob(
    image_rgb: np.ndarray,
    color: str,
    min_area_px: int = 15,
    exclude_boxes: Optional[list] = None,
) -> Optional[tuple[int, int, int, int]]:
    """Largest connected region of ``color``; returns ``(x1, y1, x2, y2)``.

    Uses connected-component labelling (OpenCV) on the colour mask and returns
    the bounding box of the single largest component, so scattered specular
    pixels can't inflate the box.  This is the classical-CV fallback for
    categories a pretrained detector cannot name.

    ``exclude_boxes`` zeroes the mask inside the given ``(x1, y1, x2, y2)``
    regions (slightly padded).  Selection passes the boxes of *known-class*
    detections here, so a colour-fallback target ("red can") is chosen as the
    red region that is **not** already explained by a recognised object -- this
    is what stops it from locking onto a red-capped bottle or a red mug.
    """
    color = color.lower().strip()
    if color not in SUPPORTED_COLORS:
        return None
    img = np.asarray(image_rgb)
    if img.ndim != 3:
        return None
    mask = _color_masks(img)[color].astype(np.uint8)

    if exclude_boxes:
        h, w = mask.shape[:2]
        pad = 3  # cover a little bleed past the reported box edges
        for box in exclude_boxes:
            x1, y1, x2, y2 = (int(v) for v in box)
            xa, ya = max(0, x1 - pad), max(0, y1 - pad)
            xb, yb = min(w, x2 + pad), min(h, y2 + pad)
            if xb > xa and yb > ya:
                mask[ya:yb, xa:xb] = 0

    if int(mask.sum()) < min_area_px:
        return None

    try:
        import cv2

        kernel = np.ones((3, 3), np.uint8)
        merged = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
        merged = cv2.dilate(merged, kernel, iterations=1)
        count, _labels, stats, _cent = cv2.connectedComponentsWithStats(merged, connectivity=8)
        if count <= 1:
            return None
        areas = stats[1:, cv2.CC_STAT_AREA]  # skip background (label 0)
        idx = 1 + int(np.argmax(areas))
        if int(stats[idx, cv2.CC_STAT_AREA]) < min_area_px:
            return None
        x = int(stats[idx, cv2.CC_STAT_LEFT])
        y = int(stats[idx, cv2.CC_STAT_TOP])
        w = int(stats[idx, cv2.CC_STAT_WIDTH])
        h = int(stats[idx, cv2.CC_STAT_HEIGHT])
        return x, y, x + w, y + h
    except ImportError:
        # numpy-only fallback: dominant-cluster box via robust percentiles.
        ys, xs = np.where(mask.astype(bool))
        x_lo, x_hi = np.percentile(xs, [8, 92])
        y_lo, y_hi = np.percentile(ys, [8, 92])
        core = (xs >= x_lo) & (xs <= x_hi) & (ys >= y_lo) & (ys <= y_hi)
        xs, ys = xs[core], ys[core]
        if xs.size < min_area_px:
            return None
        return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def crop_bbox(image_rgb: np.ndarray, bbox_xyxy: tuple[int, int, int, int]) -> np.ndarray:
    """Safe crop of a bounding box from an image (clamped to bounds)."""
    h, w = image_rgb.shape[:2]
    x1, y1, x2, y2 = bbox_xyxy
    x1 = max(0, min(int(x1), w - 1))
    y1 = max(0, min(int(y1), h - 1))
    x2 = max(x1 + 1, min(int(x2), w))
    y2 = max(y1 + 1, min(int(y2), h))
    return image_rgb[y1:y2, x1:x2]
