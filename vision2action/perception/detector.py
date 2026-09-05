"""The perception contract shared by every detector backend.

The key design decision (per the V2 spec): detection is **open** and not
target-conditioned.  A detector looks at an image and returns *everything* it
recognises; target selection happens downstream. This keeps detector backends
interchangeable behind one interface::

    detector.detect(rgb) -> list[Detection]

Each :class:`Detection` carries a class name, confidence, and pixel box.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import List

import numpy as np


@dataclass(frozen=True)
class Detection:
    """A single detected object in image space.

    Example::

        Detection(class_name="bottle", confidence=0.91, bbox_xyxy=(x1, y1, x2, y2))
    """

    class_name: str
    confidence: float
    bbox_xyxy: tuple[int, int, int, int]

    @property
    def centroid(self) -> tuple[int, int]:
        x1, y1, x2, y2 = self.bbox_xyxy
        return ((x1 + x2) // 2, (y1 + y2) // 2)

    @property
    def area(self) -> int:
        x1, y1, x2, y2 = self.bbox_xyxy
        return max(0, x2 - x1) * max(0, y2 - y1)

    def as_dict(self) -> dict:
        return {
            "class": self.class_name,
            "confidence": round(float(self.confidence), 4),
            "bbox": [int(v) for v in self.bbox_xyxy],
        }


class ObjectDetector(abc.ABC):
    """Interface every detection backend implements."""

    @abc.abstractmethod
    def detect(self, image: np.ndarray) -> List[Detection]:
        """Return all detections in ``image`` (RGB, HxWx3 uint8)."""
        raise NotImplementedError

    @property
    def name(self) -> str:
        return type(self).__name__


def clip_bbox(
    x1: float, y1: float, x2: float, y2: float, width: int, height: int
) -> tuple[int, int, int, int]:
    """Order, round, and clamp a box to image bounds; guarantees x1<x2, y1<y2."""
    xa, xb = sorted((int(round(x1)), int(round(x2))))
    ya, yb = sorted((int(round(y1)), int(round(y2))))
    xa = max(0, min(xa, width - 1))
    ya = max(0, min(ya, height - 1))
    xb = max(xa + 1, min(xb, width))
    yb = max(ya + 1, min(yb, height))
    return xa, ya, xb, yb
