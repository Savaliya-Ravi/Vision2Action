"""Build a detection backend from configuration.

Keeps backend choice in one place so ``main`` and the evaluator stay agnostic.
The oracle needs the live ``model`` and ``data`` because it reads ground truth.
"""

from __future__ import annotations

from typing import Optional

import mujoco

from vision2action.config import DetectorConfig
from vision2action.perception.detector import ObjectDetector


def build_detector(
    cfg: DetectorConfig,
    *,
    model: Optional[mujoco.MjModel] = None,
    data: Optional[mujoco.MjData] = None,
    width: int = 320,
    height: int = 240,
    fovy_deg: float = 60.0,
    cam_local_offset: tuple[float, float, float] = (0.08, 0.0, 0.0),
) -> ObjectDetector:
    backend = cfg.backend.lower()

    if backend == "oracle":
        from vision2action.perception.oracle_detector import OracleDetector

        if model is None or data is None:
            raise ValueError("oracle backend requires model and data")
        return OracleDetector(
            model,
            data,
            width=width,
            height=height,
            fovy_deg=fovy_deg,
            cam_local_offset=cam_local_offset,
            known_classes=None,
            min_bbox_area_px=cfg.min_bbox_area_px,
        )

    raise ValueError(f"Unknown detector backend: {cfg.backend!r} (use 'oracle')")
