"""Localize a detection in the world: bounding box + depth -> (x, y).

This wraps the *already-validated* back-projection pipeline
(:mod:`vision2action.perception.depth_utils`) and only adds robustness: instead
of trusting a single centroid pixel's depth, it takes the median depth over a
small window inside the detected box, which is far less sensitive to edge/noise
pixels.  The camera->world transform itself is unchanged.

Ground truth is never used here; the estimate comes purely from the rendered
depth buffer and the robot's own pose.  Callers compare against ground truth
only afterwards, via :func:`localization_error`, for the evaluation metric.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from vision2action.config import CameraConfig
from vision2action.perception.depth_utils import backproject_pixel_to_camera, camera_xyz_to_world
from vision2action.perception.detector import Detection


@dataclass(frozen=True)
class Localization:
    ok: bool
    world_xy: Optional[np.ndarray] = None  # estimated planar position (m)
    depth_m: Optional[float] = None  # sampled optical-axis depth (m)
    pixel: Optional[tuple[int, int]] = None  # pixel the estimate was taken at


def _sample_depth(
    depth: np.ndarray,
    bbox: tuple[int, int, int, int],
    znear: float,
    zfar: float,
    window_frac: float = 0.5,
) -> Optional[float]:
    """Median valid depth over the central ``window_frac`` of ``bbox``."""
    x1, y1, x2, y2 = bbox
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    hw = max(1, int((x2 - x1) * window_frac / 2))
    hh = max(1, int((y2 - y1) * window_frac / 2))
    h, w = depth.shape[:2]
    wx1, wx2 = max(0, cx - hw), min(w, cx + hw + 1)
    wy1, wy2 = max(0, cy - hh), min(h, cy + hh + 1)
    patch = depth[wy1:wy2, wx1:wx2].reshape(-1)

    valid = patch[np.isfinite(patch) & (patch > znear) & (patch < zfar)]
    if valid.size == 0:
        return None
    return float(np.median(valid))


def localize(
    detection: Detection,
    depth: np.ndarray,
    robot_state,
    camera: CameraConfig,
    *,
    znear: float = 0.05,
    zfar: float = 25.0,
) -> Localization:
    """Estimate a detection's world (x, y) from the depth buffer."""
    if depth.ndim != 2 or depth.shape != (camera.height, camera.width):
        return Localization(ok=False)

    depth_m = _sample_depth(depth, detection.bbox_xyxy, znear, zfar)
    if depth_m is None:
        return Localization(ok=False)

    cx, cy = detection.centroid
    if not (0 <= cx < camera.width and 0 <= cy < camera.height):
        return Localization(ok=False)

    cam_xyz = backproject_pixel_to_camera(cx, cy, depth_m, camera.fovy_deg, camera.width, camera.height)
    world_xy = camera_xyz_to_world(cam_xyz, robot_state, camera.local_offset)
    if world_xy.shape != (2,) or not np.all(np.isfinite(world_xy)):
        return Localization(ok=False)
    return Localization(ok=True, world_xy=world_xy, depth_m=depth_m, pixel=(cx, cy))


def localization_error(estimate_xy: np.ndarray, ground_truth_xy: tuple[float, float]) -> float:
    """Planar distance (m) between an estimate and ground truth. Eval-only."""
    est = np.asarray(estimate_xy, dtype=float)
    gt = np.asarray(ground_truth_xy, dtype=float)
    return float(np.linalg.norm(est - gt))
