"""Central, typed configuration for the Vision2Action pipeline.

Everything that used to be a magic number scattered through ``main.py`` lives
here so scenarios, tests, and the interactive demo can share one source of
truth.  All values are plain dataclasses; construct :class:`Config` and pass it
around, or use :func:`default_config`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class CameraConfig:
    """Robot camera used for perception (matches ``robot_cam`` in scene.xml)."""

    name: str = "robot_cam"
    width: int = 320
    height: int = 240
    fovy_deg: float = 100.0
    # Camera body offset from the robot origin, robot-local frame (metres).
    local_offset: tuple[float, float, float] = (0.08, 0.0, 0.0)
    # Robot body height (z of the camera body in the world), metres.
    mount_height_m: float = 1.16


@dataclass
class DetectorConfig:
    """Perception backend selection and thresholds.

    ``backend`` picks the implementation of the ``ObjectDetector`` interface:

    * ``"oracle"``    - ground-truth projection; **debug/eval only**.
    """

    backend: str = "oracle"
    # Minimum bounding-box area in pixels to trust a detection.
    min_bbox_area_px: int = 4


@dataclass
class NavConfig:
    """Navigation / control policy parameters (see navigation.py)."""

    # Distance from the object at which the robot stops.  Configurable per your
    # spec (~0.7-1.0 m); this is the estimated robot-centre to object-centre gap.
    stop_distance_m: float = 0.85
    stop_epsilon_m: float = 0.02
    search_yaw_step: float = 0.01
    centering_yaw_step: float = 0.006
    center_tolerance_px: int = 24
    forward_center_tolerance_px: int = 14
    forward_step_m: float = 0.02
    max_search_steps: int = 5000
    forward_step_safety_factor: float = 2.5
    forward_step_margin: int = 200
    drift_realign_threshold_m: float = 0.05


@dataclass
class Config:
    seed: Optional[int] = None
    camera: CameraConfig = field(default_factory=CameraConfig)
    detector: DetectorConfig = field(default_factory=DetectorConfig)
    nav: NavConfig = field(default_factory=NavConfig)


def default_config() -> Config:
    return Config()
