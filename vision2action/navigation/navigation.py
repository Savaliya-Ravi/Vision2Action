from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

from vision2action.control.controller import RobotState, rotate_in_place


@dataclass
class NavParams:
    search_yaw_step: float = 0.001
    center_tolerance_px: int = 24
    centering_yaw_step: float = 0.001


def navigation_step(
    state: RobotState,
    detection: Optional[Tuple[int, int]],
    image_width: int,
    params: NavParams,
) -> bool:
    """Rotate to search or center the target; return True when centered."""
    if detection is None:
        rotate_in_place(state, yaw_step=params.search_yaw_step)
        return False

    cx, _ = detection
    center_x = image_width // 2
    error_px = cx - center_x

    if abs(error_px) <= params.center_tolerance_px:
        return True

    turn_step = params.centering_yaw_step if error_px < 0 else -params.centering_yaw_step
    rotate_in_place(state, yaw_step=turn_step)
    return False
