"""Perception-driven navigation mission: the search -> center -> forward FSM.

This is a faithful extraction of the finite-state machine that used to live
inside ``main.run_demo``, with two deliberate changes:

* the navigation target comes from **perception** (detection -> selection ->
  depth localization), never from ground truth;
* every threshold is sourced from :class:`~vision2action.config.NavConfig` so it
  is configurable (stop distance, search/centering steps, tolerances, ...).

The control primitives (:func:`navigation_step`, :func:`rotate_in_place`,
:func:`move_forward`, :func:`apply_robot_state`) are reused unchanged, so the
working navigation behaviour is preserved.  Keeping the FSM UI-agnostic lets the
interactive viewer and the headless evaluator run the *identical* logic.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional

import mujoco
import numpy as np

from vision2action.config import Config
from vision2action.control.controller import (
    RobotState,
    apply_robot_state,
    distance_to_world_target,
    move_forward,
    rotate_in_place,
)
from vision2action.navigation.navigation import NavParams, navigation_step
from vision2action.perception.detector import Detection, ObjectDetector
from vision2action.perception.localization import Localization, localize
from vision2action.perception.selection import Selection, SelectStatus, select_target
from vision2action.targets import TargetQuery


class Phase(str, Enum):
    IDLE = "idle"
    SEARCH = "search"
    FORWARD = "forward"
    DONE = "done"
    FAILED = "failed"


@dataclass
class StepResult:
    phase: Phase
    rgb: np.ndarray
    detections: List[Detection]
    selection: Optional[Selection]
    localization: Optional[Localization]
    distance_m: Optional[float]
    message: str = ""

    @property
    def target_xy(self) -> Optional[np.ndarray]:
        return self.localization.world_xy if (self.localization and self.localization.ok) else None


def _render_depth(renderer: mujoco.Renderer) -> np.ndarray:
    """Render a depth frame, tolerating both MuJoCo depth-render APIs."""
    try:
        return renderer.render(depth=True)
    except TypeError:
        renderer.enable_depth_rendering()
        depth = renderer.render()
        renderer.disable_depth_rendering()
        return depth


@dataclass
class Mission:
    model: mujoco.MjModel
    data: mujoco.MjData
    detector: ObjectDetector
    config: Config = field(default_factory=Config)

    def __post_init__(self) -> None:
        cam = self.config.camera
        self._rgb_renderer = mujoco.Renderer(self.model, height=cam.height, width=cam.width)
        self._depth_renderer = mujoco.Renderer(self.model, height=cam.height, width=cam.width)
        # Hide the robot's own body (geom group 2) from its camera: a camera does
        # not see its own mount, and the coloured chassis would corrupt detection.
        self._cam_option = mujoco.MjvOption()
        self._cam_option.geomgroup[2] = 0
        self._cam_option.geomgroup[3] = 0
        self._nav_params = NavParams(
            search_yaw_step=self.config.nav.search_yaw_step,
            center_tolerance_px=self.config.nav.center_tolerance_px,
            centering_yaw_step=self.config.nav.centering_yaw_step,
        )
        self.reset()

    # ------------------------------------------------------------------ setup
    def reset(self) -> None:
        self.query: Optional[TargetQuery] = None
        self.phase = Phase.IDLE
        self._step = 0
        self._tick = 0
        self._forward_step_limit = 0
        self._best_distance = float("inf")
        self._last_selection: Optional[Selection] = None
        self._search_direction = 1.0

    def start(self, query: TargetQuery) -> None:
        self.reset()
        self.query = query
        self.phase = Phase.SEARCH

    def close(self) -> None:
        self._rgb_renderer.close()
        self._depth_renderer.close()

    @property
    def active(self) -> bool:
        return self.phase in (Phase.SEARCH, Phase.FORWARD)

    # ------------------------------------------------------------- perception
    def _perceive(self, robot_state: RobotState):
        cam = self.config.camera
        self._rgb_renderer.update_scene(self.data, camera=cam.name, scene_option=self._cam_option)
        rgb = self._rgb_renderer.render()
        self._depth_renderer.update_scene(self.data, camera=cam.name, scene_option=self._cam_option)
        depth = _render_depth(self._depth_renderer)

        # Oracle backends need the authoritative pose (duck-typed, eval only).
        setter = getattr(self.detector, "set_robot_state", None)
        if callable(setter):
            setter(robot_state)

        detections = self.detector.detect(rgb)
        use_color_fallback = self.detector.name != "OracleDetector(omniscient)"
        selection = select_target(
            detections,
            self.query,
            rgb,
            allow_color_fallback=use_color_fallback,
        )
        loc: Optional[Localization] = None
        if selection.found:
            loc = localize(selection.detection, depth, robot_state, cam)
        return rgb, detections, selection, loc

    # -------------------------------------------------------------- FSM step
    def step(self, robot_state: RobotState) -> StepResult:
        if self.query is None or not self.active:
            return StepResult(self.phase, self._render_rgb_only(), [], None, None, None, "inactive")

        rgb, detections, selection, loc = self._perceive(robot_state)
        self._last_selection = selection
        valid = selection.found and loc is not None and loc.ok
        centroid = selection.detection.centroid if selection.found else None
        distance_m = (
            distance_to_world_target(robot_state, loc.world_xy) if valid else None
        )

        if self.phase == Phase.SEARCH:
            msg = self._search(robot_state, valid, centroid, loc, distance_m)
        else:
            msg = self._forward(robot_state, valid, centroid, loc, distance_m)

        self._tick += 1
        return StepResult(self.phase, rgb, detections, selection, loc, distance_m, msg)

    def _render_rgb_only(self) -> np.ndarray:
        self._rgb_renderer.update_scene(self.data, camera=self.config.camera.name, scene_option=self._cam_option)
        return self._rgb_renderer.render()

    # --------------------------------------------------------- search phase
    def _search(self, state, valid, centroid, loc, distance_m) -> str:
        nav = self.config.nav
        cam = self.config.camera

        if valid:
            error = centroid[0] - cam.width // 2
            if abs(error) > nav.center_tolerance_px:
                self._search_direction = 1.0 if error < 0 else -1.0
            centered = navigation_step(state, centroid, cam.width, self._nav_params)
        else:
            rotate_in_place(
                state,
                yaw_step=self._search_direction * nav.search_yaw_step,
            )
            centered = False
        apply_robot_state(self.model, self.data, state)

        if centered and valid and distance_m is not None and np.isfinite(distance_m):
            stop = nav.stop_distance_m + nav.stop_epsilon_m
            if distance_m <= stop:
                self.phase = Phase.DONE
                return f"already within stop distance ({distance_m:.2f} m)"
            remaining = max(distance_m - stop, 0.0)
            est_steps = int(np.ceil(remaining / max(nav.forward_step_m, 1e-9)))
            self._forward_step_limit = int(est_steps * nav.forward_step_safety_factor) + nav.forward_step_margin
            self._best_distance = distance_m
            self._step = 0
            self.phase = Phase.FORWARD
            return f"centered; advancing (dist={distance_m:.2f} m, limit={self._forward_step_limit})"

        self._step += 1
        if self._step > nav.max_search_steps:
            self.phase = Phase.FAILED
            return "search timeout"
        return "searching" if not valid else "centering"

    # -------------------------------------------------------- forward phase
    def _forward(self, state, valid, centroid, loc, distance_m) -> str:
        nav = self.config.nav
        cam = self.config.camera

        if not valid or distance_m is None or not np.isfinite(distance_m):
            self.phase = Phase.SEARCH
            self._step = 0
            return "target temporarily lost; searching again"

        # Re-center: coarse turn if far off, fine proportional nudge if close.
        pixel_error = centroid[0] - cam.width // 2
        if abs(pixel_error) > nav.forward_center_tolerance_px:
            if abs(pixel_error) > nav.center_tolerance_px:
                navigation_step(state, centroid, cam.width, self._nav_params)
            else:
                normalized = pixel_error / float(max(cam.width // 2, 1))
                yaw_delta = -float(np.clip(normalized, -1.0, 1.0)) * nav.centering_yaw_step * 1.5
                rotate_in_place(state, yaw_step=yaw_delta)
            apply_robot_state(self.model, self.data, state)

        stop = nav.stop_distance_m + nav.stop_epsilon_m
        if distance_m <= stop:
            self.phase = Phase.DONE
            return f"reached target (dist={distance_m:.2f} m)"

        # Guard: perceived target must be in front before moving.
        target_vec = loc.world_xy - np.array([state.x, state.y], dtype=float)
        forward = np.array([math.cos(state.yaw), math.sin(state.yaw)], dtype=float)
        if float(np.dot(target_vec, forward)) <= 0.0:
            self.phase = Phase.FAILED
            return "target not in front; aborting"

        step_m = min(nav.forward_step_m, max(distance_m - stop, 0.0))
        move_forward(state, forward_step=step_m)
        apply_robot_state(self.model, self.data, state)
        new_distance = distance_to_world_target(state, loc.world_xy)

        if np.isfinite(new_distance) and new_distance <= stop:
            self.phase = Phase.DONE
            return f"reached target (dist={new_distance:.2f} m)"
        if new_distance > self._best_distance + nav.drift_realign_threshold_m:
            self.phase = Phase.SEARCH
            self._step = 0
            return f"drift ({new_distance:.2f} > best {self._best_distance:.2f}); re-aligning"
        self._step += 1
        if self._step > self._forward_step_limit:
            self.phase = Phase.FAILED
            return "forward timeout"
        self._best_distance = min(self._best_distance, new_distance)
        return f"advancing (dist={new_distance:.2f} m)"
