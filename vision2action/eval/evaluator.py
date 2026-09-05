"""Headless evaluation of the perception -> localization -> navigation pipeline.

Each episode runs the *identical* :class:`~vision2action.navigation.mission.Mission`
FSM the interactive demo uses, so what is measured is what ships.  Ground truth
is read **only after** the episode, to score localization error and to confirm
the true final distance -- it never influences perception or navigation.

Metrics per episode (spec item 9):

* ``detected``            - was the target ever selected from detections?
* ``confidence``          - confidence of the selected detection (last valid)
* ``localization_error_m``- planar error of the perceived position vs ground truth
* ``nav_success``         - reached and stopped within tolerance of the object
* ``final_distance_m``    - true robot->object distance at the end (ground truth)
* ``steps`` / ``wall_time_s`` - time-to-reach, deterministic and wall-clock

The default backend is the COCO-limited oracle, so the suite is reproducible
offline.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

import mujoco
import numpy as np

from vision2action.config import Config
from vision2action.control.controller import (
    apply_robot_state,
    initialize_robot_state,
    robot_to_target_distance,
)
from vision2action.env.objects import ground_truth_xy, resolve_object
from vision2action.env.randomization import randomize_scene_objects
from vision2action.eval.scenarios import Scenario, default_scenarios
from vision2action.navigation.mission import Mission, Phase
from vision2action.perception.factory import build_detector
from vision2action.perception.localization import localization_error
from vision2action.targets import parse_target_instruction

_SCENE = Path(__file__).resolve().parent.parent / "env" / "scene.xml"


@dataclass
class EpisodeResult:
    instruction: str
    seed: int
    target: str  # resolved ground-truth object name (eval bookkeeping)
    detected: bool
    confidence: float
    localization_error_m: float
    nav_success: bool
    final_distance_m: float
    steps: int
    wall_time_s: float
    phase: str
    status: str

    def as_row(self) -> dict:
        d = asdict(self)
        return d


@dataclass
class Evaluator:
    config: Config = field(default_factory=Config)
    backend: str = "oracle"
    max_steps: int = 6000
    success_margin_m: float = 0.25  # allowed slack above stop distance
    save_overlays_to: Optional[Path] = None

    def __post_init__(self) -> None:
        self.model = mujoco.MjModel.from_xml_path(str(_SCENE))
        self.data = mujoco.MjData(self.model)
        mujoco.mj_forward(self.model, self.data)
        self.config.detector.backend = self.backend
        cam = self.config.camera
        self.detector = build_detector(
            self.config.detector,
            model=self.model,
            data=self.data,
            width=cam.width,
            height=cam.height,
            fovy_deg=cam.fovy_deg,
            cam_local_offset=cam.local_offset,
        )
        self.mission = Mission(model=self.model, data=self.data, detector=self.detector, config=self.config)

    def close(self) -> None:
        self.mission.close()

    # ------------------------------------------------------------------ episode
    def run_episode(self, instruction: str, seed: int) -> EpisodeResult:
        query = parse_target_instruction(instruction)
        randomize_scene_objects(self.model, self.data, seed=seed)
        robot_state = initialize_robot_state(self.model, self.data)
        robot_state.x, robot_state.y, robot_state.yaw = 0.0, 0.0, 0.0
        apply_robot_state(self.model, self.data, robot_state)

        if query is None:
            return EpisodeResult(instruction, seed, "?", False, 0.0, float("nan"),
                                 False, float("nan"), 0, 0.0, "unparsed", "unparsed")

        self.mission.start(query)
        detected = False
        confidence = 0.0
        est_xy: Optional[np.ndarray] = None
        last = None
        steps = 0
        t0 = time.perf_counter()
        while self.mission.active and steps < self.max_steps:
            last = self.mission.step(robot_state)
            steps += 1
            if last.selection is not None and last.selection.found:
                detected = True
                confidence = float(last.selection.detection.confidence)
            if last.target_xy is not None:
                est_xy = last.target_xy
        wall = time.perf_counter() - t0

        if self.save_overlays_to is not None and last is not None:
            self._save_overlay(instruction, seed, last)

        # --- ground-truth scoring (eval only) --------------------------------
        gt_obj = resolve_object(query.category, query.color)
        true_d = robot_to_target_distance(self.model, self.data, gt_obj.body_name)
        gt_xy = ground_truth_xy(self.model, self.data, gt_obj.name)
        loc_err = localization_error(est_xy, gt_xy) if est_xy is not None else float("nan")
        # The mission stops from the visible surface. Ground truth is measured
        # to the body centre, so include the object's physical footprint.
        stop = (
            self.config.nav.stop_distance_m
            + self.success_margin_m
            + gt_obj.footprint_radius
        )
        nav_success = self.mission.phase == Phase.DONE and np.isfinite(true_d) and true_d <= stop
        status = last.selection.status.value if (last and last.selection) else "no_step"

        return EpisodeResult(
            instruction=instruction,
            seed=seed,
            target=gt_obj.name,
            detected=detected,
            confidence=round(confidence, 4),
            localization_error_m=round(loc_err, 4) if np.isfinite(loc_err) else float("nan"),
            nav_success=bool(nav_success),
            final_distance_m=round(float(true_d), 4) if np.isfinite(true_d) else float("nan"),
            steps=steps,
            wall_time_s=round(wall, 3),
            phase=self.mission.phase.value,
            status=status,
        )

    def run_suite(self, scenarios: Optional[list[Scenario]] = None) -> list[EpisodeResult]:
        scenarios = scenarios if scenarios is not None else default_scenarios()
        return [self.run_episode(s.instruction, s.seed) for s in scenarios]

    def _save_overlay(self, instruction: str, seed: int, result) -> None:
        try:
            import cv2

            from vision2action.perception.visualization import draw_overlay
        except ImportError:
            return
        frame = draw_overlay(
            result.rgb, result.detections, result.selection,
            distance_m=result.distance_m, world_xy=result.target_xy, phase=result.phase.value,
        )
        self.save_overlays_to.mkdir(parents=True, exist_ok=True)
        slug = instruction.lower().replace(" ", "_")
        cv2.imwrite(str(self.save_overlays_to / f"{slug}_seed{seed}.png"), frame[..., ::-1])


# ------------------------------------------------------------------- aggregate
def summarize(results: list[EpisodeResult]) -> dict:
    """Aggregate metrics over a suite."""
    n = len(results)
    if n == 0:
        return {"episodes": 0}
    det = [r for r in results if r.detected]
    nav_ok = [r for r in results if r.nav_success]
    loc_errs = [r.localization_error_m for r in results if np.isfinite(r.localization_error_m)]
    confs = [r.confidence for r in det]
    reached_steps = [r.steps for r in nav_ok]
    return {
        "episodes": n,
        "detection_rate": round(len(det) / n, 3),
        "navigation_success_rate": round(len(nav_ok) / n, 3),
        "mean_confidence": round(float(np.mean(confs)), 3) if confs else 0.0,
        "mean_localization_error_m": round(float(np.mean(loc_errs)), 4) if loc_errs else float("nan"),
        "median_localization_error_m": round(float(np.median(loc_errs)), 4) if loc_errs else float("nan"),
        "mean_final_distance_m": round(
            float(np.mean([r.final_distance_m for r in results if np.isfinite(r.final_distance_m)])), 3
        ),
        "mean_steps_to_reach": round(float(np.mean(reached_steps)), 1) if reached_steps else float("nan"),
    }
