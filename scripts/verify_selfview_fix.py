"""Re-run the Mission episodes that exposed the robot self-view bug.

Before the fix the robot's own coloured front geom was visible to its camera,
so "go to the red can" locked the colour fallback onto the chassis and stopped
at the wrong place.  This script re-runs the four canonical commands headlessly
with the COCO-limited oracle and prints, per episode: final phase, true robot->
object distance (ground truth, eval-only), and localization error.
"""

from __future__ import annotations

import os
import sys

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vision2action.config import Config
from vision2action.control.controller import (
    initialize_robot_state,
    apply_robot_state,
    robot_to_target_distance,
)
from vision2action.env.objects import ground_truth_xy, resolve_object
from vision2action.env.randomization import randomize_scene_objects
from vision2action.navigation.mission import Mission, Phase
from vision2action.perception.localization import localization_error
from vision2action.perception.oracle_detector import OracleDetector, coco_known_classes
from vision2action.targets import parse_target_instruction

SCENE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     "..", "vision2action", "env", "scene.xml")
COMMANDS = ["go to the bottle", "go to the banana", "go to the red can", "go to the mug"]
SEED = 42


def main() -> int:
    model = mujoco.MjModel.from_xml_path(os.path.abspath(SCENE))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    poses = randomize_scene_objects(model, data, seed=SEED)
    robot_state = initialize_robot_state(model, data)
    apply_robot_state(model, data, robot_state)  # pin robot at z=0.6 (no free-fall)

    cfg = Config(seed=SEED)
    detector = OracleDetector(
        model, data,
        width=cfg.camera.width, height=cfg.camera.height,
        fovy_deg=cfg.camera.fovy_deg, cam_local_offset=cfg.camera.local_offset,
        known_classes=coco_known_classes(),
    )
    mission = Mission(model=model, data=data, detector=detector, config=cfg)

    print(f"seed={SEED}  detector={detector.name}  stop={cfg.nav.stop_distance_m} m\n")
    ok = 0
    for cmd in COMMANDS:
        query = parse_target_instruction(cmd)
        # reset the robot to the origin for a fair, independent episode
        robot_state = initialize_robot_state(model, data)
        robot_state.x, robot_state.y, robot_state.yaw = 0.0, 0.0, 0.0
        apply_robot_state(model, data, robot_state)
        mission.start(query)

        last = None
        steps = 0
        while mission.active and steps < 20000:
            last = mission.step(robot_state)
            steps += 1

        gt_obj = resolve_object(query.category, query.color)
        true_d = robot_to_target_distance(model, data, gt_obj.body_name)
        gt_xy = ground_truth_xy(model, data, gt_obj.name)
        est = last.target_xy if last is not None else None
        loc_err = localization_error(est, gt_xy) if est is not None else float("nan")
        rx, ry = robot_state.x, robot_state.y

        success = mission.phase == Phase.DONE and true_d <= cfg.nav.stop_distance_m + 0.25
        ok += int(success)
        flag = "OK " if success else "!! "
        print(f"{flag}{cmd:<22} phase={mission.phase.value:<7} steps={steps:<5} "
              f"true_d={true_d:5.2f}m  locErr={loc_err*100:5.1f}cm  "
              f"robot=({rx:5.2f},{ry:5.2f})  obj={gt_obj.name}")

    print(f"\n{ok}/{len(COMMANDS)} episodes succeeded")
    mission.close()
    return 0 if ok == len(COMMANDS) else 1


if __name__ == "__main__":
    raise SystemExit(main())
