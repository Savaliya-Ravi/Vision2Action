"""Interactive V2 demo with oracle-assisted navigation and red-can handling.

The V3 Octo arm trial is a separate entry point in :mod:`vision2action.vla`.
"""

from __future__ import annotations

import argparse
import logging
import os
import re
import sys
import threading
from pathlib import Path
from typing import Optional

import mujoco
import mujoco.viewer
import numpy as np

from vision2action.config import Config
from vision2action.control.controller import apply_robot_state, initialize_robot_state
from vision2action.env.randomization import randomize_scene_objects
from vision2action.manipulation import (
    initialize_gripper,
    pick_up_red_can,
    put_down_red_can,
    stow_right_arm,
    sync_held_red_can,
)
from vision2action.navigation.mission import Mission, Phase, StepResult
from vision2action.perception.factory import build_detector
from vision2action.perception.visualization import draw_overlay
from vision2action.targets import SUPPORTED_CATEGORIES, parse_target_instruction
from vision2action.world.world_model import WorldModel

logger = logging.getLogger(__name__)

_system_fonts = Path("/usr/share/fonts/truetype/dejavu")
if _system_fonts.exists():
    os.environ.setdefault("QT_QPA_FONTDIR", str(_system_fonts))

_OVERLAY_WINDOW = "Vision2Action - robot camera"


def set_user_free_camera(viewer: mujoco.viewer.Handle) -> None:
    """Default interactive camera for mouse exploration."""
    viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    viewer.cam.lookat[:] = np.array([0.0, 0.0, 0.9])
    viewer.cam.distance = 8.5
    viewer.cam.azimuth = 135
    viewer.cam.elevation = -20


def _command_reader(command_state: dict, lock: threading.Lock, running: threading.Event) -> None:
    prompt = "\nCommand (e.g. 'go to the fridge', 'pick up the red can', 'put down the can', 'stop', 'quit'): "
    while running.is_set():
        try:
            text = input(prompt).strip()
        except EOFError:
            break
        if not text:
            continue
        with lock:
            command_state["pending"] = text


def _draw_markers(viewer: mujoco.viewer.Handle, world_model: WorldModel, active_label: Optional[str]) -> None:
    """Show each remembered perceived position as a sphere (active = red)."""
    viewer.user_scn.ngeom = 0
    max_geoms = min(int(viewer.user_scn.maxgeom), len(viewer.user_scn.geoms))
    for label, world_xy in world_model.get_all().items():
        if viewer.user_scn.ngeom >= max_geoms or world_xy is None:
            break
        rgba = np.array([1, 0, 0, 1] if label == active_label else [0.6, 0.6, 0.6, 1], dtype=np.float32)
        g = viewer.user_scn.geoms[viewer.user_scn.ngeom]
        mujoco.mjv_initGeom(
            g, mujoco.mjtGeom.mjGEOM_SPHERE, np.zeros(3), np.zeros(3), np.eye(3).flatten(), rgba
        )
        g.size[:] = [0.08, 0.08, 0.08]
        g.pos[:] = [float(world_xy[0]), float(world_xy[1]), 0.5]
        viewer.user_scn.ngeom += 1


def _has_display() -> bool:
    """Whether a GUI window can really be opened.

    OpenCV/Qt aborts the whole process (uncatchable) when it cannot reach an X
    server, so a mere ``DISPLAY`` env var is not enough -- a stale ``:0`` with no
    server behind it still crashes.  We actually probe the X11 socket and treat
    *any* failure (including sandbox socket restrictions) as "no display".
    """
    if not sys.platform.startswith("linux"):
        return True
    if os.environ.get("WAYLAND_DISPLAY"):
        return True
    disp = os.environ.get("DISPLAY", "")
    if not disp:
        return False
    host, _, num = disp.partition(":")
    if host not in ("", "unix"):
        return True  # remote/TCP display; assume the user knows it works
    import socket

    dnum = num.split(".")[0] or "0"
    path = f"/tmp/.X11-unix/X{dnum}"
    try:
        s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        s.settimeout(0.3)
        s.connect(path)
        s.close()
        return True
    except OSError:
        return False


def _show_overlay(result: StepResult) -> bool:
    """Render the debug overlay in an OpenCV window. Returns False to disable."""
    if not _has_display():
        return False  # headless: cv2.imshow would abort the process via Qt
    try:
        import cv2
    except ImportError:
        return False
    frame = draw_overlay(
        result.rgb,
        result.detections,
        result.selection,
        distance_m=result.distance_m,
        world_xy=result.target_xy,
        phase=result.phase.value,
    )
    try:
        cv2.imshow(_OVERLAY_WINDOW, frame[..., ::-1])  # RGB -> BGR for OpenCV
        cv2.waitKey(1)
        return True
    except cv2.error:
        return False  # no GUI backend; disable quietly


def _handle_command(text: str, mission: Mission, world_model: WorldModel, action_state: dict) -> bool:
    """Apply a typed command. Returns False if the demo should quit."""
    low = text.lower().strip()
    if low in ("quit", "exit"):
        return False
    if low == "stop":
        mission.reset()
        mission.config.nav.stop_distance_m = action_state["normal_stop"]
        action_state["pick_up"] = False
        action_state["put_down"] = None
        action_state["target_xy"] = None
        world_model.clear()
        logger.info("Stopped; idle.")
        return True

    words = set(re.findall(r"[a-z]+", low))
    if "put" in words and words.intersection(("down", "back")) and words.intersection(("can", "it")):
        if not action_state["held_can"]:
            logger.info("The red can is not in the robot's hand.")
            return True
        mission.reset()
        action_state["pick_up"] = False
        action_state["put_down"] = "back" if "back" in words else "down"
        action_state["target_xy"] = None
        world_model.clear()
        logger.info("Putting %s the red can.", "back" if action_state["put_down"] == "back" else "down")
        return True

    query = parse_target_instruction(text)
    if query is None:
        logger.info("Unsupported target. Try one of: %s", ", ".join(SUPPORTED_CATEGORIES))
        return True
    wants_pickup = "pick up" in low or "pickup" in low or "grab" in low
    if wants_pickup and not (query.category == "can" and query.color == "red"):
        logger.info("V2 pickup currently supports only the red can.")
        return True
    if wants_pickup and action_state["held_can"]:
        logger.info("Already holding the red can; put it down before picking it up again.")
        return True

    world_model.clear()
    action_state["pick_up"] = wants_pickup
    action_state["put_down"] = None
    action_state["target_xy"] = None
    mission.config.nav.stop_distance_m = action_state["normal_stop"]
    mission.start(query)
    if query.supported_by_pretrained:
        note = ""
    elif mission.detector.name == "OracleDetector(omniscient)":
        note = " (using oracle scene identity)"
    else:
        note = " (no COCO class; using colour fallback)"
    action = "Picking up" if wants_pickup else "Going to"
    logger.info("%s %s%s. Searching.", action, query.label, note)
    return True


def run_demo(
    seed: Optional[int] = None,
    backend: str = "oracle",
    show_overlay: bool = False,
    stop_distance_m: Optional[float] = None,
) -> None:
    package_dir = Path(__file__).resolve().parent
    xml_path = package_dir / "env" / "scene.xml"

    cfg = Config(seed=seed)
    cfg.detector.backend = backend
    if stop_distance_m is not None:
        cfg.nav.stop_distance_m = stop_distance_m

    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    placements = randomize_scene_objects(model, data, seed=seed)
    initialize_gripper(model, data)
    can_joint = model.joint("target_can_red_free").id
    can_qpos = int(model.jnt_qposadr[can_joint])
    can_qvel = int(model.jnt_dofadr[can_joint])

    robot_state = initialize_robot_state(model, data)
    apply_robot_state(model, data, robot_state)  # pin robot at mount height

    detector = build_detector(
        cfg.detector,
        model=model,
        data=data,
        width=cfg.camera.width,
        height=cfg.camera.height,
        fovy_deg=cfg.camera.fovy_deg,
        cam_local_offset=cfg.camera.local_offset,
    )
    mission = Mission(model=model, data=data, detector=detector, config=cfg)
    world_model = WorldModel()
    action_state = {
        "pick_up": False,
        "put_down": None,
        "held_can": False,
        "can_home_qpos": data.qpos[can_qpos:can_qpos + 7].copy(),
        "can_base_z": float(data.qpos[can_qpos + 2]),
        "target_xy": None,
        "normal_stop": cfg.nav.stop_distance_m,
    }

    command_state: dict = {"pending": None}
    lock = threading.Lock()
    running = threading.Event()
    running.set()
    reader = threading.Thread(target=_command_reader, args=(command_state, lock, running), daemon=True)

    overlay_on = show_overlay
    with mujoco.viewer.launch_passive(model, data) as viewer:
        set_user_free_camera(viewer)
        logger.info("Detector=%s seed=%s stop=%.2fm", detector.name, seed, cfg.nav.stop_distance_m)
        logger.info("Layout: %s", {k: tuple(round(v, 2) for v in xyz) for k, xyz in placements.items()})
        reader.start()
        viewer.sync()

        try:
            while viewer.is_running():
                with lock:
                    pending, command_state["pending"] = command_state["pending"], None
                if pending is not None and not _handle_command(pending, mission, world_model, action_state):
                    break

                put_down = action_state["put_down"]
                if put_down is not None:
                    action_state["put_down"] = None
                    placed = put_down_red_can(
                        model, data, viewer, robot_state,
                        action_state["can_home_qpos"], put_back=put_down == "back",
                    )
                    if placed is None:
                        logger.info("PUT-DOWN FAILED: could not reach a placement spot; the can stays held. Move near the table or counter and retry.")
                    else:
                        surface, can_base_z = placed
                        action_state["held_can"] = False
                        action_state["can_base_z"] = can_base_z
                        world_model.clear()
                        logger.info("PUT-DOWN DONE: red can placed on the %s; arm retracted.", surface)

                if mission.active:
                    result = mission.step(robot_state)
                    if action_state["held_can"]:
                        sync_held_red_can(model, data)
                    if result.target_xy is not None:
                        world_model.update(mission.query.label, result.target_xy)
                        action_state["target_xy"] = result.target_xy.copy()
                    if result.phase in (Phase.DONE, Phase.FAILED):
                        logger.info("%s: %s", result.phase.value.upper(), result.message)
                    if result.phase == Phase.DONE and action_state["pick_up"]:
                        target_xy = action_state["target_xy"]
                        action_state["pick_up"] = False
                        if target_xy is None:
                            logger.info("PICKUP FAILED: no camera position available")
                        else:
                            logger.info("Red can reached; moving the arm to the camera position.")
                            can_before = data.qpos[can_qpos:can_qpos + 7].copy()
                            picked = pick_up_red_can(
                                model, data, viewer, target_xy, robot_state,
                                can_base_z=action_state["can_base_z"],
                            )
                            if picked:
                                action_state["held_can"] = True
                                sync_held_red_can(model, data)
                            else:
                                data.qpos[can_qpos:can_qpos + 7] = can_before
                                data.qvel[can_qvel:can_qvel + 6] = 0.0
                                mujoco.mj_forward(model, data)
                                stow_right_arm(model, data, viewer, open_hand=True)
                            logger.info("PICKUP %s", "DONE" if picked else "FAILED: arm could not reach the can")
                        mission.config.nav.stop_distance_m = action_state["normal_stop"]
                    if overlay_on:
                        overlay_on = _show_overlay(result)

                _draw_markers(viewer, world_model, mission.query.label if mission.query else None)
                viewer.sync()
                if not mission.active:
                    running.wait(0.01)
        finally:
            running.clear()
            mission.close()
            if overlay_on:
                try:
                    import cv2

                    cv2.destroyAllWindows()
                except Exception:  # noqa: BLE001 - cleanup must never raise
                    pass


def _parse_args(argv: Optional[list] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Vision2Action interactive demo")
    p.add_argument("--seed", type=int, default=None, help="reproducible object placement seed")
    p.add_argument("--backend", choices=("oracle",), default="oracle",
                   help="oracle ground-truth projection backend")
    p.add_argument("--stop-distance", type=float, default=None,
                   help="override stop distance in metres (default from config, ~0.85)")
    p.add_argument("--overlay", action="store_true", help="show the robot-camera debug window")
    p.add_argument("--log", default="INFO", help="logging level (DEBUG/INFO/WARNING)")
    return p.parse_args(argv)


if __name__ == "__main__":
    args = _parse_args()
    logging.basicConfig(level=args.log.upper(), format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    run_demo(
        seed=args.seed,
        backend=args.backend,
        show_overlay=args.overlay,
        stop_distance_m=args.stop_distance,
    )
