"""Run V3 Octo control of the G1 at a fixed fridge starting pose."""

from __future__ import annotations

import argparse
import json
import os
import queue
import socket
import sys
import threading
import time
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import mujoco
import numpy as np

from vision2action.vla.g1_adapter import G1ActionAdapter, hold_current_joint_positions
from vision2action.vla.octo_policy import OctoPolicy


ROOT = Path(__file__).resolve().parents[2]
SCENE = ROOT / "vision2action" / "env" / "scene.xml"
CHECKPOINT = ROOT / "checkpoints" / "octo-small-1.5"
START_XY = (-3.0, 2.05)
START_YAW = np.pi / 2
DOOR_OPEN_DEG = 30.0
# A fixed, near-handle initial posture makes V3 a local manipulation trial.
# These values are an initial condition; they are never replayed as a trajectory.
START_ARM = {
    "right_shoulder_pitch_joint": -1.2804,
    "right_shoulder_roll_joint": -0.1427,
    "right_shoulder_yaw_joint": -0.1423,
    "right_elbow_joint": 0.4916,
    "right_wrist_roll_joint": 0.0971,
    "right_wrist_pitch_joint": -0.2932,
    "right_wrist_yaw_joint": -0.0293,
}
START_LEFT_ARM = {
    "left_shoulder_pitch_joint": 0.2,
    "left_shoulder_roll_joint": 0.2,
    "left_elbow_joint": 1.28,
}


def initialize_fridge_trial(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    """Place the robot at the known fridge location; no motion policy is run here."""
    free = model.joint("robot_free").id
    adr = int(model.jnt_qposadr[free])
    data.qpos[adr:adr + 3] = [START_XY[0], START_XY[1], 0.793]
    data.qpos[adr + 3:adr + 7] = [np.cos(START_YAW / 2), 0.0, 0.0, np.sin(START_YAW / 2)]
    for name, value in (START_LEFT_ARM | START_ARM).items():
        data.qpos[model.joint(name).qposadr[0]] = value
    hold_current_joint_positions(model, data)
    mujoco.mj_forward(model, data)


def _camera_rgb(renderer: mujoco.Renderer, data: mujoco.MjData, name: str) -> np.ndarray:
    renderer.update_scene(data, camera=name)
    return renderer.render().copy()


def run_trial(
    instruction: str,
    decisions: int,
    checkpoint: Path = CHECKPOINT,
    viewer_enabled: bool = False,
    trace_path: Path | None = None,
) -> dict:
    if decisions < 1:
        raise ValueError("--decisions must be positive")
    if viewer_enabled:
        _require_desktop_viewer()
    policy = OctoPolicy(checkpoint, instruction)
    model = mujoco.MjModel.from_xml_path(str(SCENE))
    data = mujoco.MjData(model)
    initialize_fridge_trial(model, data)
    adapter = G1ActionAdapter(model, data)
    hinge = model.joint("fridge_door_hinge").id
    hinge_q = int(model.jnt_qposadr[hinge])
    primary_renderer = mujoco.Renderer(model, height=256, width=256)
    wrist_renderer = mujoco.Renderer(model, height=128, width=128)
    viewer = None
    records = []

    try:
        if viewer_enabled:
            from mujoco import viewer as mujoco_viewer
            viewer = mujoco_viewer.launch_passive(model, data)
            viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
            viewer.cam.fixedcamid = model.camera("vla_fridge_cam").id
        for index in range(decisions):
            if viewer is not None and not viewer.is_running():
                break
            primary = _camera_rgb(primary_renderer, data, "vla_fridge_cam")
            wrist = _camera_rgb(wrist_renderer, data, "vla_wrist_cam")
            action = policy.predict(primary, wrist)
            if viewer is None:
                adapter.apply(action)
            else:
                adapter.apply(action, on_step=lambda: _show_simulation_step(viewer, model))
            door_deg = float(np.rad2deg(data.qpos[hinge_q]))
            records.append({"decision": index + 1, "action": action.tolist(), "door_deg": door_deg})
            print(f"decision={index + 1} door_deg={door_deg:.2f} action={np.round(action, 3).tolist()}", flush=True)
            if viewer is not None:
                viewer.sync()
            if door_deg >= DOOR_OPEN_DEG:
                break
    finally:
        primary_renderer.close()
        wrist_renderer.close()
        if viewer is not None:
            viewer.close()

    result = {
        "version": "3.0.0",
        "model": str(checkpoint),
        "devices": policy.devices,
        "action_statistics": policy.statistics_dataset,
        "instruction": instruction,
        "success_threshold_deg": DOOR_OPEN_DEG,
        "start_xy": list(START_XY),
        "decisions": len(records),
        "door_deg": float(np.rad2deg(data.qpos[hinge_q])),
        "opened": bool(np.rad2deg(data.qpos[hinge_q]) >= DOOR_OPEN_DEG),
        "actions": records,
    }
    if trace_path is not None:
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        trace_path.write_text(json.dumps(result, indent=2) + "\n")
    return result


def _read_terminal_commands(commands: queue.Queue[str]) -> None:
    while True:
        try:
            command = input("Command (or stop, reset, quit): ")
        except EOFError:
            commands.put("quit")
            return
        commands.put(command)


def _show_simulation_step(viewer, model: mujoco.MjModel) -> None:
    if viewer.is_running():
        viewer.sync()
        time.sleep(model.opt.timestep)


def _require_desktop_viewer() -> None:
    if os.environ.get("MUJOCO_GL", "").lower() in {"egl", "osmesa"}:
        raise RuntimeError("Interactive viewer needs desktop OpenGL. Run without MUJOCO_GL=egl.")
    if not sys.platform.startswith("linux"):
        return
    display = os.environ.get("DISPLAY", "")
    if not (display or os.environ.get("WAYLAND_DISPLAY")):
        raise RuntimeError("No desktop display found. Run --interactive from a graphical desktop session.")
    host, separator, number = display.partition(":")
    if separator and host in {"", "unix"}:
        display_number = number.split(".")[0] or "0"
        path = f"/tmp/.X11-unix/X{display_number}"
        try:
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
                connection.settimeout(0.3)
                connection.connect(path)
        except OSError as exc:
            raise RuntimeError("Cannot connect to the desktop display. Run from the graphical login session.") from exc


def run_interactive(decisions: int, checkpoint: Path = CHECKPOINT, trace_path: Path | None = None) -> list[dict]:
    """Show the scene first, then send each typed instruction directly to Octo."""
    if decisions < 1:
        raise ValueError("--decisions must be positive")
    _require_desktop_viewer()
    from mujoco import viewer as mujoco_viewer

    model = mujoco.MjModel.from_xml_path(str(SCENE))
    data = mujoco.MjData(model)
    initialize_fridge_trial(model, data)
    adapter = G1ActionAdapter(model, data)
    hinge_q = int(model.joint("fridge_door_hinge").qposadr[0])
    primary_renderer = mujoco.Renderer(model, height=256, width=256)
    wrist_renderer = mujoco.Renderer(model, height=128, width=128)
    commands: queue.Queue[str] = queue.Queue()
    sessions: list[dict] = []
    active: dict | None = None
    policy: OctoPolicy | None = None
    viewer = None

    try:
        viewer = mujoco_viewer.launch_passive(model, data)
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
        viewer.cam.fixedcamid = model.camera("vla_fridge_cam").id
        viewer.sync()
        threading.Thread(target=_read_terminal_commands, args=(commands,), daemon=True).start()
        print("MuJoCo is ready. Type a command in this terminal, for example: open the fridge door", flush=True)

        running = True
        while running and viewer.is_running():
            while True:
                try:
                    command = commands.get_nowait().strip()
                except queue.Empty:
                    break
                if not command:
                    continue
                control = command.lower()
                if control in {"quit", "exit"}:
                    running = False
                    break
                if control in {"stop", "reset"}:
                    if active is not None:
                        active["outcome"] = "stopped"
                        sessions.append(active)
                        active = None
                    if control == "reset":
                        mujoco.mj_resetData(model, data)
                        initialize_fridge_trial(model, data)
                        viewer.sync()
                        print("Scene reset. Door closed; waiting for a command.", flush=True)
                    else:
                        print("Stopped policy actions. Waiting for a command.", flush=True)
                    continue

                if active is not None:
                    active["outcome"] = "replaced"
                    sessions.append(active)
                    active = None
                if policy is None:
                    print("Loading Octo checkpoint for the first command...", flush=True)
                    policy = OctoPolicy(checkpoint, command)
                    print(f"Octo ready on {policy.devices}", flush=True)
                else:
                    policy.set_instruction(command)
                active = {"instruction": command, "actions": []}
                print(f"VLA instruction: {command}", flush=True)

            if not running:
                break
            if active is None:
                viewer.sync()
                time.sleep(0.05)
                continue

            primary = _camera_rgb(primary_renderer, data, "vla_fridge_cam")
            wrist = _camera_rgb(wrist_renderer, data, "vla_wrist_cam")
            action = policy.predict(primary, wrist)
            adapter.apply(action, on_step=lambda: _show_simulation_step(viewer, model))
            door_deg = float(np.rad2deg(data.qpos[hinge_q]))
            count = len(active["actions"]) + 1
            active["actions"].append({"decision": count, "action": action.tolist(), "door_deg": door_deg})
            print(f"decision={count} door_deg={door_deg:.2f}", flush=True)
            if door_deg >= DOOR_OPEN_DEG or count >= decisions:
                opened = door_deg >= DOOR_OPEN_DEG
                active["outcome"] = "opened" if opened else "limit_reached"
                active["door_deg"] = door_deg
                print(f"{'OPENED' if opened else 'STOPPED'}: door={door_deg:.2f} degrees; waiting for a command.", flush=True)
                sessions.append(active)
                active = None
    finally:
        if active is not None:
            active["outcome"] = "interrupted"
            sessions.append(active)
        primary_renderer.close()
        wrist_renderer.close()
        if viewer is not None:
            viewer.close()
        if trace_path is not None:
            trace_path.parent.mkdir(parents=True, exist_ok=True)
            trace_path.write_text(json.dumps({"version": "3.0.0", "sessions": sessions}, indent=2) + "\n")
    return sessions


def main() -> None:
    parser = argparse.ArgumentParser(description="V3: closed-loop Octo fridge-door trial on Unitree G1")
    parser.add_argument("--instruction", default="open the fridge door")
    parser.add_argument("--decisions", type=int, default=80)
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    parser.add_argument("--viewer", action="store_true", help="show the MuJoCo desktop viewer")
    parser.add_argument("--interactive", action="store_true", help="open the viewer and wait for typed commands")
    parser.add_argument("--trace", type=Path, help="save actions and door-angle results as JSON")
    args = parser.parse_args()
    if args.interactive:
        run_interactive(args.decisions, args.checkpoint, args.trace)
        return
    result = run_trial(args.instruction, args.decisions, args.checkpoint, args.viewer, args.trace)
    print(json.dumps({key: value for key, value in result.items() if key != "actions"}, indent=2))


if __name__ == "__main__":
    main()
