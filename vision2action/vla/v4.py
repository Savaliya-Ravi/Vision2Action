"""V4 demonstration adaptation and learned Octo-to-G1 fridge policy."""

from __future__ import annotations

import argparse
import json
import os
import queue
import threading
import time
from pathlib import Path
from typing import Callable

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import mujoco
import numpy as np

from vision2action.vla.__main__ import (
    CHECKPOINT as OCTO_CHECKPOINT,
    DOOR_OPEN_DEG,
    SCENE,
    START_LEFT_ARM,
    _camera_rgb,
    _read_terminal_commands,
    _require_desktop_viewer,
    _show_simulation_step,
)
from vision2action.vla.g1_adapter import (
    G1ActionAdapter,
    hold_current_joint_positions,
)
from vision2action.vla.octo_policy import OctoPolicy
from vision2action.vla.v4_policy import (
    ACTION_DIM,
    DEFAULT_INTENT_HEAD,
    HEAD_FORMAT_VERSION,
    INTENT_FORMAT_VERSION,
    V4Policy,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET = ROOT / "data" / "v4_fridge_demonstrations.npz"
DEFAULT_ACTION_HEAD = ROOT / "vision2action" / "vla" / "checkpoints" / "v4_g1_fridge_action_head.npz"
VERSION = "4.1.0"
INSTRUCTION = "open the fridge door"
INTENT_TRAIN_PHRASES = (
    ("open the fridge door", 1),
    ("please open the fridge door", 1),
    ("open fridge door", 1),
    ("pull open the fridge door", 1),
    ("leave the fridge door closed", 0),
    ("do not open the fridge door", 0),
    ("wait beside the fridge", 0),
    ("do not touch the fridge", 0),
    ("close the fridge door", 0),
)
INTENT_HOLDOUT_PHRASES = (
    ("open the refrigerator door", 1),
    ("could you open the fridge", 1),
    ("keep the fridge closed", 0),
    ("leave it closed", 0),
    ("pick up the red can", 0),
)
V4_START_XY = (-3.0, 1.8)
V4_START_YAW = np.pi / 2
V4_START_JOINTS = {
    "waist_yaw_joint": 0.5556741101342628,
    "waist_roll_joint": 0.22145331290988188,
    "waist_pitch_joint": 0.30839792805906496,
    "right_shoulder_pitch_joint": -2.755620091412494,
    "right_shoulder_roll_joint": -0.4338654365028243,
    "right_shoulder_yaw_joint": -1.3546734827100508,
    "right_elbow_joint": 1.1327992177660964,
    "right_wrist_roll_joint": -1.97222,
    "right_wrist_pitch_joint": 0.8099239406141652,
    "right_wrist_yaw_joint": 0.20722677683483162,
}
# Used only to create imitation-learning demonstrations. The V4 evaluator and
# interactive runner never call or inspect this expert.
EXPERT_ACTION = np.array([-0.025, 0.025, -0.025, 0.0, 0.0, 0.0, 1.0], dtype=float)
_EXPERT_ALTERNATE_ACTION = np.array([0.0, -0.025, -0.025, 0.0, 0.0, 0.0, 1.0], dtype=float)


def _expert_action(decision: int) -> np.ndarray:
    """Return one label from the successful alternating demonstration."""
    return (EXPERT_ACTION if decision % 2 == 0 else _EXPERT_ALTERNATE_ACTION).copy()


def initialize_v4_trial(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    """Set the collision-free V4 start pose beside the closed fridge."""
    free = model.joint("robot_free").id
    qpos = int(model.jnt_qposadr[free])
    qvel = int(model.jnt_dofadr[free])
    data.qpos[qpos:qpos + 3] = [V4_START_XY[0], V4_START_XY[1], 0.793]
    data.qpos[qpos + 3:qpos + 7] = [
        np.cos(V4_START_YAW / 2), 0.0, 0.0, np.sin(V4_START_YAW / 2)
    ]
    data.qvel[qvel:qvel + 6] = 0.0
    data.qpos[model.joint("fridge_door_hinge").qposadr[0]] = 0.0
    for name, value in (START_LEFT_ARM | V4_START_JOINTS).items():
        data.qpos[model.joint(name).qposadr[0]] = value
    for name in (
        "right_hand_thumb_0_joint", "right_hand_thumb_1_joint", "right_hand_thumb_2_joint",
        "right_hand_index_0_joint", "right_hand_index_1_joint",
        "right_hand_middle_0_joint", "right_hand_middle_1_joint",
    ):
        data.qpos[model.joint(name).qposadr[0]] = 0.0
    hold_current_joint_positions(model, data)
    mujoco.mj_forward(model, data)


def _is_descendant(model: mujoco.MjModel, body: int, root: int) -> bool:
    while body:
        if body == root:
            return True
        body = int(model.body_parentid[body])
    return False


def robot_door_contact(model: mujoco.MjModel, data: mujoco.MjData) -> bool:
    """Return whether a current physics contact joins the robot and door."""
    robot = model.body("robot").id
    door = model.body("fridge_door").id
    for contact in data.contact[:data.ncon]:
        body1 = int(model.geom_bodyid[contact.geom1])
        body2 = int(model.geom_bodyid[contact.geom2])
        if (body1 == door and _is_descendant(model, body2, robot)) or (
            body2 == door and _is_descendant(model, body1, robot)
        ):
            return True
    return False


def _make_trial() -> tuple[mujoco.MjModel, mujoco.MjData, G1ActionAdapter, int]:
    model = mujoco.MjModel.from_xml_path(str(SCENE))
    data = mujoco.MjData(model)
    initialize_v4_trial(model, data)
    adapter = G1ActionAdapter(model, data, control_waist=True)
    hinge_q = int(model.joint("fridge_door_hinge").qposadr[0])
    return model, data, adapter, hinge_q


def expert_physics_rollout(max_decisions: int = 80) -> dict:
    """Run the demonstration expert without rendering, for validation/tests."""
    model, data, adapter, hinge_q = _make_trial()
    contact_steps = 0
    records = []

    def on_step() -> None:
        nonlocal contact_steps
        contact_steps += int(robot_door_contact(model, data))

    for decision in range(max_decisions):
        adapter.apply(_expert_action(decision), on_step=on_step)
        door_deg = float(np.rad2deg(data.qpos[hinge_q]))
        records.append(door_deg)
        if door_deg >= DOOR_OPEN_DEG:
            break
    return {
        "door_deg": records[-1] if records else 0.0,
        "decisions": len(records),
        "contact_steps": contact_steps,
        "opened": bool(records and records[-1] >= DOOR_OPEN_DEG and contact_steps > 0),
    }


def collect_demonstrations(
    output: Path = DEFAULT_DATASET,
    episodes: int = 4,
    max_decisions: int = 80,
    instruction: str = INSTRUCTION,
) -> dict:
    """Render successful MuJoCo expert episodes into a compact NumPy dataset."""
    if episodes < 1 or max_decisions < 1:
        raise ValueError("episodes and max_decisions must be positive")
    primary_frames: list[np.ndarray] = []
    wrist_frames: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    door_angles: list[float] = []
    episode_ids: list[int] = []
    instructions: list[str] = []
    outcomes = []

    for episode in range(episodes):
        model, data, adapter, hinge_q = _make_trial()
        primary_renderer = mujoco.Renderer(model, height=256, width=256)
        wrist_renderer = mujoco.Renderer(model, height=128, width=128)
        contact_steps = 0

        def on_step() -> None:
            nonlocal contact_steps
            contact_steps += int(robot_door_contact(model, data))

        try:
            for decision in range(max_decisions):
                action = _expert_action(decision)
                primary_frames.append(_camera_rgb(primary_renderer, data, "vla_fridge_cam"))
                wrist_frames.append(_camera_rgb(wrist_renderer, data, "vla_wrist_cam"))
                actions.append(action)
                door_angles.append(float(np.rad2deg(data.qpos[hinge_q])))
                episode_ids.append(episode)
                instructions.append(instruction)
                adapter.apply(action, on_step=on_step)
                if np.rad2deg(data.qpos[hinge_q]) >= DOOR_OPEN_DEG:
                    break
        finally:
            primary_renderer.close()
            wrist_renderer.close()

        door_deg = float(np.rad2deg(data.qpos[hinge_q]))
        opened = door_deg >= DOOR_OPEN_DEG and contact_steps > 0
        outcomes.append({
            "episode": episode,
            "door_deg": door_deg,
            "contact_steps": contact_steps,
            "opened": opened,
        })
        if not opened:
            raise RuntimeError(f"Expert demonstration {episode} failed: {outcomes[-1]}")

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        format_version=np.array(1, dtype=np.int64),
        primary=np.stack(primary_frames).astype(np.uint8),
        wrist=np.stack(wrist_frames).astype(np.uint8),
        action=np.stack(actions).astype(np.float32),
        door_deg=np.asarray(door_angles, dtype=np.float32),
        episode=np.asarray(episode_ids, dtype=np.int32),
        instruction=np.asarray(instructions),
    )
    summary = {
        "version": VERSION,
        "path": str(output),
        "episodes": episodes,
        "samples": len(actions),
        "success_rate": float(np.mean([item["opened"] for item in outcomes])),
        "outcomes": outcomes,
    }
    print(json.dumps(summary, indent=2))
    return summary


def _load_dataset(path: Path) -> dict[str, np.ndarray]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"V4 demonstration dataset missing: {path}")
    with np.load(path, allow_pickle=False) as saved:
        dataset = {key: np.asarray(saved[key]) for key in saved.files}
    if int(dataset.get("format_version", -1)) != 1:
        raise ValueError("Unsupported V4 demonstration format")
    count = len(dataset["action"])
    expected = {
        "primary": (count, 256, 256, 3),
        "wrist": (count, 128, 128, 3),
        "action": (count, ACTION_DIM),
        "door_deg": (count,),
        "episode": (count,),
        "instruction": (count,),
    }
    for name, shape in expected.items():
        if dataset[name].shape != shape:
            raise ValueError(f"Dataset {name} must have shape {shape}, got {dataset[name].shape}")
    if dataset["primary"].dtype != np.uint8 or dataset["wrist"].dtype != np.uint8:
        raise ValueError("V4 demonstration images must be uint8")
    if not np.all(np.isfinite(dataset["action"])):
        raise ValueError("V4 demonstration actions must be finite")
    return dataset


def train_action_head(
    dataset_path: Path = DEFAULT_DATASET,
    output: Path = DEFAULT_ACTION_HEAD,
    octo_checkpoint: Path = OCTO_CHECKPOINT,
    ridge: float = 1e-3,
) -> dict:
    """Fit a calibrated G1 action head on frozen Octo RGB/text features."""
    if ridge <= 0:
        raise ValueError("ridge must be positive")
    dataset = _load_dataset(dataset_path)
    instructions = dataset["instruction"].astype(str)
    encoder = OctoPolicy(octo_checkpoint, instructions[0])
    features = []
    previous_episode = None
    for index, episode in enumerate(dataset["episode"]):
        episode = int(episode)
        if episode != previous_episode:
            encoder.set_instruction(instructions[index])
            previous_episode = episode
        features.append(encoder.encode(dataset["primary"][index], dataset["wrist"][index]))
        print(f"encoded {index + 1}/{len(instructions)}", flush=True)

    features = np.asarray(features, dtype=np.float64)
    targets = np.asarray(dataset["action"], dtype=np.float64)
    feature_mean = features.mean(axis=0)
    feature_std = np.maximum(features.std(axis=0), 1e-4)
    normalized = (features - feature_mean) / feature_std
    target_mean = targets.mean(axis=0)
    centered_targets = targets - target_mean
    # Dual ridge is stable when demonstrations are fewer than Octo's 384 features.
    gram = normalized @ normalized.T + ridge * np.eye(len(normalized))
    weights = normalized.T @ np.linalg.solve(gram, centered_targets)
    predictions = normalized @ weights + target_mean
    mse = float(np.mean(np.square(predictions - targets)))
    action_prototypes = np.unique(targets, axis=0)

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        format_version=np.array(HEAD_FORMAT_VERSION, dtype=np.int64),
        weights=weights.astype(np.float32),
        bias=target_mean.astype(np.float32),
        feature_mean=feature_mean.astype(np.float32),
        feature_std=feature_std.astype(np.float32),
        action_prototypes=action_prototypes.astype(np.float32),
        training_samples=np.array(len(targets), dtype=np.int64),
        training_episodes=np.array(len(np.unique(dataset["episode"])), dtype=np.int64),
        instruction=np.array(instructions[0]),
        base_checkpoint=np.array(str(octo_checkpoint)),
        train_mse=np.array(mse, dtype=np.float64),
        ridge=np.array(ridge, dtype=np.float64),
    )
    result = {
        "version": VERSION,
        "path": str(output),
        "device": encoder.devices,
        "samples": len(targets),
        "episodes": len(np.unique(dataset["episode"])),
        "train_mse": mse,
    }
    print(json.dumps(result, indent=2))
    return result


def train_intent_head(
    dataset_path: Path = DEFAULT_DATASET,
    output: Path = DEFAULT_INTENT_HEAD,
    octo_checkpoint: Path = OCTO_CHECKPOINT,
    ridge: float = 1.0,
) -> dict:
    """Learn a text-conditioned open/stop decision from paired RGB observations.

    Every phrase is encoded against the same initial camera images, so the
    supervision cannot be solved by the visual state alone. This is still a
    narrow command set, not a general language planner.
    """
    if ridge <= 0:
        raise ValueError("ridge must be positive")
    dataset = _load_dataset(dataset_path)
    primary, wrist = dataset["primary"][0], dataset["wrist"][0]
    encoder = OctoPolicy(octo_checkpoint, INTENT_TRAIN_PHRASES[0][0])
    features = []
    for phrase, _ in INTENT_TRAIN_PHRASES + INTENT_HOLDOUT_PHRASES:
        encoder.set_instruction(phrase)
        features.append(encoder.encode(primary, wrist))
        print(f"encoded intent: {phrase}", flush=True)
    features = np.asarray(features, dtype=np.float64)
    train_count = len(INTENT_TRAIN_PHRASES)
    targets = np.asarray([label for _, label in INTENT_TRAIN_PHRASES], dtype=np.float64)
    train_features = features[:train_count]
    feature_mean = train_features.mean(axis=0)
    feature_std = np.maximum(train_features.std(axis=0), 1e-3)
    normalized = (train_features - feature_mean) / feature_std
    gram = normalized @ normalized.T + ridge * np.eye(train_count)
    weights = normalized.T @ np.linalg.solve(gram, targets - targets.mean())
    bias = float(targets.mean())
    scores = ((features - feature_mean) / feature_std) @ weights + bias
    labels = np.asarray([label for _, label in INTENT_TRAIN_PHRASES + INTENT_HOLDOUT_PHRASES])
    predictions = scores >= 0.5

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        format_version=np.array(INTENT_FORMAT_VERSION, dtype=np.int64),
        weights=weights.astype(np.float32),
        bias=np.array(bias, dtype=np.float32),
        feature_mean=feature_mean.astype(np.float32),
        feature_std=feature_std.astype(np.float32),
        threshold=np.array(0.5, dtype=np.float32),
        train_phrases=np.asarray([phrase for phrase, _ in INTENT_TRAIN_PHRASES]),
        train_labels=targets.astype(np.int8),
        holdout_phrases=np.asarray([phrase for phrase, _ in INTENT_HOLDOUT_PHRASES]),
        holdout_labels=labels[train_count:].astype(np.int8),
        holdout_scores=scores[train_count:].astype(np.float32),
    )
    result = {
        "version": VERSION,
        "path": str(output),
        "train_accuracy": float(np.mean(predictions[:train_count] == labels[:train_count])),
        "held_out_phrase_accuracy": float(np.mean(predictions[train_count:] == labels[train_count:])),
        "held_out_phrases": [
            {"instruction": phrase, "score": float(score), "requests_opening": bool(prediction)}
            for (phrase, _), score, prediction in zip(
                INTENT_HOLDOUT_PHRASES, scores[train_count:], predictions[train_count:]
            )
        ],
    }
    print(json.dumps(result, indent=2))
    return result


def run_v4_trial(
    instruction: str = INSTRUCTION,
    decisions: int = 80,
    octo_checkpoint: Path = OCTO_CHECKPOINT,
    action_head: Path = DEFAULT_ACTION_HEAD,
    viewer_enabled: bool = False,
    trace_path: Path | None = None,
    policy_factory: Callable[[Path, Path, str], object] = V4Policy,
    expected_outcome: str = "open",
) -> dict:
    """Evaluate only model-produced actions against contact and door angle."""
    if decisions < 1:
        raise ValueError("decisions must be positive")
    if expected_outcome not in {"open", "stop"}:
        raise ValueError("expected_outcome must be 'open' or 'stop'")
    if viewer_enabled:
        _require_desktop_viewer()
    policy = policy_factory(octo_checkpoint, action_head, instruction)
    model, data, adapter, hinge_q = _make_trial()
    primary_renderer = mujoco.Renderer(model, height=256, width=256)
    wrist_renderer = mujoco.Renderer(model, height=128, width=128)
    records = []
    contact_steps = 0
    policy_stopped = False
    viewer = None

    def on_step() -> None:
        nonlocal contact_steps
        contact_steps += int(robot_door_contact(model, data))
        if viewer is not None:
            _show_simulation_step(viewer, model)

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
            prediction = policy.predict(primary, wrist)
            if prediction is None:
                policy_stopped = True
                print("policy selected STOP", flush=True)
                break
            action = np.asarray(prediction, dtype=float)
            adapter.apply(action, on_step=on_step)
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

    door_deg = float(np.rad2deg(data.qpos[hinge_q]))
    result = {
        "version": VERSION,
        "policy": "octo_frozen_encoder_learned_intent_and_g1_action_heads",
        "model": str(octo_checkpoint),
        "action_head": str(action_head),
        "intent_head": str(DEFAULT_INTENT_HEAD),
        "devices": list(policy.devices),
        "instruction": instruction,
        "success_threshold_deg": DOOR_OPEN_DEG,
        "start_xy": list(V4_START_XY),
        "decisions": len(records),
        "door_deg": door_deg,
        "contact_steps": contact_steps,
        "policy_stopped": policy_stopped,
        "opened": bool(door_deg >= DOOR_OPEN_DEG and contact_steps > 0),
        "expected_outcome": expected_outcome,
        "goal_satisfied": bool(
            (door_deg >= DOOR_OPEN_DEG and contact_steps > 0)
            if expected_outcome == "open"
            else (policy_stopped and abs(door_deg) < 1.0)
        ),
        "actions": records,
    }
    if trace_path is not None:
        trace_path = Path(trace_path)
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        trace_path.write_text(json.dumps(result, indent=2) + "\n")
    return result


def run_interactive_v4(
    decisions: int = 80,
    octo_checkpoint: Path = OCTO_CHECKPOINT,
    action_head: Path = DEFAULT_ACTION_HEAD,
    trace_path: Path | None = None,
) -> list[dict]:
    """Open MuJoCo, then execute each typed instruction with the V4 policy."""
    if decisions < 1:
        raise ValueError("decisions must be positive")
    _require_desktop_viewer()
    from mujoco import viewer as mujoco_viewer

    model, data, adapter, hinge_q = _make_trial()
    primary_renderer = mujoco.Renderer(model, height=256, width=256)
    wrist_renderer = mujoco.Renderer(model, height=128, width=128)
    commands: queue.Queue[str] = queue.Queue()
    sessions: list[dict] = []
    active: dict | None = None
    policy: V4Policy | None = None
    contact_steps = 0
    viewer = None

    def on_step() -> None:
        nonlocal contact_steps
        contact_steps += int(robot_door_contact(model, data))
        if viewer is not None:
            _show_simulation_step(viewer, model)

    try:
        viewer = mujoco_viewer.launch_passive(model, data)
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
        viewer.cam.fixedcamid = model.camera("vla_fridge_cam").id
        viewer.sync()
        threading.Thread(target=_read_terminal_commands, args=(commands,), daemon=True).start()
        print("V4 MuJoCo is ready. Type: open the fridge door", flush=True)
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
                        active["contact_steps"] = contact_steps
                        sessions.append(active)
                        active = None
                    if control == "reset":
                        mujoco.mj_resetData(model, data)
                        initialize_v4_trial(model, data)
                        adapter = G1ActionAdapter(model, data, control_waist=True)
                        contact_steps = 0
                        viewer.sync()
                        print("V4 scene reset; waiting for a command.", flush=True)
                    continue
                if active is not None:
                    active["outcome"] = "replaced"
                    active["contact_steps"] = contact_steps
                    sessions.append(active)
                if policy is None:
                    print("Loading Octo and the learned V4 action head...", flush=True)
                    policy = V4Policy(octo_checkpoint, action_head, command)
                    print(f"V4 ready on {policy.devices}", flush=True)
                else:
                    policy.set_instruction(command)
                active = {"instruction": command, "actions": []}
                contact_steps = 0

            if not running:
                break
            if active is None:
                viewer.sync()
                time.sleep(0.05)
                continue
            primary = _camera_rgb(primary_renderer, data, "vla_fridge_cam")
            wrist = _camera_rgb(wrist_renderer, data, "vla_wrist_cam")
            action = policy.predict(primary, wrist)
            if action is None:
                active.update(
                    outcome="policy_stopped",
                    door_deg=float(np.rad2deg(data.qpos[hinge_q])),
                    contact_steps=contact_steps,
                )
                print("Policy selected STOP; waiting for another command.", flush=True)
                sessions.append(active)
                active = None
                continue
            adapter.apply(action, on_step=on_step)
            door_deg = float(np.rad2deg(data.qpos[hinge_q]))
            count = len(active["actions"]) + 1
            active["actions"].append({"decision": count, "action": action.tolist(), "door_deg": door_deg})
            print(f"decision={count} door_deg={door_deg:.2f}", flush=True)
            if door_deg >= DOOR_OPEN_DEG or count >= decisions:
                opened = door_deg >= DOOR_OPEN_DEG and contact_steps > 0
                active.update(
                    outcome="opened" if opened else "limit_reached",
                    door_deg=door_deg,
                    contact_steps=contact_steps,
                )
                print(f"{'OPENED' if opened else 'STOPPED'}: door={door_deg:.2f}°", flush=True)
                sessions.append(active)
                active = None
    finally:
        if active is not None:
            active["outcome"] = "interrupted"
            active["contact_steps"] = contact_steps
            sessions.append(active)
        primary_renderer.close()
        wrist_renderer.close()
        if viewer is not None:
            viewer.close()
        if trace_path is not None:
            trace_path = Path(trace_path)
            trace_path.parent.mkdir(parents=True, exist_ok=True)
            trace_path.write_text(json.dumps({"version": VERSION, "sessions": sessions}, indent=2) + "\n")
    return sessions


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="V4.1: text-conditioned Octo policy for the G1 fridge")
    sub = parser.add_subparsers(dest="command", required=True)

    collect = sub.add_parser("collect", help="collect successful MuJoCo demonstrations")
    collect.add_argument("--output", type=Path, default=DEFAULT_DATASET)
    collect.add_argument("--episodes", type=int, default=4)
    collect.add_argument("--decisions", type=int, default=80)
    collect.add_argument("--instruction", default=INSTRUCTION)

    train = sub.add_parser("train", help="train the G1 action head on frozen Octo features")
    train.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    train.add_argument("--output", type=Path, default=DEFAULT_ACTION_HEAD)
    train.add_argument("--checkpoint", type=Path, default=OCTO_CHECKPOINT)
    train.add_argument("--ridge", type=float, default=1e-3)

    train_intent = sub.add_parser("train-intent", help="train text-conditioned opening or stopping")
    train_intent.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    train_intent.add_argument("--output", type=Path, default=DEFAULT_INTENT_HEAD)
    train_intent.add_argument("--checkpoint", type=Path, default=OCTO_CHECKPOINT)
    train_intent.add_argument("--ridge", type=float, default=1.0)

    evaluate = sub.add_parser("eval", help="run a headless or desktop V4 evaluation")
    evaluate.add_argument("--instruction", default=INSTRUCTION)
    evaluate.add_argument("--decisions", type=int, default=80)
    evaluate.add_argument("--checkpoint", type=Path, default=OCTO_CHECKPOINT)
    evaluate.add_argument("--action-head", type=Path, default=DEFAULT_ACTION_HEAD)
    evaluate.add_argument("--viewer", action="store_true")
    evaluate.add_argument("--trace", type=Path)
    evaluate.add_argument("--expect", choices=("open", "stop"), default="open")

    interactive = sub.add_parser("interactive", help="open MuJoCo and wait for typed commands")
    interactive.add_argument("--decisions", type=int, default=80)
    interactive.add_argument("--checkpoint", type=Path, default=OCTO_CHECKPOINT)
    interactive.add_argument("--action-head", type=Path, default=DEFAULT_ACTION_HEAD)
    interactive.add_argument("--trace", type=Path)

    bootstrap = sub.add_parser("bootstrap", help="collect, train both heads, and evaluate V4.1")
    bootstrap.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    bootstrap.add_argument("--action-head", type=Path, default=DEFAULT_ACTION_HEAD)
    bootstrap.add_argument("--checkpoint", type=Path, default=OCTO_CHECKPOINT)
    bootstrap.add_argument("--episodes", type=int, default=4)
    bootstrap.add_argument("--decisions", type=int, default=80)
    bootstrap.add_argument("--trace", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.command == "collect":
        collect_demonstrations(args.output, args.episodes, args.decisions, args.instruction)
        return 0
    if args.command == "train":
        train_action_head(args.dataset, args.output, args.checkpoint, args.ridge)
        return 0
    if args.command == "train-intent":
        train_intent_head(args.dataset, args.output, args.checkpoint, args.ridge)
        return 0
    if args.command == "eval":
        result = run_v4_trial(
            args.instruction, args.decisions, args.checkpoint, args.action_head,
            args.viewer, args.trace, expected_outcome=args.expect,
        )
        print(json.dumps({key: value for key, value in result.items() if key != "actions"}, indent=2))
        return 0 if result["goal_satisfied"] else 1
    if args.command == "interactive":
        run_interactive_v4(args.decisions, args.checkpoint, args.action_head, args.trace)
        return 0

    collect_demonstrations(args.dataset, args.episodes, args.decisions, INSTRUCTION)
    train_action_head(args.dataset, args.action_head, args.checkpoint)
    train_intent_head(args.dataset, DEFAULT_INTENT_HEAD, args.checkpoint)
    result = run_v4_trial(
        INSTRUCTION, args.decisions, args.checkpoint, args.action_head, False, args.trace
    )
    print(json.dumps({key: value for key, value in result.items() if key != "actions"}, indent=2))
    return 0 if result["goal_satisfied"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
