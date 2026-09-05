"""Safe, seeded placement for the movable kitchen objects.

Places every :data:`~vision2action.env.objects.MOVABLES` item so that it:

* stays inside the room (away from walls),
* does not spawn inside a fixed appliance (fridge, counter, table, ...),
* does not overlap another movable,
* keeps clear of the robot's start position,
* gets a reasonable random yaw and rests at its authored height.

``seed`` makes a layout exactly reproducible.  The returned poses are for
logging / evaluation only; navigation must use the camera pipeline.
"""

from __future__ import annotations

from typing import Optional

import mujoco
import numpy as np

from vision2action.env.objects import FLOOR_OBSTACLES, MOVABLES, BY_NAME


_BANANA_TABLE_OFFSET = (-0.48, 0.25)  # tabletop corner facing the room centre
_RED_CAN_TABLE_OFFSET = (0.32, 0.12)
_BANANA_TABLE_JITTER = 0.04


def _appliance_obstacles(model: mujoco.MjModel, data: mujoco.MjData) -> list[tuple[float, float, float]]:
    obstacles: list[tuple[float, float, float]] = []
    for name in FLOOR_OBSTACLES:
        obj = BY_NAME[name]
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, obj.body_name)
        if bid < 0:
            continue
        x, y = data.xpos[bid][:2]
        obstacles.append((float(x), float(y), obj.footprint_radius))
    return obstacles


def randomize_scene_objects(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    seed: Optional[int] = None,
    *,
    robot_clearance: float = 0.9,
    wall_margin: float = 0.6,
) -> dict[str, tuple[float, float, float]]:
    """Randomise movable object poses; return ``name -> (x, y, yaw)`` ground truth."""
    # Callers may pass a newly-created MjData whose derived body positions have
    # not been populated yet.
    mujoco.mj_forward(model, data)
    rng = np.random.default_rng(seed)
    obstacles = _appliance_obstacles(model, data)
    placed: list[tuple[float, float, float]] = []
    poses: dict[str, tuple[float, float, float]] = {}

    lo = -3.5 + wall_margin
    hi = 3.5 - wall_margin
    # Keep movables south of the counter zone (counter spans y ~ 3.0-3.7).
    y_hi = 2.4

    for obj in MOVABLES:
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{obj.body_name}_free")
        if joint_id < 0:
            raise RuntimeError(f"Movable object {obj.name!r} has no free joint {obj.body_name}_free")
        adr = model.jnt_qposadr[joint_id]
        dof = model.jnt_dofadr[joint_id]
        z = obj.rest_z if obj.rest_z is not None else float(data.qpos[adr + 2])

        # A standing humanoid can reach these two tabletop objects.
        if obj.name in ("banana", "can_red"):
            table_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "table")
            if table_id < 0:
                raise RuntimeError("Scene has no table body")
            table_x, table_y = (float(v) for v in data.xpos[table_id][:2])
            offset = _BANANA_TABLE_OFFSET if obj.name == "banana" else _RED_CAN_TABLE_OFFSET
            x = table_x + offset[0] + float(
                rng.uniform(-_BANANA_TABLE_JITTER, _BANANA_TABLE_JITTER)
            )
            y = table_y + offset[1] + float(
                rng.uniform(-_BANANA_TABLE_JITTER, _BANANA_TABLE_JITTER)
            )
            yaw = float(rng.uniform(-np.pi, np.pi))
            data.qpos[adr : adr + 3] = (x, y, z)
            data.qpos[adr + 3 : adr + 7] = (np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2))
            data.qvel[dof : dof + 6] = 0.0
            poses[obj.name] = (x, y, yaw)
            continue

        for _ in range(2000):
            x = float(rng.uniform(lo, hi))
            y = float(rng.uniform(lo, y_hi))
            if float(np.hypot(x, y)) < robot_clearance:
                continue
            if any(np.hypot(x - ox, y - oy) < obj.footprint_radius + r for ox, oy, r in obstacles):
                continue
            if any(
                np.hypot(x - px, y - py) < obj.footprint_radius + pr + 0.15 for px, py, pr in placed
            ):
                continue
            yaw = float(rng.uniform(-np.pi, np.pi))
            data.qpos[adr : adr + 3] = (x, y, z)
            data.qpos[adr + 3 : adr + 7] = (np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2))
            data.qvel[dof : dof + 6] = 0.0
            placed.append((x, y, obj.footprint_radius))
            poses[obj.name] = (x, y, yaw)
            break
        else:
            raise RuntimeError(f"Could not find a safe placement for {obj.name}")

    mujoco.mj_forward(model, data)
    return poses
