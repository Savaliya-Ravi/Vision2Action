"""Scene object registry: the single source of truth about what is in the scene.

Every object the pipeline cares about is described once here: its MuJoCo body,
the matching COCO class (or ``None`` if COCO has no
such class), its dominant colour, whether it is a fixed appliance or a movable
item, and a footprint radius used for collision-free placement.

``coco_class = None`` is deliberate and important: COCO has no ``can`` /
``coffee machine`` / ``trash bin`` class, so a pretrained detector cannot label
those.  Target selection uses this to decide when to fall back (colour/geometry
or, later, a custom-trained model) instead of silently failing.

``category`` is the human word a command uses ("can", "bottle", "fridge").  A
category can map to several concrete objects (a blue ``can`` and a red
``can_red`` are both category ``"can"``); ``color`` then disambiguates.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import mujoco


@dataclass(frozen=True)
class SceneObject:
    name: str  # canonical id used by commands / evaluation (unique)
    body_name: str  # MuJoCo body name
    category: str  # command word / class family, e.g. "can" (shared by can + can_red)
    coco_class: Optional[str]  # matching COCO class, or None
    color: Optional[str]  # dominant colour name, or None
    kind: str  # "appliance" | "movable"
    footprint_radius: float  # metres, for placement spacing
    rest_z: Optional[float] = None  # movable: z to reset to; None keeps authored z


APPLIANCES: tuple[SceneObject, ...] = (
    SceneObject("counter", "counter", "counter", None, None, "appliance", 1.75),
    SceneObject("sink", "sink", "sink", "sink", None, "appliance", 0.35),
    SceneObject("microwave", "microwave", "microwave", "microwave", None, "appliance", 0.30),
    SceneObject("coffee_machine", "coffee_machine", "coffee_machine", None, None, "appliance", 0.20),
    SceneObject("fridge", "fridge_carcass", "fridge", "refrigerator", "white", "appliance", 0.60),
    SceneObject("oven", "oven", "oven", "oven", "white", "appliance", 0.55),
    SceneObject("table", "table", "table", "dining table", "brown", "appliance", 0.95),
    SceneObject("chair", "chair", "chair", "chair", "brown", "appliance", 0.30),
    SceneObject("trash_bin", "trash_bin", "trash_bin", None, None, "appliance", 0.28),
)

MOVABLES: tuple[SceneObject, ...] = (
    SceneObject("bottle", "target_bottle", "bottle", "bottle", "yellow", "movable", 0.14, rest_z=0.30),
    SceneObject("mug", "target_mug", "mug", "cup", "red", "movable", 0.14, rest_z=0.14),
    SceneObject("can", "target_can", "can", None, "blue", "movable", 0.12, rest_z=0.16),
    SceneObject("can_red", "target_can_red", "can", None, "red", "movable", 0.08, rest_z=0.795),
    SceneObject("banana", "target_banana", "banana", "banana", "yellow", "movable", 0.10, rest_z=0.795),
)

# Floor-standing appliances that a movable object must not spawn inside.
FLOOR_OBSTACLES: tuple[str, ...] = ("counter", "fridge", "oven", "table", "chair", "trash_bin")

ALL_OBJECTS: tuple[SceneObject, ...] = APPLIANCES + MOVABLES
BY_NAME: dict[str, SceneObject] = {o.name: o for o in ALL_OBJECTS}
BY_BODY: dict[str, SceneObject] = {o.body_name: o for o in ALL_OBJECTS}


def objects_in_category(category: str) -> list[SceneObject]:
    """All scene objects sharing a command category (e.g. both cans for "can")."""
    return [o for o in ALL_OBJECTS if o.category == category]


def resolve_object(category: str, color: Optional[str] = None) -> Optional[SceneObject]:
    """Best concrete object for a (category, colour) command.

    **Evaluation / debugging only** - used to look up ground truth for the
    localization-error metric, never to steer navigation.  When several objects
    share a category, ``color`` disambiguates; without it the first is returned.
    """
    candidates = objects_in_category(category)
    if not candidates:
        return None
    if color is not None:
        colored = [o for o in candidates if o.color == color]
        if colored:
            return colored[0]
    return candidates[0]


def ground_truth_xy(model: mujoco.MjModel, data: mujoco.MjData, name: str) -> Optional[tuple[float, float]]:
    """True planar position of an object's body centre.

    **Evaluation / debugging only** - never fed to navigation.
    """
    obj = BY_NAME.get(name)
    if obj is None:
        return None
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, obj.body_name)
    if bid < 0:
        return None
    x, y = data.xpos[bid][:2]
    return float(x), float(y)
