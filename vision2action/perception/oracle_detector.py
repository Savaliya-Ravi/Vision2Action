"""OracleDetector - a deterministic, ground-truth-based detector *test double*.

This provides a repeatable way to test selection, localization, and navigation
without detector noise. The oracle projects each
object's true geometry into the camera to produce exact boxes.

STRICT SCOPE: this is for tests, evaluation harnessing, and debugging only.  It
reads ground-truth body/geom poses, so it must **never** be used as the
perception backend when measuring the system, and it never feeds positions to
navigation directly. It only emits image-space boxes, and
the normal depth pipeline turns those into positions.

Two modes:

* ``known_classes`` given (e.g. :data:`coco_known_classes`) - the oracle behaves
  like a *perfect but COCO-limited* detector: it labels only objects whose COCO
  class is in the set and stays blind to "can"/"coffee machine"/... exactly like
  a pretrained model.  This exercises the colour-fallback path honestly.
* ``known_classes=None`` - omniscient: every object is labelled by its category.
  Use for pure geometry/localization/navigation testing.
"""

from __future__ import annotations

import itertools
import math
from typing import List, Optional

import mujoco
import numpy as np

from vision2action.control.controller import RobotState
from vision2action.env.objects import ALL_OBJECTS, SceneObject
from vision2action.perception.depth_utils import project_world_to_pixel
from vision2action.perception.detector import Detection, ObjectDetector, clip_bbox

_COLLISION_GROUP = 3  # group used by hidden collision geoms in scene.xml


def coco_known_classes() -> set[str]:
    """COCO classes present in the scene registry."""
    return {o.coco_class for o in ALL_OBJECTS if o.coco_class}


class OracleDetector(ObjectDetector):
    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        *,
        width: int = 320,
        height: int = 240,
        fovy_deg: float = 60.0,
        cam_local_offset: tuple[float, float, float] = (0.08, 0.0, 0.0),
        known_classes: Optional[set[str]] = None,
        min_bbox_area_px: int = 24,
        objects: Optional[tuple[SceneObject, ...]] = None,
    ) -> None:
        self.model = model
        self.data = data
        self.width = width
        self.height = height
        self.fovy_deg = fovy_deg
        self.cam_local_offset = cam_local_offset
        self.known_classes = known_classes
        self.min_bbox_area_px = min_bbox_area_px
        self.objects = objects if objects is not None else ALL_OBJECTS
        self._robot_state: Optional[RobotState] = None

    @property
    def name(self) -> str:
        mode = "omniscient" if self.known_classes is None else "coco-limited"
        return f"OracleDetector({mode})"

    def set_robot_state(self, state: RobotState) -> None:
        """Provide the authoritative robot pose (else it is read from ``data``)."""
        self._robot_state = state

    def _robot_pose(self) -> RobotState:
        if self._robot_state is not None:
            return self._robot_state
        bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "robot")
        x, y = (float(v) for v in self.data.xpos[bid][:2])
        m = self.data.xmat[bid]
        yaw = math.atan2(float(m[3]), float(m[0]))
        return RobotState(x=x, y=y, yaw=yaw)

    def _label(self, obj: SceneObject) -> Optional[str]:
        if self.known_classes is None:
            return obj.name
        if obj.coco_class in self.known_classes:
            return obj.coco_class
        return None  # blind to this object, like a pretrained model

    def _body_geoms(self, body_name: str) -> list[int]:
        bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        if bid < 0:
            return []
        gids = []
        for gid in range(self.model.ngeom):
            if int(self.model.geom_bodyid[gid]) != bid:
                continue
            if int(self.model.geom_group[gid]) == _COLLISION_GROUP:
                continue  # skip hidden collision geoms
            gids.append(gid)
        return gids

    def _project_body(self, body_name: str, state: RobotState) -> Optional[tuple[int, int, int, int]]:
        gids = self._body_geoms(body_name)
        if not gids:
            return None
        us, vs, any_in_front = [], [], False
        for gid in gids:
            center = self.model.geom_aabb[gid, :3]
            half = self.model.geom_aabb[gid, 3:]
            gpos = self.data.geom_xpos[gid]
            gmat = self.data.geom_xmat[gid].reshape(3, 3)
            for sx, sy, sz in itertools.product((-1, 1), repeat=3):
                local = center + np.array([sx, sy, sz], dtype=float) * half
                world = gpos + gmat.dot(local)
                pr = project_world_to_pixel(
                    world, state, self.fovy_deg, self.width, self.height, self.cam_local_offset
                )
                if pr is None:
                    continue
                any_in_front = True
                us.append(pr[0])
                vs.append(pr[1])
        if not any_in_front or not us:
            return None

        raw_x1, raw_x2 = min(us), max(us)
        raw_y1, raw_y2 = min(vs), max(vs)
        # Fully off to one side of the frame -> not visible.
        if raw_x2 < 0 or raw_x1 > self.width or raw_y2 < 0 or raw_y1 > self.height:
            return None
        bbox = clip_bbox(raw_x1, raw_y1, raw_x2, raw_y2, self.width, self.height)
        return bbox

    def detect(self, image: np.ndarray) -> List[Detection]:
        state = self._robot_pose()
        detections: List[Detection] = []
        for obj in self.objects:
            label = self._label(obj)
            if label is None:
                continue
            bbox = self._project_body(obj.body_name, state)
            if bbox is None:
                continue
            det = Detection(class_name=label, confidence=1.0, bbox_xyxy=bbox)
            if det.area >= self.min_bbox_area_px:
                detections.append(det)
        return detections
