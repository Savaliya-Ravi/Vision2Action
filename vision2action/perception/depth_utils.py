from __future__ import annotations

import math

import numpy as np

from vision2action.world.world_model import WorldModel

# --- robot_cam optical basis (must match `robot_cam` xyaxes in scene.xml) -----
# The camera images right/down/forward.  In the robot frame the optical axes are:
#   image-right  ->  CAMERA_RIGHT
#   image-up     ->  CAMERA_UP        (v/"down" uses -CAMERA_UP)
#   forward/depth->  CAMERA_FORWARD = -cross(right, up)   (pitched slightly down)
# These three are orthonormal, so `project_world_to_pixel` is an exact inverse of
# `backproject_pixel_to_camera` + `camera_xyz_to_world` and shares these constants.
CAMERA_RIGHT = np.array([0.0, -1.0, 0.0], dtype=float)
CAMERA_UP = np.array([0.259, 0.0, 0.966], dtype=float)
CAMERA_FORWARD = -np.cross(CAMERA_RIGHT, CAMERA_UP)  # -> (0.940, 0, -0.342)
ROBOT_CAM_HEIGHT = 1.16


def _yaw_rot(yaw: float) -> np.ndarray:
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=float)


def backproject_pixel_to_camera(
    u: int,
    v: int,
    depth_val: float,
    fovy_deg: float,
    width: int,
    height: int,
) -> np.ndarray:
    fy = (height / 2.0) / math.tan(math.radians(fovy_deg / 2.0))
    fx = fy
    cx = width / 2.0
    cy = height / 2.0

    z = float(depth_val)
    x = (float(u) - cx) * z / fx
    y = (float(v) - cy) * z / fy
    return np.array([x, y, z], dtype=float)


def camera_xyz_to_world(
    cam_xyz: np.ndarray,
    robot_state,
    cam_local_offset: tuple[float, float, float] = (0.08, 0.0, 0.0),
) -> np.ndarray:
    """Map ``backproject_pixel_to_camera`` output into the robot/world frame.

    The camera basis matches the ``robot_cam`` ``xyaxes`` in ``scene.xml``.
    Image coordinates are right/down/forward, so they are not directly robot
    X/Y/Z coordinates.
    """
    c = math.cos(robot_state.yaw)
    s = math.sin(robot_state.yaw)
    rot = np.array(
        [
            [c, -s, 0.0],
            [s, c, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )

    # The XML camera is high and only slightly pitched down. Its wide view sees
    # both floor objects and appliances sitting on the counter.
    # MuJoCo looks along camera -Z, so the optical axis is pitched downward.
    image_right, image_down, forward = np.asarray(cam_xyz, dtype=float)
    camera_right = CAMERA_RIGHT
    camera_up = CAMERA_UP
    camera_forward = CAMERA_FORWARD
    robot_camera_xyz = (
        image_right * camera_right
        - image_down * camera_up
        + forward * camera_forward
    )

    robot_world = np.array([robot_state.x, robot_state.y, ROBOT_CAM_HEIGHT], dtype=float)
    cam_offset = np.array(cam_local_offset, dtype=float)
    world_xyz = robot_world + rot.dot(cam_offset + robot_camera_xyz)
    return world_xyz[:2]


def project_world_to_pixel(
    world_xyz,
    robot_state,
    fovy_deg: float,
    width: int,
    height: int,
    cam_local_offset: tuple[float, float, float] = (0.08, 0.0, 0.0),
):
    """Project a 3D world point to ``(u, v, depth)`` for ``robot_cam``.

    Exact inverse of :func:`backproject_pixel_to_camera` composed with
    :func:`camera_xyz_to_world`, sharing the same optical basis.  Returns
    ``None`` when the point is behind the camera (``depth <= 0``).  ``depth`` is
    distance along the optical axis, matching the value the depth buffer stores
    and that back-projection consumes.
    """
    p = np.asarray(world_xyz, dtype=float)
    if p.shape == (2,):
        p = np.array([p[0], p[1], 0.0], dtype=float)

    rot = _yaw_rot(robot_state.yaw)
    robot_world = np.array([robot_state.x, robot_state.y, ROBOT_CAM_HEIGHT], dtype=float)
    cam_world = robot_world + rot.dot(np.asarray(cam_local_offset, dtype=float))

    # Vector camera -> point, expressed in the (yaw-rotated) robot frame.
    rel_robot = rot.T.dot(p - cam_world)

    depth = float(rel_robot.dot(CAMERA_FORWARD))
    if depth <= 1e-6:
        return None
    right = float(rel_robot.dot(CAMERA_RIGHT))
    down = -float(rel_robot.dot(CAMERA_UP))

    f = (height / 2.0) / math.tan(math.radians(fovy_deg / 2.0))
    u = width / 2.0 + right * f / depth
    v = height / 2.0 + down * f / depth
    return float(u), float(v), depth


__all__ = [
    "WorldModel",
    "backproject_pixel_to_camera",
    "camera_xyz_to_world",
    "project_world_to_pixel",
]
