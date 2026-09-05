import numpy as np

from vision2action.config import CameraConfig
from vision2action.control.controller import RobotState, distance_to_world_target, move_forward
from vision2action.perception.depth_utils import camera_xyz_to_world
from vision2action.perception.detector import Detection
from vision2action.perception.localization import localize


def test_forward_step_matches_robot_yaw():
    state = RobotState(x=1.0, y=-2.0, yaw=0.7)
    before = np.array([state.x, state.y])
    step = 0.072

    move_forward(state, forward_step=step)

    displacement = np.array([state.x, state.y]) - before
    expected = step * np.array([np.cos(state.yaw), np.sin(state.yaw)])
    assert np.allclose(displacement, expected)
    assert float(np.dot(expected, displacement)) > 0.0


def test_perceived_distance_uses_robot_state_and_world_target():
    state = RobotState(x=0.0, y=0.0, yaw=0.0)
    assert np.isclose(distance_to_world_target(state, np.array([2.0, 0.0])), 2.0)
    move_forward(state, forward_step=0.072)
    assert distance_to_world_target(state, np.array([2.0, 0.0])) < 2.0


def test_camera_projection_uses_robot_cam_pitch():
    state = RobotState(x=0.0, y=0.0, yaw=0.0)
    world_xy = camera_xyz_to_world(np.array([0.0, 0.0, 1.0]), state)
    assert np.allclose(world_xy, np.array([0.08 + 0.966, 0.0]), atol=1e-3)


def test_invalid_detection_cannot_produce_valid_observation():
    depth = np.ones((240, 320), dtype=float)
    invalid = Detection("bottle", float("nan"), (320, 10, 330, 100))
    observation = localize(
        invalid,
        depth,
        RobotState(x=0.0, y=0.0, yaw=0.0),
        CameraConfig(),
    )
    assert not observation.ok
    assert observation.world_xy is None
    assert observation.depth_m is None
