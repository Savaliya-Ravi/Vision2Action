from __future__ import annotations

import logging
import threading
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

from vision2action.control.controller import (
    apply_robot_state,
    camera_forward_distance,
    initialize_robot_state,
    move_forward,
    robot_to_target_distance,
    rotate_in_place,
)
from vision2action.navigation.navigation import NavParams, navigation_step
from vision2action.perception.color_detector import ColorDetector


def parse_instruction(instruction: str) -> str | None:
    text = instruction.lower().strip()
    for color in ["blue", "red", "green", "brown"]:
        if color in text:
            return color
    return None


def set_user_free_camera(viewer: mujoco.viewer.Handle) -> None:
    """Default interactive camera for mouse exploration."""
    viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    viewer.cam.lookat[:] = np.array([0.0, 0.0, 0.9])
    viewer.cam.distance = 8.5
    viewer.cam.azimuth = 135
    viewer.cam.elevation = -20


def _command_reader(command_state: dict, lock: threading.Lock, running: threading.Event) -> None:
    while running.is_set():
        try:
            text = input("\nType command (e.g., 'go to green'), or 'stop': ").strip()
        except EOFError:
            break
        if not text:
            continue
        with lock:
            command_state["pending"] = text


def run_demo() -> None:
    package_dir = Path(__file__).resolve().parent
    xml_path = package_dir / "env" / "scene.xml"

    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    robot_state = initialize_robot_state(model, data)
    apply_robot_state(model, data, robot_state)
    params = NavParams(search_yaw_step=0.001, centering_yaw_step=0.0008)
    detector = ColorDetector()

    image_h, image_w = 240, 320
    max_search_steps = 5000
    desired_clearance_m = 0.3
    center_contact_distance_m = 0.30
    stop_distance_m = desired_clearance_m + center_contact_distance_m
    stop_distance_epsilon_m = 0.005
    forward_step_m = 0.0009
    min_forward_step_m = 1e-6
    forward_step_safety_factor = 2.5
    forward_step_margin = 200
    drift_realign_threshold_m = 0.01
    full_frame_mask_threshold = 0.20
    centroid_hold_steps = 20
    forward_correction_interval = 6
    forward_center_tolerance_px = 14

    renderer = mujoco.Renderer(model, height=image_h, width=image_w)

    command_state = {"pending": None}
    lock = threading.Lock()
    running = threading.Event()
    running.set()
    reader_thread = threading.Thread(
        target=_command_reader,
        args=(command_state, lock, running),
        daemon=True,
    )

    with mujoco.viewer.launch_passive(model, data) as viewer:
        set_user_free_camera(viewer)
        logger = logging.getLogger(__name__)
        logger.info("Viewer started; use terminal to send commands (e.g., 'go to green').")
        logger.info("Robot will center the color then move toward it and stop near the door.")
        reader_thread.start()

        active_target = None
        active_phase = None
        command_step = 0
        forward_step_limit = 0
        best_forward_distance_m = float("inf")
        last_seen_centroid = None
        centroid_lost_steps = 0

        viewer.sync()

        try:
            while viewer.is_running():
                pending = None
                with lock:
                    if command_state["pending"] is not None:
                        pending = command_state["pending"]
                        command_state["pending"] = None

                if pending is not None:
                    if pending.lower().strip() == "stop":
                        active_target = None
                        active_phase = None
                        best_forward_distance_m = float("inf")
                        last_seen_centroid = None
                        centroid_lost_steps = 0
                        logger.info("Active command stopped.")
                    else:
                        parsed = parse_instruction(pending)
                        if parsed is None:
                            logger.info("Unsupported instruction. Use blue/red/green/brown.")
                        else:
                            active_target = parsed
                            active_phase = "search"
                            command_step = 0
                            best_forward_distance_m = float("inf")
                            last_seen_centroid = None
                            centroid_lost_steps = 0
                            logger.info(f"Target parsed: {active_target}. Searching and aligning.")

                if active_target is not None:
                    if active_phase == "search":
                        center_distance_now_m = robot_to_target_distance(
                            model, data, target_body_name=active_target
                        )
                        if center_distance_now_m <= stop_distance_m + stop_distance_epsilon_m:
                            logger.info(f"Already near {active_target}. Current dist={center_distance_now_m:.3f}m")
                            active_target = None
                            active_phase = None
                            best_forward_distance_m = float("inf")
                            last_seen_centroid = None
                            centroid_lost_steps = 0
                            continue

                        renderer.update_scene(data, camera="robot_cam")
                        rgb = renderer.render()
                        mask_uint8, centroid = detector.detect(rgb, target=active_target)
                        total_pixels = image_w * image_h
                        matched = int(mask_uint8.sum() // 255)
                        matched_frac = matched / float(max(total_pixels, 1))

                        if centroid is not None:
                            last_seen_centroid = centroid
                            centroid_lost_steps = 0

                        if centroid is None:
                            if matched_frac > full_frame_mask_threshold:
                                centroid = (image_w // 2, image_h // 2)
                                last_seen_centroid = centroid
                                centroid_lost_steps = 0
                                if command_step % 60 == 0:
                                    logger.debug(
                                        f"full-frame {active_target} mask (frac={matched_frac:.2f}), using image center"
                                    )
                            elif last_seen_centroid is not None and centroid_lost_steps < centroid_hold_steps:
                                centroid = last_seen_centroid
                                centroid_lost_steps += 1
                            else:
                                centroid_lost_steps += 1

                        target_centered = navigation_step(
                            state=robot_state,
                            detection=centroid,
                            image_width=image_w,
                            params=params,
                        )
                        apply_robot_state(model, data, robot_state)

                        if command_step % 60 == 0:
                            seen = "yes" if centroid is not None else "no"
                            logger.debug(f"search target={active_target} step={command_step:04d} seen={seen}")

                        if target_centered:
                            distance_m = camera_forward_distance(
                                model, data, robot_state, target_body_name=active_target
                            )
                            if distance_m <= stop_distance_m + stop_distance_epsilon_m:
                                logger.info(f"Already near {active_target}. Current dist={distance_m:.3f}m")
                                active_target = None
                                active_phase = None
                                best_forward_distance_m = float("inf")
                                last_seen_centroid = None
                                centroid_lost_steps = 0
                            else:
                                remaining_m = max(distance_m - (stop_distance_m + stop_distance_epsilon_m), 0.0)
                                estimated_steps = int(np.ceil(remaining_m / max(forward_step_m, 1e-9)))
                                forward_step_limit = int(estimated_steps * forward_step_safety_factor) + forward_step_margin
                                active_phase = "forward"
                                command_step = 0
                                best_forward_distance_m = distance_m
                                last_seen_centroid = None
                                centroid_lost_steps = 0
                                logger.info(
                                    f"Target {active_target} centered. Moving forward. "
                                    f"dist={distance_m:.3f}m step_limit={forward_step_limit}"
                                )
                        elif command_step > max_search_steps:
                            logger.info(f"Search timeout for target={active_target}.")
                            active_target = None
                            active_phase = None
                            best_forward_distance_m = float("inf")
                            last_seen_centroid = None
                            centroid_lost_steps = 0
                        else:
                            command_step += 1

                    elif active_phase == "forward":
                        distance_before_m = camera_forward_distance(
                            model, data, robot_state, target_body_name=active_target
                        )
                        if distance_before_m <= stop_distance_m + stop_distance_epsilon_m:
                            logger.info(f"Reached {active_target}. Stopped at {distance_before_m:.3f}m.")
                            active_target = None
                            active_phase = None
                            best_forward_distance_m = float("inf")
                            last_seen_centroid = None
                            centroid_lost_steps = 0
                        else:
                            step_m = min(
                                forward_step_m,
                                max(distance_before_m - (stop_distance_m + stop_distance_epsilon_m), 0.0),
                            )
                            if step_m <= min_forward_step_m:
                                logger.info(
                                    f"Reached {active_target}. Stopped at {distance_before_m:.3f}m (epsilon clamp)."
                                )
                                active_target = None
                                active_phase = None
                                best_forward_distance_m = float("inf")
                                last_seen_centroid = None
                                centroid_lost_steps = 0
                                continue
                            move_forward(robot_state, forward_step=step_m)
                            apply_robot_state(model, data, robot_state)

                            if command_step % forward_correction_interval == 0:
                                renderer.update_scene(data, camera="robot_cam")
                                rgb_corr = renderer.render()
                                _, corr_centroid = detector.detect(rgb_corr, target=active_target)
                                if corr_centroid is not None:
                                    cx, _ = corr_centroid
                                    center_x = image_w // 2
                                    error_px = cx - center_x
                                    if abs(error_px) > forward_center_tolerance_px:
                                        normalized = error_px / float(max(center_x, 1))
                                        yaw_delta = -float(np.clip(normalized, -1.0, 1.0)) * params.centering_yaw_step * 1.5
                                        rotate_in_place(robot_state, yaw_step=yaw_delta)
                                        apply_robot_state(model, data, robot_state)

                            distance_m = camera_forward_distance(
                                model, data, robot_state, target_body_name=active_target
                            )

                            if command_step % 60 == 0:
                                logger.debug(
                                    f"forward target={active_target} step={command_step:04d} dist={distance_m:.3f}m"
                                )

                            if distance_m <= stop_distance_m + stop_distance_epsilon_m:
                                logger.info(f"Reached {active_target}. Stopped at {distance_m:.3f}m.")
                                active_target = None
                                active_phase = None
                                best_forward_distance_m = float("inf")
                                last_seen_centroid = None
                                centroid_lost_steps = 0
                            elif distance_m > best_forward_distance_m + drift_realign_threshold_m:
                                logger.info(
                                    f"Drift detected for target={active_target}. "
                                    f"best={best_forward_distance_m:.3f}m now={distance_m:.3f}m -> re-aligning."
                                )
                                active_phase = "search"
                                command_step = 0
                                last_seen_centroid = None
                                centroid_lost_steps = 0
                            elif command_step > forward_step_limit:
                                logger.info(
                                    f"Forward timeout for target={active_target}. "
                                    f"Last dist={distance_m:.3f}m step_limit={forward_step_limit}"
                                )
                                active_target = None
                                active_phase = None
                                best_forward_distance_m = float("inf")
                                last_seen_centroid = None
                                centroid_lost_steps = 0
                            else:
                                best_forward_distance_m = min(best_forward_distance_m, distance_m)
                                command_step += 1

                mujoco.mj_step(model, data)
                viewer.sync()
        finally:
            running.clear()
            renderer.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    run_demo()
