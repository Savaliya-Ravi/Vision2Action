# Changelog

## 3.0.0

- Connected the existing Octo Small checkpoint to the Unitree G1 right arm and hand in a separate, closed-loop VLA trial.
- Added a fixed near-fridge starting pose, fridge workspace and wrist policy cameras, bounded action conversion, and door-angle evaluation with JSON traces.
- Kept the V2 oracle detector, object locations, navigation, and scripted pickup path untouched; the V3 trial uses RGB and language directly, with no scripted door trajectory.
- Verified model inference and control in an 80-decision CPU trial. The door moved 0.74 degrees, so zero-shot door opening is **not** achieved.
- Added an interactive MuJoCo viewer that waits for terminal instructions and sends their text directly to Octo; `stop`, `reset`, and `quit` control the session.

## 2.1.0

- Replaced the Franka Panda with the Unitree G1 humanoid with hands.
- Moved the red can to a reachable table position.
- Added G1 right-arm pickup control.
- Kept camera navigation and oracle perception.
- Rewrote setup, usage, limitations, and future goals.

## 2.0.0

- Added camera-based red-can navigation and pickup.
- Added sink and microwave navigation.
- Improved target reacquisition and slower rotation.

## 1.0.0

- Added the first MuJoCo navigation prototype.
