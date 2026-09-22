# Changelog

## 4.1.0

- Trained a text-conditioned intent head on paired RGB observations so the same starting view can yield either the existing fridge-opening skill or a model-selected STOP.
- Added held-out phrase checks and MuJoCo evaluations: `open the refrigerator door` opened to 30.90 degrees, while `leave the fridge door closed` stopped before arm motion.
- Added `train-intent`, `eval --expect open|stop`, and regression coverage for the learned STOP path.
- Kept the V4 action head, fixed start, and perception path. This is command selection for a single skill, not general task understanding or learned navigation.

## 4.0.0

- Added a demonstration-adapted VLA policy that combines frozen Octo RGB and language features with a trained G1 action head.
- Added collection, training, headless evaluation, and interactive MuJoCo commands under `python -m vision2action.vla.v4`.
- Calibrated local end-effector actions for the G1 right arm and waist while keeping the floating base fixed at the known fridge pose.
- Updated the fridge door and handle contacts and added convex fingertip contact pads so the articulated hand can pull the door through MuJoCo physics.
- Bundled four successful demonstration episodes and their small trained action head. The learned policy opens the door to 30.90 degrees in 12 decisions in the reference evaluation.
- Added physical-contact, dataset, action-head, and trial regression tests. The existing oracle perception and V2 object locations remain unchanged.

## 3.1.0

- Kept the red can attached to the G1 hand during V2 navigation, including turns and approaches.
- Retracted the right arm after pickup and after placement.
- Added `put down the can` at a nearby clear table or counter spot and `put the can back` near its original table spot. Failed placements keep the can held.
- Added a MuJoCo carry and placement regression test and validated the seed 1 pickup and sink navigation route.
- Left V3 Octo arm control separate. Its seven-value action has no leg or base control; learned walking and reliable fridge opening remain V5 and V4 work, respectively.

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
