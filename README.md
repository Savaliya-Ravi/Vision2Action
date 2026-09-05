# Humanoid Vision2Action

A small MuJoCo project for testing language-directed humanoid navigation and
manipulation in a kitchen. The simulated robot is the Unitree G1 with two arms
and two articulated hands.

The project currently uses an oracle detector. It creates camera-space object
detections from MuJoCo scene data, then uses the normal RGB, depth, localization,
and navigation pipeline. This is useful for testing control logic, but it is not
a real vision model.

## What works

- A Unitree G1 humanoid with two arms, two hands, legs, head, and torso.
- Seeded object placement for repeatable tests.
- Text commands for kitchen objects.
- Camera search, horizontal centering, depth localization, and navigation.
- Recovery when a small target briefly leaves the camera view.
- Right-hand pickup and lifting of the red can from the table.
- Headless evaluation and unit tests.
- A separate Octo checkpoint-loading test for future VLA work.

## Current limitations

- Navigation moves the floating base directly. The G1 does not walk or balance.
- Only the red-can pickup action is connected to the interactive program.
- Pickup uses camera-estimated X/Y and a known table height.
- The grasped can is attached to the hand in code after the fingers close.
- The left hand is modeled but is not used by a task yet.
- The fridge door and complete can-to-fridge task are not autonomous.
- The oracle detector reads simulation state and must not be presented as real
  object detection.
- Octo is tested separately and does not control the humanoid.

## Requirements

- Ubuntu or another Linux desktop
- Python 3.10, 3.11, or 3.12
- An OpenGL-capable desktop session
- NVIDIA GPU and driver only if you want GPU Octo inference

MuJoCo physics and project control code run mainly on the CPU. The viewer and
camera rendering use OpenGL. Octo uses JAX and can run on a CUDA GPU.

## Installation

Clone the repository and enter it:

```bash
git clone https://github.com/Savaliya-Ravi/Vision2Action.git
cd Vision2Action
```

Create the main environment:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Install test dependencies:

```bash
python -m pip install -r requirements-dev.txt
```

## Run the simulation

```bash
source .venv/bin/activate
python -m vision2action.main --seed 1
```

Type commands in the same terminal:

```text
go to the sink
go to the microwave
go to the red can
pick up the red can
stop
quit
```

`--seed 1` keeps the object layout repeatable. Omit it for a different layout
on each run.

Useful options:

```bash
python -m vision2action.main --help
python -m vision2action.main --seed 1 --overlay
python -m vision2action.main --seed 1 --log DEBUG
```

The robot-camera window is disabled by default. Add `--overlay` when you need it.

## Run tests

```bash
PYTHONPATH=. PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest -q
```

Run repeatable navigation evaluation:

```bash
MUJOCO_GL=egl python -m vision2action.eval --seeds 1
```

## Optional Octo inference setup

Octo uses a separate environment because its JAX and TensorFlow dependencies
do not belong in the small MuJoCo environment. See [OCTO_SETUP.md](OCTO_SETUP.md).

The Octo test only verifies that a checkpoint loads and returns an action array:

```bash
.venv-octo/bin/python scripts/test_octo_inference.py
```

## Project layout

```text
vision2action/
  assets/robots/unitree_g1/  official G1 model and meshes
  control/                   floating-base navigation helpers
  env/                       kitchen scene and object placement
  eval/                      repeatable headless evaluation
  manipulation/              right-arm pickup controller
  navigation/                search and approach state machine
  perception/                detection, selection, depth, localization
  world/                     remembered perceived positions
  config.py                  camera and navigation settings
  main.py                    interactive program
scripts/
  test_octo_inference.py     optional Octo model check
tests/                       fast unit tests
```

## Future goals

1. Add stable G1 walking and balance control instead of moving its base directly.
2. Replace the oracle with a real RGB detector or VLA perception model.
3. Estimate full 3D grasp poses without using a known table height.
4. Use physical finger contacts instead of attaching the object in code.
5. Add left-hand and bimanual actions, including opening the fridge.
6. Map and normalize VLA actions for the G1 embodiment.
7. Add collision-aware whole-body motion planning and safety limits.

## Troubleshooting

`ModuleNotFoundError: vision2action`

Run commands from the repository root and activate `.venv`. For tests, keep
`PYTHONPATH=.` as shown above.

`Failed to open display` or the viewer cannot open

Run from a graphical desktop session. A remote or headless shell normally cannot
open the passive MuJoCo viewer.

Slow viewer

Keep the optional `--overlay` window disabled. The program renders RGB and depth
while a command is active; Octo is not loaded by `vision2action.main`.

CUDA or JAX errors

These apply only to the optional Octo environment. Confirm the driver with
`nvidia-smi`, then follow [OCTO_SETUP.md](OCTO_SETUP.md).

## Model license

The Unitree G1 MJCF and meshes come from Google DeepMind's MuJoCo Menagerie and
retain their BSD-3-Clause license in
`vision2action/assets/robots/unitree_g1/LICENSE`.
