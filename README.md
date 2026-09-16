# Humanoid Vision2Action

A small MuJoCo project for testing language-directed humanoid navigation and
manipulation in a kitchen. The simulated robot is the Unitree G1 with two arms
and two articulated hands.

The V2 interactive program uses an oracle detector. It creates camera-space
object detections from MuJoCo scene data, then uses the RGB, depth,
localization, and navigation pipeline. The V3 trial uses an Octo vision-language-
action model to command the G1 arm from RGB images and a text instruction.
The oracle and hardcoded object locations are unchanged.

## What works

- A Unitree G1 humanoid with two arms, two hands, legs, head, and torso.
- Seeded object placement for repeatable tests.
- Text commands for kitchen objects.
- Camera search, horizontal centering, depth localization, and navigation.
- Recovery when a small target briefly leaves the camera view.
- Right-hand pickup and lifting of the red can from the table.
- Headless evaluation and unit tests.
- A separate Octo checkpoint-loading test.
- A separate closed-loop Octo-to-G1 fridge-door trial with bounded arm and
  finger actions, two RGB cameras, and measured door-angle output.

## Current limitations

- Navigation moves the floating base directly. The G1 does not walk or balance.
- Only the red-can pickup action is connected to the interactive program.
- Pickup uses camera-estimated X/Y and a known table height.
- The grasped can is attached to the hand in code after the fingers close.
- The left hand is modeled but is not used by a task yet.
- The fridge door and complete can-to-fridge task are not autonomous.
- The oracle detector reads simulation state and must not be presented as real
  object detection.
- V3 starts the G1 at a fixed fridge pose with its hand near the handle. It
  does not navigate there.
- The pretrained Octo Small checkpoint has not opened the fridge zero-shot.
  In an 80-decision CPU trial the door moved about 0.74 degrees; success is
  defined as at least 30 degrees.
- The Octo action coordinate frame is an embodiment assumption until calibrated
  with G1 demonstrations. The current checkpoint was not trained on this G1
  kitchen scene.

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

## V3: VLA fridge-door trial

To open MuJoCo first and then type a command, run this from a **desktop terminal**:

```bash
env -u MUJOCO_GL .venv-octo/bin/python -m vision2action.vla --interactive
```

When the MuJoCo window appears, type `open the fridge door` in the same
terminal and press Enter. The instruction goes directly to Octo. `stop` stops
policy actions, `reset` restores the closed-door start, and `quit` exits.
The first instruction loads the checkpoint and can take a while. `--decisions`
sets the maximum number of model decisions per instruction. This is typed
language input; microphone speech is not wired up. Do **not** use
`MUJOCO_GL=egl` for the desktop viewer.

For an automatic, headless trial with the existing local checkpoint:

```bash
MUJOCO_GL=egl .venv-octo/bin/python -m vision2action.vla \
  --instruction "open the fridge door" --decisions 80 \
  --trace /tmp/vla_v3_trial.json
```

For a one-shot desktop window, omit `MUJOCO_GL=egl` and add `--viewer`. The command
starts the G1 beside the known fridge, with an open hand about 3 cm from the
upper handle. The model sees a fixed view of the fridge and a camera on the
right wrist, then supplies every 7D end-effector and gripper action. Numerical
inverse kinematics, joint limits, and base pinning translate those actions to
the G1; no target point or door angle is fed to the policy, and no task code
assigns the door joint. The door angle is read only for evaluation and early
stopping.

`--checkpoint` accepts another compatible local Octo checkpoint. The included
one is Octo Small; GPU inference is optional. This V3 run was tested on CPU in
the available environment because `nvidia-smi` could not access the NVIDIA
driver here. Check that `.venv-octo/bin/python -c "import jax; print(jax.devices())"`
shows a GPU on the laptop before expecting GPU speed. Octo's T5 tokenizer and
checkpoint must be cached locally; the V3 command loads them offline.

The trial writes raw model actions and door angles to JSON. A result with
`"opened": false` is an unsuccessful trial, even if the model issued actions.
The current Octo Small checkpoint does not reliably open the door: its
80-decision CPU baseline reached only 0.74 degrees. Interactive mode shows
the VLA attempt in MuJoCo; V4 demonstration training is needed for the
requested door-opening behavior.

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
vision2action/vla/           V3 Octo policy, G1 adapter, fridge trial
tests/                       fast unit tests
```

## Version roadmap

| Version | Scope | Completion check |
| --- | --- | --- |
| V3 (current) | Closed-loop Octo control of the G1 right arm and hand from RGB and text at a fixed fridge start. | Model actions reach G1 actuators; trial reports door angle and success honestly. |
| V4 | Collect or import G1 fridge-opening demonstrations, calibrate the action frame, and adapt a VLA on this scene. | Held-out fridge trials open the door at least 30 degrees through physical contact, without a scripted door or hand path. |
| V5 | Add learned G1 base/navigation actions and train the combined command “go to the fridge and open the door.” | From a remote start, held-out trials reach the fridge and open it without scripted navigation or manipulation. |

The existing hardcoded/oracle location path can remain as a separate V2
baseline. V4 needs demonstration data; the V3 zero-shot result alone is not
evidence of a learned fridge-opening policy.

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
