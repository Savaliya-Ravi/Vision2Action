# Octo inference test

Octo is installed separately from the MuJoCo project because its older JAX and
TensorFlow versions conflict with the main environment.

Run the test from the project directory:

```bash
.venv-octo/bin/python scripts/test_octo_inference.py
```

The first line lists the device used by JAX. It should contain `GpuDevice` when
the NVIDIA driver is available. The script then prints the raw action array.

The environment uses JAX 0.5.3 because the original Octo JAX 0.4.20 wheel
cannot compile code for RTX 50-series GPUs.

To force a CPU run for troubleshooting:

```bash
JAX_PLATFORMS=cpu .venv-octo/bin/python scripts/test_octo_inference.py
```

The downloaded checkpoint and the isolated environment are ignored by Git.

V3 connects unmodified Octo to the Unitree G1 in a separate zero-shot
fridge-door trial. V4 adds a demonstration-trained G1 action head on Octo's
RGB and language features. V4.1 adds a learned open/stop command choice. The
V4.2 uses Octo image features to recognize task completion without reading the
door angle for runtime control. The floating base remains fixed in all versions.

Run the trained V4 policy in the desktop viewer:

```bash
env -u MUJOCO_GL .venv-octo/bin/python -m vision2action.vla.v4 interactive
```

Then type `open the fridge door` or `leave the fridge door closed` in the same
terminal. For a repeatable
headless check:

```bash
MUJOCO_GL=egl XLA_PYTHON_CLIENT_PREALLOCATE=false \
  .venv-octo/bin/python -m vision2action.vla.v4 eval --decisions 40
```

The Octo Small base checkpoint remains local under `checkpoints/`. The small
V4 action head, V4.1 intent head, V4.2 completion head, and reference
demonstration dataset are included in the project.
