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

V3 connects Octo to the Unitree G1 in a separate fridge-door trial. Its seven
action values are converted to bounded right-hand motion and finger targets;
the floating base is held at the fixed starting pose. See README.md for the
command and the limits of this zero-shot trial.
