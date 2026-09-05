from pathlib import Path

import jax
import numpy as np
from octo.model.octo_model import OctoModel


PROJECT_DIR = Path(__file__).resolve().parents[1]
CHECKPOINT_DIR = PROJECT_DIR / "checkpoints" / "octo-small-1.5"


def main():
    devices = jax.devices()
    print("JAX devices:", devices)
    if any(device.platform == "gpu" for device in devices):
        print("Running on GPU")
    else:
        print("Running on CPU. JAX cannot see an NVIDIA GPU.")
    print("Loading:", CHECKPOINT_DIR)

    model = OctoModel.load_pretrained(str(CHECKPOINT_DIR))
    image_example = model.example_batch["observation"]["image_primary"]
    height, width = image_example.shape[-3:-1]

    random_image = np.random.default_rng(0).integers(
        0,
        256,
        size=(1, 1, height, width, 3),
        dtype=np.uint8,
    )
    observation = {
        "image_primary": random_image,
        "timestep_pad_mask": np.ones((1, 1), dtype=bool),
    }
    task = model.create_tasks(texts=["pick up the can"])
    actions = model.sample_actions(
        observation,
        task,
        rng=jax.random.PRNGKey(0),
    )
    actions = np.asarray(jax.device_get(actions))

    print("Action shape:", actions.shape)
    print("Raw actions:")
    print(actions)


if __name__ == "__main__":
    main()
