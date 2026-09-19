"""Small, lazy Octo wrapper for the isolated .venv-octo environment."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np


class OctoPolicy:
    def __init__(self, checkpoint: Path, instruction: str, statistics_dataset: str = "nyu_door_opening_surprising_effectiveness"):
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        try:
            import jax
            from octo.model.octo_model import OctoModel
        except ImportError as exc:
            raise RuntimeError("Octo is unavailable. Run this command with .venv-octo/bin/python.") from exc

        if not checkpoint.is_dir():
            raise FileNotFoundError(f"Octo checkpoint missing: {checkpoint}")
        self.jax = jax
        self.devices = [str(device) for device in jax.devices()]
        self.model = OctoModel.load_pretrained(str(checkpoint))
        if statistics_dataset not in self.model.dataset_statistics:
            raise ValueError(f"No Octo action statistics for {statistics_dataset!r}")
        self.statistics = self.model.dataset_statistics[statistics_dataset]["action"]
        self.statistics_dataset = statistics_dataset
        self.set_instruction(instruction)

    def set_instruction(self, instruction: str) -> None:
        """Pass a new free-form instruction directly to Octo and reset image history."""
        instruction = instruction.strip()
        if not instruction:
            raise ValueError("VLA instruction cannot be empty")
        self.task = self.model.create_tasks(texts=[instruction])
        self.primary_history: list[np.ndarray] = []
        self.wrist_history: list[np.ndarray] = []
        self.decision = 0

    def _observation(self, primary: np.ndarray, wrist: np.ndarray) -> dict:
        if primary.shape != (256, 256, 3) or wrist.shape != (128, 128, 3):
            raise ValueError("Octo requires 256x256 primary and 128x128 wrist RGB images")
        if primary.dtype != np.uint8 or wrist.dtype != np.uint8:
            raise ValueError("Octo images must have uint8 RGB pixels")
        self.primary_history = (self.primary_history + [primary.copy()])[-2:]
        self.wrist_history = (self.wrist_history + [wrist.copy()])[-2:]
        # Keep the history length fixed so JAX compiles only one input shape.
        pad = len(self.primary_history) == 1
        primary_pair = [self.primary_history[0]] * (2 - len(self.primary_history)) + self.primary_history
        wrist_pair = [self.wrist_history[0]] * (2 - len(self.wrist_history)) + self.wrist_history
        timestep_mask = np.array([[not pad, True]], dtype=bool)
        return {
            "image_primary": np.stack(primary_pair)[None],
            "image_wrist": np.stack(wrist_pair)[None],
            "timestep_pad_mask": timestep_mask,
            "pad_mask_dict": {
                "image_primary": timestep_mask,
                "image_wrist": timestep_mask,
                "timestep": timestep_mask,
            },
            "timestep": np.array([[max(0, self.decision - 1), self.decision]], dtype=np.int32),
            "task_completed": np.zeros((1, 2, 4), dtype=bool),
        }

    def encode(self, primary: np.ndarray, wrist: np.ndarray) -> np.ndarray:
        """Return Octo's final action-readout token for an RGB/text observation."""
        observation = self._observation(primary, wrist)
        outputs = self.model.run_transformer(
            observation,
            self.task,
            observation["timestep_pad_mask"],
            train=False,
        )
        self.decision += 1
        tokens = np.asarray(self.jax.device_get(outputs["readout_action"].tokens))
        feature = tokens[0, -1].mean(axis=0)
        if feature.shape != (384,) or not np.all(np.isfinite(feature)):
            raise ValueError("Octo returned an invalid action-readout feature")
        return feature

    def predict(self, primary: np.ndarray, wrist: np.ndarray) -> np.ndarray:
        observation = self._observation(primary, wrist)
        actions = self.model.sample_actions(
            observation,
            self.task,
            unnormalization_statistics=self.statistics,
            rng=self.jax.random.PRNGKey(self.decision),
        )
        self.decision += 1
        action = np.asarray(self.jax.device_get(actions))[0, 0]
        if action.shape != (7,) or not np.all(np.isfinite(action)):
            raise ValueError("Octo returned an invalid action")
        return action
