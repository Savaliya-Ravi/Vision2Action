"""Frozen Octo encoder plus a demonstration-trained G1 action head."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from vision2action.vla.octo_policy import OctoPolicy


FEATURE_DIM = 384
ACTION_DIM = 7
HEAD_FORMAT_VERSION = 2


class LearnedActionHead:
    """Small calibrated action head trained on Octo vision-language features."""

    def __init__(self, checkpoint: Path):
        checkpoint = Path(checkpoint)
        if not checkpoint.is_file():
            raise FileNotFoundError(
                f"V4 action head missing: {checkpoint}. Run the V4 train or bootstrap command first."
            )
        with np.load(checkpoint, allow_pickle=False) as saved:
            version = int(saved["format_version"])
            if version != HEAD_FORMAT_VERSION:
                raise ValueError(f"Unsupported V4 action-head format {version}")
            self.weights = np.asarray(saved["weights"], dtype=np.float32)
            self.bias = np.asarray(saved["bias"], dtype=np.float32)
            self.feature_mean = np.asarray(saved["feature_mean"], dtype=np.float32)
            self.feature_std = np.asarray(saved["feature_std"], dtype=np.float32)
            self.action_prototypes = np.asarray(saved["action_prototypes"], dtype=np.float32)
            self.training_samples = int(saved["training_samples"])
            self.training_episodes = int(saved["training_episodes"])
            self.instruction = str(saved["instruction"].item())

        if self.weights.shape != (FEATURE_DIM, ACTION_DIM):
            raise ValueError(f"Expected V4 weights {(FEATURE_DIM, ACTION_DIM)}, got {self.weights.shape}")
        if self.bias.shape != (ACTION_DIM,):
            raise ValueError(f"Expected V4 bias {(ACTION_DIM,)}, got {self.bias.shape}")
        if self.feature_mean.shape != (FEATURE_DIM,) or self.feature_std.shape != (FEATURE_DIM,):
            raise ValueError("V4 feature normalization has the wrong shape")
        if (
            self.action_prototypes.ndim != 2
            or self.action_prototypes.shape[0] < 1
            or self.action_prototypes.shape[1] != ACTION_DIM
        ):
            raise ValueError("V4 action prototypes have the wrong shape")
        if not all(np.all(np.isfinite(x)) for x in (
            self.weights, self.bias, self.feature_mean, self.feature_std, self.action_prototypes
        )):
            raise ValueError("V4 action head contains non-finite values")
        if np.any(self.feature_std <= 0):
            raise ValueError("V4 feature standard deviations must be positive")

    def predict(self, feature: np.ndarray) -> np.ndarray:
        feature = np.asarray(feature, dtype=np.float32)
        if feature.shape != (FEATURE_DIM,) or not np.all(np.isfinite(feature)):
            raise ValueError(f"Expected one finite {FEATURE_DIM}-value Octo feature")
        normalized = (feature - self.feature_mean) / self.feature_std
        continuous = normalized @ self.weights + self.bias
        if continuous.shape != (ACTION_DIM,) or not np.all(np.isfinite(continuous)):
            raise ValueError("V4 action head returned an invalid action")
        # The demonstrations define a small calibrated action vocabulary. Octo's
        # feature selects the nearest learned prototype, avoiding unsafe drift
        # between contact-sensitive G1 actions.
        distances = np.linalg.norm(self.action_prototypes - continuous[None], axis=1)
        return self.action_prototypes[int(np.argmin(distances))].astype(float)


class V4Policy:
    """VLA policy whose RGB/text features and learned head produce every action."""

    def __init__(self, octo_checkpoint: Path, action_head: Path, instruction: str):
        self.encoder = OctoPolicy(octo_checkpoint, instruction)
        self.head = LearnedActionHead(action_head)
        self.devices = self.encoder.devices

    def set_instruction(self, instruction: str) -> None:
        self.encoder.set_instruction(instruction)

    def predict(self, primary: np.ndarray, wrist: np.ndarray) -> np.ndarray:
        return self.head.predict(self.encoder.encode(primary, wrist))
