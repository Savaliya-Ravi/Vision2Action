"""Frozen Octo encoder plus a demonstration-trained G1 action head."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from vision2action.vla.octo_policy import OctoPolicy


FEATURE_DIM = 384
ACTION_DIM = 7
HEAD_FORMAT_VERSION = 2
INTENT_FORMAT_VERSION = 1
DEFAULT_INTENT_HEAD = Path(__file__).resolve().parent / "checkpoints" / "v4_1_intent_head.npz"


class LearnedIntentHead:
    """Choose whether a command requests the trained fridge-opening skill."""

    def __init__(self, checkpoint: Path = DEFAULT_INTENT_HEAD):
        checkpoint = Path(checkpoint)
        if not checkpoint.is_file():
            raise FileNotFoundError(
                f"V4.1 intent head missing: {checkpoint}. Run the V4 train-intent command first."
            )
        with np.load(checkpoint, allow_pickle=False) as saved:
            if int(saved["format_version"]) != INTENT_FORMAT_VERSION:
                raise ValueError("Unsupported V4.1 intent-head format")
            self.weights = np.asarray(saved["weights"], dtype=np.float32)
            self.bias = float(saved["bias"])
            self.feature_mean = np.asarray(saved["feature_mean"], dtype=np.float32)
            self.feature_std = np.asarray(saved["feature_std"], dtype=np.float32)
            self.threshold = float(saved["threshold"])
        if any(array.shape != (FEATURE_DIM,) for array in (
            self.weights, self.feature_mean, self.feature_std
        )):
            raise ValueError("V4.1 intent head expects one weight per Octo feature")
        if not all(np.all(np.isfinite(array)) for array in (
            self.weights, self.feature_mean, self.feature_std
        )) or not np.isfinite([self.bias, self.threshold]).all():
            raise ValueError("V4.1 intent head contains non-finite values")
        if np.any(self.feature_std <= 0):
            raise ValueError("V4.1 feature standard deviations must be positive")

    def score(self, feature: np.ndarray) -> float:
        feature = np.asarray(feature, dtype=np.float32)
        if feature.shape != (FEATURE_DIM,) or not np.all(np.isfinite(feature)):
            raise ValueError(f"Expected one finite {FEATURE_DIM}-value Octo feature")
        normalized = (feature - self.feature_mean) / self.feature_std
        return float(normalized @ self.weights + self.bias)

    def requests_opening(self, feature: np.ndarray) -> bool:
        return self.score(feature) >= self.threshold


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
    """Octo features drive learned task selection and G1 actions."""

    def __init__(
        self,
        octo_checkpoint: Path,
        action_head: Path,
        instruction: str,
        intent_head: Path = DEFAULT_INTENT_HEAD,
    ):
        self.encoder = OctoPolicy(octo_checkpoint, instruction)
        self.head = LearnedActionHead(action_head)
        self.intent = LearnedIntentHead(intent_head)
        self.devices = self.encoder.devices
        self._opening: bool | None = None

    def set_instruction(self, instruction: str) -> None:
        self.encoder.set_instruction(instruction)
        self._opening = None

    def predict(self, primary: np.ndarray, wrist: np.ndarray) -> np.ndarray | None:
        feature = self.encoder.encode(primary, wrist)
        if self._opening is None:
            self._opening = self.intent.requests_opening(feature)
        if not self._opening:
            return None
        return self.head.predict(feature)
