from __future__ import annotations

from typing import Optional

import numpy as np


class WorldModel:
    def __init__(self) -> None:
        self._objects: dict[str, np.ndarray] = {}
        self.age: dict[str, int] = {}

    def update(self, label: str, world_xy: np.ndarray) -> None:
        self._objects[label] = np.asarray(world_xy, dtype=float)
        self.age[label] = 0

    def update_all(self, detections: dict[str, np.ndarray]) -> None:
        for label, world_xy in detections.items():
            self.update(label, world_xy)

    def get(self, label: str) -> Optional[np.ndarray]:
        return self._objects.get(label)

    def get_all(self) -> dict[str, np.ndarray]:
        return {label: value.copy() for label, value in self._objects.items()}

    def clear(self, label: Optional[str] = None) -> None:
        if label is None:
            self._objects.clear()
            self.age.clear()
            return
        self._objects.pop(label, None)
        self.age.pop(label, None)

    def tick(self) -> None:
        for label in list(self.age.keys()):
            self.age[label] += 1

    def get_fresh(self, label: str, max_age: int = 300) -> Optional[np.ndarray]:
        world_xy = self._objects.get(label)
        if world_xy is None:
            return None
        return world_xy if self.age.get(label, max_age + 1) < max_age else None
