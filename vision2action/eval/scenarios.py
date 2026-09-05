"""Evaluation scenarios: the (instruction, seed) matrix to score.

The instructions cover the spec's required test cases -- a large appliance
(fridge), a counter fixture (sink), a COCO movable (bottle, banana), and the
colour-attribute / no-COCO-class case (red can) -- and each runs across several
seeds so success is measured over different reproducible layouts, not one lucky
placement.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Scenario:
    instruction: str
    seed: int


# Required instructions (see spec test-case list).
DEFAULT_INSTRUCTIONS: tuple[str, ...] = (
    "go to the fridge",
    "go to the sink",
    "go to the bottle",
    "go to the red can",
    "go to the banana",
)

# A handful of reproducible layouts.
DEFAULT_SEEDS: tuple[int, ...] = (0, 1, 7, 42)


def default_scenarios(
    instructions: tuple[str, ...] = DEFAULT_INSTRUCTIONS,
    seeds: tuple[int, ...] = DEFAULT_SEEDS,
) -> list[Scenario]:
    """Cross-product of instructions x seeds."""
    return [Scenario(instruction=i, seed=s) for s in seeds for i in instructions]
