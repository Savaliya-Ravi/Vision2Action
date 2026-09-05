"""Parse a natural-but-simple instruction into a structured target query.

No LLM, no VLA - just a small deterministic vocabulary, exactly as the spec
asks.  An instruction like "go to the red can" becomes::

    TargetQuery(category="can", coco_class=None, color="red", ...)

``coco_class`` is looked up from the scene registry, so this module never
hard-codes what a pretrained detector can or cannot see.  ``coco_class is None``
means "no COCO class for this category" - downstream selection then uses the
colour fallback (or, later, a custom model).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from vision2action.env.objects import ALL_OBJECTS
from vision2action.perception.color import SUPPORTED_COLORS

# category -> COCO class (or None), derived from the single-source registry.
CATEGORY_TO_COCO: dict[str, Optional[str]] = {o.category: o.coco_class for o in ALL_OBJECTS}

# Command word/phrase -> canonical category.  Longer phrases are matched first
# so "trash can" and "coffee machine" win over the bare "can"/"coffee".
CATEGORY_SYNONYMS: dict[str, str] = {
    "refrigerator": "fridge",
    "fridge": "fridge",
    "sink": "sink",
    "microwave": "microwave",
    "coffee machine": "coffee_machine",
    "coffee maker": "coffee_machine",
    "coffee": "coffee_machine",
    "oven": "oven",
    "stove": "oven",
    "dining table": "table",
    "table": "table",
    "chair": "chair",
    "trash bin": "trash_bin",
    "trash can": "trash_bin",
    "garbage bin": "trash_bin",
    "trash": "trash_bin",
    "bin": "trash_bin",
    "bottle": "bottle",
    "mug": "mug",
    "cup": "mug",
    "banana": "banana",
    "soda can": "can",
    "soda": "can",
    "can": "can",
    "tin": "can",
}


@dataclass(frozen=True)
class TargetQuery:
    category: str  # canonical scene category, e.g. "can", "bottle", "fridge"
    coco_class: Optional[str]  # COCO class to match in detections, or None
    color: Optional[str]  # requested colour attribute, or None
    text: str  # original instruction
    label: str  # human-friendly label, e.g. "red can"

    @property
    def supported_by_pretrained(self) -> bool:
        """Whether a pretrained COCO detector can label this category directly."""
        return self.coco_class is not None


def _normalize(text: str) -> str:
    return " ".join(text.lower().replace("-", " ").split())


def parse_target_instruction(instruction: str) -> Optional[TargetQuery]:
    """Extract a :class:`TargetQuery` from an instruction, or ``None``."""
    text = _normalize(instruction)
    if not text:
        return None

    color = next((c for c in SUPPORTED_COLORS if f" {c} " in f" {text} "), None)

    # Match the longest category phrase present (handles "trash can" vs "can").
    category: Optional[str] = None
    for phrase in sorted(CATEGORY_SYNONYMS, key=len, reverse=True):
        if f" {phrase} " in f" {text} ":
            category = CATEGORY_SYNONYMS[phrase]
            break
    if category is None:
        return None

    coco_class = CATEGORY_TO_COCO.get(category)
    label = f"{color} {category}" if color else category
    return TargetQuery(category=category, coco_class=coco_class, color=color, text=instruction, label=label)


SUPPORTED_CATEGORIES: tuple[str, ...] = tuple(sorted(set(CATEGORY_SYNONYMS.values())))
