"""Target selection: choose ONE detection to navigate to, from open detections.

Detection is open (a detector returns everything it sees); selection is where
the command's target and colour attribute are applied.  This keeps detectors
interchangeable and puts all the "which one do I want" logic in one place.

Flow for ``select_target(detections, query, image)``:

1. Keep detections whose class matches ``query.coco_class``.
2. If a colour was requested and there is more than one candidate, keep those
   whose crop's dominant colour matches (the "red can" / "blue bottle" case).
3. Return the best remaining candidate (highest confidence, ties broken by area).

If the category has no COCO class (``query.coco_class is None``) and a colour was
given, fall back to a colour blob so the demo still works. The chosen
:class:`~vision2action.perception.detector.Detection` plus a status code is
returned so callers/visualisation can explain *why* nothing was picked.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import List, Optional

import numpy as np

from vision2action.perception.color import crop_bbox, estimate_dominant_color, find_color_blob, matches_color
from vision2action.perception.detector import Detection
from vision2action.targets import TargetQuery


class SelectStatus(str, Enum):
    OK = "ok"  # class match selected
    OK_COLOR_FALLBACK = "ok_color_fallback"  # no COCO class; colour blob used
    NO_DETECTIONS = "no_detections"  # detector returned nothing
    NO_CLASS_MATCH = "no_class_match"  # class supported but not seen this frame
    NO_COLOR_MATCH = "no_color_match"  # class seen but none of the right colour
    UNSUPPORTED_NO_FALLBACK = "unsupported_no_fallback"  # no class + no usable colour


@dataclass(frozen=True)
class Selection:
    detection: Optional[Detection]
    status: SelectStatus
    query: TargetQuery
    num_candidates: int = 0  # class-matched candidates considered
    color_estimate: Optional[str] = None  # dominant colour of the chosen crop

    @property
    def found(self) -> bool:
        return self.detection is not None


def _class_eq(a: str, b: str) -> bool:
    return a.strip().lower() == b.strip().lower()


def _best(dets: List[Detection]) -> Detection:
    return max(dets, key=lambda d: (round(d.confidence, 3), d.area))


def select_target(
    detections: List[Detection],
    query: TargetQuery,
    image: np.ndarray,
    *,
    allow_color_fallback: bool = True,
    fallback_min_area_px: int = 15,
) -> Selection:
    """Pick the detection matching ``query``; see module docstring for the flow."""
    # The oracle uses stable scene names ("fridge", "mug", "can_red") so it
    # can keep following one object without depending on changing pixel colour.
    stable_name = f"{query.category}_{query.color}" if query.color else query.category
    stable = [d for d in detections if _class_eq(d.class_name, stable_name)]
    if stable:
        chosen = _best(stable)
        return Selection(chosen, SelectStatus.OK, query, len(stable), query.color)

    # --- category with no pretrained class: colour-blob fallback --------------
    if query.coco_class is None:
        candidates = [
            detection
            for detection in detections
            if _class_eq(detection.class_name, query.category)
        ]
        if candidates:
            if query.color is not None:
                colored = [
                    detection
                    for detection in candidates
                    if matches_color(crop_bbox(image, detection.bbox_xyxy), query.color)
                ]
                if colored:
                    chosen = _best(colored)
                    return Selection(
                        chosen,
                        SelectStatus.OK,
                        query,
                        len(candidates),
                        query.color,
                    )
            else:
                chosen = _best(candidates)
                return Selection(chosen, SelectStatus.OK, query, len(candidates))

        if allow_color_fallback and query.color is not None:
            # Exclude regions already explained by a recognised class, so the
            # fallback picks the coloured object the detector *cannot* name (the
            # can), not a red-capped bottle or a red mug that a detector sees as
            # "bottle"/"cup".
            exclude = [d.bbox_xyxy for d in detections]
            bbox = find_color_blob(
                image, query.color, min_area_px=fallback_min_area_px, exclude_boxes=exclude
            )
            if bbox is not None:
                det = Detection(class_name=query.category, confidence=0.5, bbox_xyxy=bbox)
                return Selection(det, SelectStatus.OK_COLOR_FALLBACK, query, 0, query.color)
            return Selection(None, SelectStatus.NO_COLOR_MATCH, query, 0)
        return Selection(None, SelectStatus.UNSUPPORTED_NO_FALLBACK, query, 0)

    # --- class-based selection ------------------------------------------------
    if not detections:
        return Selection(None, SelectStatus.NO_DETECTIONS, query, 0)

    candidates = [d for d in detections if _class_eq(d.class_name, query.coco_class)]
    if not candidates:
        return Selection(None, SelectStatus.NO_CLASS_MATCH, query, 0)

    # Colour disambiguation only matters when there is a choice to make.
    if query.color is not None and len(candidates) > 1:
        colored = [d for d in candidates if matches_color(crop_bbox(image, d.bbox_xyxy), query.color)]
        if not colored:
            return Selection(None, SelectStatus.NO_COLOR_MATCH, query, len(candidates))
        candidates = colored

    chosen = _best(candidates)
    est = estimate_dominant_color(crop_bbox(image, chosen.bbox_xyxy))
    return Selection(chosen, SelectStatus.OK, query, len(candidates), est)
