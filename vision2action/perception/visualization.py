"""Visual debugging overlay for the perception/navigation pipeline.

Draws, on a copy of the camera RGB frame:

* every detection as a thin box with ``class conf`` label,
* the *selected* target as a thick highlighted box tagged ``TARGET``,
* a small HUD with the command, selection status, perceived distance, and the
  estimated 3D position.

Uses OpenCV when available (nice text); degrades to a numpy box-only drawer if
not, so the pipeline never hard-depends on cv2 for a headless run.
"""

from __future__ import annotations

from typing import Iterable, Optional

import numpy as np

from vision2action.perception.detector import Detection
from vision2action.perception.selection import Selection

_OTHER = (170, 170, 170)
_TARGET = (255, 45, 45)
_HUD = (60, 220, 90)
_HUD_BG = (18, 18, 18)


def _draw_box(img, bbox, color, thickness, label=None):
    import cv2

    x1, y1, x2, y2 = (int(v) for v in bbox)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
    if label:
        cv2.putText(img, label, (x1, max(10, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)


def _numpy_box(img, bbox, color):
    h, w = img.shape[:2]
    x1, y1, x2, y2 = (int(v) for v in bbox)
    x1, x2 = max(0, x1), min(w - 1, x2)
    y1, y2 = max(0, y1), min(h - 1, y2)
    c = np.array(color, dtype=np.uint8)
    img[y1, x1:x2 + 1] = c
    img[y2, x1:x2 + 1] = c
    img[y1:y2 + 1, x1] = c
    img[y1:y2 + 1, x2] = c


def draw_overlay(
    rgb: np.ndarray,
    detections: Iterable[Detection],
    selection: Optional[Selection] = None,
    *,
    distance_m: Optional[float] = None,
    world_xy: Optional[np.ndarray] = None,
    phase: Optional[str] = None,
) -> np.ndarray:
    """Return an annotated copy of ``rgb`` (never mutates the input)."""
    out = np.ascontiguousarray(rgb.copy())
    selected = selection.detection if selection else None

    try:
        import cv2  # noqa: F401

        have_cv2 = True
    except ImportError:
        have_cv2 = False

    for det in detections:
        if selected is not None and det.bbox_xyxy == selected.bbox_xyxy:
            continue
        label = f"{det.class_name} {det.confidence:.2f}"
        if have_cv2:
            _draw_box(out, det.bbox_xyxy, _OTHER, 1, label)
        else:
            _numpy_box(out, det.bbox_xyxy, _OTHER)

    if selected is not None:
        tag = "TARGET"
        if selection and selection.query.label:
            tag = f"TARGET: {selection.query.label}"
        conf = f"{selected.confidence:.2f}"
        if have_cv2:
            _draw_box(out, selected.bbox_xyxy, _TARGET, 2, f"{selected.class_name} {conf}")
            x1, y1, _, _ = selected.bbox_xyxy
            import cv2

            cv2.putText(out, tag, (int(x1), min(out.shape[0] - 4, int(y1) + 14)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, _TARGET, 1, cv2.LINE_AA)
        else:
            _numpy_box(out, selected.bbox_xyxy, _TARGET)

    _draw_hud(out, selection, distance_m, world_xy, phase, have_cv2)
    return out


def _draw_hud(out, selection, distance_m, world_xy, phase, have_cv2):
    if not have_cv2:
        return
    import cv2

    lines = []
    if selection is not None:
        lines.append(f"cmd: {selection.query.label}")
        lines.append(f"status: {selection.status.value}")
    if phase is not None:
        lines.append(f"phase: {phase}")
    if distance_m is not None and np.isfinite(distance_m):
        lines.append(f"dist: {distance_m:.2f} m")
    if world_xy is not None:
        lines.append(f"est xy: ({world_xy[0]:.2f}, {world_xy[1]:.2f})")
    if not lines:
        return

    pad = 4
    line_h = 15
    box_h = pad * 2 + line_h * len(lines)
    box_w = 8 + max(len(s) for s in lines) * 7
    overlay = out.copy()
    cv2.rectangle(overlay, (0, 0), (min(out.shape[1], box_w), box_h), _HUD_BG, -1)
    cv2.addWeighted(overlay, 0.5, out, 0.5, 0, out)
    for i, s in enumerate(lines):
        cv2.putText(out, s, (5, pad + line_h * (i + 1) - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, _HUD, 1, cv2.LINE_AA)
