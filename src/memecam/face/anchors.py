"""Where to put things on a face: anchor points, size and head tilt from face-mesh landmarks.

Pure numpy geometry; no drawing, no MediaPipe.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum

import numpy as np

# MediaPipe face-mesh landmark indices (subject's right/left).
RIGHT_EYE_OUTER, RIGHT_EYE_INNER = 33, 133
LEFT_EYE_OUTER, LEFT_EYE_INNER = 263, 362
FOREHEAD_TOP = 10
NOSE_TIP = 1
UPPER_LIP, LOWER_LIP = 13, 14
CHIN = 152
MIN_LANDMARKS = 468

KEY_POINTS = (
    RIGHT_EYE_OUTER, RIGHT_EYE_INNER, LEFT_EYE_OUTER, LEFT_EYE_INNER,
    FOREHEAD_TOP, NOSE_TIP, UPPER_LIP, LOWER_LIP, CHIN,
)  # fmt: skip


class Anchor(StrEnum):
    EYES = "eyes"  # midpoint between the eyes
    FOREHEAD = "forehead"  # top of the forehead (roughly the hairline)
    NOSE = "nose"  # nose tip
    MOUTH = "mouth"  # between the lips
    CHIN = "chin"
    FACE = "face"  # center of the face


@dataclass(frozen=True, slots=True)
class Placement:
    """Where an anchor is on screen for one face.

    - ``center``: anchor position in pixels.
    - ``scale``: distance between the outer eye corners, in pixels. Overlay sizes are
      multiples of this, so they grow and shrink with the face.
    - ``angle``: head roll in radians (0 = level; positive = clockwise on screen).
    """

    center: tuple[float, float]
    scale: float
    angle: float


def to_pixels(landmarks: np.ndarray, width: int, height: int) -> np.ndarray:
    """Normalized (N, 2+) landmarks → (N, 2) pixel coordinates."""
    return landmarks[:, :2] * np.array([width, height], dtype=np.float64)


def _anchor_point(pts: np.ndarray, anchor: Anchor) -> np.ndarray:
    match anchor:
        case Anchor.EYES:
            return pts[[RIGHT_EYE_OUTER, RIGHT_EYE_INNER, LEFT_EYE_OUTER, LEFT_EYE_INNER]].mean(0)
        case Anchor.FOREHEAD:
            return pts[FOREHEAD_TOP]
        case Anchor.NOSE:
            return pts[NOSE_TIP]
        case Anchor.MOUTH:
            return pts[[UPPER_LIP, LOWER_LIP]].mean(0)
        case Anchor.CHIN:
            return pts[CHIN]
        case Anchor.FACE:
            return pts[[FOREHEAD_TOP, CHIN]].mean(0)


def head_roll(pts: np.ndarray) -> float:
    """Angle of the eye line, folded into (-90°, 90°].

    Folding makes it independent of which eye is on which side of the image, which flips
    when the frame is mirrored; an overlay should never end up upside down.
    """
    dx, dy = pts[LEFT_EYE_OUTER] - pts[RIGHT_EYE_OUTER]
    angle = math.atan2(dy, dx)
    if angle > math.pi / 2:
        angle -= math.pi
    elif angle <= -math.pi / 2:
        angle += math.pi
    return angle


def place(pts_px: np.ndarray, anchor: Anchor) -> Placement | None:
    """Placement of ``anchor`` for one face given pixel landmarks. None if degenerate."""
    if len(pts_px) < MIN_LANDMARKS:
        return None
    scale = float(np.linalg.norm(pts_px[LEFT_EYE_OUTER] - pts_px[RIGHT_EYE_OUTER]))
    if scale < 1.0:
        return None
    cx, cy = _anchor_point(pts_px, anchor)
    return Placement((float(cx), float(cy)), scale, head_roll(pts_px))


class FaceSmoother:
    """Exponential moving average of each face's landmarks, to stop overlays jittering.

    Faces are matched frame-to-frame by order. If the number of faces changes, or a face
    jumps further than one eye-distance, it snaps instead of sliding across the screen.
    """

    def __init__(self, smoothing: float) -> None:
        if not 0.0 <= smoothing < 1.0:
            raise ValueError("smoothing must be in [0, 1)")
        self._alpha = smoothing
        self._prev: list[np.ndarray] = []

    def update(self, faces_px: Sequence[np.ndarray]) -> list[np.ndarray]:
        if len(faces_px) != len(self._prev) or self._alpha == 0.0:
            self._prev = [f.astype(np.float64) for f in faces_px]
            return list(self._prev)
        out: list[np.ndarray] = []
        for prev, cur in zip(self._prev, faces_px, strict=True):
            eye_dist = float(np.linalg.norm(cur[LEFT_EYE_OUTER] - cur[RIGHT_EYE_OUTER]))
            jump = float(np.linalg.norm(cur[NOSE_TIP] - prev[NOSE_TIP]))
            if prev.shape != cur.shape or jump > max(eye_dist, 1.0):
                out.append(cur.astype(np.float64))
            else:
                out.append(self._alpha * prev + (1 - self._alpha) * cur)
        self._prev = out
        return list(out)

    def reset(self) -> None:
        self._prev = []
