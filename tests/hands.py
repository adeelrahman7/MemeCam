"""Synthetic hand landmarks for tests (no camera or model needed).

Shapes are built in a canonical frame: a right hand, palm facing the camera, fingers up,
wrist at (0, 0), middle knuckle at (0, -1). Image y grows downward.
"""

from __future__ import annotations

import math

import numpy as np

from memecam.core.types import HandDetection, Handedness, Landmark

# (base x, base y) of each finger's knuckle (MCP): index, middle, ring, pinky.
_FINGER_BASES = [(-0.3, -1.0), (0.0, -1.0), (0.25, -0.95), (0.5, -0.85)]

POSES: dict[str, tuple[bool, bool, bool, bool, bool]] = {
    # thumb, index, middle, ring, pinky extended?
    "open": (True, True, True, True, True),
    "fist": (False, False, False, False, False),
    "peace": (False, True, True, False, False),
    "point": (False, True, False, False, False),
    "rock": (True, True, False, False, True),
}


def canonical(pose: str) -> np.ndarray:
    thumb, *fingers = POSES[pose]
    pts = [(0.0, 0.0)]
    if thumb:
        pts += [(-0.4, -0.3), (-0.7, -0.55), (-0.9, -0.8), (-1.05, -1.0)]
    else:
        pts += [(-0.4, -0.3), (-0.5, -0.6), (-0.3, -0.75), (-0.1, -0.8)]
    for (bx, by), extended in zip(_FINGER_BASES, fingers, strict=True):
        pts.append((bx, by))
        if extended:
            pts += [(bx, by - 0.4), (bx, by - 0.75), (bx, by - 1.05)]
        else:
            pts += [(bx, by - 0.3), (bx, by - 0.1), (bx, by + 0.1)]
    return np.array(pts)


def landmarks(
    pose: str | np.ndarray,
    *,
    center: tuple[float, float] = (0.5, 0.6),
    size: float = 0.15,
    angle_deg: float = 0.0,
    aspect: float = 16 / 9,
    left: bool = False,
    jitter: float = 0.0,
    seed: int = 0,
) -> tuple[Landmark, ...]:
    """Place a canonical pose into normalized image coords.

    ``size`` is the palm length as a fraction of frame height. ``left=True`` mirrors the
    shape, which is what a left hand looks like.
    """
    pts = canonical(pose) if isinstance(pose, str) else np.asarray(pose, dtype=float).copy()
    if left:
        pts[:, 0] *= -1
    t = math.radians(angle_deg)
    rot = np.array([[math.cos(t), -math.sin(t)], [math.sin(t), math.cos(t)]])
    pts = pts @ rot.T * size
    if jitter:
        pts += np.random.default_rng(seed).normal(0, jitter * size, pts.shape)
    # Height-units → normalized coords: x is divided by the aspect ratio.
    xs = center[0] + pts[:, 0] / aspect
    ys = center[1] + pts[:, 1]
    return tuple(Landmark(float(x), float(y), 0.0) for x, y in zip(xs, ys, strict=True))


def detection(
    pose: str,
    *,
    left: bool = False,
    gesture: str | None = None,
    score: float = 0.9,
    **kwargs: object,
) -> HandDetection:
    return HandDetection(
        handedness=Handedness.LEFT if left else Handedness.RIGHT,
        handedness_score=0.95,
        gesture=gesture,
        gesture_score=score if gesture else 0.0,
        landmarks=landmarks(pose, left=left, **kwargs),  # type: ignore[arg-type]
    )
