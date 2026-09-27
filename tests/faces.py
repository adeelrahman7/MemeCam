"""Synthetic face-mesh landmarks for tests (no camera or model needed).

Key points are laid out in 'eye-distance units' around the midpoint between the eyes
(outer eye corners are 1 unit apart). The rest of the 478 points sit on an oval.
"""

from __future__ import annotations

import math

import numpy as np

from memecam.core.types import FaceDetection
from memecam.face import anchors as fa

_KEY = {
    fa.RIGHT_EYE_OUTER: (-0.5, 0.0),
    fa.RIGHT_EYE_INNER: (-0.2, 0.0),
    fa.LEFT_EYE_INNER: (0.2, 0.0),
    fa.LEFT_EYE_OUTER: (0.5, 0.0),
    fa.FOREHEAD_TOP: (0.0, -1.0),
    fa.NOSE_TIP: (0.0, 0.55),
    fa.UPPER_LIP: (0.0, 0.95),
    fa.LOWER_LIP: (0.0, 1.05),
    fa.CHIN: (0.0, 1.6),
}


def canonical() -> np.ndarray:
    t = np.linspace(0, 2 * math.pi, 478, endpoint=False)
    pts = np.stack([0.8 * np.cos(t), 0.3 + 1.3 * np.sin(t)], axis=1)
    for idx, xy in _KEY.items():
        pts[idx] = xy
    return pts


def face_px(
    center: tuple[float, float] = (640, 300),
    eye_dist: float = 100.0,
    angle_deg: float = 0.0,
    mirrored: bool = False,
) -> np.ndarray:
    """Pixel landmarks (478, 2). ``mirrored`` swaps which eye is on which side."""
    pts = canonical()
    if mirrored:
        pts[:, 0] *= -1
    t = math.radians(angle_deg)
    rot = np.array([[math.cos(t), -math.sin(t)], [math.sin(t), math.cos(t)]])
    return pts @ rot.T * eye_dist + np.array(center)


def face_detection(frame_size=(1280, 720), blendshapes=None, **kwargs) -> FaceDetection:
    w, h = frame_size
    px = face_px(**kwargs)
    norm = np.column_stack([px[:, 0] / w, px[:, 1] / h, np.zeros(len(px))])
    return FaceDetection(norm.astype(np.float32), blendshapes)


# --- expressions (blendshape scores) ---------------------------------------------------
RESTING = {"eyeBlinkLeft": 0.05, "eyeBlinkRight": 0.05, "mouthClose": 0.05}
EXPRESSIONS: dict[str, dict[str, float]] = {
    "neutral": {},
    "shocked": {"jawOpen": 0.8, "mouthFunnel": 0.3, "browInnerUp": 0.6, "eyeWideLeft": 0.5,
                "eyeWideRight": 0.5},
    "smile": {"mouthSmileLeft": 0.85, "mouthSmileRight": 0.85, "cheekSquintLeft": 0.4,
              "cheekSquintRight": 0.4},
    "kiss": {"mouthPucker": 0.9, "mouthFunnel": 0.4},
    "wink": {"eyeBlinkLeft": 0.95, "eyeSquintLeft": 0.5, "mouthSmileLeft": 0.3},
    "slight_smile": {"mouthSmileLeft": 0.2, "mouthSmileRight": 0.2},
}  # fmt: skip


def expression(name: str, *, jitter: float = 0.0, seed: int = 0, **extra: float) -> np.ndarray:
    """52 blendshape scores for a named expression (plus optional overrides / noise)."""
    from memecam.gestures.expressions import BLENDSHAPE_NAMES

    scores = np.zeros(len(BLENDSHAPE_NAMES))
    for key, value in {**RESTING, **EXPRESSIONS[name], **extra}.items():
        scores[BLENDSHAPE_NAMES.index(key)] = value
    if jitter:
        scores += np.random.default_rng(seed).normal(0, jitter, scores.shape)
    return np.clip(scores, 0.0, 1.0)
