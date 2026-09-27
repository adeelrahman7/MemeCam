"""Plain data passed between modules. Tracking code returns these; it never draws."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import numpy as np


class BuiltinGesture(StrEnum):
    """Gesture labels produced by MediaPipe's canned gesture_recognizer.task model."""

    CLOSED_FIST = "Closed_Fist"
    OPEN_PALM = "Open_Palm"
    POINTING_UP = "Pointing_Up"
    THUMB_DOWN = "Thumb_Down"
    THUMB_UP = "Thumb_Up"
    VICTORY = "Victory"
    I_LOVE_YOU = "ILoveYou"


class Handedness(StrEnum):
    LEFT = "Left"
    RIGHT = "Right"


@dataclass(frozen=True, slots=True)
class Landmark:
    """A landmark in normalized image coordinates (0..1), valid for any frame size."""

    x: float
    y: float
    z: float


@dataclass(frozen=True, slots=True)
class HandDetection:
    """One detected hand.

    ``handedness`` is the user's real hand, and ``landmarks`` are in the coordinates of the
    mirrored (selfie) frame the user sees. See FrameWorker for why these agree.
    """

    handedness: Handedness
    handedness_score: float
    gesture: str | None  # None when the model says "None" or nothing scored high enough
    gesture_score: float
    landmarks: tuple[Landmark, ...]


@dataclass(frozen=True, slots=True)
class GestureFrame:
    """Everything the gesture tracker found in one frame."""

    hands: tuple[HandDetection, ...] = ()


@dataclass(frozen=True, slots=True)
class FaceDetection:
    """One detected face: MediaPipe face-mesh landmarks as an (N, 3) array of normalized
    x, y (0..1 of the frame) and relative depth z. N is 478 (468 mesh points + 10 iris)."""

    landmarks: np.ndarray
    # 52 expression scores (0..1) in MediaPipe's blendshape order, or None if unavailable.
    blendshapes: np.ndarray | None = None


@dataclass(frozen=True, slots=True)
class FaceFrame:
    """Everything the face tracker found in one frame."""

    faces: tuple[FaceDetection, ...] = ()


@dataclass(slots=True)
class FrameStats:
    fps: float = 0.0
    inference_ms: float = 0.0
