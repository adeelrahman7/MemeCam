"""Plain data passed between modules. Tracking code returns these; it never draws."""

from __future__ import annotations

from dataclasses import dataclass, field
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


@dataclass(slots=True)
class FrameStats:
    fps: float = 0.0
    inference_ms: float = 0.0


@dataclass(slots=True)
class ProcessedFrame:
    """A fully rendered BGR frame plus the data used to render it."""

    image_bgr: np.ndarray
    gestures: GestureFrame = field(default_factory=GestureFrame)
    stats: FrameStats = field(default_factory=FrameStats)
    fired_reaction: str | None = None
