"""Teach mode: countdown, then capture a held hand pose into a GestureTemplate."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum

import numpy as np

from memecam.core.types import HandDetection, Handedness
from memecam.gestures.templates import GestureTemplate, Pose, most_diverse, normalize


class RecordPhase(StrEnum):
    COUNTDOWN = "countdown"
    CAPTURING = "capturing"
    DONE = "done"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class RecorderStatus:
    name: str
    phase: RecordPhase
    seconds_left: float = 0.0  # countdown remaining
    progress: float = 0.0  # capture progress, 0..1
    hand_visible: bool = False
    message: str = ""
    template: GestureTemplate | None = None  # set only when phase is DONE


class GestureRecorder:
    """State machine driven one frame at a time; time is passed in explicitly.

    idle → start() → COUNTDOWN → CAPTURING → DONE | FAILED → idle

    During capture, frames without a hand are skipped (not counted), and the capture
    fails if ``capture_frames`` good frames aren't collected within ``timeout_seconds``.
    The collected poses are thinned to the ``max_samples`` most different ones.
    """

    def __init__(
        self,
        *,
        countdown_seconds: float,
        capture_frames: int,
        max_samples: int = 20,
        timeout_seconds: float = 10.0,
    ) -> None:
        if capture_frames < 1 or max_samples < 1:
            raise ValueError("capture_frames and max_samples must be >= 1")
        self._countdown = countdown_seconds
        self._capture_frames = capture_frames
        self._max_samples = max_samples
        self._timeout = timeout_seconds
        self._name: str | None = None
        self._phase = RecordPhase.COUNTDOWN
        self._phase_start = 0.0
        self._poses: list[Pose] = []
        self._hands: list[Handedness] = []

    @property
    def active(self) -> bool:
        return self._name is not None

    def start(self, name: str, now: float) -> None:
        self._name = name
        self._phase = RecordPhase.COUNTDOWN
        self._phase_start = now
        self._poses.clear()
        self._hands.clear()

    def cancel(self) -> None:
        self._name = None

    def update(
        self, hands: Sequence[HandDetection], aspect: float, now: float
    ) -> RecorderStatus | None:
        """Advance one frame. Returns None when idle; DONE/FAILED are reported exactly once."""
        name = self._name
        if name is None:
            return None
        hand = max(hands, key=lambda h: h.handedness_score, default=None)
        visible = hand is not None

        if self._phase is RecordPhase.COUNTDOWN:
            left = self._countdown - (now - self._phase_start)
            if left > 0:
                return RecorderStatus(name, RecordPhase.COUNTDOWN, seconds_left=left,
                                      hand_visible=visible)  # fmt: skip
            self._phase = RecordPhase.CAPTURING
            self._phase_start = now

        if hand is not None:
            pose = normalize(hand.landmarks, hand.handedness, aspect)
            if pose is not None:
                self._poses.append(pose)
                self._hands.append(hand.handedness)

        if len(self._poses) >= self._capture_frames:
            self._name = None
            return RecorderStatus(
                name, RecordPhase.DONE, progress=1.0, hand_visible=visible,
                message=f"Recorded '{name}'", template=self._build_template(name),
            )  # fmt: skip

        if now - self._phase_start > self._timeout:
            self._name = None
            return RecorderStatus(
                name, RecordPhase.FAILED, hand_visible=visible,
                message="Couldn't see your hand long enough. Keep it in frame and try again.",
            )  # fmt: skip

        return RecorderStatus(
            name, RecordPhase.CAPTURING, progress=len(self._poses) / self._capture_frames,
            hand_visible=visible, message="" if visible else "Show your hand to the camera",
        )  # fmt: skip

    def _build_template(self, name: str) -> GestureTemplate:
        idx = most_diverse(self._poses, self._max_samples)
        samples = np.stack([self._poses[i] for i in idx])
        recorded_with = Counter(self._hands).most_common(1)[0][0]
        return GestureTemplate(name=name, samples=samples, recorded_with=recorded_with)
