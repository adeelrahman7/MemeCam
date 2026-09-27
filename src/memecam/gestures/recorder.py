"""Teach mode: countdown, then capture a held hand pose or face expression into a template."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum

import numpy as np

from memecam.core.types import FaceDetection, HandDetection, Handedness
from memecam.gestures.expressions import (
    SUBTLE_EXPRESSION,
    expression_distances,
    expression_strength,
)
from memecam.gestures.templates import (
    GestureKind,
    GestureTemplate,
    hand_distances,
    most_diverse,
    normalize,
)


class RecordPhase(StrEnum):
    COUNTDOWN = "countdown"
    CAPTURING = "capturing"
    DONE = "done"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class RecorderStatus:
    name: str
    phase: RecordPhase
    kind: GestureKind = GestureKind.HAND
    seconds_left: float = 0.0  # countdown remaining
    progress: float = 0.0  # capture progress, 0..1
    visible: bool = False  # is the hand / face being recorded in view?
    message: str = ""
    template: GestureTemplate | None = None  # set only when phase is DONE
    warning: str = ""  # set on DONE when the recording looks unreliable


class GestureRecorder:
    """State machine driven one frame at a time; time is passed in explicitly.

    idle → start() → COUNTDOWN → CAPTURING → DONE | FAILED → idle

    Records either a hand pose (the most confident hand) or a face expression (the first
    face's blendshape scores). Frames where it isn't visible are skipped, and the capture
    fails if ``capture_frames`` good frames aren't collected within ``timeout_seconds``.
    The collected samples are thinned to the ``max_samples`` most different ones.
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
        self._kind = GestureKind.HAND
        self._phase = RecordPhase.COUNTDOWN
        self._phase_start = 0.0
        self._samples: list[np.ndarray] = []
        self._hands: list[Handedness] = []

    @property
    def active(self) -> bool:
        return self._name is not None

    @property
    def kind(self) -> GestureKind | None:
        """What's being recorded, or None when idle."""
        return self._kind if self._name is not None else None

    def start(self, name: str, now: float, kind: GestureKind = GestureKind.HAND) -> None:
        self._name = name
        self._kind = kind
        self._phase = RecordPhase.COUNTDOWN
        self._phase_start = now
        self._samples.clear()
        self._hands.clear()

    def cancel(self) -> None:
        self._name = None

    def update(
        self,
        hands: Sequence[HandDetection],
        aspect: float,
        now: float,
        faces: Sequence[FaceDetection] = (),
    ) -> RecorderStatus | None:
        """Advance one frame. Returns None when idle; DONE/FAILED are reported exactly once."""
        name = self._name
        if name is None:
            return None
        kind = self._kind
        sample, handedness = self._sample(hands, faces, aspect)
        visible = sample is not None

        if self._phase is RecordPhase.COUNTDOWN:
            left = self._countdown - (now - self._phase_start)
            if left > 0:
                return RecorderStatus(name, RecordPhase.COUNTDOWN, kind, seconds_left=left,
                                      visible=visible)  # fmt: skip
            self._phase = RecordPhase.CAPTURING
            self._phase_start = now

        if sample is not None:
            self._samples.append(sample)
            if handedness is not None:
                self._hands.append(handedness)

        what = "face" if kind is GestureKind.FACE else "hand"
        if len(self._samples) >= self._capture_frames:
            self._name = None
            template = self._build_template(name)
            return RecorderStatus(
                name, RecordPhase.DONE, kind, progress=1.0, visible=visible,
                message=f"Recorded '{name}'", template=template,
                warning=self._warning(template),
            )  # fmt: skip

        if now - self._phase_start > self._timeout:
            self._name = None
            return RecorderStatus(
                name, RecordPhase.FAILED, kind, visible=visible,
                message=f"Couldn't see your {what} long enough. Keep it in frame and try again.",
            )  # fmt: skip

        return RecorderStatus(
            name, RecordPhase.CAPTURING, kind,
            progress=len(self._samples) / self._capture_frames, visible=visible,
            message="" if visible else f"Show your {what} to the camera",
        )  # fmt: skip

    def _sample(
        self, hands: Sequence[HandDetection], faces: Sequence[FaceDetection], aspect: float
    ) -> tuple[np.ndarray | None, Handedness | None]:
        if self._kind is GestureKind.FACE:
            face = next((f for f in faces if f.blendshapes is not None), None)
            return (None if face is None else np.asarray(face.blendshapes, np.float64)), None
        hand = max(hands, key=lambda h: h.handedness_score, default=None)
        if hand is None:
            return None, None
        return normalize(hand.landmarks, hand.handedness, aspect), hand.handedness

    def _build_template(self, name: str) -> GestureTemplate:
        if self._kind is GestureKind.FACE:
            idx = most_diverse(self._samples, self._max_samples, expression_distances)
            samples = np.stack([self._samples[i] for i in idx])
            return GestureTemplate(name=name, samples=samples, kind=GestureKind.FACE)
        idx = most_diverse(self._samples, self._max_samples, hand_distances)
        samples = np.stack([self._samples[i] for i in idx])
        recorded_with = Counter(self._hands).most_common(1)[0][0]
        return GestureTemplate(name=name, samples=samples, recorded_with=recorded_with)

    @staticmethod
    def _warning(template: GestureTemplate) -> str:
        if template.kind is not GestureKind.FACE:
            return ""
        strength = expression_strength(template.samples)
        if strength >= SUBTLE_EXPRESSION:
            return ""
        return (
            f"'{template.name}' looks close to a resting face (strongest movement "
            f"{strength:.0%}), so it may trigger by accident. Consider recording it again "
            "with a bigger, more exaggerated expression."
        )
