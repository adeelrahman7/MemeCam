"""Decides each hand's gesture: a recorded custom pose wins over MediaPipe's built-in label."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum

import numpy as np

from memecam.core.types import HandDetection, Handedness
from memecam.gestures.templates import MatchResult, TemplateMatcher, normalize


class GestureSource(StrEnum):
    BUILTIN = "builtin"
    CUSTOM = "custom"


@dataclass(frozen=True, slots=True)
class HandGesture:
    hand: HandDetection
    name: str | None
    score: float
    source: GestureSource | None
    custom: MatchResult | None  # nearest custom template, even if it wasn't accepted


@dataclass(slots=True)
class _HandState:
    pose: np.ndarray  # smoothed normalized pose
    sticky: str | None = None  # custom gesture matched last frame


class GestureClassifier:
    """Per-frame classifier with a little memory per hand (tracked by handedness).

    - ``smoothing``: exponential moving average of the hand shape (0 = off, 0.5 = each
      frame counts half). Removes landmark jitter so ``d`` doesn't spike for one frame.
    - ``release_factor``: once a custom gesture matches, it's kept until its distance
      exceeds ``threshold * release_factor`` (hysteresis), so it doesn't flicker at the edge.
    """

    def __init__(
        self,
        matcher: TemplateMatcher,
        threshold: float,
        *,
        smoothing: float = 0.5,
        release_factor: float = 1.3,
    ) -> None:
        if not 0.0 <= smoothing < 1.0:
            raise ValueError("smoothing must be in [0, 1)")
        if release_factor < 1.0:
            raise ValueError("release_factor must be >= 1")
        self._matcher = matcher
        self._threshold = threshold
        self._smoothing = smoothing
        self._release = threshold * release_factor
        self._state: dict[Handedness, _HandState] = {}

    def classify_frame(
        self, hands: Sequence[HandDetection], aspect: float
    ) -> tuple[HandGesture, ...]:
        """Classify every hand in a frame and forget hands that disappeared."""
        seen = {h.handedness for h in hands}
        for gone in set(self._state) - seen:
            del self._state[gone]
        return tuple(self._classify(h, aspect) for h in hands)

    def _classify(self, hand: HandDetection, aspect: float) -> HandGesture:
        custom: MatchResult | None = None
        pose = normalize(hand.landmarks, hand.handedness, aspect)
        if pose is not None and self._matcher.names:
            state = self._state.get(hand.handedness)
            if state is None:
                state = self._state[hand.handedness] = _HandState(pose)
            else:
                a = self._smoothing
                state.pose = a * state.pose + (1 - a) * pose
            custom = self._matcher.match(
                state.pose, sticky=state.sticky, release_threshold=self._release
            )
            state.sticky = custom.name
            if custom.name is not None:
                # Map distance 0..threshold onto a 1..0 confidence-style score.
                score = max(0.0, 1.0 - custom.distance / self._threshold)
                return HandGesture(hand, custom.name, score, GestureSource.CUSTOM, custom)
        source = GestureSource.BUILTIN if hand.gesture else None
        return HandGesture(hand, hand.gesture, hand.gesture_score, source, custom)
