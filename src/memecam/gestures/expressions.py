"""Face expressions as gestures: compare MediaPipe blendshape scores against recordings.

A face "pose" is the 52 blendshape scores FaceLandmarker reports (each 0..1: how much your
jaw is open, how much you're smiling, ...). They're already independent of where your face
is, how big it is and how it's tilted, so no normalization step is needed.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from memecam.core.types import FaceDetection
from memecam.gestures.templates import (
    NUM_BLENDSHAPES,
    GestureKind,
    GestureTemplate,
    MatchResult,
    choose_match,
)

# Index order matches MediaPipe's FaceLandmarker output.
BLENDSHAPE_NAMES: tuple[str, ...] = (
    "_neutral", "browDownLeft", "browDownRight", "browInnerUp", "browOuterUpLeft",
    "browOuterUpRight", "cheekPuff", "cheekSquintLeft", "cheekSquintRight", "eyeBlinkLeft",
    "eyeBlinkRight", "eyeLookDownLeft", "eyeLookDownRight", "eyeLookInLeft", "eyeLookInRight",
    "eyeLookOutLeft", "eyeLookOutRight", "eyeLookUpLeft", "eyeLookUpRight", "eyeSquintLeft",
    "eyeSquintRight", "eyeWideLeft", "eyeWideRight", "jawForward", "jawLeft", "jawOpen",
    "jawRight", "mouthClose", "mouthDimpleLeft", "mouthDimpleRight", "mouthFrownLeft",
    "mouthFrownRight", "mouthFunnel", "mouthLeft", "mouthLowerDownLeft", "mouthLowerDownRight",
    "mouthPressLeft", "mouthPressRight", "mouthPucker", "mouthRight", "mouthRollLower",
    "mouthRollUpper", "mouthShrugLower", "mouthShrugUpper", "mouthSmileLeft", "mouthSmileRight",
    "mouthStretchLeft", "mouthStretchRight", "mouthUpperUpLeft", "mouthUpperUpRight",
    "noseSneerLeft", "noseSneerRight",
)  # fmt: skip
assert len(BLENDSHAPE_NAMES) == NUM_BLENDSHAPES

# Ignored when comparing expressions: "_neutral" isn't a muscle, and where you're looking
# (eyeLook*) shouldn't change which face you're making.
IGNORED = frozenset(
    i for i, n in enumerate(BLENDSHAPE_NAMES) if n == "_neutral" or n.startswith("eyeLook")
)
USED = np.array([i for i in range(NUM_BLENDSHAPES) if i not in IGNORED])

# Recordings whose strongest movement is below this probably look like a resting face.
SUBTLE_EXPRESSION = 0.3


def expression_distances(stack: np.ndarray, scores: np.ndarray) -> np.ndarray:
    """Euclidean distance over the used scores, from each row of ``stack`` (n, 52) to
    ``scores`` (52,). A big smile vs a neutral face is roughly 1.0; noise is ~0.1."""
    diff = stack[..., USED] - scores[USED]
    return np.sqrt((diff**2).sum(axis=-1))


def expression_strength(samples: np.ndarray) -> float:
    """The largest average score among the used blendshapes (0..1)."""
    return float(samples[:, USED].mean(axis=0).max())


def top_blendshapes(scores: np.ndarray, count: int = 3) -> list[tuple[str, float]]:
    """The strongest ``count`` used blendshapes, for the debug overlay."""
    order = sorted(USED, key=lambda i: -scores[i])[:count]
    return [(BLENDSHAPE_NAMES[i], float(scores[i])) for i in order]


class ExpressionMatcher:
    def __init__(
        self,
        templates: Sequence[GestureTemplate],
        *,
        threshold: float,
        ambiguity_ratio: float = 0.85,
    ) -> None:
        if threshold <= 0:
            raise ValueError("threshold must be > 0")
        self._templates = [(t.name, t.samples) for t in templates if t.kind is GestureKind.FACE]
        self._threshold = threshold
        self._ratio = ambiguity_ratio

    @property
    def names(self) -> list[str]:
        return [name for name, _ in self._templates]

    def match(
        self, scores: np.ndarray, *, sticky: str | None = None, release_threshold: float = 0.0
    ) -> MatchResult:
        scored = [
            (float(expression_distances(samples, scores).min()), name)
            for name, samples in self._templates
        ]
        return choose_match(
            scored,
            threshold=self._threshold,
            ambiguity_ratio=self._ratio,
            sticky=sticky,
            release_threshold=release_threshold,
        )


@dataclass(frozen=True, slots=True)
class FaceGesture:
    face_index: int
    name: str | None  # matched custom expression, or None
    score: float  # 1 at a perfect match, 0 at the threshold
    match: MatchResult | None  # nearest expression even if not accepted (debug HUD)
    scores: np.ndarray | None  # smoothed blendshape scores (debug HUD)


@dataclass(slots=True)
class _FaceState:
    scores: np.ndarray
    sticky: str | None = None


class ExpressionClassifier:
    """Per-frame expression matching with smoothing and hysteresis, like hand gestures.

    Faces are tracked by order; if the number of faces changes, memory is reset.
    """

    def __init__(
        self,
        matcher: ExpressionMatcher,
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
        self._state: list[_FaceState] = []

    def classify_frame(self, faces: Sequence[FaceDetection]) -> tuple[FaceGesture, ...]:
        usable = [f.blendshapes for f in faces]
        if len(usable) != len(self._state) or any(b is None for b in usable):
            self._state = []
        out: list[FaceGesture] = []
        for i, raw in enumerate(usable):
            if raw is None:
                out.append(FaceGesture(i, None, 0.0, None, None))
                continue
            if i < len(self._state):
                st = self._state[i]
                a = self._smoothing
                st.scores = a * st.scores + (1 - a) * raw
            else:
                st = _FaceState(raw.astype(np.float64))
                self._state.append(st)
            if not self._matcher.names:
                out.append(FaceGesture(i, None, 0.0, None, st.scores))
                continue
            m = self._matcher.match(st.scores, sticky=st.sticky, release_threshold=self._release)
            st.sticky = m.name
            score = max(0.0, 1.0 - m.distance / self._threshold) if m.name else 0.0
            out.append(FaceGesture(i, m.name, score, m, st.scores))
        return tuple(out)
