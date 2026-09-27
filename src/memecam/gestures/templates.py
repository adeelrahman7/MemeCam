"""Hand-pose templates: normalize 21 landmarks into a comparable shape and match against samples.

Pure numpy, no I/O, no MediaPipe, which keeps it easy to unit-test.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import StrEnum

import numpy as np

from memecam.core.types import Handedness, Landmark

NUM_LANDMARKS = 21
NUM_BLENDSHAPES = 52  # MediaPipe face expression scores (see gestures/expressions.py)
WRIST = 0
MIDDLE_MCP = 9  # base knuckle of the middle finger

# A pose is a (21, 2) float array produced by ``normalize``.
Pose = np.ndarray


def normalize(landmarks: Sequence[Landmark], handedness: Handedness, aspect: float) -> Pose | None:
    """Turn raw landmarks into a position-, size- and hand-independent (21, 2) pose.

    - ``aspect`` (frame width / height) undoes MediaPipe's per-axis normalization, so a
      square hand stays square.
    - Translate so the wrist is at (0, 0).
    - Scale so wrist→middle knuckle has length 1 (distance from camera stops mattering).
    - Mirror left hands onto right hands, so one recording works with either hand.
      Use a reaction's ``"hand"`` filter if you want a gesture to be hand-specific.

    Returns None for degenerate input (wrong landmark count, collapsed hand).
    """
    if len(landmarks) != NUM_LANDMARKS:
        return None
    pts = np.array([(lm.x * aspect, lm.y) for lm in landmarks], dtype=np.float64)
    pts -= pts[WRIST]
    scale = float(np.linalg.norm(pts[MIDDLE_MCP]))
    if scale < 1e-6:
        return None
    pts /= scale
    if handedness is Handedness.LEFT:
        pts[:, 0] *= -1.0
    return pts


def align_rotation(pose: Pose) -> Pose:
    """Rotate a normalized pose so the wrist→middle knuckle direction points straight up."""
    vx, vy = pose[MIDDLE_MCP]
    theta = -math.pi / 2 - math.atan2(vy, vx)  # image y grows downward, so "up" is (0, -1)
    c, s = math.cos(theta), math.sin(theta)
    rot = np.array([[c, -s], [s, c]])
    return pose @ rot.T


def pose_distance(a: Pose, b: Pose) -> float:
    """Root-mean-square landmark distance, in units of 'palm lengths'.

    RMS (rather than a plain mean) lets one clearly different finger dominate, so a
    peace sign and a pointing finger don't look "close" just because 17 points agree.
    """
    return float(np.sqrt(((a - b) ** 2).sum(axis=-1).mean()))


class GestureKind(StrEnum):
    HAND = "hand"  # a hand pose: samples are (n, 21, 2) normalized landmarks
    FACE = "face"  # a face expression: samples are (n, 52) blendshape scores


SAMPLE_SHAPES: dict[GestureKind, tuple[int, ...]] = {
    GestureKind.HAND: (NUM_LANDMARKS, 2),
    GestureKind.FACE: (NUM_BLENDSHAPES,),
}


@dataclass(frozen=True, slots=True)
class GestureTemplate:
    name: str
    samples: np.ndarray  # hand: (n, 21, 2) poses, not rotation-aligned; face: (n, 52) scores
    recorded_with: Handedness | None = None  # hands only
    kind: GestureKind = GestureKind.HAND

    def __post_init__(self) -> None:
        expected = SAMPLE_SHAPES[self.kind]
        if self.samples.ndim != len(expected) + 1 or self.samples.shape[1:] != expected:
            shape = "(n, " + ", ".join(map(str, expected)) + ")"
            raise ValueError(
                f"{self.kind} samples must have shape {shape}, got {self.samples.shape}"
            )
        if len(self.samples) == 0:
            raise ValueError("a template needs at least one sample")


@dataclass(frozen=True, slots=True)
class MatchResult:
    """Outcome of comparing one live pose against all templates."""

    name: str | None  # the accepted gesture, or None
    nearest: str | None  # closest template even if rejected (for the debug HUD)
    distance: float  # distance to ``nearest`` (inf if there are no templates)


class TemplateMatcher:
    """Nearest-template classifier with an absolute threshold and an ambiguity check.

    A pose matches template T when:
    1. its distance to T's closest sample is <= ``threshold``, and
    2. it's clearly closer to T than to any other template:
       best <= ``ambiguity_ratio`` * second_best.
    """

    def __init__(
        self,
        templates: Sequence[GestureTemplate],
        *,
        threshold: float,
        rotation_invariant: bool,
        ambiguity_ratio: float = 0.85,
    ) -> None:
        if threshold <= 0:
            raise ValueError("threshold must be > 0")
        self._threshold = threshold
        self._rotation_invariant = rotation_invariant
        self._ratio = ambiguity_ratio
        self._templates: list[tuple[str, np.ndarray]] = []
        for t in templates:
            samples = t.samples
            if rotation_invariant:
                samples = np.stack([align_rotation(s) for s in samples])
            self._templates.append((t.name, samples))

    @property
    def names(self) -> list[str]:
        return [name for name, _ in self._templates]

    def match(
        self, pose: Pose, *, sticky: str | None = None, release_threshold: float = 0.0
    ) -> MatchResult:
        """Classify ``pose``.

        ``sticky`` is the gesture this hand matched last frame. It's kept while its distance
        stays within ``release_threshold`` (a bit looser than ``threshold``), so a pose that
        hovers around the threshold doesn't flicker on and off.
        """
        if not self._templates:
            return MatchResult(None, None, math.inf)
        if self._rotation_invariant:
            pose = align_rotation(pose)

        scored = [
            (float(hand_distances(samples, pose).min()), name) for name, samples in self._templates
        ]
        return choose_match(
            scored,
            threshold=self._threshold,
            ambiguity_ratio=self._ratio,
            sticky=sticky,
            release_threshold=release_threshold,
        )


def choose_match(
    scored: Sequence[tuple[float, str]],
    *,
    threshold: float,
    ambiguity_ratio: float,
    sticky: str | None,
    release_threshold: float,
) -> MatchResult:
    """Pick a winner from (distance, name) pairs; shared by hand and face matching.

    - A gesture matched last frame (``sticky``) is kept while within ``release_threshold``.
    - Otherwise the closest wins if it's within ``threshold`` and clearly closer than the
      runner-up (best <= ``ambiguity_ratio`` * second best).
    """
    if not scored:
        return MatchResult(None, None, math.inf)
    ranked = sorted(scored)
    if sticky is not None:
        for dist, name in ranked:
            if name == sticky and dist <= release_threshold:
                return MatchResult(name, name, dist)
    best_dist, best_name = ranked[0]
    second = ranked[1][0] if len(ranked) > 1 else math.inf
    accepted = best_dist <= threshold and best_dist <= ambiguity_ratio * second
    return MatchResult(best_name if accepted else None, best_name, best_dist)


def hand_distances(stack: np.ndarray, pose: Pose) -> np.ndarray:
    """RMS landmark distance from every pose in ``stack`` (n, 21, 2) to ``pose``."""
    return np.sqrt(((stack - pose) ** 2).sum(axis=-1).mean(axis=1))


def most_diverse(
    poses: Sequence[np.ndarray],
    count: int,
    distances: Callable[[np.ndarray, np.ndarray], np.ndarray] = hand_distances,
) -> list[int]:
    """Pick ``count`` indices that spread across the recorded variation (farthest-point).

    Evenly spaced frames from a steady hold are nearly identical; this keeps the frames
    that differ most, so whatever variety you showed while recording ends up in the template.
    ``distances(stack, item)`` returns each stacked sample's distance to ``item``.
    """
    n = len(poses)
    if count >= n:
        return list(range(n))
    stack = np.stack(poses)
    chosen = [n // 2]  # start from the middle of the hold, the most "typical" frame
    nearest = distances(stack, stack[chosen[0]])
    while len(chosen) < count:
        nxt = int(nearest.argmax())
        chosen.append(nxt)
        nearest = np.minimum(nearest, distances(stack, stack[nxt]))
    return sorted(chosen)
