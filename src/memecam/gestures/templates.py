"""Hand-pose templates: normalize 21 landmarks into a comparable shape and match against samples.

Pure numpy, no I/O, no MediaPipe, which keeps it easy to unit-test.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from memecam.core.types import Handedness, Landmark

NUM_LANDMARKS = 21
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


@dataclass(frozen=True, slots=True)
class GestureTemplate:
    name: str
    samples: np.ndarray  # (n, 21, 2) normalized poses, not rotation-aligned
    recorded_with: Handedness | None = None

    def __post_init__(self) -> None:
        if self.samples.ndim != 3 or self.samples.shape[1:] != (NUM_LANDMARKS, 2):
            raise ValueError(f"samples must have shape (n, 21, 2), got {self.samples.shape}")
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

        scored = sorted(
            (float(np.sqrt(((samples - pose) ** 2).sum(axis=-1).mean(axis=1)).min()), name)
            for name, samples in self._templates
        )
        if sticky is not None:
            for dist, name in scored:
                if name == sticky and dist <= release_threshold:
                    return MatchResult(name, name, dist)
        best_dist, best_name = scored[0]
        second = scored[1][0] if len(scored) > 1 else math.inf
        accepted = best_dist <= self._threshold and best_dist <= self._ratio * second
        return MatchResult(best_name if accepted else None, best_name, best_dist)


def most_diverse(poses: Sequence[Pose], count: int) -> list[int]:
    """Pick ``count`` indices that spread across the recorded variation (farthest-point).

    Evenly spaced frames from a steady hold are nearly identical; this keeps the frames
    that differ most, so whatever variety you showed while recording ends up in the template.
    """
    n = len(poses)
    if count >= n:
        return list(range(n))
    stack = np.stack(poses)
    chosen = [n // 2]  # start from the middle of the hold, the most "typical" frame
    nearest = np.sqrt(((stack - stack[chosen[0]]) ** 2).sum(axis=-1).mean(axis=1))
    while len(chosen) < count:
        nxt = int(nearest.argmax())
        chosen.append(nxt)
        d = np.sqrt(((stack - stack[nxt]) ** 2).sum(axis=-1).mean(axis=1))
        nearest = np.minimum(nearest, d)
    return sorted(chosen)
