"""Turns a noisy per-frame gesture stream into discrete, deliberate trigger events."""

from __future__ import annotations


class GestureDebouncer:
    """Fires a gesture only after it has been held for ``hold_frames`` consecutive frames.

    Rules:
    - A frame with a different gesture (or ``None``) resets the streak.
    - Once a gesture fires, it won't fire again until it's released or changed, so holding
      a thumbs-up for ten seconds triggers one reaction, not ten.
    - After any fire, a global cooldown blocks every gesture for ``cooldown_seconds``.
      A gesture that finishes its hold during the cooldown fires as soon as it ends
      (if it's still being held).

    Time is passed in explicitly (monotonic seconds), which keeps this class pure and testable.
    """

    def __init__(self, hold_frames: int, cooldown_seconds: float) -> None:
        if hold_frames < 1:
            raise ValueError("hold_frames must be >= 1")
        if cooldown_seconds < 0:
            raise ValueError("cooldown_seconds must be >= 0")
        self._hold_frames = hold_frames
        self._cooldown_seconds = cooldown_seconds
        self._candidate: str | None = None
        self._streak = 0
        self._fired_current = False
        self._cooldown_until = float("-inf")

    @property
    def candidate(self) -> str | None:
        """The gesture currently being held (not necessarily fired yet)."""
        return self._candidate

    @property
    def progress(self) -> float:
        """How far through the hold the current candidate is, 0..1 (handy for a debug HUD)."""
        if self._candidate is None:
            return 0.0
        return min(1.0, self._streak / self._hold_frames)

    def in_cooldown(self, now: float) -> bool:
        return now < self._cooldown_until

    def update(self, gesture: str | None, now: float) -> str | None:
        """Feed one frame's gesture. Returns the gesture name on the frame it fires, else None."""
        if gesture != self._candidate:
            self._candidate = gesture
            self._streak = 0
            self._fired_current = False

        if gesture is None:
            return None

        self._streak += 1
        if self._fired_current or self._streak < self._hold_frames or self.in_cooldown(now):
            return None

        self._fired_current = True
        self._cooldown_until = now + self._cooldown_seconds
        return gesture

    def reset(self) -> None:
        self._candidate = None
        self._streak = 0
        self._fired_current = False
        self._cooldown_until = float("-inf")
