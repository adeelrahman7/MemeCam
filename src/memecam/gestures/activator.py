"""Turns "is the gesture present this frame?" into a steady on/off state."""

from __future__ import annotations


class HoldActivator:
    """On after the gesture is seen for ``hold_frames`` frames in a row; stays on while it's
    held; turns off ``linger_seconds`` after it's released.

    Used for gesture-triggered face overlays (e.g. hold a fist → sunglasses stay on).
    Time is passed in explicitly, keeping this pure and testable.
    """

    def __init__(self, hold_frames: int, linger_seconds: float) -> None:
        if hold_frames < 1:
            raise ValueError("hold_frames must be >= 1")
        if linger_seconds < 0:
            raise ValueError("linger_seconds must be >= 0")
        self._hold_frames = hold_frames
        self._linger = linger_seconds
        self._streak = 0
        self._on_until = float("-inf")

    def update(self, present: bool, now: float) -> bool:
        if present:
            self._streak += 1
            if self._streak >= self._hold_frames:
                self._on_until = now + self._linger
        else:
            self._streak = 0
        return now <= self._on_until

    def reset(self) -> None:
        self._streak = 0
        self._on_until = float("-inf")
