"""Frame-rate measurement."""

from __future__ import annotations


class FpsMeter:
    """Exponential moving average of frames per second, driven by explicit timestamps."""

    def __init__(self, smoothing: float = 0.9) -> None:
        if not 0.0 <= smoothing < 1.0:
            raise ValueError("smoothing must be in [0, 1)")
        self._smoothing = smoothing
        self._last: float | None = None
        self._fps = 0.0

    @property
    def fps(self) -> float:
        return self._fps

    def tick(self, now: float) -> float:
        if self._last is not None:
            dt = now - self._last
            if dt > 0:
                instant = 1.0 / dt
                self._fps = (
                    instant
                    if self._fps == 0.0
                    else self._smoothing * self._fps + (1 - self._smoothing) * instant
                )
        self._last = now
        return self._fps
