"""The per-frame pipeline: mirror → downscale → infer → debounce → draw. No Qt, no camera."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Protocol

import cv2
import numpy as np

from memecam.config import AppConfig
from memecam.core.fps import FpsMeter
from memecam.core.types import FrameStats, GestureFrame, ProcessedFrame
from memecam.gestures.debouncer import GestureDebouncer
from memecam.paths import resolve_resource
from memecam.render.debug import DebugRenderer
from memecam.render.overlay import CornerOverlay


class GestureSource(Protocol):
    def process(self, frame_rgb: np.ndarray, timestamp_ms: int) -> GestureFrame: ...


def downscale(frame: np.ndarray, max_width: int) -> np.ndarray:
    """Return a copy no wider than ``max_width`` (aspect preserved)."""
    h, w = frame.shape[:2]
    if w <= max_width:
        return frame.copy()
    scale = max_width / w
    return cv2.resize(frame, (max_width, round(h * scale)), interpolation=cv2.INTER_AREA)


class FramePipeline:
    def __init__(self, config: AppConfig, tracker: GestureSource, root: Path) -> None:
        """``root`` is what relative image paths in the config resolve against."""
        self._config = config
        self._tracker = tracker
        self._debouncer = GestureDebouncer(
            config.debounce.hold_frames, config.debounce.cooldown_seconds
        )
        self._overlay = CornerOverlay(
            {r.key: resolve_resource(r.image, root) for r in config.reactions},
            corner=config.overlay.corner,
            width_fraction=config.overlay.width_fraction,
            margin_px=config.overlay.margin_px,
            display_seconds=config.overlay.display_seconds,
        )
        self._debug_renderer = DebugRenderer()
        self._fps = FpsMeter()
        self.debug = False

    def _select_reaction_key(self, gestures: GestureFrame) -> str | None:
        """Pick the most confident hand whose gesture maps to a configured reaction."""
        best: tuple[float, str] | None = None
        for hand in gestures.hands:
            if hand.gesture is None:
                continue
            reaction = self._config.find_reaction(hand.gesture, hand.handedness)
            if reaction is not None and (best is None or hand.gesture_score > best[0]):
                best = (hand.gesture_score, reaction.key)
        return best[1] if best else None

    def process(self, frame_bgr: np.ndarray, now: float) -> ProcessedFrame:
        """Process one camera frame. ``frame_bgr`` is modified in place and returned."""
        if self._config.camera.mirror:
            frame_bgr = cv2.flip(frame_bgr, 1)

        small_rgb = cv2.cvtColor(
            downscale(frame_bgr, self._config.inference.max_width), cv2.COLOR_BGR2RGB
        )
        t0 = time.perf_counter()
        gestures = self._tracker.process(small_rgb, round(now * 1000))
        inference_ms = (time.perf_counter() - t0) * 1000

        fired = self._debouncer.update(self._select_reaction_key(gestures), now)
        if fired is not None:
            self._overlay.trigger(fired, now)

        # Landmarks are normalized, so they map straight onto the full-size frame.
        self._overlay.draw(frame_bgr, now)
        stats = FrameStats(fps=self._fps.tick(now), inference_ms=inference_ms)
        if self.debug:
            self._debug_renderer.draw(
                frame_bgr,
                gestures,
                stats,
                candidate=self._debouncer.candidate,
                progress=self._debouncer.progress,
                cooldown=self._debouncer.in_cooldown(now),
            )
        return ProcessedFrame(frame_bgr, gestures, stats, fired)
