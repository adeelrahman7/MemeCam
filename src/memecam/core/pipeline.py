"""The per-frame pipeline: mirror → downscale → infer → classify → debounce → draw.

No Qt and no camera in here, so it can be driven from tests with fake frames.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Protocol

import cv2
import numpy as np

from memecam.config import Settings
from memecam.core.fps import FpsMeter
from memecam.core.types import FrameStats, GestureFrame
from memecam.gestures.classifier import GestureClassifier, HandGesture
from memecam.gestures.debouncer import GestureDebouncer
from memecam.gestures.recorder import GestureRecorder, RecorderStatus, RecordPhase
from memecam.gestures.templates import TemplateMatcher
from memecam.paths import resolve_resource
from memecam.render.debug import DebugRenderer
from memecam.render.overlay import CornerOverlay
from memecam.render.recording import RecordingRenderer

# After a recording ends you're usually still holding the pose; don't fire its meme.
POST_RECORDING_QUIET_SECONDS = 1.5


class GestureSource(Protocol):
    def process(self, frame_rgb: np.ndarray, timestamp_ms: int) -> GestureFrame: ...


@dataclass(slots=True)
class PipelineResult:
    """A fully rendered BGR frame plus what happened in it."""

    image_bgr: np.ndarray
    hands: tuple[HandGesture, ...] = ()
    stats: FrameStats = field(default_factory=FrameStats)
    fired_reaction: str | None = None
    recording: RecorderStatus | None = None


def downscale(frame: np.ndarray, max_width: int) -> np.ndarray:
    """Return a copy no wider than ``max_width`` (aspect preserved)."""
    h, w = frame.shape[:2]
    if w <= max_width:
        return frame.copy()
    scale = max_width / w
    return cv2.resize(frame, (max_width, round(h * scale)), interpolation=cv2.INTER_AREA)


class FramePipeline:
    def __init__(self, settings: Settings, tracker: GestureSource) -> None:
        config = settings.config
        self._config = config
        self._tracker = tracker
        custom = config.custom_gestures
        self._classifier = GestureClassifier(
            TemplateMatcher(
                settings.library.templates,
                threshold=custom.threshold,
                rotation_invariant=custom.rotation_invariant,
            ),
            custom.threshold,
            smoothing=custom.smoothing,
            release_factor=custom.release_factor,
        )
        self._recorder = GestureRecorder(
            countdown_seconds=custom.countdown_seconds, capture_frames=custom.capture_frames
        )
        self._debouncer = GestureDebouncer(
            config.debounce.hold_frames, config.debounce.cooldown_seconds
        )
        self._overlay = CornerOverlay(
            {r.key: resolve_resource(r.image, settings.root) for r in config.reactions},
            corner=config.overlay.corner,
            width_fraction=config.overlay.width_fraction,
            margin_px=config.overlay.margin_px,
            display_seconds=config.overlay.display_seconds,
        )
        self._debug_renderer = DebugRenderer()
        self._recording_renderer = RecordingRenderer()
        self._fps = FpsMeter()
        self.debug = False

    def start_recording(self, name: str, now: float) -> None:
        self._recorder.start(name, now)

    def cancel_recording(self) -> None:
        self._recorder.cancel()

    def _select_reaction_key(self, hands: tuple[HandGesture, ...]) -> str | None:
        """Pick the most confident hand whose gesture maps to a configured reaction."""
        best: tuple[float, str] | None = None
        for hg in hands:
            if hg.name is None:
                continue
            reaction = self._config.find_reaction(hg.name, hg.hand.handedness)
            if reaction is not None and (best is None or hg.score > best[0]):
                best = (hg.score, reaction.key)
        return best[1] if best else None

    def process(self, frame_bgr: np.ndarray, now: float) -> PipelineResult:
        """Process one camera frame (BGR). May modify ``frame_bgr`` in place."""
        if self._config.camera.mirror:
            frame_bgr = cv2.flip(frame_bgr, 1)
        h, w = frame_bgr.shape[:2]
        aspect = w / h

        small_rgb = cv2.cvtColor(
            downscale(frame_bgr, self._config.inference.max_width), cv2.COLOR_BGR2RGB
        )
        t0 = time.perf_counter()
        gestures = self._tracker.process(small_rgb, round(now * 1000))
        inference_ms = (time.perf_counter() - t0) * 1000

        hands = self._classifier.classify_frame(gestures.hands, aspect)
        recording = self._recorder.update(gestures.hands, aspect, now)

        fired: str | None = None
        if recording is None:
            key = self._select_reaction_key(hands)
            fired = self._debouncer.update(key, now)
            if fired is not None:
                self._overlay.trigger(fired, now)
            elif key is not None and self._config.overlay.stay_while_held:
                self._overlay.hold(key, now, self._config.overlay.linger_seconds)
            # Landmarks are normalized, so they map straight onto the full-size frame.
            self._overlay.draw(frame_bgr, now)
        else:
            self._debouncer.update(None, now)  # no memes while you're teaching a pose
            if recording.phase in (RecordPhase.DONE, RecordPhase.FAILED):
                self._debouncer.suppress_until(now + POST_RECORDING_QUIET_SECONDS)

        stats = FrameStats(fps=self._fps.tick(now), inference_ms=inference_ms)
        if self.debug:
            self._debug_renderer.draw(
                frame_bgr,
                hands,
                stats,
                candidate=self._debouncer.candidate,
                progress=self._debouncer.progress,
                cooldown=self._debouncer.in_cooldown(now),
            )
        if recording is not None:
            self._recording_renderer.draw(frame_bgr, recording)
        return PipelineResult(frame_bgr, hands, stats, fired, recording)
