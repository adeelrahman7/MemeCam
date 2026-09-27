"""QThread worker: owns the camera and MediaPipe, emits rendered frames to the UI."""

from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass

import numpy as np
from PySide6.QtCore import QThread, Signal
from PySide6.QtGui import QImage

from memecam.config import Settings
from memecam.core.camera import Camera, CameraError
from memecam.core.pipeline import FramePipeline
from memecam.paths import resolve_resource
from memecam.tracking.gesture_tracker import GestureTracker, ModelNotFoundError

_MAX_CONSECUTIVE_READ_FAILURES = 30


def bgr_to_qimage(frame_bgr: np.ndarray) -> QImage:
    """Wrap a BGR frame in a QImage and deep-copy it so it outlives the numpy buffer."""
    frame = np.ascontiguousarray(frame_bgr)
    h, w = frame.shape[:2]
    return QImage(frame.data, w, h, frame.strides[0], QImage.Format.Format_BGR888).copy()


# Commands sent from the UI thread; the worker applies them between frames.
@dataclass(frozen=True, slots=True)
class _StartRecording:
    name: str


@dataclass(frozen=True, slots=True)
class _CancelRecording:
    pass


@dataclass(frozen=True, slots=True)
class _ApplySettings:
    settings: Settings


_Command = _StartRecording | _CancelRecording | _ApplySettings


class FrameWorker(QThread):
    """Capture + inference loop. The UI thread never touches the camera or MediaPipe.

    Both are created *inside* run(), so they live entirely on this thread. The UI talks
    to the loop only through thread-safe methods (a command queue and an Event).
    """

    frame_ready = Signal(QImage)
    stats_ready = Signal(object)  # FrameStats
    reaction_fired = Signal(str)
    recording_updated = Signal(object)  # RecorderStatus
    failed = Signal(str)

    def __init__(self, settings: Settings) -> None:
        super().__init__()
        self._settings = settings
        self._debug = threading.Event()
        self._commands: queue.SimpleQueue[_Command] = queue.SimpleQueue()

    # --- thread-safe API for the UI thread -------------------------------------------
    def set_debug(self, enabled: bool) -> None:
        if enabled:
            self._debug.set()
        else:
            self._debug.clear()

    def start_recording(self, name: str) -> None:
        self._commands.put(_StartRecording(name))

    def cancel_recording(self) -> None:
        self._commands.put(_CancelRecording())

    def apply_settings(self, settings: Settings) -> None:
        """Swap in new reactions/custom gestures without restarting the camera."""
        self._commands.put(_ApplySettings(settings))

    def stop(self) -> None:
        self.requestInterruption()
        self.wait(3000)

    # --- worker thread ----------------------------------------------------------------
    def run(self) -> None:
        cfg = self._settings.config
        try:
            camera = Camera(cfg.camera.index, cfg.camera.width, cfg.camera.height)
        except CameraError as exc:
            self.failed.emit(str(exc))
            return

        try:
            tracker = GestureTracker(
                resolve_resource(cfg.inference.model, self._settings.root),
                num_hands=cfg.inference.num_hands,
                min_detection_confidence=cfg.inference.min_detection_confidence,
                min_gesture_score=cfg.inference.min_gesture_score,
                input_is_mirrored=cfg.camera.mirror,
                swap_handedness=cfg.camera.swap_handedness,
            )
        except (ModelNotFoundError, RuntimeError, ValueError) as exc:
            camera.release()
            self.failed.emit(str(exc))
            return

        try:
            self._loop(camera, tracker)
        except Exception as exc:  # surface anything unexpected instead of dying silently
            self.failed.emit(f"Frame worker crashed: {exc!r}")
        finally:
            tracker.close()
            camera.release()

    def _drain_commands(self, pipeline: FramePipeline, tracker: GestureTracker) -> FramePipeline:
        while True:
            try:
                cmd = self._commands.get_nowait()
            except queue.Empty:
                return pipeline
            match cmd:
                case _StartRecording(name):
                    pipeline.start_recording(name, time.monotonic())
                case _CancelRecording():
                    pipeline.cancel_recording()
                case _ApplySettings(settings):
                    # Camera/model options need a restart; reactions and gestures don't.
                    self._settings = settings
                    pipeline = FramePipeline(settings, tracker)

    def _loop(self, camera: Camera, tracker: GestureTracker) -> None:
        pipeline = FramePipeline(self._settings, tracker)
        failures = 0
        while not self.isInterruptionRequested():
            pipeline = self._drain_commands(pipeline, tracker)
            frame = camera.read()
            if frame is None:
                failures += 1
                if failures >= _MAX_CONSECUTIVE_READ_FAILURES:
                    self.failed.emit("The camera stopped delivering frames.")
                    return
                self.msleep(10)
                continue
            failures = 0

            pipeline.debug = self._debug.is_set()
            result = pipeline.process(frame, time.monotonic())
            if result.fired_reaction is not None:
                self.reaction_fired.emit(result.fired_reaction)
            if result.recording is not None:
                self.recording_updated.emit(result.recording)
            self.frame_ready.emit(bgr_to_qimage(result.image_bgr))
            self.stats_ready.emit(result.stats)
