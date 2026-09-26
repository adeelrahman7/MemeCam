"""QThread worker: owns the camera and MediaPipe, emits rendered frames to the UI."""

from __future__ import annotations

import threading
import time
from pathlib import Path

import numpy as np
from PySide6.QtCore import QThread, Signal
from PySide6.QtGui import QImage

from memecam.config import AppConfig
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


class FrameWorker(QThread):
    """Capture + inference loop. The UI thread never touches the camera or MediaPipe.

    Both are created *inside* run(), so they live entirely on this thread.
    """

    frame_ready = Signal(QImage)
    stats_ready = Signal(object)  # FrameStats
    reaction_fired = Signal(str)
    failed = Signal(str)

    def __init__(self, config: AppConfig, root: Path) -> None:
        super().__init__()
        self._config = config
        self._root = root
        self._debug = threading.Event()

    def set_debug(self, enabled: bool) -> None:
        """Thread-safe; can be called from the UI thread at any time."""
        if enabled:
            self._debug.set()
        else:
            self._debug.clear()

    def stop(self) -> None:
        self.requestInterruption()
        self.wait(3000)

    def run(self) -> None:
        cfg = self._config
        try:
            camera = Camera(cfg.camera.index, cfg.camera.width, cfg.camera.height)
        except CameraError as exc:
            self.failed.emit(str(exc))
            return

        try:
            tracker = GestureTracker(
                resolve_resource(cfg.inference.model, self._root),
                num_hands=cfg.inference.num_hands,
                min_detection_confidence=cfg.inference.min_detection_confidence,
                min_gesture_score=cfg.inference.min_gesture_score,
                input_is_mirrored=cfg.camera.mirror,
            )
        except (ModelNotFoundError, RuntimeError, ValueError) as exc:
            camera.release()
            self.failed.emit(str(exc))
            return

        try:
            self._loop(camera, FramePipeline(cfg, tracker, self._root))
        except Exception as exc:  # surface anything unexpected instead of dying silently
            self.failed.emit(f"Frame worker crashed: {exc!r}")
        finally:
            tracker.close()
            camera.release()

    def _loop(self, camera: Camera, pipeline: FramePipeline) -> None:
        failures = 0
        while not self.isInterruptionRequested():
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
            self.frame_ready.emit(bgr_to_qimage(result.image_bgr))
            self.stats_ready.emit(result.stats)
