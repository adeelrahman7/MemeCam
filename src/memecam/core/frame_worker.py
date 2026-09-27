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
from memecam.gestures.recorder import RecorderStatus, RecordPhase
from memecam.gestures.templates import GestureKind
from memecam.paths import resolve_resource
from memecam.tracking.face_tracker import FACE_MODEL_URL, FaceTracker
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
    kind: GestureKind


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
    warning = Signal(str)  # non-fatal problem, e.g. face model missing
    failed = Signal(str)

    def __init__(self, settings: Settings) -> None:
        super().__init__()
        self._settings = settings
        self._debug = threading.Event()
        self._commands: queue.SimpleQueue[_Command] = queue.SimpleQueue()
        self._face_tracker: FaceTracker | None = None
        self._face_error: str | None = None

    # --- thread-safe API for the UI thread -------------------------------------------
    def set_debug(self, enabled: bool) -> None:
        if enabled:
            self._debug.set()
        else:
            self._debug.clear()

    def start_recording(self, name: str, kind: GestureKind = GestureKind.HAND) -> None:
        self._commands.put(_StartRecording(name, kind))

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
            if self._face_tracker is not None:
                self._face_tracker.close()
            camera.release()

    def _ensure_face_tracker(
        self, settings: Settings, *, force: bool = False
    ) -> FaceTracker | None:
        """Create the face tracker the first time something needs it (face overlays, recorded
        face expressions, or ``force`` when starting to record a face expression).

        A missing/broken face model isn't fatal: hand gestures keep working, and the user
        gets a warning explaining how to enable face features.
        """
        if self._face_tracker is not None or not (force or settings.needs_face_tracking):
            return self._face_tracker
        face = settings.config.face
        try:
            self._face_tracker = FaceTracker(
                resolve_resource(face.model, settings.root),
                num_faces=face.num_faces,
                min_confidence=face.min_confidence,
            )
        except (ModelNotFoundError, RuntimeError, ValueError) as exc:
            message = f"Face overlays and face expressions are disabled.\n\n{exc}"
            if not force and message != self._face_error:  # don't repeat the same warning
                self._face_error = message
                self.warning.emit(message)
        return self._face_tracker

    def _make_pipeline(self, settings: Settings, tracker: GestureTracker) -> FramePipeline:
        return FramePipeline(settings, tracker, self._ensure_face_tracker(settings))

    def _drain_commands(self, pipeline: FramePipeline, tracker: GestureTracker) -> FramePipeline:
        while True:
            try:
                cmd = self._commands.get_nowait()
            except queue.Empty:
                return pipeline
            match cmd:
                case _StartRecording(name, kind):
                    pipeline = self._start_recording(pipeline, tracker, name, kind)
                case _CancelRecording():
                    pipeline.cancel_recording()
                case _ApplySettings(settings):
                    # Camera/model options need a restart; reactions and gestures don't.
                    self._settings = settings
                    pipeline = self._make_pipeline(settings, tracker)

    def _start_recording(
        self, pipeline: FramePipeline, tracker: GestureTracker, name: str, kind: GestureKind
    ) -> FramePipeline:
        if kind is GestureKind.FACE and not pipeline.has_face_tracker:
            if self._ensure_face_tracker(self._settings, force=True) is None:
                self.recording_updated.emit(
                    RecorderStatus(
                        name,
                        RecordPhase.FAILED,
                        kind,
                        message="Recording a face expression needs the face model.\n\n"
                        f"Download it from:\n  {FACE_MODEL_URL}\n"
                        "and save it as face_landmarker.task in the models/ folder.",
                    )
                )
                return pipeline
            pipeline = self._make_pipeline(self._settings, tracker)
        pipeline.start_recording(name, time.monotonic(), kind)
        return pipeline

    def _loop(self, camera: Camera, tracker: GestureTracker) -> None:
        pipeline = self._make_pipeline(self._settings, tracker)
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
