"""MediaPipe FaceLandmarker (Tasks API) wrapper. Returns data only; never draws."""

from __future__ import annotations

from pathlib import Path
from types import TracebackType

import mediapipe as mp
import numpy as np

from memecam.core.types import FaceDetection, FaceFrame
from memecam.gestures.templates import NUM_BLENDSHAPES
from memecam.tracking.gesture_tracker import ModelNotFoundError

FACE_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/latest/face_landmarker.task"
)


class FaceTracker:
    """Runs FaceLandmarker in VIDEO mode on RGB frames (feed it the same mirrored frame
    the user sees, so landmarks line up with the preview)."""

    def __init__(self, model_path: Path, *, num_faces: int, min_confidence: float) -> None:
        if not model_path.is_file():
            raise ModelNotFoundError(
                f"Face model not found at {model_path}.\n"
                f"Face overlays need it. Download it from:\n  {FACE_MODEL_URL}\n"
                f"and save it as {model_path.name} in the models/ folder."
            )
        vision = mp.tasks.vision
        options = vision.FaceLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(model_path)),
            running_mode=vision.RunningMode.VIDEO,
            num_faces=num_faces,
            min_face_detection_confidence=min_confidence,
            min_face_presence_confidence=min_confidence,
            min_tracking_confidence=min_confidence,
            output_face_blendshapes=True,  # expression scores, for face-expression gestures
        )
        self._landmarker = vision.FaceLandmarker.create_from_options(options)
        self._last_timestamp_ms = -1

    def process(self, frame_rgb: np.ndarray, timestamp_ms: int) -> FaceFrame:
        # VIDEO mode requires strictly increasing timestamps.
        timestamp_ms = max(timestamp_ms, self._last_timestamp_ms + 1)
        self._last_timestamp_ms = timestamp_ms

        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(frame_rgb))
        result = self._landmarker.detect_for_video(image, timestamp_ms)
        faces = []
        for i, face in enumerate(result.face_landmarks):
            points = np.array([(lm.x, lm.y, lm.z) for lm in face], dtype=np.float32)
            blendshapes = None
            if i < len(result.face_blendshapes) and result.face_blendshapes[i]:
                blendshapes = np.zeros(NUM_BLENDSHAPES, dtype=np.float64)
                for j, cat in enumerate(result.face_blendshapes[i]):
                    idx = cat.index if cat.index is not None else j  # results come in order
                    if 0 <= idx < NUM_BLENDSHAPES:
                        blendshapes[idx] = cat.score
            faces.append(FaceDetection(points, blendshapes))
        return FaceFrame(tuple(faces))

    def close(self) -> None:
        self._landmarker.close()

    def __enter__(self) -> FaceTracker:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()
