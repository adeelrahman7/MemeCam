"""MediaPipe GestureRecognizer (Tasks API) wrapper. Returns data only; never draws."""

from __future__ import annotations

from pathlib import Path
from types import TracebackType

import mediapipe as mp
import numpy as np

from memecam.core.types import GestureFrame, HandDetection, Handedness, Landmark

GESTURE_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/gesture_recognizer/"
    "gesture_recognizer/float16/latest/gesture_recognizer.task"
)


class ModelNotFoundError(FileNotFoundError):
    pass


class GestureTracker:
    """Runs GestureRecognizer in VIDEO mode on RGB frames.

    Handedness: MediaPipe labels hands *assuming the input is mirrored* (a selfie view).
    When we feed it the mirrored frame, "Right" really is the user's right hand, and that
    hand also appears on the right side of the mirrored preview. If the input is not mirrored
    we swap the label so ``HandDetection.handedness`` always means the user's real hand.
    """

    def __init__(
        self,
        model_path: Path,
        *,
        num_hands: int,
        min_detection_confidence: float,
        min_gesture_score: float,
        input_is_mirrored: bool,
    ) -> None:
        if not model_path.is_file():
            raise ModelNotFoundError(
                f"Gesture model not found at {model_path}.\n"
                f"Download it from:\n  {GESTURE_MODEL_URL}\n"
                f"and save it as {model_path.name} in the models/ folder."
            )
        vision = mp.tasks.vision
        options = vision.GestureRecognizerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(model_path)),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=num_hands,
            min_hand_detection_confidence=min_detection_confidence,
            min_hand_presence_confidence=min_detection_confidence,
            min_tracking_confidence=min_detection_confidence,
        )
        self._recognizer = vision.GestureRecognizer.create_from_options(options)
        self._min_gesture_score = min_gesture_score
        self._input_is_mirrored = input_is_mirrored
        self._last_timestamp_ms = -1

    def process(self, frame_rgb: np.ndarray, timestamp_ms: int) -> GestureFrame:
        # VIDEO mode requires strictly increasing timestamps.
        timestamp_ms = max(timestamp_ms, self._last_timestamp_ms + 1)
        self._last_timestamp_ms = timestamp_ms

        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(frame_rgb))
        result = self._recognizer.recognize_for_video(image, timestamp_ms)

        hands: list[HandDetection] = []
        for i, landmarks in enumerate(result.hand_landmarks):
            hand_cat = result.handedness[i][0]
            handedness = Handedness(hand_cat.category_name)
            if not self._input_is_mirrored:
                handedness = Handedness.LEFT if handedness is Handedness.RIGHT else Handedness.RIGHT

            gesture: str | None = None
            gesture_score = 0.0
            if i < len(result.gestures) and result.gestures[i]:
                top = result.gestures[i][0]
                gesture_score = float(top.score)
                if (
                    top.category_name not in ("", "None")
                    and gesture_score >= self._min_gesture_score
                ):
                    gesture = top.category_name

            hands.append(
                HandDetection(
                    handedness=handedness,
                    handedness_score=float(hand_cat.score),
                    gesture=gesture,
                    gesture_score=gesture_score,
                    landmarks=tuple(Landmark(lm.x, lm.y, lm.z) for lm in landmarks),
                )
            )
        return GestureFrame(hands=tuple(hands))

    def close(self) -> None:
        self._recognizer.close()

    def __enter__(self) -> GestureTracker:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()
