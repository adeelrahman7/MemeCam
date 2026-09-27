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


def _opposite(hand: Handedness) -> Handedness:
    return Handedness.LEFT if hand is Handedness.RIGHT else Handedness.RIGHT


def resolve_handedness(label: str, *, input_is_mirrored: bool, swap: bool) -> Handedness:
    """Convert MediaPipe's handedness label into the user's real hand.

    Tested on a Windows webcam, the Tasks API labels hands as they'd be named in an
    ordinary (non-mirrored) photo. So when we feed it the mirrored selfie frame, its
    labels come out backwards and get swapped here. ``swap`` is a manual override for
    cameras that don't match this (e.g. drivers that already mirror their output).
    """
    hand = Handedness(label)
    if input_is_mirrored:
        hand = _opposite(hand)
    if swap:
        hand = _opposite(hand)
    return hand


class GestureTracker:
    """Runs GestureRecognizer in VIDEO mode on RGB frames.

    ``HandDetection.handedness`` always means the user's real hand; see
    ``resolve_handedness`` for how MediaPipe's label is converted.
    """

    def __init__(
        self,
        model_path: Path,
        *,
        num_hands: int,
        min_detection_confidence: float,
        min_gesture_score: float,
        input_is_mirrored: bool,
        swap_handedness: bool = False,
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
        self._swap_handedness = swap_handedness
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
            handedness = resolve_handedness(
                hand_cat.category_name,
                input_is_mirrored=self._input_is_mirrored,
                swap=self._swap_handedness,
            )

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
