"""Webcam access via OpenCV. Only ever used from the worker thread."""

from __future__ import annotations

import sys

import cv2
import numpy as np


class CameraError(RuntimeError):
    pass


class Camera:
    def __init__(self, index: int, width: int, height: int) -> None:
        # DirectShow opens much faster than the default MSMF backend on Windows.
        backend = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY
        self._cap = cv2.VideoCapture(index, backend)
        if not self._cap.isOpened():
            raise CameraError(
                f"Could not open camera {index}. Is another app using it, or does the "
                "OS need camera permission for your terminal/Python?"
            )
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    def read(self) -> np.ndarray | None:
        """Return the next BGR frame, or None if the camera didn't deliver one."""
        ok, frame = self._cap.read()
        return frame if ok else None

    def release(self) -> None:
        self._cap.release()
