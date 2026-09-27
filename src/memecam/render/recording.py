"""Teach-mode HUD: countdown, capture progress bar and hints, drawn over the video."""

from __future__ import annotations

import math

import cv2
import numpy as np

from memecam.gestures.recorder import RecorderStatus, RecordPhase
from memecam.gestures.templates import GestureKind

_FONT = cv2.FONT_HERSHEY_SIMPLEX


def _centered_text(
    img: np.ndarray, text: str, cy: int, scale: float, color: tuple[int, int, int]
) -> None:
    thickness = max(2, round(scale * 2.5))
    (tw, th), _ = cv2.getTextSize(text, _FONT, scale, thickness)
    org = ((img.shape[1] - tw) // 2, cy + th // 2)
    cv2.putText(img, text, org, _FONT, scale, (0, 0, 0), thickness + 6, cv2.LINE_AA)
    cv2.putText(img, text, org, _FONT, scale, color, thickness, cv2.LINE_AA)


class RecordingRenderer:
    def draw(self, frame_bgr: np.ndarray, status: RecorderStatus) -> None:
        h, w = frame_bgr.shape[:2]
        s = max(0.6, w / 1280)
        face = status.kind is GestureKind.FACE

        # Red frame border = "recording".
        cv2.rectangle(frame_bgr, (0, 0), (w - 1, h - 1), (40, 40, 230), max(4, round(8 * s)))
        title = f"Recording {'face' if face else 'hand'}: {status.name}"
        _centered_text(frame_bgr, title, round(60 * s), 1.0 * s, (255, 255, 255))

        if status.phase is RecordPhase.COUNTDOWN:
            # Kept in the top third so it doesn't cover the hand you're posing.
            count = str(max(1, math.ceil(status.seconds_left)))
            ready = "Get your face ready" if face else "Get your pose ready"
            _centered_text(frame_bgr, ready, round(115 * s), 0.9 * s,
                           (255, 255, 255))  # fmt: skip
            _centered_text(frame_bgr, count, round(200 * s), 2.6 * s, (80, 220, 255))
        elif status.phase is RecordPhase.CAPTURING:
            hold = (
                "Hold the expression... tilt your head a little"
                if face
                else "Hold it... move it slightly for variety"
            )
            _centered_text(frame_bgr, hold, h - round(110 * s),
                           0.9 * s, (255, 255, 255))  # fmt: skip
            bar_w, bar_h = round(w * 0.5), round(22 * s)
            x0, y0 = (w - bar_w) // 2, h - round(70 * s)
            cv2.rectangle(frame_bgr, (x0, y0), (x0 + bar_w, y0 + bar_h), (60, 60, 60), -1)
            fill = round(bar_w * status.progress)
            cv2.rectangle(frame_bgr, (x0, y0), (x0 + fill, y0 + bar_h), (80, 220, 80), -1)
            cv2.rectangle(frame_bgr, (x0, y0), (x0 + bar_w, y0 + bar_h), (255, 255, 255), 2)

        if status.message and status.phase is not RecordPhase.DONE:
            _centered_text(frame_bgr, status.message, round(170 * s), 1.0 * s, (80, 80, 255))
