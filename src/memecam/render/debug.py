"""Debug HUD: hand skeletons, handedness/gesture labels, FPS and hold progress."""

from __future__ import annotations

from collections.abc import Sequence

import cv2
import numpy as np

from memecam.core.types import FrameStats, Handedness
from memecam.face.anchors import Anchor, place
from memecam.gestures.classifier import GestureSource, HandGesture
from memecam.gestures.expressions import FaceGesture, top_blendshapes

# MediaPipe's 21-point hand topology (wrist=0, thumb 1-4, index 5-8, middle 9-12,
# ring 13-16, pinky 17-20).
HAND_CONNECTIONS: tuple[tuple[int, int], ...] = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20),
)  # fmt: skip

_HAND_COLORS: dict[Handedness, tuple[int, int, int]] = {
    Handedness.LEFT: (255, 160, 60),  # BGR: blue-ish
    Handedness.RIGHT: (60, 200, 255),  # BGR: orange-ish
}
_FONT = cv2.FONT_HERSHEY_SIMPLEX


def _text(
    img: np.ndarray, text: str, org: tuple[int, int], scale: float, color: tuple[int, int, int]
) -> None:
    thickness = max(1, round(scale * 2))
    cv2.putText(img, text, org, _FONT, scale, (0, 0, 0), thickness + 3, cv2.LINE_AA)
    cv2.putText(img, text, org, _FONT, scale, color, thickness, cv2.LINE_AA)


def _boxed_text(
    img: np.ndarray, text: str, org: tuple[int, int], scale: float, color: tuple[int, int, int]
) -> None:
    """Small text on a dark box; outlines smear at small sizes."""
    (tw, th), base = cv2.getTextSize(text, _FONT, scale, 1)
    x, y = org
    cv2.rectangle(img, (x - 3, y - th - 3), (x + tw + 3, y + base), (20, 20, 20), -1)
    cv2.putText(img, text, org, _FONT, scale, color, 1, cv2.LINE_AA)


_FACE_COLOR = (200, 200, 120)
_ANCHOR_COLOR = (60, 255, 255)


class DebugRenderer:
    def draw_faces(
        self,
        frame_bgr: np.ndarray,
        faces_px: Sequence[np.ndarray],
        face_gestures: Sequence[FaceGesture] = (),
    ) -> None:
        """Face mesh dots, anchors, head tilt, and the expression match for each face."""
        h, w = frame_bgr.shape[:2]
        scale = max(0.5, w / 1600)
        for pts in faces_px:
            for x, y in pts[:468].astype(int):
                if 0 <= x < w and 0 <= y < h:
                    frame_bgr[y, x] = _FACE_COLOR
            for anchor in Anchor:
                p = place(pts, anchor)
                if p is None:
                    continue
                cx, cy = round(p.center[0]), round(p.center[1])
                cv2.circle(frame_bgr, (cx, cy), 5, _ANCHOR_COLOR, 2, cv2.LINE_AA)
                _boxed_text(frame_bgr, anchor.value, (cx + 8, cy + 4), scale * 0.7, _ANCHOR_COLOR)
            eyes = place(pts, Anchor.EYES)
            if eyes is not None:
                deg = np.degrees(eyes.angle)
                label = f"tilt {deg:+.0f} deg  eye-dist {eyes.scale:.0f}px"
                x = round(eyes.center[0] - eyes.scale)
                y = round(eyes.center[1] - eyes.scale * 1.6)
                _boxed_text(frame_bgr, label, (max(0, x), max(12, y)), scale * 0.8, _ANCHOR_COLOR)
        for fg in face_gestures:
            if fg.face_index >= len(faces_px):
                continue
            chin = place(faces_px[fg.face_index], Anchor.CHIN)
            if chin is None:
                continue
            x = max(0, round(chin.center[0] - chin.scale))
            y = round(chin.center[1] + chin.scale * 0.5)
            line = round(22 * scale)
            label = f"face: {fg.name} {fg.score:.2f} (custom)" if fg.name else "face: -"
            _boxed_text(frame_bgr, label, (x, y), scale * 0.9, _ANCHOR_COLOR)
            extra: list[str] = []
            if fg.match is not None and fg.match.nearest is not None:
                extra.append(f"nearest {fg.match.nearest} d={fg.match.distance:.2f}")
            if fg.scores is not None:
                extra.append("  ".join(f"{n} {v:.2f}" for n, v in top_blendshapes(fg.scores)))
            for k, text in enumerate(extra, start=1):
                _boxed_text(frame_bgr, text, (x, y + k * line), scale * 0.75, (230, 230, 230))

    def draw(
        self,
        frame_bgr: np.ndarray,
        hands: Sequence[HandGesture],
        stats: FrameStats,
        *,
        candidate: str | None,
        progress: float,
        cooldown: bool,
    ) -> None:
        h, w = frame_bgr.shape[:2]
        scale = max(0.5, w / 1600)

        for hg in hands:
            hand = hg.hand
            color = _HAND_COLORS[hand.handedness]
            pts = [(round(lm.x * w), round(lm.y * h)) for lm in hand.landmarks]
            for a, b in HAND_CONNECTIONS:
                if a < len(pts) and b < len(pts):
                    cv2.line(frame_bgr, pts[a], pts[b], color, 2, cv2.LINE_AA)
            for p in pts:
                cv2.circle(frame_bgr, p, 4, (255, 255, 255), -1, cv2.LINE_AA)
                cv2.circle(frame_bgr, p, 4, color, 1, cv2.LINE_AA)
            if not pts:
                continue
            label = f"{hand.handedness.value}: {hg.name or '-'}"
            if hg.name:
                label += f" {hg.score:.2f}"
                if hg.source is GestureSource.CUSTOM:
                    label += " (custom)"
            wx, wy = pts[0]
            y = min(h - 10 - round(28 * scale), wy + round(30 * scale))
            _text(frame_bgr, label, (wx - 40, y), scale, color)
            # Nearest custom template + distance: the number to compare against the threshold.
            if hg.custom is not None and hg.custom.nearest is not None:
                near = f"nearest {hg.custom.nearest} d={hg.custom.distance:.2f}"
                _boxed_text(
                    frame_bgr, near, (wx - 40, y + round(30 * scale)), scale * 0.8, (230, 230, 230)
                )

        line = round(34 * scale)
        _text(
            frame_bgr,
            f"FPS {stats.fps:5.1f}  infer {stats.inference_ms:4.1f} ms",
            (12, line),
            scale,
            (255, 255, 255),
        )
        status = f"holding {candidate} {progress:.0%}" if candidate else "no gesture"
        if cooldown:
            status += "  (cooldown)"
        _text(frame_bgr, status, (12, 2 * line), scale, (180, 255, 180))
