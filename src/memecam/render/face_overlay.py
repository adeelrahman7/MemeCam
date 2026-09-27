"""Draws images (sunglasses, crowns, ...) attached to a face: scaled, rotated and offset."""

from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np

from memecam.face.anchors import Placement
from memecam.render.overlay import load_bgra

_SIZE_BUCKET_PX = 8  # cache pre-shrunk images in 8px width steps


def _premultiply(bgra: np.ndarray) -> np.ndarray:
    """Premultiplied alpha avoids dark fringes when the image is rotated and resampled."""
    out = bgra.astype(np.float32)
    out[:, :, :3] *= out[:, :, 3:4] / 255.0
    return out


def blend_premultiplied(dst_bgr: np.ndarray, src: np.ndarray, x: int, y: int) -> None:
    """Composite a premultiplied float BGRA patch onto ``dst_bgr`` at (x, y), in place.

    The caller guarantees the patch lies inside the frame.
    """
    h, w = src.shape[:2]
    roi = dst_bgr[y : y + h, x : x + w].astype(np.float32)
    alpha = src[:, :, 3:4] / 255.0
    out = src[:, :, :3] + roi * (1.0 - alpha)
    dst_bgr[y : y + h, x : x + w] = np.clip(np.rint(out), 0, 255).astype(np.uint8)


class FaceOverlayRenderer:
    """Holds decoded overlay images and draws them at a face Placement."""

    def __init__(self, images: dict[str, Path]) -> None:
        self._sources = {key: _premultiply(load_bgra(path)) for key, path in images.items()}
        self._shrunk: dict[tuple[str, int], np.ndarray] = {}

    def _source_for_width(self, key: str, target_w: float) -> np.ndarray:
        """A copy of the image already close to ``target_w`` wide.

        warpAffine alone aliases badly when shrinking a lot; INTER_AREA first fixes that.
        """
        src = self._sources[key]
        if target_w >= src.shape[1] * 0.75:
            return src
        bucket = max(1, math.ceil(target_w / _SIZE_BUCKET_PX))
        cached = self._shrunk.get((key, bucket))
        if cached is None:
            w = bucket * _SIZE_BUCKET_PX
            h = max(1, round(src.shape[0] * w / src.shape[1]))
            cached = cv2.resize(src, (w, h), interpolation=cv2.INTER_AREA)
            self._shrunk[(key, bucket)] = cached
        return cached

    def draw(
        self,
        frame_bgr: np.ndarray,
        key: str,
        placement: Placement,
        *,
        width: float,
        offset_x: float = 0.0,
        offset_y: float = 0.0,
    ) -> None:
        """Draw image ``key`` centered on ``placement``.

        - ``width``: image width as a multiple of ``placement.scale`` (eye distance).
        - ``offset_x/offset_y``: shift in units of the drawn image's own width/height,
          along the head's axes (so they tilt with the head). ``offset_y=-0.5`` puts the
          image's bottom edge on the anchor, e.g. a hat sitting on the forehead.
        """
        target_w = placement.scale * width
        if target_w < 2:
            return
        img = self._source_for_width(key, target_w)
        ih, iw = img.shape[:2]
        k = target_w / iw
        target_h = ih * k
        c, s = math.cos(placement.angle), math.sin(placement.angle)

        ox, oy = offset_x * target_w, offset_y * target_h
        cx = placement.center[0] + c * ox - s * oy
        cy = placement.center[1] + s * ox + c * oy

        # Image → frame: scale by k, rotate by angle, put the image center at (cx, cy).
        m = np.array([[k * c, -k * s, 0.0], [k * s, k * c, 0.0]])
        m[:, 2] = (cx, cy) - m[:, :2] @ np.array([iw / 2, ih / 2])

        corners = np.array([[0, 0, 1], [iw, 0, 1], [0, ih, 1], [iw, ih, 1]], dtype=np.float64)
        pts = corners @ m.T
        fh, fw = frame_bgr.shape[:2]
        x0 = max(0, math.floor(pts[:, 0].min()))
        y0 = max(0, math.floor(pts[:, 1].min()))
        x1 = min(fw, math.ceil(pts[:, 0].max()))
        y1 = min(fh, math.ceil(pts[:, 1].max()))
        if x0 >= x1 or y0 >= y1:
            return  # entirely off-screen

        m[:, 2] -= (x0, y0)
        patch = cv2.warpAffine(
            img, m, (x1 - x0, y1 - y0), flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0),
        )  # fmt: skip
        blend_premultiplied(frame_bgr, patch, x0, y0)
