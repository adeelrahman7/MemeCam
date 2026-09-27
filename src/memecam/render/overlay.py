"""Corner meme overlay: loads RGBA PNGs with Pillow and alpha-blends them onto BGR frames."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from memecam.config import Corner


def load_bgra(path: Path) -> np.ndarray:
    """Decode any Pillow-readable image into a uint8 BGRA array (alpha preserved)."""
    with Image.open(path) as img:
        rgba = np.asarray(img.convert("RGBA"))
    return cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA)


def alpha_blend(dst_bgr: np.ndarray, src_bgra: np.ndarray, x: int, y: int) -> None:
    """Blend ``src_bgra`` onto ``dst_bgr`` in place with its top-left at (x, y). Clips at edges."""
    h, w = src_bgra.shape[:2]
    dh, dw = dst_bgr.shape[:2]
    x0, y0 = max(x, 0), max(y, 0)
    x1, y1 = min(x + w, dw), min(y + h, dh)
    if x0 >= x1 or y0 >= y1:
        return
    src = src_bgra[y0 - y : y1 - y, x0 - x : x1 - x]
    alpha = src[:, :, 3:4].astype(np.float32) / 255.0
    roi = dst_bgr[y0:y1, x0:x1].astype(np.float32)
    blended = src[:, :, :3].astype(np.float32) * alpha + roi * (1.0 - alpha)
    dst_bgr[y0:y1, x0:x1] = blended.astype(np.uint8)


class CornerOverlay:
    """Draws the currently active reaction image in a frame corner.

    Images are decoded once up front; resized copies are cached per target width so the
    per-frame cost is just the blend. ``corners`` can give individual reactions their own
    corner; everything else uses ``corner``.
    """

    def __init__(
        self,
        images: dict[str, Path],
        *,
        corner: Corner,
        corners: dict[str, Corner] | None = None,
        width_fraction: float,
        margin_px: int,
        display_seconds: float,
    ) -> None:
        self._sources = {key: load_bgra(path) for key, path in images.items()}
        self._corner = corner
        self._corners = dict(corners or {})
        self._width_fraction = width_fraction
        self._margin = margin_px
        self._display_seconds = display_seconds
        self._scaled: dict[tuple[str, int], np.ndarray] = {}
        self._active: str | None = None
        self._active_until = 0.0

    def trigger(self, key: str, now: float) -> None:
        if key not in self._sources:
            raise KeyError(f"no overlay image loaded for reaction {key!r}")
        self._active = key
        self._active_until = now + self._display_seconds

    def hold(self, key: str, now: float, linger_seconds: float) -> None:
        """Keep the current meme up while its gesture is still held.

        No-op unless ``key`` is the meme already showing; it never starts a new one.
        After the gesture is released the meme stays ``linger_seconds`` longer (or until
        its normal ``display_seconds`` end, whichever is later).
        """
        if self._active == key:
            self._active_until = max(self._active_until, now + linger_seconds)

    def active(self, now: float) -> str | None:
        if self._active is not None and now >= self._active_until:
            self._active = None
        return self._active

    def draw(self, frame_bgr: np.ndarray, now: float) -> None:
        key = self.active(now)
        if key is None:
            return
        img = self._scaled_image(key, frame_bgr.shape[1])
        fh, fw = frame_bgr.shape[:2]
        ih, iw = img.shape[:2]
        corner = self._corners.get(key, self._corner)
        left = corner in (Corner.TOP_LEFT, Corner.BOTTOM_LEFT)
        top = corner in (Corner.TOP_LEFT, Corner.TOP_RIGHT)
        x = self._margin if left else fw - iw - self._margin
        y = self._margin if top else fh - ih - self._margin
        alpha_blend(frame_bgr, img, x, y)

    def _scaled_image(self, key: str, frame_width: int) -> np.ndarray:
        target_w = max(1, round(frame_width * self._width_fraction))
        cached = self._scaled.get((key, target_w))
        if cached is None:
            src = self._sources[key]
            scale = target_w / src.shape[1]
            target_h = max(1, round(src.shape[0] * scale))
            interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
            cached = cv2.resize(src, (target_w, target_h), interpolation=interp)
            self._scaled[(key, target_w)] = cached
        return cached
