"""Face overlays for one frame: which are switched on, and where they go on each face."""

from __future__ import annotations

from collections.abc import Collection, Sequence
from pathlib import Path

import numpy as np

from memecam.config import FaceConfig, FaceOverlay
from memecam.core.types import FaceFrame
from memecam.face.anchors import FaceSmoother, place, to_pixels
from memecam.gestures.activator import HoldActivator
from memecam.paths import resolve_resource
from memecam.render.face_overlay import FaceOverlayRenderer


class FaceOverlayLayer:
    def __init__(
        self,
        overlays: Sequence[FaceOverlay],
        face_config: FaceConfig,
        *,
        root: Path,
        hold_frames: int,
    ) -> None:
        self._overlays = list(overlays)
        self._renderer = FaceOverlayRenderer(
            {self._key(i): resolve_resource(o.image, root) for i, o in enumerate(overlays)}
        )
        self._activators: list[HoldActivator | None] = [
            None if o.trigger is None else HoldActivator(hold_frames, o.linger_seconds)
            for o in overlays
        ]
        self._smoother = FaceSmoother(face_config.smoothing)
        self._faces_px: list[np.ndarray] = []
        self._active: list[bool] = [False] * len(overlays)

    @staticmethod
    def _key(index: int) -> str:
        return f"face_overlay_{index}"

    @property
    def faces_px(self) -> list[np.ndarray]:
        """Smoothed pixel landmarks of each face from the last update()."""
        return self._faces_px

    @property
    def active(self) -> list[bool]:
        return list(self._active)

    def update(
        self,
        faces: FaceFrame,
        gestures_present: Collection[str],
        *,
        frame_size: tuple[int, int],
        now: float,
        allow_triggers: bool = True,
    ) -> None:
        """Advance one frame. ``gestures_present``: gesture names on any hand right now."""
        w, h = frame_size
        self._faces_px = self._smoother.update([to_pixels(f.landmarks, w, h) for f in faces.faces])
        for i, (overlay, activator) in enumerate(
            zip(self._overlays, self._activators, strict=True)
        ):
            if activator is None:
                self._active[i] = True
            else:
                held = allow_triggers and overlay.trigger in gestures_present
                self._active[i] = activator.update(held, now)

    def draw(self, frame_bgr: np.ndarray) -> None:
        for face in self._faces_px:
            for i, overlay in enumerate(self._overlays):
                if not self._active[i]:
                    continue
                placement = place(face, overlay.anchor)
                if placement is None:
                    continue
                self._renderer.draw(
                    frame_bgr,
                    self._key(i),
                    placement,
                    width=overlay.width,
                    offset_x=overlay.offset_x,
                    offset_y=overlay.offset_y,
                )
