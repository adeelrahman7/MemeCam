"""Where a gesture's image shows up: a screen corner, or a spot on your face."""

from __future__ import annotations

from dataclasses import dataclass

from memecam.config import Corner
from memecam.face.anchors import Anchor

# Sensible starting sizes/offsets per face spot. width = multiple of your eye distance;
# offset_y shifts by the image's own height (-0.5 = sit on top of the anchor, like a hat).
FACE_DEFAULTS: dict[Anchor, tuple[float, float]] = {
    Anchor.EYES: (1.4, 0.0),
    Anchor.FOREHEAD: (1.3, -0.45),
    Anchor.NOSE: (0.9, 0.0),
    Anchor.MOUTH: (1.1, 0.0),
    Anchor.CHIN: (1.2, 0.3),
    Anchor.FACE: (2.0, 0.0),
}

_CORNER_NAMES = {
    Corner.TOP_LEFT: "top left",
    Corner.TOP_RIGHT: "top right",
    Corner.BOTTOM_LEFT: "bottom left",
    Corner.BOTTOM_RIGHT: "bottom right",
}
_ANCHOR_NAMES = {
    Anchor.EYES: "eyes",
    Anchor.FOREHEAD: "forehead / head",
    Anchor.NOSE: "nose",
    Anchor.MOUTH: "mouth",
    Anchor.CHIN: "chin",
    Anchor.FACE: "whole face",
}


@dataclass(frozen=True, slots=True)
class Placement:
    """Exactly one of: a corner meme (``anchor`` is None) or a face overlay (``anchor`` set).

    For a corner meme, ``corner=None`` means "the default corner from the config".
    For a face overlay, ``width=None`` means "the default size for that spot".
    """

    corner: Corner | None = None
    anchor: Anchor | None = None
    width: float | None = None

    @classmethod
    def in_corner(cls, corner: Corner | None = None) -> Placement:
        return cls(corner=corner)

    @classmethod
    def on_face(cls, anchor: Anchor, width: float | None = None) -> Placement:
        return cls(anchor=anchor, width=width)

    @property
    def is_face(self) -> bool:
        return self.anchor is not None

    def face_width(self) -> float:
        assert self.anchor is not None
        return self.width if self.width is not None else FACE_DEFAULTS[self.anchor][0]

    def label(self, default_corner: Corner) -> str:
        """Short human description, e.g. 'bottom-left corner' or 'on face: eyes'."""
        if self.anchor is not None:
            return f"on face: {_ANCHOR_NAMES[self.anchor]}"
        corner = _CORNER_NAMES[self.corner or default_corner].replace(" ", "-")
        return f"{corner} corner" + (" (default)" if self.corner is None else "")


def placement_choices(default_corner: Corner) -> list[tuple[str, Placement]]:
    """Menu entries for the placement picker, in display order."""
    choices = [(f"Corner: default ({_CORNER_NAMES[default_corner]})", Placement.in_corner())]
    choices += [
        (f"Corner: {name}", Placement.in_corner(c))
        for c, name in _CORNER_NAMES.items()
        if c is not default_corner
    ]
    choices += [(f"On face: {name}", Placement.on_face(a)) for a, name in _ANCHOR_NAMES.items()]
    return choices
