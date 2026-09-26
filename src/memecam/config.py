"""reactions.json schema (pydantic) and loader with human-readable errors."""

from __future__ import annotations

import json
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    ValidationInfo,
    field_validator,
    model_validator,
)

from memecam.core.types import BuiltinGesture, Handedness
from memecam.paths import resolve_resource, resource_root


class ConfigError(Exception):
    """Raised when reactions.json is missing, not JSON, or fails validation."""


class _Strict(BaseModel):
    # extra="forbid" turns typos like "hold_frame" into errors instead of silent defaults.
    model_config = ConfigDict(extra="forbid", frozen=True)


class Corner(StrEnum):
    TOP_LEFT = "top_left"
    TOP_RIGHT = "top_right"
    BOTTOM_LEFT = "bottom_left"
    BOTTOM_RIGHT = "bottom_right"


class HandFilter(StrEnum):
    ANY = "any"
    LEFT = "left"
    RIGHT = "right"

    def matches(self, hand: Handedness) -> bool:
        return self is HandFilter.ANY or self.value == hand.value.lower()


class CameraConfig(_Strict):
    index: Annotated[int, Field(ge=0)] = 0
    width: Annotated[int, Field(ge=160, le=3840)] = 1280
    height: Annotated[int, Field(ge=120, le=2160)] = 720
    mirror: bool = True


class InferenceConfig(_Strict):
    model: str = "models/gesture_recognizer.task"
    max_width: Annotated[int, Field(ge=160, le=1920)] = 640
    num_hands: Annotated[int, Field(ge=1, le=4)] = 2
    min_detection_confidence: Annotated[float, Field(ge=0.0, le=1.0)] = 0.5
    min_gesture_score: Annotated[float, Field(ge=0.0, le=1.0)] = 0.6


class DebounceConfig(_Strict):
    hold_frames: Annotated[int, Field(ge=1, le=120)] = 6
    cooldown_seconds: Annotated[float, Field(ge=0.0, le=60.0)] = 1.5


class OverlayConfig(_Strict):
    corner: Corner = Corner.TOP_RIGHT
    width_fraction: Annotated[float, Field(gt=0.0, le=1.0)] = 0.3
    margin_px: Annotated[int, Field(ge=0, le=500)] = 16
    display_seconds: Annotated[float, Field(gt=0.0, le=60.0)] = 2.0


class Reaction(_Strict):
    gesture: str
    image: str
    hand: HandFilter = HandFilter.ANY

    @property
    def key(self) -> str:
        """Stable identifier fed to the debouncer, e.g. 'Thumb_Up' or 'Victory:left'."""
        return self.gesture if self.hand is HandFilter.ANY else f"{self.gesture}:{self.hand}"

    @field_validator("gesture")
    @classmethod
    def _known_gesture(cls, value: str) -> str:
        known = {g.value for g in BuiltinGesture}
        if value not in known:
            raise ValueError(f"unknown gesture {value!r}; expected one of {sorted(known)}")
        return value

    @field_validator("image")
    @classmethod
    def _image_exists(cls, value: str, info: ValidationInfo) -> str:
        root = (info.context or {}).get("root")
        if root is None:
            return value
        path = resolve_resource(value, Path(root))
        if not path.is_file():
            raise ValueError(f"image file not found: {path}")
        if path.suffix.lower() != ".png":
            raise ValueError(f"only .png reactions are supported for now, got {path.name}")
        return value


class AppConfig(_Strict):
    camera: CameraConfig = CameraConfig()
    inference: InferenceConfig = InferenceConfig()
    debounce: DebounceConfig = DebounceConfig()
    overlay: OverlayConfig = OverlayConfig()
    reactions: Annotated[list[Reaction], Field(min_length=1)]

    @model_validator(mode="after")
    def _no_conflicting_reactions(self) -> Self:
        seen: dict[str, set[HandFilter]] = {}
        for reaction in self.reactions:
            hands = seen.setdefault(reaction.gesture, set())
            if reaction.hand in hands:
                raise ValueError(
                    f"duplicate reaction for gesture {reaction.gesture!r} (hand={reaction.hand})"
                )
            if HandFilter.ANY in hands or (hands and reaction.hand is HandFilter.ANY):
                raise ValueError(
                    f"gesture {reaction.gesture!r} has both a hand='any' reaction and a "
                    "hand-specific one; use either 'any' or left/right"
                )
            hands.add(reaction.hand)
        return self

    def find_reaction(self, gesture: str, hand: Handedness) -> Reaction | None:
        """Return the reaction for this gesture/hand, if one is configured."""
        for reaction in self.reactions:
            if reaction.gesture == gesture and reaction.hand.matches(hand):
                return reaction
        return None


def _format_loc(loc: tuple[int | str, ...]) -> str:
    out = ""
    for part in loc:
        out += f"[{part}]" if isinstance(part, int) else (f".{part}" if out else str(part))
    return out or "(root)"


def load_config(path: Path, root: Path | None = None) -> AppConfig:
    """Load and validate reactions.json.

    ``root`` is the directory that relative image/model paths are resolved against; it defaults
    to the resource root (project dir in dev, sys._MEIPASS when bundled).
    """
    root = root or resource_root()
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise ConfigError(f"Config file not found: {path}") from exc
    except OSError as exc:
        raise ConfigError(f"Could not read config file {path}: {exc}") from exc

    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ConfigError(
            f"{path.name} is not valid JSON (line {exc.lineno}, column {exc.colno}): {exc.msg}"
        ) from exc

    try:
        return AppConfig.model_validate(data, context={"root": root})
    except ValidationError as exc:
        lines = [f"{path.name} has {exc.error_count()} problem(s):"]
        for err in exc.errors(include_url=False):
            msg = err["msg"].removeprefix("Value error, ")
            lines.append(f"  • {_format_loc(tuple(err['loc']))}: {msg}")
        raise ConfigError("\n".join(lines)) from exc
