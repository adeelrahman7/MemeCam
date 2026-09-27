"""reactions.json schema (pydantic) and loader with human-readable errors."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
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
from memecam.gestures.library import (
    CUSTOM_NAME_PATTERN,
    GestureLibrary,
    LibraryError,
    load_library,
)
from memecam.paths import resolve_resource, resource_root

BUILTIN_GESTURES = frozenset(g.value for g in BuiltinGesture)


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
    # Flip if the debug overlay shows "Left" on your right hand (camera-dependent).
    swap_handedness: bool = False


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
    # Minimum time a meme is shown after it fires.
    display_seconds: Annotated[float, Field(gt=0.0, le=60.0)] = 2.0
    # true: the meme stays up as long as you keep holding the gesture.
    stay_while_held: bool = True
    # With stay_while_held, how long the meme remains after you let go.
    linger_seconds: Annotated[float, Field(ge=0.0, le=10.0)] = 0.5


class CustomGesturesConfig(_Strict):
    file: str = "config/custom_gestures.json"
    # Max RMS landmark distance (in palm lengths) that still counts as a match.
    # Lower = stricter. The debug overlay shows live distances to help tune this.
    threshold: Annotated[float, Field(gt=0.0, le=2.0)] = 0.2
    # true: a tilted hand still matches. false: orientation matters (thumb up != thumb down).
    rotation_invariant: bool = False
    # Averages the hand shape over recent frames to remove jitter (0 = off).
    smoothing: Annotated[float, Field(ge=0.0, lt=1.0)] = 0.5
    # A match is kept until distance exceeds threshold * release_factor (prevents flicker).
    release_factor: Annotated[float, Field(ge=1.0, le=2.0)] = 1.3
    countdown_seconds: Annotated[float, Field(ge=0.0, le=10.0)] = 3.0
    capture_frames: Annotated[int, Field(ge=5, le=300)] = 45


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
        # Custom names are checked against custom_gestures.json in load_settings().
        if value not in BUILTIN_GESTURES and not re.fullmatch(CUSTOM_NAME_PATTERN, value):
            raise ValueError(
                f"unknown gesture {value!r}; built-in gestures are {sorted(BUILTIN_GESTURES)}, "
                "and custom gesture names are lowercase_with_underscores"
            )
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
    custom_gestures: CustomGesturesConfig = CustomGesturesConfig()
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


def config_to_json(config: AppConfig) -> str:
    """Serialize for writing back to reactions.json, keeping only keys the file had or we set."""
    return json.dumps(config.model_dump(mode="json", exclude_unset=True), indent=2) + "\n"


@dataclass(frozen=True, slots=True)
class Settings:
    """Everything loaded from disk at startup: the validated config plus custom gestures."""

    config: AppConfig
    library: GestureLibrary
    config_path: Path
    root: Path

    @property
    def library_path(self) -> Path:
        return resolve_resource(self.config.custom_gestures.file, self.root)


def load_settings(config_path: Path, root: Path | None = None) -> Settings:
    """Load reactions.json and custom_gestures.json and check they agree."""
    root = root or resource_root()
    config = load_config(config_path, root)
    library_path = resolve_resource(config.custom_gestures.file, root)
    try:
        library = load_library(library_path)
    except LibraryError as exc:
        raise ConfigError(str(exc)) from exc

    problems = [
        f"  • reactions[{i}].gesture: no recorded custom gesture named {r.gesture!r} "
        f"(record it in the app, or check {library_path.name})"
        for i, r in enumerate(config.reactions)
        if r.gesture not in BUILTIN_GESTURES and r.gesture not in library
    ]
    if problems:
        raise ConfigError(
            "\n".join([f"{config_path.name} has {len(problems)} problem(s):", *problems])
        )
    return Settings(config, library, config_path, root)
