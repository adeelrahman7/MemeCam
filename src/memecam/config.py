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
from memecam.face.anchors import Anchor
from memecam.gestures.library import (
    CUSTOM_NAME_PATTERN,
    GestureLibrary,
    LibraryError,
    load_library,
)
from memecam.gestures.templates import GestureKind
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


def _check_png(value: str, info: ValidationInfo) -> str:
    """Validate an image path from the config (only when a resource root is provided)."""
    root = (info.context or {}).get("root")
    if root is None:
        return value
    path = resolve_resource(value, Path(root))
    if not path.is_file():
        raise ValueError(f"image file not found: {path}")
    if path.suffix.lower() != ".png":
        raise ValueError(f"only .png images are supported for now, got {path.name}")
    return value


def _check_gesture_name(value: str) -> str:
    # Custom names are checked against custom_gestures.json in load_settings().
    if value not in BUILTIN_GESTURES and not re.fullmatch(CUSTOM_NAME_PATTERN, value):
        raise ValueError(
            f"unknown gesture {value!r}; built-in gestures are {sorted(BUILTIN_GESTURES)}, "
            "and custom gesture names are lowercase_with_underscores"
        )
    return value


class CustomGesturesConfig(_Strict):
    file: str = "config/custom_gestures.json"
    # Max RMS landmark distance (in palm lengths) that still counts as a match.
    # Lower = stricter. The debug overlay shows live distances to help tune this.
    threshold: Annotated[float, Field(gt=0.0, le=2.0)] = 0.2
    # Same idea for recorded face expressions: max distance between expression scores.
    # Roughly: noise ~0.1, a big smile vs a resting face ~1.0.
    face_threshold: Annotated[float, Field(gt=0.0, le=3.0)] = 0.4
    # true: a tilted hand still matches. false: orientation matters (thumb up != thumb down).
    rotation_invariant: bool = False
    # Averages the hand shape / expression over recent frames to remove jitter (0 = off).
    smoothing: Annotated[float, Field(ge=0.0, lt=1.0)] = 0.5
    # A match is kept until distance exceeds threshold * release_factor (prevents flicker).
    release_factor: Annotated[float, Field(ge=1.0, le=2.0)] = 1.3
    countdown_seconds: Annotated[float, Field(ge=0.0, le=10.0)] = 3.0
    capture_frames: Annotated[int, Field(ge=5, le=300)] = 45


class Reaction(_Strict):
    gesture: str
    image: str
    hand: HandFilter = HandFilter.ANY
    # Which corner this meme appears in; omitted = overlay.corner (the default corner).
    corner: Corner | None = None

    @property
    def key(self) -> str:
        """Stable identifier fed to the debouncer, e.g. 'Thumb_Up' or 'Victory:left'."""
        return self.gesture if self.hand is HandFilter.ANY else f"{self.gesture}:{self.hand}"

    @field_validator("gesture")
    @classmethod
    def _known_gesture(cls, value: str) -> str:
        return _check_gesture_name(value)

    @field_validator("image")
    @classmethod
    def _image_exists(cls, value: str, info: ValidationInfo) -> str:
        return _check_png(value, info)


class FaceConfig(_Strict):
    model: str = "models/face_landmarker.task"
    num_faces: Annotated[int, Field(ge=1, le=4)] = 1
    min_confidence: Annotated[float, Field(ge=0.0, le=1.0)] = 0.5
    # Averages face landmarks over recent frames so overlays don't jitter (0 = off).
    smoothing: Annotated[float, Field(ge=0.0, lt=1.0)] = 0.5


class FaceOverlay(_Strict):
    """An image attached to every detected face, e.g. sunglasses on the eyes."""

    image: str
    anchor: Anchor
    # Width as a multiple of the distance between your outer eye corners.
    width: Annotated[float, Field(gt=0.0, le=10.0)] = 1.5
    # Shift in units of the image's own width/height, along the head's axes.
    # offset_y = -0.5 sits the image's bottom edge on the anchor (hats, crowns).
    offset_x: Annotated[float, Field(ge=-5.0, le=5.0)] = 0.0
    offset_y: Annotated[float, Field(ge=-5.0, le=5.0)] = 0.0
    # null = always on; otherwise shown while this gesture is held (any hand).
    trigger: str | None = None
    # With a trigger: how long it stays after you drop the gesture.
    linger_seconds: Annotated[float, Field(ge=0.0, le=30.0)] = 0.5

    @field_validator("image")
    @classmethod
    def _image_exists(cls, value: str, info: ValidationInfo) -> str:
        return _check_png(value, info)

    @field_validator("trigger")
    @classmethod
    def _known_trigger(cls, value: str | None) -> str | None:
        return None if value is None else _check_gesture_name(value)


class AppConfig(_Strict):
    camera: CameraConfig = CameraConfig()
    inference: InferenceConfig = InferenceConfig()
    debounce: DebounceConfig = DebounceConfig()
    overlay: OverlayConfig = OverlayConfig()
    custom_gestures: CustomGesturesConfig = CustomGesturesConfig()
    face: FaceConfig = FaceConfig()
    reactions: list[Reaction] = []
    face_overlays: list[FaceOverlay] = []

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

    def find_reaction(self, gesture: str, hand: Handedness | None) -> Reaction | None:
        """Return the reaction for this gesture/hand, if one is configured.

        ``hand`` is None for face expressions, which only match hand="any" reactions.
        """
        for reaction in self.reactions:
            if reaction.gesture != gesture:
                continue
            if hand is None:
                if reaction.hand is HandFilter.ANY:
                    return reaction
            elif reaction.hand.matches(hand):
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

    @property
    def needs_face_tracking(self) -> bool:
        """The face model must run for face overlays or recorded face expressions."""
        return bool(self.config.face_overlays or self.library.of_kind(GestureKind.FACE))


def load_settings(config_path: Path, root: Path | None = None) -> Settings:
    """Load reactions.json and custom_gestures.json and check they agree."""
    root = root or resource_root()
    config = load_config(config_path, root)
    library_path = resolve_resource(config.custom_gestures.file, root)
    try:
        library = load_library(library_path)
    except LibraryError as exc:
        raise ConfigError(str(exc)) from exc

    def missing(name: str | None) -> bool:
        return name is not None and name not in BUILTIN_GESTURES and name not in library

    hint = f"(record it in the app, or check {library_path.name})"
    problems = (
        [
            f"  • reactions[{i}].gesture: no recorded custom gesture named {r.gesture!r} {hint}"
            for i, r in enumerate(config.reactions)
            if missing(r.gesture)
        ]
        + [
            f"  • face_overlays[{i}].trigger: no recorded custom gesture named {o.trigger!r} {hint}"
            for i, o in enumerate(config.face_overlays)
            if missing(o.trigger)
        ]
        + [
            f"  • reactions[{i}].hand: {r.gesture!r} is a face expression, so a left/right hand "
            'filter doesn\'t apply; use "any" or remove "hand"'
            for i, r in enumerate(config.reactions)
            if r.hand is not HandFilter.ANY and library.kind_of(r.gesture) is GestureKind.FACE
        ]
    )
    if problems:
        raise ConfigError(
            "\n".join([f"{config_path.name} has {len(problems)} problem(s):", *problems])
        )
    return Settings(config, library, config_path, root)
