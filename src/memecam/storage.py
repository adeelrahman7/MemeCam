"""Saving and deleting gestures: templates, reaction images and reactions.json entries."""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from pydantic import ValidationError

from memecam.config import (
    BUILTIN_GESTURES,
    AppConfig,
    ConfigError,
    Corner,
    FaceOverlay,
    HandFilter,
    Reaction,
    Settings,
    config_to_json,
    load_settings,
)
from memecam.gestures.library import atomic_write_text, save_library
from memecam.gestures.templates import GestureKind, GestureTemplate
from memecam.placement import FACE_DEFAULTS, Placement

MEMES_DIR = Path("assets") / "memes"


def _image_config_path(image: Path, name: str, root: Path) -> str:
    """Copy ``image`` into assets/memes/ (unless it's already inside the project).

    Returns the path to store in reactions.json: relative to the root when possible, so the
    config keeps working if the project folder moves or gets bundled.
    """
    image = image.resolve()
    if image.suffix.lower() != ".png":
        raise ConfigError(f"Reaction images must be .png for now (got {image.name}).")
    try:
        return image.relative_to(root.resolve()).as_posix()
    except ValueError:
        pass
    dest = root / MEMES_DIR / f"{name}.png"
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(image, dest)
    return (MEMES_DIR / dest.name).as_posix()


def _entry_for(
    gesture: str,
    image: str,
    placement: Placement,
    previous: Reaction | FaceOverlay | None = None,
) -> Reaction | FaceOverlay:
    """Build the config entry that shows ``image`` for ``gesture`` at ``placement``.

    Carries over what still makes sense from ``previous``: a reaction's hand filter when it
    stays a corner meme; an overlay's offsets when it stays on the same face spot; and an
    overlay's linger time whenever it stays on the face.
    """
    if placement.anchor is None:
        hand = previous.hand if isinstance(previous, Reaction) else HandFilter.ANY
        if placement.corner is None:
            return Reaction(gesture=gesture, image=image, hand=hand)
        return Reaction(gesture=gesture, image=image, hand=hand, corner=placement.corner)

    width, offset_y = FACE_DEFAULTS[placement.anchor]
    offset_x, linger = 0.0, 0.5
    if isinstance(previous, FaceOverlay):
        linger = previous.linger_seconds
        if previous.anchor is placement.anchor:
            width, offset_x, offset_y = previous.width, previous.offset_x, previous.offset_y
    if placement.width is not None:
        width = placement.width
    return FaceOverlay(
        image=image,
        anchor=placement.anchor,
        width=width,
        offset_x=offset_x,
        offset_y=offset_y,
        trigger=gesture,
        linger_seconds=linger,
    )


def _with_entry(
    config: AppConfig,
    entry: Reaction | FaceOverlay,
    *,
    reaction_at: int | None = None,
    overlay_at: int | None = None,
) -> AppConfig:
    """Insert ``entry`` into the right list (at a position, or at the end)."""
    reactions, overlays = list(config.reactions), list(config.face_overlays)
    if isinstance(entry, Reaction):
        reactions.insert(len(reactions) if reaction_at is None else reaction_at, entry)
    else:
        overlays.insert(len(overlays) if overlay_at is None else overlay_at, entry)
    return config.model_copy(update={"reactions": reactions, "face_overlays": overlays})


def _validate(config: AppConfig, root: Path) -> None:
    """Check the edited config before touching any file on disk."""
    try:
        AppConfig.model_validate(json.loads(config_to_json(config)), context={"root": root})
    except ValidationError as exc:
        messages = "; ".join(e["msg"].removeprefix("Value error, ") for e in exc.errors())
        raise ConfigError(f"That change would make the config invalid: {messages}") from exc


def save_recorded_gesture(
    settings: Settings,
    template: GestureTemplate,
    image: Path | None,
    placement: Placement | None = None,
) -> Settings:
    """Save ``template`` and (optionally) show ``image`` at ``placement``.

    Re-recording an existing name replaces its template. Choosing an image replaces
    everything the gesture showed before (its corner memes and face overlays) with that one
    image. ``placement`` defaults to the default corner. Returns freshly loaded settings.
    """
    config = settings.config
    if image is not None:
        rel = _image_config_path(image, template.name, settings.root)
        config = config.model_copy(
            update={
                "reactions": [r for r in config.reactions if r.gesture != template.name],
                "face_overlays": [o for o in config.face_overlays if o.trigger != template.name],
            }
        )
        config = _with_entry(
            config, _entry_for(template.name, rel, placement or Placement.in_corner())
        )
        _validate(config, settings.root)

    save_library(settings.library.with_template(template), settings.library_path)
    if image is not None:
        atomic_write_text(settings.config_path, config_to_json(config))
    return load_settings(settings.config_path, settings.root)


@dataclass(frozen=True, slots=True)
class ReactionItem:
    """One image a gesture shows: a corner meme or a face overlay."""

    gesture: str
    image: str
    placement: Placement
    reaction_index: int | None = None  # position in config.reactions, if a corner meme
    overlay_index: int | None = None  # position in config.face_overlays, if on the face
    hand: HandFilter = HandFilter.ANY  # corner memes only

    def describe(self, default_corner: Corner) -> str:
        name = PurePosixPath(self.image).name
        hand = "" if self.hand is HandFilter.ANY else f", {self.hand.value} hand only"
        return f"{name} · {self.placement.label(default_corner)}{hand}"


def reaction_items(settings: Settings, gesture: str) -> list[ReactionItem]:
    """Every image ``gesture`` shows: corner memes first, then face overlays."""
    config = settings.config
    items = [
        ReactionItem(gesture, r.image, Placement.in_corner(r.corner), reaction_index=i, hand=r.hand)
        for i, r in enumerate(config.reactions)
        if r.gesture == gesture
    ]
    items += [
        ReactionItem(gesture, o.image, Placement.on_face(o.anchor, o.width), overlay_index=i)
        for i, o in enumerate(config.face_overlays)
        if o.trigger == gesture
    ]
    return items


def set_placement(settings: Settings, item: ReactionItem, placement: Placement) -> Settings:
    """Move one of a gesture's images to a new corner or face spot; keeps the image.

    Returns freshly loaded settings. Raises ConfigError if the move isn't possible, e.g. a
    gesture can only have one "any hand" corner meme.
    """
    config = settings.config
    reactions, overlays = list(config.reactions), list(config.face_overlays)
    previous: Reaction | FaceOverlay
    if item.reaction_index is not None:
        previous = reactions.pop(item.reaction_index)
    elif item.overlay_index is not None:
        previous = overlays.pop(item.overlay_index)
    else:
        raise ConfigError("Nothing to move.")

    entry = _entry_for(item.gesture, item.image, placement, previous)
    if isinstance(entry, Reaction) and any(
        r.gesture == item.gesture
        and (r.hand is entry.hand or HandFilter.ANY in (r.hand, entry.hand))
        for r in reactions
    ):
        raise ConfigError(
            f"'{item.gesture}' already shows a corner meme. A gesture can have one corner "
            "meme (per hand); pick a spot on the face instead, or delete the other one."
        )
    config = config.model_copy(update={"reactions": reactions, "face_overlays": overlays})
    # Stay in the same slot when the type doesn't change, so the file order is stable.
    config = _with_entry(
        config,
        entry,
        reaction_at=item.reaction_index if isinstance(previous, Reaction) else None,
        overlay_at=item.overlay_index if isinstance(previous, FaceOverlay) else None,
    )
    _validate(config, settings.root)
    atomic_write_text(settings.config_path, config_to_json(config))
    return load_settings(settings.config_path, settings.root)


@dataclass(frozen=True, slots=True)
class GestureUsage:
    """One gesture and everything it's wired to (for the Manage gestures window)."""

    name: str
    custom: bool  # recorded by you (deletable pose) vs built into MediaPipe's model
    kind: GestureKind
    reactions: tuple[Reaction, ...]
    face_overlays: tuple[FaceOverlay, ...]


def list_gestures(settings: Settings) -> list[GestureUsage]:
    """Every custom gesture, plus every built-in one that triggers something.

    Custom gestures come first (in recording order), then built-ins alphabetically.
    """
    config = settings.config
    used_builtins = sorted(
        {r.gesture for r in config.reactions if r.gesture in BUILTIN_GESTURES}
        | {o.trigger for o in config.face_overlays if o.trigger in BUILTIN_GESTURES}
    )
    return [
        GestureUsage(
            name=name,
            custom=name in settings.library,
            kind=settings.library.kind_of(name) or GestureKind.HAND,
            reactions=tuple(r for r in config.reactions if r.gesture == name),
            face_overlays=tuple(o for o in config.face_overlays if o.trigger == name),
        )
        for name in [*settings.library.names, *used_builtins]
    ]


def delete_gesture(settings: Settings, name: str) -> Settings:
    """Remove a gesture and everything it triggers; return freshly loaded settings.

    - Its reactions (corner memes) and the face overlays it triggers are removed from
      reactions.json.
    - A custom gesture's recorded pose is removed from custom_gestures.json.
    - A built-in gesture can't be un-learned by the model; after this it just does nothing.
    - Image files are kept: they may be shared, and deleting user files is irreversible.
    """
    config = settings.config
    is_custom = name in settings.library
    reactions = [r for r in config.reactions if r.gesture != name]
    overlays = [o for o in config.face_overlays if o.trigger != name]
    touches_config = len(reactions) != len(config.reactions) or len(overlays) != len(
        config.face_overlays
    )
    if not is_custom and not touches_config:
        raise ConfigError(f"No gesture named {name!r} is set up, so there's nothing to delete.")

    # Config first: if the library write then failed, the config would merely stop
    # referencing a pose that still exists, which is still a valid state.
    if touches_config:
        config = config.model_copy(update={"reactions": reactions, "face_overlays": overlays})
        atomic_write_text(settings.config_path, config_to_json(config))
    if is_custom:
        save_library(settings.library.without(name), settings.library_path)
    return load_settings(settings.config_path, settings.root)
