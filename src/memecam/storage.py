"""Persisting a newly recorded gesture: template, reaction image and reactions.json entry."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from pydantic import ValidationError

from memecam.config import (
    AppConfig,
    ConfigError,
    Reaction,
    Settings,
    config_to_json,
    load_settings,
)
from memecam.gestures.library import atomic_write_text, save_library
from memecam.gestures.templates import GestureTemplate

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


def save_recorded_gesture(
    settings: Settings, template: GestureTemplate, image: Path | None
) -> Settings:
    """Save ``template`` and (optionally) map it to ``image``; return freshly loaded settings.

    Re-recording an existing name replaces its template. Choosing an image replaces any
    existing reactions for that gesture with a single hand="any" reaction.
    """
    config = settings.config
    if image is not None:
        rel = _image_config_path(image, template.name, settings.root)
        reactions = [r for r in config.reactions if r.gesture != template.name]
        reactions.append(Reaction(gesture=template.name, image=rel))
        config = config.model_copy(update={"reactions": reactions})
        # Validate before touching any file on disk.
        try:
            AppConfig.model_validate(
                json.loads(config_to_json(config)), context={"root": settings.root}
            )
        except ValidationError as exc:
            raise ConfigError(f"The updated config would be invalid:\n{exc}") from exc

    save_library(settings.library.with_template(template), settings.library_path)
    if image is not None:
        atomic_write_text(settings.config_path, config_to_json(config))
    return load_settings(settings.config_path, settings.root)
