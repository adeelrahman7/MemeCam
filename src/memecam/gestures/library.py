"""custom_gestures.json: saved hand poses and face expressions (schema, load, save)."""

from __future__ import annotations

import os
import re
import tempfile
from collections.abc import Iterable
from pathlib import Path
from typing import Annotated, Literal, Self

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from memecam.core.types import Handedness
from memecam.gestures.templates import (
    NUM_BLENDSHAPES,
    NUM_LANDMARKS,
    GestureKind,
    GestureTemplate,
)

# Custom names are lowercase snake_case, so they can never collide with MediaPipe's
# built-in labels (which are Capitalized, e.g. "Thumb_Up").
CUSTOM_NAME_PATTERN = r"^[a-z][a-z0-9_]{1,31}$"
_NAME_RE = re.compile(CUSTOM_NAME_PATTERN)


class LibraryError(Exception):
    pass


def validate_custom_name(name: str) -> str | None:
    """Return an error message if ``name`` isn't a valid custom gesture name, else None."""
    if not _NAME_RE.fullmatch(name):
        return (
            "use 2-32 characters: lowercase letters, digits and underscores, "
            "starting with a letter (e.g. 'rock_on')"
        )
    return None


_Point = tuple[float, float]
_HandSample = Annotated[list[_Point], Field(min_length=NUM_LANDMARKS, max_length=NUM_LANDMARKS)]
_FaceSample = Annotated[list[float], Field(min_length=NUM_BLENDSHAPES, max_length=NUM_BLENDSHAPES)]


class _TemplateModel(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: Annotated[str, Field(pattern=CUSTOM_NAME_PATTERN)]
    kind: GestureKind = GestureKind.HAND  # files from before face gestures have no "kind"
    recorded_with: Handedness | None = None
    samples: Annotated[list[_HandSample] | list[_FaceSample], Field(min_length=1)]

    @model_validator(mode="after")
    def _samples_match_kind(self) -> Self:
        is_face_data = isinstance(self.samples[0][0], float | int)
        if is_face_data != (self.kind is GestureKind.FACE):
            expected = (
                f"lists of {NUM_BLENDSHAPES} scores"
                if self.kind is GestureKind.FACE
                else f"lists of {NUM_LANDMARKS} [x, y] points"
            )
            raise ValueError(f"samples for a {self.kind} gesture must be {expected}")
        return self


class _LibraryModel(BaseModel):
    model_config = ConfigDict(extra="forbid")
    version: Literal[1] = 1
    gestures: list[_TemplateModel] = []

    @model_validator(mode="after")
    def _unique_names(self) -> Self:
        names = [g.name for g in self.gestures]
        dupes = sorted({n for n in names if names.count(n) > 1})
        if dupes:
            raise ValueError(f"duplicate gesture names: {dupes}")
        return self


class GestureLibrary:
    """An immutable collection of named templates."""

    def __init__(self, templates: Iterable[GestureTemplate] = ()) -> None:
        self._templates = {t.name: t for t in templates}

    def __len__(self) -> int:
        return len(self._templates)

    def __contains__(self, name: object) -> bool:
        return name in self._templates

    @property
    def names(self) -> list[str]:
        return list(self._templates)

    @property
    def templates(self) -> list[GestureTemplate]:
        return list(self._templates.values())

    def with_template(self, template: GestureTemplate) -> GestureLibrary:
        """Return a copy with ``template`` added (replacing any template with the same name)."""
        return GestureLibrary([*(t for t in self.templates if t.name != template.name), template])

    def of_kind(self, kind: GestureKind) -> list[GestureTemplate]:
        return [t for t in self._templates.values() if t.kind is kind]

    def kind_of(self, name: str) -> GestureKind | None:
        t = self._templates.get(name)
        return t.kind if t else None

    def without(self, name: str) -> GestureLibrary:
        """Return a copy with the template called ``name`` removed (no-op if absent)."""
        return GestureLibrary(t for t in self.templates if t.name != name)

    def to_json(self) -> str:
        model = _LibraryModel(
            gestures=[
                _TemplateModel(
                    name=t.name,
                    kind=t.kind,
                    recorded_with=t.recorded_with,
                    samples=_rounded_samples(t),
                )
                for t in self.templates
            ]
        )
        return model.model_dump_json(indent=2) + "\n"


def _rounded_samples(t: GestureTemplate) -> list[list[float]] | list[list[_Point]]:
    if t.kind is GestureKind.FACE:
        return [[round(float(v), 4) for v in s] for s in t.samples]
    return [[(round(float(x), 4), round(float(y), 4)) for x, y in s] for s in t.samples]


def load_library(path: Path) -> GestureLibrary:
    """Load custom_gestures.json. A missing file just means no custom gestures yet."""
    if not path.exists():
        return GestureLibrary()
    try:
        model = _LibraryModel.model_validate_json(path.read_text(encoding="utf-8"))
    except ValidationError as exc:
        first = exc.errors(include_url=False)[0]
        loc = ".".join(str(p) for p in first["loc"])
        raise LibraryError(
            f"{path.name} is invalid ({exc.error_count()} problem(s)); first: "
            f"{loc or '(root)'}: {first['msg']}"
        ) from exc
    except OSError as exc:
        raise LibraryError(f"Could not read {path}: {exc}") from exc
    return GestureLibrary(
        GestureTemplate(
            name=g.name,
            samples=np.array(g.samples, dtype=np.float64),
            recorded_with=g.recorded_with,
            kind=g.kind,
        )
        for g in model.gestures
    )


def atomic_write_text(path: Path, text: str) -> None:
    """Write via a temp file + rename, so a crash never leaves a half-written file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as f:
            f.write(text)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def save_library(library: GestureLibrary, path: Path) -> None:
    atomic_write_text(path, library.to_json())
