"""Locating bundled resources (models/, assets/, config/) in dev and PyInstaller builds."""

from __future__ import annotations

import sys
from pathlib import Path


def resource_root() -> Path:
    """Return the directory that contains models/, assets/ and config/.

    - Inside a PyInstaller bundle, data files are unpacked under ``sys._MEIPASS``.
    - In development (src layout), it's the project root: src/memecam/paths.py -> ../../..
    """
    meipass = getattr(sys, "_MEIPASS", None)
    if meipass is not None:
        return Path(meipass)
    return Path(__file__).resolve().parents[2]


def resolve_resource(relative: str | Path, root: Path | None = None) -> Path:
    """Resolve a path from config. Absolute paths are kept; relative ones hang off the root."""
    path = Path(relative)
    if path.is_absolute():
        return path
    return (root or resource_root()) / path


def default_config_path() -> Path:
    return resource_root() / "config" / "reactions.json"


def models_dir() -> Path:
    return resource_root() / "models"
