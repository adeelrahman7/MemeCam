import json

import numpy as np
import pytest

from memecam.core.types import Handedness
from memecam.gestures.library import (
    GestureLibrary,
    LibraryError,
    load_library,
    save_library,
    validate_custom_name,
)
from memecam.gestures.templates import GestureTemplate


def tpl(name: str, value: float = 0.5) -> GestureTemplate:
    return GestureTemplate(name, np.full((2, 21, 2), value), Handedness.RIGHT)


def test_missing_file_is_empty_library(tmp_path):
    assert len(load_library(tmp_path / "nope.json")) == 0


def test_roundtrip(tmp_path):
    path = tmp_path / "sub" / "custom_gestures.json"
    save_library(GestureLibrary([tpl("rock_on"), tpl("peace_out", 0.25)]), path)
    lib = load_library(path)
    assert lib.names == ["rock_on", "peace_out"]
    rock = lib.templates[0]
    assert rock.recorded_with is Handedness.RIGHT
    assert rock.samples.shape == (2, 21, 2)
    assert np.allclose(rock.samples, 0.5)
    assert json.loads(path.read_text())["version"] == 1
    assert not list(path.parent.glob("*.tmp"))  # atomic write cleaned up


def test_with_template_replaces_same_name():
    lib = GestureLibrary([tpl("rock_on", 0.1)]).with_template(tpl("rock_on", 0.9))
    assert len(lib) == 1
    assert np.allclose(lib.templates[0].samples, 0.9)


@pytest.mark.parametrize(
    ("content", "match"),
    [
        ('{"gestures": [{"name": "Bad Name", "samples": []}]}', "gestures.0.name"),
        ('{"gestures": [{"name": "ok_name", "samples": [[[0, 0]]]}]}', "samples"),
        ('{"version": 2}', "version"),
        ("not json", "invalid"),
    ],
)
def test_invalid_files(tmp_path, content, match):
    path = tmp_path / "custom_gestures.json"
    path.write_text(content)
    with pytest.raises(LibraryError, match=match):
        load_library(path)


def test_duplicate_names_rejected(tmp_path):
    path = tmp_path / "custom_gestures.json"
    sample = [[0.0, 0.0]] * 21
    g = {"name": "dup_name", "samples": [sample]}
    path.write_text(json.dumps({"gestures": [g, g]}))
    with pytest.raises(LibraryError, match="duplicate"):
        load_library(path)


@pytest.mark.parametrize("name", ["rock_on", "ab", "gun2", "a_b_c"])
def test_valid_names(name):
    assert validate_custom_name(name) is None


@pytest.mark.parametrize("name", ["", "a", "Rock", "Thumb_Up", "2cool", "has space", "x" * 33])
def test_invalid_names(name):
    assert validate_custom_name(name) is not None
