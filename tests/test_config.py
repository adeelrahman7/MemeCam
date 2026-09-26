import json
from pathlib import Path

import pytest
from PIL import Image

from memecam.config import AppConfig, ConfigError, Corner, HandFilter, load_config
from memecam.core.types import Handedness
from memecam.paths import default_config_path, resource_root


@pytest.fixture
def root(tmp_path: Path) -> Path:
    memes = tmp_path / "assets" / "memes"
    memes.mkdir(parents=True)
    for name in ("a.png", "b.png"):
        Image.new("RGBA", (8, 8), (255, 0, 0, 128)).save(memes / name)
    (memes / "c.gif").write_bytes(b"GIF89a")
    return tmp_path


def write(root: Path, data: object) -> Path:
    path = root / "reactions.json"
    path.write_text(json.dumps(data) if not isinstance(data, str) else data, encoding="utf-8")
    return path


def minimal(**overrides: object) -> dict:
    data: dict = {"reactions": [{"gesture": "Thumb_Up", "image": "assets/memes/a.png"}]}
    data.update(overrides)
    return data


def test_shipped_config_is_valid():
    config = load_config(default_config_path(), resource_root())
    assert config.reactions


def test_minimal_config_gets_defaults(root):
    config = load_config(write(root, minimal()), root)
    assert config.camera.mirror is True
    assert config.debounce.hold_frames == 6
    assert config.overlay.corner is Corner.TOP_RIGHT
    assert config.reactions[0].hand is HandFilter.ANY
    assert config.reactions[0].key == "Thumb_Up"


def test_values_are_parsed(root):
    data = minimal(
        debounce={"hold_frames": 10, "cooldown_seconds": 0.25}, overlay={"corner": "bottom_left"}
    )
    config = load_config(write(root, data), root)
    assert config.debounce.hold_frames == 10
    assert config.debounce.cooldown_seconds == 0.25
    assert config.overlay.corner is Corner.BOTTOM_LEFT


def test_missing_file(tmp_path):
    with pytest.raises(ConfigError, match="not found"):
        load_config(tmp_path / "nope.json", tmp_path)


def test_invalid_json_reports_position(root):
    with pytest.raises(ConfigError, match=r"not valid JSON \(line 2, column"):
        load_config(write(root, '{\n  "reactions": [,]\n}'), root)


def test_unknown_gesture(root):
    data = {"reactions": [{"gesture": "Thumbs_Up", "image": "assets/memes/a.png"}]}
    with pytest.raises(ConfigError) as exc:
        load_config(write(root, data), root)
    msg = str(exc.value)
    assert "reactions[0].gesture" in msg
    assert "unknown gesture 'Thumbs_Up'" in msg
    assert "Thumb_Up" in msg  # lists valid options


def test_missing_image(root):
    data = {"reactions": [{"gesture": "Victory", "image": "assets/memes/missing.png"}]}
    with pytest.raises(ConfigError, match=r"reactions\[0\]\.image: image file not found"):
        load_config(write(root, data), root)


def test_non_png_image_rejected_for_now(root):
    data = {"reactions": [{"gesture": "Victory", "image": "assets/memes/c.gif"}]}
    with pytest.raises(ConfigError, match=r"only \.png"):
        load_config(write(root, data), root)


def test_typo_in_key_is_rejected(root):
    with pytest.raises(ConfigError, match=r"debounce\.hold_frame: Extra inputs are not permitted"):
        load_config(write(root, minimal(debounce={"hold_frame": 3})), root)


def test_out_of_range_value(root):
    with pytest.raises(ConfigError, match=r"debounce\.hold_frames: .*greater than or equal to 1"):
        load_config(write(root, minimal(debounce={"hold_frames": 0})), root)


def test_bad_enum_value(root):
    with pytest.raises(ConfigError, match=r"overlay\.corner"):
        load_config(write(root, minimal(overlay={"corner": "middle"})), root)


def test_reactions_required_and_non_empty(root):
    with pytest.raises(ConfigError, match=r"reactions: Field required"):
        load_config(write(root, {}), root)
    with pytest.raises(ConfigError, match=r"reactions: List should have at least 1 item"):
        load_config(write(root, {"reactions": []}), root)


def test_all_problems_reported_at_once(root):
    data = {
        "debounce": {"hold_frames": -1},
        "reactions": [{"gesture": "Nope", "image": "assets/memes/a.png"}],
    }
    with pytest.raises(ConfigError, match=r"has 2 problem\(s\)"):
        load_config(write(root, data), root)


def test_duplicate_reaction(root):
    data = {
        "reactions": [
            {"gesture": "Thumb_Up", "image": "assets/memes/a.png"},
            {"gesture": "Thumb_Up", "image": "assets/memes/b.png"},
        ]
    }
    with pytest.raises(ConfigError, match="duplicate reaction for gesture 'Thumb_Up'"):
        load_config(write(root, data), root)


def test_any_plus_specific_hand_conflict(root):
    data = {
        "reactions": [
            {"gesture": "Victory", "image": "assets/memes/a.png", "hand": "left"},
            {"gesture": "Victory", "image": "assets/memes/b.png"},
        ]
    }
    with pytest.raises(ConfigError, match="both a hand='any' reaction and a hand-specific"):
        load_config(write(root, data), root)


def test_left_and_right_reactions_for_same_gesture(root):
    data = {
        "reactions": [
            {"gesture": "Victory", "image": "assets/memes/a.png", "hand": "left"},
            {"gesture": "Victory", "image": "assets/memes/b.png", "hand": "right"},
        ]
    }
    config = load_config(write(root, data), root)
    left = config.find_reaction("Victory", Handedness.LEFT)
    right = config.find_reaction("Victory", Handedness.RIGHT)
    assert left is not None and left.key == "Victory:left"
    assert right is not None and right.image.endswith("b.png")
    assert config.find_reaction("Thumb_Up", Handedness.LEFT) is None


def test_model_validate_without_root_skips_file_checks():
    config = AppConfig.model_validate(minimal(reactions=[{"gesture": "Victory", "image": "x.png"}]))
    assert config.reactions[0].image == "x.png"
