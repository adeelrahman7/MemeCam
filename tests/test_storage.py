import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from memecam.config import ConfigError, load_settings
from memecam.core.types import Handedness
from memecam.gestures.templates import GestureTemplate
from memecam.storage import save_recorded_gesture


@pytest.fixture
def settings(tmp_path: Path):
    memes = tmp_path / "assets" / "memes"
    memes.mkdir(parents=True)
    Image.new("RGBA", (8, 8), (0, 255, 0, 255)).save(memes / "a.png")
    (tmp_path / "config").mkdir()
    cfg = {
        "debounce": {"hold_frames": 4},
        "reactions": [{"gesture": "Thumb_Up", "image": "assets/memes/a.png"}],
    }
    (tmp_path / "config" / "reactions.json").write_text(json.dumps(cfg))
    return load_settings(tmp_path / "config" / "reactions.json", tmp_path)


def tpl(name="rock_on") -> GestureTemplate:
    return GestureTemplate(name, np.zeros((3, 21, 2)))


@pytest.fixture
def outside_png(tmp_path_factory) -> Path:
    path = tmp_path_factory.mktemp("downloads") / "my meme.png"
    Image.new("RGBA", (8, 8), (255, 0, 0, 255)).save(path)
    return path


def test_saves_gesture_and_links_copied_image(settings, outside_png):
    new = save_recorded_gesture(settings, tpl(), outside_png)
    assert "rock_on" in new.library
    reaction = new.config.find_reaction("rock_on", Handedness.LEFT)
    assert reaction is not None
    assert reaction.image == "assets/memes/rock_on.png"
    assert (settings.root / "assets/memes/rock_on.png").is_file()
    # Existing settings survive the rewrite, and defaults aren't dumped into the file.
    raw = json.loads(settings.config_path.read_text())
    assert raw["debounce"] == {"hold_frames": 4}
    assert "camera" not in raw
    assert [r["gesture"] for r in raw["reactions"]] == ["Thumb_Up", "rock_on"]


def test_image_already_in_project_is_not_copied(settings):
    new = save_recorded_gesture(settings, tpl(), settings.root / "assets/memes/a.png")
    assert new.config.reactions[-1].image == "assets/memes/a.png"
    assert not (settings.root / "assets/memes/rock_on.png").exists()


def test_gesture_only_leaves_reactions_untouched(settings):
    before = settings.config_path.read_text()
    new = save_recorded_gesture(settings, tpl(), None)
    assert "rock_on" in new.library
    assert settings.config_path.read_text() == before


def test_rerecording_replaces_reaction(settings, outside_png):
    s1 = save_recorded_gesture(settings, tpl(), settings.root / "assets/memes/a.png")
    s2 = save_recorded_gesture(s1, tpl(), outside_png)
    rock = [r for r in s2.config.reactions if r.gesture == "rock_on"]
    assert len(rock) == 1 and rock[0].image == "assets/memes/rock_on.png"
    assert len(s2.library) == 1


def test_non_png_rejected_before_writing(settings, tmp_path):
    gif = tmp_path / "x.gif"
    gif.write_bytes(b"GIF89a")
    with pytest.raises(ConfigError, match=r"\.png"):
        save_recorded_gesture(settings, tpl(), gif)
    assert not settings.library_path.exists()
