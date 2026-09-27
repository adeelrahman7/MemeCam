import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from memecam.config import ConfigError, load_settings
from memecam.core.types import Handedness
from memecam.gestures.templates import GestureTemplate
from memecam.storage import delete_gesture, list_gestures, save_recorded_gesture


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


# --- listing and deleting gestures ----------------------------------------------------
@pytest.fixture
def wired(settings, outside_png):
    """A custom 'rock_on' with a meme, plus face overlays on Thumb_Up and rock_on."""
    s = save_recorded_gesture(settings, tpl("rock_on"), outside_png)
    s = save_recorded_gesture(s, tpl("unused_pose"), None)
    raw = json.loads(s.config_path.read_text())
    raw["face_overlays"] = [
        {"image": "assets/memes/a.png", "anchor": "forehead", "trigger": "Thumb_Up"},
        {"image": "assets/memes/a.png", "anchor": "eyes", "trigger": "rock_on"},
        {"image": "assets/memes/a.png", "anchor": "nose"},  # always on
    ]
    s.config_path.write_text(json.dumps(raw))
    return load_settings(s.config_path, s.root)


def test_list_gestures(wired):
    by_name = {g.name: g for g in list_gestures(wired)}
    assert list(by_name) == ["rock_on", "unused_pose", "Thumb_Up"]  # custom first
    assert by_name["rock_on"].custom and len(by_name["rock_on"].reactions) == 1
    assert len(by_name["rock_on"].face_overlays) == 1
    assert by_name["unused_pose"].reactions == () and by_name["unused_pose"].face_overlays == ()
    assert not by_name["Thumb_Up"].custom
    assert len(by_name["Thumb_Up"].reactions) == 1 and len(by_name["Thumb_Up"].face_overlays) == 1


def test_delete_custom_gesture_removes_pose_meme_and_overlays(wired):
    image = wired.root / "assets/memes/rock_on.png"
    new = delete_gesture(wired, "rock_on")
    assert "rock_on" not in new.library and "unused_pose" in new.library
    assert all(r.gesture != "rock_on" for r in new.config.reactions)
    assert all(o.trigger != "rock_on" for o in new.config.face_overlays)
    assert len(new.config.face_overlays) == 2  # the others stay
    assert image.is_file()  # image files are never deleted


def test_delete_builtin_gesture_unwires_it(wired):
    new = delete_gesture(wired, "Thumb_Up")
    assert all(r.gesture != "Thumb_Up" for r in new.config.reactions)
    assert all(o.trigger != "Thumb_Up" for o in new.config.face_overlays)
    assert new.library.names == wired.library.names  # custom poses untouched
    assert "Thumb_Up" not in [g.name for g in list_gestures(new)]


def test_delete_unused_custom_gesture_leaves_config_file_alone(wired):
    before = wired.config_path.read_text()
    new = delete_gesture(wired, "unused_pose")
    assert "unused_pose" not in new.library
    assert wired.config_path.read_text() == before


def test_delete_unknown_gesture(wired):
    with pytest.raises(ConfigError, match="nothing to delete"):
        delete_gesture(wired, "Victory")


def test_deleting_everything_leaves_a_loadable_config(wired):
    s = wired
    for g in list_gestures(wired):
        s = delete_gesture(s, g.name)
    assert list_gestures(s) == []
    assert s.config.reactions == []
    assert [o.trigger for o in s.config.face_overlays] == [None]  # always-on overlay kept


# --- placement -------------------------------------------------------------------------
from memecam.config import Corner  # noqa: E402
from memecam.face.anchors import Anchor  # noqa: E402
from memecam.placement import Placement, placement_choices  # noqa: E402
from memecam.storage import reaction_items, set_placement  # noqa: E402


def test_save_with_corner_placement(settings, outside_png):
    s = save_recorded_gesture(settings, tpl(), outside_png, Placement.in_corner(Corner.BOTTOM_LEFT))
    (item,) = reaction_items(s, "rock_on")
    assert item.placement == Placement.in_corner(Corner.BOTTOM_LEFT)
    assert json.loads(s.config_path.read_text())["reactions"][-1]["corner"] == "bottom_left"


def test_save_with_face_placement_uses_spot_defaults(settings, outside_png):
    s = save_recorded_gesture(settings, tpl(), outside_png, Placement.on_face(Anchor.FOREHEAD))
    assert all(r.gesture != "rock_on" for r in s.config.reactions)
    (overlay,) = s.config.face_overlays
    assert overlay.trigger == "rock_on" and overlay.anchor is Anchor.FOREHEAD
    assert (overlay.width, overlay.offset_y) == (1.3, -0.45)
    assert s.needs_face_tracking


def test_rerecording_with_image_replaces_both_kinds_of_entry(settings, outside_png):
    s = save_recorded_gesture(settings, tpl(), outside_png, Placement.on_face(Anchor.EYES))
    s = save_recorded_gesture(s, tpl(), outside_png, Placement.in_corner())
    assert [i.placement.is_face for i in reaction_items(s, "rock_on")] == [False]


def test_move_corner_to_corner_keeps_hand_and_position(settings):
    raw = json.loads(settings.config_path.read_text())
    raw["reactions"] = [
        {"gesture": "Victory", "image": "assets/memes/a.png", "hand": "left"},
        {"gesture": "Thumb_Up", "image": "assets/memes/a.png"},
    ]
    settings.config_path.write_text(json.dumps(raw))
    s = load_settings(settings.config_path, settings.root)
    (item,) = reaction_items(s, "Victory")
    s = set_placement(s, item, Placement.in_corner(Corner.TOP_LEFT))
    first = s.config.reactions[0]
    assert (first.gesture, first.hand.value, first.corner) == ("Victory", "left", Corner.TOP_LEFT)


def test_move_between_corner_and_face_and_back(settings):
    (item,) = reaction_items(settings, "Thumb_Up")
    s = set_placement(settings, item, Placement.on_face(Anchor.EYES, width=2.0))
    assert s.config.reactions == []
    (face_item,) = reaction_items(s, "Thumb_Up")
    assert face_item.placement == Placement.on_face(Anchor.EYES, 2.0)
    assert face_item.image == "assets/memes/a.png"  # same image

    # Same spot, new size: offsets kept; different spot: that spot's defaults.
    s = set_placement(s, face_item, Placement.on_face(Anchor.FOREHEAD))
    assert (s.config.face_overlays[0].width, s.config.face_overlays[0].offset_y) == (1.3, -0.45)
    (face_item,) = reaction_items(s, "Thumb_Up")
    s = set_placement(s, face_item, Placement.in_corner())
    assert [r.corner for r in s.config.reactions] == [None] and s.config.face_overlays == []


def test_cannot_have_two_any_hand_corner_memes(settings):
    (item,) = reaction_items(settings, "Thumb_Up")
    s = set_placement(settings, item, Placement.on_face(Anchor.NOSE))
    raw = json.loads(s.config_path.read_text())
    raw["reactions"] = [{"gesture": "Thumb_Up", "image": "assets/memes/a.png"}]
    s.config_path.write_text(json.dumps(raw))
    s = load_settings(s.config_path, s.root)
    face_item = next(i for i in reaction_items(s, "Thumb_Up") if i.placement.is_face)
    before = s.config_path.read_text()
    with pytest.raises(ConfigError, match="already shows a corner meme"):
        set_placement(s, face_item, Placement.in_corner(Corner.BOTTOM_RIGHT))
    assert s.config_path.read_text() == before  # nothing written


def test_placement_labels_and_choices():
    choices = placement_choices(Corner.TOP_RIGHT)
    labels = [label for label, _ in choices]
    assert labels[0] == "Corner: default (top right)"
    assert "Corner: top right" not in labels  # the default isn't listed twice
    assert "On face: eyes" in labels and len(choices) == 4 + 6
    assert Placement.in_corner().label(Corner.TOP_RIGHT) == "top-right corner (default)"
    assert Placement.in_corner(Corner.BOTTOM_LEFT).label(Corner.TOP_RIGHT) == "bottom-left corner"
    assert Placement.on_face(Anchor.MOUTH).label(Corner.TOP_RIGHT) == "on face: mouth"
