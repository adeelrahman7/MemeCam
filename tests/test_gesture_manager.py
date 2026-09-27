import json
import os

import numpy as np
import pytest
from PIL import Image

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication, QMessageBox

from memecam.config import load_settings
from memecam.gestures.templates import GestureTemplate
from memecam.storage import list_gestures, save_recorded_gesture
from memecam.ui.gesture_manager import GestureManagerDialog, describe_deletion


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def settings(tmp_path):
    (tmp_path / "assets").mkdir()
    Image.new("RGBA", (8, 8), (0, 255, 0, 255)).save(tmp_path / "assets/up.png")
    (tmp_path / "config").mkdir()
    cfg = {
        "reactions": [{"gesture": "Thumb_Up", "image": "assets/up.png"}],
        "face_overlays": [{"image": "assets/up.png", "anchor": "eyes", "trigger": "Thumb_Up"}],
    }
    (tmp_path / "config/reactions.json").write_text(json.dumps(cfg))
    s = load_settings(tmp_path / "config/reactions.json", tmp_path)
    return save_recorded_gesture(s, GestureTemplate("rock_on", np.zeros((1, 21, 2))), None)


def test_confirmation_text(settings):
    by_name = {g.name: g for g in list_gestures(settings)}
    custom = describe_deletion(by_name["rock_on"])
    assert "its recorded hand pose" in custom
    builtin = describe_deletion(by_name["Thumb_Up"])
    assert "corner meme: up.png" in builtin and "face overlay: up.png on eyes" in builtin
    assert "still be recognized" in builtin


def test_dialog_deletes_after_confirmation(app, settings, monkeypatch):
    dlg = GestureManagerDialog(settings)
    assert [g.name for g in dlg._gestures] == ["rock_on", "Thumb_Up"]
    changed = []
    dlg.settings_changed.connect(changed.append)

    dlg._table.selectRow(0)
    monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: QMessageBox.StandardButton.Cancel)
    dlg._delete_selected()
    assert changed == []  # cancelled: nothing happens

    monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: QMessageBox.StandardButton.Yes)
    dlg._delete_selected()
    assert len(changed) == 1 and "rock_on" not in changed[0].library
    assert [g.name for g in dlg._gestures] == ["Thumb_Up"]
    assert not dlg._delete_button.isEnabled()  # selection cleared after refresh


def test_record_dialog_validates_and_reports_kind(app, settings, monkeypatch):
    from memecam.gestures.templates import GestureKind
    from memecam.ui.record_dialog import RecordDialog

    dlg = RecordDialog(settings.library)
    assert dlg.kind is GestureKind.HAND
    dlg._name.setText("Bad Name")
    dlg._try_accept()
    assert dlg.result() == 0 and not dlg._error.isHidden()

    dlg._face.setChecked(True)
    assert dlg.kind is GestureKind.FACE and "face" in dlg._hint.text().lower()
    dlg._name.setText("shocked_face")
    dlg._try_accept()
    assert dlg.result() == RecordDialog.DialogCode.Accepted and dlg.name == "shocked_face"

    # Re-using an existing name (recorded as a hand) asks first and mentions the change.
    asked = []
    monkeypatch.setattr(
        QMessageBox, "question",
        lambda *a, **k: asked.append(a[2]) or QMessageBox.StandardButton.No,
    )  # fmt: skip
    dlg2 = RecordDialog(settings.library, kind=GestureKind.FACE)
    dlg2._name.setText("rock_on")
    dlg2._try_accept()
    assert dlg2.result() == 0 and "currently a hand gesture" in asked[0]


def test_manage_window_shows_kind(app, settings):
    dlg = GestureManagerDialog(settings)
    assert dlg._table.item(0, 1).text() == "Custom hand"
    assert dlg._table.item(1, 1).text() == "Built-in hand"
    assert dlg._table.item(0, 2).text() == "— (no image)"


# --- placement UI ---------------------------------------------------------------------
def test_placement_picker_roundtrip(app):
    from memecam.config import Corner
    from memecam.face.anchors import Anchor
    from memecam.placement import Placement
    from memecam.ui.placement_picker import PlacementPicker

    picker = PlacementPicker(Corner.TOP_RIGHT)
    assert picker.placement() == Placement.in_corner()
    assert picker._size.isHidden()
    for p in (Placement.in_corner(Corner.BOTTOM_LEFT), Placement.on_face(Anchor.MOUTH, 2.5)):
        picker.set_placement(p)
        assert picker.placement() == p
    assert not picker._size.isHidden()
    picker.set_placement(Placement.on_face(Anchor.FOREHEAD))
    assert picker.placement() == Placement.on_face(Anchor.FOREHEAD, 1.3)  # spot default


def test_record_dialog_remembers_placement(app, settings):
    from memecam.config import Corner
    from memecam.face.anchors import Anchor
    from memecam.placement import Placement
    from memecam.ui.record_dialog import RecordDialog

    dlg = RecordDialog(settings.library)
    assert dlg.placement == Placement.in_corner()
    p = Placement.on_face(Anchor.EYES, 1.4)
    dlg2 = RecordDialog(settings.library, default_corner=Corner.BOTTOM_LEFT, placement=p)
    assert dlg2.placement == p


def test_change_placement_from_manage_window(app, settings, monkeypatch):
    from memecam.face.anchors import Anchor
    from memecam.placement import Placement
    from memecam.ui import gesture_manager
    from memecam.ui.placement_picker import PlacementDialog

    # Thumb_Up shows up.png in the corner and on the eyes; move the corner one to the nose.
    pd = PlacementDialog(settings, "Thumb_Up")
    assert pd._item_combo.count() == 2
    pd._item_combo.setCurrentIndex(0)
    pd._picker.set_placement(Placement.on_face(Anchor.NOSE))
    pd._save()
    new = pd.result_settings
    assert new is not None and new.config.reactions == []
    assert sorted(o.anchor.value for o in new.config.face_overlays) == ["eyes", "nose"]

    # The manage window refreshes and passes the new settings on.
    class FakeDialog:
        def __init__(self, *a):
            self.result_settings = new

        def exec(self):
            return gesture_manager.QDialog.DialogCode.Accepted

    monkeypatch.setattr(gesture_manager, "PlacementDialog", FakeDialog)
    dlg = GestureManagerDialog(settings)
    got = []
    dlg.settings_changed.connect(got.append)
    dlg._table.selectRow(1)  # Thumb_Up
    assert dlg._place_button.isEnabled()
    dlg._change_placement()
    assert got == [new]
    assert "on face: nose" in dlg._table.item(1, 2).text()

    dlg._table.selectRow(0)  # rock_on has no image: nothing to move
    assert not dlg._place_button.isEnabled()
