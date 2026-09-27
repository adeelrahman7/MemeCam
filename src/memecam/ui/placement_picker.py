"""Choosing where a gesture's image appears: a picker widget and the 'Change placement' dialog."""

from __future__ import annotations

from PySide6.QtCore import Slot
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QVBoxLayout,
    QWidget,
)

from memecam.config import ConfigError, Corner, Settings
from memecam.placement import FACE_DEFAULTS, Placement, placement_choices
from memecam.storage import ReactionItem, reaction_items, set_placement


class PlacementPicker(QWidget):
    """A dropdown of corners and face spots, plus a size box for face spots."""

    def __init__(self, default_corner: Corner, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._choices = placement_choices(default_corner)
        self._combo = QComboBox()
        for label, _ in self._choices:
            self._combo.addItem(label)
        self._combo.insertSeparator(4)  # between the corners and the face spots

        self._size = QDoubleSpinBox()
        self._size.setRange(0.3, 6.0)
        self._size.setSingleStep(0.1)
        self._size.setDecimals(1)
        self._size.setSuffix(" x eye gap")
        self._size.setToolTip("How wide the image is, relative to the distance between your eyes")
        self._size_label = QLabel("Size:")
        self._keep_width: tuple[Placement, float] | None = None

        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(8)
        row.addWidget(self._combo, 1)
        row.addWidget(self._size_label)
        row.addWidget(self._size)
        self._combo.currentIndexChanged.connect(self._on_choice_changed)
        self._on_choice_changed()

    def _choice(self) -> Placement:
        label = self._combo.currentText()
        return next(p for text, p in self._choices if text == label)

    @Slot()
    def _on_choice_changed(self) -> None:
        choice = self._choice()
        self._size.setVisible(choice.is_face)
        self._size_label.setVisible(choice.is_face)
        if choice.anchor is not None:
            # Returning to the spot an image already uses keeps its size; else spot default.
            if self._keep_width and self._keep_width[0].anchor is choice.anchor:
                self._size.setValue(self._keep_width[1])
            else:
                self._size.setValue(FACE_DEFAULTS[choice.anchor][0])

    def placement(self) -> Placement:
        choice = self._choice()
        if choice.anchor is None:
            return choice
        return Placement.on_face(choice.anchor, round(self._size.value(), 2))

    def set_placement(self, placement: Placement) -> None:
        target = Placement(corner=placement.corner, anchor=placement.anchor)
        for i in range(self._combo.count()):
            text = self._combo.itemText(i)
            if any(text == label and p == target for label, p in self._choices):
                if placement.anchor is not None:
                    self._keep_width = (target, placement.face_width())
                self._combo.setCurrentIndex(i)
                self._on_choice_changed()
                return


class PlacementDialog(QDialog):
    """Move one of a gesture's images to another corner or face spot."""

    def __init__(self, settings: Settings, gesture: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle(f"Change placement: {gesture}")
        self._settings = settings
        self._default_corner = settings.config.overlay.corner
        self._items = reaction_items(settings, gesture)
        self.result_settings: Settings | None = None

        self._item_combo = QComboBox()
        for item in self._items:
            self._item_combo.addItem(item.describe(self._default_corner))
        self._picker = PlacementPicker(self._default_corner)
        self._item_combo.currentIndexChanged.connect(self._load_item)

        form = QFormLayout()
        form.setHorizontalSpacing(14)
        form.setVerticalSpacing(12)
        if len(self._items) > 1:
            form.addRow("Image:", self._item_combo)
        else:
            form.addRow("Image:", QLabel(self._items[0].describe(self._default_corner)))
        form.addRow("Show it:", self._picker)

        note = QLabel(
            "Face spots need the face model and follow your head. Fine-tune sizes and offsets "
            "in config/reactions.json."
        )
        note.setWordWrap(True)
        note.setObjectName("Hint")

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.button(QDialogButtonBox.StandardButton.Save).setObjectName("Primary")
        buttons.accepted.connect(self._save)
        buttons.rejected.connect(self.reject)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(22, 20, 22, 18)
        layout.setSpacing(12)
        layout.addLayout(form)
        layout.addWidget(note)
        layout.addWidget(buttons)
        self.setMinimumWidth(520)
        self._load_item()

    def _current(self) -> ReactionItem:
        return self._items[max(0, self._item_combo.currentIndex())]

    @Slot()
    def _load_item(self) -> None:
        self._picker.set_placement(self._current().placement)

    @Slot()
    def _save(self) -> None:
        item, placement = self._current(), self._picker.placement()
        if placement == item.placement:
            self.reject()  # nothing changed
            return
        try:
            self.result_settings = set_placement(self._settings, item, placement)
        except (ConfigError, OSError) as exc:
            QMessageBox.warning(self, "Can't move it there", str(exc))
            return
        self.accept()
