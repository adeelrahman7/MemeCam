"""The 'Record a gesture' dialog: a name, and whether it's a hand pose or a face expression."""

from __future__ import annotations

from PySide6.QtCore import Qt, Slot
from PySide6.QtWidgets import (
    QButtonGroup,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QRadioButton,
    QVBoxLayout,
    QWidget,
)

from memecam.config import Corner
from memecam.gestures.library import GestureLibrary, validate_custom_name
from memecam.gestures.templates import GestureKind
from memecam.placement import Placement
from memecam.ui.placement_picker import PlacementPicker

_HINTS = {
    GestureKind.HAND: "Hold a hand pose in view. Either hand works.",
    GestureKind.FACE: (
        "Make a clear, exaggerated face (e.g. mouth wide open, big smile, eyebrows up). "
        "Where you look doesn't matter. Needs the face model."
    ),
}


class RecordDialog(QDialog):
    """Ask for a name and a kind; validates the name and confirms replacing an existing one."""

    def __init__(
        self,
        library: GestureLibrary,
        parent: QWidget | None = None,
        *,
        kind: GestureKind = GestureKind.HAND,
        default_corner: Corner = Corner.TOP_RIGHT,
        placement: Placement | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Record a gesture")
        self._library = library

        self._name = QLineEdit()
        self._name.setPlaceholderText("e.g. rock_on, shocked_face")
        self._error = QLabel()
        self._error.setObjectName("Error")
        self._error.setWordWrap(True)
        self._error.hide()
        self._name.textChanged.connect(lambda _t: self._error.hide())

        self._hand = QRadioButton("Hand pose")
        self._face = QRadioButton("Face expression")
        self._group = QButtonGroup(self)
        self._group.addButton(self._hand)
        self._group.addButton(self._face)
        (self._face if kind is GestureKind.FACE else self._hand).setChecked(True)
        kinds = QHBoxLayout()
        kinds.setSpacing(20)
        kinds.addWidget(self._hand)
        kinds.addWidget(self._face)
        kinds.addStretch()

        self._hint = QLabel()
        self._hint.setWordWrap(True)
        self._hint.setObjectName("Hint")
        self._group.buttonToggled.connect(lambda *_: self._update_hint())
        self._update_hint()

        self._placement = PlacementPicker(default_corner)
        if placement is not None:
            self._placement.set_placement(placement)

        form = QFormLayout()
        form.setHorizontalSpacing(14)
        form.setVerticalSpacing(12)
        form.setLabelAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        form.addRow("Record with:", kinds)
        form.addRow("Name:", self._name)
        form.addRow("Show image:", self._placement)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        start = buttons.button(QDialogButtonBox.StandardButton.Ok)
        start.setText("Start recording")
        start.setObjectName("Primary")
        buttons.accepted.connect(self._try_accept)
        buttons.rejected.connect(self.reject)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(22, 20, 22, 18)
        layout.setSpacing(12)
        layout.addLayout(form)
        layout.addWidget(self._hint)
        layout.addWidget(self._error)
        layout.addWidget(buttons)
        self.setMinimumWidth(520)

    @property
    def name(self) -> str:
        return self._name.text().strip()

    @property
    def placement(self) -> Placement:
        """Where the image you pick after recording will appear."""
        return self._placement.placement()

    @property
    def kind(self) -> GestureKind:
        return GestureKind.FACE if self._face.isChecked() else GestureKind.HAND

    def _update_hint(self) -> None:
        self._hint.setText(_HINTS[self.kind])

    def _show_error(self, text: str) -> None:
        self._error.setText(text)
        self._error.show()

    @Slot()
    def _try_accept(self) -> None:
        name = self.name
        if error := validate_custom_name(name):
            self._show_error(f"'{name}': {error}" if name else "Enter a name.")
            return
        existing = self._library.kind_of(name)
        if existing is not None:
            change = "" if existing is self.kind else f" (it's currently a {existing} gesture)"
            answer = QMessageBox.question(
                self, "Replace gesture?", f"'{name}' already exists{change}. Record it again?"
            )
            if answer != QMessageBox.StandardButton.Yes:
                return
        self.accept()
