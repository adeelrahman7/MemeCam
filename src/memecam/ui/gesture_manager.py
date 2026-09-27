"""'Manage gestures' window: see what every gesture does and delete the ones you don't want."""

from __future__ import annotations

from pathlib import PurePosixPath

from PySide6.QtCore import Qt, Signal, Slot
from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QAbstractItemView,
    QDialog,
    QDialogButtonBox,
    QHeaderView,
    QLabel,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from memecam.config import ConfigError, FaceOverlay, Reaction, Settings
from memecam.gestures.templates import GestureKind
from memecam.storage import GestureUsage, delete_gesture, list_gestures, reaction_items
from memecam.ui.placement_picker import PlacementDialog

_NONE = "—"


def _describe_reaction(r: Reaction) -> str:
    name = PurePosixPath(r.image).name
    return name if r.hand.value == "any" else f"{name} ({r.hand.value} hand)"


def _describe_overlay(o: FaceOverlay) -> str:
    return f"{PurePosixPath(o.image).name} on {o.anchor.value}"


def describe_deletion(g: GestureUsage) -> str:
    """The confirmation text: exactly what will be removed."""
    lines = []
    if g.custom:
        what = "face expression" if g.kind is GestureKind.FACE else "hand pose"
        lines.append(f"• its recorded {what}")
    lines += [f"• corner meme: {_describe_reaction(r)}" for r in g.reactions]
    lines += [f"• face overlay: {_describe_overlay(o)}" for o in g.face_overlays]
    items = "\n".join(lines)
    if g.custom:
        head = f"Delete the gesture '{g.name}'?\n\nThis removes:\n{items}"
    else:
        head = (
            f"Remove everything '{g.name}' triggers?\n\nThis removes:\n{items}\n\n"
            f"'{g.name}' is built into MediaPipe's model, so it will still be recognized; "
            "it just won't do anything."
        )
    return head + "\n\nImage files stay in assets/."


class GestureManagerDialog(QDialog):
    """Emits ``settings_changed`` with freshly loaded Settings after each deletion."""

    settings_changed = Signal(object)

    def __init__(self, settings: Settings, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Manage gestures")
        self.resize(780, 420)
        self._settings = settings
        self._gestures: list[GestureUsage] = []

        self._table = QTableWidget(0, 3)
        self._table.setHorizontalHeaderLabels(["Gesture", "Type", "Shows"])
        self._table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self._table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self._table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self._table.verticalHeader().setVisible(False)
        self._table.verticalHeader().setDefaultSectionSize(38)
        self._table.setShowGrid(False)
        self._table.setAlternatingRowColors(True)
        self._table.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        header = self._table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        header.setStretchLastSection(True)
        header.setDefaultAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        self._table.itemSelectionChanged.connect(self._update_buttons)
        self._table.itemDoubleClicked.connect(lambda _item: self._delete_selected())

        self._empty = QLabel("No gestures are set up yet. Close this and press R to record one.")
        self._empty.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._empty.setObjectName("Hint")

        hint = QLabel(
            "Select a gesture to move its image or delete it. Deleting a custom gesture "
            "erases its recording; built-in ones just stop triggering anything."
        )
        hint.setWordWrap(True)
        hint.setObjectName("Hint")

        self._delete_button = QPushButton("Delete…")
        self._delete_button.clicked.connect(self._delete_selected)
        self._place_button = QPushButton("Change placement…")
        self._place_button.setToolTip("Move this gesture's image to another corner or face spot")
        self._place_button.clicked.connect(self._change_placement)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.addButton(self._place_button, QDialogButtonBox.ButtonRole.ActionRole)
        buttons.addButton(self._delete_button, QDialogButtonBox.ButtonRole.ActionRole)
        buttons.rejected.connect(self.reject)
        QShortcut(QKeySequence(QKeySequence.StandardKey.Delete), self, self._delete_selected)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(22, 20, 22, 18)
        layout.setSpacing(12)
        layout.addWidget(hint)
        layout.addWidget(self._table)
        layout.addWidget(self._empty)
        layout.addWidget(buttons)
        self._refresh()

    def _refresh(self) -> None:
        self._gestures = list_gestures(self._settings)
        self._table.setRowCount(len(self._gestures))
        for row, g in enumerate(self._gestures):
            corner = self._settings.config.overlay.corner
            shows = [i.describe(corner) for i in reaction_items(self._settings, g.name)]
            cells = [
                g.name,
                f"Custom {g.kind}" if g.custom else "Built-in hand",
                ", ".join(shows) or f"{_NONE} (no image)",
            ]
            for col, text in enumerate(cells):
                self._table.setItem(row, col, QTableWidgetItem(text))
        self._table.clearSelection()  # never leave the next row armed for another Delete
        has_rows = bool(self._gestures)
        self._table.setVisible(has_rows)
        self._empty.setVisible(not has_rows)
        self._update_buttons()

    def _selected(self) -> GestureUsage | None:
        rows = self._table.selectionModel().selectedRows()
        return self._gestures[rows[0].row()] if rows else None

    @Slot()
    def _update_buttons(self) -> None:
        g = self._selected()
        self._delete_button.setEnabled(g is not None)
        self._place_button.setEnabled(g is not None and bool(g.reactions or g.face_overlays))

    @Slot()
    def _change_placement(self) -> None:
        g = self._selected()
        if g is None or not (g.reactions or g.face_overlays):
            return
        dialog = PlacementDialog(self._settings, g.name, self)
        if dialog.exec() != QDialog.DialogCode.Accepted or dialog.result_settings is None:
            return
        self._settings = dialog.result_settings
        self._refresh()
        self.settings_changed.emit(self._settings)

    @Slot()
    def _delete_selected(self) -> None:
        g = self._selected()
        if g is None:
            return
        answer = QMessageBox.question(
            self,
            "Delete gesture",
            describe_deletion(g),
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Cancel,
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        try:
            self._settings = delete_gesture(self._settings, g.name)
        except (ConfigError, OSError) as exc:
            QMessageBox.critical(self, "Couldn't delete gesture", str(exc))
            return
        self._refresh()
        self.settings_changed.emit(self._settings)
