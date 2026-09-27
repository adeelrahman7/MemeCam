"""Main window: shows the processed feed, the debug toggle and gesture recording."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt, Slot
from PySide6.QtGui import QCloseEvent, QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from memecam.config import ConfigError, Settings
from memecam.core.frame_worker import FrameWorker
from memecam.core.types import FrameStats
from memecam.gestures.recorder import RecorderStatus, RecordPhase
from memecam.gestures.templates import GestureKind, GestureTemplate
from memecam.placement import Placement
from memecam.storage import MEMES_DIR, save_recorded_gesture
from memecam.ui.gesture_manager import GestureManagerDialog
from memecam.ui.record_dialog import RecordDialog
from memecam.ui.widgets import VideoView


class MainWindow(QMainWindow):
    def __init__(self, worker: FrameWorker, settings: Settings) -> None:
        super().__init__()
        self._settings = settings
        self._recording = False
        self._last_kind = GestureKind.HAND  # the record dialog remembers your last choices
        self._placement = Placement.in_corner()
        self.setWindowTitle("MemeCam")
        self.resize(1100, 700)

        self._worker = worker
        self._view = VideoView()

        # --- top bar: title · record · gestures · debug ········· fps ---
        title = QLabel("MemeCam")
        title.setObjectName("AppTitle")
        self._record_button = QPushButton()
        self._record_button.clicked.connect(self._on_record_clicked)
        self._manage_button = QPushButton("Gestures")
        self._manage_button.setToolTip("See, move or delete your gestures (G)")
        self._manage_button.clicked.connect(self._open_gesture_manager)
        self._debug_button = QPushButton("Debug")
        self._debug_button.setCheckable(True)
        self._debug_button.setToolTip(
            "Show hand/face tracking, match distances and FPS on the video (D)"
        )
        self._debug_button.toggled.connect(self._worker.set_debug)
        self._fps_label = QLabel("-- fps")
        self._fps_label.setObjectName("Pill")

        bar = QWidget()
        bar.setObjectName("TopBar")
        row = QHBoxLayout(bar)
        row.setContentsMargins(16, 10, 16, 10)
        row.setSpacing(8)
        row.addWidget(title)
        row.addSpacing(12)
        row.addWidget(self._record_button)
        row.addWidget(self._manage_button)
        row.addWidget(self._debug_button)
        row.addStretch()
        row.addWidget(self._fps_label)

        central = QWidget()
        column = QVBoxLayout(central)
        column.setContentsMargins(0, 0, 0, 0)
        column.setSpacing(0)
        column.addWidget(bar)
        column.addWidget(self._view, 1)
        self.setCentralWidget(central)
        self._set_recording(False)

        for key, slot in (
            ("R", self._on_record_clicked),
            ("G", self._open_gesture_manager),
            ("D", self._debug_button.toggle),
        ):
            QShortcut(QKeySequence(key), self, activated=slot)
        QShortcut(QKeySequence(Qt.Key.Key_Escape), self, activated=self._cancel_recording)

        worker.frame_ready.connect(self._view.show_image)
        worker.stats_ready.connect(self._on_stats)
        worker.reaction_fired.connect(self._on_reaction)
        worker.recording_updated.connect(self._on_recording_updated)
        worker.warning.connect(self._on_warning)
        worker.failed.connect(self._on_failed)

    def _toast(self, text: str, ms: int = 2500) -> None:
        self._view.toast.show_message(text, ms)

    def enable_debug(self) -> None:
        self._debug_button.setChecked(True)

    @Slot(object)
    def _on_stats(self, stats: FrameStats) -> None:
        self._fps_label.setText(f"{stats.fps:.0f} fps")

    @Slot(str)
    def _on_reaction(self, key: str) -> None:
        pass  # the meme on screen is feedback enough

    @Slot(str)
    def _on_warning(self, message: str) -> None:
        self._toast(message.splitlines()[0], 6000)
        QMessageBox.warning(self, "MemeCam", message)

    @Slot(str)
    def _on_failed(self, message: str) -> None:
        self._view.set_message("Camera stopped.")
        QMessageBox.critical(self, "MemeCam", message)

    # --- gesture recording ------------------------------------------------------------
    @Slot()
    def _on_record_clicked(self) -> None:
        if self._recording:
            self._cancel_recording()
            return
        dialog = RecordDialog(
            self._settings.library,
            self,
            kind=self._last_kind,
            default_corner=self._settings.config.overlay.corner,
            placement=self._placement,
        )
        if dialog.exec() != RecordDialog.DialogCode.Accepted:
            return
        self._last_kind = dialog.kind
        self._placement = dialog.placement
        self._set_recording(True)
        self._worker.start_recording(dialog.name, dialog.kind)

    @Slot()
    def _open_gesture_manager(self) -> None:
        if self._recording:
            self._toast("Finish or cancel the recording first (Esc)")
            return
        dialog = GestureManagerDialog(self._settings, self)
        dialog.settings_changed.connect(self._on_settings_changed)
        dialog.exec()

    @Slot(object)
    def _on_settings_changed(self, settings: Settings) -> None:
        """Apply new settings live: the worker swaps them in between frames."""
        self._settings = settings
        self._worker.apply_settings(settings)

    @Slot()
    def _cancel_recording(self) -> None:
        if self._recording:
            self._worker.cancel_recording()
            self._set_recording(False)
            self._toast("Recording cancelled")

    def _set_recording(self, active: bool) -> None:
        self._recording = active
        b = self._record_button
        b.setText("■  Stop recording" if active else "●  Record")
        b.setToolTip("Cancel the recording (Esc)" if active else "Record a new gesture (R)")
        b.setObjectName("Danger" if active else "Primary")
        b.style().unpolish(b)  # re-apply the stylesheet for the new objectName
        b.style().polish(b)
        self._manage_button.setEnabled(not active)

    @Slot(object)
    def _on_recording_updated(self, status: RecorderStatus) -> None:
        if not self._recording:
            return  # a stale status from just before a cancel
        if status.phase is RecordPhase.FAILED:
            self._set_recording(False)
            QMessageBox.warning(self, "Recording failed", status.message)
        elif status.phase is RecordPhase.DONE and status.template is not None:
            self._set_recording(False)
            self._finish_recording(status.template, status.warning)

    def _finish_recording(self, template: GestureTemplate, warning: str = "") -> None:
        image, _ = QFileDialog.getOpenFileName(
            self,
            f"Pick a reaction image for '{template.name}' (Cancel = save the gesture only)",
            str(self._settings.root / MEMES_DIR),
            "PNG images (*.png)",
        )
        try:
            self._settings = save_recorded_gesture(
                self._settings, template, Path(image) if image else None, self._placement
            )
        except (ConfigError, OSError) as exc:
            QMessageBox.critical(self, "Couldn't save gesture", str(exc))
            return
        self._on_settings_changed(self._settings)
        where = self._placement.label(self._settings.config.overlay.corner)
        linked = where if image else "no image yet"
        self._toast(f"Saved '{template.name}' · {linked}", 4000)
        if warning:
            QMessageBox.information(self, "Gesture saved", warning)

    def closeEvent(self, event: QCloseEvent) -> None:  # noqa: N802 (Qt override)
        self._worker.stop()
        super().closeEvent(event)
