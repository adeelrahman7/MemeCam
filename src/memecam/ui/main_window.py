"""Main window: shows the processed feed, the debug toggle and gesture recording."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt, Slot
from PySide6.QtGui import (
    QAction,
    QCloseEvent,
    QImage,
    QKeySequence,
    QPixmap,
    QResizeEvent,
    QShortcut,
)
from PySide6.QtWidgets import (
    QCheckBox,
    QFileDialog,
    QInputDialog,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QSizePolicy,
    QToolBar,
)

from memecam.config import ConfigError, Settings
from memecam.core.frame_worker import FrameWorker
from memecam.core.types import FrameStats
from memecam.gestures.library import validate_custom_name
from memecam.gestures.recorder import RecorderStatus, RecordPhase
from memecam.gestures.templates import GestureTemplate
from memecam.storage import MEMES_DIR, save_recorded_gesture


class VideoView(QLabel):
    """Displays frames scaled to fit while keeping aspect ratio."""

    def __init__(self) -> None:
        super().__init__("Starting camera…")
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(320, 180)
        self.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
        self.setStyleSheet("background: #111; color: #aaa;")
        self._image: QImage | None = None

    def show_image(self, image: QImage) -> None:
        self._image = image
        self._repaint_pixmap()

    def resizeEvent(self, event: QResizeEvent) -> None:  # noqa: N802 (Qt override)
        super().resizeEvent(event)
        self._repaint_pixmap()

    def _repaint_pixmap(self) -> None:
        if self._image is None:
            return
        pixmap = QPixmap.fromImage(self._image).scaled(
            self.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.setPixmap(pixmap)


class MainWindow(QMainWindow):
    def __init__(self, worker: FrameWorker, settings: Settings) -> None:
        super().__init__()
        self._settings = settings
        self._recording = False
        self.setWindowTitle("MemeCam")
        self.resize(1100, 680)

        self._worker = worker
        self._view = VideoView()
        self.setCentralWidget(self._view)

        toolbar = QToolBar("Controls")
        toolbar.setMovable(False)
        self._debug_box = QCheckBox("Debug overlay (D)")
        self._debug_box.setShortcut(QKeySequence("D"))
        self._debug_box.toggled.connect(self._worker.set_debug)
        toolbar.addWidget(self._debug_box)
        toolbar.addSeparator()
        self._record_action = QAction("Record gesture (R)", self)
        self._record_action.setShortcut(QKeySequence("R"))
        self._record_action.triggered.connect(self._on_record_clicked)
        toolbar.addAction(self._record_action)
        self.addToolBar(toolbar)
        QShortcut(QKeySequence(Qt.Key.Key_Escape), self, activated=self._cancel_recording)

        self._fps_label = QLabel()
        self.statusBar().addPermanentWidget(self._fps_label)

        worker.frame_ready.connect(self._view.show_image)
        worker.stats_ready.connect(self._on_stats)
        worker.reaction_fired.connect(self._on_reaction)
        worker.recording_updated.connect(self._on_recording_updated)
        worker.failed.connect(self._on_failed)

    def enable_debug(self) -> None:
        self._debug_box.setChecked(True)

    @Slot(object)
    def _on_stats(self, stats: FrameStats) -> None:
        self._fps_label.setText(f"{stats.fps:.1f} fps")

    @Slot(str)
    def _on_reaction(self, key: str) -> None:
        self.statusBar().showMessage(f"Reaction: {key}", 2500)

    @Slot(str)
    def _on_failed(self, message: str) -> None:
        self._view.setText("Camera stopped.")
        QMessageBox.critical(self, "MemeCam", message)

    # --- gesture recording ------------------------------------------------------------
    @Slot()
    def _on_record_clicked(self) -> None:
        if self._recording:
            self._cancel_recording()
            return
        name = self._ask_gesture_name()
        if name is None:
            return
        self._set_recording(True)
        self._worker.start_recording(name)

    def _ask_gesture_name(self) -> str | None:
        name = ""
        while True:
            name, ok = QInputDialog.getText(
                self,
                "Record a gesture",
                "Name for the new gesture (lowercase_with_underscores):",
                QLineEdit.EchoMode.Normal,
                name,
            )
            if not ok:
                return None
            name = name.strip()
            if error := validate_custom_name(name):
                QMessageBox.warning(self, "Invalid name", f"'{name}': {error}")
                continue
            if name in self._settings.library:
                answer = QMessageBox.question(
                    self, "Replace gesture?", f"'{name}' already exists. Record it again?"
                )
                if answer != QMessageBox.StandardButton.Yes:
                    continue
            return name

    @Slot()
    def _cancel_recording(self) -> None:
        if self._recording:
            self._worker.cancel_recording()
            self._set_recording(False)
            self.statusBar().showMessage("Recording cancelled", 2500)

    def _set_recording(self, active: bool) -> None:
        self._recording = active
        self._record_action.setText("Cancel recording (Esc)" if active else "Record gesture (R)")

    @Slot(object)
    def _on_recording_updated(self, status: RecorderStatus) -> None:
        if not self._recording:
            return  # a stale status from just before a cancel
        if status.phase is RecordPhase.FAILED:
            self._set_recording(False)
            QMessageBox.warning(self, "Recording failed", status.message)
        elif status.phase is RecordPhase.DONE and status.template is not None:
            self._set_recording(False)
            self._finish_recording(status.template)

    def _finish_recording(self, template: GestureTemplate) -> None:
        image, _ = QFileDialog.getOpenFileName(
            self,
            f"Pick a reaction image for '{template.name}' (Cancel = save the gesture only)",
            str(self._settings.root / MEMES_DIR),
            "PNG images (*.png)",
        )
        try:
            self._settings = save_recorded_gesture(
                self._settings, template, Path(image) if image else None
            )
        except (ConfigError, OSError) as exc:
            QMessageBox.critical(self, "Couldn't save gesture", str(exc))
            return
        self._worker.apply_settings(self._settings)
        linked = "with its reaction" if image else "(no reaction image yet)"
        self.statusBar().showMessage(f"Saved gesture '{template.name}' {linked}", 5000)

    def closeEvent(self, event: QCloseEvent) -> None:  # noqa: N802 (Qt override)
        self._worker.stop()
        super().closeEvent(event)
