"""Main window: shows the processed feed and hosts the debug toggle."""

from __future__ import annotations

from PySide6.QtCore import Qt, Slot
from PySide6.QtGui import QCloseEvent, QImage, QKeySequence, QPixmap, QResizeEvent
from PySide6.QtWidgets import QCheckBox, QLabel, QMainWindow, QMessageBox, QSizePolicy, QToolBar

from memecam.core.frame_worker import FrameWorker
from memecam.core.types import FrameStats


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
    def __init__(self, worker: FrameWorker) -> None:
        super().__init__()
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
        self.addToolBar(toolbar)

        self._fps_label = QLabel()
        self.statusBar().addPermanentWidget(self._fps_label)

        worker.frame_ready.connect(self._view.show_image)
        worker.stats_ready.connect(self._on_stats)
        worker.reaction_fired.connect(self._on_reaction)
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

    def closeEvent(self, event: QCloseEvent) -> None:  # noqa: N802 (Qt override)
        self._worker.stop()
        super().closeEvent(event)
