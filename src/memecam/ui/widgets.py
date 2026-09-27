"""Small reusable widgets: the video view and fading toast messages."""

from __future__ import annotations

from PySide6.QtCore import QEasingCurve, QPropertyAnimation, QRect, Qt, QTimer
from PySide6.QtGui import QColor, QImage, QPainter, QPaintEvent, QResizeEvent
from PySide6.QtWidgets import QGraphicsOpacityEffect, QLabel, QSizePolicy, QWidget

from memecam.ui import theme


class VideoView(QWidget):
    """Paints the latest frame letterboxed to fit, with smooth scaling.

    Painting the QImage directly (instead of converting to a QPixmap and scaling it on every
    frame) keeps the preview cheap and smooth, especially when the window is resized.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setMinimumSize(320, 180)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setAttribute(Qt.WidgetAttribute.WA_OpaquePaintEvent)
        self._image: QImage | None = None
        self._message = "Starting camera…"
        self.toast = Toast(self)

    def show_image(self, image: QImage) -> None:
        self._image = image
        self._message = ""
        self.update()

    def set_message(self, text: str) -> None:
        """Replace the video with a centered message (e.g. 'Camera stopped.')."""
        self._image = None
        self._message = text
        self.update()

    def image_rect(self) -> QRect:
        """Where the frame is drawn inside the widget (letterboxed)."""
        if self._image is None or self._image.isNull():
            return self.rect()
        iw, ih = self._image.width(), self._image.height()
        scale = min(self.width() / iw, self.height() / ih)
        w, h = round(iw * scale), round(ih * scale)
        return QRect((self.width() - w) // 2, (self.height() - h) // 2, w, h)

    def paintEvent(self, event: QPaintEvent) -> None:  # noqa: N802 (Qt override)
        p = QPainter(self)
        p.fillRect(self.rect(), QColor(theme.BG))
        if self._image is not None and not self._image.isNull():
            p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
            p.drawImage(self.image_rect(), self._image)
        elif self._message:
            p.setPen(QColor(theme.MUTED))
            p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, self._message)
        p.end()

    def resizeEvent(self, event: QResizeEvent) -> None:  # noqa: N802 (Qt override)
        super().resizeEvent(event)
        self.toast.reposition()


class Toast(QLabel):
    """A short message that fades in near the bottom of its parent, then fades out."""

    def __init__(self, parent: QWidget) -> None:
        super().__init__(parent)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.setStyleSheet(
            f"background: rgba(20, 21, 26, 220); color: {theme.TEXT};"
            f"border: 1px solid {theme.BORDER}; border-radius: 12px; padding: 8px 16px;"
        )
        self._opacity = QGraphicsOpacityEffect(self)
        self._opacity.setOpacity(0.0)
        self.setGraphicsEffect(self._opacity)
        self._fade = QPropertyAnimation(self._opacity, b"opacity", self)
        self._fade.setEasingCurve(QEasingCurve.Type.OutCubic)
        self._hide_timer = QTimer(self)
        self._hide_timer.setSingleShot(True)
        self._hide_timer.timeout.connect(lambda: self._animate_to(0.0, 400))
        self.hide()

    def show_message(self, text: str, ms: int = 2500) -> None:
        self.setText(text)
        self.adjustSize()
        self.reposition()
        self.show()
        self.raise_()
        self._animate_to(1.0, 180)
        self._hide_timer.start(ms)

    def reposition(self) -> None:
        parent = self.parentWidget()
        if parent is None:
            return
        self.setMaximumWidth(max(200, parent.width() - 40))
        self.adjustSize()
        x = (parent.width() - self.width()) // 2
        y = parent.height() - self.height() - 24
        self.move(max(0, x), max(0, y))

    def _animate_to(self, value: float, ms: int) -> None:
        self._fade.stop()
        self._fade.setDuration(ms)
        self._fade.setStartValue(self._opacity.opacity())
        self._fade.setEndValue(value)
        self._fade.start()
