"""One clean, dark look for every window: palette + stylesheet."""

from __future__ import annotations

from PySide6.QtGui import QColor, QFont, QPalette
from PySide6.QtWidgets import QApplication

BG = "#15161a"
SURFACE = "#1e2026"
SURFACE_HI = "#262932"
BORDER = "#30333d"
TEXT = "#e8e9ec"
MUTED = "#9aa0ab"
ACCENT = "#6d8cff"
ACCENT_HI = "#86a0ff"
DANGER = "#ff5d5d"
DANGER_HI = "#ff7a7a"

STYLESHEET = f"""
* {{ outline: none; }}
QWidget {{ color: {TEXT}; font-size: 10pt; }}
QMainWindow, QDialog {{ background: {BG}; }}
QToolTip {{ background: {SURFACE_HI}; color: {TEXT}; border: 1px solid {BORDER};
            padding: 4px 6px; border-radius: 6px; }}

#TopBar {{ background: {SURFACE}; border-bottom: 1px solid {BORDER}; }}
#AppTitle {{ font-size: 12pt; font-weight: 600; }}
#Pill {{ background: {SURFACE_HI}; color: {MUTED}; border-radius: 10px; padding: 3px 10px;
         font-size: 9pt; }}
#Hint {{ color: {MUTED}; }}
#Error {{ color: {DANGER}; }}

QPushButton {{ background: {SURFACE_HI}; border: 1px solid {BORDER}; border-radius: 8px;
               padding: 6px 14px; }}
QPushButton:hover {{ background: #2e323c; border-color: #3b3f4b; }}
QPushButton:pressed {{ background: #23262e; }}
QPushButton:disabled {{ color: #5d626d; background: {SURFACE}; border-color: {SURFACE_HI}; }}
QPushButton:checked {{ background: #2b3350; border-color: {ACCENT}; color: {ACCENT_HI}; }}
QPushButton#Primary {{ background: {ACCENT}; border-color: {ACCENT}; color: white;
                       font-weight: 600; }}
QPushButton#Primary:hover {{ background: {ACCENT_HI}; border-color: {ACCENT_HI}; }}
QPushButton#Danger {{ background: {DANGER}; border-color: {DANGER}; color: white;
                      font-weight: 600; }}
QPushButton#Danger:hover {{ background: {DANGER_HI}; border-color: {DANGER_HI}; }}

QLineEdit, QComboBox, QDoubleSpinBox {{ background: {SURFACE}; border: 1px solid {BORDER};
    border-radius: 8px; padding: 5px 8px; selection-background-color: {ACCENT}; }}
QLineEdit:focus, QComboBox:focus, QDoubleSpinBox:focus {{ border-color: {ACCENT}; }}
QComboBox::drop-down {{ border: none; width: 22px; }}
QComboBox QAbstractItemView {{ background: {SURFACE}; border: 1px solid {BORDER};
    selection-background-color: {SURFACE_HI}; selection-color: {TEXT}; padding: 4px; }}
QDoubleSpinBox::up-button, QDoubleSpinBox::down-button {{ width: 16px; border: none; }}

QRadioButton {{ spacing: 6px; }}
QRadioButton::indicator {{ width: 14px; height: 14px; border-radius: 8px;
                           border: 2px solid {BORDER}; background: {SURFACE}; }}
QRadioButton::indicator:checked {{ border-color: {ACCENT}; background: {ACCENT}; }}

QTableWidget {{ background: {SURFACE}; alternate-background-color: #22252c;
                border: 1px solid {BORDER}; border-radius: 10px; gridline-color: transparent;
                selection-background-color: #2b3350; selection-color: {TEXT}; }}
QTableWidget::item {{ padding: 6px 8px; border: none; }}
QHeaderView::section {{ background: {SURFACE}; color: {MUTED}; border: none;
                        border-bottom: 1px solid {BORDER}; padding: 8px; font-weight: 600; }}
QTableCornerButton::section {{ background: {SURFACE}; border: none; }}
QScrollBar:vertical {{ background: transparent; width: 10px; margin: 4px; }}
QScrollBar::handle:vertical {{ background: {BORDER}; border-radius: 3px; min-height: 24px; }}
QScrollBar::add-line, QScrollBar::sub-line {{ height: 0; }}

QMessageBox QLabel {{ min-width: 260px; }}
"""


def apply_theme(app: QApplication) -> None:
    """Fusion style + dark palette (so native bits match) + the stylesheet above."""
    app.setStyle("Fusion")
    pal = QPalette()
    roles = {
        QPalette.ColorRole.Window: BG,
        QPalette.ColorRole.WindowText: TEXT,
        QPalette.ColorRole.Base: SURFACE,
        QPalette.ColorRole.AlternateBase: SURFACE_HI,
        QPalette.ColorRole.Text: TEXT,
        QPalette.ColorRole.Button: SURFACE_HI,
        QPalette.ColorRole.ButtonText: TEXT,
        QPalette.ColorRole.Highlight: ACCENT,
        QPalette.ColorRole.HighlightedText: "#ffffff",
        QPalette.ColorRole.ToolTipBase: SURFACE_HI,
        QPalette.ColorRole.ToolTipText: TEXT,
        QPalette.ColorRole.PlaceholderText: MUTED,
    }
    for role, color in roles.items():
        pal.setColor(role, QColor(color))
    pal.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.ButtonText, QColor("#5d626d"))
    pal.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.Text, QColor("#5d626d"))
    app.setPalette(pal)
    font = QFont(app.font())
    font.setPointSizeF(10)
    app.setFont(font)
    app.setStyleSheet(STYLESHEET)
