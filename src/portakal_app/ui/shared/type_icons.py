"""Orange-compatible variable type badge icons.

Colors match Orange3 source (gui.py __AttributeIconDict):
  - Continuous (numeric): N  red    (202, 0, 32)
  - Discrete (categorical): C green (26, 150, 65)
  - String (text): S  black         (0, 0, 0)
  - Time: T  blue                   (68, 170, 255)
  - Unknown: ?  grey                (128, 128, 128)
"""
from __future__ import annotations

from PySide6.QtCore import QSize, Qt
from PySide6.QtGui import QColor, QFont, QIcon, QPainter, QPixmap

_TYPE_MAP: dict[str, tuple[tuple[int, int, int], str]] = {
    "numeric": ((202, 0, 32), "N"),
    "categorical": ((26, 150, 65), "C"),
    "boolean": ((26, 150, 65), "C"),
    "text": ((0, 0, 0), "S"),
    "string": ((0, 0, 0), "S"),
    "date": ((68, 170, 255), "T"),
    "time": ((68, 170, 255), "T"),
    "datetime": ((68, 170, 255), "T"),
    "duration": ((68, 170, 255), "T"),
}

_UNKNOWN = ((128, 128, 128), "?")


def type_badge_icon(logical_type: str) -> QIcon:
    """Return a small coloured badge QIcon for the given logical type."""
    color_rgb, letter = _TYPE_MAP.get(logical_type, _UNKNOWN)
    size = 13
    pixmap = QPixmap(QSize(size, size))
    pixmap.fill(QColor(0, 0, 0, 0))
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    painter.setPen(Qt.PenStyle.NoPen)
    painter.setBrush(QColor(*color_rgb))
    painter.drawRoundedRect(0, 0, size, size, 2, 2)
    painter.setPen(QColor(Qt.GlobalColor.white))
    font = QFont("Arial", 8, QFont.Weight.Bold)
    painter.setFont(font)
    painter.drawText(pixmap.rect(), Qt.AlignmentFlag.AlignCenter, letter)
    painter.end()
    return QIcon(pixmap)
