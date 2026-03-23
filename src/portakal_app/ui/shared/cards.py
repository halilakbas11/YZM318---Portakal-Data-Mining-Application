from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QFrame, QLabel, QVBoxLayout, QWidget


class SectionHeader(QWidget):
    def __init__(self, title: str, subtitle: str = "", parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        title_label = QLabel(title)
        title_label.setProperty("sectionTitle", True)
        layout.addWidget(title_label)
        if subtitle:
            subtitle_label = QLabel(subtitle)
            subtitle_label.setWordWrap(True)
            subtitle_label.setProperty("muted", True)
            layout.addWidget(subtitle_label)


class EmptyStateCard(QFrame):
    def __init__(self, title: str, message: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setProperty("panel", True)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(8)
        title_label = QLabel(title)
        title_label.setAlignment(Qt.AlignmentFlag.AlignLeft)
        title_label.setProperty("sectionTitle", True)
        body_label = QLabel(message)
        body_label.setWordWrap(True)
        body_label.setProperty("muted", True)
        layout.addWidget(title_label)
        layout.addWidget(body_label)


class InfoCard(QFrame):
    def __init__(self, title: str, value: str, subtitle: str = "", parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setProperty("infoCard", True)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 12, 14, 12)
        layout.setSpacing(6)
        title_label = QLabel(title)
        title_label.setProperty("muted", True)
        value_label = QLabel(value)
        value_label.setStyleSheet("font-size: 18pt; font-weight: 700; color: #3b2a10;")
        layout.addWidget(title_label)
        layout.addWidget(value_label)
        if subtitle:
            subtitle_label = QLabel(subtitle)
            subtitle_label.setWordWrap(True)
            subtitle_label.setProperty("muted", True)
            layout.addWidget(subtitle_label)
