from __future__ import annotations

from PySide6.QtWidgets import QVBoxLayout, QWidget

from portakal_app.ui.shared.cards import EmptyStateCard, SectionHeader


class PlaceholderScreen(QWidget):
    def __init__(self, title: str, message: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(16)
        layout.addWidget(SectionHeader(title, "This module is intentionally left as a placeholder for another team."))
        layout.addWidget(EmptyStateCard(title, message))
        layout.addStretch(1)
