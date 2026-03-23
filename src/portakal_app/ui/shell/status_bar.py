from __future__ import annotations

from PySide6.QtWidgets import QLabel, QStatusBar


class StatusBarController(QStatusBar):
    def __init__(self) -> None:
        super().__init__()
        self._label = QLabel("Ready")
        self.addPermanentWidget(self._label, 1)

    def set_message(self, message: str) -> None:
        self._label.setText(message)
        self.showMessage(message, 3000)

    def current_message(self) -> str:
        return self._label.text()
