from __future__ import annotations

from PySide6.QtWidgets import QFrame, QStackedWidget, QVBoxLayout, QWidget


class ContentHost(QFrame):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("contentPanel")
        self._screens: dict[str, QWidget] = {}

        layout = QVBoxLayout(self)
        layout.setContentsMargins(18, 18, 18, 18)
        layout.setSpacing(0)

        self._stack = QStackedWidget()
        layout.addWidget(self._stack)

    def register_screen(self, widget_id: str, screen: QWidget) -> None:
        if widget_id in self._screens:
            return
        self._screens[widget_id] = screen
        self._stack.addWidget(screen)

    def show_widget(self, widget_id: str) -> None:
        screen = self._screens.get(widget_id)
        if screen is None:
            raise KeyError(f"Unknown widget id: {widget_id}")
        self._stack.setCurrentWidget(screen)

    def current_widget(self) -> QWidget:
        return self._stack.currentWidget()
