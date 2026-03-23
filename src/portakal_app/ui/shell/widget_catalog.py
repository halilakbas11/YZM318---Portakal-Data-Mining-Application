from __future__ import annotations

from collections.abc import Iterable

from PySide6.QtCore import QMimeData, QPoint, QSize, Qt, QTimer, Signal
from PySide6.QtGui import QDrag
from PySide6.QtWidgets import QFrame, QGridLayout, QLabel, QLineEdit, QPushButton, QSizePolicy, QVBoxLayout, QWidget

from portakal_app.models import WidgetDefinition
from portakal_app.ui.icons import get_widget_icon


class WidgetCatalogButton(QPushButton):
    activateRequested = Signal(str)

    def __init__(self, widget_definition: WidgetDefinition, parent: QWidget | None = None) -> None:
        label = widget_definition.label
        if not widget_definition.enabled:
            label = f"{label}\nComing soon"
        super().__init__(label, parent)
        self._widget_definition = widget_definition
        self._drag_start_position = QPoint()
        self._drag_in_progress = False
        self.widget_id = widget_definition.id
        self._activate_timer = QTimer(self)
        self._activate_timer.setSingleShot(True)
        self._activate_timer.setInterval(180)
        self._activate_timer.timeout.connect(lambda: self.activateRequested.emit(self.widget_id))
        self.setProperty("card", True)
        self.setProperty("comingSoon", not widget_definition.enabled)
        self.setMinimumHeight(72)
        self.setMaximumHeight(76)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setToolTip(widget_definition.description)
        icon = get_widget_icon(widget_definition.icon_name)
        if icon is not None:
            self.setIcon(icon)
            self.setIconSize(QSize(24, 24))
        self.setStyleSheet("text-align: left; padding-top: 8px;")

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag_start_position = event.position().toPoint()
            self._drag_in_progress = False
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:
        if not self._widget_definition.enabled:
            return super().mouseMoveEvent(event)
        if not (event.buttons() & Qt.MouseButton.LeftButton):
            return super().mouseMoveEvent(event)
        current_position = event.position().toPoint()
        if (current_position - self._drag_start_position).manhattanLength() < 10:
            return super().mouseMoveEvent(event)
        self._activate_timer.stop()
        self._drag_in_progress = True
        drag = QDrag(self)
        mime_data = QMimeData()
        mime_data.setData("application/x-portakal-widget", self.widget_id.encode("utf-8"))
        drag.setMimeData(mime_data)
        drag.setHotSpot(current_position)
        drag.setPixmap(self.grab())
        drag.exec(Qt.DropAction.CopyAction)

    def mouseReleaseEvent(self, event) -> None:
        super().mouseReleaseEvent(event)
        if not self._widget_definition.enabled:
            return
        if event.button() != Qt.MouseButton.LeftButton:
            return
        if self._drag_in_progress:
            self._drag_in_progress = False
            return
        if self.rect().contains(event.position().toPoint()):
            self._activate_timer.start()

    def mouseDoubleClickEvent(self, event) -> None:
        self._activate_timer.stop()
        if self._widget_definition.enabled and event.button() == Qt.MouseButton.LeftButton:
            self.activateRequested.emit(self.widget_id)
            event.accept()
            return
        super().mouseDoubleClickEvent(event)


class WidgetCatalogPanel(QFrame):
    widgetSelected = Signal(str)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("catalogPanel")
        self._widget_definitions: list[WidgetDefinition] = []

        outer_layout = QVBoxLayout(self)
        outer_layout.setContentsMargins(14, 14, 14, 14)
        outer_layout.setSpacing(12)

        self._title = QLabel("Data")
        self._title.setProperty("sectionTitle", True)
        outer_layout.addWidget(self._title)

        self._search = QLineEdit()
        self._search.setObjectName("catalogSearch")
        self._search.setPlaceholderText("Filter widgets...")
        self._search.textChanged.connect(self._render_grid)
        outer_layout.addWidget(self._search)

        self._content = QWidget()
        outer_layout.addWidget(self._content, 1)
        self._grid = QGridLayout(self._content)
        self._grid.setContentsMargins(0, 0, 0, 0)
        self._grid.setHorizontalSpacing(6)
        self._grid.setVerticalSpacing(6)

    def set_widgets(self, category_label: str, widgets: Iterable[WidgetDefinition]) -> None:
        self._title.setText(category_label)
        self._widget_definitions = list(widgets)
        self._search.clear()
        self._render_grid()

    def current_widget_ids(self) -> list[str]:
        return [widget.id for widget in self._widget_definitions]

    def _render_grid(self) -> None:
        while self._grid.count():
            item = self._grid.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

        query = self._search.text().strip().lower()
        filtered = [
            widget
            for widget in self._widget_definitions
            if not query or query in widget.label.lower() or query in widget.description.lower()
        ]

        for index, widget_definition in enumerate(filtered):
            button = WidgetCatalogButton(widget_definition)
            button.activateRequested.connect(self.widgetSelected.emit)
            row = index // 2
            column = index % 2
            self._grid.addWidget(button, row, column)

        row_count = max(1, (len(filtered) + 1) // 2)
        for row in range(row_count):
            self._grid.setRowStretch(row, 0)
        self._grid.setRowStretch(row_count, 1)
        self._grid.setColumnStretch(0, 1)
        self._grid.setColumnStretch(1, 1)
