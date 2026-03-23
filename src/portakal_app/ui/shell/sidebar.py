from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFontMetrics
from PySide6.QtWidgets import QSizePolicy, QPushButton, QFrame, QListWidget, QListWidgetItem, QVBoxLayout, QWidget

from portakal_app.models import CategoryDefinition


class SidebarCategoryList(QFrame):
    categorySelected = Signal(str)
    workflowInfoRequested = Signal()

    def __init__(self, categories: list[CategoryDefinition], parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("categoryPanel")
        self._categories = categories
        self._items_by_category: dict[str, QListWidgetItem] = {}

        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 10, 6, 10)
        layout.setSpacing(4)

        self._list = QListWidget()
        self._list.setObjectName("categoryList")
        self._list.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._list.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._list.setTextElideMode(Qt.TextElideMode.ElideNone)
        self._list.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self._list.currentItemChanged.connect(self._emit_selected_category)
        layout.addWidget(self._list)

        for category in self._categories:
            item = QListWidgetItem(category.label)
            item.setData(Qt.ItemDataRole.UserRole, category.id)
            self._list.addItem(item)
            self._items_by_category[category.id] = item

        layout.addStretch(1)
        self._workflow_info_button = QPushButton("Workflow Info")
        self._workflow_info_button.setObjectName("workflowInfoButton")
        self._workflow_info_button.clicked.connect(self.workflowInfoRequested.emit)
        layout.addWidget(self._workflow_info_button)

        self._sync_list_height()

    def _emit_selected_category(self, current: QListWidgetItem | None, _previous: QListWidgetItem | None) -> None:
        if current is None:
            return
        category_id = current.data(Qt.ItemDataRole.UserRole)
        self.categorySelected.emit(category_id)

    def set_current_category(self, category_id: str) -> None:
        item = self._items_by_category.get(category_id)
        if item is not None:
            self._list.setCurrentItem(item)

    def current_category_id(self) -> str | None:
        item = self._list.currentItem()
        return item.data(Qt.ItemDataRole.UserRole) if item is not None else None

    def recommended_width(self) -> int:
        metrics = QFontMetrics(self._list.font())
        longest_label = max((category.label for category in self._categories), key=len, default="")
        text_width = metrics.horizontalAdvance(longest_label)
        layout_margins = self.layout().contentsMargins()
        frame_width = layout_margins.left() + layout_margins.right()
        item_padding = 92
        return max(188, text_width + frame_width + item_padding)

    def _sync_list_height(self) -> None:
        if self._list.count() == 0:
            return
        row_height = self._list.sizeHintForRow(0)
        spacing = 2
        frame = self._list.frameWidth() * 2
        viewport_margins = 8
        total = (row_height * self._list.count()) + (spacing * (self._list.count() - 1)) + frame + viewport_margins
        self._list.setFixedHeight(total)
