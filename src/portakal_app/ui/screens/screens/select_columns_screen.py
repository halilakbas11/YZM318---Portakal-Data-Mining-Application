from __future__ import annotations

from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QVBoxLayout,
    QWidget,
)
from PySide6.QtGui import QIcon, QPixmap, QPainter, QColor, QFont, QDrag
from PySide6.QtCore import Qt, QRectF, QMimeData, Signal

from portakal_app.data.models import DatasetHandle
from portakal_app.data.services.select_columns_service import SelectColumnsService
from portakal_app.ui import i18n
from portakal_app.ui.screens.node_screen import WorkflowNodeScreenSupport


def _create_type_icon(logical_type: str) -> QIcon:
    pixmap = QPixmap(16, 16)
    pixmap.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)

    if logical_type == "numeric":
        color = QColor("#ef4444")
        text = "N"
    elif logical_type in ("categorical", "boolean"):
        color = QColor("#22c55e")
        text = "C"
    elif logical_type in ("text", "string"):
        color = QColor("#8b5cf6")
        text = "S"
    elif logical_type in ("datetime", "date", "time"):
        color = QColor("#3b82f6")
        text = "D"
    else:
        color = QColor("#6b7280")
        text = "?"

    painter.setBrush(color)
    painter.setPen(Qt.PenStyle.NoPen)
    painter.drawRoundedRect(0, 0, 16, 16, 3, 3)

    painter.setPen(QColor("white"))
    font = QFont("Arial", 9, QFont.Weight.Bold)
    painter.setFont(font)
    painter.drawText(QRectF(0, 0, 16, 16), Qt.AlignmentFlag.AlignCenter, text)
    painter.end()

    return QIcon(pixmap)


# ─────────────────────────────────────────────────────────────────────────────
#  Drag & Drop enabled list widget
# ─────────────────────────────────────────────────────────────────────────────

_COLUMN_MIME = "application/x-portakal-column-items"


class DraggableColumnList(QListWidget):
    """
    A QListWidget subclass that supports:
    • Drag-and-drop between sibling list widgets (move items)
    • Internal reordering (drag within the same list)
    • Extended multi-selection

    Inspired by Orange3's ``VariablesListItemView`` but built entirely with
    Qt's built-in ``QListWidget`` drag-drop infrastructure.
    """

    itemsDropped = Signal()  # emitted after a successful drop

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setSelectionMode(QListWidget.SelectionMode.ExtendedSelection)
        self.setDragEnabled(True)
        self.setAcceptDrops(True)
        self.setDropIndicatorShown(True)
        self.setDragDropMode(QAbstractItemView.DragDropMode.DragDrop)
        self.setDefaultDropAction(Qt.DropAction.MoveAction)

    # ── Drag start ──────────────────────────────────────────────────────
    def startDrag(self, supportedActions):
        items = self.selectedItems()
        if not items:
            return

        mime = QMimeData()
        names = [item.text() for item in items]
        types = [item.data(Qt.ItemDataRole.UserRole) or "" for item in items]
        # Encode as plain text lines: "name\ttype"
        mime.setData(_COLUMN_MIME, b"")
        mime.setProperty("_names", names)
        mime.setProperty("_types", types)
        mime.setProperty("_source", self)

        drag = QDrag(self)
        drag.setMimeData(mime)
        result = drag.exec(Qt.DropAction.MoveAction)

        if result == Qt.DropAction.MoveAction:
            # Remove dragged items from source (this list)
            for item in items:
                row = self.row(item)
                if row >= 0:
                    self.takeItem(row)
            self.itemsDropped.emit()

    # ── Drop receive ────────────────────────────────────────────────────
    def dragEnterEvent(self, event):
        if event.mimeData().hasFormat(_COLUMN_MIME):
            event.acceptProposedAction()
        else:
            event.ignore()

    def dragMoveEvent(self, event):
        if event.mimeData().hasFormat(_COLUMN_MIME):
            event.acceptProposedAction()
        else:
            event.ignore()

    def dropEvent(self, event):
        mime = event.mimeData()
        if not mime.hasFormat(_COLUMN_MIME):
            event.ignore()
            return

        names = mime.property("_names")
        types = mime.property("_types")
        source = mime.property("_source")

        if names is None:
            event.ignore()
            return

        # Figure out insertion row from drop position
        drop_row = self.indexAt(event.position().toPoint()).row()
        if drop_row < 0:
            drop_row = self.count()

        for name, logical_type in zip(names, types):
            item = QListWidgetItem(name)
            item.setIcon(_create_type_icon(logical_type))
            item.setData(Qt.ItemDataRole.UserRole, logical_type)
            self.insertItem(drop_row, item)
            drop_row += 1

        event.acceptProposedAction()
        self.itemsDropped.emit()


# ─────────────────────────────────────────────────────────────────────────────
#  Filterable column group (QGroupBox + filter QLineEdit + DraggableColumnList)
# ─────────────────────────────────────────────────────────────────────────────

class FilterableColumnGroup(QWidget):
    """
    A composite widget that wraps a ``DraggableColumnList`` with:
    • A titled ``QGroupBox`` frame
    • A ``QLineEdit`` filter (like Orange's *Filter* box)

    Filtering hides non-matching items rather than removing them so the
    underlying data list remains intact.
    """

    def __init__(self, title: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self._group = QGroupBox(title)
        group_layout = QVBoxLayout(self._group)
        group_layout.setContentsMargins(6, 6, 6, 6)
        group_layout.setSpacing(4)

        self._filter = QLineEdit()
        self._filter.setPlaceholderText(i18n.t("Filter"))
        self._filter.setClearButtonEnabled(True)
        self._filter.textChanged.connect(self._apply_filter)
        group_layout.addWidget(self._filter)

        self.list_widget = DraggableColumnList()
        group_layout.addWidget(self.list_widget)

        layout.addWidget(self._group)

    # ── Public helpers ──────────────────────────────────────────────────
    @property
    def title(self) -> str:
        return self._group.title()

    def set_title(self, title: str) -> None:
        self._group.setTitle(title)

    def update_title_count(self, base_title: str) -> None:
        total = self.list_widget.count()
        visible = sum(
            1
            for i in range(total)
            if not self.list_widget.item(i).isHidden()
        )
        if self._filter.text():
            self._group.setTitle(f"{base_title} ({visible}/{total})")
        elif total:
            self._group.setTitle(f"{base_title} ({total})")
        else:
            self._group.setTitle(base_title)

    # ── Filter logic ────────────────────────────────────────────────────
    def _apply_filter(self, text: str) -> None:
        needle = text.lower().strip()
        for i in range(self.list_widget.count()):
            item = self.list_widget.item(i)
            item.setHidden(
                needle != "" and needle not in item.text().lower()
            )


# ─────────────────────────────────────────────────────────────────────────────
#  The actual Select Columns screen
# ─────────────────────────────────────────────────────────────────────────────

class SelectColumnsScreen(QWidget, WorkflowNodeScreenSupport):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._init_workflow_node_support()
        self._service = SelectColumnsService()
        self._dataset_handle: DatasetHandle | None = None
        self._output_dataset: DatasetHandle | None = None
        self._saved_roles: dict[str, str] = {}

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(10)

        self._dataset_label = QLabel(i18n.t("Dataset: none"))
        self._dataset_label.setProperty("sectionTitle", True)
        self._dataset_label.setStyleSheet("font-size: 12pt; background: transparent;")
        layout.addWidget(self._dataset_label)

        main_boxes_layout = QHBoxLayout()

        # ── LEFT PANE: Ignored (with filter) ────────────────────────────
        self._ignored_group = FilterableColumnGroup(i18n.t("Ignored"))
        self._ignored_list = self._ignored_group.list_widget
        self._ignored_list.itemsDropped.connect(self._on_lists_changed)
        main_boxes_layout.addWidget(self._ignored_group, 1)

        # ── MIDDLE PANE: Arrow buttons ──────────────────────────────────
        btn_layout = QVBoxLayout()
        btn_layout.addStretch(1)

        self._to_features = QPushButton(i18n.t("Features >"))
        self._to_features.clicked.connect(lambda: self._move_selected("features"))
        btn_layout.addWidget(self._to_features)

        self._to_target = QPushButton(i18n.t("Target >"))
        self._to_target.clicked.connect(lambda: self._move_selected("target"))
        btn_layout.addWidget(self._to_target)

        self._to_meta = QPushButton(i18n.t("Meta >"))
        self._to_meta.clicked.connect(lambda: self._move_selected("meta"))
        btn_layout.addWidget(self._to_meta)

        btn_layout.addSpacing(20)

        self._to_ignored = QPushButton(i18n.t("< Ignored"))
        self._to_ignored.clicked.connect(lambda: self._move_selected("ignored"))
        btn_layout.addWidget(self._to_ignored)

        btn_layout.addStretch(1)
        main_boxes_layout.addLayout(btn_layout)

        # ── RIGHT PANE: Features (with filter), Target, Meta ────────────
        right_panel = QVBoxLayout()

        self._features_group = FilterableColumnGroup(i18n.t("Features"))
        self._features_list = self._features_group.list_widget
        self._features_list.itemsDropped.connect(self._on_lists_changed)
        right_panel.addWidget(self._features_group, 3)

        target_group = QGroupBox(i18n.t("Target"))
        target_layout = QVBoxLayout(target_group)
        target_layout.setContentsMargins(6, 6, 6, 6)
        self._target_list = DraggableColumnList()
        self._target_list.itemsDropped.connect(self._on_lists_changed)
        target_layout.addWidget(self._target_list)
        right_panel.addWidget(target_group, 1)

        meta_group = QGroupBox(i18n.t("Meta"))
        meta_layout = QVBoxLayout(meta_group)
        meta_layout.setContentsMargins(6, 6, 6, 6)
        self._meta_list = DraggableColumnList()
        self._meta_list.itemsDropped.connect(self._on_lists_changed)
        meta_layout.addWidget(self._meta_list)
        right_panel.addWidget(meta_group, 1)

        main_boxes_layout.addLayout(right_panel, 1)
        layout.addLayout(main_boxes_layout, 1)

        # ── Result info ─────────────────────────────────────────────────
        self._result_label = QLabel("")
        self._result_label.setWordWrap(True)
        self._result_label.setStyleSheet("font-size: 9pt; color: #6b5d50; background: transparent;")
        layout.addWidget(self._result_label)

        # ── Bottom bar ──────────────────────────────────────────────────
        bottom_layout = QHBoxLayout()
        self._reset_btn = QPushButton(i18n.t("Reset"))
        self._reset_btn.clicked.connect(self._reset_columns)
        bottom_layout.addWidget(self._reset_btn)

        self.cb_apply_auto = QCheckBox(i18n.t("Apply Automatically"))
        self.cb_apply_auto.setChecked(True)
        bottom_layout.addWidget(self.cb_apply_auto)

        bottom_layout.addStretch(1)
        self._apply_button = QPushButton(i18n.t("Apply"))
        self._apply_button.setProperty("primary", True)
        self._apply_button.clicked.connect(self._apply)
        bottom_layout.addWidget(self._apply_button)

        layout.addLayout(bottom_layout)

    # ── i18n ────────────────────────────────────────────────────────────
    def refresh_translations(self) -> None:
        if self._dataset_handle is None:
            self._dataset_label.setText(i18n.t("Dataset: none"))
        else:
            self._dataset_label.setText(
                i18n.tf("Dataset: {name}", name=self._dataset_handle.display_name)
            )

    # ── Data pipeline ───────────────────────────────────────────────────
    def set_input_payload(self, payload) -> None:
        dataset = payload.dataset if payload is not None else None
        self._dataset_handle = dataset
        self._output_dataset = None
        self._reset_columns()
        if self._dataset_handle is not None:
            self._apply()

    def current_output_dataset(self) -> DatasetHandle | None:
        return self._output_dataset

    def serialize_node_state(self) -> dict[str, object]:
        return {
            "features": _list_items(self._features_list),
            "target": _list_items(self._target_list),
            "metas": _list_items(self._meta_list),
            "ignored": _list_items(self._ignored_list),
            "auto_apply": self.cb_apply_auto.isChecked(),
        }

    def restore_node_state(self, payload: dict[str, object]) -> None:
        self._saved_roles = {}
        for col_name in payload.get("features", []):
            self._saved_roles[col_name] = "feature"
        for col_name in payload.get("target", []):
            self._saved_roles[col_name] = "target"
        for col_name in payload.get("metas", []):
            self._saved_roles[col_name] = "meta"
        for col_name in payload.get("ignored", []):
            self._saved_roles[col_name] = "ignored"

        self.cb_apply_auto.setChecked(bool(payload.get("auto_apply", True)))

        if self._dataset_handle:
            self._reset_columns()

    def help_text(self) -> str:
        return (
            "Assign columns to Features, Target, Meta, or Ignored roles "
            "using drag-and-drop or the arrow buttons. "
            "Use the Filter boxes to search within large column lists."
        )

    def documentation_url(self) -> str:
        return "https://orangedatamining.com/widget-catalog/transform/selectcolumns/"

    # ── Internal helpers ────────────────────────────────────────────────
    def _check_auto_apply(self) -> None:
        if self._is_auto_apply() and self._dataset_handle is not None:
            self._apply()

    def _on_lists_changed(self) -> None:
        """Called after any drag-drop completes so we can update counts."""
        self._update_group_titles()
        self._check_auto_apply()

    def _update_group_titles(self) -> None:
        self._ignored_group.update_title_count(i18n.t("Ignored"))
        self._features_group.update_title_count(i18n.t("Features"))

    def _reset_columns(self) -> None:
        self._ignored_list.clear()
        self._features_list.clear()
        self._target_list.clear()
        self._meta_list.clear()

        if self._dataset_handle is None:
            self._dataset_label.setText(i18n.t("Dataset: none"))
            self._update_group_titles()
            return

        self._dataset_label.setText(
            i18n.tf("Dataset: {name}", name=self._dataset_handle.display_name)
        )
        for col in self._dataset_handle.domain.columns:
            item = QListWidgetItem(col.name)
            item.setIcon(_create_type_icon(col.logical_type))
            item.setData(Qt.ItemDataRole.UserRole, col.logical_type)

            role = self._saved_roles.get(col.name, col.role)
            if role == "target":
                self._target_list.addItem(item)
            elif role in ("meta", "metas"):
                self._meta_list.addItem(item)
            elif role == "ignored":
                self._ignored_list.addItem(item)
            else:
                self._features_list.addItem(item)

        self._update_group_titles()

    def _move_selected(self, target: str) -> None:
        target_list_map = {
            "ignored": self._ignored_list,
            "features": self._features_list,
            "target": self._target_list,
            "meta": self._meta_list,
        }
        target_list = target_list_map[target]

        for source_list in [
            self._ignored_list,
            self._features_list,
            self._target_list,
            self._meta_list,
        ]:
            if source_list is target_list:
                continue
            selected = source_list.selectedItems()
            for item in selected:
                row = source_list.row(item)
                taken_item = source_list.takeItem(row)
                target_list.addItem(taken_item)

        self._update_group_titles()
        self._check_auto_apply()

    def _apply(self) -> None:
        if self._dataset_handle is None:
            self._output_dataset = None
            self._result_label.setText("")
            self._notify_output_changed()
            return

        target_names = _list_items(self._target_list)
        meta_names = _list_items(self._meta_list)
        feature_names = _list_items(self._features_list)
        ignored_names = _list_items(self._ignored_list)

        self._output_dataset = self._service.select(
            self._dataset_handle,
            features=feature_names,
            target=target_names,
            metas=meta_names,
        )

        kept = len(target_names) + len(meta_names) + len(feature_names)
        dropped = len(ignored_names)
        parts: list[str] = []
        if target_names:
            parts.append(i18n.tf("Target: {col}", col=target_names[0]))
        parts.append(i18n.tf("{n} features", n=len(feature_names)))
        if meta_names:
            parts.append(i18n.tf("{n} meta", n=len(meta_names)))
        if dropped:
            parts.append(i18n.tf("{n} ignored (dropped)", n=dropped))
        self._result_label.setText(" | ".join(parts))
        self._notify_output_changed()


def _list_items(list_widget: QListWidget) -> list[str]:
    return [list_widget.item(i).text() for i in range(list_widget.count())]
