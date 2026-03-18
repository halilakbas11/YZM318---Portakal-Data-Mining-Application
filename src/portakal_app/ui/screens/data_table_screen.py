from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PySide6.QtCore import QAbstractTableModel, QModelIndex, Qt
from PySide6.QtGui import QColor, QPainter, QPalette, QBrush
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QCheckBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QSplitter,
    QStackedWidget,
    QStyle,
    QStyledItemDelegate,
    QStyleOptionViewItem,
    QTableView,
    QVBoxLayout,
    QWidget,
)

from portakal_app.data.models import DatasetHandle, PreviewPage
from portakal_app.data.services.preview_service import PreviewService


TYPE_COLORS = {
    "numeric": "#4db7eb",
    "categorical": "#aeb0b4",
    "text": "#cab56d",
    "datetime": "#7d9de8",
}

ROW_CLASS_COLORS = ["#eef1f4", "#f3eee6", "#ece8f7", "#eef7eb", "#f8eded"]
DEFAULT_PAGE_SIZE = 500


@dataclass
class DataTableColumn:
    name: str
    type_name: str
    role_name: str
    values_preview: str = ""


def _describe_dataset(dataset_handle: Any) -> str:
    if dataset_handle is None:
        return "none"
    if isinstance(dataset_handle, str):
        return Path(dataset_handle).name
    for attr in ("display_name", "name", "dataset_id", "id"):
        value = getattr(dataset_handle, attr, None)
        if value:
            return str(value)
    return dataset_handle.__class__.__name__


class DataTableModel(QAbstractTableModel):
    BAR_VALUE_ROLE = Qt.ItemDataRole.UserRole + 1
    BAR_RANGE_ROLE = Qt.ItemDataRole.UserRole + 2

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._headers: list[str] = []
        self._columns: list[DataTableColumn] = []
        self._all_rows: list[list[str]] = []
        self._display_rows: list[list[str]] = []
        self._numeric_ranges: dict[int, tuple[float, float]] = {}
        self._class_colors: dict[str, QColor] = {}
        self._target_column_index: int | None = None
        self._show_labels = True
        self._color_by_classes = True

    def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:  # noqa: N802
        if parent.isValid():
            return 0
        return len(self._display_rows)

    def columnCount(self, parent: QModelIndex = QModelIndex()) -> int:  # noqa: N802
        if parent.isValid():
            return 0
        return len(self._headers)

    def data(self, index: QModelIndex, role: int = Qt.ItemDataRole.DisplayRole) -> Any:
        if not index.isValid():
            return None

        row = self._display_rows[index.row()]
        value = row[index.column()] if index.column() < len(row) else ""
        column = self._columns[index.column()]

        if role == Qt.ItemDataRole.DisplayRole:
            return value
        if role == Qt.ItemDataRole.TextAlignmentRole and column.type_name == "numeric":
            return int(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        if role == Qt.ItemDataRole.BackgroundRole and self._color_by_classes:
            class_value = self._row_class_value(row)
            if class_value in self._class_colors:
                return self._class_colors[class_value]
        if role == self.BAR_VALUE_ROLE and column.type_name == "numeric" and self._is_float(value):
            return float(value)
        if role == self.BAR_RANGE_ROLE:
            return self._numeric_ranges.get(index.column())
        return None

    def headerData(self, section: int, orientation: Qt.Orientation, role: int = Qt.ItemDataRole.DisplayRole) -> Any:  # noqa: N802
        if role != Qt.ItemDataRole.DisplayRole:
            return None
        if orientation == Qt.Orientation.Horizontal:
            if not (0 <= section < len(self._headers)):
                return None
            if self._show_labels:
                column = self._columns[section]
                return f"{column.name}\n{column.type_name}"
            return self._headers[section]
        if orientation == Qt.Orientation.Vertical:
            return str(section + 1)
        return None

    def sort(self, column: int, order: Qt.SortOrder = Qt.SortOrder.AscendingOrder) -> None:
        if not (0 <= column < len(self._headers)):
            return
        reverse = order == Qt.SortOrder.DescendingOrder

        def sort_key(row: list[str]) -> tuple[int, Any]:
            value = row[column] if column < len(row) else ""
            if self._columns[column].type_name == "numeric" and self._is_float(value):
                return (0, float(value))
            return (1, value.lower())

        self.layoutAboutToBeChanged.emit()
        self._display_rows.sort(key=sort_key, reverse=reverse)
        self.layoutChanged.emit()

    def set_dataset(
        self,
        headers: list[str],
        columns: list[DataTableColumn],
        rows: list[list[str]],
        numeric_ranges: dict[int, tuple[float, float]],
        target_column_index: int | None,
    ) -> None:
        self.beginResetModel()
        self._headers = headers
        self._columns = columns
        self._all_rows = rows
        self._display_rows = list(rows)
        self._numeric_ranges = numeric_ranges
        self._target_column_index = target_column_index
        self._class_colors = self._build_class_color_map()
        self.endResetModel()

    def clear(self) -> None:
        self.beginResetModel()
        self._headers = []
        self._columns = []
        self._all_rows = []
        self._display_rows = []
        self._numeric_ranges = {}
        self._class_colors = {}
        self._target_column_index = None
        self.endResetModel()

    def restore_original_order(self) -> None:
        self.layoutAboutToBeChanged.emit()
        self._display_rows = list(self._all_rows)
        self.layoutChanged.emit()

    def set_show_labels(self, enabled: bool) -> None:
        self._show_labels = enabled
        if self._headers:
            self.headerDataChanged.emit(Qt.Orientation.Horizontal, 0, len(self._headers) - 1)

    def set_color_by_classes(self, enabled: bool) -> None:
        self._color_by_classes = enabled
        if self.rowCount() and self.columnCount():
            top_left = self.index(0, 0)
            bottom_right = self.index(self.rowCount() - 1, self.columnCount() - 1)
            self.dataChanged.emit(top_left, bottom_right, [Qt.ItemDataRole.BackgroundRole])

    def rows_for_indexes(self, row_indexes: list[int]) -> list[list[str]]:
        return [self._display_rows[index] for index in row_indexes if 0 <= index < len(self._display_rows)]

    def all_display_rows(self) -> list[list[str]]:
        return self._display_rows

    def _build_class_color_map(self) -> dict[str, QColor]:
        if self._target_column_index is None:
            return {}
        mapping: dict[str, QColor] = {}
        color_index = 0
        for row in self._all_rows:
            value = self._row_class_value(row)
            if not value or value in mapping:
                continue
            mapping[value] = QColor(ROW_CLASS_COLORS[color_index % len(ROW_CLASS_COLORS)])
            color_index += 1
        return mapping

    def _row_class_value(self, row: list[str]) -> str:
        if self._target_column_index is None or self._target_column_index >= len(row):
            return ""
        return row[self._target_column_index].strip()

    def _is_float(self, value: str) -> bool:
        try:
            float(value)
        except ValueError:
            return False
        return True


class NumericBarDelegate(QStyledItemDelegate):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._enabled = True

    def set_enabled(self, enabled: bool) -> None:
        self._enabled = enabled

    def paint(self, painter: QPainter, option, index) -> None:
        styled_option = QStyleOptionViewItem(option)
        self.initStyleOption(styled_option, index)

        if option.state & QStyle.StateFlag.State_Selected:
            selection_color = QColor("#8fc8ff")
            painter.save()
            painter.fillRect(option.rect, selection_color)
            painter.setPen(QColor("#10202f"))
            text_rect = option.rect.adjusted(8, 0, -8, 0)
            alignment = Qt.AlignmentFlag.AlignVCenter
            if index.data(Qt.ItemDataRole.TextAlignmentRole):
                alignment = Qt.AlignmentFlag(index.data(Qt.ItemDataRole.TextAlignmentRole))
            else:
                alignment |= Qt.AlignmentFlag.AlignLeft
            painter.drawText(text_rect, int(alignment), str(index.data(Qt.ItemDataRole.DisplayRole) or ""))
            painter.restore()
            return
        else:
            row_color = index.data(Qt.ItemDataRole.BackgroundRole)
            if row_color is not None:
                styled_option.backgroundBrush = row_color

        super().paint(painter, styled_option, index)

        if not self._enabled:
            return

        bar_value = index.data(DataTableModel.BAR_VALUE_ROLE)
        bar_range = index.data(DataTableModel.BAR_RANGE_ROLE)
        if bar_value is None or not isinstance(bar_range, tuple):
            return

        minimum, maximum = bar_range
        ratio = 1.0 if maximum <= minimum else max(0.0, min(1.0, (float(bar_value) - minimum) / (maximum - minimum)))
        if ratio <= 0:
            return

        rect = option.rect.adjusted(6, option.rect.height() - 10, -6, -4)
        bar_width = max(10, int(rect.width() * ratio))
        bar_rect = rect.adjusted(0, 0, bar_width - rect.width(), 0)
        painter.save()
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor("#4db7eb"))
        painter.drawRoundedRect(bar_rect, 3, 3)
        painter.restore()


class _LoadingWidget(QWidget):
    """Centered loading indicator shown while data is being prepared."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._label = QLabel("⏳ Veriler yükleniyor...")
        self._label.setStyleSheet(
            "font-size: 14pt; font-weight: 600; color: #8b7355; background: transparent;"
        )
        self._label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self._label)

        self._detail = QLabel("Polars DataFrame hazırlanıyor, lütfen bekleyiniz.")
        self._detail.setStyleSheet(
            "font-size: 10pt; color: #a39580; background: transparent;"
        )
        self._detail.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self._detail)


class _EmptyWidget(QWidget):
    """Centered placeholder shown when no dataset is loaded."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._label = QLabel("📂 Henüz veri yüklenmedi")
        self._label.setStyleSheet(
            "font-size: 14pt; font-weight: 600; color: #8b7355; background: transparent;"
        )
        self._label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self._label)

        self._detail = QLabel("File widget'tan bir dosya seçerek başlayabilirsiniz.")
        self._detail.setStyleSheet(
            "font-size: 10pt; color: #a39580; background: transparent;"
        )
        self._detail.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self._detail)


class DataTableScreen(QWidget):
    def __init__(self, parent: QWidget | None = None, preview_service: PreviewService | None = None) -> None:
        super().__init__(parent)
        self._preview_service = preview_service or PreviewService()
        self._dataset_handle: DatasetHandle | None = None
        self._headers: list[str] = []
        self._rows: list[list[str]] = []
        self._columns: list[DataTableColumn] = []
        self._numeric_ranges: dict[int, tuple[float, float]] = {}
        self._target_column_index: int | None = None
        self._total_rows = 0
        self._missing_count = 0

        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(10)

        # --- stacked widget for empty / loading / data states ---
        self._stack = QStackedWidget(self)
        self._empty_widget = _EmptyWidget(self._stack)
        self._loading_widget = _LoadingWidget(self._stack)
        self._data_widget = QWidget(self._stack)

        self._stack.addWidget(self._empty_widget)     # index 0
        self._stack.addWidget(self._loading_widget)    # index 1
        self._stack.addWidget(self._data_widget)       # index 2
        self._stack.setCurrentIndex(0)
        layout.addWidget(self._stack, 1)

        # --- data widget inner layout ---
        data_layout = QVBoxLayout(self._data_widget)
        data_layout.setContentsMargins(0, 0, 0, 0)
        data_layout.setSpacing(10)

        splitter = QSplitter(Qt.Orientation.Horizontal, self._data_widget)
        splitter.setChildrenCollapsible(False)
        data_layout.addWidget(splitter, 1)

        self._sidebar = QWidget(self)
        self._sidebar.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Expanding)
        self._sidebar.setMinimumWidth(240)
        self._sidebar.setMaximumWidth(280)
        sidebar_layout = QVBoxLayout(self._sidebar)
        sidebar_layout.setContentsMargins(0, 0, 0, 0)
        sidebar_layout.setSpacing(10)

        self._info_card = self._build_panel("Info")
        self._info_label = QLabel("No dataset loaded.")
        self._info_label.setWordWrap(True)
        self._info_card.layout().addWidget(self._info_label)
        sidebar_layout.addWidget(self._info_card)

        self._variables_card = self._build_panel("Variables")
        self._show_labels_checkbox = QCheckBox("Show variable labels (if present)")
        self._show_labels_checkbox.setChecked(True)
        self._show_labels_checkbox.toggled.connect(self._refresh_headers)
        self._variables_card.layout().addWidget(self._show_labels_checkbox)

        self._visualize_checkbox = QCheckBox("Visualize numeric values")
        self._visualize_checkbox.setChecked(True)
        self._visualize_checkbox.toggled.connect(self._toggle_numeric_bars)
        self._variables_card.layout().addWidget(self._visualize_checkbox)

        self._color_checkbox = QCheckBox("Color by instance classes")
        self._color_checkbox.setChecked(True)
        self._color_checkbox.toggled.connect(self._refresh_row_colors)
        self._variables_card.layout().addWidget(self._color_checkbox)
        sidebar_layout.addWidget(self._variables_card)

        self._selection_card = self._build_panel("Selection")
        self._clear_selection_button = QPushButton("Clear Selection")
        self._clear_selection_button.setProperty("secondary", True)
        self._selection_card.layout().addWidget(self._clear_selection_button)

        self._select_full_rows_checkbox = QCheckBox("Select full rows")
        self._select_full_rows_checkbox.setChecked(True)
        self._selection_card.layout().addWidget(self._select_full_rows_checkbox)
        sidebar_layout.addWidget(self._selection_card)

        sidebar_layout.addStretch(1)

        self._restore_order_button = QPushButton("Restore Original Order")
        self._restore_order_button.setProperty("secondary", True)
        self._restore_order_button.clicked.connect(self._restore_original_order)
        self._restore_order_button.setEnabled(False)
        sidebar_layout.addWidget(self._restore_order_button)

        self._send_auto_checkbox = QCheckBox("Send Automatically")
        self._send_auto_checkbox.setChecked(True)
        sidebar_layout.addWidget(self._send_auto_checkbox)

        splitter.addWidget(self._sidebar)

        self._table_container = QWidget(self)
        table_layout = QVBoxLayout(self._table_container)
        table_layout.setContentsMargins(0, 0, 0, 0)
        table_layout.setSpacing(8)

        self._summary_label = QLabel("Data: none")
        self._summary_label.setWordWrap(True)
        table_layout.addWidget(self._summary_label)

        self._selected_summary_label = QLabel("Data Subset: -")
        self._selected_summary_label.setWordWrap(True)
        table_layout.addWidget(self._selected_summary_label)

        self._table = QTableView(self)
        self._table.setAlternatingRowColors(False)
        self._table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self._table.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self._table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self._table.setSortingEnabled(True)
        self._table.verticalHeader().setDefaultSectionSize(30)
        self._table.verticalHeader().setMinimumSectionSize(28)
        self._table.doubleClicked.connect(lambda index: self._table.selectRow(index.row()))

        self._model = DataTableModel(self._table)
        self._table.setModel(self._model)
        self._delegate = NumericBarDelegate(self._table)
        self._table.setItemDelegate(self._delegate)
        self._clear_selection_button.clicked.connect(self._table.clearSelection)
        self._select_full_rows_checkbox.toggled.connect(self._update_selection_mode)
        self._table.selectionModel().selectionChanged.connect(lambda *_args: self._update_selection_summary())
        table_layout.addWidget(self._table, 1)

        splitter.addWidget(self._table_container)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([255, 820])

    def set_preview_service(self, service: PreviewService) -> None:
        """Allow injection of PreviewService after construction."""
        self._preview_service = service

    def _build_panel(self, title: str) -> QFrame:
        frame = QFrame(self)
        frame.setProperty("panel", True)
        layout = QVBoxLayout(frame)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)
        label = QLabel(title)
        label.setProperty("sectionTitle", True)
        label.setStyleSheet("font-size: 11pt;")
        layout.addWidget(label)
        return frame

    def set_dataset(self, dataset_handle: Any) -> None:
        """Accept a DatasetHandle (preferred) or a string path to load data."""
        if dataset_handle is None:
            self._dataset_handle = None
            self._set_empty_state()
            return

        # Show loading state
        self._stack.setCurrentIndex(1)
        QApplication.processEvents()

        if isinstance(dataset_handle, DatasetHandle):
            self._dataset_handle = dataset_handle
        elif isinstance(dataset_handle, str):
            # Fallback: try to create a DatasetHandle from path via FileImportService
            self._dataset_handle = self._try_load_from_path(dataset_handle)
        else:
            self._dataset_handle = None

        if self._dataset_handle is None:
            self._set_empty_state()
            return

        self._load_from_handle()

    def _try_load_from_path(self, path: str) -> DatasetHandle | None:
        """Best-effort fallback to load a path into a DatasetHandle."""
        try:
            from portakal_app.data.services.file_import_service import FileImportService
            return FileImportService().load(path)
        except Exception:
            return None

    def _load_from_handle(self) -> None:
        """Load all data from the DatasetHandle via PreviewService."""
        dataset = self._dataset_handle
        if dataset is None:
            self._set_empty_state()
            return

        # Get column specs from the domain
        col_specs = self._preview_service.get_column_specs(dataset)
        self._columns = [
            DataTableColumn(
                name=spec["name"],
                type_name=spec["type_name"],
                role_name=spec["role_name"],
                values_preview=spec["values_preview"],
            )
            for spec in col_specs
        ]

        # Determine headers
        self._headers = [col.name for col in self._columns]

        # Get numeric ranges
        self._numeric_ranges = self._preview_service.get_numeric_ranges(dataset)

        # Get total rows
        self._total_rows = self._preview_service.get_total_rows(dataset)

        # Get target column
        self._target_column_index = self._find_target_index()

        # Load all rows via PreviewService (paginated internally for large datasets)
        self._rows = self._load_all_rows(dataset)

        # Count missing values
        self._missing_count = self._count_missing(self._rows)

        # Set the model
        self._model.set_dataset(
            self._headers,
            self._columns,
            self._rows,
            self._numeric_ranges,
            self._target_column_index,
        )

        # Update UI
        self._refresh_headers()
        self._refresh_info()
        self._toggle_numeric_bars(self._visualize_checkbox.isChecked())
        self._refresh_row_colors()
        self._table.resizeColumnsToContents()
        self._table.horizontalHeader().setStretchLastSection(True)
        self._restore_order_button.setEnabled(True)
        self._update_selection_summary()

        # Switch to data view
        self._stack.setCurrentIndex(2)

    def _load_all_rows(self, dataset: DatasetHandle) -> list[list[str]]:
        """Load all rows page by page using PreviewService for responsiveness."""
        total = self._preview_service.get_total_rows(dataset)
        all_rows: list[list[str]] = []
        offset = 0
        page_size = DEFAULT_PAGE_SIZE

        while offset < total:
            page = self._preview_service.get_page(dataset, offset, page_size)
            for row_tuple in page.rows:
                all_rows.append(list(row_tuple))
            offset += page_size
            # Keep UI responsive during large loads
            if offset < total:
                QApplication.processEvents()

        return all_rows

    def _count_missing(self, rows: list[list[str]]) -> int:
        count = 0
        for row in rows:
            for cell in row:
                if not cell.strip():
                    count += 1
        return count

    def _find_target_index(self) -> int | None:
        for index, col in enumerate(self._columns):
            if col.role_name == "target":
                return index
        return None

    def help_text(self) -> str:
        return (
            "Inspect the loaded dataset in tabular form, toggle numeric bars, "
            "color rows by inferred target classes, and manage row selection."
        )

    def documentation_url(self) -> str:
        return "https://orangedatamining.com/widget-catalog/data/data-table/"

    def footer_status_text(self) -> str:
        total = self._total_rows
        selected = self._selected_row_count()
        if total == 0:
            return "0"
        if selected:
            return f"{selected} | {total}"
        return str(total)

    def data_preview_snapshot(self) -> dict[str, object]:
        return {
            "summary": self._summary_label.text(),
            "headers": self._headers,
            "rows": self._rows[:200],
        }

    def detailed_data_snapshot(self) -> dict[str, object]:
        selected_row_indexes, selected_column_indexes = self._selected_selection_axes()
        selected_headers = [self._headers[index] for index in selected_column_indexes]
        selected_rows = self._selected_rows_data(selected_row_indexes, selected_column_indexes)[:200]
        all_rows = self._model.all_display_rows()[:200]
        selected_columns = [self._columns[index] for index in selected_column_indexes]
        selected_numeric_features = [
            column for index, column in zip(selected_column_indexes, selected_columns)
            if column.type_name == "numeric" and index != self._target_column_index
        ]
        selected_missing = 0
        for row in selected_rows:
            selected_missing += sum(1 for cell in row if not str(cell).strip())
        selected_target = "-"
        if self._target_column_index in selected_column_indexes:
            target_position = selected_column_indexes.index(self._target_column_index)
            selected_target = self._headers[self._target_column_index]
            selected_target_values = {
                row[target_position]
                for row in selected_rows
                if target_position < len(row) and str(row[target_position]).strip()
            }
            selected_target = f"{selected_target} ({len(selected_target_values)} values)"

        dataset_name = _describe_dataset(self._dataset_handle)
        selected_summary = "\n".join(
            [
                f"Selected Data: {dataset_name}: {len(selected_row_indexes)} instances, {len(selected_column_indexes)} variables",
                f"Features: {len(selected_numeric_features)} numeric ({'no missing values' if selected_missing == 0 else 'contains missing values'})",
                f"Target: {selected_target}",
            ]
        )
        return {
            "selected_summary": selected_summary,
            "selected_headers": selected_headers,
            "selected_rows": selected_rows,
            "data_summary": self._summary_label.text(),
            "data_headers": self._headers,
            "data_rows": all_rows,
        }

    def _refresh_headers(self) -> None:
        self._model.set_show_labels(self._show_labels_checkbox.isChecked())

    def _refresh_info(self) -> None:
        feature_columns = [column for index, column in enumerate(self._columns) if index != self._target_column_index]
        numeric_features = [column for column in feature_columns if column.type_name == "numeric"]
        target_values = set()
        if self._target_column_index is not None:
            for row in self._rows:
                if self._target_column_index < len(row) and row[self._target_column_index].strip():
                    target_values.add(row[self._target_column_index])

        self._info_label.setText(
            "\n".join(
                [
                    f"{self._total_rows} instances ({'no missing data' if self._missing_count == 0 else f'{self._missing_count} missing values'})",
                    f"{len(numeric_features)} numeric features",
                    f"Target with {len(target_values)} values" if self._target_column_index is not None else "No target variable inferred",
                    "No meta attributes.",
                ]
            )
        )

        dataset_name = _describe_dataset(self._dataset_handle)
        target_label = self._columns[self._target_column_index].name if self._target_column_index is not None else "-"
        self._summary_label.setText(
            "\n".join(
                [
                    f"Data: {dataset_name}: {self._total_rows} instances, {len(self._headers)} variables",
                    f"Features: {len(numeric_features)} numeric ({'no missing values' if self._missing_count == 0 else 'contains missing values'})",
                    f"Target: {target_label}",
                    "Showing all rows in the table",
                ]
            )
        )

    def _toggle_numeric_bars(self, enabled: bool) -> None:
        self._delegate.set_enabled(enabled)
        self._table.viewport().update()

    def _update_selection_mode(self, full_rows: bool) -> None:
        behavior = QAbstractItemView.SelectionBehavior.SelectRows if full_rows else QAbstractItemView.SelectionBehavior.SelectItems
        self._table.setSelectionBehavior(behavior)

    def _refresh_row_colors(self) -> None:
        self._model.set_color_by_classes(self._color_checkbox.isChecked())
        self._table.viewport().update()

    def _restore_original_order(self) -> None:
        self._model.restore_original_order()
        self._update_selection_summary()

    def _selected_row_count(self) -> int:
        selection_model = self._table.selectionModel()
        return len({index.row() for index in selection_model.selectedRows()}) if selection_model else 0

    def _selected_selection_axes(self) -> tuple[list[int], list[int]]:
        selection_model = self._table.selectionModel()
        if selection_model is None:
            return [], list(range(len(self._headers)))

        selected_indexes = selection_model.selectedIndexes()
        if not selected_indexes:
            return [], list(range(len(self._headers)))

        row_indexes = sorted({index.row() for index in selected_indexes})
        column_indexes = sorted({index.column() for index in selected_indexes})
        if self._table.selectionBehavior() == QAbstractItemView.SelectionBehavior.SelectRows:
            column_indexes = list(range(len(self._headers)))
        return row_indexes, column_indexes

    def _selected_rows_data(self, row_indexes: list[int], column_indexes: list[int] | None = None) -> list[list[str]]:
        if column_indexes is None:
            column_indexes = list(range(len(self._headers)))
        rows = self._model.rows_for_indexes(row_indexes)
        return [
            [row[column_index] if column_index < len(row) else "" for column_index in column_indexes]
            for row in rows
        ]

    def _update_selection_summary(self) -> None:
        selected_row_indexes, selected_column_indexes = self._selected_selection_axes()
        selected_rows = len(selected_row_indexes)
        if selected_rows == 0:
            self._selected_summary_label.setText("Data Subset: -")
        else:
            self._selected_summary_label.setText(f"Selected Data: {selected_rows} instances, {len(selected_column_indexes)} variables")

    def _set_empty_state(self) -> None:
        self._headers = []
        self._rows = []
        self._columns = []
        self._numeric_ranges = {}
        self._target_column_index = None
        self._total_rows = 0
        self._missing_count = 0
        self._summary_label.setText("Data: none")
        self._selected_summary_label.setText("Data Subset: -")
        self._info_label.setText("No dataset loaded.")
        self._model.clear()
        self._restore_order_button.setEnabled(False)
        self._stack.setCurrentIndex(0)

    def _is_float(self, value: str) -> bool:
        try:
            float(value)
        except ValueError:
            return False
        return True
