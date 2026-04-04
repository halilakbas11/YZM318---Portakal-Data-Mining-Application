from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QPainter, QPen
from PySide6.QtWidgets import (
    QComboBox,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from portakal_app.data.models import ColumnStatisticsResult, DatasetHandle, HistogramBin, ValueFrequency
from portakal_app.data.services.column_statistics_service import ColumnStatisticsService
from portakal_app.data.services.file_import_service import FileImportService
from portakal_app.ui.screens.node_screen import WorkflowNodeScreenSupport


class _HistogramWidget(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._bins: tuple[HistogramBin, ...] = ()
        self.setMinimumHeight(120)

    def set_bins(self, bins: tuple[HistogramBin, ...]) -> None:
        self._bins = bins
        self.update()

    def paintEvent(self, _event) -> None:
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#fffdf9"))
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        if not self._bins:
            painter.setPen(QColor("#8d877d"))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "No histogram available")
            return
        max_count = max(bin_item.count for bin_item in self._bins) or 1
        usable_width = max(1, self.width() - 16)
        usable_height = max(1, self.height() - 26)
        bar_width = max(6, usable_width // max(1, len(self._bins)))
        for index, bin_item in enumerate(self._bins):
            height = int((bin_item.count / max_count) * usable_height)
            x = 8 + (index * bar_width)
            y = self.height() - 18 - height
            painter.setBrush(QColor("#ddb26b"))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawRoundedRect(x, y, max(4, bar_width - 4), height, 3, 3)
        painter.setPen(QPen(QColor("#9b9488"), 1))
        painter.drawLine(8, self.height() - 18, self.width() - 8, self.height() - 18)


class _FrequencyBarsWidget(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._values: tuple[ValueFrequency, ...] = ()
        self.setMinimumHeight(140)

    def set_values(self, values: tuple[ValueFrequency, ...]) -> None:
        self._values = values
        self.update()

    def paintEvent(self, _event) -> None:
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#fffdf9"))
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        if not self._values:
            painter.setPen(QColor("#8d877d"))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "No frequency bars available")
            return
        row_height = max(18, (self.height() - 16) // max(1, len(self._values)))
        label_width = max(80, self.width() // 3)
        for index, value in enumerate(self._values):
            y = 8 + (index * row_height)
            painter.setPen(QColor("#534b40"))
            painter.drawText(8, y + 12, value.value[:18])
            bar_x = label_width
            bar_width = int((self.width() - label_width - 12) * value.ratio)
            painter.setBrush(QColor("#91b4d8"))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawRoundedRect(bar_x, y, max(6, bar_width), 12, 4, 4)
            painter.setPen(QColor("#534b40"))
            painter.drawText(self.width() - 54, y + 12, f"{value.count}")


class ColumnStatisticsScreen(QWidget, WorkflowNodeScreenSupport):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._init_workflow_node_support()
        self._import_service = FileImportService()
        self._statistics_service = ColumnStatisticsService()
        self._dataset_handle: DatasetHandle | None = None
        self._all_column_names: list[str] = []

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(10)

        self._dataset_label = QLabel("Dataset: none")
        self._dataset_label.setProperty("sectionTitle", True)
        self._dataset_label.setStyleSheet("font-size: 12pt; background: transparent;")
        layout.addWidget(self._dataset_label)

        search_row = QHBoxLayout()
        search_row.addWidget(QLabel("Search"))
        self._search_input = QLineEdit(self)
        self._search_input.setPlaceholderText("Filter columns...")
        self._search_input.textChanged.connect(self._refilter_columns)
        search_row.addWidget(self._search_input, 1)
        search_row.addWidget(QLabel("Column"))
        self._column_combo = QComboBox(self)
        self._column_combo.currentTextChanged.connect(self._refresh_stats)
        search_row.addWidget(self._column_combo, 1)
        layout.addLayout(search_row)

        self._warning_badges_frame = QFrame(self)
        self._warning_badges_layout = QHBoxLayout(self._warning_badges_frame)
        self._warning_badges_layout.setContentsMargins(0, 0, 0, 0)
        self._warning_badges_layout.setSpacing(6)
        layout.addWidget(self._warning_badges_frame)

        stats_group = QGroupBox("Statistics")
        stats_layout = QVBoxLayout(stats_group)
        stats_layout.setContentsMargins(10, 10, 10, 10)
        self._stats_table = QTableWidget(0, 2, self)
        self._stats_table.setHorizontalHeaderLabels(["Metric", "Value"])
        self._stats_table.horizontalHeader().setStretchLastSection(True)
        self._stats_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        stats_layout.addWidget(self._stats_table)
        layout.addWidget(stats_group)

        viz_row = QHBoxLayout()
        viz_row.setSpacing(10)
        histogram_group = QGroupBox("Distribution")
        histogram_layout = QVBoxLayout(histogram_group)
        histogram_layout.setContentsMargins(10, 10, 10, 10)
        self._histogram_widget = _HistogramWidget(self)
        histogram_layout.addWidget(self._histogram_widget)
        viz_row.addWidget(histogram_group, 1)

        top_values_group = QGroupBox("Top Values")
        top_values_layout = QVBoxLayout(top_values_group)
        top_values_layout.setContentsMargins(10, 10, 10, 10)
        self._top_values_table = QTableWidget(0, 3, self)
        self._top_values_table.setHorizontalHeaderLabels(["Value", "Count", "Ratio"])
        self._top_values_table.horizontalHeader().setStretchLastSection(True)
        self._top_values_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        top_values_layout.addWidget(self._top_values_table)
        self._frequency_bars_widget = _FrequencyBarsWidget(self)
        top_values_layout.addWidget(self._frequency_bars_widget)
        viz_row.addWidget(top_values_group, 1)
        layout.addLayout(viz_row, 1)

        self._status_label = QLabel("Load a dataset to inspect per-column statistics.")
        self._status_label.setWordWrap(True)
        layout.addWidget(self._status_label)

    def set_dataset(self, dataset_handle: DatasetHandle | str | None) -> None:
        if isinstance(dataset_handle, str):
            try:
                dataset_handle = self._import_service.load(dataset_handle)
            except Exception:
                dataset_handle = None
        self._dataset_handle = dataset_handle
        self._column_combo.clear()
        self._all_column_names = []
        if dataset_handle is None:
            self._dataset_label.setText("Dataset: none")
            self._status_label.setText("Load a dataset to inspect per-column statistics.")
            self._stats_table.setRowCount(0)
            self._top_values_table.setRowCount(0)
            self._histogram_widget.set_bins(())
            self._frequency_bars_widget.set_values(())
            self._render_warning_badges(())
            return

        self._dataset_label.setText(f"Dataset: {dataset_handle.source.path.name}")
        self._all_column_names = [column.name for column in dataset_handle.domain.columns]
        self._search_input.clear()
        self._refilter_columns()

    def footer_status_text(self) -> str:
        return self._column_combo.currentText() or "Stats"

    def set_input_payload(self, payload) -> None:
        dataset = payload.dataset if payload is not None else None
        self.set_dataset(dataset)

    def help_text(self) -> str:
        return (
            "Inspect one column at a time with search, warning badges, numeric histogram, "
            "and categorical frequency bars."
        )

    def documentation_url(self) -> str:
        return "https://orangedatamining.com/widget-catalog/data/column-statistics/"

    def _refilter_columns(self) -> None:
        current_text = self._column_combo.currentText()
        search = self._search_input.text().strip().lower()
        filtered = [name for name in self._all_column_names if search in name.lower()]
        self._column_combo.blockSignals(True)
        self._column_combo.clear()
        self._column_combo.addItems(filtered)
        if current_text in filtered:
            self._column_combo.setCurrentText(current_text)
        self._column_combo.blockSignals(False)
        if filtered:
            self._refresh_stats()
        else:
            self._status_label.setText("No columns matched the current search.")
            self._stats_table.setRowCount(0)
            self._top_values_table.setRowCount(0)
            self._histogram_widget.set_bins(())
            self._frequency_bars_widget.set_values(())
            self._render_warning_badges(())

    def _refresh_stats(self) -> None:
        if self._dataset_handle is None or not self._column_combo.currentText():
            return
        result = self._statistics_service.describe(self._dataset_handle, self._column_combo.currentText())
        self._populate_result(result)

    def _populate_result(self, result: ColumnStatisticsResult) -> None:
        self._stats_table.setRowCount(len(result.metrics))
        for row_index, (name, value) in enumerate(result.metrics):
            self._stats_table.setItem(row_index, 0, QTableWidgetItem(name))
            self._stats_table.setItem(row_index, 1, QTableWidgetItem(value))
        self._stats_table.resizeColumnsToContents()

        self._top_values_table.setRowCount(len(result.top_values))
        for row_index, value in enumerate(result.top_values):
            self._top_values_table.setItem(row_index, 0, QTableWidgetItem(value.value))
            self._top_values_table.setItem(row_index, 1, QTableWidgetItem(str(value.count)))
            self._top_values_table.setItem(row_index, 2, QTableWidgetItem(f"{value.ratio * 100:.1f}%"))
        self._top_values_table.resizeColumnsToContents()

        self._histogram_widget.set_bins(result.histogram_bins)
        self._frequency_bars_widget.set_values(result.top_values)
        self._render_warning_badges(result.warning_tags)
        self._status_label.setText(f"Showing statistics for '{result.column_name}'.")

    def _render_warning_badges(self, warning_tags: tuple[str, ...]) -> None:
        while self._warning_badges_layout.count():
            item = self._warning_badges_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        if not warning_tags:
            badge = QLabel("No warnings")
            badge.setProperty("muted", True)
            self._warning_badges_layout.addWidget(badge)
            self._warning_badges_layout.addStretch(1)
            return
        for tag in warning_tags:
            badge = QLabel(tag)
            badge.setStyleSheet(
                "background: #f5e4c2; color: #5b4020; border-radius: 10px; padding: 4px 10px; font-weight: 600;"
            )
            self._warning_badges_layout.addWidget(badge)
        self._warning_badges_layout.addStretch(1)
