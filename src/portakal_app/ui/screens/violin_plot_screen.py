from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats
from PySide6.QtCore import QPoint, QRect, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QSizePolicy,
    QToolTip,
    QVBoxLayout,
    QWidget,
)

from portakal_app.data.models import DatasetHandle
from portakal_app.data.services.generated_dataset_service import GeneratedDatasetService
from portakal_app.ui import i18n
from portakal_app.ui.screens.node_screen import WorkflowNodeScreenSupport
from portakal_app.ui.screens.visualize_common import (
    PALETTE,
    PlotColumn,
    build_selection_outputs,
    discrete_columns,
    kernel_density,
    nice_ticks,
    numeric_columns,
    prepared_column,
)


@dataclass(frozen=True)
class _ViolinEntry:
    label: str
    support: np.ndarray
    density: np.ndarray
    values: np.ndarray
    row_indices: np.ndarray
    color: QColor


class _ViolinCanvas(QWidget):
    selectionChanged = Signal(list)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._entries: list[_ViolinEntry] = []
        self._orientation = Qt.Orientation.Vertical
        self._show_rug = False
        self._show_box_plot = True
        self._show_strip_plot = False
        self._show_grid = False
        self._scale_mode = "Area"
        self._selection_entry = -1
        self._selection_range: tuple[float, float] | None = None
        self._selected_rows: set[int] = set()
        self._drag_start: QPoint | None = None
        self._hover_regions: list[tuple[QRect, str]] = []
        self.setMinimumHeight(280)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMouseTracking(True)

    def set_state(
        self,
        entries: list[_ViolinEntry],
        *,
        orientation: Qt.Orientation,
        show_rug: bool,
        show_box_plot: bool,
        show_strip_plot: bool,
        show_grid: bool,
        scale_mode: str,
    ) -> None:
        self._entries = entries
        self._orientation = orientation
        self._show_rug = show_rug
        self._show_box_plot = show_box_plot
        self._show_strip_plot = show_strip_plot
        self._show_grid = show_grid
        self._scale_mode = scale_mode
        self._hover_regions = []
        self.update()

    def set_selected_rows(self, rows: list[int]) -> None:
        self._selected_rows = set(rows)
        self.update()

    def clear_selection(self) -> None:
        self._selection_entry = -1
        self._selection_range = None
        if self._selected_rows:
            self._selected_rows.clear()
            self.selectionChanged.emit([])
        self.update()

    def selection_state(self) -> tuple[int, tuple[float, float] | None]:
        return self._selection_entry, self._selection_range

    def entry_label(self, index: int) -> str | None:
        if 0 <= index < len(self._entries):
            return self._entries[index].label
        return None

    def restore_selection_state(
        self,
        entry_index: int,
        value_range: tuple[float, float] | None,
    ) -> None:
        if entry_index < 0 or value_range is None or entry_index >= len(self._entries):
            self._selection_entry = -1
            self._selection_range = None
            return
        low, high = value_range
        entry = self._entries[entry_index]
        mask = (entry.values >= low) & (entry.values <= high)
        rows = [int(row) for row in entry.row_indices[mask]]
        self._selection_entry = entry_index
        self._selection_range = (float(low), float(high))
        self._selected_rows = set(rows)
        self.selectionChanged.emit(rows)
        self.update()

    def mousePressEvent(self, event) -> None:
        if event.button() != Qt.MouseButton.LeftButton:
            return super().mousePressEvent(event)
        self._drag_start = event.position().toPoint()
        self.update()

    def mouseMoveEvent(self, event) -> None:
        if self._drag_start is None:
            for rect, tooltip in self._hover_regions:
                if rect.contains(event.position().toPoint()):
                    QToolTip.showText(event.globalPosition().toPoint(), tooltip, self)
                    return
            QToolTip.hideText()
        self.update()

    def mouseReleaseEvent(self, event) -> None:
        if event.button() != Qt.MouseButton.LeftButton or self._drag_start is None:
            return super().mouseReleaseEvent(event)
        self._apply_drag_selection(self._drag_start, event.position().toPoint())
        self._drag_start = None
        self.update()

    def leaveEvent(self, _event) -> None:
        QToolTip.hideText()

    def paintEvent(self, _event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), QColor("#fffdf9"))
        self._hover_regions = []

        if not self._entries:
            painter.setPen(QColor("#8d877d"))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, i18n.t("Load numeric data to plot violins."))
            return

        if self._orientation == Qt.Orientation.Vertical:
            margin_left = 78
            margin_right = 24
            margin_top = 24
            margin_bottom = 72
        else:
            margin_left = 128
            margin_right = 24
            margin_top = 24
            margin_bottom = 56
        chart = QRect(
            margin_left,
            margin_top,
            max(10, self.width() - margin_left - margin_right),
            max(10, self.height() - margin_top - margin_bottom),
        )

        support_low = min(float(np.min(entry.support)) for entry in self._entries)
        support_high = max(float(np.max(entry.support)) for entry in self._entries)
        if abs(support_high - support_low) < 1e-12:
            support_high = support_low + 1.0
        max_density = self._global_density_scale() or 1.0
        ticks = nice_ticks(support_low, support_high, count=5)

        if self._show_grid:
            painter.setPen(QPen(QColor("#e7dfd3"), 1, Qt.PenStyle.DotLine))
            for tick in ticks:
                if self._orientation == Qt.Orientation.Vertical:
                    py = chart.bottom() - int((tick - support_low) / (support_high - support_low) * chart.height())
                    painter.drawLine(chart.left(), py, chart.right(), py)
                else:
                    px = chart.left() + int((tick - support_low) / (support_high - support_low) * chart.width())
                    painter.drawLine(px, chart.top(), px, chart.bottom())

        painter.setPen(QPen(QColor("#675f53"), 1.2))
        painter.drawLine(chart.left(), chart.bottom(), chart.right(), chart.bottom())
        painter.drawLine(chart.left(), chart.top(), chart.left(), chart.bottom())
        painter.setPen(QColor("#534b40"))
        for tick in ticks:
            label = f"{tick:.3g}"
            if self._orientation == Qt.Orientation.Vertical:
                py = chart.bottom() - int((tick - support_low) / (support_high - support_low) * chart.height())
                painter.drawLine(chart.left() - 5, py, chart.left(), py)
                painter.drawText(
                    QRect(4, py - 10, margin_left - 12, 20),
                    Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
                    label,
                )
            else:
                px = chart.left() + int((tick - support_low) / (support_high - support_low) * chart.width())
                painter.drawLine(px, chart.bottom(), px, chart.bottom() + 5)
                painter.drawText(
                    QRect(px - 26, chart.bottom() + 8, 52, 18),
                    Qt.AlignmentFlag.AlignCenter,
                    label,
                )
        for index, entry in enumerate(self._entries):
            if self._orientation == Qt.Orientation.Vertical:
                x_center = chart.left() + int((index + 0.5) / len(self._entries) * chart.width())
                self._paint_vertical_entry(
                    painter, chart, entry, x_center, support_low, support_high, max_density
                )
                painter.drawText(
                    QRect(x_center - 54, chart.bottom() + 10, 108, 22),
                    Qt.AlignmentFlag.AlignCenter,
                    entry.label,
                )
            else:
                y_center = chart.top() + int((index + 0.5) / len(self._entries) * chart.height())
                self._paint_horizontal_entry(
                    painter, chart, entry, y_center, support_low, support_high, max_density
                )
                painter.drawText(
                    QRect(6, y_center - 10, margin_left - 14, 20),
                    Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
                    entry.label,
                )

        if self._selection_entry >= 0 and self._selection_range is not None:
            self._draw_selection_overlay(painter, chart, support_low, support_high)

    def _paint_vertical_entry(
        self,
        painter: QPainter,
        chart: QRect,
        entry: _ViolinEntry,
        x_center: int,
        support_low: float,
        support_high: float,
        max_density: float,
    ) -> None:
        half_width = max(12.0, chart.width() / max(1, len(self._entries)) * 0.38)
        scaled = self._scaled_density(entry, max_density) * half_width
        path = QPainterPath()
        top_y = chart.bottom() - int((entry.support[0] - support_low) / (support_high - support_low) * chart.height())
        path.moveTo(x_center, top_y)
        for value, density in zip(entry.support, scaled):
            y = chart.bottom() - int((value - support_low) / (support_high - support_low) * chart.height())
            path.lineTo(x_center - density, y)
        for value, density in zip(entry.support[::-1], scaled[::-1]):
            y = chart.bottom() - int((value - support_low) / (support_high - support_low) * chart.height())
            path.lineTo(x_center + density, y)
        path.closeSubpath()
        fill = QColor(entry.color)
        fill.setAlpha(180)
        painter.setPen(QPen(entry.color.darker(150), 1.5))
        painter.setBrush(fill)
        painter.drawPath(path)

        q0, q1, median, q3, q100 = np.percentile(entry.values, [0, 25, 50, 75, 100])
        y_q0 = chart.bottom() - int((q0 - support_low) / (support_high - support_low) * chart.height())
        y_q1 = chart.bottom() - int((q1 - support_low) / (support_high - support_low) * chart.height())
        y_median = chart.bottom() - int((median - support_low) / (support_high - support_low) * chart.height())
        y_q3 = chart.bottom() - int((q3 - support_low) / (support_high - support_low) * chart.height())
        y_q100 = chart.bottom() - int((q100 - support_low) / (support_high - support_low) * chart.height())
        if self._show_box_plot:
            painter.setPen(QPen(QColor("#111111"), 1.2))
            painter.drawLine(x_center, y_q0, x_center, y_q1)
            painter.drawLine(x_center, y_q3, x_center, y_q100)
            painter.setPen(QPen(QColor("#111111"), 6, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
            painter.drawLine(x_center, y_q1, x_center, y_q3)
            painter.setPen(QPen(QColor("#111111"), 1.0))
            painter.setBrush(QColor("#ffffff"))
            painter.drawEllipse(QRectF(x_center - 4, y_median - 4, 8, 8))

        if self._show_rug:
            painter.setPen(QPen(QColor("#111111"), 1))
            for value in entry.values:
                y = chart.bottom() - int((float(value) - support_low) / (support_high - support_low) * chart.height())
                density_at_value = float(np.interp(float(value), entry.support, scaled))
                painter.drawLine(
                    int(x_center - density_at_value),
                    y,
                    int(x_center + density_at_value),
                    y,
                )
        if self._show_strip_plot:
            for row_index, value in zip(entry.row_indices, entry.values):
                y = chart.bottom() - int((float(value) - support_low) / (support_high - support_low) * chart.height())
                seed = int(row_index) * 2654435761
                density_at_value = float(np.interp(float(value), entry.support, scaled))
                offset = ((seed % 997) / 997.0 - 0.5) * density_at_value * 2.0
                radius = 3 if int(row_index) in self._selected_rows else 2
                color = QColor(entry.color)
                color.setAlpha(180 if int(row_index) in self._selected_rows else 110)
                painter.setBrush(color)
                painter.setPen(QPen(QColor("#111111") if int(row_index) in self._selected_rows else color.darker(140), 1.0))
                painter.drawEllipse(QRectF(x_center + offset - radius, y - radius, radius * 2, radius * 2))

        tooltip = "\n".join(
            [
                entry.label,
                f"Rows: {len(entry.row_indices)}",
                f"Median: {median:.3g}",
            ]
        )
        self._hover_regions.append((QRect(int(x_center - half_width), chart.top(), int(half_width * 2), chart.height()), tooltip))

    def _paint_horizontal_entry(
        self,
        painter: QPainter,
        chart: QRect,
        entry: _ViolinEntry,
        y_center: int,
        support_low: float,
        support_high: float,
        max_density: float,
    ) -> None:
        half_height = max(12.0, chart.height() / max(1, len(self._entries)) * 0.38)
        scaled = self._scaled_density(entry, max_density) * half_height
        path = QPainterPath()
        left_x = chart.left() + int((entry.support[0] - support_low) / (support_high - support_low) * chart.width())
        path.moveTo(left_x, y_center)
        for value, density in zip(entry.support, scaled):
            x = chart.left() + int((value - support_low) / (support_high - support_low) * chart.width())
            path.lineTo(x, y_center - density)
        for value, density in zip(entry.support[::-1], scaled[::-1]):
            x = chart.left() + int((value - support_low) / (support_high - support_low) * chart.width())
            path.lineTo(x, y_center + density)
        path.closeSubpath()
        fill = QColor(entry.color)
        fill.setAlpha(180)
        painter.setPen(QPen(entry.color.darker(150), 1.5))
        painter.setBrush(fill)
        painter.drawPath(path)

        q0, q1, median, q3, q100 = np.percentile(entry.values, [0, 25, 50, 75, 100])
        x_q0 = chart.left() + int((q0 - support_low) / (support_high - support_low) * chart.width())
        x_q1 = chart.left() + int((q1 - support_low) / (support_high - support_low) * chart.width())
        x_median = chart.left() + int((median - support_low) / (support_high - support_low) * chart.width())
        x_q3 = chart.left() + int((q3 - support_low) / (support_high - support_low) * chart.width())
        x_q100 = chart.left() + int((q100 - support_low) / (support_high - support_low) * chart.width())
        if self._show_box_plot:
            painter.setPen(QPen(QColor("#111111"), 1.2))
            painter.drawLine(x_q0, y_center, x_q1, y_center)
            painter.drawLine(x_q3, y_center, x_q100, y_center)
            painter.setPen(QPen(QColor("#111111"), 6, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
            painter.drawLine(x_q1, y_center, x_q3, y_center)
            painter.setPen(QPen(QColor("#111111"), 1.0))
            painter.setBrush(QColor("#ffffff"))
            painter.drawEllipse(QRectF(x_median - 4, y_center - 4, 8, 8))

        if self._show_rug:
            painter.setPen(QPen(QColor("#111111"), 1))
            for value in entry.values:
                x = chart.left() + int((float(value) - support_low) / (support_high - support_low) * chart.width())
                density_at_value = float(np.interp(float(value), entry.support, scaled))
                painter.drawLine(
                    x,
                    int(y_center - density_at_value),
                    x,
                    int(y_center + density_at_value),
                )
        if self._show_strip_plot:
            for row_index, value in zip(entry.row_indices, entry.values):
                x = chart.left() + int((float(value) - support_low) / (support_high - support_low) * chart.width())
                seed = int(row_index) * 2654435761
                density_at_value = float(np.interp(float(value), entry.support, scaled))
                offset = ((seed % 997) / 997.0 - 0.5) * density_at_value * 2.0
                radius = 3 if int(row_index) in self._selected_rows else 2
                color = QColor(entry.color)
                color.setAlpha(180 if int(row_index) in self._selected_rows else 110)
                painter.setBrush(color)
                painter.setPen(QPen(QColor("#111111") if int(row_index) in self._selected_rows else color.darker(140), 1.0))
                painter.drawEllipse(QRectF(x - radius, y_center + offset - radius, radius * 2, radius * 2))

        tooltip = "\n".join(
            [
                entry.label,
                f"Rows: {len(entry.row_indices)}",
                f"Median: {median:.3g}",
            ]
        )
        self._hover_regions.append((QRect(chart.left(), int(y_center - half_height), chart.width(), int(half_height * 2)), tooltip))

    def _apply_drag_selection(self, start: QPoint, end: QPoint) -> None:
        if not self._entries:
            return
        if self._orientation == Qt.Orientation.Vertical:
            margin_left = 78
            margin_right = 24
            margin_top = 24
            margin_bottom = 72
        else:
            margin_left = 128
            margin_right = 24
            margin_top = 24
            margin_bottom = 56
        chart = QRect(
            margin_left,
            margin_top,
            max(10, self.width() - margin_left - margin_right),
            max(10, self.height() - margin_top - margin_bottom),
        )
        support_low = min(float(np.min(entry.support)) for entry in self._entries)
        support_high = max(float(np.max(entry.support)) for entry in self._entries)
        if self._orientation == Qt.Orientation.Vertical:
            if not chart.contains(start) and not chart.contains(end):
                return
            relative = max(chart.left(), min(chart.right(), end.x())) - chart.left()
            entry_index = min(len(self._entries) - 1, max(0, int(relative / max(chart.width(), 1) * len(self._entries))))
            low_y = max(chart.top(), min(start.y(), end.y()))
            high_y = min(chart.bottom(), max(start.y(), end.y()))
            min_value = support_high - (high_y - chart.top()) / max(chart.height(), 1) * (support_high - support_low)
            max_value = support_high - (low_y - chart.top()) / max(chart.height(), 1) * (support_high - support_low)
        else:
            if not chart.contains(start) and not chart.contains(end):
                return
            relative = max(chart.top(), min(chart.bottom(), end.y())) - chart.top()
            entry_index = min(len(self._entries) - 1, max(0, int(relative / max(chart.height(), 1) * len(self._entries))))
            low_x = max(chart.left(), min(start.x(), end.x()))
            high_x = min(chart.right(), max(start.x(), end.x()))
            min_value = support_low + (low_x - chart.left()) / max(chart.width(), 1) * (support_high - support_low)
            max_value = support_low + (high_x - chart.left()) / max(chart.width(), 1) * (support_high - support_low)

        entry = self._entries[entry_index]
        mask = (entry.values >= min_value) & (entry.values <= max_value)
        rows = [int(row) for row in entry.row_indices[mask]]
        self._selection_entry = entry_index
        self._selection_range = (float(min_value), float(max_value))
        self._selected_rows = set(rows)
        self.selectionChanged.emit(rows)

    def _draw_selection_overlay(self, painter: QPainter, chart: QRect, support_low: float, support_high: float) -> None:
        if self._selection_entry < 0 or self._selection_range is None:
            return
        low, high = self._selection_range
        painter.setPen(QPen(QColor(255, 255, 100), 1))
        painter.setBrush(QColor(255, 255, 0, 100))
        if self._orientation == Qt.Orientation.Vertical:
            x0 = chart.left() + int(self._selection_entry / len(self._entries) * chart.width())
            x1 = chart.left() + int((self._selection_entry + 1) / len(self._entries) * chart.width())
            y_low = chart.bottom() - int((high - support_low) / (support_high - support_low) * chart.height())
            y_high = chart.bottom() - int((low - support_low) / (support_high - support_low) * chart.height())
            painter.drawRect(QRect(x0 + 4, y_low, max(8, x1 - x0 - 8), max(8, y_high - y_low)))
        else:
            y0 = chart.top() + int(self._selection_entry / len(self._entries) * chart.height())
            y1 = chart.top() + int((self._selection_entry + 1) / len(self._entries) * chart.height())
            x_low = chart.left() + int((low - support_low) / (support_high - support_low) * chart.width())
            x_high = chart.left() + int((high - support_low) / (support_high - support_low) * chart.width())
            painter.drawRect(QRect(x_low, y0 + 4, max(8, x_high - x_low), max(8, y1 - y0 - 8)))

    def _global_density_scale(self) -> float:
        scales = [self._scaled_density_raw(entry) for entry in self._entries]
        return max((float(np.max(scale)) for scale in scales if scale.size), default=1.0)

    def _scaled_density_raw(self, entry: _ViolinEntry) -> np.ndarray:
        density = entry.density.astype(float, copy=False)
        if density.size == 0:
            return density
        local_max = max(float(np.max(density)), 1e-9)
        if self._scale_mode == i18n.t("Count"):
            return density * len(entry.values) / local_max
        if self._scale_mode == i18n.t("Width"):
            return density / local_max
        return density

    def _scaled_density(self, entry: _ViolinEntry, global_max_density: float) -> np.ndarray:
        return self._scaled_density_raw(entry) / max(global_max_density, 1e-9)


class ViolinPlotScreen(QWidget, WorkflowNodeScreenSupport):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._init_workflow_node_support()
        self._builder = GeneratedDatasetService()
        self._dataset: DatasetHandle | None = None
        self._selected_dataset: DatasetHandle | None = None
        self._annotated_dataset: DatasetHandle | None = None
        self._pending_selected_rows: list[int] = []
        self._pending_selection_entry = -1
        self._pending_selection_range: tuple[float, float] | None = None
        self._pending_selection_label: str | None = None
        self._pending_value_name = ""
        self._pending_group_name = ""

        root = QHBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(12)

        self._sidebar_scroll = QScrollArea(self)
        self._sidebar_scroll.setWidgetResizable(True)
        self._sidebar_scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        self._sidebar_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._sidebar_scroll.setMinimumWidth(320)
        self._sidebar_scroll.setMaximumWidth(360)
        root.addWidget(self._sidebar_scroll, 0)

        sidebar_host = QWidget(self._sidebar_scroll)
        self._sidebar_scroll.setWidget(sidebar_host)
        sidebar = QVBoxLayout(sidebar_host)
        sidebar.setContentsMargins(4, 4, 4, 4)
        sidebar.setSpacing(12)

        variable_box = QGroupBox(i18n.t("Variable"))
        variable_layout = QVBoxLayout(variable_box)
        self._value_list = QListWidget()
        self._value_list.setMinimumHeight(124)
        self._order_value_cb = QCheckBox(i18n.t("Order by relevance to subgroups"))
        variable_layout.addWidget(self._value_list)
        variable_layout.addWidget(self._order_value_cb)
        sidebar.addWidget(variable_box)

        subgroup_box = QGroupBox(i18n.t("Subgroups"))
        subgroup_layout = QVBoxLayout(subgroup_box)
        self._group_list = QListWidget()
        self._group_list.setMinimumHeight(124)
        self._order_group_cb = QCheckBox(i18n.t("Order by relevance to variable"))
        subgroup_layout.addWidget(self._group_list)
        subgroup_layout.addWidget(self._order_group_cb)
        sidebar.addWidget(subgroup_box)

        display_box = QGroupBox(i18n.t("Display"))
        display_layout = QVBoxLayout(display_box)
        self._show_box_cb = QCheckBox(i18n.t("Box plot"))
        self._show_box_cb.setChecked(True)
        self._show_strip_cb = QCheckBox(i18n.t("Density dots"))
        self._show_rug_cb = QCheckBox(i18n.t("Density lines"))
        self._show_grid_cb = QCheckBox(i18n.t("Show grid"))
        self._order_cb = QCheckBox(i18n.t("Order subgroups"))
        self._orientation_group = QButtonGroup(self)
        orientation_row = QHBoxLayout()
        orientation_row.addWidget(QLabel(i18n.t("Orientation:")))
        self._horizontal_rb = QRadioButton(i18n.t("Horizontal"))
        self._vertical_rb = QRadioButton(i18n.t("Vertical"))
        self._vertical_rb.setChecked(True)
        self._orientation_group.addButton(self._horizontal_rb, 0)
        self._orientation_group.addButton(self._vertical_rb, 1)
        orientation_row.addWidget(self._horizontal_rb)
        orientation_row.addWidget(self._vertical_rb)
        orientation_row.addStretch(1)
        display_layout.addWidget(self._show_box_cb)
        display_layout.addWidget(self._show_strip_cb)
        display_layout.addWidget(self._show_rug_cb)
        display_layout.addWidget(self._order_cb)
        display_layout.addWidget(self._show_grid_cb)
        display_layout.addLayout(orientation_row)
        sidebar.addWidget(display_box)

        density_box = QGroupBox(i18n.t("Density Estimation"))
        density_layout = QVBoxLayout(density_box)
        kernel_row = QHBoxLayout()
        kernel_row.addWidget(QLabel(i18n.t("Kernel:")))
        self._kernel_combo = QComboBox()
        self._kernel_combo.addItems([i18n.t("Normal"), i18n.t("Epanechnikov"), i18n.t("Linear")])
        kernel_row.addWidget(self._kernel_combo, 1)
        density_layout.addLayout(kernel_row)
        scale_row = QHBoxLayout()
        scale_row.addWidget(QLabel(i18n.t("Scale:")))
        self._scale_combo = QComboBox()
        self._scale_combo.addItems([i18n.t("Area"), i18n.t("Count"), i18n.t("Width")])
        scale_row.addWidget(self._scale_combo, 1)
        density_layout.addLayout(scale_row)
        sidebar.addWidget(density_box)

        self._dataset_label = QLabel(i18n.t("Dataset: none"))
        self._status_label = QLabel(i18n.t("Load numeric data to plot violins."))
        self._status_label.setWordWrap(True)
        sidebar.addWidget(self._dataset_label)
        sidebar.addWidget(self._status_label)

        self._clear_button = QPushButton(i18n.t("Clear Selection"))
        self._clear_button.clicked.connect(self._clear_selection)
        sidebar.addWidget(self._clear_button)

        self._selection_label = QLabel(i18n.t("Selected: 0"))
        sidebar.addWidget(self._selection_label)
        sidebar.addStretch(1)

        self._canvas = _ViolinCanvas(self)
        self._canvas.selectionChanged.connect(self._handle_selection_changed)
        self._canvas.setMinimumWidth(560)
        root.addWidget(self._canvas, 1)

        self._value_list.currentItemChanged.connect(self._handle_value_changed)
        self._group_list.currentItemChanged.connect(self._handle_group_changed)
        self._order_value_cb.toggled.connect(self._handle_order_value_toggle)
        self._order_group_cb.toggled.connect(self._handle_order_group_toggle)
        self._show_box_cb.toggled.connect(self._refresh_plot)
        self._show_strip_cb.toggled.connect(self._refresh_plot)
        self._show_rug_cb.toggled.connect(self._refresh_plot)
        self._order_cb.toggled.connect(self._refresh_plot)
        self._show_grid_cb.toggled.connect(self._refresh_plot)
        self._orientation_group.idClicked.connect(lambda _index: self._refresh_plot())
        self._kernel_combo.currentTextChanged.connect(self._refresh_plot)
        self._scale_combo.currentTextChanged.connect(self._refresh_plot)

    def sizeHint(self):
        from PySide6.QtCore import QSize

        return QSize(1132, 708)

    def set_input_payload(self, payload) -> None:
        self._dataset = payload.dataset if payload is not None else None
        self._sync_controls()
        self._refresh_plot()

    def current_output_dataset(self) -> DatasetHandle | None:
        return self._selected_dataset

    def current_output_datasets(self) -> dict[str, DatasetHandle | None] | None:
        return {
            "Selected Data": self._selected_dataset,
            "Annotated Data": self._annotated_dataset,
        }

    def serialize_node_state(self) -> dict[str, object]:
        return {
            "value": self._current_value_name(),
            "group": self._current_group_name(),
            "order_by_importance": self._order_value_cb.isChecked(),
            "order_grouping_by_importance": self._order_group_cb.isChecked(),
            "orientation": self._current_orientation_label(),
            "kernel": self._kernel_combo.currentText(),
            "scale": self._scale_combo.currentText(),
            "show_box_plot": self._show_box_cb.isChecked(),
            "show_strip_plot": self._show_strip_cb.isChecked(),
            "show_rug": self._show_rug_cb.isChecked(),
            "show_grid": self._show_grid_cb.isChecked(),
            "order_subgroups": self._order_cb.isChecked(),
            "selected_rows": list(self._pending_selected_rows),
            "selection_entry": self._canvas.selection_state()[0],
            "selection_range": self._canvas.selection_state()[1],
            "selection_label": self._canvas.entry_label(self._canvas.selection_state()[0]),
        }

    def restore_node_state(self, payload: dict[str, object]) -> None:
        self._pending_selected_rows = [
            int(index)
            for index in payload.get("selected_rows", [])
            if isinstance(index, int | float)
        ]
        entry_index = payload.get("selection_entry", -1)
        self._pending_selection_entry = int(entry_index) if isinstance(entry_index, int | float) else -1
        selection_range = payload.get("selection_range")
        if (
            isinstance(selection_range, (list, tuple))
            and len(selection_range) == 2
            and all(isinstance(value, int | float) for value in selection_range)
        ):
            self._pending_selection_range = (float(selection_range[0]), float(selection_range[1]))
        else:
            self._pending_selection_range = None
        selection_label = payload.get("selection_label")
        self._pending_selection_label = str(selection_label) if selection_label else None
        self._order_value_cb.setChecked(bool(payload.get("order_by_importance", False)))
        self._order_group_cb.setChecked(bool(payload.get("order_grouping_by_importance", False)))
        self._set_combo_value(self._kernel_combo, str(payload.get("kernel", i18n.t("Normal"))))
        self._set_combo_value(self._scale_combo, str(payload.get("scale", i18n.t("Area"))))
        self._show_box_cb.setChecked(bool(payload.get("show_box_plot", True)))
        self._show_strip_cb.setChecked(bool(payload.get("show_strip_plot", False)))
        self._show_rug_cb.setChecked(bool(payload.get("show_rug", False)))
        self._show_grid_cb.setChecked(bool(payload.get("show_grid", False)))
        self._order_cb.setChecked(bool(payload.get("order_subgroups", False)))
        orientation = str(payload.get("orientation", i18n.t("Vertical")))
        self._horizontal_rb.setChecked(orientation == i18n.t("Horizontal"))
        self._vertical_rb.setChecked(orientation != i18n.t("Horizontal"))
        self._pending_value_name = str(payload.get("value", ""))
        self._pending_group_name = str(payload.get("group", i18n.t("None")))

    def documentation_url(self) -> str:
        return "https://orangedatamining.com/widget-catalog/visualize/violinplot/"

    def _sync_controls(self) -> None:
        dataset = self._dataset
        current_value = self._pending_value_name or self._current_value_name()
        current_group = self._pending_group_name or self._current_group_name()

        if dataset is None:
            self._populate_list(self._value_list, [], "")
            self._populate_list(self._group_list, [i18n.t("None")], i18n.t("None"))
            self._update_control_enablement()
            return

        default_value = next(iter(numeric_columns(dataset)), "")
        default_group = self._default_group_name()
        group_name = current_group or default_group
        value_names = self._sorted_value_names(group_name)
        value_name = current_value or default_value or (value_names[0] if value_names else "")
        group_names = [i18n.t("None"), *self._sorted_group_names(value_name)]
        group_name = current_group or default_group or i18n.t("None")

        self._populate_list(self._value_list, value_names, value_name)
        self._populate_list(self._group_list, group_names, group_name)
        self._pending_value_name = ""
        self._pending_group_name = ""
        self._update_control_enablement()

    def _refresh_plot(self) -> None:
        dataset = self._dataset
        self._selected_dataset = None
        self._annotated_dataset = None
        if dataset is None:
            self._dataset_label.setText(i18n.t("Dataset: none"))
            self._status_label.setText(i18n.t("Load numeric data to plot violins."))
            self._canvas.set_state(
                [],
                orientation=Qt.Orientation.Vertical,
                show_rug=False,
                show_box_plot=True,
                show_strip_plot=False,
                show_grid=False,
                scale_mode=i18n.t("Area"),
            )
            self._handle_selection_changed([])
            return

        self._dataset_label.setText(i18n.tf("Dataset: {name}", name=dataset.display_name))
        value_name = self._current_value_name()
        group_name = self._current_group_name()
        value = prepared_column(dataset, value_name)
        group = prepared_column(dataset, None if group_name == i18n.t("None") else group_name)
        if value is None or value.is_discrete:
            self._status_label.setText(i18n.t("Choose a numeric value column."))
            self._canvas.set_state(
                [],
                orientation=self._current_orientation(),
                show_rug=self._show_rug_cb.isChecked(),
                show_box_plot=self._show_box_cb.isChecked(),
                show_strip_plot=self._show_strip_cb.isChecked(),
                show_grid=self._show_grid_cb.isChecked(),
                scale_mode=self._scale_combo.currentText(),
            )
            self._handle_selection_changed([])
            return

        entries = self._build_entries(value, group)
        selection_entry, selection_range = self._canvas.selection_state()
        selection_label = self._pending_selection_label
        if selection_label is None:
            selection_label = self._canvas.entry_label(selection_entry)
        orientation = self._current_orientation()
        status = i18n.tf("Groups: {count}", count=len(entries))
        if self._show_strip_cb.isChecked():
            status += i18n.t(" | density dots")
        if self._order_cb.isChecked() and group is not None:
            status += i18n.t(" | ordered by median")
        self._status_label.setText(status)
        self._canvas.set_state(
            entries,
            orientation=orientation,
            show_rug=self._show_rug_cb.isChecked(),
            show_box_plot=self._show_box_cb.isChecked(),
            show_strip_plot=self._show_strip_cb.isChecked(),
            show_grid=self._show_grid_cb.isChecked(),
            scale_mode=self._scale_combo.currentText(),
        )
        restore_range = self._pending_selection_range or selection_range
        restore_label = self._pending_selection_label or selection_label
        if restore_label and restore_range is not None:
            restore_index = next((index for index, entry in enumerate(entries) if entry.label == restore_label), -1)
            self._canvas.restore_selection_state(restore_index, restore_range)
            self._pending_selection_entry = -1
            self._pending_selection_range = None
            self._pending_selection_label = None
            return
        self._handle_selection_changed(self._pending_selected_rows)

    def _build_entries(self, value: PlotColumn, group: PlotColumn | None) -> list[_ViolinEntry]:
        kernel_name = self._selected_kernel()
        if group is None:
            support, density = kernel_density(value.values, kernel=kernel_name)
            return [
                _ViolinEntry(
                    label=i18n.t("All data"),
                    support=support,
                    density=density,
                    values=value.values,
                    row_indices=value.row_indices,
                    color=QColor(Qt.GlobalColor.lightGray),
                )
            ]

        group_lookup = {row: raw for row, raw in zip(group.row_indices, group.raw_values)}
        grouped_values: dict[str, list[float]] = {}
        grouped_rows: dict[str, list[int]] = {}
        for row, raw in zip(value.row_indices, value.raw_values):
            if row not in group_lookup:
                continue
            label = str(group_lookup[row])
            grouped_values.setdefault(label, []).append(float(raw))
            grouped_rows.setdefault(label, []).append(int(row))

        ordered_labels = list(grouped_values)
        if self._order_cb.isChecked():
            ordered_labels.sort(key=lambda label: float(np.median(np.asarray(grouped_values[label], dtype=float))))

        entries: list[_ViolinEntry] = []
        for index, label in enumerate(ordered_labels):
            values = np.asarray(grouped_values[label], dtype=float)
            support, density = kernel_density(values, kernel=kernel_name)
            entries.append(
                _ViolinEntry(
                    label=label,
                    support=support,
                    density=density,
                    values=values,
                    row_indices=np.asarray(grouped_rows[label], dtype=int),
                    color=QColor(PALETTE[index % len(PALETTE)]),
                )
            )
        return entries

    def _selected_kernel(self) -> str:
        label = self._kernel_combo.currentText()
        if label == i18n.t("Epanechnikov"):
            return "epanechnikov"
        if label == i18n.t("Linear"):
            return "linear"
        return "gaussian"

    def _current_orientation(self) -> Qt.Orientation:
        return Qt.Orientation.Horizontal if self._horizontal_rb.isChecked() else Qt.Orientation.Vertical

    def _current_orientation_label(self) -> str:
        return i18n.t("Horizontal") if self._horizontal_rb.isChecked() else i18n.t("Vertical")

    def _current_value_name(self) -> str:
        item = self._value_list.currentItem()
        return item.text() if item is not None else ""

    def _current_group_name(self) -> str:
        item = self._group_list.currentItem()
        return item.text() if item is not None else i18n.t("None")

    def _handle_value_changed(self, _current, _previous) -> None:
        self._sync_controls()
        self._refresh_plot()

    def _handle_group_changed(self, _current, _previous) -> None:
        self._sync_controls()
        self._refresh_plot()

    def _handle_order_value_toggle(self) -> None:
        self._sync_controls()
        self._refresh_plot()

    def _handle_order_group_toggle(self) -> None:
        self._sync_controls()
        self._refresh_plot()

    def _handle_selection_changed(self, rows: list[int]) -> None:
        self._pending_selected_rows = sorted({index for index in rows})
        self._canvas.set_selected_rows(self._pending_selected_rows)
        self._selection_label.setText(i18n.tf("Selected: {count}", count=len(self._pending_selected_rows)))
        self._selected_dataset, self._annotated_dataset = build_selection_outputs(
            self._dataset,
            self._pending_selected_rows,
            generated_by="violin-plot",
            service=self._builder,
        )
        self._notify_output_changed()

    def _clear_selection(self) -> None:
        self._canvas.clear_selection()

    @staticmethod
    def _set_combo_value(combo: QComboBox, value: str) -> None:
        if not value:
            return
        index = combo.findText(value)
        if index >= 0:
            combo.setCurrentIndex(index)

    def _populate_list(self, widget: QListWidget, values: list[str], current: str) -> None:
        widget.blockSignals(True)
        widget.clear()
        for value in values:
            widget.addItem(QListWidgetItem(value))
        matches = widget.findItems(current, Qt.MatchFlag.MatchExactly)
        if matches:
            widget.setCurrentItem(matches[0])
        elif widget.count():
            widget.setCurrentRow(0)
        widget.blockSignals(False)

    def _sorted_value_names(self, group_name: str) -> list[str]:
        dataset = self._dataset
        names = sorted(numeric_columns(dataset), key=str.lower)
        if dataset is None or not self._order_value_cb.isChecked() or group_name == i18n.t("None"):
            return names
        group = prepared_column(dataset, group_name)
        if group is None or not group.is_discrete:
            return names
        return sorted(names, key=lambda name: (self._anova_score(name, group), name.lower()))

    def _sorted_group_names(self, value_name: str) -> list[str]:
        dataset = self._dataset
        names = sorted(discrete_columns(dataset), key=str.lower)
        if dataset is None or not self._order_group_cb.isChecked() or not value_name:
            return names
        value = prepared_column(dataset, value_name)
        if value is None or value.is_discrete:
            return names
        return sorted(names, key=lambda name: (self._group_score(name, value), name.lower()))

    def _anova_score(self, value_name: str, group: PlotColumn) -> float:
        value = prepared_column(self._dataset, value_name)
        if value is None or value.is_discrete:
            return 2.0
        group_lookup = {int(row): str(raw) for row, raw in zip(group.row_indices, group.raw_values)}
        grouped: dict[str, list[float]] = {}
        for row, raw in zip(value.row_indices, value.raw_values):
            label = group_lookup.get(int(row))
            if label is None:
                continue
            grouped.setdefault(label, []).append(float(raw))
        return self._anova_p_value(grouped)

    def _group_score(self, group_name: str, value: PlotColumn) -> float:
        group = prepared_column(self._dataset, group_name)
        if group is None or not group.is_discrete:
            return 2.0
        group_lookup = {int(row): str(raw) for row, raw in zip(group.row_indices, group.raw_values)}
        grouped: dict[str, list[float]] = {}
        for row, raw in zip(value.row_indices, value.raw_values):
            label = group_lookup.get(int(row))
            if label is None:
                continue
            grouped.setdefault(label, []).append(float(raw))
        return self._anova_p_value(grouped)

    @staticmethod
    def _anova_p_value(grouped: dict[str, list[float]]) -> float:
        samples = [np.asarray(values, dtype=float) for values in grouped.values() if len(values) > 0]
        if len(samples) < 2:
            return 2.0
        try:
            p_value = float(stats.f_oneway(*samples).pvalue)
        except ValueError:
            return 2.0
        if np.isnan(p_value):
            return 2.0
        return p_value

    def _default_group_name(self) -> str:
        dataset = self._dataset
        if dataset is None:
            return i18n.t("None")
        for column in dataset.domain.target_columns:
            if column.logical_type in {"categorical", "boolean", "string", "text"}:
                return column.name
        return i18n.t("None")

    def _update_control_enablement(self) -> None:
        group_selected = self._current_group_name() != i18n.t("None")
        has_dataset = self._dataset is not None
        self._order_cb.setEnabled(group_selected or not has_dataset)
        self._scale_combo.setEnabled(group_selected or not has_dataset)
        self._order_value_cb.setEnabled(group_selected and has_dataset)
        self._order_group_cb.setEnabled(bool(self._current_value_name()) and has_dataset)
