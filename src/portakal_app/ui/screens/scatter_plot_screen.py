from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from PySide6.QtCore import QPoint, QRect, QRectF, QSize, Qt, Signal
from PySide6.QtGui import QColor, QFont, QPainter, QPen, QPolygon
from PySide6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QSlider,
    QSizePolicy,
    QSpinBox,
    QToolTip,
    QVBoxLayout,
    QWidget,
)

from portakal_app.data.models import DatasetHandle
from portakal_app.data.services.generated_dataset_service import GeneratedDatasetService
from portakal_app.models import WorkflowPayload
from portakal_app.ui import i18n
from portakal_app.ui.screens.node_screen import WorkflowNodeScreenSupport
from portakal_app.ui.screens.visualize_common import (
    PALETTE,
    PlotColumn,
    build_selection_outputs,
    discrete_columns,
    feature_names_from_payload,
    feature_output_payload,
    gradient_color,
    nice_ticks,
    prepared_column,
    primitive_columns,
    rank_scatter_pairs,
    suggest_scatter_pair,
    subset_row_indices,
)


@dataclass(frozen=True)
class _AxisErrorConfig:
    upper: str = ""
    lower: str = ""
    absolute: bool = False

    def summary(self) -> str:
        parts = [name for name in (self.lower, self.upper) if name]
        if not parts:
            return i18n.t("None")
        text = " / ".join(parts)
        if self.absolute:
            text += i18n.t(" (absolute)")
        return text


@dataclass(frozen=True)
class _ScatterPoint:
    row_index: int
    x: float
    y: float
    raw_x: object
    raw_y: object
    color: QColor
    legend: str
    subset: bool
    label: str = ""
    radius: float = 4.0
    shape: str = "circle"
    x_error_left: float = 0.0
    x_error_right: float = 0.0
    y_error_bottom: float = 0.0
    y_error_top: float = 0.0
    tooltip: str = ""


@dataclass(frozen=True)
class _AggregateMarker:
    row_indices: tuple[int, ...]
    x: float
    y: float
    color: QColor
    count: int


class _AxisErrorDialog(QDialog):
    def __init__(self, axis_label: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle(i18n.tf("{axis} Error Bars", axis=axis_label))
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)

        self._lower_combo = QComboBox(self)
        self._upper_combo = QComboBox(self)
        self._absolute_cb = QCheckBox(i18n.t("Input values are absolute bounds"), self)

        for label, combo in (
            (i18n.t("Lower"), self._lower_combo),
            (i18n.t("Upper"), self._upper_combo),
        ):
            layout.addWidget(QLabel(label, self))
            layout.addWidget(combo)
        layout.addWidget(self._absolute_cb)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel,
            parent=self,
        )
        clear_button = buttons.addButton(i18n.t("Clear"), QDialogButtonBox.ButtonRole.ResetRole)
        clear_button.clicked.connect(self._clear)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def set_options(self, numeric_columns: list[str], current: _AxisErrorConfig) -> None:
        for combo in (self._lower_combo, self._upper_combo):
            combo.blockSignals(True)
            combo.clear()
            combo.addItem(i18n.t("None"))
            combo.addItems(numeric_columns)
            combo.blockSignals(False)
        self._set_combo_value(self._lower_combo, current.lower or i18n.t("None"))
        self._set_combo_value(self._upper_combo, current.upper or i18n.t("None"))
        self._absolute_cb.setChecked(current.absolute)

    def selected_config(self) -> _AxisErrorConfig:
        lower = self._lower_combo.currentText()
        upper = self._upper_combo.currentText()
        none_text = i18n.t("None")
        return _AxisErrorConfig(
            lower="" if lower == none_text else lower,
            upper="" if upper == none_text else upper,
            absolute=self._absolute_cb.isChecked(),
        )

    def _clear(self) -> None:
        self._lower_combo.setCurrentIndex(0)
        self._upper_combo.setCurrentIndex(0)
        self._absolute_cb.setChecked(False)

    @staticmethod
    def _set_combo_value(combo: QComboBox, value: str) -> None:
        index = combo.findText(value)
        combo.setCurrentIndex(index if index >= 0 else 0)


class _VizRankDialog(QDialog):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle(i18n.t("Find Informative Projections"))
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)

        self._description = QLabel(
            i18n.t("Ranked pairs of numeric attributes based on class separation or variance."),
            self,
        )
        self._description.setWordWrap(True)
        layout.addWidget(self._description)

        self._list = QListWidget(self)
        layout.addWidget(self._list, 1)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel,
            parent=self,
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        self._list.itemDoubleClicked.connect(lambda _item: self.accept())

    def set_ranked_pairs(
        self,
        pairs: list[tuple[str, str, float]],
        *,
        current_pair: tuple[str, str] | None = None,
    ) -> None:
        self._list.clear()
        selected_row = 0
        for row, (x_name, y_name, score) in enumerate(pairs):
            item = QListWidgetItem(f"{x_name} vs {y_name}    [{score:.3f}]", self._list)
            item.setData(Qt.ItemDataRole.UserRole, (x_name, y_name))
            if current_pair == (x_name, y_name):
                selected_row = row
        if self._list.count():
            self._list.setCurrentRow(selected_row)

    def selected_pair(self) -> tuple[str, str] | None:
        item = self._list.currentItem()
        if item is None:
            return None
        value = item.data(Qt.ItemDataRole.UserRole)
        if isinstance(value, tuple) and len(value) == 2:
            return str(value[0]), str(value[1])
        return None


class _ScatterCanvas(QWidget):
    selectionChanged = Signal(list)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._points: list[_ScatterPoint] = []
        self._x_col: PlotColumn | None = None
        self._y_col: PlotColumn | None = None
        self._selected_rows: set[int] = set()
        self._legend_items: list[tuple[str, QColor]] = []
        self._show_regression = False
        self._show_ellipse = False
        self._show_subset = True
        self._show_grid = True
        self._show_legend = True
        self._show_density = False
        self._aggregate_points = True
        self._label_only_selected = False
        self._point_size = 10
        self._alpha_value = 180
        self._jitter_continuous = False
        self._orthonormal_regression = False
        self._jitter = 0
        self._hit_regions: list[tuple[QRect, tuple[int, ...], str]] = []
        self._selection_rect: QRect | None = None
        self._drag_start: QPoint | None = None
        self._drag_mode = "select"
        self._pan_origin: QPoint | None = None
        self._pan_ranges: tuple[tuple[float, float], tuple[float, float]] | None = None
        self._x_range = (0.0, 1.0)
        self._y_range = (0.0, 1.0)
        self._data_x_range = (0.0, 1.0)
        self._data_y_range = (0.0, 1.0)
        self._regression: tuple[float, float] | None = None
        self.setMinimumHeight(280)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMouseTracking(True)

    def set_state(
        self,
        *,
        points: list[_ScatterPoint],
        x_col: PlotColumn | None,
        y_col: PlotColumn | None,
        legend_items: list[tuple[str, QColor]],
        show_regression: bool,
        show_ellipse: bool,
        show_subset: bool,
        show_grid: bool,
        show_legend: bool,
        show_density: bool,
        aggregate_points: bool,
        label_only_selected: bool,
        point_size: int,
        alpha_value: int,
        jitter_continuous: bool,
        orthonormal_regression: bool,
        jitter: int,
    ) -> None:
        prev_x_name = self._x_col.name if self._x_col is not None else ""
        prev_y_name = self._y_col.name if self._y_col is not None else ""
        prev_x_range = self._x_range
        prev_y_range = self._y_range
        prev_data_x_range = self._data_x_range
        prev_data_y_range = self._data_y_range
        self._points = points
        self._x_col = x_col
        self._y_col = y_col
        self._legend_items = legend_items
        self._show_regression = show_regression
        self._show_ellipse = show_ellipse
        self._show_subset = show_subset
        self._show_grid = show_grid
        self._show_legend = show_legend
        self._show_density = show_density
        self._aggregate_points = aggregate_points
        self._label_only_selected = label_only_selected
        self._point_size = point_size
        self._alpha_value = alpha_value
        self._jitter_continuous = jitter_continuous
        self._orthonormal_regression = orthonormal_regression
        self._jitter = jitter
        self._hit_regions = []
        self._selection_rect = None
        self._recompute_ranges()
        same_axes = prev_x_name == (x_col.name if x_col is not None else "") and prev_y_name == (
            y_col.name if y_col is not None else ""
        )
        had_custom_view = not (
            self._ranges_close(prev_x_range, prev_data_x_range)
            and self._ranges_close(prev_y_range, prev_data_y_range)
        )
        if same_axes and had_custom_view and self._points:
            self._x_range = self._clamp_view(prev_x_range, self._data_x_range)
            self._y_range = self._clamp_view(prev_y_range, self._data_y_range)
        else:
            self.reset_view()
        self.update()
        if self._show_density:
            print("Color region repaint requested", flush=True)
            self.repaint()

    def set_selected_rows(self, rows: list[int]) -> None:
        self._selected_rows = set(rows)
        self.update()

    def clear_selection(self) -> None:
        if not self._selected_rows:
            return
        self._selected_rows.clear()
        self.selectionChanged.emit([])
        self.update()

    def set_mode(self, mode: str) -> None:
        self._drag_mode = mode
        self._selection_rect = None
        self._drag_start = None
        self._pan_origin = None
        self._pan_ranges = None
        if mode == "pan":
            self.setCursor(Qt.CursorShape.OpenHandCursor)
        elif mode == "zoom":
            self.setCursor(Qt.CursorShape.CrossCursor)
        else:
            self.setCursor(Qt.CursorShape.ArrowCursor)
        self.update()

    def reset_view(self) -> None:
        self._x_range = self._data_x_range
        self._y_range = self._data_y_range
        self.update()

    def zoom(self, factor: float, center: tuple[float, float] | None = None) -> None:
        if factor <= 0:
            return
        cx = (self._x_range[0] + self._x_range[1]) / 2 if center is None else center[0]
        cy = (self._y_range[0] + self._y_range[1]) / 2 if center is None else center[1]
        x0, x1 = self._x_range
        y0, y1 = self._y_range
        self._x_range = (cx - (cx - x0) * factor, cx + (x1 - cx) * factor)
        self._y_range = (cy - (cy - y0) * factor, cy + (y1 - cy) * factor)
        self._x_range = self._clamp_view(self._x_range, self._data_x_range)
        self._y_range = self._clamp_view(self._y_range, self._data_y_range)
        self.update()

    def zoom_in(self) -> None:
        self.zoom(0.85)

    def zoom_out(self) -> None:
        self.zoom(1.18)

    def mousePressEvent(self, event) -> None:
        if event.button() != Qt.MouseButton.LeftButton:
            return super().mousePressEvent(event)
        chart = self._chart_rect()
        pos = event.position().toPoint()
        if self._drag_mode == "pan" and chart.contains(pos):
            self._pan_origin = pos
            self._pan_ranges = (self._x_range, self._y_range)
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            return
        self._drag_start = pos
        self._selection_rect = QRect(pos, pos)
        self.update()

    def mouseMoveEvent(self, event) -> None:
        pos = event.position().toPoint()
        if self._pan_origin is not None and self._pan_ranges is not None:
            chart = self._chart_rect()
            if chart.width() > 0 and chart.height() > 0:
                (x0, x1), (y0, y1) = self._pan_ranges
                dx = (pos.x() - self._pan_origin.x()) / chart.width() * (x1 - x0)
                dy = (pos.y() - self._pan_origin.y()) / chart.height() * (y1 - y0)
                self._x_range = (x0 - dx, x1 - dx)
                self._y_range = (y0 + dy, y1 + dy)
                self.update()
            return
        if self._drag_start is not None:
            self._selection_rect = QRect(self._drag_start, pos).normalized()
            self.update()
            return

        for rect, _rows, tooltip in self._hit_regions:
            if rect.contains(pos):
                QToolTip.showText(event.globalPosition().toPoint(), tooltip, self)
                return
        QToolTip.hideText()

    def mouseReleaseEvent(self, event) -> None:
        if event.button() != Qt.MouseButton.LeftButton:
            return super().mouseReleaseEvent(event)
        if self._pan_origin is not None:
            self._pan_origin = None
            self._pan_ranges = None
            self.setCursor(Qt.CursorShape.OpenHandCursor)
            return
        if self._selection_rect is None:
            return super().mouseReleaseEvent(event)

        rect = QRect(self._selection_rect)
        self._drag_start = None
        self._selection_rect = None
        if self._drag_mode == "zoom":
            chart = self._chart_rect()
            if rect.width() >= 6 and rect.height() >= 6 and chart.intersects(rect):
                clipped = rect.intersected(chart)
                left, top = self._pixel_to_data(chart, clipped.left(), clipped.top())
                right, bottom = self._pixel_to_data(chart, clipped.right(), clipped.bottom())
                self._x_range = (min(left, right), max(left, right))
                self._y_range = (min(top, bottom), max(top, bottom))
                self.update()
            return
        if self._drag_mode != "select":
            return
        if rect.width() < 6 or rect.height() < 6:
            clicked_rows = {
                row
                for point_rect, rows, _tooltip in self._hit_regions
                if point_rect.contains(event.position().toPoint())
                for row in rows
            }
            modifiers = event.modifiers()
            if modifiers & Qt.KeyboardModifier.AltModifier:
                self._selected_rows.difference_update(clicked_rows)
            elif modifiers & (Qt.KeyboardModifier.ShiftModifier | Qt.KeyboardModifier.ControlModifier):
                self._selected_rows.update(clicked_rows)
            else:
                self._selected_rows = set(clicked_rows)
        else:
            incoming = {
                row
                for point_rect, rows, _tooltip in self._hit_regions
                if rect.intersects(point_rect)
                for row in rows
            }
            modifiers = event.modifiers()
            if modifiers & Qt.KeyboardModifier.AltModifier:
                self._selected_rows.difference_update(incoming)
            elif modifiers & (Qt.KeyboardModifier.ShiftModifier | Qt.KeyboardModifier.ControlModifier):
                self._selected_rows.update(incoming)
            else:
                self._selected_rows = set(incoming)
        self.selectionChanged.emit(sorted(self._selected_rows))
        self.update()

    def leaveEvent(self, _event) -> None:
        QToolTip.hideText()

    def wheelEvent(self, event) -> None:
        chart = self._chart_rect()
        pos = event.position().toPoint()
        if not chart.contains(pos):
            return super().wheelEvent(event)
        if event.angleDelta().y() == 0:
            return
        factor = 0.85 if event.angleDelta().y() > 0 else 1.18
        center = self._pixel_to_data(chart, pos.x(), pos.y())
        self.zoom(factor, center=center)
        event.accept()

    def _recompute_ranges(self) -> None:
        if not self._points:
            self._data_x_range = (0.0, 1.0)
            self._data_y_range = (0.0, 1.0)
            self._regression = None
            return
        x_values = np.asarray([point.x for point in self._points], dtype=float)
        y_values = np.asarray([point.y for point in self._points], dtype=float)
        x_lows = np.asarray([point.x - point.x_error_left for point in self._points], dtype=float)
        x_highs = np.asarray([point.x + point.x_error_right for point in self._points], dtype=float)
        y_lows = np.asarray([point.y - point.y_error_bottom for point in self._points], dtype=float)
        y_highs = np.asarray([point.y + point.y_error_top for point in self._points], dtype=float)
        x_low, x_high = float(np.min(x_lows)), float(np.max(x_highs))
        y_low, y_high = float(np.min(y_lows)), float(np.max(y_highs))
        if self._x_col is not None and self._x_col.is_discrete:
            x_low, x_high = -0.5, max(float(len(self._x_col.categories)) - 0.5, 0.5)
        else:
            x_pad = (x_high - x_low) * 0.05 or 0.5
            x_low -= x_pad
            x_high += x_pad
        if self._y_col is not None and self._y_col.is_discrete:
            y_low, y_high = -0.5, max(float(len(self._y_col.categories)) - 0.5, 0.5)
        else:
            y_pad = (y_high - y_low) * 0.05 or 0.5
            y_low -= y_pad
            y_high += y_pad
        self._data_x_range = (x_low, x_high)
        self._data_y_range = (y_low, y_high)

        if self._show_regression and len(self._points) >= 2 and self._x_col and self._y_col:
            if not self._x_col.is_discrete and not self._y_col.is_discrete:
                xs = x_values
                ys = y_values
                x_mean = float(np.mean(xs))
                y_mean = float(np.mean(ys))
                var_x = float(np.var(xs))
                if var_x > 1e-12:
                    if self._orthonormal_regression:
                        centered = np.column_stack((xs - x_mean, ys - y_mean))
                        _, _, vh = np.linalg.svd(centered, full_matrices=False)
                        direction = vh[0]
                        if abs(float(direction[0])) < 1e-12:
                            self._regression = None
                        else:
                            slope = float(direction[1] / direction[0])
                            intercept = y_mean - slope * x_mean
                            self._regression = (slope, intercept)
                    else:
                        cov = float(np.mean((xs - x_mean) * (ys - y_mean)))
                        slope = cov / var_x
                        intercept = y_mean - slope * x_mean
                        self._regression = (slope, intercept)
                else:
                    self._regression = None
            else:
                self._regression = None
        else:
            self._regression = None

    @staticmethod
    def _ranges_close(left: tuple[float, float], right: tuple[float, float]) -> bool:
        return math.isclose(left[0], right[0], rel_tol=1e-6, abs_tol=1e-6) and math.isclose(
            left[1], right[1], rel_tol=1e-6, abs_tol=1e-6
        )

    @staticmethod
    def _clamp_view(view: tuple[float, float], data: tuple[float, float]) -> tuple[float, float]:
        low, high = sorted((float(view[0]), float(view[1])))
        data_low, data_high = float(data[0]), float(data[1])
        if data_high <= data_low:
            return data_low, data_high
        data_span = data_high - data_low
        view_span = max(high - low, 1e-9)
        if view_span >= data_span:
            return data_low, data_high
        if low < data_low:
            high += data_low - low
            low = data_low
        if high > data_high:
            low -= high - data_high
            high = data_high
        low = max(low, data_low)
        high = min(high, data_high)
        if high - low >= 1e-9:
            return low, high
        return data_low, data_high

    def _map_to_pixel(self, rect: QRect, x: float, y: float) -> tuple[int, int]:
        x_low, x_high = self._x_range
        y_low, y_high = self._y_range
        span_x = x_high - x_low or 1.0
        span_y = y_high - y_low or 1.0
        px = rect.left() + int((x - x_low) / span_x * rect.width())
        py = rect.bottom() - int((y - y_low) / span_y * rect.height())
        return px, py

    def _pixel_to_data(self, rect: QRect, px: int, py: int) -> tuple[float, float]:
        x_low, x_high = self._x_range
        y_low, y_high = self._y_range
        span_x = x_high - x_low or 1.0
        span_y = y_high - y_low or 1.0
        x = x_low + ((px - rect.left()) / max(rect.width(), 1)) * span_x
        y = y_high - ((py - rect.top()) / max(rect.height(), 1)) * span_y
        return float(x), float(y)

    def _jittered(self, point: _ScatterPoint) -> tuple[float, float]:
        if self._jitter <= 0:
            return point.x, point.y
        seed = point.row_index * 1103515245 + 12345
        dx = ((seed % 997) / 997.0 - 0.5) * self._jitter / 12.0
        dy = (((seed // 997) % 997) / 997.0 - 0.5) * self._jitter / 12.0
        if self._x_col is not None and (self._x_col.is_discrete or self._jitter_continuous):
            x = point.x + dx
        else:
            x = point.x
        if self._y_col is not None and (self._y_col.is_discrete or self._jitter_continuous):
            y = point.y + dy
        else:
            y = point.y
        return x, y

    def paintEvent(self, _event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), QColor("#fffdf9"))

        if not self._points or self._x_col is None or self._y_col is None:
            painter.setPen(QColor("#8d877d"))
            painter.drawText(
                self.rect(),
                Qt.AlignmentFlag.AlignCenter,
                i18n.t("Load data and choose X/Y variables."),
            )
            return

        chart = self._chart_rect()
        margin_left = chart.left()
        margin_right = 160 if self._show_legend and self._legend_items else 20
        self._hit_regions = []

        if self._show_density:
            self._draw_density(painter, chart)

        if self._show_grid:
            painter.setPen(QPen(QColor("#e7dfd3"), 1, Qt.PenStyle.DotLine))
            for tick in self._x_ticks():
                px, _ = self._map_to_pixel(chart, tick, self._y_range[0])
                painter.drawLine(px, chart.top(), px, chart.bottom())
            for tick in self._y_ticks():
                _, py = self._map_to_pixel(chart, self._x_range[0], tick)
                painter.drawLine(chart.left(), py, chart.right(), py)

        painter.setPen(QPen(QColor("#9b9488"), 1))
        painter.drawRect(chart)

        painter.setPen(QColor("#534b40"))
        painter.setFont(QFont(self.font().family(), 9))
        painter.drawText(chart.adjusted(0, chart.height() + 14, 0, 34), Qt.AlignmentFlag.AlignCenter, self._x_col.name)
        painter.save()
        painter.translate(20, chart.center().y())
        painter.rotate(-90)
        painter.drawText(QRect(-90, -8, 180, 16), Qt.AlignmentFlag.AlignCenter, self._y_col.name)
        painter.restore()

        painter.setFont(QFont(self.font().family(), 8))
        for tick in self._x_ticks():
            px, _ = self._map_to_pixel(chart, tick, self._y_range[0])
            text = self._tick_label(self._x_col, tick)
            painter.drawText(QRect(px - 36, chart.bottom() + 4, 72, 16), Qt.AlignmentFlag.AlignCenter, text)
        for tick in self._y_ticks():
            _, py = self._map_to_pixel(chart, self._x_range[0], tick)
            text = self._tick_label(self._y_col, tick)
            painter.drawText(QRect(2, py - 8, margin_left - 8, 16), Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter, text)

        if self._show_ellipse and len(self._points) >= 3 and not self._x_col.is_discrete and not self._y_col.is_discrete:
            self._draw_ellipse(painter, chart)
        if self._regression is not None:
            slope, intercept = self._regression
            x0 = self._x_range[0]
            x1 = self._x_range[1]
            p0 = self._map_to_pixel(chart, x0, slope * x0 + intercept)
            p1 = self._map_to_pixel(chart, x1, slope * x1 + intercept)
            painter.setPen(QPen(QColor(80, 80, 80, 170), 2, Qt.PenStyle.DashLine))
            painter.drawLine(p0[0], p0[1], p1[0], p1[1])

        aggregate_lookup = self._aggregate_lookup(chart) if self._aggregate_points else {}
        drawn_rows: set[int] = set()
        for marker in aggregate_lookup.values():
            for row in marker.row_indices:
                drawn_rows.add(row)
            px, py = self._map_to_pixel(chart, marker.x, marker.y)
            radius = max(8, 5 + int(math.sqrt(marker.count)))
            color = QColor(marker.color)
            color.setAlpha(max(140, self._alpha_value))
            painter.setPen(QPen(QColor("#3a3127"), 1.5))
            painter.setBrush(color)
            painter.drawEllipse(px - radius, py - radius, radius * 2, radius * 2)
            painter.setPen(QColor("#ffffff"))
            painter.drawText(QRect(px - radius, py - 8, radius * 2, 16), Qt.AlignmentFlag.AlignCenter, str(marker.count))
            tooltip = i18n.tf("Aggregated points: {count}", count=marker.count)
            self._hit_regions.append((QRect(px - radius - 3, py - radius - 3, radius * 2 + 6, radius * 2 + 6), marker.row_indices, tooltip))

        for point in self._points:
            if point.row_index in drawn_rows:
                continue
            draw_x, draw_y = self._jittered(point)
            px, py = self._map_to_pixel(chart, draw_x, draw_y)
            if (point.x_error_left > 0 or point.x_error_right > 0) and not self._x_col.is_discrete:
                px0, _ = self._map_to_pixel(chart, draw_x - point.x_error_left, draw_y)
                px1, _ = self._map_to_pixel(chart, draw_x + point.x_error_right, draw_y)
                painter.setPen(QPen(QColor("#505050"), 1.1))
                painter.drawLine(px0, py, px1, py)
                painter.drawLine(px0, py - 4, px0, py + 4)
                painter.drawLine(px1, py - 4, px1, py + 4)
            if (point.y_error_bottom > 0 or point.y_error_top > 0) and not self._y_col.is_discrete:
                _, py0 = self._map_to_pixel(chart, draw_x, draw_y - point.y_error_bottom)
                _, py1 = self._map_to_pixel(chart, draw_x, draw_y + point.y_error_top)
                painter.setPen(QPen(QColor("#505050"), 1.1))
                painter.drawLine(px, py0, px, py1)
                painter.drawLine(px - 4, py0, px + 4, py0)
                painter.drawLine(px - 4, py1, px + 4, py1)
            radius = max(3, int(round(point.radius * max(self._point_size, 1) / 10.0)))
            if point.row_index in self._selected_rows:
                radius += 1
            rect = QRect(px - radius, py - radius, radius * 2, radius * 2)
            fill = QColor(point.color)
            fill.setAlpha(self._alpha_value)
            border = QColor("#111111") if point.row_index in self._selected_rows else QColor(fill.darker(150))
            if point.subset and self._show_subset:
                painter.setPen(QPen(QColor("#111111"), 2))
                painter.setBrush(QColor("#ffffff"))
                self._draw_marker(painter, point.shape, px, py, radius + 2)
            painter.setPen(QPen(border, 1.4))
            painter.setBrush(fill)
            self._draw_marker(painter, point.shape, px, py, radius)
            tooltip = point.tooltip or "\n".join(
                [
                    f"Row: {point.row_index}",
                    f"{self._x_col.name}: {point.raw_x}",
                    f"{self._y_col.name}: {point.raw_y}",
                ]
            )
            self._hit_regions.append((rect.adjusted(-4, -4, 4, 4), (point.row_index,), tooltip))
            if point.label and (not self._label_only_selected or point.row_index in self._selected_rows):
                painter.setPen(QColor("#3b2a10"))
                painter.drawText(
                    QRect(px + radius + 4, py - 10, 120, 20),
                    Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
                    point.label,
                )

        if self._selection_rect is not None:
            painter.setPen(QPen(QColor("#2563eb"), 1.5, Qt.PenStyle.DashLine))
            painter.setBrush(QColor(37, 99, 235, 32))
            painter.drawRect(self._selection_rect)

        if self._show_legend and self._legend_items:
            legend_rect = QRect(chart.right() + 18, chart.top(), margin_right - 28, min(chart.height(), 180))
            painter.setPen(QColor("#534b40"))
            painter.drawText(legend_rect.adjusted(0, 0, 0, -legend_rect.height() + 18), Qt.AlignmentFlag.AlignLeft, i18n.t("Legend"))
            y = legend_rect.top() + 22
            for label, color in self._legend_items[:8]:
                painter.setPen(QPen(QColor("#534b40"), 1))
                painter.setBrush(color)
                painter.drawEllipse(legend_rect.left(), y + 2, 10, 10)
                painter.drawText(QRect(legend_rect.left() + 16, y, legend_rect.width() - 16, 16), Qt.AlignmentFlag.AlignLeft, label)
                y += 18

    def _chart_rect(self) -> QRect:
        margin_left = 72
        margin_right = 160 if self._show_legend and self._legend_items else 20
        margin_top = 18
        margin_bottom = 64
        return QRect(
            margin_left,
            margin_top,
            max(10, self.width() - margin_left - margin_right),
            max(10, self.height() - margin_top - margin_bottom),
        )

    def _aggregate_lookup(self, chart: QRect) -> dict[tuple[int, int], _AggregateMarker]:
        if not self._points:
            return {}
        buckets: dict[tuple[int, int], list[_ScatterPoint]] = {}
        cell = 18
        for point in self._points:
            draw_x, draw_y = self._jittered(point)
            px, py = self._map_to_pixel(chart, draw_x, draw_y)
            key = ((px - chart.left()) // cell, (py - chart.top()) // cell)
            buckets.setdefault(key, []).append(point)
        markers: dict[tuple[int, int], _AggregateMarker] = {}
        for key, group in buckets.items():
            if len(group) < 4:
                continue
            xs = [point.x for point in group]
            ys = [point.y for point in group]
            color = QColor(group[0].color)
            markers[key] = _AggregateMarker(
                row_indices=tuple(point.row_index for point in group),
                x=float(np.mean(xs)),
                y=float(np.mean(ys)),
                color=color,
                count=len(group),
            )
        return markers

    def _draw_density(self, painter: QPainter, chart: QRect) -> None:
        print("Color region calc started", flush=True)
        if self._x_col is None or self._y_col is None or self._x_col.is_discrete or self._y_col.is_discrete:
            print("Color region skipped: X/Y axes are missing or discrete", flush=True)
            return
        if not self._legend_items or not self._points:
            print("Color region skipped: no legend items or plot points", flush=True)
            return
        step = max(8, min(16, int(max(chart.width(), chart.height()) / 70)))
        cols = max(1, int(math.ceil(chart.width() / step)))
        rows = max(1, int(math.ceil(chart.height() / step)))
        xs = chart.left() + np.arange(cols, dtype=float) * step + step / 2.0
        ys = chart.top() + np.arange(rows, dtype=float) * step + step / 2.0
        if xs.size == 0 or ys.size == 0:
            print("Color region skipped: empty meshgrid", flush=True)
            return
        pixel_grid = np.array([(x, y) for y in ys for x in xs], dtype=float)
        print(
            f"Meshgrid created: rows={rows}, cols={cols}, cells={pixel_grid.shape[0]}, step={step}",
            flush=True,
        )
        x_low, x_high = self._x_range
        y_low, y_high = self._y_range
        span_x = x_high - x_low or 1.0
        span_y = y_high - y_low or 1.0
        print(
            f"Color region bounds: x=({x_low:.6g}, {x_high:.6g}), y=({y_low:.6g}, {y_high:.6g})",
            flush=True,
        )
        query = np.column_stack(
            (
                x_low + ((pixel_grid[:, 0] - chart.left()) / max(chart.width(), 1)) * span_x,
                y_high - ((pixel_grid[:, 1] - chart.top()) / max(chart.height(), 1)) * span_y,
            )
        )
        samples = np.asarray([(point.x, point.y) for point in self._points], dtype=float)
        if samples.ndim != 2 or samples.shape[0] == 0:
            print("Color region skipped: no valid samples", flush=True)
            return
        color_map = {label: color for label, color in self._legend_items}
        legends = [point.legend for point in self._points]
        nearest: list[int] = []
        chunk = 2048
        for start in range(0, query.shape[0], chunk):
            block = query[start:start + chunk]
            distances = np.sum((block[:, np.newaxis, :] - samples[np.newaxis, :, :]) ** 2, axis=2)
            nearest.extend(np.argmin(distances, axis=1).astype(int).tolist())
        print(f"Adding to plot: nearest-neighbor cells={len(nearest)}", flush=True)
        painter.save()
        painter.setClipRect(chart)
        painter.setPen(Qt.PenStyle.NoPen)
        for index, sample_index in enumerate(nearest):
            legend = legends[sample_index]
            color = QColor(color_map.get(legend, QColor("#94a3b8")))
            color.setAlpha(82)
            left = int(pixel_grid[index, 0] - step / 2.0)
            top = int(pixel_grid[index, 1] - step / 2.0)
            painter.setBrush(color)
            painter.drawRect(QRectF(left, top, step + 1, step + 1))
        painter.restore()
        print("Color region draw complete", flush=True)

    def _x_ticks(self) -> list[float]:
        if self._x_col and self._x_col.is_discrete:
            return [float(index) for index, _ in enumerate(self._x_col.categories)]
        return nice_ticks(*self._x_range, count=5)

    def _y_ticks(self) -> list[float]:
        if self._y_col and self._y_col.is_discrete:
            return [float(index) for index, _ in enumerate(self._y_col.categories)]
        return nice_ticks(*self._y_range, count=5)

    @staticmethod
    def _tick_label(column: PlotColumn, value: float) -> str:
        if column.is_discrete:
            index = int(round(value))
            if 0 <= index < len(column.categories):
                return column.categories[index]
        return f"{value:.3g}"

    def _draw_ellipse(self, painter: QPainter, chart: QRect) -> None:
        xs = np.asarray([point.x for point in self._points], dtype=float)
        ys = np.asarray([point.y for point in self._points], dtype=float)
        cov = np.cov(xs, ys)
        if cov.shape != (2, 2) or not np.isfinite(cov).all():
            return
        values, vectors = np.linalg.eigh(cov)
        if not np.isfinite(values).all():
            return
        order = values.argsort()[::-1]
        values = values[order]
        vectors = vectors[:, order]
        if values[0] <= 0 or values[1] <= 0:
            return
        mean = np.array([np.mean(xs), np.mean(ys)])
        theta = math.atan2(vectors[1, 0], vectors[0, 0])
        width = 2.4477 * math.sqrt(values[0]) * 2
        height = 2.4477 * math.sqrt(values[1]) * 2

        painter.save()
        center_px = self._map_to_pixel(chart, float(mean[0]), float(mean[1]))
        x_low, x_high = self._x_range
        y_low, y_high = self._y_range
        span_x = x_high - x_low or 1.0
        span_y = y_high - y_low or 1.0
        px_width = width / span_x * chart.width()
        px_height = height / span_y * chart.height()
        painter.translate(center_px[0], center_px[1])
        painter.rotate(-math.degrees(theta))
        painter.setPen(QPen(QColor(37, 99, 235, 130), 1.5, Qt.PenStyle.DashDotLine))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawEllipse(QRectF(-px_width / 2, -px_height / 2, px_width, px_height))
        painter.restore()

    @staticmethod
    def _draw_marker(painter: QPainter, shape: str, px: int, py: int, radius: int) -> None:
        if shape == "square":
            painter.drawRect(px - radius, py - radius, radius * 2, radius * 2)
        elif shape == "triangle":
            painter.drawPolygon(QPolygon([QPoint(px, py - radius), QPoint(px - radius, py + radius), QPoint(px + radius, py + radius)]))
        elif shape == "diamond":
            painter.drawPolygon(QPolygon([QPoint(px, py - radius), QPoint(px - radius, py), QPoint(px, py + radius), QPoint(px + radius, py)]))
        else:
            painter.drawEllipse(px - radius, py - radius, radius * 2, radius * 2)


class ScatterPlotScreen(QWidget, WorkflowNodeScreenSupport):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._init_workflow_node_support()
        self._builder = GeneratedDatasetService()
        self._dataset: DatasetHandle | None = None
        self._subset: DatasetHandle | None = None
        self._selected_dataset: DatasetHandle | None = None
        self._annotated_dataset: DatasetHandle | None = None
        self._requested_features: tuple[str, ...] = ()
        self._feature_input_active = False
        self._pending_selected_rows: list[int] = []
        self._x_error_config = _AxisErrorConfig()
        self._y_error_config = _AxisErrorConfig()
        self._x_error_dialog = _AxisErrorDialog("X", self)
        self._y_error_dialog = _AxisErrorDialog("Y", self)
        self._vizrank_dialog = _VizRankDialog(self)

        root = QHBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(12)

        sidebar_host = QWidget(self)
        sidebar_host.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.MinimumExpanding)
        sidebar = QVBoxLayout(sidebar_host)
        sidebar.setContentsMargins(0, 0, 0, 0)
        sidebar.setSpacing(10)

        self._sidebar_scroll = QScrollArea(self)
        self._sidebar_scroll.setWidgetResizable(True)
        self._sidebar_scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        self._sidebar_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._sidebar_scroll.setWidget(sidebar_host)
        self._sidebar_scroll.setFixedWidth(308)
        self._sidebar_scroll.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Expanding)
        root.addWidget(self._sidebar_scroll, 0)

        info_box = QGroupBox(i18n.t("Info"))
        info_layout = QVBoxLayout(info_box)
        self._dataset_label = QLabel(i18n.t("Dataset: none"))
        self._dataset_label.setProperty("sectionTitle", True)
        self._status_label = QLabel(i18n.t("Load data and choose X/Y variables."))
        self._status_label.setWordWrap(True)
        info_layout.addWidget(self._dataset_label)
        info_layout.addWidget(self._status_label)
        sidebar.addWidget(info_box)

        self._axes_box = QGroupBox(i18n.t("Axes"))
        axes_layout = QVBoxLayout(self._axes_box)
        axes_layout.setSpacing(8)

        self._x_combo = QComboBox()
        self._y_combo = QComboBox()
        self._x_error_button = QPushButton("...")
        self._y_error_button = QPushButton("...")
        self._x_error_button.clicked.connect(self._configure_x_errors)
        self._y_error_button.clicked.connect(self._configure_y_errors)
        self._x_error_button.setToolTip(i18n.t("Configure X-axis error bars"))
        self._y_error_button.setToolTip(i18n.t("Configure Y-axis error bars"))

        for label, combo, button in (
            (i18n.t("Axis x"), self._x_combo, self._x_error_button),
            (i18n.t("Axis y"), self._y_combo, self._y_error_button),
        ):
            axes_layout.addWidget(QLabel(label))
            row = QHBoxLayout()
            row.setSpacing(6)
            row.addWidget(combo, 1)
            row.addWidget(button, 0)
            axes_layout.addLayout(row)

        self._suggest_button = QPushButton(i18n.t("Find Informative Projections"))
        self._suggest_button.clicked.connect(self._open_vizrank)
        axes_layout.addWidget(self._suggest_button)
        sidebar.addWidget(self._axes_box)

        self._points_box = QGroupBox(i18n.t("Attributes"))
        points_layout = QVBoxLayout(self._points_box)
        points_layout.setSpacing(8)

        self._color_combo = QComboBox()
        self._color_combo.addItem(i18n.t("None"))
        self._label_combo = QComboBox()
        self._label_combo.addItem(i18n.t("None"))
        self._shape_combo = QComboBox()
        self._shape_combo.addItem(i18n.t("None"))
        self._size_combo = QComboBox()
        self._size_combo.addItem(i18n.t("None"))
        for label, combo in (
            (i18n.t("Color"), self._color_combo),
            (i18n.t("Shape"), self._shape_combo),
            (i18n.t("Size"), self._size_combo),
            (i18n.t("Label"), self._label_combo),
        ):
            points_layout.addWidget(QLabel(label))
            points_layout.addWidget(combo)
        self._label_only_selected_cb = QCheckBox(i18n.t("Label only selection and subset"))
        points_layout.addWidget(self._label_only_selected_cb)
        sidebar.addWidget(self._points_box)

        self._effects_box = QGroupBox(i18n.t("Effects"))
        effects_layout = QVBoxLayout(self._effects_box)
        effects_layout.setSpacing(8)

        self._point_size_slider = QSlider(Qt.Orientation.Horizontal)
        self._point_size_slider.setRange(1, 20)
        self._point_size_slider.setValue(10)
        self._alpha_slider = QSlider(Qt.Orientation.Horizontal)
        self._alpha_slider.setRange(0, 255)
        self._alpha_slider.setValue(180)
        self._jitter_slider = QSlider(Qt.Orientation.Horizontal)
        self._jitter_slider.setRange(0, 20)
        self._jitter_slider.setValue(0)
        self._jitter_numeric_cb = QCheckBox(i18n.t("Jitter numeric values"))
        self._jitter_numeric_cb.setChecked(False)
        for label, slider in (
            (i18n.t("Symbol size"), self._point_size_slider),
            (i18n.t("Opacity"), self._alpha_slider),
            (i18n.t("Jittering"), self._jitter_slider),
        ):
            effects_layout.addWidget(QLabel(label))
            effects_layout.addWidget(slider)
        effects_layout.addWidget(self._jitter_numeric_cb)
        sidebar.addWidget(self._effects_box)

        self._plot_box = QGroupBox(i18n.t("Plot"))
        plot_layout = QVBoxLayout(self._plot_box)
        plot_layout.setSpacing(8)

        self._aggregate_cb = QCheckBox(i18n.t("Aggregate points in dense regions"))
        self._aggregate_cb.setChecked(True)
        self._class_density_cb = QCheckBox(i18n.t("Show color regions"))
        self._legend_cb = QCheckBox(i18n.t("Show legend"))
        self._legend_cb.setChecked(True)
        self._show_grid_cb = QCheckBox(i18n.t("Show gridlines"))
        self._show_grid_cb.setChecked(True)
        self._tooltip_all_cb = QCheckBox(i18n.t("Show all data on mouse hover"))
        self._tooltip_all_cb.setChecked(True)
        self._regression_cb = QCheckBox(i18n.t("Show regression line"))
        self._regression_cb.setChecked(False)
        self._orthonormal_cb = QCheckBox(i18n.t("Treat variables as independent"))
        self._ellipse_cb = QCheckBox(i18n.t("Show confidence ellipse"))
        self._subset_cb = QCheckBox(i18n.t("Highlight subset input"))
        self._subset_cb.setChecked(True)
        self._clear_button = QPushButton(i18n.t("Clear Selection"))
        self._clear_button.clicked.connect(self._clear_selection)

        for widget in (
            self._aggregate_cb,
            self._class_density_cb,
            self._legend_cb,
            self._show_grid_cb,
            self._tooltip_all_cb,
            self._regression_cb,
            self._orthonormal_cb,
            self._ellipse_cb,
            self._subset_cb,
            self._clear_button,
        ):
            plot_layout.addWidget(widget)
        sidebar.addWidget(self._plot_box)

        self._tools_box = QGroupBox(i18n.t("Zoom/Select"))
        tools_layout = QVBoxLayout(self._tools_box)
        tools_layout.setSpacing(6)
        self._mode_group = QButtonGroup(self)
        self._select_mode_rb = QRadioButton(i18n.t("Select"))
        self._pan_mode_rb = QRadioButton(i18n.t("Pan"))
        self._zoom_mode_rb = QRadioButton(i18n.t("Zoom"))
        self._select_mode_rb.setChecked(True)
        for index, button in enumerate((self._select_mode_rb, self._pan_mode_rb, self._zoom_mode_rb)):
            self._mode_group.addButton(button, index)
            tools_layout.addWidget(button)
        zoom_button_row = QHBoxLayout()
        zoom_button_row.setSpacing(6)
        self._zoom_in_button = QPushButton(i18n.t("Zoom In"))
        self._zoom_out_button = QPushButton(i18n.t("Zoom Out"))
        zoom_button_row.addWidget(self._zoom_in_button)
        zoom_button_row.addWidget(self._zoom_out_button)
        tools_layout.addLayout(zoom_button_row)
        self._reset_zoom_button = QPushButton(i18n.t("Reset Zoom"))
        tools_layout.addWidget(self._reset_zoom_button)
        sidebar.addWidget(self._tools_box)

        self._selection_label = QLabel(i18n.t("Selected: 0"))
        sidebar.addWidget(self._selection_label)
        sidebar.addStretch(1)

        self._canvas = _ScatterCanvas(self)
        self._canvas.selectionChanged.connect(self._handle_selection_changed)
        root.addWidget(self._canvas, 1)

        for combo in (self._x_combo, self._y_combo):
            combo.currentTextChanged.connect(self._handle_axes_changed)
        self._color_combo.currentTextChanged.connect(self._handle_color_changed)
        self._label_combo.currentTextChanged.connect(self._refresh_plot)
        self._shape_combo.currentTextChanged.connect(self._refresh_plot)
        self._size_combo.currentTextChanged.connect(self._refresh_plot)
        self._label_only_selected_cb.toggled.connect(self._refresh_plot)
        self._point_size_slider.valueChanged.connect(self._refresh_plot)
        self._alpha_slider.valueChanged.connect(self._refresh_plot)
        self._jitter_slider.valueChanged.connect(self._refresh_plot)
        self._jitter_numeric_cb.toggled.connect(self._refresh_plot)
        self._aggregate_cb.toggled.connect(self._refresh_plot)
        self._class_density_cb.toggled.connect(self._refresh_plot)
        self._legend_cb.toggled.connect(self._refresh_plot)
        self._show_grid_cb.toggled.connect(self._refresh_plot)
        self._tooltip_all_cb.toggled.connect(self._refresh_plot)
        self._regression_cb.toggled.connect(self._refresh_plot)
        self._orthonormal_cb.toggled.connect(self._refresh_plot)
        self._ellipse_cb.toggled.connect(self._refresh_plot)
        self._subset_cb.toggled.connect(self._refresh_plot)
        self._select_mode_rb.toggled.connect(lambda checked: checked and self._canvas.set_mode("select"))
        self._pan_mode_rb.toggled.connect(lambda checked: checked and self._canvas.set_mode("pan"))
        self._zoom_mode_rb.toggled.connect(lambda checked: checked and self._canvas.set_mode("zoom"))
        self._zoom_in_button.clicked.connect(self._canvas.zoom_in)
        self._zoom_out_button.clicked.connect(self._canvas.zoom_out)
        self._reset_zoom_button.clicked.connect(self._canvas.reset_view)
        self._canvas.set_mode("select")

    def sizeHint(self) -> QSize:
        return QSize(1132, 708)

    def set_input_payload(self, payload) -> None:
        if payload is None:
            self._dataset = None
            self._subset = None
            self._requested_features = ()
            self._feature_input_active = False
        elif payload.port_label == "Data":
            self._dataset = payload.dataset
        elif payload.port_label == "Data Subset":
            self._subset = payload.dataset
        elif payload.port_label == "Features":
            names = feature_names_from_payload(payload)
            self._requested_features = names[:2]
            self._feature_input_active = len(self._requested_features) >= 2
        self._sync_controls()
        self._refresh_plot()

    def current_output_dataset(self) -> DatasetHandle | None:
        return self._selected_dataset

    def current_output_datasets(self) -> dict[str, DatasetHandle | None] | None:
        return {
            "Selected Data": self._selected_dataset,
            "Annotated Data": self._annotated_dataset,
        }

    def current_output_payloads(self):
        return {
            "Selected Data": None
            if self._selected_dataset is None
            else WorkflowPayload("Selected Data", self._selected_dataset),
            "Annotated Data": (
                None
                if self._annotated_dataset is None
                else WorkflowPayload("Annotated Data", self._annotated_dataset)
            ),
            "Features": feature_output_payload(
                [self._x_combo.currentText(), self._y_combo.currentText()],
                port_label="Features",
            ),
        }

    def serialize_node_state(self) -> dict[str, object]:
        return {
            "x": self._x_combo.currentText(),
            "y": self._y_combo.currentText(),
            "color": self._color_combo.currentText(),
            "label": self._label_combo.currentText(),
            "size": self._size_combo.currentText(),
            "shape": self._shape_combo.currentText(),
            "label_only_selected": self._label_only_selected_cb.isChecked(),
            "point_size": self._point_size_slider.value(),
            "alpha_value": self._alpha_slider.value(),
            "x_error_upper": self._x_error_config.upper,
            "x_error_lower": self._x_error_config.lower,
            "x_error_absolute": self._x_error_config.absolute,
            "y_error_upper": self._y_error_config.upper,
            "y_error_lower": self._y_error_config.lower,
            "y_error_absolute": self._y_error_config.absolute,
            "jitter": self._jitter_slider.value(),
            "jitter_continuous": self._jitter_numeric_cb.isChecked(),
            "aggregate_points": self._aggregate_cb.isChecked(),
            "class_density": self._class_density_cb.isChecked(),
            "show_legend": self._legend_cb.isChecked(),
            "show_grid": self._show_grid_cb.isChecked(),
            "tooltip_shows_all": self._tooltip_all_cb.isChecked(),
            "show_regression": self._regression_cb.isChecked(),
            "orthonormal_regression": self._orthonormal_cb.isChecked(),
            "show_ellipse": self._ellipse_cb.isChecked(),
            "highlight_subset": self._subset_cb.isChecked(),
            "selected_rows": list(self._pending_selected_rows),
        }

    def restore_node_state(self, payload: dict[str, object]) -> None:
        self._pending_selected_rows = [
            int(index)
            for index in payload.get("selected_rows", [])
            if isinstance(index, int | float)
        ]
        self._set_combo_value(self._x_combo, str(payload.get("x", "")))
        self._set_combo_value(self._y_combo, str(payload.get("y", "")))
        self._set_combo_value(self._color_combo, str(payload.get("color", i18n.t("None"))))
        self._set_combo_value(self._label_combo, str(payload.get("label", i18n.t("None"))))
        self._set_combo_value(self._size_combo, str(payload.get("size", i18n.t("None"))))
        self._set_combo_value(self._shape_combo, str(payload.get("shape", i18n.t("None"))))
        self._label_only_selected_cb.setChecked(bool(payload.get("label_only_selected", False)))
        self._point_size_slider.setValue(int(payload.get("point_size", 10)))
        self._alpha_slider.setValue(int(payload.get("alpha_value", 180)))
        self._x_error_config = _AxisErrorConfig(
            upper=str(payload.get("x_error_upper", "")),
            lower=str(payload.get("x_error_lower", "")),
            absolute=bool(payload.get("x_error_absolute", False)),
        )
        self._y_error_config = _AxisErrorConfig(
            upper=str(payload.get("y_error_upper", "")),
            lower=str(payload.get("y_error_lower", "")),
            absolute=bool(payload.get("y_error_absolute", False)),
        )
        self._jitter_slider.setValue(int(payload.get("jitter", 0)))
        self._jitter_numeric_cb.setChecked(bool(payload.get("jitter_continuous", False)))
        self._aggregate_cb.setChecked(bool(payload.get("aggregate_points", True)))
        self._class_density_cb.setChecked(bool(payload.get("class_density", False)))
        self._legend_cb.setChecked(bool(payload.get("show_legend", True)))
        self._show_grid_cb.setChecked(bool(payload.get("show_grid", True)))
        self._tooltip_all_cb.setChecked(bool(payload.get("tooltip_shows_all", True)))
        self._regression_cb.setChecked(bool(payload.get("show_regression", False)))
        self._orthonormal_cb.setChecked(bool(payload.get("orthonormal_regression", False)))
        self._ellipse_cb.setChecked(bool(payload.get("show_ellipse", False)))
        self._subset_cb.setChecked(bool(payload.get("highlight_subset", True)))
        self._update_error_button_labels()

    def documentation_url(self) -> str:
        return "https://orangedatamining.com/widget-catalog/visualize/scatterplot/"

    def _sync_controls(self) -> None:
        columns = primitive_columns(self._dataset)
        numeric = [
            column.name
            for column in self._dataset.domain.columns
            if column.logical_type == "numeric"
        ] if self._dataset is not None else []
        discrete = discrete_columns(self._dataset)
        target_name = ""
        if self._dataset is not None:
            for column in self._dataset.domain.target_columns:
                if column.name in self._dataset.dataframe.columns:
                    target_name = column.name
                    break
            if not target_name:
                target_name = self._default_region_column_name()
        current_x = self._x_combo.currentText()
        current_y = self._y_combo.currentText()
        current_color = self._color_combo.currentText()
        current_label = self._label_combo.currentText()
        current_size = self._size_combo.currentText()
        current_shape = self._shape_combo.currentText()
        none_text = i18n.t("None")

        self._x_combo.blockSignals(True)
        self._y_combo.blockSignals(True)
        self._color_combo.blockSignals(True)
        self._label_combo.blockSignals(True)
        self._size_combo.blockSignals(True)
        self._shape_combo.blockSignals(True)
        self._x_combo.clear()
        self._y_combo.clear()
        self._color_combo.clear()
        self._label_combo.clear()
        self._size_combo.clear()
        self._shape_combo.clear()
        self._color_combo.addItem(i18n.t("None"))
        self._label_combo.addItem(i18n.t("None"))
        self._size_combo.addItem(i18n.t("None"))
        self._shape_combo.addItem(i18n.t("None"))
        self._x_combo.addItems(columns)
        self._y_combo.addItems(columns)
        self._color_combo.addItems(columns)
        self._label_combo.addItems(columns)
        self._size_combo.addItems(numeric)
        self._shape_combo.addItems(discrete)
        self._x_combo.blockSignals(False)
        self._y_combo.blockSignals(False)
        self._color_combo.blockSignals(False)
        self._label_combo.blockSignals(False)
        self._size_combo.blockSignals(False)
        self._shape_combo.blockSignals(False)

        if columns:
            if self._feature_input_active:
                requested_x = self._requested_features[0] if len(self._requested_features) > 0 else ""
                requested_y = self._requested_features[1] if len(self._requested_features) > 1 else ""
                if self._x_combo.findText(requested_x) >= 0 and self._y_combo.findText(requested_y) >= 0:
                    self._set_combo_value(self._x_combo, requested_x)
                    self._set_combo_value(self._y_combo, requested_y)
                else:
                    self._x_combo.setCurrentIndex(-1)
                    self._y_combo.setCurrentIndex(-1)
            else:
                self._set_combo_value(self._x_combo, current_x or columns[0])
                self._set_combo_value(self._y_combo, current_y or columns[min(1, len(columns) - 1)])
        color_value = current_color if current_color and current_color != none_text else (target_name or none_text)
        label_value = current_label if current_label and current_label != none_text else none_text
        size_value = current_size if current_size and current_size != none_text else none_text
        shape_value = current_shape if current_shape and current_shape != none_text else none_text
        self._set_combo_value(self._color_combo, color_value)
        self._set_combo_value(self._label_combo, label_value)
        self._set_combo_value(self._size_combo, size_value)
        self._set_combo_value(self._shape_combo, shape_value)
        self._update_error_button_labels()
        self._axes_box.setEnabled(not self._feature_input_active)
        self._x_error_button.setEnabled(bool(numeric) and not self._feature_input_active)
        self._y_error_button.setEnabled(bool(numeric) and not self._feature_input_active)
        self._update_vizrank_state()
        self._update_regression_controls()
        self._update_density_controls()

    def _refresh_plot(self) -> None:
        self._selected_dataset = None
        self._annotated_dataset = None
        dataset = self._dataset
        self._update_vizrank_state()
        self._update_regression_controls()
        if dataset is None:
            self._dataset_label.setText(i18n.t("Dataset: none"))
            self._status_label.setText(i18n.t("Load data and choose X/Y variables."))
            self._canvas.set_state(
                points=[],
                x_col=None,
                y_col=None,
                legend_items=[],
                show_regression=self._regression_cb.isChecked(),
                show_ellipse=self._ellipse_cb.isChecked(),
                show_subset=self._subset_cb.isChecked(),
                show_grid=self._show_grid_cb.isChecked(),
                show_legend=self._legend_cb.isChecked(),
                show_density=self._class_density_cb.isChecked(),
                aggregate_points=self._aggregate_cb.isChecked(),
                label_only_selected=self._label_only_selected_cb.isChecked(),
                point_size=self._point_size_slider.value(),
                alpha_value=self._alpha_slider.value(),
                jitter_continuous=self._jitter_numeric_cb.isChecked(),
                orthonormal_regression=self._orthonormal_cb.isChecked(),
                jitter=self._jitter_slider.value(),
            )
            self._handle_selection_changed([])
            return

        self._dataset_label.setText(i18n.tf("Dataset: {name}", name=dataset.display_name))
        x_name = self._x_combo.currentText()
        y_name = self._y_combo.currentText()
        color_name = self._color_combo.currentText()
        label_name = self._label_combo.currentText()
        size_name = self._size_combo.currentText()
        shape_name = self._shape_combo.currentText()

        x_col = prepared_column(dataset, x_name)
        y_col = prepared_column(dataset, y_name)
        if x_col is None or y_col is None:
            if self._feature_input_active:
                self._status_label.setText(i18n.t("Incoming Features do not match available data columns."))
            else:
                self._status_label.setText(i18n.t("Choose valid X/Y variables."))
            self._canvas.set_state(
                points=[],
                x_col=None,
                y_col=None,
                legend_items=[],
                show_regression=self._regression_cb.isChecked(),
                show_ellipse=self._ellipse_cb.isChecked(),
                show_subset=self._subset_cb.isChecked(),
                show_grid=self._show_grid_cb.isChecked(),
                show_legend=self._legend_cb.isChecked(),
                show_density=self._class_density_cb.isChecked(),
                aggregate_points=self._aggregate_cb.isChecked(),
                label_only_selected=self._label_only_selected_cb.isChecked(),
                point_size=self._point_size_slider.value(),
                alpha_value=self._alpha_slider.value(),
                jitter_continuous=self._jitter_numeric_cb.isChecked(),
                orthonormal_regression=self._orthonormal_cb.isChecked(),
                jitter=self._jitter_slider.value(),
            )
            self._handle_selection_changed([])
            return

        x_lookup = {row: (value, raw) for row, value, raw in zip(x_col.row_indices, x_col.values, x_col.raw_values)}
        y_lookup = {row: (value, raw) for row, value, raw in zip(y_col.row_indices, y_col.values, y_col.raw_values)}
        subset_rows = subset_row_indices(dataset, self._subset)
        color_values = self._color_lookup(color_name)
        label_values = self._label_lookup(label_name)
        size_values = self._size_lookup(size_name)
        shape_values = self._shape_lookup(shape_name)
        x_error_values = self._axis_error_lookup(self._x_error_config, x_name)
        y_error_values = self._axis_error_lookup(self._y_error_config, y_name)
        points: list[_ScatterPoint] = []
        legend_map: dict[str, QColor] = {}
        shared_rows = sorted(set(x_lookup).intersection(y_lookup))
        if not shared_rows:
            self._status_label.setText(
                i18n.tf(
                    "Plot cannot be displayed because '{x}' or '{y}' is missing for all data points.",
                    x=x_name,
                    y=y_name,
                )
            )
            self._canvas.set_state(
                points=[],
                x_col=x_col,
                y_col=y_col,
                legend_items=[],
                show_regression=self._regression_cb.isChecked(),
                show_ellipse=self._ellipse_cb.isChecked(),
                show_subset=self._subset_cb.isChecked(),
                show_grid=self._show_grid_cb.isChecked(),
                show_legend=self._legend_cb.isChecked(),
                show_density=self._class_density_cb.isChecked(),
                aggregate_points=self._aggregate_cb.isChecked(),
                label_only_selected=self._label_only_selected_cb.isChecked(),
                point_size=self._point_size_slider.value(),
                alpha_value=self._alpha_slider.value(),
                jitter_continuous=self._jitter_numeric_cb.isChecked(),
                orthonormal_regression=self._orthonormal_cb.isChecked(),
                jitter=self._jitter_slider.value(),
            )
            self._handle_selection_changed([])
            return

        for row in shared_rows:
            x_value, raw_x = x_lookup[row]
            y_value, raw_y = y_lookup[row]
            legend, color = color_values.get(row, (i18n.t("None"), QColor(PALETTE[0])))
            if legend not in legend_map:
                legend_map[legend] = color
            x_error_left, x_error_right = x_error_values.get(row, (0.0, 0.0))
            y_error_bottom, y_error_top = y_error_values.get(row, (0.0, 0.0))
            points.append(
                _ScatterPoint(
                    row_index=row,
                    x=float(x_value),
                    y=float(y_value),
                    raw_x=raw_x,
                    raw_y=raw_y,
                    color=color,
                    legend=legend,
                    subset=row in subset_rows,
                    label=label_values.get(row, ""),
                    radius=size_values.get(row, 4.0),
                    shape=shape_values.get(row, "circle"),
                    x_error_left=x_error_left,
                    x_error_right=x_error_right,
                    y_error_bottom=y_error_bottom,
                    y_error_top=y_error_top,
                    tooltip=self._tooltip_text(row, x_name, y_name, color_name),
                )
            )

        status = i18n.tf("Points: {count}", count=len(points))
        missing_count = dataset.row_count - len(points)
        if missing_count > 0:
            status += i18n.tf(" | Missing coordinates: {count}", count=missing_count)
        if label_name and label_name != i18n.t("None"):
            status += i18n.tf(" | Label: {name}", name=label_name)
        if size_name and size_name != i18n.t("None"):
            status += i18n.tf(" | Size: {name}", name=size_name)
        if shape_name and shape_name != i18n.t("None"):
            status += i18n.tf(" | Shape: {name}", name=shape_name)
        if self._x_error_config.upper or self._x_error_config.lower:
            status += i18n.tf(" | X error: {name}", name=self._x_error_config.summary())
        if self._y_error_config.upper or self._y_error_config.lower:
            status += i18n.tf(" | Y error: {name}", name=self._y_error_config.summary())
        if self._orthonormal_cb.isChecked():
            status += i18n.t(" | orthonormal regression")
        if self._aggregate_cb.isChecked():
            status += i18n.t(" | aggregation")
        if self._class_density_cb.isChecked():
            status += i18n.t(" | density")
        self._status_label.setText(status)
        self._canvas.set_state(
            points=points,
            x_col=x_col,
            y_col=y_col,
            legend_items=list(legend_map.items()),
            show_regression=self._regression_cb.isChecked(),
            show_ellipse=self._ellipse_cb.isChecked(),
            show_subset=self._subset_cb.isChecked(),
            show_grid=self._show_grid_cb.isChecked(),
            show_legend=self._legend_cb.isChecked(),
            show_density=self._class_density_cb.isChecked(),
            aggregate_points=self._aggregate_cb.isChecked(),
            label_only_selected=self._label_only_selected_cb.isChecked(),
            point_size=self._point_size_slider.value(),
            alpha_value=self._alpha_slider.value(),
            jitter_continuous=self._jitter_numeric_cb.isChecked(),
            orthonormal_regression=self._orthonormal_cb.isChecked(),
            jitter=self._jitter_slider.value(),
        )
        self._handle_selection_changed(self._pending_selected_rows)

    def _color_lookup(self, color_name: str) -> dict[int, tuple[str, QColor]]:
        dataset = self._dataset
        if dataset is None:
            return {}
        if not color_name or color_name == i18n.t("None"):
            return {}

        column = prepared_column(dataset, color_name)
        if column is None:
            return {}
        if column.is_discrete:
            lookup: dict[int, tuple[str, QColor]] = {}
            for row, value, raw in zip(column.row_indices, column.values, column.raw_values):
                label = str(raw)
                lookup[int(row)] = (label, QColor(PALETTE[int(value) % len(PALETTE)]))
            return lookup

        low = float(np.min(column.values)) if len(column.values) else 0.0
        high = float(np.max(column.values)) if len(column.values) else 1.0
        return {
            int(row): (f"{float(value):.3g}", gradient_color(float(value), low, high))
            for row, value in zip(column.row_indices, column.values)
        }

    def _label_lookup(self, label_name: str) -> dict[int, str]:
        dataset = self._dataset
        if dataset is None or not label_name or label_name == i18n.t("None") or label_name not in dataset.dataframe.columns:
            return {}
        series = dataset.dataframe.get_column(label_name).to_list()
        return {
            index: str(value)
            for index, value in enumerate(series)
            if value is not None
        }

    def _size_lookup(self, size_name: str) -> dict[int, float]:
        dataset = self._dataset
        if dataset is None or not size_name or size_name == i18n.t("None"):
            return {}
        column = prepared_column(dataset, size_name)
        if column is None or column.is_discrete or len(column.values) == 0:
            return {}
        low = float(np.min(column.values))
        high = float(np.max(column.values))
        span = high - low or 1.0
        return {
            int(row): 3.5 + 4.5 * ((float(value) - low) / span)
            for row, value in zip(column.row_indices, column.values)
        }

    def _shape_lookup(self, shape_name: str) -> dict[int, str]:
        dataset = self._dataset
        if dataset is None or not shape_name or shape_name == i18n.t("None"):
            return {}
        column = prepared_column(dataset, shape_name)
        if column is None or not column.is_discrete:
            return {}
        shapes = ("circle", "square", "triangle", "diamond")
        return {
            int(row): shapes[int(value) % len(shapes)]
            for row, value in zip(column.row_indices, column.values)
        }

    def _axis_error_lookup(self, config: _AxisErrorConfig, center_name: str) -> dict[int, tuple[float, float]]:
        dataset = self._dataset
        if dataset is None or not center_name or (not config.lower and not config.upper):
            return {}
        center_column = prepared_column(dataset, center_name)
        if center_column is None or center_column.is_discrete:
            return {}
        center_lookup = {
            int(row): float(value)
            for row, value in zip(center_column.row_indices, center_column.values)
            if math.isfinite(float(value))
        }

        def build_lookup(name: str, *, upper: bool) -> dict[int, float]:
            if not name or name == i18n.t("None"):
                return {}
            column = prepared_column(dataset, name)
            if column is None or column.is_discrete:
                return {}
            resolved: dict[int, float] = {}
            for row, value in zip(column.row_indices, column.values):
                row_index = int(row)
                if row_index not in center_lookup or not math.isfinite(float(value)):
                    continue
                numeric = float(value)
                center_value = center_lookup[row_index]
                if config.absolute:
                    delta = numeric - center_value if upper else center_value - numeric
                else:
                    delta = numeric
                resolved[row_index] = max(0.0, float(delta))
            return resolved

        lower_lookup = build_lookup(config.lower, upper=False)
        upper_lookup = build_lookup(config.upper, upper=True)
        rows = set(lower_lookup).union(upper_lookup)
        return {
            row: (lower_lookup.get(row, 0.0), upper_lookup.get(row, 0.0))
            for row in rows
        }

    def _suggest_axes(self) -> None:
        dataset = self._dataset
        if dataset is None:
            return
        color_name = self._color_combo.currentText()
        if color_name == i18n.t("None"):
            color_name = None
        pair = suggest_scatter_pair(dataset, class_name=color_name)
        if pair is None:
            self._status_label.setText(i18n.t("No suitable numeric pair found."))
            return
        self._set_combo_value(self._x_combo, pair[0])
        self._set_combo_value(self._y_combo, pair[1])

    def _open_vizrank(self) -> None:
        dataset = self._dataset
        if dataset is None or not self._suggest_button.isEnabled():
            return
        color_name = self._current_color_name()
        ranked_pairs = rank_scatter_pairs(dataset, class_name=color_name, limit=40)
        if not ranked_pairs:
            self._status_label.setText(i18n.t("No suitable numeric pair found."))
            return
        self._vizrank_dialog.set_ranked_pairs(
            ranked_pairs,
            current_pair=(self._x_combo.currentText(), self._y_combo.currentText()),
        )
        if self._vizrank_dialog.exec() == QDialog.DialogCode.Accepted:
            pair = self._vizrank_dialog.selected_pair()
            if pair is not None:
                self._set_combo_value(self._x_combo, pair[0])
                self._set_combo_value(self._y_combo, pair[1])

    def _configure_x_errors(self) -> None:
        self._configure_axis_errors(self._x_error_dialog, self._x_error_config, axis="x")

    def _configure_y_errors(self) -> None:
        self._configure_axis_errors(self._y_error_dialog, self._y_error_config, axis="y")

    def _configure_axis_errors(
        self,
        dialog: _AxisErrorDialog,
        current: _AxisErrorConfig,
        *,
        axis: str,
    ) -> None:
        numeric = [
            column.name
            for column in self._dataset.domain.columns
            if column.logical_type == "numeric"
        ] if self._dataset is not None else []
        dialog.set_options(numeric, current)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            config = dialog.selected_config()
            if axis == "x":
                self._x_error_config = config
            else:
                self._y_error_config = config
            self._update_error_button_labels()
            self._refresh_plot()

    def _update_error_button_labels(self) -> None:
        self._x_error_button.setText("..." if not (self._x_error_config.upper or self._x_error_config.lower) else "E")
        self._y_error_button.setText("..." if not (self._y_error_config.upper or self._y_error_config.lower) else "E")
        self._x_error_button.setToolTip(i18n.tf("X error bars: {value}", value=self._x_error_config.summary()))
        self._y_error_button.setToolTip(i18n.tf("Y error bars: {value}", value=self._y_error_config.summary()))

    def _handle_axes_changed(self) -> None:
        self._update_regression_controls()
        self._update_density_controls()
        self._refresh_plot()

    def _handle_color_changed(self) -> None:
        self._update_vizrank_state()
        self._update_density_controls()
        self._refresh_plot()

    def _current_color_name(self) -> str | None:
        color_name = self._color_combo.currentText()
        if not color_name or color_name == i18n.t("None"):
            return None
        return color_name

    def _update_vizrank_state(self) -> None:
        tooltip = ""
        enabled = True
        dataset = self._dataset
        if dataset is None:
            enabled = False
            tooltip = i18n.t("No data on input")
        elif self._feature_input_active:
            enabled = False
            tooltip = i18n.t("Features input is controlling axes")
        else:
            numeric = [
                column.name
                for column in dataset.domain.columns
                if column.logical_type == "numeric"
            ]
            if len(numeric) < 3:
                enabled = False
                tooltip = i18n.t("Not enough features for ranking")
            else:
                color_name = self._current_color_name()
                if color_name is None:
                    enabled = False
                    tooltip = i18n.t("Color variable is not selected")
                else:
                    color_column = prepared_column(dataset, color_name)
                    if color_column is None or len(color_column.row_indices) == 0:
                        enabled = False
                        tooltip = i18n.t("Color variable has no values")
        self._suggest_button.setEnabled(enabled)
        self._suggest_button.setToolTip(tooltip)

    def _update_regression_controls(self) -> None:
        can_draw = self._can_draw_regression_line()
        self._regression_cb.setEnabled(can_draw)
        self._orthonormal_cb.setEnabled(can_draw and self._regression_cb.isChecked())
        self._ellipse_cb.setEnabled(can_draw)

    def _update_density_controls(self) -> None:
        dataset = self._dataset
        enabled = False
        if dataset is not None:
            color_name = self._current_color_name()
            region_name = self._default_region_column_name()
            color_col = prepared_column(dataset, color_name) if color_name is not None else None
            x_col = prepared_column(dataset, self._x_combo.currentText())
            y_col = prepared_column(dataset, self._y_combo.currentText())
            enabled = bool(
                color_col
                and color_col.is_discrete
                and color_name == region_name
                and x_col
                and y_col
                and not x_col.is_discrete
                and not y_col.is_discrete
            )
        self._class_density_cb.setEnabled(enabled)
        if not enabled:
            self._class_density_cb.setChecked(False)

    def _default_region_column_name(self) -> str:
        dataset = self._dataset
        if dataset is None:
            return ""
        lower_to_name = {name.lower(): name for name in dataset.dataframe.columns}
        for candidate in ("cluster", "class"):
            name = lower_to_name.get(candidate)
            if name:
                return name
        return ""

    def _can_draw_regression_line(self) -> bool:
        dataset = self._dataset
        if dataset is None:
            return False
        x_col = prepared_column(dataset, self._x_combo.currentText())
        y_col = prepared_column(dataset, self._y_combo.currentText())
        return bool(x_col and y_col and not x_col.is_discrete and not y_col.is_discrete)

    def _tooltip_text(self, row_index: int, x_name: str, y_name: str, color_name: str) -> str:
        dataset = self._dataset
        if dataset is None or row_index < 0 or row_index >= dataset.row_count:
            return ""
        row = dataset.dataframe.row(row_index, named=True)
        entries = [f"Row: {row_index}", f"{x_name}: {row.get(x_name)}", f"{y_name}: {row.get(y_name)}"]
        if not self._tooltip_all_cb.isChecked():
            if color_name and color_name != i18n.t("None") and color_name not in {x_name, y_name}:
                entries.append(f"{color_name}: {row.get(color_name)}")
            return "\n".join(entries)
        for name, value in row.items():
            if name in {x_name, y_name}:
                continue
            entries.append(f"{name}: {value}")
        return "\n".join(entries)

    def _handle_selection_changed(self, rows: list[int]) -> None:
        self._pending_selected_rows = sorted({index for index in rows})
        self._canvas.set_selected_rows(self._pending_selected_rows)
        self._selection_label.setText(i18n.tf("Selected: {count}", count=len(self._pending_selected_rows)))
        self._selected_dataset, self._annotated_dataset = build_selection_outputs(
            self._dataset,
            self._pending_selected_rows,
            generated_by="scatter-plot",
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
