from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import beta, expon, gamma, norm, pareto, rayleigh
from PySide6.QtCore import QRect, QSize, Qt, Signal
from PySide6.QtGui import QColor, QPainter, QPen
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QToolTip,
    QVBoxLayout,
    QWidget,
)

from portakal_app.data.models import DatasetHandle
from portakal_app.data.services.generated_dataset_service import GeneratedDatasetService
from portakal_app.ui import i18n
from portakal_app.ui.screens.node_screen import WorkflowNodeScreenSupport
from portakal_app.ui.screens.visualize_common import (
    PlotColumn,
    build_selection_outputs,
    discrete_columns,
    nice_ticks,
    prepared_column,
    primitive_columns,
)
from portakal_app.ui.shared.type_icons import type_badge_icon

ORANGE_DISCRETE_COLORS = (
    (70, 190, 250),
    (237, 70, 47),
    (170, 242, 43),
    (245, 174, 50),
    (255, 255, 0),
    (255, 0, 255),
    (0, 255, 255),
    (128, 0, 255),
)


def _unique_thresholds(unique: np.ndarray) -> np.ndarray:
    if len(unique) >= 2:
        last_boundary = 2 * unique[-1] - unique[-2]
    else:
        last_boundary = unique[0] + 1.0
    return np.hstack((unique, [last_boundary]))


@dataclass(frozen=True)
class _DistributionSegment:
    group_label: str
    count: int
    value: float
    row_indices: tuple[int, ...]
    color: QColor


@dataclass(frozen=True)
class _DistributionBar:
    label: str
    segments: tuple[_DistributionSegment, ...]
    low: float | None = None
    high: float | None = None


@dataclass(frozen=True)
class _DistributionCurve:
    label: str
    points: tuple[tuple[float, float], ...]
    color: QColor
    description: str = ""


@dataclass(frozen=True)
class _BinningDefinition:
    thresholds: np.ndarray
    width_label: str = ""


class _DistributionCanvas(QWidget):
    selectionChanged = Signal(list)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._bars: list[_DistributionBar] = []
        self._curves: list[_DistributionCurve] = []
        self._stacked = True
        self._relative = False
        self._cumulative = False
        self._show_legend = True
        self._hide_bars = False
        self._x_label = ""
        self._y_label = i18n.t("Frequency")
        self._hit_rects: list[tuple[QRect, tuple[int, ...], str]] = []
        self._selected_rows: set[int] = set()
        self.setMinimumHeight(280)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMouseTracking(True)

    def set_state(
        self,
        bars: list[_DistributionBar],
        *,
        stacked: bool,
        relative: bool,
        cumulative: bool = False,
        show_legend: bool = True,
        hide_bars: bool = False,
        curves: list[_DistributionCurve] | None = None,
        x_label: str = "",
        y_label: str = "",
    ) -> None:
        self._bars = bars
        self._curves = list(curves or [])
        self._stacked = stacked
        self._relative = relative
        self._cumulative = cumulative
        self._show_legend = show_legend
        self._hide_bars = hide_bars
        self._x_label = x_label
        self._y_label = y_label
        self._hit_rects = []
        self.update()

    def set_selected_rows(self, rows: list[int]) -> None:
        self._selected_rows = set(rows)
        self.update()

    def clear_selection(self) -> None:
        if not self._selected_rows:
            return
        self._selected_rows.clear()
        self.selectionChanged.emit([])
        self.update()

    def mouseMoveEvent(self, event) -> None:
        pos = event.position().toPoint()
        for rect, _, tooltip in self._hit_rects:
            if rect.contains(pos):
                QToolTip.showText(event.globalPosition().toPoint(), tooltip, self)
                return
        QToolTip.hideText()

    def mouseReleaseEvent(self, event) -> None:
        if event.button() != Qt.MouseButton.LeftButton:
            return super().mouseReleaseEvent(event)
        pos = event.position().toPoint()
        for rect, row_indices, _ in self._hit_rects:
            if rect.contains(pos):
                incoming = set(row_indices)
                modifiers = event.modifiers()
                if modifiers & Qt.KeyboardModifier.AltModifier:
                    self._selected_rows.difference_update(incoming)
                elif modifiers & (Qt.KeyboardModifier.ShiftModifier | Qt.KeyboardModifier.ControlModifier):
                    self._selected_rows.update(incoming)
                else:
                    self._selected_rows = set(incoming)
                self.selectionChanged.emit(sorted(self._selected_rows))
                self.update()
                return
        self._selected_rows.clear()
        self.selectionChanged.emit([])
        self.update()

    def leaveEvent(self, _event) -> None:
        QToolTip.hideText()

    def paintEvent(self, _event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), QColor("#ffffff"))
        self._hit_rects = []

        if not self._bars:
            painter.setPen(QColor("#8d877d"))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, i18n.t("Load data and choose a variable."))
            return

        numeric_axis = any(bar.low is not None and bar.high is not None for bar in self._bars)
        margin_left = 88
        legend_items = self._legend_items()
        margin_right = 24
        margin_top = 16
        margin_bottom = 58 if numeric_axis else 86
        chart = QRect(
            margin_left,
            margin_top,
            max(10, self.width() - margin_left - margin_right),
            max(10, self.height() - margin_top - margin_bottom),
        )

        totals = [self._bar_visible_height(bar) for bar in self._bars]
        curve_max = max(
            (point[1] for curve in self._curves for point in curve.points),
            default=0.0,
        )
        max_total = max(max(totals) if totals else 0.0, curve_max)
        max_total = max(max_total, 1.0)

        painter.setPen(QPen(QColor("#222222"), 1.0))
        painter.drawLine(chart.left(), chart.bottom(), chart.right(), chart.bottom())
        painter.drawLine(chart.left(), chart.top(), chart.left(), chart.bottom())

        for tick in nice_ticks(0.0, float(max_total), count=5):
            py = chart.bottom() - int(tick / max_total * chart.height())
            painter.setPen(QPen(QColor("#222222"), 1.0))
            painter.drawLine(chart.left() - 5, py, chart.left(), py)
            painter.drawText(QRect(2, py - 8, margin_left - 8, 16), Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter, f"{tick:.3g}")

        painter.setPen(QColor("#222222"))
        if numeric_axis:
            x_min = min(float(bar.low) for bar in self._bars if bar.low is not None)
            x_max = max(float(bar.high) for bar in self._bars if bar.high is not None)
            x_span = max(x_max - x_min, 1e-9)
            for tick in nice_ticks(x_min, x_max, count=6):
                px = chart.left() + int((tick - x_min) / x_span * chart.width())
                painter.drawLine(px, chart.bottom(), px, chart.bottom() + 5)
                painter.drawText(
                    QRect(px - 30, chart.bottom() + 8, 60, 18),
                    Qt.AlignmentFlag.AlignCenter,
                    f"{tick:.3g}",
                )
        else:
            bar_width = chart.width() / max(1, len(self._bars))
            for index, bar in enumerate(self._bars):
                left = chart.left() + int(index * bar_width) + 6
                right = chart.left() + int((index + 1) * bar_width) - 6
                painter.drawText(
                    QRect(left - 12, chart.bottom() + 6, (right - left) + 24, 48),
                    Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop | Qt.TextFlag.TextWordWrap,
                    bar.label,
                )
        if self._x_label:
            painter.drawText(QRect(chart.left(), self.height() - 24, chart.width(), 18), Qt.AlignmentFlag.AlignCenter, self._x_label)
        if self._y_label:
            painter.save()
            painter.translate(22, chart.center().y())
            painter.rotate(-90)
            painter.drawText(QRect(-chart.height() // 2, -12, chart.height(), 24), Qt.AlignmentFlag.AlignCenter, self._y_label)
            painter.restore()

        bar_width = chart.width() / max(1, len(self._bars))
        for index, bar in enumerate(self._bars):
            if numeric_axis and bar.low is not None and bar.high is not None:
                x_min = min(float(item.low) for item in self._bars if item.low is not None)
                x_max = max(float(item.high) for item in self._bars if item.high is not None)
                x_span = max(x_max - x_min, 1e-9)
                left = chart.left() + int((bar.low - x_min) / x_span * chart.width())
                right = chart.left() + int((bar.high - x_min) / x_span * chart.width())
                left += 1
                right -= 1
            else:
                left = chart.left() + int(index * bar_width) + 6
                right = chart.left() + int((index + 1) * bar_width) - 6
            if not self._hide_bars:
                bar_total = max(1, sum(segment.count for segment in bar.segments))
                if self._stacked:
                    current_top = chart.bottom()
                    for segment in bar.segments:
                        segment_value = segment.count / bar_total if self._relative else float(segment.count)
                        height = int(segment_value / max_total * chart.height())
                        rect = QRect(left, current_top - height, max(8, right - left), max(4, height))
                        current_top -= height
                        self._draw_segment(painter, rect, segment, bar_total)
                else:
                    padding = min(20, max(0, int((right - left) * 0.1)))
                    padded_left = left + padding
                    padded_right = right - padding
                    seg_width = max(8, (padded_right - padded_left) // max(1, len(bar.segments)))
                    for seg_index, segment in enumerate(bar.segments):
                        segment_value = segment.count / bar_total if self._relative else float(segment.count)
                        height = int(segment_value / max_total * chart.height())
                        seg_left = padded_left + seg_index * seg_width
                        rect = QRect(seg_left, chart.bottom() - height, max(6, seg_width - 4), max(4, height))
                        self._draw_segment(painter, rect, segment, bar_total)
        if self._curves:
            self._draw_curves(painter, chart, max_total)
        if self._show_legend and legend_items:
            legend_width = 130
            legend_height = min(max(40, 26 + 24 * len(legend_items)), 180)
            self._draw_legend(
                painter,
                QRect(chart.right() - legend_width - 8, chart.top() + 8, legend_width, legend_height),
                legend_items,
            )

    def _draw_segment(self, painter: QPainter, rect: QRect, segment: _DistributionSegment, bar_total: int) -> None:
        fill = QColor(segment.color)
        border = QColor("#2f5fff") if self._selected_rows.intersection(segment.row_indices) else QColor("#ffffff")
        painter.setPen(QPen(border, 1.0 if not self._stacked else 0.0))
        painter.setBrush(fill)
        painter.drawRect(rect)
        tooltip = "\n".join(
            [
                segment.group_label,
                f"Count: {segment.count}",
                f"Probability: {segment.count / max(bar_total, 1):.1%}" if self._relative else "",
                f"Share: {segment.value:.1%}" if not self._relative else "",
            ]
        ).strip()
        self._hit_rects.append((rect.adjusted(-2, -2, 2, 2), segment.row_indices, tooltip))

    def _draw_curves(self, painter: QPainter, chart: QRect, max_total: float) -> None:
        if max_total <= 0:
            return
        for curve in self._curves:
            if len(curve.points) < 2:
                continue
            polygon: list[tuple[int, int]] = []
            painter.setPen(QPen(curve.color, 2.0))
            last: tuple[int, int] | None = None
            for x_pos, y_value in curve.points:
                px = chart.left() + int(max(0.0, min(1.0, x_pos)) * chart.width())
                py = chart.bottom() - int(max(0.0, min(max_total, y_value)) / max_total * chart.height())
                polygon.append((px, py))
                if last is not None:
                    painter.drawLine(last[0], last[1], px, py)
                last = (px, py)
            if self._hide_bars and polygon:
                from PySide6.QtGui import QPainterPath

                path = QPainterPath()
                path.moveTo(polygon[0][0], chart.bottom())
                for px, py in polygon:
                    path.lineTo(px, py)
                path.lineTo(polygon[-1][0], chart.bottom())
                path.closeSubpath()
                fill = QColor(curve.color)
                fill = fill.lighter(160)
                fill.setAlpha(110)
                painter.fillPath(path, fill)

    def _legend_items(self) -> list[tuple[str, QColor]]:
        items: list[tuple[str, QColor]] = []
        seen: set[str] = set()
        for bar in self._bars:
            for segment in bar.segments:
                if segment.group_label not in seen:
                    seen.add(segment.group_label)
                    items.append((segment.group_label, segment.color))
        for curve in self._curves:
            legend_label = curve.label
            if curve.description:
                legend_label = curve.description if curve.label == i18n.t("All data") else f"{curve.label} ({curve.description})"
            if legend_label not in seen:
                seen.add(legend_label)
                items.append((legend_label, curve.color))
        if len(items) == 1 and items[0][0] in {i18n.t("All data"), ""}:
            return []
        return items

    @staticmethod
    def _draw_legend(painter: QPainter, rect: QRect, items: list[tuple[str, QColor]]) -> None:
        painter.setPen(QPen(QColor("#c9c9c9"), 1.0))
        painter.setBrush(QColor("#ffffff"))
        painter.drawRect(rect)
        y = rect.top() + 12
        for label, color in items[:8]:
            painter.setBrush(color)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawEllipse(rect.left(), y + 2, 10, 10)
            painter.setPen(QColor("#222222"))
            painter.drawText(QRect(rect.left() + 16, y - 2, rect.width() - 20, 18), Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, label)
            y += 24

    def _bar_visible_height(self, bar: _DistributionBar) -> float:
        if not bar.segments:
            return 0.0
        if self._relative:
            if self._stacked:
                return 1.0
            total = max(1, sum(segment.count for segment in bar.segments))
            return max(float(segment.count) / total for segment in bar.segments)
        if self._stacked:
            return sum(float(segment.count) for segment in bar.segments)
        return max(float(segment.count) for segment in bar.segments)


class DistributionsScreen(QWidget, WorkflowNodeScreenSupport):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._init_workflow_node_support()
        self._builder = GeneratedDatasetService()
        self._dataset: DatasetHandle | None = None
        self._selected_dataset: DatasetHandle | None = None
        self._annotated_dataset: DatasetHandle | None = None
        self._pending_selected_rows: list[int] = []
        self._pending_variable_name = ""
        self._pending_group_name = ""
        self._binnings: list[_BinningDefinition] = []

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
        self._variable_filter_edit = QLineEdit()
        self._variable_filter_edit.setPlaceholderText(i18n.t("Filter..."))
        self._variable_list = QListWidget()
        self._variable_list.setMinimumHeight(148)
        self._sort_freq_cb = QCheckBox(i18n.t("Sort categories by frequency"))
        variable_layout.addWidget(self._variable_filter_edit)
        variable_layout.addWidget(self._variable_list)
        variable_layout.addWidget(self._sort_freq_cb)
        sidebar.addWidget(variable_box)

        self._distribution_box = QGroupBox(i18n.t("Distribution"))
        distribution_layout = QVBoxLayout(self._distribution_box)
        fit_row = QHBoxLayout()
        self._fit_label = QLabel(i18n.t("Fitted distribution"))
        fit_row.addWidget(self._fit_label)
        self._fit_combo = QComboBox()
        self._fit_combo.addItems(
            [
                i18n.t("None"),
                "Normal",
                "Beta",
                "Gamma",
                "Rayleigh",
                "Pareto",
                "Exponential",
                "Kernel density",
            ]
        )
        fit_row.addWidget(self._fit_combo, 1)
        distribution_layout.addLayout(fit_row)
        bin_row = QHBoxLayout()
        bin_row.addWidget(QLabel(i18n.t("Bin width")))
        self._bins_slider = QSlider(Qt.Orientation.Horizontal)
        self._bins_slider.setRange(0, 0)
        self._bins_slider.setValue(5)
        self._bins_label = QLabel("5")
        self._bins_label.setMinimumWidth(28)
        self._bins_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        bin_row.addWidget(self._bins_slider, 1)
        bin_row.addWidget(self._bins_label)
        distribution_layout.addLayout(bin_row)
        smoothing_row = QHBoxLayout()
        self._smoothing_label = QLabel(i18n.t("Smoothing"))
        self._smoothing_slider = QSlider(Qt.Orientation.Horizontal)
        self._smoothing_slider.setRange(2, 20)
        self._smoothing_slider.setValue(10)
        self._smoothing_value_label = QLabel("10")
        self._smoothing_value_label.setMinimumWidth(28)
        self._smoothing_value_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        smoothing_row.addWidget(self._smoothing_label)
        smoothing_row.addWidget(self._smoothing_slider, 1)
        smoothing_row.addWidget(self._smoothing_value_label)
        distribution_layout.addLayout(smoothing_row)
        self._hide_bars_cb = QCheckBox(i18n.t("Hide bars"))
        distribution_layout.addWidget(self._hide_bars_cb)
        sidebar.addWidget(self._distribution_box)

        columns_box = QGroupBox(i18n.t("Columns"))
        columns_layout = QVBoxLayout(columns_box)
        split_row = QHBoxLayout()
        split_row.addWidget(QLabel(i18n.t("Split by")))
        self._group_combo = QComboBox()
        self._group_combo.addItem(i18n.t("None"))
        split_row.addWidget(self._group_combo, 1)
        columns_layout.addLayout(split_row)
        self._stacked_cb = QCheckBox(i18n.t("Stack columns"))
        self._show_probs_cb = QCheckBox(i18n.t("Show probabilities"))
        self._cumulative_cb = QCheckBox(i18n.t("Show cumulative distribution"))
        self._legend_cb = QCheckBox(i18n.t("Show legend"))
        self._legend_cb.setChecked(True)
        columns_layout.addWidget(self._stacked_cb)
        columns_layout.addWidget(self._show_probs_cb)
        columns_layout.addWidget(self._cumulative_cb)
        columns_layout.addWidget(self._legend_cb)
        sidebar.addWidget(columns_box)

        self._auto_apply_cb = QCheckBox(i18n.t("Apply Automatically"))
        self._auto_apply_cb.setChecked(True)
        sidebar.addWidget(self._auto_apply_cb)

        self._dataset_label = QLabel(i18n.t("Dataset: none"))
        self._status_label = QLabel(i18n.t("Load data and choose a variable."))
        self._status_label.setWordWrap(True)
        sidebar.addWidget(self._dataset_label)
        sidebar.addWidget(self._status_label)

        self._clear_button = QPushButton(i18n.t("Clear Selection"))
        self._clear_button.clicked.connect(self._clear_selection)
        sidebar.addWidget(self._clear_button)

        self._selection_label = QLabel(i18n.t("Selected: 0"))
        sidebar.addWidget(self._selection_label)
        sidebar.addStretch(1)

        self._canvas = _DistributionCanvas(self)
        self._canvas.selectionChanged.connect(self._handle_selection_changed)
        root.addWidget(self._canvas, 1)
        self._canvas.setMinimumWidth(560)

        self._variable_filter_edit.textChanged.connect(self._filter_variable_list)
        self._variable_list.currentItemChanged.connect(self._handle_variable_changed)
        self._group_combo.currentTextChanged.connect(self._refresh_plot)
        self._bins_slider.valueChanged.connect(self._handle_bins_changed)
        self._smoothing_slider.valueChanged.connect(self._handle_smoothing_changed)
        self._fit_combo.currentTextChanged.connect(self._refresh_plot)
        self._sort_freq_cb.toggled.connect(self._refresh_plot)
        self._stacked_cb.toggled.connect(self._refresh_plot)
        self._cumulative_cb.toggled.connect(self._refresh_plot)
        self._hide_bars_cb.toggled.connect(self._refresh_plot)
        self._show_probs_cb.toggled.connect(self._refresh_plot)
        self._legend_cb.toggled.connect(self._refresh_plot)

    def sizeHint(self) -> QSize:
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
            "variable": self._current_variable_name(),
            "group": self._group_combo.currentText(),
            "bins": self._bins_slider.value(),
            "smoothing": self._smoothing_slider.value(),
            "fit": self._fit_combo.currentText(),
            "sort_by_freq": self._sort_freq_cb.isChecked(),
            "stacked": self._stacked_cb.isChecked(),
            "cumulative": self._cumulative_cb.isChecked(),
            "hide_bars": self._hide_bars_cb.isChecked(),
            "show_probs": self._show_probs_cb.isChecked(),
            "show_legend": self._legend_cb.isChecked(),
            "selected_rows": list(self._pending_selected_rows),
        }

    def restore_node_state(self, payload: dict[str, object]) -> None:
        self._pending_selected_rows = [
            int(index)
            for index in payload.get("selected_rows", [])
            if isinstance(index, int | float)
        ]
        self._pending_variable_name = str(payload.get("variable", ""))
        self._pending_group_name = str(payload.get("group", i18n.t("None")))
        self._bins_slider.setValue(int(payload.get("bins", 10)))
        self._smoothing_slider.setValue(int(payload.get("smoothing", 10)))
        self._set_combo_value(self._fit_combo, str(payload.get("fit", i18n.t("None"))))
        self._sort_freq_cb.setChecked(bool(payload.get("sort_by_freq", False)))
        self._stacked_cb.setChecked(bool(payload.get("stacked", False)))
        self._cumulative_cb.setChecked(bool(payload.get("cumulative", False)))
        self._hide_bars_cb.setChecked(bool(payload.get("hide_bars", False)))
        self._show_probs_cb.setChecked(bool(payload.get("show_probs", False)))
        self._legend_cb.setChecked(bool(payload.get("show_legend", True)))

    def documentation_url(self) -> str:
        return "https://orangedatamining.com/widget-catalog/visualize/distributions/"

    def _sync_controls(self) -> None:
        variables = self._ordered_variable_names()
        groups = self._ordered_group_names()
        current_variable = self._pending_variable_name or self._current_variable_name()
        current_group = self._pending_group_name or self._current_group_name()
        self._group_combo.blockSignals(True)
        self._group_combo.clear()
        self._group_combo.addItem(i18n.t("None"), i18n.t("None"))
        for group_name in groups:
            column = self._column_schema(group_name)
            if column is None:
                self._group_combo.addItem(group_name, group_name)
            else:
                self._group_combo.addItem(type_badge_icon(column.logical_type), group_name, group_name)
        self._group_combo.blockSignals(False)
        self._populate_variable_list(variables, current_variable or self._default_variable_name(variables))
        self._set_combo_value(
            self._group_combo,
            current_group if current_group != i18n.t("None") else self._default_group_name(),
        )
        self._pending_variable_name = ""
        self._pending_group_name = ""
        self._filter_variable_list(self._variable_filter_edit.text())

    def _refresh_plot(self) -> None:
        dataset = self._dataset
        self._selected_dataset = None
        self._annotated_dataset = None
        if dataset is None:
            self._dataset_label.setText(i18n.t("Dataset: none"))
            self._status_label.setText(i18n.t("Load data and choose a variable."))
            self._canvas.set_state([], stacked=False, relative=False)
            self._set_binnings([])
            self._handle_selection_changed([])
            return

        variable = prepared_column(dataset, self._current_variable_name())
        group_name = self._current_group_name()
        group = prepared_column(dataset, None if group_name == i18n.t("None") else group_name)
        self._dataset_label.setText(i18n.tf("Dataset: {name}", name=dataset.display_name))
        if variable is None:
            self._status_label.setText(i18n.t("Choose a variable."))
            self._set_binnings([])
            return

        if variable.is_discrete:
            self._set_binnings([])
        else:
            self._update_binnings(variable.values)

        curves: list[_DistributionCurve] = []
        if variable.is_discrete:
            bars = self._build_discrete_bars(variable, group)
        else:
            bars, curves = self._build_numeric_distribution_state(variable, group)
        can_show_probs = group is not None
        if not can_show_probs:
            self._show_probs_cb.setChecked(False)
        fit_active = self._fit_combo.currentText() not in {"", i18n.t("None")}
        can_hide_bars = fit_active and not variable.is_discrete
        if not can_hide_bars:
            self._hide_bars_cb.setChecked(False)
        self._update_control_states()
        relative_mode = self._show_probs_cb.isChecked()
        status = i18n.tf("Bars: {count}", count=len(bars))
        if curves:
            status += i18n.tf(" | Fit: {name}", name=self._fit_combo.currentText())
        if self._cumulative_cb.isChecked():
            status += i18n.t(" | cumulative")
        if self._show_probs_cb.isChecked():
            status += i18n.t(" | probabilities")
        self._status_label.setText(status)
        self._canvas.set_state(
            bars,
            stacked=self._stacked_cb.isChecked(),
            relative=relative_mode,
            cumulative=self._cumulative_cb.isChecked(),
            show_legend=self._legend_cb.isChecked(),
            hide_bars=self._hide_bars_cb.isChecked(),
            curves=curves,
            x_label=variable.name,
            y_label=(
                i18n.tf("Probability of '{group}' at given '{var}'", group=group.name, var=variable.name)
                if self._show_probs_cb.isChecked() and group is not None
                else i18n.t("Frequency")
            ),
        )
        self._handle_selection_changed(self._pending_selected_rows)

    def _handle_variable_changed(self, _current, _previous) -> None:
        self._update_control_states()
        self._refresh_plot()

    def _handle_bins_changed(self, value: int) -> None:
        if self._binnings and 0 <= value < len(self._binnings):
            self._bins_label.setText(self._binnings[value].width_label or str(value))
        else:
            self._bins_label.setText(str(value))
        self._refresh_plot()

    def _handle_smoothing_changed(self, value: int) -> None:
        self._smoothing_value_label.setText(str(value))
        if self._fit_combo.currentText() == "Kernel density":
            self._refresh_plot()

    def _current_variable_name(self) -> str:
        item = self._variable_list.currentItem()
        return item.text() if item is not None else ""

    def _populate_variable_list(self, values: list[str], current: str) -> None:
        self._variable_list.blockSignals(True)
        self._variable_list.clear()
        for value in values:
            column = self._column_schema(value)
            item = QListWidgetItem(type_badge_icon(column.logical_type), value) if column is not None else QListWidgetItem(value)
            self._variable_list.addItem(item)
        matches = self._variable_list.findItems(current, Qt.MatchFlag.MatchExactly)
        if matches:
            self._variable_list.setCurrentItem(matches[0])
        elif self._variable_list.count():
            self._variable_list.setCurrentRow(0)
        self._variable_list.blockSignals(False)

    def _filter_variable_list(self, text: str) -> None:
        needle = text.strip().lower()
        first_visible = None
        for row in range(self._variable_list.count()):
            item = self._variable_list.item(row)
            visible = not needle or needle in item.text().lower()
            item.setHidden(not visible)
            if visible and first_visible is None:
                first_visible = item
        current = self._variable_list.currentItem()
        if current is None or current.isHidden():
            if first_visible is not None:
                self._variable_list.setCurrentItem(first_visible)

    def _ordered_variable_names(self) -> list[str]:
        dataset = self._dataset
        if dataset is None:
            return []
        primitive = primitive_columns(dataset)
        targets = [column.name for column in dataset.domain.target_columns if column.name in primitive]
        others = [name for name in primitive if name not in targets]
        return [*targets, *others]

    def _ordered_group_names(self) -> list[str]:
        dataset = self._dataset
        if dataset is None:
            return []
        discrete = discrete_columns(dataset)
        targets = [column.name for column in dataset.domain.target_columns if column.name in discrete]
        others = [name for name in discrete if name not in targets]
        return [*targets, *others]

    def _default_variable_name(self, variables: list[str]) -> str:
        dataset = self._dataset
        if dataset is None or not variables:
            return ""
        target_names = {column.name for column in dataset.domain.target_columns}
        for name in variables:
            if name not in target_names:
                return name
        return variables[0]

    def _default_group_name(self) -> str:
        dataset = self._dataset
        if dataset is None:
            return i18n.t("None")
        for column in dataset.domain.target_columns:
            if column.logical_type in {"categorical", "boolean"}:
                return column.name
        return i18n.t("None")

    def _update_control_states(self) -> None:
        variable = prepared_column(self._dataset, self._current_variable_name())
        group_name = self._current_group_name()
        group_selected = bool(group_name and group_name != i18n.t("None"))
        variable_is_discrete = bool(variable is not None and variable.is_discrete)
        fit_active = self._fit_combo.currentText() not in {"", i18n.t("None")}
        kde_active = self._fit_combo.currentText() == "Kernel density" and not variable_is_discrete

        self._sort_freq_cb.setEnabled(variable_is_discrete)
        self._distribution_box.setDisabled(variable_is_discrete)
        self._hide_bars_cb.setEnabled(fit_active and not variable_is_discrete)
        self._smoothing_label.setEnabled(kde_active)
        self._smoothing_slider.setEnabled(kde_active)
        self._smoothing_value_label.setEnabled(kde_active)
        self._show_probs_cb.setEnabled(group_selected)
        self._stacked_cb.setEnabled(group_selected)
        self._fit_label.setText(i18n.t("Fitted probability") if self._show_probs_cb.isChecked() else i18n.t("Fitted distribution"))
        self._fit_label.setToolTip(
            i18n.t("Chosen distribution is used to compute Bayesian probabilities")
            if self._show_probs_cb.isChecked()
            else ""
        )

    def _column_schema(self, name: str):
        dataset = self._dataset
        if dataset is None:
            return None
        for column in dataset.domain.columns:
            if column.name == name:
                return column
        return None

    def _current_group_name(self) -> str:
        data = self._group_combo.currentData()
        if isinstance(data, str) and data:
            return data
        text = self._group_combo.currentText()
        return text if text else i18n.t("None")

    def _group_color_map(self, group: PlotColumn | None, *, lighten: bool = False) -> dict[str, QColor]:
        labels = list(group.categories) if group is not None else [i18n.t("All data")]
        return {label: self._orange_color(index, lighten=lighten) for index, label in enumerate(labels)}

    @staticmethod
    def _orange_color(index: int, *, lighten: bool = False) -> QColor:
        color = QColor(*ORANGE_DISCRETE_COLORS[index % len(ORANGE_DISCRETE_COLORS)])
        return color.lighter(130) if lighten else color

    def _set_binnings(self, binnings: list[_BinningDefinition]) -> None:
        self._binnings = binnings
        self._bins_slider.blockSignals(True)
        if binnings:
            self._bins_slider.setRange(0, len(binnings) - 1)
            current = min(self._bins_slider.value(), len(binnings) - 1)
            default_index = min(5, len(binnings) - 1)
            if self._bins_slider.value() > len(binnings) - 1:
                current = default_index
            self._bins_slider.setValue(current)
            self._bins_label.setText(binnings[current].width_label or str(current))
        else:
            self._bins_slider.setRange(0, 0)
            self._bins_slider.setValue(0)
            self._bins_label.setText("")
        self._bins_slider.blockSignals(False)

    def _update_binnings(self, values: np.ndarray) -> None:
        binnings = self._decimal_binnings(values)
        self._set_binnings(binnings)

    def _current_thresholds(self, values: np.ndarray) -> np.ndarray:
        if self._binnings and 0 <= self._bins_slider.value() < len(self._binnings):
            return self._binnings[self._bins_slider.value()].thresholds
        _hist, edges = np.histogram(values, bins=max(3, self._bins_slider.value()))
        return edges

    @staticmethod
    def _decimal_binnings(values: np.ndarray) -> list[_BinningDefinition]:
        values = np.asarray(values, dtype=float)
        values = values[np.isfinite(values)]
        if values.size == 0:
            return []
        unique = np.unique(values)
        bins: list[_BinningDefinition] = []
        if len(unique) <= 10:
            bins.append(_BinningDefinition(_unique_thresholds(unique), "unique"))
        diff = float(unique[-1] - unique[0])
        if diff <= 1e-12:
            return bins or [_BinningDefinition(np.array([unique[0], unique[0] + 1.0]), "1")]
        f10 = 10 ** -np.floor(np.log10(diff))
        factors = (0.01, 0.02, 0.025, 0.05, 0.1, 0.2, 0.25, 0.5, 1, 2, 5, 10, 20)
        max_bins = min(50, len(unique))
        for factor in factors:
            width = factor / f10
            mn = np.floor(unique[0] / width) * width
            mx = np.ceil(unique[-1] / width) * width
            nbins = int(round((mx - mn) / width))
            if 2 <= nbins <= max_bins and (not bins or len(bins[-1].thresholds) - 1 != nbins):
                thresholds = mn + width * np.arange(nbins + 1)
                thresholds = np.around(thresholds, decimals=np.finfo(thresholds.dtype).precision)
                bins.append(_BinningDefinition(thresholds=thresholds, width_label=f"{width:g}"))
        return bins or [_BinningDefinition(np.histogram_bin_edges(values, bins=5), "5")]

    def _build_discrete_bars(self, variable: PlotColumn, group: PlotColumn | None) -> list[_DistributionBar]:
        group_lookup = {row: str(raw) for row, raw in zip(group.row_indices, group.raw_values)} if group else {}
        color_map = self._group_color_map(group, lighten=self._fit_combo.currentText() not in {"", i18n.t("None")})
        rows_by_value_group: dict[str, dict[str, list[int]]] = {}
        for row, raw in zip(variable.row_indices, variable.raw_values):
            value_label = str(raw)
            group_label = group_lookup.get(int(row), i18n.t("All data")) if group else i18n.t("All data")
            rows_by_value_group.setdefault(value_label, {}).setdefault(group_label, []).append(int(row))

        total = len(variable.row_indices) or 1
        bars: list[_DistributionBar] = []
        ordered_items = list(rows_by_value_group.items())
        if self._sort_freq_cb.isChecked():
            ordered_items.sort(key=lambda item: -sum(len(rows) for rows in item[1].values()))
        for bar_index, (value_label, group_map) in enumerate(ordered_items):
            segments = []
            for seg_index, (group_label, rows) in enumerate(group_map.items()):
                segments.append(
                    _DistributionSegment(
                        group_label=group_label,
                        count=len(rows),
                        value=len(rows) / total,
                        row_indices=tuple(rows),
                        color=QColor(color_map.get(group_label, self._orange_color(seg_index))),
                    )
                )
            bars.append(_DistributionBar(label=value_label, segments=tuple(segments)))
        return bars

    def _build_numeric_bars(self, variable: PlotColumn, group: PlotColumn | None) -> list[_DistributionBar]:
        values = variable.values
        row_indices = variable.row_indices
        edges = self._current_thresholds(values)
        total = len(values) or 1
        group_lookup = {row: str(raw) for row, raw in zip(group.row_indices, group.raw_values)} if group else {}
        color_map = self._group_color_map(group, lighten=self._fit_combo.currentText() not in {"", i18n.t("None")})
        bars: list[_DistributionBar] = []
        for index in range(len(edges) - 1):
            low = float(edges[index])
            high = float(edges[index + 1])
            desc = self._format_interval_label(variable.name, low, high, index == 0, index == len(edges) - 2)
            if index == len(edges) - 2:
                mask = (values >= low) & (values <= high)
            else:
                mask = (values >= low) & (values < high)
            value_rows = row_indices[mask]
            if group is None:
                rows = tuple(int(row) for row in value_rows)
                bars.append(
                    _DistributionBar(
                        label=desc,
                        segments=(
                            _DistributionSegment(
                                group_label=i18n.t("All data"),
                                count=len(rows),
                                value=len(rows) / total,
                                row_indices=rows,
                                color=QColor(self._orange_color(0, lighten=self._fit_combo.currentText() not in {"", i18n.t("None")})),
                            ),
                        ),
                        low=low,
                        high=high,
                    )
                )
                continue

            grouped_rows: dict[str, list[int]] = {}
            for row in value_rows:
                label = group_lookup.get(int(row))
                if label is None:
                    continue
                grouped_rows.setdefault(label, []).append(int(row))
            segments = [
                _DistributionSegment(
                    group_label=label,
                    count=len(rows),
                    value=len(rows) / total,
                    row_indices=tuple(rows),
                    color=QColor(color_map.get(label, self._orange_color(seg_index))),
                )
                for seg_index, (label, rows) in enumerate(grouped_rows.items())
            ]
            bars.append(_DistributionBar(label=desc, segments=tuple(segments), low=low, high=high))
        return bars

    def _build_numeric_distribution_state(
        self,
        variable: PlotColumn,
        group: PlotColumn | None,
    ) -> tuple[list[_DistributionBar], list[_DistributionCurve]]:
        values = variable.values
        row_indices = variable.row_indices
        total = len(values) or 1
        edges = self._current_thresholds(values)
        group_lookup = {row: str(raw) for row, raw in zip(group.row_indices, group.raw_values)} if group else {}
        color_map = self._group_color_map(group, lighten=self._fit_combo.currentText() not in {"", i18n.t("None")})
        cumulative = self._cumulative_cb.isChecked()
        bars: list[_DistributionBar] = []
        grouped_values: dict[str, list[float]] = {}
        running_rows_all: list[int] = []
        running_rows_by_group: dict[str, list[int]] = {}

        for index in range(len(edges) - 1):
            low = float(edges[index])
            high = float(edges[index + 1])
            desc = self._format_interval_label(variable.name, low, high, index == 0, index == len(edges) - 2)
            if index == len(edges) - 2:
                mask = (values >= low) & (values <= high)
            else:
                mask = (values >= low) & (values < high)
            value_rows = row_indices[mask]
            value_values = values[mask]
            if group is None:
                rows = [int(row) for row in value_rows]
                if cumulative:
                    running_rows_all.extend(rows)
                    rows = list(dict.fromkeys(running_rows_all))
                bars.append(
                    _DistributionBar(
                        label=desc,
                        segments=(
                            _DistributionSegment(
                                group_label=i18n.t("All data"),
                                count=len(rows),
                                value=len(rows) / total,
                                row_indices=tuple(rows),
                                color=QColor(self._orange_color(0, lighten=self._fit_combo.currentText() not in {"", i18n.t("None")})),
                            ),
                        ),
                        low=low,
                        high=high,
                    )
                )
                continue

            grouped_rows: dict[str, list[int]] = {}
            for row, value in zip(value_rows, value_values):
                label = group_lookup.get(int(row))
                if label is None:
                    continue
                grouped_rows.setdefault(label, []).append(int(row))
                grouped_values.setdefault(label, []).append(float(value))
            segments = []
            for seg_index, (label, rows) in enumerate(grouped_rows.items()):
                if cumulative:
                    running_rows_by_group.setdefault(label, []).extend(rows)
                    segment_rows = list(dict.fromkeys(running_rows_by_group[label]))
                else:
                    segment_rows = rows
                segments.append(
                    _DistributionSegment(
                        group_label=label,
                        count=len(segment_rows),
                        value=len(segment_rows) / total,
                        row_indices=tuple(segment_rows),
                        color=QColor(color_map.get(label, self._orange_color(seg_index))),
                    )
                )
            bars.append(_DistributionBar(label=desc, segments=tuple(segments), low=low, high=high))

        curves = self._build_numeric_curves(variable, grouped_values if group else None, edges)
        return bars, curves

    def _build_numeric_curves(
        self,
        variable: PlotColumn,
        grouped_values: dict[str, list[float]] | None,
        edges: np.ndarray,
    ) -> list[_DistributionCurve]:
        fit_name = self._fit_combo.currentText()
        if fit_name in {"", i18n.t("None")} or len(edges) < 2:
            return []
        curves: list[_DistributionCurve] = []
        if grouped_values and self._show_probs_cb.isChecked():
            return self._build_probability_curves(grouped_values, edges, total=max(1, len(variable.values)))
        if grouped_values:
            group = prepared_column(self._dataset, self._current_group_name())
            color_map = self._group_color_map(group, lighten=False)
            for index, (label, values) in enumerate(grouped_values.items()):
                curve = self._fit_curve(
                    np.asarray(values, dtype=float),
                    edges,
                    label,
                    QColor(color_map.get(label, self._orange_color(index))),
                    total=max(1, len(variable.values)),
                )
                if curve is not None:
                    curves.append(curve)
            return curves
        curve = self._fit_curve(
            variable.values.astype(float),
            edges,
            i18n.t("All data"),
            QColor("#111111"),
            total=max(1, len(variable.values)),
        )
        return [curve] if curve is not None else []

    def _fit_curve(
        self,
        values: np.ndarray,
        edges: np.ndarray,
        label: str,
        color: QColor,
        *,
        total: int,
    ) -> _DistributionCurve | None:
        if len(values) < 2:
            return None
        support = np.linspace(float(edges[0]), float(edges[-1]), 120)
        cumulative = self._cumulative_cb.isChecked()
        relative = self._show_probs_cb.isChecked()
        span = float(edges[-1] - edges[0]) or 1.0
        y_values = self._evaluate_fit(values, support)
        if y_values is None:
            return None
        if not np.isfinite(y_values).all():
            return None
        if cumulative:
            scaled = y_values if relative else y_values * total
        else:
            bin_width = span / max(1, len(edges) - 1)
            scaled = y_values * bin_width
            if not relative:
                scaled = scaled * total
        description = self._fit_description(values)
        points = tuple(
            (float((x_value - float(edges[0])) / span), float(max(0.0, y_value)))
            for x_value, y_value in zip(support, scaled)
        )
        return _DistributionCurve(label=label, points=points, color=color, description=description)

    def _fit_description(self, values: np.ndarray) -> str:
        fit_name = self._fit_combo.currentText()
        if fit_name == "Normal":
            mu = float(np.nanmean(values))
            sigma = float(np.nanstd(values, ddof=1))
            return f"μ={mu:.3g}, σ={sigma:.3g}"
        if fit_name == "Exponential":
            scale = float(np.nanmean(np.maximum(values - np.nanmin(values), 0.0)))
            return f"λ={scale:.3g}"
        if fit_name == "Gamma":
            mean = float(np.nanmean(values))
            var = float(np.nanvar(values, ddof=1))
            alpha = mean * mean / var if var > 1e-12 else 1.0
            beta_param = var / mean if abs(mean) > 1e-12 else 1.0
            return f"α={alpha:.3g}, β={beta_param:.3g}"
        if fit_name == "Beta":
            return "α, β fit"
        if fit_name in {"Rayleigh", "Pareto", "Kernel density"}:
            return fit_name
        return ""

    def _evaluate_fit(self, values: np.ndarray, support: np.ndarray) -> np.ndarray | None:
        fit_name = self._fit_combo.currentText()
        cumulative = self._cumulative_cb.isChecked()
        try:
            if fit_name == "Normal":
                params = norm.fit(values)
                fn = norm.cdf if cumulative else norm.pdf
                return np.asarray(fn(support, *params), dtype=float)
            if fit_name == "Beta":
                params = beta.fit(values)
                fn = beta.cdf if cumulative else beta.pdf
                return np.asarray(fn(support, *params), dtype=float)
            if fit_name == "Gamma":
                params = gamma.fit(values)
                fn = gamma.cdf if cumulative else gamma.pdf
                return np.asarray(fn(support, *params), dtype=float)
            if fit_name == "Rayleigh":
                params = rayleigh.fit(values)
                fn = rayleigh.cdf if cumulative else rayleigh.pdf
                return np.asarray(fn(support, *params), dtype=float)
            if fit_name == "Pareto":
                params = pareto.fit(values)
                fn = pareto.cdf if cumulative else pareto.pdf
                return np.asarray(fn(support, *params), dtype=float)
            if fit_name == "Exponential":
                params = expon.fit(values)
                fn = expon.cdf if cumulative else expon.pdf
                return np.asarray(fn(support, *params), dtype=float)
            if fit_name == "Kernel density":
                from portakal_app.ui.screens.visualize_common import kernel_density

                support2, density = kernel_density(values, points=len(support), kernel="gaussian")
                density = self._smooth_density(density)
                if cumulative:
                    cumulative_density = np.cumsum(density)
                    total_density = float(cumulative_density[-1]) or 1.0
                    return cumulative_density / total_density
                return np.interp(support, support2, density)
        except Exception:
            return None
        return None

    def _smooth_density(self, density: np.ndarray) -> np.ndarray:
        if density.size < 3:
            return density
        radius = max(1, self._smoothing_slider.value() // 3)
        sigma = max(self._smoothing_slider.value() / 6.0, 1e-6)
        offsets = np.arange(-radius, radius + 1, dtype=float)
        kernel = np.exp(-0.5 * (offsets / sigma) ** 2)
        kernel /= max(float(np.sum(kernel)), 1e-9)
        return np.convolve(density, kernel, mode="same")

    def _build_probability_curves(
        self,
        grouped_values: dict[str, list[float]],
        edges: np.ndarray,
        *,
        total: int,
    ) -> list[_DistributionCurve]:
        support = np.linspace(float(edges[0]), float(edges[-1]), 120)
        span = float(edges[-1] - edges[0]) or 1.0
        priors = {
            label: len(values) / max(total, 1)
            for label, values in grouped_values.items()
            if len(values) >= 2
        }
        if not priors:
            return []

        fitted_curves: dict[str, np.ndarray] = {}
        for label, values in grouped_values.items():
            if len(values) < 2:
                continue
            fitted = self._evaluate_fit(np.asarray(values, dtype=float), support)
            if fitted is None:
                continue
            if np.isfinite(fitted).all():
                fitted_curves[label] = np.maximum(fitted, 0.0)

        if not fitted_curves:
            return []

        labels = list(fitted_curves)
        weighted = np.vstack([fitted_curves[label] * priors[label] for label in labels])
        totals = weighted.sum(axis=0)
        totals[totals <= 1e-12] = 1.0

        curves: list[_DistributionCurve] = []
        group = prepared_column(self._dataset, self._current_group_name())
        color_map = self._group_color_map(group, lighten=False)
        for index, label in enumerate(labels):
            probabilities = weighted[index] / totals
            curves.append(
                _DistributionCurve(
                    label=label,
                    points=tuple(
                        (float((x_value - float(edges[0])) / span), float(probability))
                        for x_value, probability in zip(support, probabilities)
                    ),
                    color=QColor(color_map.get(label, self._orange_color(index))),
                )
            )
        return curves

    def _handle_selection_changed(self, rows: list[int]) -> None:
        self._pending_selected_rows = sorted({index for index in rows})
        self._canvas.set_selected_rows(self._pending_selected_rows)
        self._selection_label.setText(i18n.tf("Selected: {count}", count=len(self._pending_selected_rows)))
        self._selected_dataset, self._annotated_dataset = build_selection_outputs(
            self._dataset,
            self._pending_selected_rows,
            generated_by="distributions",
            service=self._builder,
        )
        self._notify_output_changed()

    def _clear_selection(self) -> None:
        self._canvas.clear_selection()

    @staticmethod
    def _format_interval_label(name: str, low: float, high: float, first: bool, last: bool) -> str:
        low_text = f"{low:.3g}"
        high_text = f"{high:.3g}"
        if first and last:
            return f"{name} = {low_text}"
        if first:
            return f"{name} < {high_text}"
        if last:
            return f"{name} ≥ {low_text}"
        if low_text == high_text:
            return f"{name} = {low_text}"
        return f"{low_text} ≤ {name} < {high_text}"

    @staticmethod
    def _set_combo_value(combo: QComboBox, value: str) -> None:
        if not value:
            return
        index = combo.findText(value)
        if index >= 0:
            combo.setCurrentIndex(index)
