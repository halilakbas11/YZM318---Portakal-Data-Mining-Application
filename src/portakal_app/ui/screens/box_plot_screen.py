from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import chi2_contingency, f_oneway, ttest_ind
from PySide6.QtCore import QRect, QSize, Qt, Signal
from PySide6.QtGui import QColor, QFont, QPainter, QPen
from PySide6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
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
    nice_ticks,
    prepared_column,
    primitive_columns,
)


@dataclass(frozen=True)
class _NumericEntry:
    label: str
    group_value: str
    a_min: float
    a_max: float
    q1: float
    median: float
    q3: float
    mean: float
    dev: float
    row_indices: tuple[int, ...]
    color: QColor


@dataclass(frozen=True)
class _DiscreteSegment:
    label: str
    group_value: str
    count: int
    row_indices: tuple[int, ...]
    color: QColor


@dataclass(frozen=True)
class _DiscreteRow:
    label: str
    group_value: str
    total_count: int
    segments: tuple[_DiscreteSegment, ...]


class _BoxPlotCanvas(QWidget):
    selectionChanged = Signal(list)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._attribute: PlotColumn | None = None
        self._numeric_entries: list[_NumericEntry] = []
        self._discrete_rows: list[_DiscreteRow] = []
        self._selected_rows: set[int] = set()
        self._hit_regions: list[tuple[QRect, tuple[int, ...], str]] = []
        self._show_annotations = True
        self._show_stretched = True
        self._show_labels = True
        self._compare_mode = 2
        self._stat_text = ""
        self.setMinimumHeight(300)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMouseTracking(True)

    def set_numeric_state(
        self,
        attribute: PlotColumn | None,
        entries: list[_NumericEntry],
        *,
        show_annotations: bool,
        compare_mode: int,
        stat_text: str,
    ) -> None:
        self._attribute = attribute
        self._numeric_entries = entries
        self._discrete_rows = []
        self._show_annotations = show_annotations
        self._compare_mode = compare_mode
        self._stat_text = stat_text
        self._hit_regions = []
        self.update()

    def set_discrete_state(
        self,
        attribute: PlotColumn | None,
        rows: list[_DiscreteRow],
        *,
        stretched: bool,
        show_labels: bool,
        stat_text: str,
    ) -> None:
        self._attribute = attribute
        self._numeric_entries = []
        self._discrete_rows = rows
        self._show_stretched = stretched
        self._show_labels = show_labels
        self._stat_text = stat_text
        self._hit_regions = []
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
        for rect, _rows, tooltip in self._hit_regions:
            if rect.contains(pos):
                QToolTip.showText(event.globalPosition().toPoint(), tooltip, self)
                return
        QToolTip.hideText()

    def mouseReleaseEvent(self, event) -> None:
        if event.button() != Qt.MouseButton.LeftButton:
            return super().mouseReleaseEvent(event)
        pos = event.position().toPoint()
        for rect, row_indices, _tooltip in self._hit_regions:
            if not rect.contains(pos):
                continue
            incoming = set(row_indices)
            modifiers = event.modifiers()
            if modifiers & Qt.KeyboardModifier.AltModifier:
                self._selected_rows.difference_update(incoming)
            elif modifiers & (Qt.KeyboardModifier.ControlModifier | Qt.KeyboardModifier.ShiftModifier):
                self._selected_rows.update(incoming)
            else:
                self._selected_rows = incoming
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
        painter.fillRect(self.rect(), QColor("#fffdf9"))
        self._hit_regions = []

        if self._attribute is None or (not self._numeric_entries and not self._discrete_rows):
            painter.setPen(QColor("#8d877d"))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, i18n.t("Load data and choose a variable."))
            return

        if self._numeric_entries:
            self._paint_numeric(painter)
        else:
            self._paint_discrete(painter)

    def _paint_numeric(self, painter: QPainter) -> None:
        margin_left = 44
        margin_right = 28
        margin_top = 22
        margin_bottom = 82
        axis_y = self.height() - margin_bottom
        chart = QRect(margin_left, margin_top, max(10, self.width() - margin_left - margin_right), max(10, axis_y - margin_top - 18))
        values = [entry.a_min for entry in self._numeric_entries] + [entry.a_max for entry in self._numeric_entries]
        low = min(values)
        high = max(values)
        if abs(high - low) < 1e-12:
            high = low + 1.0

        ticks = nice_ticks(low, high, count=6)
        axis_pen = QPen(QColor("#5b534c"), 2.2)
        axis_pen.setCosmetic(True)
        tick_pen = QPen(QColor("#efefef"), 5)
        tick_pen.setCosmetic(True)
        tick_pen.setCapStyle(Qt.PenCapStyle.FlatCap)
        painter.setPen(axis_pen)
        painter.drawLine(chart.left() - 4, axis_y, chart.right() + 4, axis_y)

        painter.setPen(QColor("#534b40"))
        painter.setFont(QFont(self.font().family(), 8))
        for tick in ticks:
            px = chart.left() + int((tick - low) / (high - low) * chart.width())
            painter.setPen(tick_pen)
            painter.drawLine(px, axis_y - 2, px, axis_y + 2)
            painter.setPen(QColor("#534b40"))
            painter.drawText(QRect(px - 34, axis_y + 8, 68, 16), Qt.AlignmentFlag.AlignCenter, f"{tick:.3g}")

        if self._stat_text:
            painter.setPen(QColor("#534b40"))
            painter.setFont(QFont(self.font().family(), 9))
            painter.drawText(QRect(chart.left(), axis_y + 48, chart.width(), 18), Qt.AlignmentFlag.AlignLeft, self._stat_text)

        title_font = QFont(self.font().family(), 9)
        title_font.setBold(True)
        painter.setFont(title_font)
        painter.drawText(
            QRect(chart.left(), axis_y + 28, chart.width(), 18),
            Qt.AlignmentFlag.AlignCenter,
            self._attribute.name,
        )

        row_height = 90 if self._show_annotations else 60
        metric_positions: list[tuple[int, int]] = []
        for row, entry in enumerate(self._numeric_entries):
            center_y = axis_y - (len(self._numeric_entries) - row) * row_height + 10
            box_top = center_y - 10
            box_bottom = center_y + 10

            def xpos(value: float) -> int:
                return chart.left() + int((value - low) / (high - low) * chart.width())

            x_min = xpos(entry.a_min)
            x_max = xpos(entry.a_max)
            x_q1 = xpos(entry.q1)
            x_med = xpos(entry.median)
            x_q3 = xpos(entry.q3)
            x_mean = xpos(entry.mean)
            x_dev_low = xpos(entry.mean - entry.dev)
            x_dev_high = xpos(entry.mean + entry.dev)

            painter.setPen(QPen(QColor("#54627a"), 1, Qt.PenStyle.DotLine))
            painter.drawLine(x_dev_low, center_y, x_dev_high, center_y)

            painter.setPen(QPen(QColor("#5f5f5f"), 1.3))
            painter.drawLine(x_min, center_y, x_q1, center_y)
            painter.drawLine(x_q3, center_y, x_max, center_y)
            painter.drawLine(x_min, center_y - 4, x_min, center_y + 4)
            painter.drawLine(x_max, center_y - 4, x_max, center_y + 4)

            box_rect = QRect(min(x_q1, x_q3), box_top, max(8, abs(x_q3 - x_q1)), box_bottom - box_top)
            fill = QColor("#3388ff")
            fill.setAlpha(176)
            border = QColor("#111111") if self._selected_rows.intersection(entry.row_indices) else QColor("#2958b0")
            painter.setBrush(fill)
            painter.setPen(QPen(border, 1.5))
            painter.drawRect(box_rect)

            painter.setPen(QPen(QColor("#ffdd33"), 2))
            painter.drawLine(x_med, box_top, x_med, box_bottom)
            painter.setPen(QPen(QColor("#3300ff"), 2))
            painter.drawLine(x_mean, box_top - 4, x_mean, box_bottom + 4)

            if self._show_annotations:
                painter.setPen(QColor("#534b40"))
                painter.setFont(QFont(self.font().family(), 8))
                mean_text = f"{entry.mean:.3g} +- {entry.dev:.3g}"
                if entry.label and entry.label != i18n.t("All data"):
                    mean_text = f"{entry.label}: {mean_text}"
                painter.drawText(
                    QRect(x_mean - 120, center_y - 34, 240, 16),
                    Qt.AlignmentFlag.AlignCenter,
                    mean_text,
                )
                for value, x_pos, y_off in (
                    (entry.q1, x_q1, 18),
                    (entry.median, x_med, 32),
                    (entry.q3, x_q3, 18),
                ):
                    painter.drawText(
                        QRect(x_pos - 30, center_y + y_off, 60, 14),
                        Qt.AlignmentFlag.AlignCenter,
                        f"{value:.3g}",
                    )
            elif entry.label:
                painter.setPen(QColor("#2f2417"))
                painter.setFont(QFont(self.font().family(), 8))
                if self._compare_mode == 1:
                    label_x = x_med
                elif self._compare_mode == 2:
                    label_x = x_mean
                else:
                    label_x = x_q1
                painter.drawText(
                    QRect(label_x - 80, center_y - 28, 160, 16),
                    Qt.AlignmentFlag.AlignCenter,
                    entry.label,
                )

            if self._compare_mode == 1:
                metric_positions.append((x_med, center_y))
            elif self._compare_mode == 2:
                metric_positions.append((x_mean, center_y))

            tooltip = "\n".join(
                [
                    entry.label or i18n.t("All data"),
                    f"Min: {entry.a_min:.3g}",
                    f"Q1: {entry.q1:.3g}",
                    f"Median: {entry.median:.3g}",
                    f"Q3: {entry.q3:.3g}",
                    f"Max: {entry.a_max:.3g}",
                    f"Mean: {entry.mean:.3g}",
                ]
            )
            self._hit_regions.append((box_rect.adjusted(-6, -4, 6, 4), entry.row_indices, tooltip))

        self._draw_numeric_posthoc(painter, chart, metric_positions)

    def _paint_discrete(self, painter: QPainter) -> None:
        margin_left = 140
        margin_right = 40
        margin_top = 22
        margin_bottom = 82
        axis_y = self.height() - margin_bottom
        chart = QRect(margin_left, margin_top, max(10, self.width() - margin_left - margin_right), max(10, axis_y - margin_top - 16))
        if not self._discrete_rows:
            return

        if self._show_stretched:
            axis_max = 100.0
            ticks = [0, 20, 40, 60, 80, 100]
        else:
            axis_max = max(float(row.total_count) for row in self._discrete_rows)
            axis_max = max(axis_max, 1.0)
            ticks = nice_ticks(0.0, axis_max, count=6)

        axis_pen = QPen(QColor("#5b534c"), 2.2)
        axis_pen.setCosmetic(True)
        tick_pen = QPen(QColor("#efefef"), 5)
        tick_pen.setCosmetic(True)
        tick_pen.setCapStyle(Qt.PenCapStyle.FlatCap)
        painter.setPen(axis_pen)
        painter.drawLine(chart.left(), axis_y, chart.right(), axis_y)

        painter.setPen(QColor("#534b40"))
        painter.setFont(QFont(self.font().family(), 8))
        for tick in ticks:
            px = chart.left() + int((float(tick) / axis_max) * chart.width())
            painter.setPen(tick_pen)
            painter.drawLine(px, axis_y - 2, px, axis_y + 2)
            painter.setPen(QColor("#534b40"))
            label = f"{int(tick)}%" if self._show_stretched else f"{int(tick)}"
            painter.drawText(QRect(px - 28, axis_y + 8, 56, 16), Qt.AlignmentFlag.AlignCenter, label)

        if self._stat_text:
            painter.setPen(QColor("#534b40"))
            painter.setFont(QFont(self.font().family(), 9))
            painter.drawText(QRect(chart.left(), axis_y + 48, chart.width(), 18), Qt.AlignmentFlag.AlignLeft, self._stat_text)

        title_font = QFont(self.font().family(), 9)
        title_font.setBold(True)
        painter.setFont(title_font)
        painter.drawText(
            QRect(chart.left(), axis_y + 28, chart.width(), 18),
            Qt.AlignmentFlag.AlignCenter,
            self._attribute.name,
        )

        row_height = 40
        for row_index, row in enumerate(self._discrete_rows):
            center_y = axis_y - (len(self._discrete_rows) - row_index) * row_height + 10
            top = center_y - 6
            left = chart.left()
            row_total = max(row.total_count, 1)

            painter.setFont(QFont(self.font().family(), 8))
            painter.setPen(QColor("#534b40"))
            painter.drawText(
                QRect(4, center_y - 12, margin_left - 12, 24),
                Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
                row.label,
            )

            for segment in row.segments:
                if self._show_stretched:
                    value = 100.0 * segment.count / row_total
                else:
                    value = float(segment.count)
                width = max(6, int((value / axis_max) * chart.width()))
                rect = QRect(left, top, width, 12)
                color = QColor(segment.color)
                color.setAlpha(210)
                border = QColor("#111111") if self._selected_rows.intersection(segment.row_indices) else QColor(color.darker(140))
                painter.setBrush(color)
                painter.setPen(QPen(border, 1.2))
                painter.drawRect(rect)

                tooltip = (
                    f"{segment.label}: {100.0 * segment.count / row_total:.2f}%"
                    if self._show_stretched
                    else f"{segment.label}: ({segment.count})"
                )
                self._hit_regions.append((rect.adjusted(-2, -2, 2, 2), segment.row_indices, tooltip))

                if self._show_labels and rect.width() >= 34:
                    painter.save()
                    painter.setClipRect(rect.adjusted(2, -18, -2, 0))
                    painter.setPen(QColor("#ffffff") if color.lightness() < 150 else QColor("#3b2a10"))
                    painter.drawText(rect.adjusted(4, -18, -4, -2), Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignBottom, segment.label)
                    painter.restore()
                left += width

            if not self._show_stretched:
                painter.setPen(QColor("#534b40"))
                painter.drawText(
                    QRect(chart.right() + 8, center_y - 10, margin_right - 8, 20),
                    Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
                    str(row.total_count),
                )

    def _draw_numeric_posthoc(self, painter: QPainter, chart: QRect, metric_positions: list[tuple[int, int]]) -> None:
        if self._compare_mode == 0 or len(metric_positions) < 2:
            return
        vertical_pen = QPen(QColor("#d0d0d0"), 1.5)
        connector_pen = QPen(QColor("#d0d0d0"), 2.5)
        painter.save()
        painter.setPen(vertical_pen)
        guide_y = chart.bottom() + 10
        xs = []
        for x_pos, center_y in metric_positions:
            xs.append(x_pos)
            painter.drawLine(x_pos, guide_y, x_pos, center_y - 12)
        xs.sort()
        if len(xs) < 2:
            painter.restore()
            return
        painter.setPen(connector_pen)
        groups: list[list[int]] = []
        current = [xs[0]]
        for x_pos in xs[1:]:
            if x_pos - current[-1] <= 42:
                current.append(x_pos)
            else:
                groups.append(current)
                current = [x_pos]
        groups.append(current)
        for level, group in enumerate(group for group in groups if len(group) > 1):
            y = guide_y - 8 - level * 6
            painter.drawLine(group[0] - 2, y, group[-1] + 2, y)
        painter.restore()


class BoxPlotScreen(QWidget, WorkflowNodeScreenSupport):
    CompareNone = 0
    CompareMedians = 1
    CompareMeans = 2

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._init_workflow_node_support()
        self._builder = GeneratedDatasetService()
        self._dataset: DatasetHandle | None = None
        self._selected_dataset: DatasetHandle | None = None
        self._annotated_dataset: DatasetHandle | None = None
        self._pending_selected_rows: list[int] = []

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

        variable_box = QGroupBox(i18n.t("Variable"))
        variable_layout = QVBoxLayout(variable_box)
        self._attribute_list = QListWidget(self)
        self._attribute_list.setSelectionMode(QListWidget.SelectionMode.SingleSelection)
        self._attribute_list.setMinimumHeight(120)
        self._order_attr_cb = QCheckBox(i18n.t("Order by relevance to subgroups"))
        variable_layout.addWidget(self._attribute_list)
        variable_layout.addWidget(self._order_attr_cb)
        sidebar.addWidget(variable_box)

        subgroup_box = QGroupBox(i18n.t("Subgroups"))
        subgroup_layout = QVBoxLayout(subgroup_box)
        self._group_list = QListWidget(self)
        self._group_list.setSelectionMode(QListWidget.SelectionMode.SingleSelection)
        self._group_list.setMinimumHeight(120)
        self._order_group_cb = QCheckBox(i18n.t("Order by relevance to variable"))
        subgroup_layout.addWidget(self._group_list)
        subgroup_layout.addWidget(self._order_group_cb)
        sidebar.addWidget(subgroup_box)

        self._display_box = QGroupBox(i18n.t("Display"))
        display_layout = QVBoxLayout(self._display_box)
        self._show_annotations_cb = QCheckBox(i18n.t("Annotate"))
        self._show_annotations_cb.setChecked(True)
        display_layout.addWidget(self._show_annotations_cb)
        self._compare_group = QButtonGroup(self)
        self._compare_none_rb = QRadioButton(i18n.t("No comparison"))
        self._compare_medians_rb = QRadioButton(i18n.t("Compare medians"))
        self._compare_means_rb = QRadioButton(i18n.t("Compare means"))
        self._compare_means_rb.setChecked(True)
        for index, button in enumerate((self._compare_none_rb, self._compare_medians_rb, self._compare_means_rb)):
            self._compare_group.addButton(button, index)
            display_layout.addWidget(button)
        sidebar.addWidget(self._display_box)

        self._stretching_box = QGroupBox(i18n.t("Display"))
        stretching_layout = QVBoxLayout(self._stretching_box)
        self._stretched_cb = QCheckBox(i18n.t("Stretch bars"))
        self._stretched_cb.setChecked(True)
        self._show_labels_cb = QCheckBox(i18n.t("Show box labels"))
        self._show_labels_cb.setChecked(True)
        self._sort_freq_cb = QCheckBox(i18n.t("Sort by subgroup frequencies"))
        stretching_layout.addWidget(self._stretched_cb)
        stretching_layout.addWidget(self._show_labels_cb)
        stretching_layout.addWidget(self._sort_freq_cb)
        sidebar.addWidget(self._stretching_box)

        self._status_label = QLabel("")
        self._status_label.setWordWrap(True)
        sidebar.addWidget(self._status_label)

        self._clear_button = QPushButton(i18n.t("Clear Selection"))
        self._clear_button.clicked.connect(self._clear_selection)
        sidebar.addWidget(self._clear_button)

        self._selection_label = QLabel(i18n.t("Selected: 0"))
        sidebar.addWidget(self._selection_label)
        sidebar.addStretch(1)

        self._canvas = _BoxPlotCanvas(self)
        self._canvas.selectionChanged.connect(self._handle_selection_changed)
        root.addWidget(self._canvas, 1)
        self._canvas.setMinimumWidth(520)

        self._attribute_list.currentItemChanged.connect(self._handle_attribute_changed)
        self._group_list.currentItemChanged.connect(self._handle_group_changed)
        self._order_attr_cb.toggled.connect(self._handle_order_toggle)
        self._order_group_cb.toggled.connect(self._handle_order_toggle)
        self._show_annotations_cb.toggled.connect(self._refresh_plot)
        self._compare_group.idClicked.connect(lambda _index: self._refresh_plot())
        self._stretched_cb.toggled.connect(self._refresh_plot)
        self._show_labels_cb.toggled.connect(self._refresh_plot)
        self._sort_freq_cb.toggled.connect(self._refresh_plot)

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
            "attribute": self._current_attribute_name(),
            "group": self._current_group_name(),
            "order_by_importance": self._order_attr_cb.isChecked(),
            "order_grouping_by_importance": self._order_group_cb.isChecked(),
            "show_annotations": self._show_annotations_cb.isChecked(),
            "compare_mode": self._current_compare_mode(),
            "stretched": self._stretched_cb.isChecked(),
            "show_labels": self._show_labels_cb.isChecked(),
            "sort_freqs": self._sort_freq_cb.isChecked(),
            "selected_rows": list(self._pending_selected_rows),
        }

    def restore_node_state(self, payload: dict[str, object]) -> None:
        self._pending_selected_rows = [
            int(index)
            for index in payload.get("selected_rows", [])
            if isinstance(index, int | float)
        ]
        self._order_attr_cb.setChecked(bool(payload.get("order_by_importance", False)))
        self._order_group_cb.setChecked(bool(payload.get("order_grouping_by_importance", False)))
        self._show_annotations_cb.setChecked(bool(payload.get("show_annotations", True)))
        self._set_compare_mode(int(payload.get("compare_mode", self.CompareMeans)))
        self._stretched_cb.setChecked(bool(payload.get("stretched", True)))
        self._show_labels_cb.setChecked(bool(payload.get("show_labels", True)))
        self._sort_freq_cb.setChecked(bool(payload.get("sort_freqs", False)))
        self._select_list_text(self._attribute_list, str(payload.get("attribute", "")))
        self._select_list_text(self._group_list, str(payload.get("group", i18n.t("None"))))

    def documentation_url(self) -> str:
        return "https://orangedatamining.com/widget-catalog/visualize/boxplot/"

    def _sync_controls(self) -> None:
        dataset = self._dataset
        current_attr = self._current_attribute_name()
        current_group = self._current_group_name()

        if dataset is None:
            self._populate_list(self._attribute_list, [])
            self._populate_list(self._group_list, [i18n.t("None")])
            self._display_box.hide()
            self._stretching_box.hide()
            return

        attributes = self._sorted_attribute_names(current_group)
        groups = self._sorted_group_names(current_attr)
        default_attr = self._default_attribute_name(attributes)
        default_group = self._default_group_name(groups)
        self._populate_list(self._attribute_list, attributes, current_attr or default_attr)
        self._populate_list(self._group_list, groups, current_group or default_group)
        self._update_display_boxes()

    def _refresh_plot(self) -> None:
        dataset = self._dataset
        self._selected_dataset = None
        self._annotated_dataset = None
        if dataset is None:
            self._status_label.setText(i18n.t("Load data and choose a variable."))
            self._canvas.set_numeric_state(
                None,
                [],
                show_annotations=self._show_annotations_cb.isChecked(),
                compare_mode=self._current_compare_mode(),
                stat_text="",
            )
            self._handle_selection_changed([])
            return

        attr_name = self._current_attribute_name()
        group_name = self._current_group_name()
        attribute = prepared_column(dataset, attr_name)
        group = None if group_name == i18n.t("None") else prepared_column(dataset, group_name)
        if attribute is None:
            self._status_label.setText(i18n.t("Choose a variable."))
            self._canvas.set_numeric_state(
                None,
                [],
                show_annotations=self._show_annotations_cb.isChecked(),
                compare_mode=self._current_compare_mode(),
                stat_text="",
            )
            self._handle_selection_changed([])
            return

        self._update_display_boxes()

        if attribute.is_discrete:
            rows = self._build_discrete_rows(attribute, group)
            stat = self._discrete_stat_summary(attribute, group)
            self._canvas.set_discrete_state(
                attribute,
                rows,
                stretched=self._show_stretched(),
                show_labels=self._show_labels_cb.isChecked(),
                stat_text=stat,
            )
            status = i18n.tf("Rows: {count}", count=len(rows))
            if stat:
                status += f" | {stat}"
        else:
            entries = self._build_numeric_entries(attribute, group)
            compare_mode = self._current_compare_mode()
            if compare_mode == self.CompareMedians:
                entries.sort(key=lambda entry: entry.median)
            elif compare_mode == self.CompareMeans:
                entries.sort(key=lambda entry: entry.mean)
            stat = self._numeric_stat_summary(entries)
            self._canvas.set_numeric_state(
                attribute,
                entries,
                show_annotations=self._show_annotations_cb.isChecked(),
                compare_mode=compare_mode,
                stat_text=stat,
            )
            status = i18n.tf("Groups: {count}", count=len(entries))
            if stat:
                status += f" | {stat}"

        self._status_label.setText(status)
        self._handle_selection_changed(self._pending_selected_rows)

    def _build_numeric_entries(self, attribute: PlotColumn, group: PlotColumn | None) -> list[_NumericEntry]:
        if group is None:
            return [self._numeric_entry(i18n.t("All data"), "", attribute.values, attribute.row_indices, QColor("#3388ff"))]

        grouped_rows: dict[str, list[int]] = {}
        grouped_values: dict[str, list[float]] = {}
        group_lookup = {int(row): str(raw) for row, raw in zip(group.row_indices, group.raw_values)}
        for row, value in zip(attribute.row_indices, attribute.values):
            row_index = int(row)
            group_label = group_lookup.get(row_index)
            if group_label is None:
                continue
            grouped_rows.setdefault(group_label, []).append(row_index)
            grouped_values.setdefault(group_label, []).append(float(value))

        entries = []
        order = list(group.categories) if group.categories else list(grouped_values)
        for index, label in enumerate(order):
            if label not in grouped_values or not grouped_values[label]:
                continue
            values = np.asarray(grouped_values[label], dtype=float)
            row_indices = np.asarray(grouped_rows[label], dtype=int)
            entries.append(
                self._numeric_entry(label, label, values, row_indices, QColor(PALETTE[index % len(PALETTE)]))
            )
        return entries

    def _build_discrete_rows(self, attribute: PlotColumn, group: PlotColumn | None) -> list[_DiscreteRow]:
        attr_lookup = {int(row): str(raw) for row, raw in zip(attribute.row_indices, attribute.raw_values)}
        categories = list(attribute.categories) if attribute.categories else list(dict.fromkeys(attr_lookup.values()))
        color_map = {label: QColor(PALETTE[index % len(PALETTE)]) for index, label in enumerate(categories)}

        if group is None:
            rows_by_value = {label: [] for label in categories}
            for row, value in attr_lookup.items():
                rows_by_value.setdefault(value, []).append(row)
            segments = [
                _DiscreteSegment(
                    label=value,
                    group_value="",
                    count=len(rows_by_value.get(value, [])),
                    row_indices=tuple(rows_by_value.get(value, [])),
                    color=color_map.get(value, QColor(PALETTE[0])),
                )
                for value in categories
                if rows_by_value.get(value)
            ]
            total = sum(segment.count for segment in segments)
            return [_DiscreteRow(label="", group_value="", total_count=total, segments=tuple(segments))]

        group_lookup = {int(row): str(raw) for row, raw in zip(group.row_indices, group.raw_values)}
        rows: list[_DiscreteRow] = []
        group_order = list(group.categories) if group.categories else list(dict.fromkeys(group_lookup.values()))
        for group_label in group_order:
            members = [row for row, value in group_lookup.items() if value == group_label and row in attr_lookup]
            if not members:
                continue
            segments: list[_DiscreteSegment] = []
            for value in categories:
                matched = tuple(row for row in members if attr_lookup[row] == value)
                if not matched:
                    continue
                segments.append(
                    _DiscreteSegment(
                        label=value,
                        group_value=group_label,
                        count=len(matched),
                        row_indices=matched,
                        color=color_map.get(value, QColor(PALETTE[0])),
                    )
                )
            total = sum(segment.count for segment in segments)
            rows.append(_DiscreteRow(label=group_label, group_value=group_label, total_count=total, segments=tuple(segments)))

        if self._sort_freq_cb.isChecked():
            rows.sort(key=lambda row: row.total_count, reverse=True)
        return rows

    @staticmethod
    def _numeric_entry(
        label: str,
        group_value: str,
        values: np.ndarray,
        row_indices: np.ndarray,
        color: QColor,
    ) -> _NumericEntry:
        q1, median, q3 = np.quantile(values, [0.25, 0.5, 0.75], method="midpoint")
        mask = (values >= q1) & (values <= q3)
        selected_rows = tuple(int(index) for index in row_indices[mask])
        if not selected_rows:
            selected_rows = tuple(int(index) for index in row_indices)
        return _NumericEntry(
            label=label,
            group_value=group_value,
            a_min=float(np.min(values)),
            a_max=float(np.max(values)),
            q1=float(q1),
            median=float(median),
            q3=float(q3),
            mean=float(np.mean(values)),
            dev=float(np.std(values)),
            row_indices=selected_rows,
            color=color,
        )

    def _numeric_stat_summary(self, entries: list[_NumericEntry]) -> str:
        compare = self._current_compare_mode()
        if compare == self.CompareNone or len(entries) < 2:
            return ""
        if compare == self.CompareMedians:
            return ""
        samples = [np.asarray([float(value) for value in self._values_for_rows(entry.row_indices)], dtype=float) for entry in entries]
        samples = [sample for sample in samples if len(sample) > 0]
        if len(samples) < 2:
            return ""
        try:
            if len(samples) == 2 and all(len(sample) > 1 for sample in samples):
                stat, p_value = ttest_ind(samples[0], samples[1], equal_var=False)
                if np.isnan(p_value):
                    return ""
                return f"Student's t: {stat:.3f} (p={p_value:.3f}, N={sum(len(sample) for sample in samples)})"
            stat, p_value = f_oneway(*samples)
            if np.isnan(p_value):
                return ""
            return f"ANOVA: {stat:.3f} (p={p_value:.3f}, N={sum(len(sample) for sample in samples)})"
        except Exception:
            return ""

    def _discrete_stat_summary(self, attribute: PlotColumn, group: PlotColumn | None) -> str:
        if group is None:
            return ""
        attr_lookup = {int(row): str(raw) for row, raw in zip(attribute.row_indices, attribute.raw_values)}
        group_lookup = {int(row): str(raw) for row, raw in zip(group.row_indices, group.raw_values)}
        attr_categories = list(attribute.categories) if attribute.categories else list(dict.fromkeys(attr_lookup.values()))
        group_categories = list(group.categories) if group.categories else list(dict.fromkeys(group_lookup.values()))
        matrix = []
        for group_label in group_categories:
            counts = [
                sum(1 for row, attr_value in attr_lookup.items() if attr_value == value and group_lookup.get(row) == group_label)
                for value in attr_categories
            ]
            if sum(counts) > 0:
                matrix.append(counts)
        if len(matrix) < 2:
            return ""
        matrix_np = np.asarray(matrix, dtype=float)
        if matrix_np.shape[1] < 2:
            return ""
        try:
            chi2, p_value, dof, _expected = chi2_contingency(matrix_np)
            if np.isnan(p_value):
                return ""
            return f"chi2: {chi2:.2f} (p={p_value:.3f}, dof={dof})"
        except Exception:
            return ""

    def _sorted_attribute_names(self, current_group: str) -> list[str]:
        dataset = self._dataset
        attributes = primitive_columns(dataset)
        if dataset is None or not self._order_attr_cb.isChecked() or current_group == i18n.t("None"):
            return attributes
        base_order = {name: index for index, name in enumerate(attributes)}
        return sorted(
            attributes,
            key=lambda name: (self._attribute_importance_score(name, current_group), base_order[name]),
        )

    def _sorted_group_names(self, current_attr: str) -> list[str]:
        dataset = self._dataset
        groups = [i18n.t("None"), *discrete_columns(dataset)]
        if dataset is None or not self._order_group_cb.isChecked() or not current_attr:
            return groups
        base_order = {name: index for index, name in enumerate(groups)}
        ordered = [i18n.t("None")]
        others = sorted(
            [name for name in groups if name != i18n.t("None")],
            key=lambda name: (self._group_importance_score(name, current_attr), base_order[name]),
        )
        ordered.extend(others)
        return ordered

    def _attribute_importance_score(self, attr_name: str, group_name: str) -> float:
        dataset = self._dataset
        if dataset is None or group_name == i18n.t("None"):
            return float("inf")
        attribute = prepared_column(dataset, attr_name)
        group = prepared_column(dataset, group_name)
        if attribute is None or group is None or attr_name == group_name:
            return float("inf")
        if attribute.is_discrete:
            return self._chi_square_p(attribute, group)
        return self._anova_p(attribute, group)

    def _group_importance_score(self, group_name: str, attr_name: str) -> float:
        dataset = self._dataset
        if dataset is None:
            return float("inf")
        attribute = prepared_column(dataset, attr_name)
        group = prepared_column(dataset, group_name)
        if attribute is None or group is None or group_name == attr_name:
            return float("inf")
        if attribute.is_discrete:
            return self._chi_square_p(attribute, group)
        return self._anova_p(attribute, group)

    @staticmethod
    def _chi_square_p(attribute: PlotColumn, group: PlotColumn) -> float:
        attr_lookup = {int(row): str(raw) for row, raw in zip(attribute.row_indices, attribute.raw_values)}
        group_lookup = {int(row): str(raw) for row, raw in zip(group.row_indices, group.raw_values)}
        attr_categories = list(attribute.categories) if attribute.categories else list(dict.fromkeys(attr_lookup.values()))
        group_categories = list(group.categories) if group.categories else list(dict.fromkeys(group_lookup.values()))
        matrix = []
        for group_label in group_categories:
            counts = [
                sum(1 for row, value in attr_lookup.items() if value == attr_label and group_lookup.get(row) == group_label)
                for attr_label in attr_categories
            ]
            if sum(counts) > 0:
                matrix.append(counts)
        if len(matrix) < 2:
            return float("inf")
        matrix_np = np.asarray(matrix, dtype=float)
        if matrix_np.shape[1] < 2:
            return float("inf")
        try:
            _chi2, p_value, _dof, _expected = chi2_contingency(matrix_np)
            return float(p_value) if np.isfinite(p_value) else float("inf")
        except Exception:
            return float("inf")

    @staticmethod
    def _anova_p(attribute: PlotColumn, group: PlotColumn) -> float:
        value_lookup = {int(row): float(value) for row, value in zip(attribute.row_indices, attribute.values)}
        group_lookup = {int(row): str(raw) for row, raw in zip(group.row_indices, group.raw_values)}
        categories = list(group.categories) if group.categories else list(dict.fromkeys(group_lookup.values()))
        samples = []
        for label in categories:
            sample = [value_lookup[row] for row, group_label in group_lookup.items() if group_label == label and row in value_lookup]
            sample = [value for value in sample if np.isfinite(value)]
            if len(sample) > 1:
                samples.append(np.asarray(sample, dtype=float))
        if len(samples) < 2:
            return float("inf")
        try:
            _stat, p_value = f_oneway(*samples)
            return float(p_value) if np.isfinite(p_value) else float("inf")
        except Exception:
            return float("inf")

    def _handle_attribute_changed(self, _current: QListWidgetItem | None, _previous: QListWidgetItem | None) -> None:
        if self._order_group_cb.isChecked():
            self._sync_controls()
        self._refresh_plot()

    def _handle_group_changed(self, _current: QListWidgetItem | None, _previous: QListWidgetItem | None) -> None:
        if self._order_attr_cb.isChecked():
            self._sync_controls()
        self._refresh_plot()

    def _handle_order_toggle(self, _checked: bool) -> None:
        self._sync_controls()
        self._refresh_plot()

    def _update_display_boxes(self) -> None:
        dataset = self._dataset
        attribute = prepared_column(dataset, self._current_attribute_name()) if dataset is not None else None
        if attribute is None:
            self._display_box.hide()
            self._stretching_box.hide()
            return
        is_continuous = not attribute.is_discrete
        self._display_box.setVisible(is_continuous)
        self._stretching_box.setVisible(not is_continuous)
        has_group = self._current_group_name() != i18n.t("None")
        self._compare_none_rb.setEnabled(has_group)
        self._compare_medians_rb.setEnabled(has_group)
        self._compare_means_rb.setEnabled(has_group)
        self._stretched_cb.setEnabled(self._current_group_name() not in {i18n.t("None"), self._current_attribute_name()})
        self._sort_freq_cb.setEnabled(has_group)

    def _handle_selection_changed(self, rows: list[int]) -> None:
        self._pending_selected_rows = sorted({index for index in rows})
        self._canvas.set_selected_rows(self._pending_selected_rows)
        self._selection_label.setText(i18n.tf("Selected: {count}", count=len(self._pending_selected_rows)))
        self._selected_dataset, self._annotated_dataset = build_selection_outputs(
            self._dataset,
            self._pending_selected_rows,
            generated_by="box-plot",
            service=self._builder,
        )
        self._notify_output_changed()

    def _clear_selection(self) -> None:
        self._canvas.clear_selection()

    def _values_for_rows(self, rows: tuple[int, ...]) -> list[float]:
        dataset = self._dataset
        attr_name = self._current_attribute_name()
        attribute = prepared_column(dataset, attr_name) if dataset is not None else None
        if attribute is None:
            return []
        lookup = {int(row): float(value) for row, value in zip(attribute.row_indices, attribute.values)}
        return [lookup[row] for row in rows if row in lookup]

    def _default_attribute_name(self, attributes: list[str]) -> str:
        dataset = self._dataset
        if dataset is None or not attributes:
            return ""
        target_names = {column.name for column in dataset.domain.target_columns}
        for name in attributes:
            if name not in target_names:
                return name
        return attributes[0]

    def _default_group_name(self, groups: list[str]) -> str:
        dataset = self._dataset
        none_text = i18n.t("None")
        if dataset is None:
            return none_text
        target_names = {column.name for column in dataset.domain.target_columns if column.logical_type != "numeric"}
        for name in groups:
            if name in target_names:
                return name
        return none_text

    def _current_attribute_name(self) -> str:
        item = self._attribute_list.currentItem()
        return item.text() if item is not None else ""

    def _current_group_name(self) -> str:
        item = self._group_list.currentItem()
        return item.text() if item is not None else i18n.t("None")

    def _current_compare_mode(self) -> int:
        if self._compare_medians_rb.isChecked():
            return self.CompareMedians
        if self._compare_means_rb.isChecked():
            return self.CompareMeans
        return self.CompareNone

    def _set_compare_mode(self, mode: int) -> None:
        if mode == self.CompareMedians:
            self._compare_medians_rb.setChecked(True)
        elif mode == self.CompareMeans:
            self._compare_means_rb.setChecked(True)
        else:
            self._compare_none_rb.setChecked(True)

    def _show_stretched(self) -> bool:
        return self._stretched_cb.isChecked() and self._current_group_name() != self._current_attribute_name()

    @staticmethod
    def _populate_list(widget: QListWidget, names: list[str], wanted: str | None = None) -> None:
        widget.blockSignals(True)
        widget.clear()
        for name in names:
            widget.addItem(name)
        target = wanted or (names[0] if names else "")
        for row in range(widget.count()):
            if widget.item(row).text() == target:
                widget.setCurrentRow(row)
                break
        if widget.currentItem() is None and widget.count():
            widget.setCurrentRow(0)
        widget.blockSignals(False)

    @staticmethod
    def _select_list_text(widget: QListWidget, value: str) -> None:
        if not value:
            return
        for row in range(widget.count()):
            if widget.item(row).text() == value:
                widget.setCurrentRow(row)
                return
