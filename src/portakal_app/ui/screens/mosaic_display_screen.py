from __future__ import annotations

import itertools
import math
from dataclasses import dataclass

import numpy as np
from PySide6.QtCore import QRect, Qt, Signal, QSize
from PySide6.QtGui import QColor, QFont, QFontMetrics, QPainter, QPen
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QToolTip,
    QVBoxLayout,
    QWidget,
)

from portakal_app.data.models import DatasetHandle
from portakal_app.data.services.generated_dataset_service import GeneratedDatasetService
from portakal_app.models import WorkflowPayload
from portakal_app.ui.screens.node_screen import WorkflowNodeScreenSupport
from portakal_app.ui.screens.visualize_common import (
    PALETTE,
    build_selection_outputs,
    categorical_candidate_columns,
    categorical_view,
    subset_row_indices,
)


_BLUE_COLORS = (
    QColor(255, 255, 255),
    QColor(210, 210, 255),
    QColor(110, 110, 255),
    QColor(0, 0, 255),
)
_RED_COLORS = (
    QColor(255, 255, 255),
    QColor(255, 200, 200),
    QColor(255, 100, 100),
    QColor(255, 0, 0),
)


def _chi2_sf(value: float, degrees: int) -> float:
    if value <= 0:
        return 1.0
    z_score = ((value / degrees) ** (1 / 3) - (1 - 2 / (9 * degrees))) / math.sqrt(2 / (9 * degrees))
    if z_score > 6:
        return 0.0
    if z_score < -6:
        return 1.0
    return 0.5 * math.erfc(z_score / math.sqrt(2))


def _residual_color(pearson: float) -> QColor:
    if pearson == 0:
        return QColor(220, 220, 220)
    index = max(0, min(int(math.log(abs(pearson), 2)), 3))
    return (_BLUE_COLORS if pearson > 0 else _RED_COLORS)[index]


@dataclass(frozen=True)
class _LabelInfo:
    rect: QRect
    text: str
    bold: bool = False
    vertical: bool = False


@dataclass(frozen=True)
class _BarSegment:
    fraction: float
    color: QColor


@dataclass(frozen=True)
class _LeafInfo:
    rect: QRect
    row_indices: tuple[int, ...]
    tooltip: str
    fill_color: QColor | None = None
    bars: tuple[_BarSegment, ...] = ()
    compare_bars: tuple[_BarSegment, ...] = ()
    subset_bars: tuple[_BarSegment, ...] = ()


class _MosaicCanvas(QWidget):
    selectionChanged = Signal(list)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._leaves: list[_LeafInfo] = []
        self._top_labels: list[_LabelInfo] = []
        self._left_labels: list[_LabelInfo] = []
        self._bottom_labels: list[_LabelInfo] = []
        self._right_labels: list[_LabelInfo] = []
        self._legend_items: list[tuple[str, QColor]] = []
        self._selected_leaf_indices: set[int] = set()
        self._selection_rows: set[int] = set()
        self._title = "Mosaic Display"
        self._hover_index: int | None = None
        self.setMouseTracking(True)
        self.setMinimumSize(620, 440)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

    def set_plot(
        self,
        *,
        leaves: list[_LeafInfo],
        top_labels: list[_LabelInfo],
        left_labels: list[_LabelInfo],
        bottom_labels: list[_LabelInfo],
        right_labels: list[_LabelInfo],
        legend_items: list[tuple[str, QColor]],
        title: str,
    ) -> None:
        self._leaves = leaves
        self._top_labels = top_labels
        self._left_labels = left_labels
        self._bottom_labels = bottom_labels
        self._right_labels = right_labels
        self._legend_items = legend_items
        self._title = title
        self._selected_leaf_indices = set()
        self._selection_rows = set()
        self.update()

    def set_selection_rows(self, rows: list[int]) -> None:
        normalized = set(rows)
        self._selection_rows = normalized
        self._selected_leaf_indices = {
            index
            for index, leaf in enumerate(self._leaves)
            if any(row in normalized for row in leaf.row_indices)
        }
        self.update()

    def clear_selection(self) -> None:
        self._selected_leaf_indices = set()
        self._selection_rows = set()
        self.selectionChanged.emit([])
        self.update()

    def sizeHint(self) -> QSize:
        return QSize(760, 560)

    def mouseMoveEvent(self, event) -> None:
        pos = event.position().toPoint()
        for index, leaf in enumerate(self._leaves):
            if leaf.rect.contains(pos):
                if self._hover_index != index:
                    self._hover_index = index
                    QToolTip.showText(event.globalPosition().toPoint(), leaf.tooltip, self)
                return
        self._hover_index = None
        QToolTip.hideText()

    def leaveEvent(self, _event) -> None:
        self._hover_index = None
        QToolTip.hideText()

    def mousePressEvent(self, event) -> None:
        if event.button() != Qt.MouseButton.LeftButton:
            return
        pos = event.position().toPoint()
        for index, leaf in enumerate(self._leaves):
            if not leaf.rect.contains(pos):
                continue
            if event.modifiers() & Qt.KeyboardModifier.ControlModifier:
                if index in self._selected_leaf_indices:
                    self._selected_leaf_indices.remove(index)
                else:
                    self._selected_leaf_indices.add(index)
            else:
                self._selected_leaf_indices = {index}
            rows = sorted({row for chosen in self._selected_leaf_indices for row in self._leaves[chosen].row_indices})
            self._selection_rows = set(rows)
            self.selectionChanged.emit(rows)
            self.update()
            return
        if not (event.modifiers() & Qt.KeyboardModifier.ControlModifier):
            self.clear_selection()

    def paintEvent(self, _event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), QColor("#fffdf9"))

        if not self._leaves:
            painter.setPen(QColor("#8d877d"))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "No mosaic available.")
            return

        painter.setPen(QColor("#3b2a10"))
        painter.setFont(QFont(self.font().family(), 9, QFont.Weight.Bold))
        painter.drawText(QRect(0, 6, self.width(), 18), Qt.AlignmentFlag.AlignCenter, self._title)

        for index, leaf in enumerate(self._leaves):
            self._draw_leaf(painter, leaf, selected=index in self._selected_leaf_indices)

        self._draw_labels(painter)
        self._draw_legend(painter)
        painter.end()

    def _draw_leaf(self, painter: QPainter, leaf: _LeafInfo, *, selected: bool) -> None:
        rect = leaf.rect
        if leaf.bars:
            self._draw_bar_stack(painter, rect, leaf.bars)
            if leaf.compare_bars and rect.width() > 18:
                compare_rect = QRect(rect.left(), rect.top(), 6, rect.height())
                self._draw_bar_stack(painter, compare_rect, leaf.compare_bars)
                painter.setPen(QPen(QColor("#ffffff"), 1))
                painter.drawLine(compare_rect.right() + 1, rect.top(), compare_rect.right() + 1, rect.bottom())
            if leaf.subset_bars and rect.width() > 18:
                subset_rect = QRect(rect.right() - 5, rect.top(), 6, rect.height())
                self._draw_bar_stack(painter, subset_rect, leaf.subset_bars)
                painter.setPen(QPen(QColor("#ffffff"), 1))
                painter.drawLine(subset_rect.left() - 1, rect.top(), subset_rect.left() - 1, rect.bottom())
        else:
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(leaf.fill_color or QColor("#e5e7eb"))
            painter.drawRect(rect)

        border = QPen(QColor("#111111"), 2.5, Qt.PenStyle.DotLine) if selected else QPen(QColor("#fffdf9"), 1)
        painter.setPen(border)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawRect(rect)

    @staticmethod
    def _draw_bar_stack(painter: QPainter, rect: QRect, bars: tuple[_BarSegment, ...]) -> None:
        painter.setPen(Qt.PenStyle.NoPen)
        top = rect.top()
        remaining = rect.height()
        for index, bar in enumerate(bars):
            height = remaining if index == len(bars) - 1 else max(1, int(round(rect.height() * bar.fraction)))
            remaining -= height
            painter.setBrush(bar.color)
            painter.drawRect(rect.left(), top, rect.width(), height)
            top += height

    def _draw_labels(self, painter: QPainter) -> None:
        for label in (*self._top_labels, *self._bottom_labels, *self._left_labels, *self._right_labels):
            font = QFont(self.font())
            font.setPointSize(8)
            font.setBold(label.bold)
            painter.setFont(font)
            painter.setPen(QColor("#534b40"))
            metrics = QFontMetrics(font)
            if label.vertical:
                painter.save()
                painter.translate(label.rect.center())
                painter.rotate(-90)
                rotated = QRect(-label.rect.height() // 2, -label.rect.width() // 2, label.rect.height(), label.rect.width())
                text = metrics.elidedText(label.text, Qt.TextElideMode.ElideRight, rotated.width())
                painter.drawText(rotated, Qt.AlignmentFlag.AlignCenter, text)
                painter.restore()
            else:
                text = metrics.elidedText(label.text, Qt.TextElideMode.ElideRight, label.rect.width())
                painter.drawText(label.rect, Qt.AlignmentFlag.AlignCenter, text)

    def _draw_legend(self, painter: QPainter) -> None:
        if not self._legend_items:
            return
        left = max(16, self.width() - 172)
        top = 38
        painter.setFont(QFont(self.font().family(), 8))
        painter.setPen(QColor("#534b40"))
        painter.drawText(QRect(left, top, 140, 16), Qt.AlignmentFlag.AlignLeft, "Legend")
        y = top + 22
        for label, color in self._legend_items[:8]:
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(color)
            painter.drawRect(left, y + 2, 12, 12)
            painter.setPen(QColor("#534b40"))
            painter.drawText(QRect(left + 18, y, 140, 16), Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, label)
            y += 20


class MosaicDisplayScreen(QWidget, WorkflowNodeScreenSupport):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._init_workflow_node_support()
        self._dataset: DatasetHandle | None = None
        self._subset: DatasetHandle | None = None
        self._builder = GeneratedDatasetService()
        self._selected_rows: list[int] = []
        self._selected_dataset: DatasetHandle | None = None
        self._annotated_dataset: DatasetHandle | None = None
        self._candidate_columns: list[str] = []

        root = QHBoxLayout(self)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(12)

        left_scroll = QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        left_scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        left_scroll.setFixedWidth(320)
        left_panel = QWidget()
        left_scroll.setWidget(left_panel)
        root.addWidget(left_scroll, 0)

        left = QVBoxLayout(left_panel)
        left.setContentsMargins(0, 0, 0, 0)
        left.setSpacing(10)

        self._variables_box = QGroupBox("Variables")
        vars_layout = QVBoxLayout(self._variables_box)
        vars_layout.setContentsMargins(10, 12, 10, 10)
        vars_layout.setSpacing(6)
        self._var1_combo = QComboBox()
        self._var2_combo = QComboBox()
        self._var3_combo = QComboBox()
        self._var4_combo = QComboBox()
        for combo in (self._var1_combo, self._var2_combo, self._var3_combo, self._var4_combo):
            combo.currentTextChanged.connect(self._refresh)
            vars_layout.addWidget(combo)
        self._vizrank_btn = QPushButton("Find Informative Mosaics")
        self._vizrank_btn.clicked.connect(self._find_informative_mosaic)
        vars_layout.addWidget(self._vizrank_btn)
        left.addWidget(self._variables_box)

        self._interior_box = QGroupBox("Interior Coloring")
        interior_layout = QVBoxLayout(self._interior_box)
        interior_layout.setContentsMargins(10, 12, 10, 10)
        interior_layout.setSpacing(8)
        self._color_combo = QComboBox()
        self._color_combo.currentTextChanged.connect(self._refresh)
        self._compare_total_cb = QCheckBox("Compare with total")
        self._compare_total_cb.toggled.connect(self._refresh)
        interior_layout.addWidget(self._color_combo)
        interior_layout.addWidget(self._compare_total_cb)
        left.addWidget(self._interior_box)

        self._status_label = QLabel("Load data to build a mosaic display.")
        self._status_label.setWordWrap(True)
        left.addWidget(self._status_label)
        self._selection_label = QLabel("Selected: 0")
        left.addWidget(self._selection_label)
        self._clear_btn = QPushButton("Clear Selection")
        self._clear_btn.clicked.connect(self._clear_selection)
        left.addWidget(self._clear_btn)
        left.addStretch(1)

        right = QVBoxLayout()
        right.setContentsMargins(0, 0, 0, 0)
        self._canvas = _MosaicCanvas()
        self._canvas.selectionChanged.connect(self._handle_selection_changed)
        right.addWidget(self._canvas, 1)
        root.addLayout(right, 1)

    def documentation_url(self) -> str:
        return "https://orangedatamining.com/widget-catalog/visualize/mosaicdisplay/"

    def sizeHint(self) -> QSize:
        return QSize(980, 640)

    def set_input_payload(self, payload) -> None:
        if payload is None:
            self._dataset = None
            self._subset = None
        elif payload.port_label == "Data":
            self._dataset = payload.dataset
        elif payload.port_label == "Data Subset":
            self._subset = payload.dataset
        self._sync_controls()
        self._refresh()

    def current_output_dataset(self) -> DatasetHandle | None:
        return self._selected_dataset

    def current_output_datasets(self) -> dict[str, DatasetHandle | None] | None:
        return {
            "Selected Data": self._selected_dataset,
            "Annotated Data": self._annotated_dataset,
        }

    def current_output_payloads(self) -> dict[str, WorkflowPayload | None] | None:
        return {
            "Selected Data": None if self._selected_dataset is None else WorkflowPayload("Selected Data", self._selected_dataset),
            "Annotated Data": None if self._annotated_dataset is None else WorkflowPayload("Annotated Data", self._annotated_dataset),
        }

    def serialize_node_state(self) -> dict[str, object]:
        return {
            "variable1": self._var1_combo.currentText(),
            "variable2": self._var2_combo.currentText(),
            "variable3": self._var3_combo.currentText(),
            "variable4": self._var4_combo.currentText(),
            "color": self._color_combo.currentText(),
            "compare_total": self._compare_total_cb.isChecked(),
            "selected_rows": list(self._selected_rows),
        }

    def restore_node_state(self, payload: dict[str, object]) -> None:
        for combo, key in (
            (self._var1_combo, "variable1"),
            (self._var2_combo, "variable2"),
            (self._var3_combo, "variable3"),
            (self._var4_combo, "variable4"),
            (self._color_combo, "color"),
        ):
            value = str(payload.get(key, ""))
            index = combo.findText(value)
            if index >= 0:
                combo.setCurrentIndex(index)
        self._compare_total_cb.setChecked(bool(payload.get("compare_total", False)))
        self._selected_rows = sorted({int(value) for value in payload.get("selected_rows", []) if isinstance(value, (int, float))})

    def _sync_controls(self) -> None:
        dataset = self._dataset
        self._candidate_columns = categorical_candidate_columns(dataset)
        self._populate_combo(self._var1_combo, self._candidate_columns, allow_none=False)
        self._populate_combo(self._var2_combo, self._candidate_columns, allow_none=True)
        self._populate_combo(self._var3_combo, self._candidate_columns, allow_none=True)
        self._populate_combo(self._var4_combo, self._candidate_columns, allow_none=True)
        self._populate_combo(self._color_combo, ["(Pearson residuals)", *self._candidate_columns], allow_none=False)

        if self._candidate_columns and self._var1_combo.currentText() not in self._candidate_columns:
            self._var1_combo.setCurrentIndex(0)
        if self._candidate_columns and self._var2_combo.currentText() in {"", "(None)"} and len(self._candidate_columns) > 1:
            self._var2_combo.setCurrentText(self._candidate_columns[1])
        if dataset is not None and self._color_combo.currentText() == "(Pearson residuals)":
            for column in dataset.domain.target_columns:
                if column.name in self._candidate_columns:
                    self._color_combo.setCurrentText(column.name)
                    break
        self._vizrank_btn.setEnabled(bool(self._candidate_columns))

    def _populate_combo(self, combo: QComboBox, values: list[str], *, allow_none: bool) -> None:
        current = combo.currentText()
        items = list(dict.fromkeys(values))
        if allow_none and "(None)" not in items:
            items = ["(None)", *items]
        combo.blockSignals(True)
        combo.clear()
        combo.addItems(items)
        if current in items:
            combo.setCurrentText(current)
        elif combo is self._var2_combo and len(values) > 1:
            combo.setCurrentText(values[1])
        elif items:
            combo.setCurrentIndex(0)
        combo.blockSignals(False)

    def _active_variables(self) -> list[str]:
        result: list[str] = []
        for combo in (self._var1_combo, self._var2_combo, self._var3_combo, self._var4_combo):
            value = combo.currentText().strip()
            if not value or value == "(None)" or value in result:
                continue
            result.append(value)
        return result

    @staticmethod
    def _count_labels(labels: tuple[str, ...]) -> dict[str, int]:
        counts: dict[str, int] = {}
        for label in labels:
            counts[label] = counts.get(label, 0) + 1
        return counts

    @staticmethod
    def _count_labels_for_rows(labels: tuple[str, ...], rows: np.ndarray) -> dict[str, int]:
        counts: dict[str, int] = {}
        for row in rows.tolist():
            label = labels[int(row)]
            counts[label] = counts.get(label, 0) + 1
        return counts

    @staticmethod
    def _segments_from_counts(counts: dict[str, int], ordered_labels: tuple[str, ...]) -> tuple[_BarSegment, ...]:
        total = sum(counts.get(label, 0) for label in ordered_labels)
        if total <= 0:
            return ()
        return tuple(
            _BarSegment(counts.get(label, 0) / total, QColor(PALETTE[index % len(PALETTE)]))
            for index, label in enumerate(ordered_labels)
            if counts.get(label, 0) > 0
        )

    def _refresh(self) -> None:
        dataset = self._dataset
        if dataset is None:
            self._canvas.set_plot(
                leaves=[],
                top_labels=[],
                left_labels=[],
                bottom_labels=[],
                right_labels=[],
                legend_items=[],
                title="Mosaic Display",
            )
            self._status_label.setText("Load data to build a mosaic display.")
            self._handle_selection_changed([])
            return

        variables = self._active_variables()
        color_variable = self._color_combo.currentText()
        color_mode = color_variable != "(Pearson residuals)"
        self._compare_total_cb.setEnabled(color_mode)
        if not color_mode and self._compare_total_cb.isChecked():
            self._compare_total_cb.setChecked(False)
            return

        if len(variables) < (1 if color_mode else 2):
            self._canvas.set_plot(
                leaves=[],
                top_labels=[],
                left_labels=[],
                bottom_labels=[],
                right_labels=[],
                legend_items=[],
                title="Mosaic Display",
            )
            self._status_label.setText("Select enough variables for the current interior-coloring mode.")
            self._handle_selection_changed([])
            return

        attr_views = [categorical_view(dataset, name, bins=4, discretize_numeric=True) for name in variables]
        if any(view is None for view in attr_views):
            self._status_label.setText("Selected variables are not available.")
            return
        attr_views = [view for view in attr_views if view is not None]
        color_view = categorical_view(dataset, color_variable, bins=4, discretize_numeric=True) if color_mode else None
        subset_rows = subset_row_indices(dataset, self._subset)
        marginals = {view.name: self._count_labels(view.labels) for view in attr_views}
        color_totals = self._count_labels(color_view.labels) if color_view is not None else {}

        chart_rect = QRect(118, 54, max(220, self._canvas.width() - 250), max(220, self._canvas.height() - 120))
        top_labels = [_LabelInfo(QRect(chart_rect.left(), 8, chart_rect.width(), 18), variables[0], bold=True)]
        left_labels = [_LabelInfo(QRect(8, chart_rect.top(), 18, chart_rect.height()), variables[1], bold=True, vertical=True)] if len(variables) >= 2 else []
        bottom_labels = [_LabelInfo(QRect(chart_rect.left(), chart_rect.bottom() + 22, chart_rect.width(), 18), variables[2], bold=True)] if len(variables) >= 3 else []
        right_labels = [_LabelInfo(QRect(chart_rect.right() + 22, chart_rect.top(), 18, chart_rect.height()), variables[3], bold=True, vertical=True)] if len(variables) >= 4 else []

        leaves: list[_LeafInfo] = []
        self._layout_recursive(
            chart_rect,
            np.arange(dataset.row_count, dtype=int),
            attr_views,
            0,
            (),
            top_labels,
            left_labels,
            bottom_labels,
            right_labels,
            leaves,
            marginals,
            subset_rows,
            color_view,
            color_totals,
        )

        legend_items = (
            [
                ("<-8", _RED_COLORS[3]),
                ("-8:-4", _RED_COLORS[2]),
                ("-4:-2", _RED_COLORS[1]),
                ("-2:2", QColor(220, 220, 220)),
                ("2:4", _BLUE_COLORS[1]),
                ("4:8", _BLUE_COLORS[2]),
                (">8", _BLUE_COLORS[3]),
            ]
            if color_view is None
            else [(label, QColor(PALETTE[index % len(PALETTE)])) for index, label in enumerate(color_view.categories)]
        )
        self._canvas.set_plot(
            leaves=leaves,
            top_labels=top_labels,
            left_labels=left_labels,
            bottom_labels=bottom_labels,
            right_labels=right_labels,
            legend_items=legend_items,
            title="Mosaic Display",
        )
        if self._selected_rows:
            self._canvas.set_selection_rows(self._selected_rows)

        chi2, p_value = self._score_combination(variables, color_variable)
        p_text = "< 0.0001" if p_value < 0.0001 else f"{p_value:.4f}"
        self._status_label.setText(f"Variables: {', '.join(variables)}  |  N = {dataset.row_count}  |  chi2 = {chi2:.2f}  |  p = {p_text}")
        self._handle_selection_changed(self._selected_rows)

    def _layout_recursive(
        self,
        rect: QRect,
        rows: np.ndarray,
        attr_views: list,
        depth: int,
        conditions: tuple[tuple[str, str], ...],
        top_labels: list[_LabelInfo],
        left_labels: list[_LabelInfo],
        bottom_labels: list[_LabelInfo],
        right_labels: list[_LabelInfo],
        leaves: list[_LeafInfo],
        marginals: dict[str, dict[str, int]],
        subset_rows: set[int],
        color_view,
        color_totals: dict[str, int],
    ) -> None:
        if depth >= len(attr_views):
            leaves.append(self._build_leaf(rect, rows, conditions, marginals, subset_rows, color_view, color_totals))
            return

        view = attr_views[depth]
        grouped: list[tuple[str, np.ndarray]] = []
        for category in view.categories:
            matched = np.asarray([row for row in rows.tolist() if view.labels[int(row)] == category], dtype=int)
            if matched.size:
                grouped.append((category, matched))
        if not grouped:
            return

        total = sum(len(group_rows) for _, group_rows in grouped)
        horizontal = depth % 2 == 0
        gap = 4 * max(1, len(attr_views) - depth - 1) if len(grouped) > 1 else 0
        available = (rect.width() if horizontal else rect.height()) - gap * (len(grouped) - 1)
        cursor = rect.left() if horizontal else rect.top()
        for index, (category, matched_rows) in enumerate(grouped):
            segment = (
                (rect.right() + 1 - cursor) if horizontal else (rect.bottom() + 1 - cursor)
            ) if index == len(grouped) - 1 else max(1, int(round(available * (len(matched_rows) / total))))
            child_rect = QRect(cursor, rect.top(), segment, rect.height()) if horizontal else QRect(rect.left(), cursor, rect.width(), segment)
            cursor += segment + gap
            self._append_category_label(depth, child_rect, category, top_labels, left_labels, bottom_labels, right_labels)
            self._layout_recursive(
                child_rect,
                matched_rows,
                attr_views,
                depth + 1,
                (*conditions, (view.name, category)),
                top_labels,
                left_labels,
                bottom_labels,
                right_labels,
                leaves,
                marginals,
                subset_rows,
                color_view,
                color_totals,
            )

    def _append_category_label(
        self,
        depth: int,
        rect: QRect,
        text: str,
        top_labels: list[_LabelInfo],
        left_labels: list[_LabelInfo],
        bottom_labels: list[_LabelInfo],
        right_labels: list[_LabelInfo],
    ) -> None:
        if depth == 0:
            top_labels.append(_LabelInfo(QRect(rect.left(), 28, rect.width(), 18), text))
        elif depth == 1:
            left_labels.append(_LabelInfo(QRect(30, rect.top(), 78, rect.height()), text))
        elif depth == 2:
            bottom_labels.append(_LabelInfo(QRect(rect.left(), rect.bottom() + 4, rect.width(), 18), text))
        elif depth == 3:
            right_labels.append(_LabelInfo(QRect(rect.right() + 4, rect.top(), 92, rect.height()), text))

    def _build_leaf(
        self,
        rect: QRect,
        rows: np.ndarray,
        conditions: tuple[tuple[str, str], ...],
        marginals: dict[str, dict[str, int]],
        subset_rows: set[int],
        color_view,
        color_totals: dict[str, int],
    ) -> _LeafInfo:
        dataset = self._dataset
        assert dataset is not None
        count = len(rows)
        header = "<br/>".join(f"<b>{name}</b>: {value}" for name, value in conditions)
        if color_view is None:
            expected = float(dataset.row_count)
            for name, value in conditions:
                expected *= marginals[name].get(value, 0) / max(1, dataset.row_count)
            pearson = (count - expected) / math.sqrt(expected) if expected > 0 else 0.0
            return _LeafInfo(
                rect=rect,
                row_indices=tuple(int(row) for row in rows.tolist()),
                tooltip=(
                    f"{header}<hr/>Expected instances: {expected:.1f}<br>"
                    f"Actual instances: {count}<br>"
                    f"Standardized (Pearson) residual: {pearson:.1f}"
                ),
                fill_color=_residual_color(pearson),
            )

        observed = self._count_labels_for_rows(color_view.labels, rows)
        bars = self._segments_from_counts(observed, color_view.categories)
        compare_bars = self._segments_from_counts(color_totals, color_view.categories) if self._compare_total_cb.isChecked() else ()
        subset_counts = self._count_labels_for_rows(
            color_view.labels,
            np.asarray([row for row in rows.tolist() if int(row) in subset_rows], dtype=int),
        )
        subset_bars = self._segments_from_counts(subset_counts, color_view.categories) if subset_counts else ()
        detail_lines = []
        total_count = max(1, count)
        for label in color_view.categories:
            actual = observed.get(label, 0)
            prior = color_totals.get(label, 0)
            expected = total_count * prior / max(1, dataset.row_count)
            detail_lines.append(f"<b>{label}</b>: {actual} / {100.0 * actual / total_count:.1f}% (Expected {expected:.1f})")
        return _LeafInfo(
            rect=rect,
            row_indices=tuple(int(row) for row in rows.tolist()),
            tooltip=f"{header}<hr/>Instances: {count}<br><br>{'<br/>'.join(detail_lines)}",
            bars=bars,
            compare_bars=compare_bars,
            subset_bars=subset_bars,
        )

    def _handle_selection_changed(self, rows: list[int]) -> None:
        dataset = self._dataset
        normalized = sorted({int(row) for row in rows if isinstance(row, (int, float))})
        self._selected_rows = [row for row in normalized if dataset is not None and 0 <= row < dataset.row_count]
        self._selection_label.setText(f"Selected: {len(self._selected_rows)}")
        self._selected_dataset, self._annotated_dataset = build_selection_outputs(
            dataset,
            self._selected_rows,
            generated_by="mosaic-display",
            service=self._builder,
        )
        self._notify_output_changed()

    def _clear_selection(self) -> None:
        self._selected_rows = []
        self._canvas.clear_selection()

    def _find_informative_mosaic(self) -> None:
        candidates = self._candidate_columns[:8]
        if not candidates:
            return
        color_variable = self._color_combo.currentText()
        if color_variable in candidates and color_variable != "(Pearson residuals)":
            candidates = [name for name in candidates if name != color_variable]
        if not candidates:
            return
        lengths = range(1 if color_variable != "(Pearson residuals)" else 2, min(4, len(candidates)) + 1)
        best_combo: tuple[str, ...] | None = None
        best_score: tuple[float, float] | None = None
        for length in lengths:
            for combo in itertools.combinations(candidates, length):
                chi2, p_value = self._score_combination(list(combo), color_variable)
                score = (p_value, -chi2)
                if best_score is None or score < best_score:
                    best_score = score
                    best_combo = combo
        if best_combo is None:
            return
        choices = list(best_combo) + ["(None)"] * (4 - len(best_combo))
        for combo, value in zip((self._var1_combo, self._var2_combo, self._var3_combo, self._var4_combo), choices):
            combo.setCurrentText(value)
        self._refresh()

    def _score_combination(self, variables: list[str], color_variable: str) -> tuple[float, float]:
        dataset = self._dataset
        if dataset is None:
            return 0.0, 1.0
        attr_views = [categorical_view(dataset, name, bins=4, discretize_numeric=True) for name in variables]
        attr_views = [view for view in attr_views if view is not None]
        if not attr_views:
            return 0.0, 1.0
        groups: dict[tuple[str, ...], int] = {}
        row_indices = np.arange(dataset.row_count, dtype=int)
        for row in row_indices.tolist():
            key = tuple(view.labels[row] for view in attr_views)
            groups[key] = groups.get(key, 0) + 1
        chi2 = 0.0
        if color_variable == "(Pearson residuals)":
            marginals = {view.name: self._count_labels(view.labels) for view in attr_views}
            for key, observed in groups.items():
                expected = float(dataset.row_count)
                for view, value in zip(attr_views, key):
                    expected *= marginals[view.name].get(value, 0) / max(1, dataset.row_count)
                if expected > 1e-9:
                    chi2 += (observed - expected) ** 2 / expected
            degrees = max(1, len(groups) - 1)
        else:
            color_view = categorical_view(dataset, color_variable, bins=4, discretize_numeric=True)
            if color_view is None:
                return 0.0, 1.0
            priors = self._count_labels(color_view.labels)
            members: dict[tuple[str, ...], list[int]] = {}
            for row in row_indices.tolist():
                key = tuple(view.labels[row] for view in attr_views)
                members.setdefault(key, []).append(row)
            for rows in members.values():
                total = len(rows)
                observed = self._count_labels_for_rows(color_view.labels, np.asarray(rows, dtype=int))
                for label in color_view.categories:
                    expected = total * priors.get(label, 0) / max(1, dataset.row_count)
                    if expected > 1e-9:
                        chi2 += (observed.get(label, 0) - expected) ** 2 / expected
            degrees = max(1, len(members) * max(1, len(color_view.categories) - 1))
        return chi2, _chi2_sf(chi2, degrees)
