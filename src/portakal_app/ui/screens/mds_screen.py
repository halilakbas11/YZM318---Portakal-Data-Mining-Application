from __future__ import annotations

import math
import uuid

import numpy as np
import polars as pl
from PySide6.QtCore import QPointF, QRect, Qt
from PySide6.QtGui import QColor, QFont, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QSlider,
    QSpinBox,
    QToolTip,
    QVBoxLayout,
    QWidget,
)

from portakal_app.data.models import DatasetHandle
from portakal_app.data.services.distance_matrix_service import coerce_distance_matrix
from portakal_app.data.services.generated_dataset_service import GeneratedDatasetService
from portakal_app.data.services.sklearn_learner_service import SklearnLearnerService
from portakal_app.models import WorkflowPayload
from portakal_app.ui import i18n
from portakal_app.ui.screens.model_base import ModelScreenBase
from portakal_app.ui.screens.visualize_common import nice_ticks, subset_row_indices

_INIT_TYPES = ["random", "PCA"]
_PALETTE = (
    QColor("#3b82f6"),
    QColor("#e07020"),
    QColor("#22c55e"),
    QColor("#a855f7"),
    QColor("#f43f5e"),
    QColor("#0ea5e9"),
    QColor("#f59e0b"),
    QColor("#10b981"),
    QColor("#8b5cf6"),
    QColor("#ec4899"),
)


def _torgerson(distances: np.ndarray, n_components: int = 2) -> np.ndarray:
    matrix = np.asarray(distances, dtype=float)
    n_rows = matrix.shape[0]
    squared = matrix**2
    row_sum = squared.sum(axis=1, keepdims=True)
    col_sum = squared.sum(axis=0, keepdims=True)
    total = col_sum.sum()
    squared -= row_sum / n_rows
    squared -= col_sum / n_rows
    squared += total / (n_rows**2)
    centered = squared * -0.5
    values, vectors = np.linalg.eigh(centered)
    order = np.argsort(values)[::-1]
    values = np.maximum(values[order][:n_components], 0.0)
    vectors = vectors[:, order][:, :n_components]
    return vectors * np.sqrt(values).reshape(1, -1)


class _MDSScatterCanvas(QWidget):
    SHAPE_CIRCLE = 0
    SHAPE_SQUARE = 1
    SHAPE_TRIANGLE = 2
    SHAPE_DIAMOND = 3

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._points: list[tuple[float, float, str, QColor, int]] = []
        self._legend_items: list[tuple[str, QColor]] = []
        self._stress: float | None = None
        self._hovered_index: int | None = None
        self._point_rects: list[QRect] = []
        self._similar_pairs: list[tuple[int, int, float]] = []
        self._shapes: list[int] = []
        self._point_sizes: list[float] = []
        self._label_texts: list[str] = []
        self._point_radius = 6
        self._opacity = 200
        self._jitter = 0.0
        self._shown_pairs_count = 0
        self.setMinimumHeight(280)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMouseTracking(True)

    def set_visual_params(self, radius: int = 6, opacity: int = 200, jitter: float = 0.0, n_similar_pairs: int = 0) -> None:
        self._point_radius = max(2, radius)
        self._opacity = max(0, min(255, opacity))
        self._jitter = jitter
        self._shown_pairs_count = n_similar_pairs
        self.update()

    def set_data(
        self,
        points: list[tuple[float, float, str, QColor, int]],
        legend_items: list[tuple[str, QColor]],
        stress: float | None = None,
        similar_pairs: list[tuple[int, int, float]] | None = None,
        shapes: list[int] | None = None,
        point_sizes: list[float] | None = None,
        label_texts: list[str] | None = None,
    ) -> None:
        count = len(points)
        self._points = points
        self._legend_items = legend_items
        self._stress = stress
        self._similar_pairs = similar_pairs or []
        self._shapes = shapes if shapes is not None else [self.SHAPE_CIRCLE] * count
        self._point_sizes = point_sizes if point_sizes is not None else [1.0] * count
        self._label_texts = label_texts if label_texts is not None else [""] * count
        self._hovered_index = None
        self._point_rects = []
        self.update()

    def clear(self) -> None:
        self._points = []
        self._legend_items = []
        self._stress = None
        self._similar_pairs = []
        self._shapes = []
        self._point_sizes = []
        self._label_texts = []
        self._hovered_index = None
        self._point_rects = []
        self.update()

    def _chart_rect(self) -> QRect:
        margin_left = 72
        margin_right = 160 if self._legend_items else 20
        margin_top = 24
        margin_bottom = 56
        return QRect(margin_left, margin_top, max(80, self.width() - margin_left - margin_right), max(60, self.height() - margin_top - margin_bottom))

    def _data_ranges(self) -> tuple[float, float, float, float]:
        if not self._points:
            return -1.0, 1.0, -1.0, 1.0
        xs = [point[0] for point in self._points]
        ys = [point[1] for point in self._points]
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)
        x_pad = max((x_max - x_min) * 0.08, 0.3)
        y_pad = max((y_max - y_min) * 0.08, 0.3)
        return x_min - x_pad, x_max + x_pad, y_min - y_pad, y_max + y_pad

    def _map_to_pixel(self, chart: QRect, x: float, y: float, ranges: tuple[float, float, float, float]) -> tuple[int, int]:
        x_min, x_max, y_min, y_max = ranges
        span_x = max(x_max - x_min, 1e-9)
        span_y = max(y_max - y_min, 1e-9)
        px = chart.left() + int((x - x_min) / span_x * chart.width())
        py = chart.bottom() - int((y - y_min) / span_y * chart.height())
        return px, py

    def paintEvent(self, _event) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), QColor("#fffdf9"))
        if not self._points:
            painter.setPen(QColor("#8d877d"))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "No MDS data available.\nConnect data and press Apply.")
            painter.end()
            return

        chart = self._chart_rect()
        ranges = self._data_ranges()
        x_min, x_max, y_min, y_max = ranges
        fm = painter.fontMetrics()

        painter.setPen(QPen(QColor("#ece6dd"), 1, Qt.PenStyle.DotLine))
        for tick in nice_ticks(x_min, x_max, 6):
            px, _ = self._map_to_pixel(chart, tick, y_min, ranges)
            if chart.left() <= px <= chart.right():
                painter.drawLine(px, chart.top(), px, chart.bottom())
        for tick in nice_ticks(y_min, y_max, 6):
            _, py = self._map_to_pixel(chart, x_min, tick, ranges)
            if chart.top() <= py <= chart.bottom():
                painter.drawLine(chart.left(), py, chart.right(), py)

        painter.setPen(QPen(QColor("#9b9488"), 1))
        painter.drawRect(chart)

        painter.setPen(QColor("#5f5649"))
        painter.setFont(QFont(self.font().family(), 8))
        for tick in nice_ticks(x_min, x_max, 6):
            px, _ = self._map_to_pixel(chart, tick, y_min, ranges)
            label = f"{tick:.2g}"
            tw = fm.horizontalAdvance(label)
            painter.drawText(QRect(px - tw // 2, chart.bottom() + 4, tw + 8, 16), Qt.AlignmentFlag.AlignCenter, label)
        for tick in nice_ticks(y_min, y_max, 6):
            _, py = self._map_to_pixel(chart, x_min, tick, ranges)
            label = f"{tick:.2g}"
            tw = fm.horizontalAdvance(label)
            painter.drawText(QRect(chart.left() - tw - 10, py - 8, tw + 8, 16), Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter, label)

        bold = QFont(self.font().family(), 9)
        bold.setBold(True)
        painter.setFont(bold)
        painter.setPen(QColor("#463c2f"))
        painter.drawText(chart.adjusted(0, chart.height() + 20, 0, 40), Qt.AlignmentFlag.AlignCenter, "MDS Dimension 1")
        painter.save()
        painter.translate(16, chart.center().y())
        painter.rotate(-90)
        painter.drawText(QRect(-90, -8, 180, 16), Qt.AlignmentFlag.AlignCenter, "MDS Dimension 2")
        painter.restore()

        if self._stress is not None:
            painter.setFont(QFont(self.font().family(), 8))
            painter.setPen(QColor("#777777"))
            painter.drawText(QRect(chart.left(), chart.top() - 20, chart.width(), 18), Qt.AlignmentFlag.AlignRight, f"Kruskal Stress: {self._stress:.4f}")

        import random as _random

        n_show = self._shown_pairs_count if self._shown_pairs_count > 0 else len(self._similar_pairs)
        if self._similar_pairs and n_show > 0:
            px_cache: dict[int, tuple[int, int]] = {}
            for index, (x, y, _lbl, _color, _row) in enumerate(self._points):
                jx, jy = x, y
                if self._jitter > 0:
                    _random.seed(index)
                    jx += _random.gauss(0, self._jitter * 0.01)
                    jy += _random.gauss(0, self._jitter * 0.01)
                px_cache[index] = self._map_to_pixel(chart, jx, jy, ranges)
            for left_idx, right_idx, strength in self._similar_pairs[:n_show]:
                if left_idx >= len(self._points) or right_idx >= len(self._points):
                    continue
                left_x, left_y = px_cache[left_idx]
                right_x, right_y = px_cache[right_idx]
                left_color = QColor(self._points[left_idx][3])
                right_color = QColor(self._points[right_idx][3])
                line_color = QColor((left_color.red() + right_color.red()) // 2, (left_color.green() + right_color.green()) // 2, (left_color.blue() + right_color.blue()) // 2)
                line_color.setAlpha(max(60, int(strength * 200)))
                painter.setPen(QPen(line_color, max(1.0, strength * 3.0)))
                painter.drawLine(left_x, left_y, right_x, right_y)

        self._point_rects = []
        painter.setFont(QFont(self.font().family(), 8))
        for index, (x, y, hover_label, color, row_idx) in enumerate(self._points):
            jx, jy = x, y
            if self._jitter > 0:
                _random.seed(index)
                jx += _random.gauss(0, self._jitter * 0.01)
                jy += _random.gauss(0, self._jitter * 0.01)
            px, py = self._map_to_pixel(chart, jx, jy, ranges)
            size_mul = self._point_sizes[index] if index < len(self._point_sizes) else 1.0
            radius = max(2, int(self._point_radius * size_mul) + (2 if index == self._hovered_index else 0))
            shape = self._shapes[index] if index < len(self._shapes) else self.SHAPE_CIRCLE
            fill = QColor(color)
            fill.setAlpha(self._opacity)
            border = QColor("#ffffff") if index != self._hovered_index else QColor("#222222")
            painter.setBrush(fill)
            painter.setPen(QPen(border, 1.5))
            r = float(radius)
            if shape == self.SHAPE_SQUARE:
                painter.drawRect(QRect(px - radius, py - radius, radius * 2, radius * 2))
            elif shape == self.SHAPE_TRIANGLE:
                path = QPainterPath()
                path.moveTo(px, py - r)
                path.lineTo(px + r, py + r)
                path.lineTo(px - r, py + r)
                path.closeSubpath()
                painter.drawPath(path)
            elif shape == self.SHAPE_DIAMOND:
                path = QPainterPath()
                path.moveTo(px, py - r)
                path.lineTo(px + r, py)
                path.lineTo(px, py + r)
                path.lineTo(px - r, py)
                path.closeSubpath()
                painter.drawPath(path)
            else:
                painter.drawEllipse(QPointF(px, py), r, r)
            self._point_rects.append(QRect(px - radius - 2, py - radius - 2, (radius + 2) * 2, (radius + 2) * 2))
            label_text = self._label_texts[index] if index < len(self._label_texts) else ""
            if label_text:
                painter.setPen(QColor("#1a1310"))
                painter.drawText(QRect(px + radius + 3, py - 8, 200, 16), Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, label_text)

        if self._legend_items:
            legend_left = chart.right() + 18
            legend_top = chart.top()
            painter.setFont(QFont(self.font().family(), 9))
            painter.setPen(QColor("#463c2f"))
            painter.drawText(QRect(legend_left, legend_top, 140, 18), Qt.AlignmentFlag.AlignLeft, "Legend")
            y = legend_top + 24
            painter.setFont(QFont(self.font().family(), 8))
            for label, color in self._legend_items[:12]:
                painter.setPen(QPen(QColor("#534b40"), 1))
                painter.setBrush(color)
                painter.drawEllipse(QPointF(legend_left + 5, y + 6), 5, 5)
                painter.setPen(QColor("#534b40"))
                painter.drawText(QRect(legend_left + 16, y - 2, 130, 16), Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, label)
                y += 18
        painter.end()

    def mouseMoveEvent(self, event) -> None:  # noqa: N802
        if not self._points or not self._point_rects:
            return
        pos = event.position().toPoint()
        new_hover = None
        for index, rect in enumerate(self._point_rects):
            if rect.contains(pos):
                new_hover = index
                break
        if new_hover != self._hovered_index:
            self._hovered_index = new_hover
            self.update()
            if new_hover is not None:
                x, y, label, _, row_idx = self._points[new_hover]
                QToolTip.showText(event.globalPosition().toPoint(), f"{label}\nRow: {row_idx}\nDim 1: {x:.4f}\nDim 2: {y:.4f}", self)
            else:
                QToolTip.hideText()

    def leaveEvent(self, _event) -> None:  # noqa: N802
        self._hovered_index = None
        self.update()
        QToolTip.hideText()


class MDSScreen(ModelScreenBase):
    _OUTPUT_PORT_LABEL = "Data"

    def __init__(self, parent: QWidget | None = None) -> None:
        self._svc = SklearnLearnerService()
        self._builder = GeneratedDatasetService()
        self._projected_dataset: DatasetHandle | None = None
        self._selected_dataset: DatasetHandle | None = None
        self._embedding: np.ndarray | None = None
        self._stress: float | None = None
        super().__init__(parent)
        self._canvas = _MDSScatterCanvas()
        plot_box = QGroupBox("MDS Projection")
        plot_layout = QVBoxLayout(plot_box)
        plot_layout.setContentsMargins(6, 6, 6, 6)
        plot_layout.addWidget(self._canvas)
        self._stress_label = QLabel("")
        self._stress_label.setStyleSheet("color: #777; background: transparent; padding: 2px 4px;")
        plot_layout.addWidget(self._stress_label)
        self.layout().insertWidget(3, plot_box, 1)

    def set_input_payload(self, payload: WorkflowPayload | None) -> None:
        super().set_input_payload(payload)
        distances = self._extra_inputs.get("Distances")
        if self._dataset is None and distances is not None:
            try:
                handle = coerce_distance_matrix(distances)
                if handle.source_dataset is not None:
                    self._dataset = handle.source_dataset
            except Exception:
                pass

    def current_output_datasets(self) -> dict[str, DatasetHandle | None] | None:
        return {"Selected Data": self._selected_dataset, "Data": self._projected_dataset}

    def help_text(self) -> str:
        return "Project instances into two dimensions using multidimensional scaling."

    def documentation_url(self) -> str:
        return "https://orangedatamining.com/widget-catalog/unsupervised/mds/"

    def _add_main_layout(self, layout: QVBoxLayout) -> None:
        opt_box = QGroupBox("Optimize")
        opt_layout = QVBoxLayout(opt_box)
        opt_layout.setContentsMargins(8, 8, 8, 8)
        btn_row = QHBoxLayout()
        self._pca_btn = QPushButton("PCA")
        self._randomize_btn = QPushButton("Randomize")
        self._jitter_btn = QPushButton("Jitter")
        btn_row.addWidget(self._pca_btn)
        btn_row.addWidget(self._randomize_btn)
        btn_row.addWidget(self._jitter_btn)
        opt_layout.addLayout(btn_row)

        refresh_row = QHBoxLayout()
        refresh_row.addWidget(QLabel("Refresh:"))
        self._refresh_combo = QComboBox()
        self._refresh_combo.addItems(["Every 5 steps", "Every 10 steps", "Every 25 steps", "Every 50 steps"])
        self._refresh_combo.setCurrentIndex(2)
        refresh_row.addWidget(self._refresh_combo)
        opt_layout.addLayout(refresh_row)

        iter_row = QHBoxLayout()
        iter_row.addWidget(QLabel("Max iterations:"))
        self._max_iter_spin = QSpinBox()
        self._max_iter_spin.setRange(10, 1000)
        self._max_iter_spin.setValue(300)
        self._max_iter_spin.setSingleStep(50)
        iter_row.addWidget(self._max_iter_spin)
        opt_layout.addLayout(iter_row)
        layout.addWidget(opt_box)

        attr_box = QGroupBox("Attributes")
        attr_form = QFormLayout(attr_box)
        attr_form.setContentsMargins(8, 8, 8, 8)
        self._color_combo = QComboBox()
        self._color_combo.addItem("(Same color)")
        attr_form.addRow("Color:", self._color_combo)
        self._shape_combo = QComboBox()
        self._shape_combo.addItem("(Same shape)")
        attr_form.addRow("Shape:", self._shape_combo)
        self._size_combo = QComboBox()
        self._size_combo.addItem("(Same size)")
        attr_form.addRow("Size:", self._size_combo)
        self._label_combo = QComboBox()
        self._label_combo.addItem("(No labels)")
        attr_form.addRow("Label:", self._label_combo)
        self._label_subset_cb = QCheckBox("Label only selection and subset")
        attr_form.addRow(self._label_subset_cb)
        layout.addWidget(attr_box)

        slider_form = QFormLayout()
        self._symbol_size_slider = QSlider(Qt.Orientation.Horizontal)
        self._symbol_size_slider.setRange(1, 20)
        self._symbol_size_slider.setValue(10)
        slider_form.addRow("Symbol size:", self._symbol_size_slider)
        self._opacity_slider = QSlider(Qt.Orientation.Horizontal)
        self._opacity_slider.setRange(0, 255)
        self._opacity_slider.setValue(200)
        slider_form.addRow("Opacity:", self._opacity_slider)
        self._jittering_slider = QSlider(Qt.Orientation.Horizontal)
        self._jittering_slider.setRange(0, 50)
        self._jittering_slider.setValue(0)
        slider_form.addRow("Jittering:", self._jittering_slider)
        self._similar_pairs_slider = QSlider(Qt.Orientation.Horizontal)
        self._similar_pairs_slider.setRange(0, 100)
        self._similar_pairs_slider.setValue(20)
        slider_form.addRow("Show similar pairs:", self._similar_pairs_slider)
        layout.addLayout(slider_form)

        zoom_box = QGroupBox("Zoom/Select")
        zoom_layout = QHBoxLayout(zoom_box)
        zoom_layout.setContentsMargins(8, 8, 8, 8)
        self._select_btn = QPushButton(">")
        self._pan_btn = QPushButton("P")
        self._zoom_btn = QPushButton("Z")
        self._fit_btn = QPushButton("F")
        for button in (self._select_btn, self._pan_btn, self._zoom_btn, self._fit_btn):
            button.setFixedSize(32, 32)
        zoom_layout.addWidget(self._select_btn)
        zoom_layout.addWidget(self._pan_btn)
        zoom_layout.addWidget(self._zoom_btn)
        zoom_layout.addWidget(self._fit_btn)
        zoom_layout.addStretch(1)
        layout.addWidget(zoom_box)

        self._n_components_spin = QSpinBox()
        self._n_components_spin.setRange(2, 10)
        self._n_components_spin.setValue(2)
        self._n_components_spin.setVisible(False)
        self._metric_cb = QCheckBox("Metric MDS")
        self._metric_cb.setChecked(True)
        self._init_combo = QComboBox()
        self._init_combo.addItems(_INIT_TYPES)
        self._init_combo.setVisible(False)

        self._pca_btn.clicked.connect(self._on_pca_clicked)
        self._randomize_btn.clicked.connect(self._on_randomize_clicked)
        self._jitter_btn.clicked.connect(self._on_jitter_clicked)
        self._max_iter_spin.valueChanged.connect(self._settings_changed)
        self._symbol_size_slider.valueChanged.connect(self._on_visual_slider_changed)
        self._opacity_slider.valueChanged.connect(self._on_visual_slider_changed)
        self._jittering_slider.valueChanged.connect(self._on_visual_slider_changed)
        self._similar_pairs_slider.valueChanged.connect(self._on_visual_slider_changed)
        self._color_combo.currentIndexChanged.connect(self._on_attr_changed)
        self._shape_combo.currentIndexChanged.connect(self._on_attr_changed)
        self._size_combo.currentIndexChanged.connect(self._on_attr_changed)
        self._label_combo.currentIndexChanged.connect(self._on_attr_changed)

    def serialize_node_state(self) -> dict[str, object]:
        return {
            **super().serialize_node_state(),
            "n_components": self._n_components_spin.value(),
            "metric": self._metric_cb.isChecked(),
            "init_idx": self._init_combo.currentIndex(),
            "max_iter": self._max_iter_spin.value(),
        }

    def restore_node_state(self, payload: dict[str, object]) -> None:
        super().restore_node_state(payload)
        self._n_components_spin.setValue(int(payload.get("n_components", 2)))
        self._metric_cb.setChecked(bool(payload.get("metric", True)))
        self._init_combo.setCurrentIndex(int(payload.get("init_idx", 0)))
        self._max_iter_spin.setValue(int(payload.get("max_iter", 300)))

    def _on_attr_changed(self) -> None:
        if self._dataset is not None:
            self._update_plot()

    def _on_visual_slider_changed(self) -> None:
        self._canvas.set_visual_params(
            radius=self._symbol_size_slider.value(),
            opacity=self._opacity_slider.value(),
            jitter=float(self._jittering_slider.value()),
            n_similar_pairs=self._similar_pairs_slider.value(),
        )

    def _on_pca_clicked(self) -> None:
        self._init_combo.setCurrentIndex(1)
        self._settings_changed()

    def _on_randomize_clicked(self) -> None:
        self._init_combo.setCurrentIndex(0)
        self._settings_changed()

    def _on_jitter_clicked(self) -> None:
        self._jittering_slider.setValue(min(self._jittering_slider.value() + 5, 50))

    def _populate_attribute_combos(self) -> None:
        dataset = self._dataset
        previous = {
            "color": self._color_combo.currentText(),
            "shape": self._shape_combo.currentText(),
            "size": self._size_combo.currentText(),
            "label": self._label_combo.currentText(),
        }
        for combo in (self._color_combo, self._shape_combo, self._size_combo, self._label_combo):
            combo.blockSignals(True)
            combo.clear()
        self._color_combo.addItem("(Same color)")
        self._shape_combo.addItem("(Same shape)")
        self._size_combo.addItem("(Same size)")
        self._label_combo.addItem("(No labels)")
        if dataset is not None:
            names = [column.name for column in dataset.domain.feature_columns] + [column.name for column in dataset.domain.target_columns]
            for name in names:
                self._color_combo.addItem(name)
                self._shape_combo.addItem(name)
                self._size_combo.addItem(name)
                self._label_combo.addItem(name)
        for combo, key in (
            (self._color_combo, "color"),
            (self._shape_combo, "shape"),
            (self._size_combo, "size"),
            (self._label_combo, "label"),
        ):
            index = combo.findText(previous[key])
            combo.setCurrentIndex(index if index >= 0 else 0)
            combo.blockSignals(False)

    def _apply(self) -> None:
        self._projected_dataset = None
        self._selected_dataset = None
        self._embedding = None
        self._stress = None
        if self._dataset is None and "Distances" not in self._extra_inputs:
            self._dataset_label.setText(i18n.t("No data on input."))
            self._status_label.setText("")
            self._canvas.clear()
            self._stress_label.setText("")
            self._notify_output_changed()
            return

        dataset = self._dataset
        if dataset is not None:
            self._dataset_label.setText(
                i18n.tf(
                    "Training on: {name} ({rows} rows, {cols} features)",
                    name=dataset.display_name,
                    rows=dataset.row_count,
                    cols=len(list(dataset.domain.feature_columns)),
                )
            )

        self._populate_attribute_combos()
        self._update_plot()
        self._on_visual_slider_changed()
        self._notify_output_changed()

    def _update_plot(self) -> None:
        dataset = self._dataset
        if dataset is None and "Distances" in self._extra_inputs:
            try:
                handle = coerce_distance_matrix(self._extra_inputs["Distances"])
                dataset = handle.source_dataset
            except Exception:
                handle = None
            else:
                if self._dataset is None and dataset is not None:
                    self._dataset = dataset
        else:
            handle = None

        if dataset is None and handle is None:
            self._canvas.clear()
            self._stress_label.setText("")
            return

        if dataset is not None and self._color_combo.count() <= 1:
            self._populate_attribute_combos()

        try:
            from sklearn.manifold import MDS

            if handle is None and "Distances" in self._extra_inputs:
                try:
                    handle = coerce_distance_matrix(self._extra_inputs["Distances"])
                except Exception:
                    handle = None

            if handle is not None:
                distance_matrix = np.asarray(handle.matrix, dtype=float)
            else:
                assert dataset is not None
                feature_columns = [
                    column for column in dataset.domain.feature_columns if column.logical_type in {"numeric", "categorical", "boolean"}
                ]
                if not feature_columns:
                    self._canvas.clear()
                    self._stress_label.setText("")
                    return
                X = self._svc.encode_X(
                    dataset,
                    tuple(column.name for column in feature_columns),
                    {
                        column.name: list(dataset.dataframe.get_column(column.name).drop_nulls().unique(maintain_order=True).to_list())
                        for column in feature_columns
                        if column.logical_type in {"categorical", "boolean"}
                    },
                    tuple(column.name for column in feature_columns if column.logical_type == "numeric"),
                    {
                        column.name: float(np.nanmean(dataset.dataframe.get_column(column.name).cast(float, strict=False).to_numpy()))
                        for column in feature_columns
                        if column.logical_type == "numeric"
                    },
                )
                if X.shape[0] < 3:
                    self._canvas.clear()
                    self._stress_label.setText("Need at least 3 data points.")
                    return
                from scipy.spatial.distance import cdist as _cdist

                distance_matrix = _cdist(X, X, metric="euclidean")

            if distance_matrix.shape[0] < 3:
                self._canvas.clear()
                self._stress_label.setText("Need at least 3 data points.")
                return

            n_components = min(self._n_components_spin.value(), distance_matrix.shape[0] - 1, 2)
            init_type = _INIT_TYPES[self._init_combo.currentIndex()].lower()
            init_array = None
            if init_type == "pca":
                try:
                    init_array = _torgerson(distance_matrix, n_components)
                except Exception:
                    init_array = None
            mds = MDS(
                n_components=n_components,
                metric=self._metric_cb.isChecked(),
                max_iter=self._max_iter_spin.value(),
                n_init=1,
                random_state=42,
                dissimilarity="precomputed",
                normalized_stress="auto",
            )
            mds.fit(distance_matrix, init=init_array)
            embedding = np.asarray(mds.embedding_, dtype=float)
            self._embedding = embedding
            self._stress = float(mds.stress_) if hasattr(mds, "stress_") else None
            self._stress_label.setText(f"Kruskal Stress: {self._stress:.4f}" if self._stress is not None else "")

            labels: list[str] = []
            colors: list[QColor] = []
            legend_items: list[tuple[str, QColor]] = []
            if dataset is not None and dataset.domain.target_columns:
                target_name = dataset.domain.target_columns[0].name
                if target_name in dataset.dataframe.columns:
                    target_series = dataset.dataframe.get_column(target_name).to_list()
                    unique_values: list[str] = []
                    for value in target_series:
                        text = str(value) if value is not None else "(missing)"
                        if text not in unique_values:
                            unique_values.append(text)
                    color_map = {value: _PALETTE[index % len(_PALETTE)] for index, value in enumerate(unique_values)}
                    legend_items = [(value, color_map[value]) for value in unique_values]
                    for value in target_series:
                        text = str(value) if value is not None else "(missing)"
                        labels.append(text)
                        colors.append(color_map[text])

            if not labels:
                labels = [f"Row {index}" for index in range(embedding.shape[0])]
                colors = [_PALETTE[0]] * embedding.shape[0]

            if dataset is not None:
                color_name = self._color_combo.currentText() if self._color_combo.currentIndex() > 0 else None
                if color_name and color_name in dataset.dataframe.columns:
                    color_series = dataset.dataframe.get_column(color_name).to_list()
                    unique_color_values: list[str] = []
                    for value in color_series:
                        text = str(value) if value is not None else "(missing)"
                        if text not in unique_color_values:
                            unique_color_values.append(text)
                    color_map = {value: _PALETTE[index % len(_PALETTE)] for index, value in enumerate(unique_color_values)}
                    legend_items = [(value, color_map[value]) for value in unique_color_values]
                    colors = [color_map.get(str(value) if value is not None else "(missing)", _PALETTE[0]) for value in color_series]

                label_name = self._label_combo.currentText() if self._label_combo.currentIndex() > 0 else None
                if label_name and label_name in dataset.dataframe.columns:
                    label_series = dataset.dataframe.get_column(label_name).to_list()
                    labels = [str(value) if value is not None else "" for value in label_series]

                shape_name = self._shape_combo.currentText() if self._shape_combo.currentIndex() > 0 else None
                shapes = [_MDSScatterCanvas.SHAPE_CIRCLE] * embedding.shape[0]
                if shape_name and shape_name in dataset.dataframe.columns:
                    shape_series = dataset.dataframe.get_column(shape_name).to_list()
                    unique_shapes = list(dict.fromkeys(str(value) for value in shape_series if value is not None))
                    shape_map = {
                        category: shape
                        for category, shape in zip(
                            unique_shapes,
                            [
                                _MDSScatterCanvas.SHAPE_CIRCLE,
                                _MDSScatterCanvas.SHAPE_SQUARE,
                                _MDSScatterCanvas.SHAPE_TRIANGLE,
                                _MDSScatterCanvas.SHAPE_DIAMOND,
                            ]
                            * 8,
                        )
                    }
                    shapes = [shape_map.get(str(value) if value is not None else "", _MDSScatterCanvas.SHAPE_CIRCLE) for value in shape_series]

                size_name = self._size_combo.currentText() if self._size_combo.currentIndex() > 0 else None
                point_sizes = [1.0] * embedding.shape[0]
                if size_name and size_name in dataset.dataframe.columns:
                    try:
                        raw_sizes = [float(value) if value is not None else 0.0 for value in dataset.dataframe.get_column(size_name).to_list()]
                        low, high = min(raw_sizes), max(raw_sizes)
                        span = max(high - low, 1e-9)
                        point_sizes = [0.4 + 1.6 * (value - low) / span for value in raw_sizes]
                    except Exception:
                        point_sizes = [1.0] * embedding.shape[0]

                label_text_name = self._label_combo.currentText() if self._label_combo.currentIndex() > 0 else None
                if self._label_subset_cb.isChecked() and dataset is not None:
                    subset_rows = subset_row_indices(dataset, self._extra_inputs.get("Data Subset"))
                    label_texts = [
                        (str(dataset.dataframe.get_column(label_text_name)[index]) if label_text_name and label_text_name in dataset.dataframe.columns else labels[index])
                        if index in subset_rows else ""
                        for index in range(embedding.shape[0])
                    ]
                elif label_text_name and dataset is not None and label_text_name in dataset.dataframe.columns:
                    label_series = dataset.dataframe.get_column(label_text_name).to_list()
                    label_texts = [str(value) if value is not None else "" for value in label_series]
                else:
                    label_texts = [""] * embedding.shape[0]
            else:
                shapes = [_MDSScatterCanvas.SHAPE_CIRCLE] * embedding.shape[0]
                point_sizes = [1.0] * embedding.shape[0]
                label_texts = [""] * embedding.shape[0]

            points = [
                (
                    float(embedding[index, 0]),
                    float(embedding[index, 1]) if embedding.shape[1] > 1 else 0.0,
                    labels[index] if index < len(labels) else f"Row {index}",
                    colors[index] if index < len(colors) else _PALETTE[0],
                    index,
                )
                for index in range(embedding.shape[0])
            ]

            all_pairs: list[tuple[int, int, float]] = []
            if embedding.shape[0] > 1:
                from scipy.spatial.distance import cdist

                pair_distances = cdist(embedding[:, :2], embedding[:, :2], metric="euclidean")
                max_distance = float(pair_distances.max()) if pair_distances.max() > 0 else 1.0
                pairs: list[tuple[float, int, int]] = []
                for left_index in range(embedding.shape[0]):
                    for right_index in range(left_index + 1, embedding.shape[0]):
                        pairs.append((float(pair_distances[left_index, right_index]), left_index, right_index))
                pairs.sort()
                for distance, left_index, right_index in pairs[:200]:
                    all_pairs.append((left_index, right_index, 1.0 - distance / max_distance))

            self._canvas.set_data(
                points,
                legend_items,
                self._stress,
                similar_pairs=all_pairs,
                shapes=shapes,
                point_sizes=point_sizes,
                label_texts=label_texts,
            )
            self._build_outputs()
            self._status_label.setText(i18n.t("Model trained successfully."))
            self._status_label.setStyleSheet("color: #2e7d32; background: transparent;")
        except Exception as exc:
            self._canvas.clear()
            self._stress_label.setText(f"Error: {exc}")
            self._status_label.setText(str(exc))
            self._status_label.setStyleSheet("color: #c62828; background: transparent;")
            self._projected_dataset = None
            self._selected_dataset = None

    def _build_outputs(self) -> None:
        if self._dataset is None or self._embedding is None:
            self._projected_dataset = None
            self._selected_dataset = None
            return
        dataframe = self._dataset.dataframe.with_columns(
            pl.Series("MDS 1", [float(value) for value in self._embedding[:, 0].tolist()]),
            pl.Series("MDS 2", [float(value) for value in self._embedding[:, 1].tolist()]),
        )
        role_overrides = {column.name: column.role for column in self._dataset.domain.columns}
        role_overrides.update({"MDS 1": "feature", "MDS 2": "feature"})
        self._projected_dataset = self._builder.build_dataset(
            dataframe,
            dataset_id=f"{self._dataset.dataset_id}-mds-{uuid.uuid4().hex[:8]}",
            display_name=f"{self._dataset.display_name} (MDS)",
            file_name=f"{self._dataset.dataset_id}-mds.csv",
            role_overrides=role_overrides,
            annotations={
                **self._dataset.annotations,
                "mds": {
                    "stress": float(self._stress or 0.0),
                },
            },
        )
        self._selected_dataset = None
