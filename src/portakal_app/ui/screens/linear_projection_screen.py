from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import polars as pl
from PySide6.QtCore import QPoint, QRect, QRectF, Qt, Signal, QSize
from PySide6.QtGui import QColor, QFont, QFontMetrics, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QToolButton,
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
    build_components_dataset,
    build_selection_outputs,
    discrete_columns,
    numeric_columns,
    primitive_columns,
    subset_row_indices,
)

PLACEMENT_CIRCULAR = 0
PLACEMENT_LDA = 1
PLACEMENT_PCA = 2

SELECT_MODE = "select"
PAN_MODE = "pan"
ZOOM_MODE = "zoom"

ORANGE_SHAPES = ("Circle", "Square", "Triangle", "Diamond", "Cross", "Plus")


def _nice_ticks(vmin: float, vmax: float, n: int = 5) -> list[float]:
    span = vmax - vmin or 1.0
    raw = span / n
    exp = math.floor(math.log10(abs(raw)) if abs(raw) > 1e-15 else 0)
    frac = raw / 10 ** exp
    nice = next((value for value in (1, 2, 2.5, 5, 10) if frac <= value), 10)
    step = nice * 10 ** exp
    start = math.ceil(vmin / step) * step
    ticks: list[float] = []
    value = start
    while value <= vmax + step * 0.01:
        ticks.append(round(value, 10))
        value += step
    return ticks


def _normalize_embedding(values: np.ndarray) -> np.ndarray:
    if values.ndim != 2 or not len(values):
        return values
    centered = values - np.mean(values, axis=0, keepdims=True)
    span = np.max(values, axis=0) - np.min(values, axis=0)
    span[span == 0] = 1.0
    return centered / span


def _circular_components(count: int) -> np.ndarray:
    if count == 0:
        return np.zeros((0, 2), dtype=float)
    if count == 1:
        angles = np.array([0.0], dtype=float)
    elif count == 2:
        angles = np.array([0.0, math.pi / 2], dtype=float)
    else:
        angles = np.linspace(0.0, 2.0 * math.pi, count, endpoint=False)
    return np.column_stack([np.cos(angles), np.sin(angles)]).astype(float, copy=False)


def _pca_projection(features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    centered = features - np.mean(features, axis=0, keepdims=True)
    std = np.std(centered, axis=0)
    std[std < 1e-10] = 1.0
    normalized = centered / std
    _, _, vt = np.linalg.svd(normalized, full_matrices=False)
    components = vt[:2].T
    coords = normalized @ components
    return _normalize_embedding(coords), components


def _lda_projection(features: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    classes = sorted({int(value) for value in labels})
    n_features = features.shape[1]
    overall_mean = np.mean(features, axis=0)
    sw = np.zeros((n_features, n_features), dtype=float)
    sb = np.zeros((n_features, n_features), dtype=float)
    for klass in classes:
        mask = labels == klass
        if not np.any(mask):
            continue
        group = features[mask]
        mean = np.mean(group, axis=0)
        sw += (group - mean).T @ (group - mean)
        diff = (mean - overall_mean).reshape(-1, 1)
        sb += len(group) * (diff @ diff.T)
    try:
        sw_inv = np.linalg.inv(sw + np.eye(n_features) * 1e-6)
        evals, evecs = np.linalg.eigh(sw_inv @ sb)
        order = np.argsort(evals)[::-1]
        components = evecs[:, order[:2]]
    except np.linalg.LinAlgError:
        return _pca_projection(features)
    coords = features @ components
    return _normalize_embedding(coords), components


def _score_feature_discrete(values: np.ndarray, labels: np.ndarray) -> float:
    overall = float(np.mean(values))
    between = 0.0
    within = 0.0
    for klass in sorted(set(labels.tolist())):
        group = values[labels == klass]
        if not len(group):
            continue
        mean = float(np.mean(group))
        between += len(group) * (mean - overall) ** 2
        within += float(np.sum((group - mean) ** 2))
    return between / (within + 1e-9)


def _score_feature_continuous(values: np.ndarray, targets: np.ndarray) -> float:
    if len(values) < 2:
        return 0.0
    corr = np.corrcoef(values, targets)[0, 1]
    return 0.0 if not math.isfinite(float(corr)) else abs(float(corr))


@dataclass(frozen=True)
class _ProjectionPoint:
    row_index: int
    x: float
    y: float
    color: QColor
    shape_index: int
    radius: int
    label: str
    legend: str
    tooltip: str
    subset: bool


class _ProjectionCanvas(QWidget):
    selectionChanged = Signal(list)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._points: list[_ProjectionPoint] = []
        self._anchors: list[tuple[str, float, float]] = []
        self._legend_items: list[tuple[str, QColor]] = []
        self._x_range = (-1.0, 1.0)
        self._y_range = (-1.0, 1.0)
        self._tool_mode = SELECT_MODE
        self._show_regions = False
        self._show_legend = True
        self._label_selected_only = False
        self._always_show_axes = True
        self._hide_radius_fraction = 0.0
        self._selected_rows: set[int] = set()
        self._point_hit_regions: list[tuple[QRect, int, str]] = []
        self._selection_rect: QRect | None = None
        self._drag_origin: QPoint | None = None
        self._zoom = 1.0
        self._pan_x = 0.0
        self._pan_y = 0.0
        self._point_size = 9
        self._opacity = 180
        self._jitter = 0.0

        self.setMouseTracking(True)
        self.setMinimumSize(600, 420)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

    def set_projection(
        self,
        points: list[_ProjectionPoint],
        anchors: list[tuple[str, float, float]],
        legend_items: list[tuple[str, QColor]],
        *,
        x_range: tuple[float, float],
        y_range: tuple[float, float],
        always_show_axes: bool,
    ) -> None:
        self._points = points
        self._anchors = anchors
        self._legend_items = legend_items
        self._x_range = x_range
        self._y_range = y_range
        self._always_show_axes = always_show_axes
        self._point_hit_regions = []
        self._zoom = 1.0
        self._pan_x = 0.0
        self._pan_y = 0.0
        self.update()

    def set_selected_rows(self, rows: list[int]) -> None:
        self._selected_rows = set(rows)
        self.update()

    def set_tool_mode(self, mode: str) -> None:
        self._tool_mode = mode

    def set_show_regions(self, checked: bool) -> None:
        self._show_regions = checked
        self.update()

    def set_show_legend(self, checked: bool) -> None:
        self._show_legend = checked
        self.update()

    def set_label_selected_only(self, checked: bool) -> None:
        self._label_selected_only = checked
        self.update()

    def set_point_size(self, value: int) -> None:
        self._point_size = max(3, int(value))
        self.update()

    def set_opacity(self, value: int) -> None:
        self._opacity = max(40, min(255, int(value)))
        self.update()

    def set_jitter(self, value: float) -> None:
        self._jitter = max(0.0, float(value))
        self.update()

    def set_hide_radius(self, value: int) -> None:
        self._hide_radius_fraction = max(0.0, min(1.0, value / 100.0))
        self.update()

    def reset_view(self) -> None:
        self._zoom = 1.0
        self._pan_x = 0.0
        self._pan_y = 0.0
        self.update()

    def wheelEvent(self, event) -> None:
        factor = 1.15 if event.angleDelta().y() > 0 else 1 / 1.15
        self._zoom = max(0.2, min(8.0, self._zoom * factor))
        self.update()

    def mousePressEvent(self, event) -> None:
        if event.button() != Qt.MouseButton.LeftButton:
            return
        self._drag_origin = event.position().toPoint()
        if self._tool_mode in {SELECT_MODE, ZOOM_MODE}:
            self._selection_rect = QRect(self._drag_origin, self._drag_origin)

    def mouseMoveEvent(self, event) -> None:
        pos = event.position().toPoint()
        if event.buttons() & Qt.MouseButton.LeftButton and self._drag_origin is not None:
            if self._tool_mode == PAN_MODE:
                self._pan_x += pos.x() - self._drag_origin.x()
                self._pan_y += pos.y() - self._drag_origin.y()
                self._drag_origin = pos
                self.update()
            else:
                self._selection_rect = QRect(self._drag_origin, pos).normalized()
                self.update()
            QToolTip.hideText()
            return
        for rect, _, tip in self._point_hit_regions:
            if rect.contains(pos):
                QToolTip.showText(event.globalPosition().toPoint(), tip, self)
                return
        QToolTip.hideText()

    def mouseReleaseEvent(self, event) -> None:
        if event.button() != Qt.MouseButton.LeftButton:
            return
        rect = self._selection_rect.normalized() if self._selection_rect is not None else None
        self._selection_rect = None
        if self._drag_origin is None:
            return
        origin = self._drag_origin
        self._drag_origin = None
        if rect is None:
            return
        if rect.width() < 4 and rect.height() < 4:
            if self._tool_mode == SELECT_MODE:
                row = self._row_at(event.position().toPoint())
                self.selectionChanged.emit([] if row is None else [row])
            self.update()
            return
        if self._tool_mode == SELECT_MODE:
            rows = [row for hit_rect, row, _ in self._point_hit_regions if rect.intersects(hit_rect)]
            self.selectionChanged.emit(sorted(set(rows)))
        elif self._tool_mode == ZOOM_MODE:
            self._zoom_to_rect(rect, origin)
        self.update()

    def mouseDoubleClickEvent(self, _event) -> None:
        self.reset_view()

    def leaveEvent(self, _event) -> None:
        QToolTip.hideText()

    def sizeHint(self) -> QSize:
        return QSize(780, 640)

    def paintEvent(self, _event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), QColor("#fffdf9"))

        if not self._points and not self._anchors:
            painter.setPen(QColor("#8d877d"))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "No projection available.")
            return

        self._point_hit_regions = []
        chart = self._chart_rect()
        x_min, x_max, y_min, y_max = self._view_bounds()

        if self._show_regions and self._legend_items:
            self._draw_regions(painter, chart, x_min, x_max, y_min, y_max)

        x_ticks = _nice_ticks(x_min, x_max, 5)
        y_ticks = _nice_ticks(y_min, y_max, 5)
        painter.setPen(QPen(QColor("#e0ddd6"), 1, Qt.PenStyle.DotLine))
        for tick in x_ticks:
            px, _ = self._map_to_pixel(chart, tick, 0.0, x_min, x_max, y_min, y_max)
            painter.drawLine(px, chart.top(), px, chart.bottom())
        for tick in y_ticks:
            _, py = self._map_to_pixel(chart, 0.0, tick, x_min, x_max, y_min, y_max)
            painter.drawLine(chart.left(), py, chart.right(), py)

        if x_min <= 0 <= x_max:
            px, _ = self._map_to_pixel(chart, 0.0, 0.0, x_min, x_max, y_min, y_max)
            painter.setPen(QPen(QColor("#c0bcb5"), 1, Qt.PenStyle.DashLine))
            painter.drawLine(px, chart.top(), px, chart.bottom())
        if y_min <= 0 <= y_max:
            _, py = self._map_to_pixel(chart, 0.0, 0.0, x_min, x_max, y_min, y_max)
            painter.setPen(QPen(QColor("#c0bcb5"), 1, Qt.PenStyle.DashLine))
            painter.drawLine(chart.left(), py, chart.right(), py)

        painter.setPen(QPen(QColor("#9b9488"), 1))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawRect(chart)

        painter.setFont(QFont(self.font().family(), 8))
        fm = QFontMetrics(painter.font())
        painter.setPen(QColor("#8d877d"))
        for tick in x_ticks:
            px, _ = self._map_to_pixel(chart, tick, 0.0, x_min, x_max, y_min, y_max)
            painter.drawText(px - 24, chart.bottom() + 6, 48, 14, Qt.AlignmentFlag.AlignCenter, f"{tick:.3g}")
        for tick in y_ticks:
            _, py = self._map_to_pixel(chart, 0.0, tick, x_min, x_max, y_min, y_max)
            painter.drawText(2, py - 8, chart.left() - 8, 16, Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter, f"{tick:.3g}")

        max_anchor_radius = max((math.hypot(ax, ay) for _, ax, ay in self._anchors), default=1.0)
        hide_radius = self._hide_radius_fraction * max_anchor_radius
        if not self._always_show_axes and hide_radius > 0:
            origin_x, origin_y = self._map_to_pixel(chart, 0.0, 0.0, x_min, x_max, y_min, y_max)
            edge_x, edge_y = self._map_to_pixel(chart, hide_radius, 0.0, x_min, x_max, y_min, y_max)
            radius = abs(edge_x - origin_x) or abs(edge_y - origin_y)
            painter.setPen(QPen(QColor(145, 135, 125, 130), 1.2, Qt.PenStyle.DashLine))
            painter.drawEllipse(origin_x - radius, origin_y - radius, radius * 2, radius * 2)

        origin_x, origin_y = self._map_to_pixel(chart, 0.0, 0.0, x_min, x_max, y_min, y_max)
        for name, ax, ay in self._anchors:
            radius = math.hypot(ax, ay)
            if not self._always_show_axes and radius <= hide_radius + 1e-5:
                continue
            tx, ty = self._map_to_pixel(chart, ax, ay, x_min, x_max, y_min, y_max)
            painter.setPen(QPen(QColor("#7a7369"), 1.5))
            painter.drawLine(origin_x, origin_y, tx, ty)
            self._draw_arrow_head(painter, origin_x, origin_y, tx, ty)
            painter.setPen(QColor("#3b2a10"))
            label_x = tx + (8 if tx >= origin_x else -8 - fm.horizontalAdvance(name))
            label_y = ty + (12 if ty >= origin_y else -4)
            painter.drawText(label_x, label_y, fm.elidedText(name, Qt.TextElideMode.ElideRight, 100))

        for point in self._points:
            jitter_x = point.x + (np.random.RandomState(point.row_index).rand() - 0.5) * self._jitter * 0.02
            jitter_y = point.y + (np.random.RandomState(point.row_index + 1).rand() - 0.5) * self._jitter * 0.02
            px, py = self._map_to_pixel(chart, jitter_x, jitter_y, x_min, x_max, y_min, y_max)
            radius = max(3, int(round(point.radius * self._point_size / 10.0)))
            if point.subset:
                painter.setPen(QPen(QColor("#111111"), 1.8))
                painter.setBrush(QColor("#ffffff"))
                self._draw_marker(painter, point.shape_index, px, py, radius + 2)
            fill = QColor(point.color)
            fill.setAlpha(self._opacity)
            border = QColor("#111111") if point.row_index in self._selected_rows else QColor(fill.darker(135))
            painter.setPen(QPen(border, 1.4))
            painter.setBrush(fill)
            self._draw_marker(painter, point.shape_index, px, py, radius)
            rect = QRect(px - radius - 4, py - radius - 4, radius * 2 + 8, radius * 2 + 8)
            self._point_hit_regions.append((rect, point.row_index, point.tooltip))
            if point.label and (not self._label_selected_only or point.row_index in self._selected_rows):
                painter.setPen(QColor("#3b2a10"))
                painter.drawText(QRect(px + radius + 4, py - 10, 140, 20), Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, point.label)

        if self._selection_rect is not None:
            painter.setPen(QPen(QColor("#2563eb"), 1.5, Qt.PenStyle.DashLine))
            painter.setBrush(QColor(37, 99, 235, 30))
            painter.drawRect(self._selection_rect)

        if self._show_legend and self._legend_items:
            left = chart.right() + 18
            painter.setPen(QColor("#534b40"))
            painter.drawText(QRect(left, chart.top(), 120, 18), Qt.AlignmentFlag.AlignLeft, "Legend")
            y = chart.top() + 24
            for label, color in self._legend_items[:8]:
                painter.setPen(Qt.PenStyle.NoPen)
                painter.setBrush(color)
                painter.drawEllipse(left, y + 2, 11, 11)
                painter.setPen(QColor("#534b40"))
                painter.drawText(QRect(left + 16, y, 116, 18), Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, fm.elidedText(label, Qt.TextElideMode.ElideRight, 110))
                y += 20

        painter.setPen(QColor(160, 155, 145, 160))
        painter.setFont(QFont(self.font().family(), 7))
        painter.drawText(chart.left() + 4, self.height() - 6, f"scroll=zoom  drag={self._tool_mode}  dbl-click=reset")
        painter.end()

    def _chart_rect(self) -> QRect:
        margin_left = 64
        margin_right = 170 if self._show_legend and self._legend_items else 24
        margin_top = 18
        margin_bottom = 48
        return QRect(margin_left, margin_top, max(10, self.width() - margin_left - margin_right), max(10, self.height() - margin_top - margin_bottom))

    def _view_bounds(self) -> tuple[float, float, float, float]:
        x0, x1 = self._x_range
        y0, y1 = self._y_range
        span_x = (x1 - x0) or 1.0
        span_y = (y1 - y0) or 1.0
        eff_x = span_x / self._zoom
        eff_y = span_y / self._zoom
        center_x = (x0 + x1) / 2.0 - self._pan_x / max(1, self.width()) * eff_x
        center_y = (y0 + y1) / 2.0 + self._pan_y / max(1, self.height()) * eff_y
        return (
            center_x - eff_x / 2.0,
            center_x + eff_x / 2.0,
            center_y - eff_y / 2.0,
            center_y + eff_y / 2.0,
        )

    @staticmethod
    def _map_to_pixel(chart: QRect, x: float, y: float, x_min: float, x_max: float, y_min: float, y_max: float) -> tuple[int, int]:
        sx = chart.left() + int((x - x_min) / ((x_max - x_min) or 1.0) * chart.width())
        sy = chart.bottom() - int((y - y_min) / ((y_max - y_min) or 1.0) * chart.height())
        return sx, sy

    def _draw_regions(self, painter: QPainter, chart: QRect, x_min: float, x_max: float, y_min: float, y_max: float) -> None:
        bins_x = 18
        bins_y = 18
        counts: dict[tuple[int, int], dict[str, int]] = {}
        palette = {label: color for label, color in self._legend_items}
        for point in self._points:
            if not point.legend:
                continue
            ix = int((point.x - x_min) / ((x_max - x_min) or 1.0) * bins_x)
            iy = int((point.y - y_min) / ((y_max - y_min) or 1.0) * bins_y)
            ix = max(0, min(bins_x - 1, ix))
            iy = max(0, min(bins_y - 1, iy))
            counts.setdefault((ix, iy), {}).setdefault(point.legend, 0)
            counts[(ix, iy)][point.legend] += 1
        if not counts:
            return
        max_density = max(sum(values.values()) for values in counts.values())
        cell_w = chart.width() / bins_x
        cell_h = chart.height() / bins_y
        painter.save()
        painter.setPen(Qt.PenStyle.NoPen)
        for (ix, iy), values in counts.items():
            legend = max(values.items(), key=lambda item: item[1])[0]
            color = QColor(palette.get(legend, QColor("#94a3b8")))
            color.setAlpha(int(18 + 78 * (sum(values.values()) / max_density)))
            left = int(chart.left() + ix * cell_w)
            top = int(chart.bottom() - (iy + 1) * cell_h)
            painter.setBrush(color)
            painter.drawRect(QRectF(left, top, cell_w + 1, cell_h + 1))
        painter.restore()

    def _row_at(self, pos: QPoint) -> int | None:
        for rect, row, _ in reversed(self._point_hit_regions):
            if rect.contains(pos):
                return row
        return None

    def _zoom_to_rect(self, rect: QRect, origin: QPoint) -> None:
        if rect.width() < 10 or rect.height() < 10:
            return
        factor = min(self.width() / rect.width(), self.height() / rect.height())
        self._zoom = max(0.2, min(8.0, self._zoom * factor * 0.7))
        center = rect.center()
        self._pan_x += origin.x() - center.x()
        self._pan_y += origin.y() - center.y()

    @staticmethod
    def _draw_arrow_head(painter: QPainter, x0: int, y0: int, x1: int, y1: int) -> None:
        dx = x1 - x0
        dy = y1 - y0
        length = math.hypot(dx, dy) or 1.0
        ux = dx / length
        uy = dy / length
        px = -uy
        py = ux
        head = 6
        p1 = QPoint(int(x1 - ux * head + px * 3), int(y1 - uy * head + py * 3))
        p2 = QPoint(int(x1 - ux * head - px * 3), int(y1 - uy * head - py * 3))
        painter.setBrush(QColor("#7a7369"))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawPolygon([QPoint(x1, y1), p1, p2])

    @staticmethod
    def _draw_marker(painter: QPainter, shape_index: int, px: int, py: int, radius: int) -> None:
        if shape_index == 1:
            painter.drawRect(px - radius, py - radius, radius * 2, radius * 2)
            return
        if shape_index == 2:
            path = QPainterPath()
            path.moveTo(px, py - radius)
            path.lineTo(px - radius, py + radius)
            path.lineTo(px + radius, py + radius)
            path.closeSubpath()
            painter.drawPath(path)
            return
        if shape_index == 3:
            path = QPainterPath()
            path.moveTo(px, py - radius)
            path.lineTo(px - radius, py)
            path.lineTo(px, py + radius)
            path.lineTo(px + radius, py)
            path.closeSubpath()
            painter.drawPath(path)
            return
        if shape_index == 4:
            painter.drawLine(px - radius, py - radius, px + radius, py + radius)
            painter.drawLine(px - radius, py + radius, px + radius, py - radius)
            return
        if shape_index == 5:
            painter.drawLine(px - radius, py, px + radius, py)
            painter.drawLine(px, py - radius, px, py + radius)
            return
        painter.drawEllipse(px - radius, py - radius, radius * 2, radius * 2)


class LinearProjectionScreen(QWidget, WorkflowNodeScreenSupport):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._init_workflow_node_support()
        self._dataset: DatasetHandle | None = None
        self._subset: DatasetHandle | None = None
        self._projection_input: DatasetHandle | None = None
        self._builder = GeneratedDatasetService()
        self._selected_rows: list[int] = []
        self._selected_dataset: DatasetHandle | None = None
        self._annotated_dataset: DatasetHandle | None = None
        self._components_dataset: DatasetHandle | None = None
        self._selected_feature_names: list[str] = []
        self._effective_rows = np.asarray([], dtype=int)
        self._current_components: np.ndarray | None = None
        self._current_embedding: np.ndarray | None = None
        self._status_message = "Load data with numeric features."
        self._info_message = ""

        root = QHBoxLayout(self)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(12)

        left_scroll = QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        left_scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        left_scroll.setFixedWidth(330)
        left_panel = QWidget()
        left_scroll.setWidget(left_panel)
        root.addWidget(left_scroll, 0)

        left = QVBoxLayout(left_panel)
        left.setContentsMargins(0, 0, 0, 0)
        left.setSpacing(10)

        features_box = QGroupBox("Features")
        features_layout = QVBoxLayout(features_box)
        features_layout.setContentsMargins(10, 12, 10, 10)
        features_layout.setSpacing(8)
        self._feature_filter_edit = QLineEdit()
        self._feature_filter_edit.setPlaceholderText("Filter...")
        self._feature_filter_edit.textChanged.connect(self._filter_feature_list)
        features_layout.addWidget(self._feature_filter_edit)
        self._feature_list = QListWidget()
        self._feature_list.setSelectionMode(QListWidget.SelectionMode.ExtendedSelection)
        self._feature_list.itemSelectionChanged.connect(self._handle_feature_selection_changed)
        features_layout.addWidget(self._feature_list)
        self._suggest_btn = QPushButton("Suggest Features")
        self._suggest_btn.clicked.connect(self._suggest_features)
        features_layout.addWidget(self._suggest_btn)
        left.addWidget(features_box)

        placement_box = QGroupBox("Placement")
        placement_layout = QVBoxLayout(placement_box)
        placement_layout.setContentsMargins(10, 12, 10, 10)
        placement_layout.setSpacing(6)
        self._placement_group = QButtonGroup(self)
        self._circular_rb = QRadioButton("Circular Placement")
        self._lda_rb = QRadioButton("Linear Discriminant Analysis")
        self._pca_rb = QRadioButton("Principal Component Analysis")
        self._placement_group.addButton(self._circular_rb, PLACEMENT_CIRCULAR)
        self._placement_group.addButton(self._lda_rb, PLACEMENT_LDA)
        self._placement_group.addButton(self._pca_rb, PLACEMENT_PCA)
        self._circular_rb.setChecked(True)
        self._placement_group.idClicked.connect(self._handle_projection_changed)
        placement_layout.addWidget(self._circular_rb)
        placement_layout.addWidget(self._lda_rb)
        placement_layout.addWidget(self._pca_rb)
        self._placement_info_label = QLabel("")
        self._placement_info_label.setWordWrap(True)
        self._placement_info_label.setStyleSheet("color: #92400e;")
        placement_layout.addWidget(self._placement_info_label)
        left.addWidget(placement_box)

        display_box = QGroupBox("Display")
        display_layout = QVBoxLayout(display_box)
        display_layout.setContentsMargins(10, 12, 10, 10)
        display_layout.setSpacing(8)
        self._color_combo = QComboBox()
        self._color_combo.currentTextChanged.connect(self._refresh_projection)
        self._label_combo = QComboBox()
        self._label_combo.currentTextChanged.connect(self._refresh_projection)
        self._shape_combo = QComboBox()
        self._shape_combo.currentTextChanged.connect(self._refresh_projection)
        self._size_combo = QComboBox()
        self._size_combo.currentTextChanged.connect(self._refresh_projection)
        display_layout.addWidget(QLabel("Color"))
        display_layout.addWidget(self._color_combo)
        display_layout.addWidget(QLabel("Label"))
        display_layout.addWidget(self._label_combo)
        display_layout.addWidget(QLabel("Shape"))
        display_layout.addWidget(self._shape_combo)
        display_layout.addWidget(QLabel("Size"))
        display_layout.addWidget(self._size_combo)
        self._label_selected_only_cb = QCheckBox("Label only selected points")
        self._label_selected_only_cb.toggled.connect(lambda checked: self._canvas.set_label_selected_only(checked))
        display_layout.addWidget(self._label_selected_only_cb)
        left.addWidget(display_box)

        effects_box = QGroupBox("Plot")
        effects_layout = QVBoxLayout(effects_box)
        effects_layout.setContentsMargins(10, 12, 10, 10)
        effects_layout.setSpacing(8)
        self._point_size_slider = QSlider(Qt.Orientation.Horizontal)
        self._point_size_slider.setRange(4, 18)
        self._point_size_slider.setValue(9)
        self._opacity_slider = QSlider(Qt.Orientation.Horizontal)
        self._opacity_slider.setRange(40, 255)
        self._opacity_slider.setValue(180)
        self._jitter_slider = QSlider(Qt.Orientation.Horizontal)
        self._jitter_slider.setRange(0, 50)
        self._jitter_slider.setValue(0)
        self._hide_radius_slider = QSlider(Qt.Orientation.Horizontal)
        self._hide_radius_slider.setRange(0, 100)
        self._hide_radius_slider.setValue(0)
        self._show_regions_cb = QCheckBox("Show color regions")
        self._show_legend_cb = QCheckBox("Show legend")
        self._show_legend_cb.setChecked(True)
        for label, control in (
            ("Symbol size", self._point_size_slider),
            ("Opacity", self._opacity_slider),
            ("Jittering", self._jitter_slider),
            ("Hide radius", self._hide_radius_slider),
        ):
            effects_layout.addWidget(QLabel(label))
            effects_layout.addWidget(control)
        effects_layout.addWidget(self._show_regions_cb)
        effects_layout.addWidget(self._show_legend_cb)
        left.addWidget(effects_box)

        tools_box = QGroupBox("Tools")
        tools_layout = QHBoxLayout(tools_box)
        tools_layout.setContentsMargins(10, 12, 10, 10)
        tools_layout.setSpacing(6)
        self._tool_group = QButtonGroup(self)
        self._select_btn = self._make_tool_button("Select", SELECT_MODE, True)
        self._pan_btn = self._make_tool_button("Pan", PAN_MODE, False)
        self._zoom_btn = self._make_tool_button("Zoom", ZOOM_MODE, False)
        self._zoom_fit_btn = QPushButton("Zoom to fit")
        tools_layout.addWidget(self._select_btn)
        tools_layout.addWidget(self._pan_btn)
        tools_layout.addWidget(self._zoom_btn)
        tools_layout.addWidget(self._zoom_fit_btn)
        left.addWidget(tools_box)

        self._status_label = QLabel(self._status_message)
        self._status_label.setWordWrap(True)
        self._selection_label = QLabel("Selected: 0")
        left.addWidget(self._status_label)
        left.addWidget(self._selection_label)
        left.addStretch(1)

        right = QVBoxLayout()
        right.setContentsMargins(0, 0, 0, 0)
        self._canvas = _ProjectionCanvas()
        self._canvas.selectionChanged.connect(self._handle_selection_changed)
        right.addWidget(self._canvas, 1)
        root.addLayout(right, 1)

        self._point_size_slider.valueChanged.connect(self._canvas.set_point_size)
        self._opacity_slider.valueChanged.connect(self._canvas.set_opacity)
        self._jitter_slider.valueChanged.connect(lambda value: self._canvas.set_jitter(value / 50.0))
        self._hide_radius_slider.valueChanged.connect(self._canvas.set_hide_radius)
        self._show_regions_cb.toggled.connect(self._canvas.set_show_regions)
        self._show_legend_cb.toggled.connect(self._canvas.set_show_legend)
        self._zoom_fit_btn.clicked.connect(self._canvas.reset_view)

    def documentation_url(self) -> str:
        return "https://orangedatamining.com/widget-catalog/visualize/linearprojection/"

    def sizeHint(self) -> QSize:
        return QSize(1132, 708)

    def set_input_payload(self, payload) -> None:
        if payload is None:
            self._dataset = None
            self._subset = None
            self._projection_input = None
        elif payload.port_label == "Data":
            self._dataset = payload.dataset
        elif payload.port_label == "Data Subset":
            self._subset = payload.dataset
        elif payload.port_label == "Projection":
            self._projection_input = payload.dataset
        self._sync_controls()
        self._refresh_projection()

    def current_output_dataset(self) -> DatasetHandle | None:
        return self._selected_dataset

    def current_output_datasets(self) -> dict[str, DatasetHandle | None] | None:
        return {
            "Selected Data": self._selected_dataset,
            "Annotated Data": self._annotated_dataset,
            "Components": self._components_dataset,
        }

    def current_output_payloads(self) -> dict[str, WorkflowPayload | None] | None:
        return {
            "Selected Data": None if self._selected_dataset is None else WorkflowPayload("Selected Data", self._selected_dataset),
            "Annotated Data": None if self._annotated_dataset is None else WorkflowPayload("Annotated Data", self._annotated_dataset),
            "Components": None if self._components_dataset is None else WorkflowPayload("Components", self._components_dataset),
        }

    def serialize_node_state(self) -> dict[str, object]:
        return {
            "features": list(self._selected_feature_names),
            "placement": self._placement_group.checkedId(),
            "color": self._color_combo.currentText(),
            "label": self._label_combo.currentText(),
            "shape": self._shape_combo.currentText(),
            "size": self._size_combo.currentText(),
            "label_selected_only": self._label_selected_only_cb.isChecked(),
            "point_size": self._point_size_slider.value(),
            "opacity": self._opacity_slider.value(),
            "jitter": self._jitter_slider.value(),
            "hide_radius": self._hide_radius_slider.value(),
            "show_regions": self._show_regions_cb.isChecked(),
            "show_legend": self._show_legend_cb.isChecked(),
            "tool_mode": self._canvas._tool_mode,
            "selected_rows": list(self._selected_rows),
        }

    def restore_node_state(self, payload: dict[str, object]) -> None:
        self._selected_feature_names = [str(item) for item in payload.get("features", [])]
        placement = int(payload.get("placement", PLACEMENT_CIRCULAR))
        button = self._placement_group.button(placement) or self._circular_rb
        button.setChecked(True)
        self._select_combo_text(self._color_combo, str(payload.get("color", "")))
        self._select_combo_text(self._label_combo, str(payload.get("label", "")))
        self._select_combo_text(self._shape_combo, str(payload.get("shape", "")))
        self._select_combo_text(self._size_combo, str(payload.get("size", "")))
        self._label_selected_only_cb.setChecked(bool(payload.get("label_selected_only", False)))
        self._point_size_slider.setValue(int(payload.get("point_size", 9)))
        self._opacity_slider.setValue(int(payload.get("opacity", 180)))
        self._jitter_slider.setValue(int(payload.get("jitter", 0)))
        self._hide_radius_slider.setValue(int(payload.get("hide_radius", 0)))
        self._show_regions_cb.setChecked(bool(payload.get("show_regions", False)))
        self._show_legend_cb.setChecked(bool(payload.get("show_legend", True)))
        tool_mode = str(payload.get("tool_mode", SELECT_MODE))
        button = {SELECT_MODE: self._select_btn, PAN_MODE: self._pan_btn, ZOOM_MODE: self._zoom_btn}.get(tool_mode, self._select_btn)
        button.setChecked(True)
        self._canvas.set_tool_mode(tool_mode)
        self._selected_rows = [int(index) for index in payload.get("selected_rows", []) if isinstance(index, int | float)]

    def _make_tool_button(self, label: str, mode: str, checked: bool) -> QToolButton:
        button = QToolButton()
        button.setText(label)
        button.setCheckable(True)
        button.setChecked(checked)
        button.clicked.connect(lambda: self._canvas.set_tool_mode(mode))
        self._tool_group.addButton(button)
        return button

    def _sync_controls(self) -> None:
        dataset = self._dataset
        numeric = numeric_columns(dataset)
        self._sync_feature_list(numeric)
        target_name = dataset.domain.target_columns[0].name if dataset is not None and dataset.domain.target_columns else ""
        self._populate_combo(self._color_combo, ["None", *primitive_columns(dataset)], preferred=target_name or "None")
        self._populate_combo(self._label_combo, ["None", *primitive_columns(dataset)], preferred="None")
        self._populate_combo(self._shape_combo, ["None", *discrete_columns(dataset)], preferred="None")
        self._populate_combo(self._size_combo, ["None", *numeric], preferred="None")
        self._update_lda_state()
        self._update_suggest_button_state()

    def _sync_feature_list(self, names: list[str]) -> None:
        previous = set(self._selected_feature_names)
        if not previous:
            previous = set(names[:3])
        self._feature_list.blockSignals(True)
        self._feature_list.clear()
        for name in names:
            item = QListWidgetItem(name)
            self._feature_list.addItem(item)
            if name in previous:
                item.setSelected(True)
        self._feature_list.blockSignals(False)
        self._filter_feature_list(self._feature_filter_edit.text())
        self._selected_feature_names = self._selected_features()

    def _populate_combo(self, combo: QComboBox, values: list[str], *, preferred: str) -> None:
        normalized: list[str] = []
        for value in values:
            if value not in normalized:
                normalized.append(value)
        current = combo.currentText()
        combo.blockSignals(True)
        combo.clear()
        combo.addItems(normalized)
        desired = current if current in normalized else preferred
        combo.setCurrentIndex(max(0, combo.findText(desired)))
        combo.blockSignals(False)

    def _update_lda_state(self) -> None:
        dataset = self._dataset
        problem = ""
        if dataset is not None:
            targets = list(dataset.domain.target_columns)
            if not targets:
                problem = "Current data has no target variable."
            elif targets[0].logical_type != "categorical":
                problem = f"{targets[0].name} is not categorical."
            else:
                raw = dataset.dataframe.get_column(targets[0].name).to_list()
                distinct = []
                for value in raw:
                    if value is None:
                        continue
                    text = str(value)
                    if text not in distinct:
                        distinct.append(text)
                if len(distinct) < 3:
                    problem = f"Target '{targets[0].name}' needs at least three distinct values for LDA."
        self._placement_info_label.setText(problem)
        self._lda_rb.setEnabled(not problem)
        self._hide_radius_slider.setEnabled(self._placement_group.checkedId() != PLACEMENT_CIRCULAR)
        if problem and self._placement_group.checkedId() == PLACEMENT_LDA:
            self._circular_rb.setChecked(True)

    def _update_suggest_button_state(self) -> None:
        dataset = self._dataset
        enabled = dataset is not None and len(numeric_columns(dataset)) >= 3
        self._suggest_btn.setEnabled(enabled)

    def _filter_feature_list(self, text: str) -> None:
        needle = text.strip().lower()
        for row in range(self._feature_list.count()):
            item = self._feature_list.item(row)
            item.setHidden(bool(needle) and needle not in item.text().lower())

    def _selected_features(self) -> list[str]:
        return [item.text() for item in self._feature_list.selectedItems()]

    def _handle_feature_selection_changed(self) -> None:
        self._selected_feature_names = self._selected_features()
        self._refresh_projection()

    def _handle_projection_changed(self, _id: int) -> None:
        self._update_lda_state()
        self._refresh_projection()

    def _suggest_features(self) -> None:
        dataset = self._dataset
        if dataset is None:
            return
        names = numeric_columns(dataset)
        if len(names) <= 3:
            self._select_feature_names(names)
            return
        color_name = self._color_combo.currentText()
        scores: list[tuple[float, str]] = []
        if color_name != "None" and color_name in dataset.dataframe.columns:
            color_series = dataset.dataframe.get_column(color_name)
            if color_series.dtype.is_numeric():
                targets = color_series.cast(pl.Float64, strict=False).to_numpy().astype(float, copy=False)
                for name in names:
                    values = dataset.dataframe.get_column(name).cast(pl.Float64, strict=False).to_numpy().astype(float, copy=False)
                    valid = np.isfinite(values) & np.isfinite(targets)
                    scores.append((_score_feature_continuous(values[valid], targets[valid]), name))
            else:
                labels = []
                encoded = np.full(dataset.row_count, -1, dtype=int)
                for index, value in enumerate(color_series.to_list()):
                    if value is None:
                        continue
                    text = str(value)
                    if text not in labels:
                        labels.append(text)
                    encoded[index] = labels.index(text)
                for name in names:
                    values = dataset.dataframe.get_column(name).cast(pl.Float64, strict=False).to_numpy().astype(float, copy=False)
                    valid = np.isfinite(values) & (encoded >= 0)
                    scores.append((_score_feature_discrete(values[valid], encoded[valid]), name))
        if not scores:
            scores = [(float(np.nanstd(dataset.dataframe.get_column(name).cast(pl.Float64, strict=False).to_numpy())), name) for name in names]
        selected = [name for _, name in sorted(scores, key=lambda item: (-item[0], item[1]))[:3]]
        self._select_feature_names(selected)

    def _select_feature_names(self, names: list[str]) -> None:
        chosen = set(names)
        self._feature_list.blockSignals(True)
        for row in range(self._feature_list.count()):
            item = self._feature_list.item(row)
            item.setSelected(item.text() in chosen)
        self._feature_list.blockSignals(False)
        self._selected_feature_names = self._selected_features()
        self._refresh_projection()

    def _refresh_projection(self) -> None:
        dataset = self._dataset
        self._selected_dataset = None
        self._annotated_dataset = None
        self._components_dataset = None
        if dataset is None:
            self._canvas.set_projection([], [], [], x_range=(-1.0, 1.0), y_range=(-1.0, 1.0), always_show_axes=True)
            self._status_message = "Load data with numeric features."
            self._sync_status()
            self._notify_output_changed()
            return

        numeric = numeric_columns(dataset)
        if not numeric:
            self._canvas.set_projection([], [], [], x_range=(-1.0, 1.0), y_range=(-1.0, 1.0), always_show_axes=True)
            self._status_message = "Plotting requires numeric features."
            self._sync_status()
            self._notify_output_changed()
            return

        selected = [name for name in self._selected_features() if name in numeric]
        if not selected:
            selected = numeric[:3]
            self._select_feature_names(selected)
            return
        self._selected_feature_names = selected

        row_mask = np.ones(dataset.row_count, dtype=bool)
        feature_arrays: list[np.ndarray] = []
        raw_values_by_feature: list[np.ndarray] = []
        for name in selected:
            values = dataset.dataframe.get_column(name).cast(pl.Float64, strict=False).to_numpy().astype(float, copy=False)
            raw_values_by_feature.append(values)
            row_mask &= np.isfinite(values)
            feature_arrays.append(values)
        row_indices = np.flatnonzero(row_mask).astype(int)
        if row_indices.size < 2:
            self._canvas.set_projection([], [], [], x_range=(-1.0, 1.0), y_range=(-1.0, 1.0), always_show_axes=True)
            self._status_message = "Not enough valid data instances."
            self._sync_status()
            self._notify_output_changed()
            return
        self._effective_rows = row_indices
        features = np.column_stack([values[row_indices] for values in feature_arrays]).astype(float, copy=False)

        placement = self._placement_group.checkedId()
        class_name = self._color_combo.currentText()
        class_labels: list[str] = []
        label_ids = np.zeros(len(row_indices), dtype=int)
        if class_name != "None" and class_name in dataset.dataframe.columns:
            color_series = dataset.dataframe.get_column(class_name)
            if not color_series.dtype.is_numeric():
                ordered: list[str] = []
                labels_for_rows: list[str] = []
                for row_index in row_indices:
                    value = dataset.dataframe[class_name][int(row_index)]
                    label = "(missing)" if value is None else str(value)
                    labels_for_rows.append(label)
                    if label not in ordered:
                        ordered.append(label)
                class_labels = ordered[:8]
                lookup = {label: index for index, label in enumerate(class_labels)}
                label_ids = np.array(
                    [lookup.get(label, 0) for label in labels_for_rows],
                    dtype=int,
                )

        if placement == PLACEMENT_CIRCULAR:
            standardized = features.copy()
            components = _circular_components(len(selected))
            coords = _normalize_embedding(standardized @ components)
            always_show_axes = True
        elif placement == PLACEMENT_PCA:
            coords, components = _pca_projection(features)
            always_show_axes = False
        else:
            target = dataset.domain.target_columns[0] if dataset.domain.target_columns else None
            if target is None or target.logical_type != "categorical":
                self._circular_rb.setChecked(True)
                self._refresh_projection()
                return
            raw_targets = dataset.dataframe.get_column(target.name).to_list()
            ordered = []
            encoded = np.full(dataset.row_count, -1, dtype=int)
            for index, value in enumerate(raw_targets):
                if value is None:
                    continue
                text = str(value)
                if text not in ordered:
                    ordered.append(text)
                encoded[index] = ordered.index(text)
            valid_targets = encoded[row_indices]
            coords, components = _lda_projection(features, valid_targets)
            always_show_axes = False

        projection_components = self._projection_components_override(selected)
        if projection_components is not None:
            components = projection_components
            coords = _normalize_embedding(features @ components)
            always_show_axes = False

        if components.shape[1] == 1:
            components = np.column_stack([components[:, 0], np.zeros(len(selected))])
        self._current_components = components
        self._current_embedding = coords

        combined = np.vstack([coords, components]) if len(components) else coords
        if combined.size == 0:
            x_range = (-1.0, 1.0)
            y_range = (-1.0, 1.0)
        else:
            xmin = float(np.min(combined[:, 0]))
            xmax = float(np.max(combined[:, 0]))
            ymin = float(np.min(combined[:, 1]))
            ymax = float(np.max(combined[:, 1]))
            xpad = (xmax - xmin) * 0.12 or 0.5
            ypad = (ymax - ymin) * 0.12 or 0.5
            x_range = (xmin - xpad, xmax + xpad)
            y_range = (ymin - ypad, ymax + ypad)

        points = self._build_points(selected, row_indices, coords, label_ids, class_labels)
        anchors = [(selected[index], float(components[index, 0]), float(components[index, 1])) for index in range(len(selected))]
        legend_items = self._legend_items()
        self._canvas.set_projection(points, anchors, legend_items, x_range=x_range, y_range=y_range, always_show_axes=always_show_axes)
        self._canvas.set_selected_rows(self._selected_rows)
        self._components_dataset = build_components_dataset(
            selected,
            components[:, 0].tolist(),
            components[:, 1].tolist(),
            dataset_id=f"{dataset.dataset_id}-linear-projection-components",
            display_name=f"{dataset.display_name} (Linear Projection Components)",
            file_name=f"{dataset.dataset_id}-linear-projection-components.csv",
            service=self._builder,
        )
        self._handle_selection_changed(self._selected_rows, notify=False)
        self._status_message = f"{len(row_indices)} instances, {len(selected)} features, {self._placement_name(placement)}"
        self._sync_status()
        self._notify_output_changed()

    def _build_points(
        self,
        selected: list[str],
        row_indices: np.ndarray,
        coords: np.ndarray,
        label_ids: np.ndarray,
        class_labels: list[str],
    ) -> list[_ProjectionPoint]:
        dataset = self._dataset
        if dataset is None:
            return []
        subset_rows = subset_row_indices(dataset, self._subset)
        colors, legends = self._resolve_colors(row_indices)
        shapes = self._resolve_shapes(row_indices)
        sizes = self._resolve_sizes(row_indices)
        labels = self._resolve_labels(row_indices)
        points: list[_ProjectionPoint] = []
        for index, row_index in enumerate(row_indices):
            class_label = class_labels[label_ids[index]] if class_labels and 0 <= label_ids[index] < len(class_labels) else ""
            tooltip_lines = [f"Row: {int(row_index)}"]
            if class_label:
                tooltip_lines.append(f"Class: {class_label}")
            for feature_name in selected:
                value = dataset.dataframe[feature_name][int(row_index)]
                tooltip_lines.append(f"{feature_name}: {value}")
            points.append(
                _ProjectionPoint(
                    row_index=int(row_index),
                    x=float(coords[index, 0]),
                    y=float(coords[index, 1]) if coords.shape[1] > 1 else 0.0,
                    color=colors[index],
                    shape_index=shapes[index],
                    radius=sizes[index],
                    label=labels[index],
                    legend=legends[index],
                    tooltip="\n".join(tooltip_lines),
                    subset=int(row_index) in subset_rows,
                )
            )
        return points

    def _resolve_colors(self, row_indices: np.ndarray) -> tuple[list[QColor], list[str]]:
        dataset = self._dataset
        attr = self._color_combo.currentText()
        if dataset is None or attr == "None" or attr not in dataset.dataframe.columns:
            return [QColor("#3b82f6")] * len(row_indices), [""] * len(row_indices)
        series = dataset.dataframe.get_column(attr)
        if series.dtype.is_numeric():
            values = series.cast(pl.Float64, strict=False).to_numpy().astype(float, copy=False)[row_indices]
            finite = values[np.isfinite(values)]
            low = float(np.min(finite)) if finite.size else 0.0
            high = float(np.max(finite)) if finite.size else 1.0
            colors = []
            for value in values:
                t = 0.5 if abs(high - low) < 1e-9 else max(0.0, min(1.0, (float(value) - low) / (high - low)))
                colors.append(QColor(int(59 + (239 - 59) * t), int(130 + (68 - 130) * t), int(246 + (68 - 246) * t)))
            return colors, [""] * len(colors)
        ordered: list[str] = []
        labels: list[str] = []
        for row_index in row_indices:
            value = dataset.dataframe[attr][int(row_index)]
            label = "(missing)" if value is None else str(value)
            labels.append(label)
            if label not in ordered:
                ordered.append(label)
        lookup = {label: QColor(PALETTE[index % len(PALETTE)]) for index, label in enumerate(ordered)}
        return [QColor(lookup[label]) for label in labels], labels

    def _resolve_shapes(self, row_indices: np.ndarray) -> list[int]:
        dataset = self._dataset
        attr = self._shape_combo.currentText()
        if dataset is None or attr == "None" or attr not in dataset.dataframe.columns:
            return [0] * len(row_indices)
        labels: list[str] = []
        output: list[int] = []
        for row_index in row_indices:
            value = dataset.dataframe[attr][int(row_index)]
            label = "(missing)" if value is None else str(value)
            if label not in labels:
                labels.append(label)
            output.append(labels.index(label) % len(ORANGE_SHAPES))
        return output

    def _resolve_sizes(self, row_indices: np.ndarray) -> list[int]:
        dataset = self._dataset
        attr = self._size_combo.currentText()
        if dataset is None or attr == "None" or attr not in dataset.dataframe.columns:
            return [6] * len(row_indices)
        values = dataset.dataframe.get_column(attr).cast(pl.Float64, strict=False).to_numpy().astype(float, copy=False)[row_indices]
        finite = values[np.isfinite(values)]
        if not finite.size or abs(float(np.max(finite)) - float(np.min(finite))) < 1e-9:
            return [6] * len(values)
        low = float(np.min(finite))
        high = float(np.max(finite))
        scaled = 4 + 6 * ((values - low) / (high - low))
        return [max(4, min(10, int(round(value)))) for value in scaled]

    def _resolve_labels(self, row_indices: np.ndarray) -> list[str]:
        dataset = self._dataset
        attr = self._label_combo.currentText()
        if dataset is None or attr == "None" or attr not in dataset.dataframe.columns:
            return [""] * len(row_indices)
        return ["" if dataset.dataframe[attr][int(row_index)] is None else str(dataset.dataframe[attr][int(row_index)]) for row_index in row_indices]

    def _legend_items(self) -> list[tuple[str, QColor]]:
        dataset = self._dataset
        attr = self._color_combo.currentText()
        if dataset is None or attr == "None" or attr not in dataset.dataframe.columns:
            return []
        series = dataset.dataframe.get_column(attr)
        if series.dtype.is_numeric():
            return []
        labels: list[str] = []
        for value in series.to_list():
            label = "(missing)" if value is None else str(value)
            if label not in labels:
                labels.append(label)
        return [(label, QColor(PALETTE[index % len(PALETTE)])) for index, label in enumerate(labels[:8])]

    def _projection_components_override(self, selected: list[str]) -> np.ndarray | None:
        projection = self._projection_input
        if projection is None:
            return None
        columns = projection.dataframe.columns
        if not {"feature", "component_1", "component_2"}.issubset(columns):
            return None
        lookup = {}
        feature_names = projection.dataframe.get_column("feature").to_list()
        comp1 = projection.dataframe.get_column("component_1").cast(pl.Float64, strict=False).to_list()
        comp2 = projection.dataframe.get_column("component_2").cast(pl.Float64, strict=False).to_list()
        for name, x_value, y_value in zip(feature_names, comp1, comp2):
            if name is None or x_value is None or y_value is None:
                continue
            lookup[str(name)] = (float(x_value), float(y_value))
        if not all(name in lookup for name in selected):
            return None
        return np.asarray([lookup[name] for name in selected], dtype=float)

    def _handle_selection_changed(self, rows: list[int], *, notify: bool = True) -> None:
        dataset = self._dataset
        normalized = sorted({int(index) for index in rows if isinstance(index, int | float)})
        self._selected_rows = [index for index in normalized if dataset is not None and 0 <= index < dataset.row_count]
        self._canvas.set_selected_rows(self._selected_rows)
        self._selection_label.setText(f"Selected: {len(self._selected_rows)}")
        self._selected_dataset, self._annotated_dataset = build_selection_outputs(
            dataset,
            self._selected_rows,
            generated_by="linear-projection",
            service=self._builder,
        )
        if notify:
            self._notify_output_changed()

    def _sync_status(self) -> None:
        self._status_label.setText(self._status_message)

    @staticmethod
    def _placement_name(placement: int) -> str:
        return {
            PLACEMENT_CIRCULAR: "Circular Placement",
            PLACEMENT_LDA: "Linear Discriminant Analysis",
            PLACEMENT_PCA: "Principal Component Analysis",
        }.get(placement, "Projection")

    @staticmethod
    def _select_combo_text(combo: QComboBox, text: str) -> None:
        index = combo.findText(text)
        if index >= 0:
            combo.setCurrentIndex(index)
