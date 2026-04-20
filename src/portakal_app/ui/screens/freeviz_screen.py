from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import polars as pl
from PySide6.QtCore import QPoint, QRect, QRectF, Qt, QTimer, Signal, QSize
from PySide6.QtGui import QColor, QFont, QFontMetrics, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
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
from portakal_app.ui import i18n
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

SELECT_MODE = "select"
PAN_MODE = "pan"
ZOOM_MODE = "zoom"

ORANGE_SHAPES = ("Circle", "Square", "Triangle", "Diamond", "Cross", "Plus")
GRAVITY_VALUES = [0.1, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2, 2.5, 3, 4, 5]


def _allclose_zero(values: np.ndarray, atol: float = 1e-5) -> bool:
    return bool(np.all(np.isclose(values, 0, atol=atol)))


def _rotate_axes(anchors: np.ndarray) -> np.ndarray:
    if anchors.ndim != 2 or anchors.shape[1] != 2 or not len(anchors):
        return anchors
    phi = math.atan2(float(anchors[0, 1]), float(anchors[0, 0]))
    rot = np.array(
        [
            [math.cos(-phi), math.sin(-phi)],
            [-math.sin(-phi), math.cos(-phi)],
        ],
        dtype=float,
    )
    return anchors @ rot


def _init_radial(count: int) -> np.ndarray:
    if count <= 0:
        return np.zeros((0, 2), dtype=float)
    if count == 1:
        angles = np.array([0.0], dtype=float)
    elif count == 2:
        angles = np.array([0.0, math.pi / 2], dtype=float)
    else:
        angles = np.linspace(0.0, 2.0 * math.pi, count, endpoint=False)
    return np.column_stack([np.cos(angles), np.sin(angles)]).astype(float, copy=False)


def _init_random(count: int, seed: int = 0) -> np.ndarray:
    rng = np.random.RandomState(seed)
    return rng.rand(count, 2) * 2.0 - 1.0


def _freeviz_forces_classification(distances: np.ndarray, labels: np.ndarray, gravity: float | None) -> np.ndarray:
    n = len(labels)
    same = labels[:, np.newaxis] == labels[np.newaxis, :]
    forces = -distances.copy()
    eps = np.finfo(float).eps * 100.0
    diff_mask = ~same
    valid = diff_mask & (distances > eps)
    forces[valid] = 1.0 / distances[valid]
    forces[diff_mask & ~valid] = 0.0
    np.fill_diagonal(forces, 0.0)
    if gravity is not None:
        repulsive_sum = float(np.sum(forces[diff_mask]))
        attractive_sum = float(np.sum(forces[same & ~np.eye(n, dtype=bool)]))
        if abs(repulsive_sum) > eps:
            forces[diff_mask] *= -attractive_sum / repulsive_sum / gravity
    return forces


def _freeviz_forces_regression(distances: np.ndarray, targets: np.ndarray) -> np.ndarray:
    diff = targets[:, np.newaxis] - targets[np.newaxis, :]
    ydist = diff * diff
    eps = np.finfo(float).eps * 100.0
    mask = distances > eps
    forces = ydist.copy()
    forces[mask] /= distances[mask]
    forces[~mask] = 0.0
    np.fill_diagonal(forces, 0.0)
    return forces


def _freeviz_gradient(
    features: np.ndarray,
    targets: np.ndarray,
    embedding: np.ndarray,
    *,
    gravity: float | None,
    is_class_discrete: bool,
) -> np.ndarray:
    diffs = embedding[:, np.newaxis, :] - embedding[np.newaxis, :, :]
    distances = np.linalg.norm(diffs, axis=2)
    direction = np.zeros_like(diffs)
    mask = distances > np.finfo(float).eps * 100.0
    direction[mask] = diffs[mask] / distances[mask][:, np.newaxis]
    if is_class_discrete:
        forces = _freeviz_forces_classification(distances, targets.astype(int, copy=False), gravity)
    else:
        forces = _freeviz_forces_regression(distances, targets.astype(float, copy=False))
    net = np.sum(direction * forces[:, :, np.newaxis], axis=0)
    return features.T @ net


def _optimize_anchors(
    features: np.ndarray,
    targets: np.ndarray,
    anchors: np.ndarray,
    *,
    alpha: float,
    gravity: float | None,
    is_class_discrete: bool,
    steps: int,
    atol: float = 1e-5,
) -> tuple[np.ndarray, bool]:
    current = anchors.astype(float, copy=True)
    converged = False
    for _ in range(max(1, steps)):
        embedding = features @ current
        gradient = _freeviz_gradient(
            features,
            targets,
            embedding,
            gravity=gravity,
            is_class_discrete=is_class_discrete,
        )
        grad_norm = np.linalg.norm(gradient, axis=1)
        anchor_norm = np.linalg.norm(current, axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            ratios = anchor_norm / grad_norm
        finite = ratios[np.isfinite(ratios)]
        if finite.size == 0:
            converged = True
            break
        step = alpha * float(np.min(finite))
        updated = current - step * gradient
        updated = updated - np.mean(updated, axis=0, keepdims=True)
        max_radius = float(np.max(np.linalg.norm(updated, axis=1)))
        if max_radius >= 1e-3:
            updated /= max_radius
        change = np.linalg.norm(updated - current, axis=1)
        current = updated
        if _allclose_zero(change, atol=atol):
            converged = True
            break
    return _rotate_axes(current), converged


@dataclass(frozen=True)
class _PointVisual:
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


class _FreeVizCanvas(QWidget):
    anchorMoved = Signal(int, float, float)
    selectionChanged = Signal(list)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._points: list[_PointVisual] = []
        self._anchors: list[tuple[str, float, float]] = []
        self._legend_items: list[tuple[str, QColor]] = []
        self._show_regions = False
        self._show_legend = True
        self._label_selected_only = False
        self._selected_rows: set[int] = set()
        self._point_hit_regions: list[tuple[QRect, int, str]] = []
        self._anchor_regions: list[tuple[QRect, int]] = []
        self._selection_rect: QRect | None = None
        self._drag_origin: QPoint | None = None
        self._drag_anchor: int | None = None
        self._tool_mode = SELECT_MODE
        self._zoom = 1.0
        self._pan_x = 0.0
        self._pan_y = 0.0
        self._point_size = 9
        self._opacity = 180
        self._jitter = 0.0
        self._hide_radius = 0.0
        self._chart_center = QPoint()
        self._chart_radius = 120.0

        self.setMouseTracking(True)
        self.setMinimumSize(560, 420)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

    def set_projection(
        self,
        points: list[_PointVisual],
        anchors: list[tuple[str, float, float]],
        legend_items: list[tuple[str, QColor]],
    ) -> None:
        self._points = points
        self._anchors = anchors
        self._legend_items = legend_items
        self._point_hit_regions = []
        self._anchor_regions = []
        self.update()

    def set_selected_rows(self, rows: list[int]) -> None:
        self._selected_rows = set(rows)
        self.update()

    def set_tool_mode(self, mode: str) -> None:
        self._tool_mode = mode

    def set_point_size(self, value: int) -> None:
        self._point_size = max(3, int(value))
        self.update()

    def set_opacity(self, value: int) -> None:
        self._opacity = max(40, min(255, int(value)))
        self.update()

    def set_jitter(self, value: float) -> None:
        self._jitter = max(0.0, float(value))
        self.update()

    def set_hide_radius(self, percent: int) -> None:
        self._hide_radius = max(0.0, min(1.0, percent / 100.0))
        self.update()

    def set_show_color_regions(self, checked: bool) -> None:
        self._show_regions = checked
        self.update()

    def set_show_legend(self, checked: bool) -> None:
        self._show_legend = checked
        self.update()

    def set_label_selected_only(self, checked: bool) -> None:
        self._label_selected_only = checked
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
        pos = event.position().toPoint()
        anchor_index = self._anchor_index_at(pos)
        if anchor_index is not None:
            self._drag_anchor = anchor_index
            return
        self._drag_origin = pos
        if self._tool_mode in {SELECT_MODE, ZOOM_MODE}:
            self._selection_rect = QRect(pos, pos)

    def mouseMoveEvent(self, event) -> None:
        pos = event.position().toPoint()
        if event.buttons() & Qt.MouseButton.LeftButton:
            if self._drag_anchor is not None:
                self._move_anchor(self._drag_anchor, pos)
                QToolTip.hideText()
                return
            if self._drag_origin is not None:
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
        if self._drag_anchor is not None:
            self._drag_anchor = None
            return
        selection_rect = self._selection_rect.normalized() if self._selection_rect is not None else None
        self._selection_rect = None
        if self._drag_origin is None:
            return
        drag_origin = self._drag_origin
        self._drag_origin = None
        if selection_rect is None:
            return
        if selection_rect.width() < 4 and selection_rect.height() < 4:
            if self._tool_mode == SELECT_MODE:
                row = self._row_at(event.position().toPoint())
                self.selectionChanged.emit([] if row is None else [row])
            self.update()
            return
        if self._tool_mode == SELECT_MODE:
            rows = [row for rect, row, _ in self._point_hit_regions if selection_rect.intersects(rect)]
            self.selectionChanged.emit(sorted(set(rows)))
        elif self._tool_mode == ZOOM_MODE:
            self._zoom_to_rect(selection_rect, drag_origin)
        self.update()

    def mouseDoubleClickEvent(self, _event) -> None:
        self.reset_view()

    def leaveEvent(self, _event) -> None:
        QToolTip.hideText()

    def sizeHint(self) -> QSize:
        return QSize(760, 620)

    def paintEvent(self, _event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), QColor("#fffdf9"))

        if not self._anchors:
            painter.setPen(QColor("#8d877d"))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "No projection available.")
            return

        legend_width = 160 if self._show_legend and self._legend_items else 26
        margin = 48
        width = max(60, self.width() - legend_width - margin * 2)
        height = max(60, self.height() - margin * 2)
        radius = min(width, height) / 2 * self._zoom
        center_x = margin + width / 2 + self._pan_x
        center_y = margin + height / 2 + self._pan_y
        self._chart_center = QPoint(int(center_x), int(center_y))
        self._chart_radius = radius

        self._point_hit_regions = []
        self._anchor_regions = []

        if self._show_regions and self._legend_items:
            self._draw_regions(painter)

        painter.setPen(QPen(QColor("#d7d1c7"), 1, Qt.PenStyle.DotLine))
        for ring in (0.25, 0.5, 0.75):
            ring_radius = int(radius * ring)
            painter.drawEllipse(int(center_x - ring_radius), int(center_y - ring_radius), ring_radius * 2, ring_radius * 2)

        painter.setPen(QPen(QColor("#c0bbb2"), 1.2))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawEllipse(int(center_x - radius), int(center_y - radius), int(radius * 2), int(radius * 2))

        if self._hide_radius > 0:
            inner = int(radius * self._hide_radius)
            painter.setPen(QPen(QColor(145, 135, 125, 130), 1.2, Qt.PenStyle.DashLine))
            painter.drawEllipse(int(center_x - inner), int(center_y - inner), inner * 2, inner * 2)

        painter.setFont(QFont(self.font().family(), 8))
        fm = QFontMetrics(painter.font())

        for point in self._points:
            px = center_x + point.x * radius
            py = center_y - point.y * radius
            radius_px = max(3, int(round(point.radius * self._point_size / 10.0)))

            if point.subset:
                painter.setPen(QPen(QColor("#111111"), 1.8))
                painter.setBrush(QColor("#ffffff"))
                self._draw_marker(painter, point.shape_index, px, py, radius_px + 2)

            fill = QColor(point.color)
            fill.setAlpha(self._opacity)
            border = QColor("#111111") if point.row_index in self._selected_rows else QColor(fill.darker(135))
            painter.setPen(QPen(border, 1.4))
            painter.setBrush(fill)
            self._draw_marker(painter, point.shape_index, px, py, radius_px)

            if point.label and (
                not self._label_selected_only
                or point.row_index in self._selected_rows
                or point.subset
            ):
                painter.setPen(QColor("#3b2a10"))
                painter.drawText(
                    QRect(int(px + radius_px + 5), int(py - 10), 140, 20),
                    Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
                    point.label,
                )

            rect = QRect(int(px - radius_px - 4), int(py - radius_px - 4), radius_px * 2 + 8, radius_px * 2 + 8)
            self._point_hit_regions.append((rect, point.row_index, point.tooltip))

        hide_threshold = self._hide_radius + 1e-5
        for index, (name, ax, ay) in enumerate(self._anchors):
            length = math.sqrt(ax * ax + ay * ay)
            if length <= hide_threshold:
                continue
            px = center_x + ax * radius
            py = center_y - ay * radius
            painter.setPen(QPen(QColor("#7a7369"), 1.5))
            painter.drawLine(int(center_x), int(center_y), int(px), int(py))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QColor("#e07020"))
            painter.drawEllipse(int(px - 5), int(py - 5), 10, 10)
            label_x = center_x + ax * (radius + 22)
            label_y = center_y - ay * (radius + 22)
            painter.setPen(QColor("#3b2a10"))
            painter.drawText(
                QRect(int(label_x - 48), int(label_y - 8), 96, 16),
                Qt.AlignmentFlag.AlignCenter,
                fm.elidedText(name, Qt.TextElideMode.ElideRight, 94),
            )
            self._anchor_regions.append((QRect(int(px - 10), int(py - 10), 20, 20), index))

        if self._selection_rect is not None:
            painter.setPen(QPen(QColor("#2563eb"), 1.5, Qt.PenStyle.DashLine))
            painter.setBrush(QColor(37, 99, 235, 30))
            painter.drawRect(self._selection_rect)

        if self._show_legend and self._legend_items:
            painter.setPen(QColor("#534b40"))
            legend_left = self.width() - legend_width + 8
            painter.drawText(QRect(legend_left, 12, legend_width - 16, 18), Qt.AlignmentFlag.AlignLeft, "Legend")
            y = 38
            for label, color in self._legend_items[:8]:
                painter.setPen(Qt.PenStyle.NoPen)
                painter.setBrush(color)
                painter.drawEllipse(legend_left, y + 2, 11, 11)
                painter.setPen(QColor("#534b40"))
                painter.drawText(
                    QRect(legend_left + 16, y, legend_width - 24, 18),
                    Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
                    fm.elidedText(label, Qt.TextElideMode.ElideRight, legend_width - 28),
                )
                y += 20

        painter.setPen(QColor(160, 155, 145, 160))
        painter.setFont(QFont(self.font().family(), 7))
        painter.drawText(6, self.height() - 6, f"scroll=zoom  drag-anchor=move  mode={self._tool_mode}")
        painter.end()

    def _draw_regions(self, painter: QPainter) -> None:
        color_lookup = {label: color for label, color in self._legend_items}
        bins = 18
        chart_radius = self._chart_radius
        center_x = self._chart_center.x()
        center_y = self._chart_center.y()
        density: dict[tuple[int, int], dict[str, int]] = {}
        for point in self._points:
            if not point.legend:
                continue
            ix = int((point.x + 1.0) / 2.0 * bins)
            iy = int((point.y + 1.0) / 2.0 * bins)
            ix = max(0, min(bins - 1, ix))
            iy = max(0, min(bins - 1, iy))
            density.setdefault((ix, iy), {}).setdefault(point.legend, 0)
            density[(ix, iy)][point.legend] += 1
        if not density:
            return
        max_density = max(sum(items.values()) for items in density.values())
        cell = chart_radius * 2 / bins
        painter.save()
        painter.setPen(Qt.PenStyle.NoPen)
        for (ix, iy), groups in density.items():
            label = max(groups.items(), key=lambda item: item[1])[0]
            color = QColor(color_lookup.get(label, QColor("#94a3b8")))
            color.setAlpha(int(18 + 78 * (sum(groups.values()) / max_density)))
            left = center_x - chart_radius + ix * cell
            top = center_y + chart_radius - (iy + 1) * cell
            painter.setBrush(color)
            painter.drawRect(QRectF(left, top, cell + 1, cell + 1))
        painter.restore()

    def _anchor_index_at(self, pos: QPoint) -> int | None:
        for rect, index in self._anchor_regions:
            if rect.contains(pos):
                return index
        return None

    def _row_at(self, pos: QPoint) -> int | None:
        for rect, row, _ in reversed(self._point_hit_regions):
            if rect.contains(pos):
                return row
        return None

    def _move_anchor(self, index: int, pos: QPoint) -> None:
        radius = self._chart_radius or 1.0
        dx = (pos.x() - self._chart_center.x()) / radius
        dy = (self._chart_center.y() - pos.y()) / radius
        self.anchorMoved.emit(index, float(dx), float(dy))

    def _zoom_to_rect(self, rect: QRect, origin: QPoint) -> None:
        if rect.width() < 10 or rect.height() < 10:
            return
        factor = min(self.width() / rect.width(), self.height() / rect.height())
        self._zoom = max(0.2, min(8.0, self._zoom * factor * 0.7))
        center = rect.center()
        self._pan_x += origin.x() - center.x()
        self._pan_y += origin.y() - center.y()

    @staticmethod
    def _draw_marker(painter: QPainter, shape_index: int, px: float, py: float, radius: int) -> None:
        if shape_index == 1:
            painter.drawRect(int(px - radius), int(py - radius), radius * 2, radius * 2)
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
            painter.drawLine(int(px - radius), int(py - radius), int(px + radius), int(py + radius))
            painter.drawLine(int(px - radius), int(py + radius), int(px + radius), int(py - radius))
            return
        if shape_index == 5:
            painter.drawLine(int(px - radius), int(py), int(px + radius), int(py))
            painter.drawLine(int(px), int(py - radius), int(px), int(py + radius))
            return
        painter.drawEllipse(int(px - radius), int(py - radius), radius * 2, radius * 2)


class FreeVizScreen(QWidget, WorkflowNodeScreenSupport):
    MAX_INSTANCES = 10000

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._init_workflow_node_support()
        self._dataset: DatasetHandle | None = None
        self._subset: DatasetHandle | None = None
        self._selected_rows: list[int] = []
        self._selected_dataset: DatasetHandle | None = None
        self._annotated_dataset: DatasetHandle | None = None
        self._components_dataset: DatasetHandle | None = None
        self._builder = GeneratedDatasetService()

        self._effective_feature_names: list[str] = []
        self._row_indices = np.asarray([], dtype=int)
        self._feature_matrix: np.ndarray | None = None
        self._target_values: np.ndarray | None = None
        self._target_name = ""
        self._target_is_discrete = False
        self._target_labels: list[str] = []
        self._anchors: np.ndarray | None = None
        self._embedding: np.ndarray | None = None
        self._removed_features: list[str] = []
        self._status_message = "Load a dataset with a target variable."
        self._warning_message = ""
        self._optimized_once = False
        self._can_resume = False
        self._iteration_count = 0

        self._timer = QTimer(self)
        self._timer.setInterval(35)
        self._timer.timeout.connect(self._run_batch)

        root = QHBoxLayout(self)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(12)

        left_scroll = QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        left_scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        left_scroll.setFixedWidth(328)
        left_panel = QWidget()
        left_scroll.setWidget(left_panel)
        root.addWidget(left_scroll, 0)

        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(10)

        optimize_box = QGroupBox("Optimize")
        optimize_form = QFormLayout(optimize_box)
        optimize_form.setContentsMargins(10, 12, 10, 10)
        optimize_form.setSpacing(8)
        self._init_combo = QComboBox()
        self._init_combo.addItems(["Circular", "Random"])
        self._init_combo.currentIndexChanged.connect(self._handle_projection_control_changed)
        optimize_form.addRow("Initialization:", self._init_combo)

        gravity_row = QWidget()
        gravity_layout = QHBoxLayout(gravity_row)
        gravity_layout.setContentsMargins(0, 0, 0, 0)
        gravity_layout.setSpacing(6)
        self._gravity_cb = QCheckBox("Gravity")
        self._gravity_cb.toggled.connect(self._handle_gravity_changed)
        gravity_layout.addWidget(self._gravity_cb)
        self._grav_slider = QSlider(Qt.Orientation.Horizontal)
        self._grav_slider.setRange(0, len(GRAVITY_VALUES) - 1)
        self._grav_slider.setValue(4)
        self._grav_slider.valueChanged.connect(self._handle_gravity_slider_changed)
        gravity_layout.addWidget(self._grav_slider, 1)
        self._grav_label = QLabel(str(GRAVITY_VALUES[self._grav_slider.value()]))
        self._grav_label.setFixedWidth(32)
        self._grav_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        gravity_layout.addWidget(self._grav_label)
        optimize_form.addRow(gravity_row)

        self._run_btn = QPushButton("Start")
        self._run_btn.clicked.connect(self._toggle_run)
        optimize_form.addRow(self._run_btn)
        left_layout.addWidget(optimize_box)

        attributes_box = QGroupBox("Display")
        attributes_form = QFormLayout(attributes_box)
        attributes_form.setContentsMargins(10, 12, 10, 10)
        attributes_form.setSpacing(8)
        self._color_combo = QComboBox()
        self._color_combo.currentTextChanged.connect(self._refresh_visuals)
        attributes_form.addRow("Color:", self._color_combo)
        self._label_combo = QComboBox()
        self._label_combo.currentTextChanged.connect(self._refresh_visuals)
        attributes_form.addRow("Label:", self._label_combo)
        self._shape_combo = QComboBox()
        self._shape_combo.currentTextChanged.connect(self._refresh_visuals)
        attributes_form.addRow("Shape:", self._shape_combo)
        self._size_combo = QComboBox()
        self._size_combo.currentTextChanged.connect(self._refresh_visuals)
        attributes_form.addRow("Size:", self._size_combo)
        self._label_only_selected_cb = QCheckBox("Label only selection and subset")
        self._label_only_selected_cb.toggled.connect(lambda checked: self._canvas.set_label_selected_only(checked))
        attributes_form.addRow(self._label_only_selected_cb)
        left_layout.addWidget(attributes_box)

        effects_box = QGroupBox("Plot")
        effects_form = QFormLayout(effects_box)
        effects_form.setContentsMargins(10, 12, 10, 10)
        effects_form.setSpacing(8)
        self._point_size_slider = QSlider(Qt.Orientation.Horizontal)
        self._point_size_slider.setRange(4, 18)
        self._point_size_slider.setValue(9)
        effects_form.addRow("Symbol size:", self._point_size_slider)
        self._opacity_slider = QSlider(Qt.Orientation.Horizontal)
        self._opacity_slider.setRange(40, 255)
        self._opacity_slider.setValue(180)
        effects_form.addRow("Opacity:", self._opacity_slider)
        self._jitter_slider = QSlider(Qt.Orientation.Horizontal)
        self._jitter_slider.setRange(0, 50)
        self._jitter_slider.setValue(0)
        effects_form.addRow("Jittering:", self._jitter_slider)
        self._hide_radius_slider = QSlider(Qt.Orientation.Horizontal)
        self._hide_radius_slider.setRange(0, 100)
        self._hide_radius_slider.setValue(0)
        effects_form.addRow("Hide radius:", self._hide_radius_slider)
        self._regions_cb = QCheckBox("Show color regions")
        effects_form.addRow(self._regions_cb)
        self._legend_cb = QCheckBox("Show legend")
        self._legend_cb.setChecked(True)
        effects_form.addRow(self._legend_cb)
        left_layout.addWidget(effects_box)

        tools_box = QGroupBox("Tools")
        tools_layout = QHBoxLayout(tools_box)
        tools_layout.setContentsMargins(10, 12, 10, 10)
        tools_layout.setSpacing(6)
        self._tool_group = QButtonGroup(self)
        self._select_btn = self._make_tool_button("Select", SELECT_MODE, True)
        self._pan_btn = self._make_tool_button("Pan", PAN_MODE, False)
        self._zoom_btn = self._make_tool_button("Zoom", ZOOM_MODE, False)
        tools_layout.addWidget(self._select_btn)
        tools_layout.addWidget(self._pan_btn)
        tools_layout.addWidget(self._zoom_btn)
        self._reset_zoom_btn = QPushButton("Zoom to fit")
        tools_layout.addWidget(self._reset_zoom_btn)
        left_layout.addWidget(tools_box)

        self._warning_label = QLabel("")
        self._warning_label.setWordWrap(True)
        self._warning_label.setStyleSheet("color: #92400e;")
        left_layout.addWidget(self._warning_label)

        self._status_label = QLabel(self._status_message)
        self._status_label.setWordWrap(True)
        self._status_label.setStyleSheet("color: #534b40;")
        left_layout.addWidget(self._status_label)

        self._selection_label = QLabel("Selected: 0")
        left_layout.addWidget(self._selection_label)
        left_layout.addStretch(1)

        right_layout = QVBoxLayout()
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(8)
        self._canvas = _FreeVizCanvas()
        self._canvas.anchorMoved.connect(self._handle_anchor_moved)
        self._canvas.selectionChanged.connect(self._handle_selection_changed)
        right_layout.addWidget(self._canvas, 1)
        root.addLayout(right_layout, 1)

        self._point_size_slider.valueChanged.connect(self._canvas.set_point_size)
        self._opacity_slider.valueChanged.connect(self._canvas.set_opacity)
        self._jitter_slider.valueChanged.connect(lambda value: self._canvas.set_jitter(value / 50.0))
        self._hide_radius_slider.valueChanged.connect(self._canvas.set_hide_radius)
        self._regions_cb.toggled.connect(self._canvas.set_show_color_regions)
        self._legend_cb.toggled.connect(self._canvas.set_show_legend)
        self._reset_zoom_btn.clicked.connect(self._canvas.reset_view)

    def documentation_url(self) -> str:
        return "https://orangedatamining.com/widget-catalog/visualize/freeviz/"

    def sizeHint(self) -> QSize:
        return QSize(1132, 708)

    def set_input_payload(self, payload) -> None:
        if payload is None:
            self._dataset = None
            self._subset = None
        elif payload.port_label == "Data":
            self._dataset = payload.dataset
        elif payload.port_label == "Data Subset":
            self._subset = payload.dataset
        self._sync_controls()
        self._reset_projection()

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
            "initialization": self._init_combo.currentText(),
            "gravity": self._gravity_cb.isChecked(),
            "gravity_index": self._grav_slider.value(),
            "color": self._color_combo.currentText(),
            "label": self._label_combo.currentText(),
            "shape": self._shape_combo.currentText(),
            "size": self._size_combo.currentText(),
            "label_selected_only": self._label_only_selected_cb.isChecked(),
            "point_size": self._point_size_slider.value(),
            "opacity": self._opacity_slider.value(),
            "jitter": self._jitter_slider.value(),
            "hide_radius": self._hide_radius_slider.value(),
            "show_regions": self._regions_cb.isChecked(),
            "show_legend": self._legend_cb.isChecked(),
            "tool_mode": self._canvas._tool_mode,
            "selected_rows": list(self._selected_rows),
        }

    def restore_node_state(self, payload: dict[str, object]) -> None:
        self._select_combo_text(self._init_combo, str(payload.get("initialization", "Circular")))
        self._gravity_cb.setChecked(bool(payload.get("gravity", False)))
        self._grav_slider.setValue(int(payload.get("gravity_index", 4)))
        self._pending_combo_value(self._color_combo, str(payload.get("color", "")))
        self._pending_combo_value(self._label_combo, str(payload.get("label", "")))
        self._pending_combo_value(self._shape_combo, str(payload.get("shape", "")))
        self._pending_combo_value(self._size_combo, str(payload.get("size", "")))
        self._label_only_selected_cb.setChecked(bool(payload.get("label_selected_only", False)))
        self._point_size_slider.setValue(int(payload.get("point_size", 9)))
        self._opacity_slider.setValue(int(payload.get("opacity", 180)))
        self._jitter_slider.setValue(int(payload.get("jitter", 0)))
        self._hide_radius_slider.setValue(int(payload.get("hide_radius", 0)))
        self._regions_cb.setChecked(bool(payload.get("show_regions", False)))
        self._legend_cb.setChecked(bool(payload.get("show_legend", True)))
        tool_mode = str(payload.get("tool_mode", SELECT_MODE))
        button = {
            SELECT_MODE: self._select_btn,
            PAN_MODE: self._pan_btn,
            ZOOM_MODE: self._zoom_btn,
        }.get(tool_mode, self._select_btn)
        button.setChecked(True)
        self._canvas.set_tool_mode(tool_mode)
        self._selected_rows = [
            int(index)
            for index in payload.get("selected_rows", [])
            if isinstance(index, int | float)
        ]

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
        primitive = primitive_columns(dataset)
        numeric = numeric_columns(dataset)
        discrete = discrete_columns(dataset)
        target_name = dataset.domain.target_columns[0].name if dataset is not None and dataset.domain.target_columns else ""
        self._populate_combo(self._color_combo, ["(Class)", "None", *primitive], preferred=target_name or "(Class)")
        self._populate_combo(self._label_combo, ["None", *primitive], preferred="None")
        self._populate_combo(self._shape_combo, ["None", *discrete], preferred="None")
        self._populate_combo(self._size_combo, ["None", *numeric], preferred="None")

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
        index = combo.findText(desired)
        combo.setCurrentIndex(max(0, index))
        combo.blockSignals(False)

    def _reset_projection(self) -> None:
        self._timer.stop()
        self._optimized_once = False
        self._can_resume = False
        self._iteration_count = 0
        self._set_run_button_text("Start")
        self._selected_rows = []
        self._canvas.reset_view()
        self._rebuild_projection()

    def _handle_projection_control_changed(self) -> None:
        was_running = self._timer.isActive()
        self._reset_projection()
        if was_running and self._anchors is not None:
            self._start_run()

    def _handle_gravity_slider_changed(self, value: int) -> None:
        self._grav_label.setText(str(GRAVITY_VALUES[value]))
        if not self._gravity_cb.isChecked():
            self._gravity_cb.setChecked(True)
            return
        self._handle_gravity_changed()

    def _handle_gravity_changed(self) -> None:
        self._grav_label.setText(str(GRAVITY_VALUES[self._grav_slider.value()]))
        if self._optimized_once and not self._timer.isActive() and self._anchors is not None:
            self._start_run()

    def _toggle_run(self) -> None:
        if self._timer.isActive():
            self._timer.stop()
            self._can_resume = True
            self._set_run_button_text("Resume")
            return
        self._start_run()

    def _start_run(self) -> None:
        if self._anchors is None or self._feature_matrix is None or self._target_values is None:
            return
        self._timer.start()
        self._set_run_button_text("Stop")

    def _run_batch(self) -> None:
        if self._anchors is None or self._feature_matrix is None or self._target_values is None:
            self._timer.stop()
            self._set_run_button_text("Start")
            return
        gravity = GRAVITY_VALUES[self._grav_slider.value()] if self._gravity_cb.isChecked() else None
        self._anchors, converged = _optimize_anchors(
            self._feature_matrix,
            self._target_values,
            self._anchors,
            alpha=0.1,
            gravity=gravity,
            is_class_discrete=self._target_is_discrete,
            steps=10,
        )
        self._iteration_count += 10
        self._optimized_once = True
        self._can_resume = False
        self._update_visuals()
        if converged:
            self._timer.stop()
            self._set_run_button_text("Start")

    def _rebuild_projection(self) -> None:
        self._selected_dataset = None
        self._annotated_dataset = None
        self._components_dataset = None
        self._feature_matrix = None
        self._target_values = None
        self._anchors = None
        self._embedding = None
        self._row_indices = np.asarray([], dtype=int)
        self._effective_feature_names = []
        self._target_labels = []
        self._removed_features = []
        self._status_message = "Load a dataset with a target variable."
        self._warning_message = ""

        dataset = self._dataset
        if dataset is None:
            self._canvas.set_projection([], [], [])
            self._sync_status_labels()
            self._notify_output_changed()
            return

        if not dataset.domain.target_columns:
            self._status_message = "Data must have a target variable."
            self._canvas.set_projection([], [], [])
            self._sync_status_labels()
            self._notify_output_changed()
            return
        if len(dataset.domain.target_columns) > 1:
            self._status_message = "Data must have a single target variable."
            self._canvas.set_projection([], [], [])
            self._sync_status_labels()
            self._notify_output_changed()
            return

        target = dataset.domain.target_columns[0]
        self._target_name = target.name
        target_series = dataset.dataframe.get_column(target.name)
        target_values_raw = target_series.to_list()

        feature_columns = list(dataset.domain.feature_columns)
        encoded_columns: list[np.ndarray] = []
        feature_names: list[str] = []
        removed: list[str] = []
        valid_mask = np.ones(dataset.row_count, dtype=bool)
        for column in feature_columns:
            series = dataset.dataframe.get_column(column.name)
            if column.logical_type == "numeric":
                values = series.cast(pl.Float64, strict=False).to_numpy().astype(float, copy=False)
                finite = np.isfinite(values)
                valid_mask &= finite
                valid_values = values[finite]
                if valid_values.size:
                    minimum = float(np.min(valid_values))
                    span = float(np.max(valid_values) - minimum)
                    if span > 1e-12:
                        values = (values - minimum) / span
                    else:
                        values = values - minimum
                encoded_columns.append(values)
                feature_names.append(column.name)
                continue
            if column.logical_type not in {"categorical", "boolean", "string", "text"}:
                removed.append(column.name)
                continue
            labels = []
            for value in series.to_list():
                if value is None:
                    continue
                text = str(value)
                if text not in labels:
                    labels.append(text)
            if len(labels) > 2:
                removed.append(column.name)
                continue
            mapping = {label: float(index) for index, label in enumerate(labels)}
            values = np.full(dataset.row_count, np.nan, dtype=float)
            for index, value in enumerate(series.to_list()):
                if value is None:
                    continue
                values[index] = mapping[str(value)]
            valid_mask &= np.isfinite(values)
            encoded_columns.append(values)
            feature_names.append(column.name)

        self._removed_features = removed
        if len(feature_names) < 2:
            self._status_message = "At least two features are required."
            if removed:
                removed_text = ", ".join(removed[:5])
                suffix = "..." if len(removed) > 5 else ""
                self._warning_message = f"Categorical features with more than two values are not shown: {removed_text}{suffix}"
            self._canvas.set_projection([], [], [])
            self._sync_status_labels()
            self._notify_output_changed()
            return

        if target.logical_type == "numeric":
            target_values = target_series.cast(pl.Float64, strict=False).to_numpy().astype(float, copy=False)
            target_mask = np.isfinite(target_values)
            self._target_is_discrete = False
            self._target_labels = []
        else:
            labels: list[str] = []
            for value in target_values_raw:
                if value is None:
                    continue
                text = str(value)
                if text not in labels:
                    labels.append(text)
            if len(labels) < 2:
                self._status_message = "Target variable must have at least two unique values."
                self._canvas.set_projection([], [], [])
                self._sync_status_labels()
                self._notify_output_changed()
                return
            mapping = {label: index for index, label in enumerate(labels)}
            target_values = np.full(dataset.row_count, np.nan, dtype=float)
            for index, value in enumerate(target_values_raw):
                if value is None:
                    continue
                target_values[index] = float(mapping[str(value)])
            target_mask = np.isfinite(target_values)
            self._target_is_discrete = True
            self._target_labels = labels[:8]

        valid_mask &= target_mask
        row_indices = np.flatnonzero(valid_mask).astype(int)
        if row_indices.size < 2:
            self._status_message = "At least two data instances are required."
            self._canvas.set_projection([], [], [])
            self._sync_status_labels()
            self._notify_output_changed()
            return
        if row_indices.size > self.MAX_INSTANCES:
            self._status_message = "Data is too large."
            self._canvas.set_projection([], [], [])
            self._sync_status_labels()
            self._notify_output_changed()
            return
        if len(feature_names) > row_indices.size:
            self._status_message = "Number of features exceeds the number of instances."
            self._canvas.set_projection([], [], [])
            self._sync_status_labels()
            self._notify_output_changed()
            return

        feature_matrix = np.column_stack([column[row_indices] for column in encoded_columns]).astype(float, copy=False)
        if not np.sum(np.std(feature_matrix, axis=0)):
            self._status_message = "All data columns are constant."
            self._canvas.set_projection([], [], [])
            self._sync_status_labels()
            self._notify_output_changed()
            return

        self._row_indices = row_indices
        self._feature_matrix = feature_matrix
        self._target_values = target_values[row_indices]
        self._effective_feature_names = feature_names
        self._anchors = _init_radial(len(feature_names)) if self._init_combo.currentText() == "Circular" else _init_random(len(feature_names))

        if removed:
            removed_text = ", ".join(removed[:5])
            suffix = "..." if len(removed) > 5 else ""
            self._warning_message = f"Categorical features with more than two values are not shown: {removed_text}{suffix}"

        self._update_visuals()

    def _update_visuals(self) -> None:
        if self._anchors is None or self._feature_matrix is None or self._target_values is None:
            self._canvas.set_projection([], [], [])
            self._selected_dataset = None
            self._annotated_dataset = None
            self._components_dataset = None
            self._sync_status_labels()
            self._notify_output_changed()
            return

        embedding = self._feature_matrix @ self._anchors
        norms = np.linalg.norm(embedding, axis=1)
        max_norm = float(np.max(norms)) if norms.size else 1.0
        if max_norm < 1e-9:
            max_norm = 1.0
        normalized = embedding / max_norm
        self._embedding = normalized

        subset_rows = subset_row_indices(self._dataset, self._subset)
        colors, legends = self._resolve_colors()
        shapes = self._resolve_shapes()
        sizes = self._resolve_sizes()
        labels = self._resolve_labels()
        legend_items = self._legend_items_for_current_color()

        points: list[_PointVisual] = []
        for index, row_index in enumerate(self._row_indices):
            tooltip_lines = [
                f"Row: {row_index}",
                f"{self._target_name}: {self._tooltip_value(self._target_name, row_index)}",
            ]
            feature_order = np.argsort(-np.abs(self._feature_matrix[index]))[:3]
            for feature_idx in feature_order:
                tooltip_lines.append(f"{self._effective_feature_names[feature_idx]}: {self._feature_matrix[index, feature_idx]:.3f}")
            points.append(
                _PointVisual(
                    row_index=int(row_index),
                    x=float(normalized[index, 0]),
                    y=float(normalized[index, 1]),
                    color=colors[index],
                    shape_index=shapes[index],
                    radius=sizes[index],
                    label=labels[index],
                    legend=legends[index],
                    tooltip="\n".join(tooltip_lines),
                    subset=int(row_index) in subset_rows,
                )
            )

        anchors = [
            (name, float(self._anchors[index, 0]), float(self._anchors[index, 1]))
            for index, name in enumerate(self._effective_feature_names)
        ]
        self._canvas.set_projection(points, anchors, legend_items)
        self._canvas.set_selected_rows(self._selected_rows)
        self._components_dataset = build_components_dataset(
            self._effective_feature_names,
            self._anchors[:, 0].tolist(),
            self._anchors[:, 1].tolist(),
            dataset_id=f"{self._dataset.dataset_id}-freeviz-components",
            display_name=f"{self._dataset.display_name} (FreeViz Components)",
            file_name=f"{self._dataset.dataset_id}-freeviz-components.csv",
            service=self._builder,
        )
        self._handle_selection_changed(self._selected_rows, notify=False)
        feature_count = len(self._effective_feature_names)
        self._status_message = (
            f"{len(self._row_indices)} instances, {feature_count} features, "
            f"target '{self._target_name}', iterations {self._iteration_count}"
        )
        self._sync_status_labels()
        self._notify_output_changed()

    def _resolve_colors(self) -> tuple[list[QColor], list[str]]:
        attr = self._color_combo.currentText()
        if self._dataset is None:
            return [], []
        if attr == "(Class)":
            attr = self._target_name
        if not attr or attr == "None" or attr not in self._dataset.dataframe.columns:
            return [QColor("#3b82f6")] * len(self._row_indices), [""] * len(self._row_indices)

        series = self._dataset.dataframe.get_column(attr)
        if attr == self._target_name and self._target_is_discrete:
            labels = [self._target_labels[int(value)] for value in self._target_values.astype(int, copy=False)]
            colors = [QColor(PALETTE[index % len(PALETTE)]) for index, _ in enumerate(self._target_labels)]
            color_lookup = {label: colors[index] for index, label in enumerate(self._target_labels)}
            return [QColor(color_lookup[label]) for label in labels], labels

        if series.dtype.is_numeric():
            values = series.cast(pl.Float64, strict=False).to_numpy().astype(float, copy=False)[self._row_indices]
            finite = values[np.isfinite(values)]
            low = float(np.min(finite)) if finite.size else 0.0
            high = float(np.max(finite)) if finite.size else 1.0
            colors = []
            for value in values:
                t = 0.5 if abs(high - low) < 1e-9 else max(0.0, min(1.0, (float(value) - low) / (high - low)))
                colors.append(
                    QColor(
                        int(59 + (239 - 59) * t),
                        int(130 + (68 - 130) * t),
                        int(246 + (68 - 246) * t),
                    )
                )
            return colors, [""] * len(colors)

        labels = []
        ordered: list[str] = []
        for row_index in self._row_indices:
            value = self._dataset.dataframe[attr][int(row_index)]
            label = "(missing)" if value is None else str(value)
            labels.append(label)
            if label not in ordered:
                ordered.append(label)
        lookup = {label: QColor(PALETTE[index % len(PALETTE)]) for index, label in enumerate(ordered)}
        return [QColor(lookup[label]) for label in labels], labels

    def _legend_items_for_current_color(self) -> list[tuple[str, QColor]]:
        attr = self._color_combo.currentText()
        if self._dataset is None:
            return []
        if attr == "(Class)" and self._target_is_discrete:
            return [(label, QColor(PALETTE[index % len(PALETTE)])) for index, label in enumerate(self._target_labels)]
        if not attr or attr in {"None", "(Class)"} or attr not in self._dataset.dataframe.columns:
            return []
        series = self._dataset.dataframe.get_column(attr)
        if series.dtype.is_numeric():
            return []
        labels: list[str] = []
        for value in series.to_list():
            label = "(missing)" if value is None else str(value)
            if label not in labels:
                labels.append(label)
        return [(label, QColor(PALETTE[index % len(PALETTE)])) for index, label in enumerate(labels[:8])]

    def _resolve_shapes(self) -> list[int]:
        attr = self._shape_combo.currentText()
        if self._dataset is None or not attr or attr == "None" or attr not in self._dataset.dataframe.columns:
            return [0] * len(self._row_indices)
        labels: list[str] = []
        indices: list[int] = []
        for row_index in self._row_indices:
            value = self._dataset.dataframe[attr][int(row_index)]
            label = "(missing)" if value is None else str(value)
            if label not in labels:
                labels.append(label)
            indices.append(labels.index(label) % len(ORANGE_SHAPES))
        return indices

    def _resolve_sizes(self) -> list[int]:
        attr = self._size_combo.currentText()
        if self._dataset is None or not attr or attr == "None" or attr not in self._dataset.dataframe.columns:
            return [6] * len(self._row_indices)
        values = self._dataset.dataframe.get_column(attr).cast(pl.Float64, strict=False).to_numpy().astype(float, copy=False)[self._row_indices]
        finite = values[np.isfinite(values)]
        if not finite.size or abs(float(np.max(finite)) - float(np.min(finite))) < 1e-9:
            return [6] * len(values)
        low = float(np.min(finite))
        high = float(np.max(finite))
        scaled = 4 + 6 * ((values - low) / (high - low))
        return [max(4, min(10, int(round(value)))) for value in scaled]

    def _resolve_labels(self) -> list[str]:
        attr = self._label_combo.currentText()
        if self._dataset is None or not attr or attr == "None" or attr not in self._dataset.dataframe.columns:
            return [""] * len(self._row_indices)
        labels = []
        for row_index in self._row_indices:
            value = self._dataset.dataframe[attr][int(row_index)]
            labels.append("" if value is None else str(value))
        return labels

    def _tooltip_value(self, column_name: str, row_index: int) -> str:
        if self._dataset is None or column_name not in self._dataset.dataframe.columns:
            return ""
        value = self._dataset.dataframe[column_name][int(row_index)]
        return "(missing)" if value is None else str(value)

    def _handle_anchor_moved(self, index: int, x: float, y: float) -> None:
        if self._anchors is None or not (0 <= index < len(self._anchors)):
            return
        self._anchors[index] = [float(x), float(y)]
        max_radius = float(np.max(np.linalg.norm(self._anchors, axis=1)))
        if max_radius >= 1e-3:
            self._anchors /= max_radius
        self._anchors = _rotate_axes(self._anchors)
        self._can_resume = self._optimized_once or self._iteration_count > 0
        if not self._timer.isActive():
            self._set_run_button_text("Resume" if self._can_resume else "Start")
        self._update_visuals()

    def _handle_selection_changed(self, rows: list[int], *, notify: bool = True) -> None:
        normalized = sorted({int(index) for index in rows if isinstance(index, int | float)})
        self._selected_rows = [index for index in normalized if self._dataset is not None and 0 <= index < self._dataset.row_count]
        self._canvas.set_selected_rows(self._selected_rows)
        self._selection_label.setText(f"Selected: {len(self._selected_rows)}")
        self._selected_dataset, self._annotated_dataset = build_selection_outputs(
            self._dataset,
            self._selected_rows,
            generated_by="freeviz",
            service=self._builder,
        )
        if notify:
            self._notify_output_changed()

    def _refresh_visuals(self, *_args) -> None:
        if self._anchors is not None:
            self._update_visuals()

    def _sync_status_labels(self) -> None:
        self._warning_label.setText(self._warning_message)
        self._status_label.setText(self._status_message)

    def _set_run_button_text(self, text: str) -> None:
        self._run_btn.setText(text)

    @staticmethod
    def _select_combo_text(combo: QComboBox, text: str) -> None:
        index = combo.findText(text)
        if index >= 0:
            combo.setCurrentIndex(index)

    @staticmethod
    def _pending_combo_value(combo: QComboBox, text: str) -> None:
        if not text:
            return
        index = combo.findText(text)
        if index >= 0:
            combo.setCurrentIndex(index)
