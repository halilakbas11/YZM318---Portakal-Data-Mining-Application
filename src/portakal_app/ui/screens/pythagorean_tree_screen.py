from __future__ import annotations

from collections import defaultdict, namedtuple
from dataclasses import dataclass
from html import escape
from math import cos, degrees, log, pi, sin, sqrt
from random import Random
from uuid import uuid4

import numpy as np
import polars as pl

from PySide6.QtCore import QPointF, QRectF, QSize, Qt, QTimer
from PySide6.QtGui import QBrush, QColor, QFontMetrics, QLinearGradient, QPainter, QPen
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFrame,
    QGraphicsItem,
    QGraphicsRectItem,
    QGraphicsScene,
    QGraphicsView,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from portakal_app.data.models import DatasetHandle
from portakal_app.data.services.generated_dataset_service import GeneratedDatasetService
from portakal_app.models import WorkflowPayload
from portakal_app.tree_artifacts import DecisionTreeArtifact
from portakal_app.ui import i18n
from portakal_app.ui.screens.node_screen import WorkflowNodeScreenSupport


OUTPUT_CHANNELS = ("Selected Data", "Annotated Data")
SELECTED_COLUMN_NAME = "Selected"
Point = namedtuple("Point", ["x", "y"])
Square = namedtuple("Square", ["center", "length", "angle"])

_CONT_LOW = QColor("#2166ac")
_CONT_MID = QColor("#f7f7f7")
_CONT_HIGH = QColor("#b2182b")
_BORDER_COLOR = QColor("#2f2419")
_BACKGROUND = QColor("#fffdf9")
_FADED_OPACITY = 0.12
_DIMMED_OPACITY = 0.5
_MAX_OPACITY = 1.0


def _is_missing(value) -> bool:
    return value is None or (isinstance(value, float) and np.isnan(value))


def _display_value(value) -> str:
    if _is_missing(value):
        return ""
    return str(value)


def _dataset_summary(dataset: DatasetHandle | None) -> str:
    if dataset is None:
        return i18n.t("No data")
    return i18n.tf("{name}: {rows} rows x {cols} cols", name=dataset.display_name, rows=dataset.row_count, cols=dataset.column_count)


def _preview_rows(dataset: DatasetHandle | None, limit: int = 200) -> list[list[str]]:
    if dataset is None:
        return []
    rows: list[list[str]] = []
    for row in dataset.dataframe.head(limit).iter_rows(named=False):
        rows.append([_display_value(value) for value in row])
    return rows


def _unique_name(existing: list[str], proposed: str) -> str:
    if proposed not in existing:
        return proposed
    index = 1
    while f"{proposed} ({index})" in existing:
        index += 1
    return f"{proposed} ({index})"


def _role_overrides(dataset: DatasetHandle) -> dict[str, str]:
    return {column.name: column.role for column in dataset.domain.columns}


def _subset_frame(dataset: DatasetHandle, indices: list[int]) -> pl.DataFrame:
    if not indices:
        return dataset.dataframe.head(0)
    mask = [False] * dataset.row_count
    for index in indices:
        if 0 <= index < dataset.row_count:
            mask[index] = True
    return dataset.dataframe.filter(pl.Series("__mask__", mask))


def _interpolate_color(start: QColor, end: QColor, t: float) -> QColor:
    t = max(0.0, min(1.0, t))
    return QColor(
        int(start.red() + (end.red() - start.red()) * t),
        int(start.green() + (end.green() - start.green()) * t),
        int(start.blue() + (end.blue() - start.blue()) * t),
    )


def _continuous_color(value: float, low: float, high: float) -> QColor:
    if not np.isfinite(value) or high <= low:
        return QColor(_CONT_MID)
    t = (value - low) / (high - low)
    if t <= 0.5:
        return _interpolate_color(_CONT_LOW, _CONT_MID, t * 2.0)
    return _interpolate_color(_CONT_MID, _CONT_HIGH, (t - 0.5) * 2.0)


def _safe_float_series(dataset: DatasetHandle, column_name: str) -> np.ndarray:
    values = dataset.dataframe.get_column(column_name).cast(pl.Float64, strict=False).to_list()
    return np.asarray([float(value) if value is not None else np.nan for value in values], dtype=float)


@dataclass
class _LayoutNode:
    node_id: int
    square: Square
    children: tuple["_LayoutNode", ...]
    parent_id: int | None = None


class _PythagorasLayout:
    def __init__(self, weight_adjustment=lambda value: value) -> None:
        self.adjust_weight = weight_adjustment
        self._slopes = defaultdict(list)

    def build(self, tree: DecisionTreeArtifact, node_id: int, square: Square) -> _LayoutNode:
        if node_id == tree.root:
            self._slopes.clear()

        child_ids = tree.children(node_id)
        child_weights = [self.adjust_weight(tree.weight(child_id)) for child_id in child_ids]
        total_weight = sum(child_weights) or 1.0
        normalized = [weight / total_weight for weight in child_weights]
        children = tuple(self._compute_child(tree, square, child_id, weight) for child_id, weight in zip(child_ids, normalized))
        layout_node = _LayoutNode(node_id=node_id, square=square, children=children)
        for child in children:
            child.parent_id = node_id
        return layout_node

    def _compute_child(self, tree: DecisionTreeArtifact, parent_square: Square, node_id: int, weight: float) -> _LayoutNode:
        alpha = weight * pi
        length = parent_square.length * sin(alpha / 2)
        previous_angles = sum(self._slopes[parent_square])
        center = self._compute_center(parent_square, length, alpha, previous_angles)
        angle = parent_square.angle - pi / 2 + previous_angles + alpha / 2
        square = Square(center, length, angle)
        self._slopes[parent_square].append(alpha)
        return self.build(tree, node_id, square)

    def _compute_center(self, initial_square: Square, length: float, alpha: float, base_angle: float = 0) -> Point:
        parent_center, parent_length, parent_angle = initial_square
        t0 = self._get_point_on_square_edge(parent_center, parent_length, parent_angle)
        diagonal = sqrt(2 * parent_length**2)
        edge = self._get_point_on_square_edge(parent_center, diagonal, parent_angle - pi / 4)
        if base_angle != 0:
            edge = self._rotate_point(edge, t0, base_angle)
        t1 = self._rotate_point(edge, t0, alpha)
        t2 = Point((t1.x + edge.x) / 2, (t1.y + edge.y) / 2)
        slope = parent_angle - pi / 2 + alpha / 2
        return self._get_point_on_square_edge(t2, length, slope + base_angle)

    @staticmethod
    def _rotate_point(point: Point, around: Point, alpha: float) -> Point:
        temp = Point(point.x - around.x, point.y - around.y)
        temp = Point(
            temp.x * cos(alpha) - temp.y * sin(alpha),
            temp.x * sin(alpha) + temp.y * cos(alpha),
        )
        return Point(temp.x + around.x, temp.y + around.y)

    @staticmethod
    def _get_point_on_square_edge(center: Point, length: float, angle: float) -> Point:
        return Point(center.x + length / 2 * cos(angle), center.y + length / 2 * sin(angle))


class _LegendWidget(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._mode = "none"
        self._items: list[tuple[str, QColor]] = []
        self._range: tuple[float, float] | None = None
        self._range_labels: tuple[str, str] = ("", "")
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)

    def clear(self) -> None:
        self._mode = "none"
        self._items = []
        self._range = None
        self._range_labels = ("", "")
        self.updateGeometry()
        self.update()

    def set_discrete_items(self, items: list[tuple[str, QColor]]) -> None:
        self._mode = "discrete"
        self._items = items
        self._range = None
        self._range_labels = ("", "")
        self.updateGeometry()
        self.update()

    def set_continuous_range(self, low: float, high: float, low_label: str, high_label: str) -> None:
        self._mode = "continuous"
        self._items = []
        self._range = (float(low), float(high))
        self._range_labels = (low_label, high_label)
        self.updateGeometry()
        self.update()

    def is_empty(self) -> bool:
        return self._mode == "none"

    def sizeHint(self) -> QSize:
        metrics = QFontMetrics(self.font())
        if self._mode == "discrete":
            width = 72
            for label, _color in self._items:
                width = max(width, metrics.horizontalAdvance(label) + 58)
            height = 18 + len(self._items) * 22
            return QSize(width, max(46, height))
        if self._mode == "continuous":
            width = max(180, metrics.horizontalAdvance(self._range_labels[0]) + metrics.horizontalAdvance(self._range_labels[1]) + 92)
            return QSize(width, 66)
        return QSize(0, 0)

    def paintEvent(self, _event) -> None:
        if self._mode == "none":
            return

        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setPen(QPen(QColor(90, 74, 54, 160), 1))
        painter.setBrush(QColor(255, 253, 249, 238))
        painter.drawRoundedRect(self.rect().adjusted(0, 0, -1, -1), 8, 8)

        painter.setPen(QColor("#463c2f"))
        painter.drawText(QRectF(12, 8, self.width() - 24, 16), Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, i18n.t("Legend"))

        if self._mode == "discrete":
            y = 28
            for label, color in self._items:
                painter.setPen(QPen(_BORDER_COLOR, 1))
                painter.setBrush(color)
                painter.drawRect(QRectF(12, y, 14, 14))
                painter.setPen(QColor("#463c2f"))
                painter.drawText(QRectF(34, y - 1, self.width() - 46, 18), Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, label)
                y += 22
            return

        if self._mode == "continuous" and self._range is not None:
            gradient = QLinearGradient(0, 0, 1, 0)
            gradient.setCoordinateMode(QLinearGradient.CoordinateMode.ObjectBoundingMode)
            gradient.setColorAt(0.0, _CONT_LOW)
            gradient.setColorAt(0.5, _CONT_MID)
            gradient.setColorAt(1.0, _CONT_HIGH)
            painter.setPen(QPen(_BORDER_COLOR, 1))
            painter.setBrush(gradient)
            painter.drawRect(QRectF(12, 30, self.width() - 24, 12))
            painter.setPen(QColor("#463c2f"))
            painter.drawText(QRectF(12, 44, 72, 16), Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, self._range_labels[0])
            painter.drawText(QRectF(self.width() - 84, 44, 72, 16), Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter, self._range_labels[1])


class _TreeViewContainer(QWidget):
    def __init__(self, view: QGraphicsView, legend: _LegendWidget, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._view = view
        self._legend = legend
        self._view.setParent(self)
        self._legend.setParent(self)

    def resizeEvent(self, event) -> None:
        self._view.setGeometry(self.rect())
        self._reposition_overlay()
        super().resizeEvent(event)

    def _reposition_overlay(self) -> None:
        if not self._legend.isVisible():
            return
        hint = self._legend.sizeHint()
        margin = 12
        width = min(hint.width(), max(0, self.width() - 2 * margin))
        height = min(hint.height(), max(0, self.height() - 2 * margin))
        x = max(margin, self.width() - width - margin)
        y = max(margin, self.height() - height - margin)
        self._legend.setGeometry(x, y, width, height)

    def refresh_overlay(self) -> None:
        self._reposition_overlay()
        self._legend.update()


class _TreeGraphicsView(QGraphicsView):
    def __init__(self, scene: QGraphicsScene, parent: QWidget | None = None) -> None:
        super().__init__(scene, parent)
        self.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        self.setViewportUpdateMode(QGraphicsView.ViewportUpdateMode.BoundingRectViewportUpdate)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorViewCenter)
        self.setDragMode(QGraphicsView.DragMode.RubberBandDrag)
        self.setRubberBandSelectionMode(Qt.ItemSelectionMode.IntersectsItemShape)
        self.setBackgroundBrush(_BACKGROUND)
        self.setFrameShape(QFrame.Shape.NoFrame)

    def wheelEvent(self, event) -> None:
        if event.angleDelta().y() == 0:
            return super().wheelEvent(event)
        factor = 1.18 if event.angleDelta().y() > 0 else 1 / 1.18
        self.scale(factor, factor)
        event.accept()

    def fit_content(self) -> None:
        scene = self.scene()
        if scene is None:
            return
        rect = scene.itemsBoundingRect()
        if not rect.isValid() or rect.isNull():
            return
        rect = rect.adjusted(-40, -40, 40, 40)
        self.resetTransform()
        self.fitInView(rect, Qt.AspectRatioMode.KeepAspectRatio)


class _SquareItem(QGraphicsRectItem):
    def __init__(self, screen: "PythagoreanTreeScreen", node_id: int, square: Square, depth: int) -> None:
        self._screen = screen
        self.node_id = int(node_id)
        self._depth = int(depth)
        center, length, _angle = square
        x = center.x - length / 2
        y = center.y - length / 2
        super().__init__(QRectF(x, y, length, length))
        self.setTransformOriginPoint(self.boundingRect().center())
        self.setRotation(degrees(square.angle))
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable, True)
        self.setAcceptHoverEvents(True)
        self.setZValue(float(depth))
        self.update_appearance()

    def update_appearance(self) -> None:
        self.setBrush(QBrush(self._screen._node_color(self.node_id)))
        tooltip = self._screen._node_tooltip(self.node_id) if self._screen.tooltips_enabled else ""
        self.setToolTip(tooltip)
        self.setOpacity(self._screen._node_opacity(self.node_id))
        pen = QPen(_BORDER_COLOR, 2.0 if self.isSelected() else 0.75)
        pen.setCosmetic(True)
        pen.setJoinStyle(Qt.PenJoinStyle.MiterJoin)
        self.setPen(pen)
        if self._screen._hovered_node_id is not None and self._screen._node_in_hover_branch(self.node_id):
            self.setZValue(1_000_000 + self._depth)
        else:
            self.setZValue(float(self._depth))

    def itemChange(self, change, value):
        if change == QGraphicsItem.GraphicsItemChange.ItemSelectedHasChanged:
            QTimer.singleShot(0, self._screen._handle_scene_selection_changed)
        return super().itemChange(change, value)

    def hoverEnterEvent(self, event) -> None:
        self._screen._set_hovered_node(self.node_id)
        super().hoverEnterEvent(event)

    def hoverLeaveEvent(self, event) -> None:
        self._screen._clear_hovered_node(self.node_id)
        super().hoverLeaveEvent(event)

    def mouseDoubleClickEvent(self, event) -> None:
        self._screen._reverse_subtree(self.node_id)
        event.accept()


class PythagoreanTreeScreen(QWidget, WorkflowNodeScreenSupport):
    SIZE_NORMAL = 0
    SIZE_SQRT = 1
    SIZE_LOG = 2

    def __init__(self) -> None:
        super().__init__()
        self._init_workflow_node_support()
        self._generated_datasets = GeneratedDatasetService()
        self._screen_token = uuid4().hex[:8]

        self._tree: DecisionTreeArtifact | None = None
        self._selected_dataset: DatasetHandle | None = None
        self._annotated_dataset: DatasetHandle | None = None
        self._selected_node_ids: set[int] = set()
        self._layout_root: _LayoutNode | None = None
        self._layout_nodes: dict[int, _LayoutNode] = {}
        self._depths: dict[int, int] = {}
        self._items: dict[int, _SquareItem] = {}
        self._hovered_node_id: int | None = None

        self._pending_depth_limit = 0
        self._pending_target_class_index = 0
        self._pending_size_calc_idx = self.SIZE_NORMAL
        self._pending_log_scale = 2
        self._pending_selection: list[int] = []

        self.tooltips_enabled = True
        self.show_legend = False

        self._refresh_timer = QTimer(self)
        self._refresh_timer.setSingleShot(True)
        self._refresh_timer.setInterval(0)
        self._refresh_timer.timeout.connect(self._rebuild_tree)

        self._build_ui()
        self._clear_state()

    def _build_ui(self) -> None:
        root = QHBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        controls = QFrame(self)
        controls.setFrameShape(QFrame.Shape.StyledPanel)
        controls.setMinimumWidth(272)
        controls.setMaximumWidth(320)
        controls_layout = QVBoxLayout(controls)
        controls_layout.setContentsMargins(14, 14, 14, 14)
        controls_layout.setSpacing(12)

        info_box = QGroupBox(i18n.t("Tree Info"), controls)
        info_layout = QVBoxLayout(info_box)
        self.infolabel = QLabel(info_box)
        self.infolabel.setWordWrap(True)
        info_layout.addWidget(self.infolabel)
        controls_layout.addWidget(info_box)

        display_box = QGroupBox(i18n.t("Display Settings"), controls)
        display_layout = QGridLayout(display_box)
        display_layout.setContentsMargins(10, 10, 10, 10)
        display_layout.setHorizontalSpacing(8)
        display_layout.setVerticalSpacing(10)

        display_layout.addWidget(QLabel(i18n.t("Depth"), display_box), 0, 0)
        self.depth_slider = QSlider(Qt.Orientation.Horizontal, display_box)
        self.depth_slider.setRange(0, 0)
        self.depth_slider.valueChanged.connect(self._on_depth_changed)
        display_layout.addWidget(self.depth_slider, 0, 1)

        self._target_label = QLabel(i18n.t("Target class"), display_box)
        display_layout.addWidget(self._target_label, 1, 0)
        self.target_class_combo = QComboBox(display_box)
        self.target_class_combo.currentIndexChanged.connect(self._on_target_class_changed)
        display_layout.addWidget(self.target_class_combo, 1, 1)

        display_layout.addWidget(QLabel(i18n.t("Size"), display_box), 2, 0)
        self.size_calc_combo = QComboBox(display_box)
        self.size_calc_combo.addItem(i18n.t("Normal"), self.SIZE_NORMAL)
        self.size_calc_combo.addItem(i18n.t("Square root"), self.SIZE_SQRT)
        self.size_calc_combo.addItem(i18n.t("Logarithmic"), self.SIZE_LOG)
        self.size_calc_combo.currentIndexChanged.connect(self._on_size_mode_changed)
        display_layout.addWidget(self.size_calc_combo, 2, 1)

        self._log_scale_label = QLabel(i18n.t("Log scale factor"), display_box)
        display_layout.addWidget(self._log_scale_label, 3, 0)
        self.log_scale_box = QSlider(Qt.Orientation.Horizontal, display_box)
        self.log_scale_box.setRange(1, 100)
        self.log_scale_box.setValue(2)
        self.log_scale_box.valueChanged.connect(self._on_log_scale_changed)
        display_layout.addWidget(self.log_scale_box, 3, 1)
        controls_layout.addWidget(display_box)

        plot_box = QGroupBox(i18n.t("Plot Properties"), controls)
        plot_layout = QVBoxLayout(plot_box)
        self.cb_show_tooltips = QCheckBox(i18n.t("Enable tooltips"), plot_box)
        self.cb_show_tooltips.setChecked(True)
        self.cb_show_tooltips.toggled.connect(self._on_tooltips_toggled)
        plot_layout.addWidget(self.cb_show_tooltips)
        self.cb_show_legend = QCheckBox(i18n.t("Show legend"), plot_box)
        self.cb_show_legend.toggled.connect(self._on_legend_toggled)
        plot_layout.addWidget(self.cb_show_legend)
        controls_layout.addWidget(plot_box)

        self._status_label = QLabel(self)
        self._status_label.setWordWrap(True)
        self._status_label.setProperty("muted", True)
        controls_layout.addWidget(self._status_label)
        controls_layout.addStretch(1)

        footer = QHBoxLayout()
        self.cb_apply_auto = QCheckBox(i18n.t("Apply Automatically"))
        self.cb_apply_auto.setChecked(True)
        footer.addWidget(self.cb_apply_auto)
        footer.addStretch(1)
        self._redraw_button = QPushButton(i18n.t("Redraw"))
        self._redraw_button.clicked.connect(self.redraw)
        footer.addWidget(self._redraw_button)
        controls_layout.addLayout(footer)

        root.addWidget(controls, 0)

        self.scene = QGraphicsScene(self)
        self.scene.selectionChanged.connect(self._handle_scene_selection_changed)
        self.view = _TreeGraphicsView(self.scene, self)
        self._legend = _LegendWidget(self)
        self._view_container = _TreeViewContainer(self.view, self._legend, self)
        self._view_container.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        root.addWidget(self._view_container, 1)

    def _clear_state(self) -> None:
        self._tree = None
        self._selected_dataset = None
        self._annotated_dataset = None
        self._selected_node_ids = set()
        self._layout_root = None
        self._layout_nodes = {}
        self._depths = {}
        self._items = {}
        self._hovered_node_id = None
        self.scene.clear()
        self._legend.clear()
        self._legend.hide()
        self._view_container.refresh_overlay()
        self._set_controls_enabled(False)
        self._populate_target_combo()
        self.infolabel.setText(i18n.t("No tree on input"))
        self._status_label.setText(i18n.t("Connect a decision tree to visualize its structure."))

    def _set_controls_enabled(self, enabled: bool) -> None:
        self.depth_slider.setEnabled(enabled)
        self.target_class_combo.setEnabled(enabled)
        self.size_calc_combo.setEnabled(enabled)
        self._redraw_button.setEnabled(enabled)
        self._update_log_scale_slider_enabled()

    def set_input_payload(self, payload: WorkflowPayload | None) -> None:
        if payload is None:
            self._clear_state()
            self._notify_output_changed()
            return

        tree = payload.value
        if not isinstance(tree, DecisionTreeArtifact):
            self._clear_state()
            self._status_label.setText(i18n.t("Pythagorean Tree expects a tree model on input."))
            self._notify_output_changed()
            return

        self._tree = tree
        self._selected_dataset = None
        self._annotated_dataset = None
        self._selected_node_ids = set()
        self._hovered_node_id = None
        if tree.meta_target_class_index is not None:
            self._pending_target_class_index = int(tree.meta_target_class_index)
        if tree.meta_size_calc_idx is not None:
            self._pending_size_calc_idx = int(tree.meta_size_calc_idx)
        if tree.meta_depth_limit is not None:
            self._pending_depth_limit = int(tree.meta_depth_limit)
        self._populate_target_combo()
        self._sync_controls_from_pending_state()
        self._update_info_box()
        self._set_controls_enabled(True)
        self._refresh_timer.start()

    def current_output_datasets(self) -> dict[str, DatasetHandle | None] | None:
        return {
            OUTPUT_CHANNELS[0]: self._selected_dataset,
            OUTPUT_CHANNELS[1]: self._annotated_dataset,
        }

    def data_preview_snapshot(self) -> dict[str, object]:
        dataset = self._selected_dataset or self._annotated_dataset
        if dataset is None:
            return {"summary": i18n.t("No preview available."), "headers": [], "rows": []}
        return {
            "summary": _dataset_summary(dataset),
            "headers": list(dataset.dataframe.columns),
            "rows": _preview_rows(dataset),
        }

    def detailed_data_snapshot(self) -> dict[str, object]:
        return {
            "selected_summary": _dataset_summary(self._selected_dataset) if self._selected_dataset is not None else i18n.t("Selected Data: -"),
            "selected_headers": list(self._selected_dataset.dataframe.columns) if self._selected_dataset is not None else [],
            "selected_rows": _preview_rows(self._selected_dataset) if self._selected_dataset is not None else [],
            "data_summary": _dataset_summary(self._annotated_dataset) if self._annotated_dataset is not None else i18n.t("Data: -"),
            "data_headers": list(self._annotated_dataset.dataframe.columns) if self._annotated_dataset is not None else [],
            "data_rows": _preview_rows(self._annotated_dataset) if self._annotated_dataset is not None else [],
        }

    def serialize_node_state(self) -> dict[str, object]:
        return {
            "depth_limit": self.depth_slider.value(),
            "target_class_index": self.target_class_combo.currentIndex(),
            "size_calc_idx": self.size_calc_combo.currentIndex(),
            "size_log_scale": self.log_scale_box.value(),
            "tooltips_enabled": self.cb_show_tooltips.isChecked(),
            "show_legend": self.cb_show_legend.isChecked(),
            "selection": sorted(self._selected_node_ids),
            "auto_apply": self.cb_apply_auto.isChecked(),
        }

    def restore_node_state(self, payload: dict[str, object]) -> None:
        self._pending_depth_limit = int(payload.get("depth_limit", 0) or 0)
        self._pending_target_class_index = int(payload.get("target_class_index", 0) or 0)
        self._pending_size_calc_idx = int(payload.get("size_calc_idx", self.SIZE_NORMAL) or self.SIZE_NORMAL)
        self._pending_log_scale = int(payload.get("size_log_scale", 2) or 2)
        self.cb_show_tooltips.setChecked(bool(payload.get("tooltips_enabled", True)))
        self.cb_show_legend.setChecked(bool(payload.get("show_legend", False)))
        self.cb_apply_auto.setChecked(bool(payload.get("auto_apply", True)))
        selection = payload.get("selection", [])
        if isinstance(selection, list):
            self._pending_selection = [int(value) for value in selection if isinstance(value, int)]
        if self._tree is not None:
            self._sync_controls_from_pending_state()
            self._refresh_timer.start()

    def help_text(self) -> str:
        return i18n.t("Visualize a decision tree as a Pythagorean tree. Select nodes to send Selected Data and Annotated Data outputs.")

    def _populate_target_combo(self) -> None:
        self.target_class_combo.blockSignals(True)
        self.target_class_combo.clear()
        if self._tree is None:
            self._target_label.setText(i18n.t("Target class"))
            self.target_class_combo.blockSignals(False)
            return

        if self._tree.has_discrete_class:
            self._target_label.setText(i18n.t("Target class"))
            self.target_class_combo.addItem(i18n.t("None"))
            values = self._tree.class_values or ()
            for value in values:
                self.target_class_combo.addItem(str(value).title())
        else:
            self._target_label.setText(i18n.t("Node color"))
            self.target_class_combo.addItems((i18n.t("None"), i18n.t("Mean"), i18n.t("Standard deviation")))
        self.target_class_combo.blockSignals(False)

    def _sync_controls_from_pending_state(self) -> None:
        if self._tree is None:
            return
        self.depth_slider.blockSignals(True)
        self.depth_slider.setMaximum(max(0, self._tree.max_depth))
        depth_limit = self._pending_depth_limit if self._pending_depth_limit else self._tree.max_depth
        self.depth_slider.setValue(max(0, min(depth_limit, self._tree.max_depth)))
        self.depth_slider.blockSignals(False)

        self.size_calc_combo.blockSignals(True)
        self.size_calc_combo.setCurrentIndex(max(0, min(self._pending_size_calc_idx, self.size_calc_combo.count() - 1)))
        self.size_calc_combo.blockSignals(False)

        self.log_scale_box.blockSignals(True)
        self.log_scale_box.setValue(max(1, min(self._pending_log_scale, self.log_scale_box.maximum())))
        self.log_scale_box.blockSignals(False)

        self.target_class_combo.blockSignals(True)
        max_index = max(0, self.target_class_combo.count() - 1)
        self.target_class_combo.setCurrentIndex(max(0, min(self._pending_target_class_index, max_index)))
        self.target_class_combo.blockSignals(False)

        self._update_log_scale_slider_enabled()

    def _update_info_box(self) -> None:
        if self._tree is None:
            self.infolabel.setText(i18n.t("No tree on input"))
            return
        self.infolabel.setText(i18n.tf("Nodes: {nodes}\nDepth: {depth}", nodes=self._tree.num_nodes, depth=self._tree.max_depth))

    def _current_size_adjustment(self):
        mode = self.size_calc_combo.currentIndex()
        if mode == self.SIZE_SQRT:
            return lambda value: sqrt(max(1.0, float(value)))
        if mode == self.SIZE_LOG:
            factor = max(1, self.log_scale_box.value())
            return lambda value: log(max(1.0, float(value)) * factor + 1.0)
        return lambda value: float(value)

    def _rebuild_tree(self) -> None:
        if self._tree is None:
            self._clear_state()
            self._notify_output_changed()
            return

        builder = _PythagorasLayout(weight_adjustment=self._current_size_adjustment())
        self._layout_root = builder.build(self._tree, self._tree.root, Square(Point(0, 0), 200, -pi / 2))
        self._layout_nodes = {}
        self._depths = {}
        self._flatten_layout(self._layout_root, depth=0)
        self._render_visible_nodes()
        self._update_legend()
        self._update_status()
        if self.cb_apply_auto.isChecked():
            self._apply()
        else:
            self._notify_output_changed()

    def _flatten_layout(self, node: _LayoutNode, depth: int) -> None:
        self._layout_nodes[node.node_id] = node
        self._depths[node.node_id] = depth
        for child in node.children:
            self._flatten_layout(child, depth + 1)

    def _render_visible_nodes(self) -> None:
        self.scene.blockSignals(True)
        self.scene.clear()
        self._items = {}
        if self._layout_root is None:
            self.scene.blockSignals(False)
            return

        visible_ids: set[int] = set()
        for node_id, layout in self._layout_nodes.items():
            depth = self._depths.get(node_id, 0)
            if depth > self.depth_slider.value():
                continue
            item = _SquareItem(self, node_id, layout.square, depth)
            self.scene.addItem(item)
            self._items[node_id] = item
            visible_ids.add(node_id)

        if self._pending_selection:
            self._selected_node_ids = {node_id for node_id in self._pending_selection if node_id in visible_ids}
            self._pending_selection = []
        else:
            self._selected_node_ids &= visible_ids
        for node_id in self._selected_node_ids:
            item = self._items.get(node_id)
            if item is not None:
                item.setSelected(True)
        self.scene.blockSignals(False)
        self._handle_scene_selection_changed()
        self.view.fit_content()

    def _handle_scene_selection_changed(self) -> None:
        selected: set[int] = set()
        for item in self.scene.selectedItems():
            if isinstance(item, _SquareItem):
                selected.add(item.node_id)
        if selected == self._selected_node_ids and self._items:
            self._update_item_styles()
            return
        self._selected_node_ids = selected
        self._update_item_styles()
        self._update_status()
        if self.cb_apply_auto.isChecked():
            self._apply()
        else:
            self._notify_output_changed()

    def _update_item_styles(self) -> None:
        for item in self._items.values():
            item.update_appearance()

    def _set_hovered_node(self, node_id: int) -> None:
        if self._hovered_node_id == node_id:
            return
        self._hovered_node_id = node_id
        self._update_item_styles()

    def _clear_hovered_node(self, node_id: int) -> None:
        if self._hovered_node_id != node_id:
            return
        self._hovered_node_id = None
        self._update_item_styles()

    def _node_in_hover_branch(self, node_id: int) -> bool:
        if self._hovered_node_id is None:
            return False
        branch = self._hover_branch_nodes(self._hovered_node_id)
        return node_id in branch

    def _hover_branch_nodes(self, node_id: int) -> set[int]:
        branch: set[int] = set()
        current = node_id
        while True:
            branch.add(current)
            parent_id = self._layout_nodes.get(current).parent_id if current in self._layout_nodes else None
            if parent_id is None:
                break
            current = parent_id
        frontier = [node_id]
        while frontier:
            label = frontier.pop()
            branch.add(label)
            frontier.extend(self._tree.children(label) if self._tree is not None else [])
        return branch

    def _node_opacity(self, node_id: int) -> float:
        if self._hovered_node_id is not None:
            if self._node_in_hover_branch(node_id):
                return _MAX_OPACITY
            if self._selected_node_ids and node_id in self._selected_node_ids:
                return _MAX_OPACITY
            return _FADED_OPACITY
        if self._selected_node_ids:
            return _MAX_OPACITY if node_id in self._selected_node_ids else _DIMMED_OPACITY
        return _MAX_OPACITY

    def _classification_palette(self) -> list[QColor]:
        if self._tree is None:
            return []
        return [QColor(color) for color in self._tree.class_colors[: max(1, len(self._tree.class_values) or 1)]]

    def _regression_summary(self, node_id: int) -> tuple[float, float]:
        assert self._tree is not None
        target_values = _safe_float_series(self._tree.instances, self._tree.target_name)
        indices = self._tree.get_indices(node_id)
        subset = target_values[indices] if indices else np.asarray([], dtype=float)
        subset = subset[np.isfinite(subset)]
        if subset.size == 0:
            return float("nan"), float("nan")
        return float(np.mean(subset)), float(np.std(subset))

    def _node_color(self, node_id: int) -> QColor:
        if self._tree is None:
            return QColor("white")
        if self._tree.has_discrete_class:
            distribution = np.asarray(self._tree.get_distribution(node_id)[0], dtype=float)
            total = float(np.sum(distribution))
            palette = self._classification_palette()
            if distribution.size == 0 or not palette:
                return QColor("white")
            if self.target_class_combo.currentIndex() > 0:
                target = self.target_class_combo.currentIndex() - 1
                p = float(distribution[target] / total) if total > 0 else 0.0
                color = QColor(palette[target % len(palette)])
                return color.lighter(int(200 - 100 * p))
            modus = int(np.argmax(distribution))
            p = float(distribution[modus] / (total or 1.0))
            color = QColor(palette[modus % len(palette)])
            return color.lighter(int(400 - 300 * p))

        mode = self.target_class_combo.currentIndex()
        if mode == 0:
            return QColor("#ffffff")
        values = _safe_float_series(self._tree.instances, self._tree.target_name)
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            return QColor(_CONT_MID)
        mean, std = self._regression_summary(node_id)
        if mode == 1:
            return _continuous_color(mean, float(np.min(finite)), float(np.max(finite)))
        return _continuous_color(std, 0.0, float(np.std(finite)))

    def _node_tooltip(self, node_id: int) -> str:
        if self._tree is None:
            return ""
        rules = "<br>".join(escape(rule) for rule in self._tree.rules(node_id))
        split_attr = escape(self._tree.attribute(node_id).name)

        if self._tree.has_discrete_class:
            distribution = np.asarray(self._tree.get_distribution(node_id)[0], dtype=float)
            total = int(np.sum(distribution))
            target_index = self.target_class_combo.currentIndex()
            if target_index > 0:
                samples = int(distribution[target_index - 1]) if distribution.size else 0
                header = ""
            else:
                modus = int(np.argmax(distribution)) if distribution.size else 0
                samples = int(distribution[modus]) if distribution.size else 0
                header = escape(str(self._tree.class_values[modus])) + "<br>" if self._tree.class_values else ""
            ratio = float(samples / total) if total else 0.0
            lines = [header + f"{samples}/{total} samples ({ratio * 100:.3f}%)"]
        else:
            mean, std = self._regression_summary(node_id)
            lines = [
                f"Mean: {mean:.3f}" if np.isfinite(mean) else i18n.t("Mean: n/a"),
                f"Standard deviation: {std:.3f}" if np.isfinite(std) else i18n.t("Standard deviation: n/a"),
                i18n.tf("{count} samples", count=self._tree.num_samples(node_id)),
            ]

        details = "<br>".join(lines)
        split_text = i18n.tf("Split by {name}", name=split_attr) if split_attr and not self._tree.is_leaf(node_id) else ""
        html = f"<p>{details}"
        if split_text:
            html += f"<hr>{split_text}"
        if rules:
            spacer = "<br><br>" if split_text else "<hr>"
            html += f"{spacer}{rules}"
        html += "</p>"
        return html

    def _update_legend(self) -> None:
        if self._tree is None:
            self._legend.clear()
            self._legend.hide()
            self._view_container.refresh_overlay()
            return

        if self._tree.has_discrete_class:
            palette = self._classification_palette()
            if self.target_class_combo.currentIndex() == 0:
                items = [
                    (str(value).title(), QColor(palette[index % len(palette)]))
                    for index, value in enumerate(self._tree.class_values)
                ]
            else:
                target = self.target_class_combo.currentIndex() - 1
                label = self.target_class_combo.currentText()
                items = [(label, QColor(palette[target % len(palette)])), (i18n.t("other"), QColor("#ffffff"))]
            self._legend.set_discrete_items(items)
        else:
            values = _safe_float_series(self._tree.instances, self._tree.target_name)
            finite = values[np.isfinite(values)]
            if self.target_class_combo.currentIndex() == 0 or finite.size == 0:
                self._legend.clear()
            elif self.target_class_combo.currentIndex() == 1:
                low = float(np.min(finite))
                high = float(np.max(finite))
                self._legend.set_continuous_range(low, high, f"{low:.3f}", f"{high:.3f}")
            else:
                high = float(np.std(finite))
                self._legend.set_continuous_range(0.0, high, "0.000", f"{high:.3f}")

        should_show = self.cb_show_legend.isChecked() and not self._legend.is_empty()
        self._legend.setVisible(should_show)
        self._view_container.refresh_overlay()

    def _update_log_scale_slider_enabled(self) -> None:
        enabled = self._tree is not None and self.size_calc_combo.currentIndex() == self.SIZE_LOG
        self._log_scale_label.setEnabled(enabled)
        self.log_scale_box.setEnabled(enabled)

    def _update_status(self) -> None:
        if self._tree is None:
            self._status_label.setText(i18n.t("Connect a decision tree to visualize its structure."))
            return
        visible_count = len(self._items)
        selected_rows = len(self._selected_row_indices())
        status = i18n.tf(
            "{name}: {visible}/{nodes} visible nodes | {selected_nodes} selected nodes | {selected_rows} selected rows",
            name=self._tree.display_name,
            visible=visible_count,
            nodes=self._tree.num_nodes,
            selected_nodes=len(self._selected_node_ids),
            selected_rows=selected_rows,
        )
        self._status_label.setText(status)

    def _selected_row_indices(self) -> list[int]:
        if self._tree is None or not self._selected_node_ids:
            return []
        return self._tree.get_indices(sorted(self._selected_node_ids))

    def _apply(self) -> None:
        if self._tree is None:
            self._selected_dataset = None
            self._annotated_dataset = None
            self._notify_output_changed()
            return

        dataset = self._tree.instances
        selected_rows = self._selected_row_indices()
        selected_mask = [False] * dataset.row_count
        for index in selected_rows:
            if 0 <= index < dataset.row_count:
                selected_mask[index] = True

        role_overrides = _role_overrides(dataset)
        role_overrides[SELECTED_COLUMN_NAME] = "meta"
        annotated_frame = dataset.dataframe.with_columns(pl.Series(SELECTED_COLUMN_NAME, selected_mask))
        self._annotated_dataset = self._generated_datasets.build_dataset(
            annotated_frame,
            dataset_id=f"pythagorean-tree-{self._screen_token}-annotated",
            display_name=i18n.t("Annotated Data"),
            file_name=f"pythagorean-tree-{self._screen_token}-annotated.csv",
            role_overrides=role_overrides,
            annotations={
                "source_row_indices": list(range(dataset.row_count)),
                "selected_row_indices": selected_rows,
                "selected_nodes": sorted(self._selected_node_ids),
                "tree_id": self._tree.tree_id,
            },
        )

        if not selected_rows:
            self._selected_dataset = None
            self._notify_output_changed()
            return

        selected_frame = _subset_frame(dataset, selected_rows)
        self._selected_dataset = self._generated_datasets.build_dataset(
            selected_frame,
            dataset_id=f"pythagorean-tree-{self._screen_token}-selected",
            display_name=i18n.t("Selected Data"),
            file_name=f"pythagorean-tree-{self._screen_token}-selected.csv",
            role_overrides=_role_overrides(dataset),
            annotations={
                "source_row_indices": selected_rows,
                "selected_nodes": sorted(self._selected_node_ids),
                "tree_id": self._tree.tree_id,
            },
        )
        self._notify_output_changed()

    def _on_depth_changed(self, value: int) -> None:
        self._pending_depth_limit = int(value)
        self._render_visible_nodes()
        self._update_status()
        if self.cb_apply_auto.isChecked():
            self._apply()
        else:
            self._notify_output_changed()

    def _on_target_class_changed(self, index: int) -> None:
        self._pending_target_class_index = int(index)
        self._update_item_styles()
        self._update_legend()

    def _on_size_mode_changed(self, index: int) -> None:
        self._pending_size_calc_idx = int(index)
        self._update_log_scale_slider_enabled()
        self._refresh_timer.start()

    def _on_log_scale_changed(self, value: int) -> None:
        self._pending_log_scale = int(value)
        if self.size_calc_combo.currentIndex() == self.SIZE_LOG:
            self._refresh_timer.start()

    def _on_tooltips_toggled(self, checked: bool) -> None:
        self.tooltips_enabled = bool(checked)
        self._update_item_styles()

    def _on_legend_toggled(self, checked: bool) -> None:
        self.show_legend = bool(checked)
        self._update_legend()

    def _reverse_subtree(self, node_id: int) -> None:
        if self._tree is None:
            return
        self._tree.reverse_children(node_id)
        self._refresh_timer.start()

    def redraw(self) -> None:
        if self._tree is None:
            return
        self._tree.shuffle_children(Random())
        self._refresh_timer.start()
