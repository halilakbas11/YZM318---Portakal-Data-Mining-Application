from __future__ import annotations

import math
import re

import polars as pl

from PySide6.QtCore import QPointF, QRectF, Qt, QSize
from PySide6.QtGui import QColor, QBrush, QFontMetricsF, QPainter, QPen, QTransform
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFormLayout,
    QGraphicsEllipseItem,
    QGraphicsItem,
    QGraphicsLineItem,
    QGraphicsScene,
    QGraphicsTextItem,
    QGraphicsView,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QSizePolicy,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from portakal_app.data.models import DatasetHandle
from portakal_app.data.services.generated_dataset_service import GeneratedDatasetService
from portakal_app.ui import i18n
from portakal_app.ui.screens.node_screen import WorkflowNodeScreenSupport
from portakal_app.ui.screens.visualize_common import PALETTE, TreeNodeData, build_selection_outputs

DEFAULT_COLOR_MODE = "Default"
INSTANCE_COLOR_MODE = "Number of instances"
MEAN_COLOR_MODE = "Mean value"
VARIANCE_COLOR_MODE = "Variance"
EDGE_WIDTH_OPTIONS = ("Fixed", "Relative to root", "Relative to parent")
TREE_LEVEL_OPTIONS = ("Unlimited", "2 levels", "3 levels", "4 levels", "5 levels", "6 levels", "7 levels", "8 levels", "9 levels")
NODE_HORIZONTAL_SPACING = 26.0
NODE_VERTICAL_SPACING = 34.0
SELECTION_HIGHLIGHT = QColor(125, 162, 206, 192)
DEFAULT_NODE_BRUSH = QColor("#f7f7f7")
DEFAULT_REGRESSION_BRUSH = QColor(192, 192, 255)


def _safe_float(value: object) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _continuous_gradient(value: float, min_value: float, max_value: float) -> QColor:
    if max_value <= min_value:
        return QColor("#60a5fa")
    ratio = max(0.0, min(1.0, (value - min_value) / (max_value - min_value)))
    start = QColor("#dbeafe")
    end = QColor("#1d4ed8")
    red = int(start.red() + (end.red() - start.red()) * ratio)
    green = int(start.green() + (end.green() - start.green()) * ratio)
    blue = int(start.blue() + (end.blue() - start.blue()) * ratio)
    return QColor(red, green, blue)


def _sample_count_from_summary(summary: str) -> int | None:
    match = re.search(r"(\d+)", summary or "")
    return int(match.group(1)) if match else None


class _TreeNodeItem(QGraphicsTextItem):
    def __init__(self, screen: "TreeViewerScreen", node: TreeNodeData, *, parent_node: TreeNodeData | None) -> None:
        super().__init__()
        self.screen = screen
        self.node = node
        self.parent_node = parent_node
        self._box_rect = QRectF(0, 0, 1, 1)
        self._rule_height = 0.0
        self._background = QBrush(DEFAULT_NODE_BRUSH)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable, True)
        font = self.font()
        font.setPointSize(10)
        self.setFont(font)
        self.document().setDocumentMargin(6)

    def set_background(self, color: QColor) -> None:
        self._background = QBrush(color)
        luminance = 0.2126 * color.red() + 0.7152 * color.green() + 0.0722 * color.blue()
        self.setDefaultTextColor(Qt.GlobalColor.black if luminance > 110 else Qt.GlobalColor.white)
        self.update()

    def configure_html(self, html: str, *, width: float) -> None:
        self.prepareGeometryChange()
        self.setHtml(f"<body>{html}</body>")
        self.setTextWidth(width)
        metrics = QFontMetricsF(self.font())
        self._rule_height = metrics.lineSpacing() + 4 if self.parent_node is not None and self.node.rule else 0.0
        document_height = self.document().size().height()
        self._box_rect = QRectF(0, self._rule_height, width, document_height)
        self.update()

    @property
    def box_rect(self) -> QRectF:
        return self._box_rect

    def boundingRect(self) -> QRectF:  # noqa: N802
        metrics = QFontMetricsF(self.font())
        rule_width = metrics.horizontalAdvance(self.node.rule) + 12 if self.parent_node is not None and self.node.rule else 0.0
        rect = self._box_rect.adjusted(-6, -6, 6, 6)
        if self._rule_height:
            rect = rect | QRectF(0, 0, max(rule_width, rect.width()), self._rule_height)
        return rect.adjusted(-5, -5, 5, 5)

    def paint(self, painter: QPainter, option, widget=None) -> None:  # noqa: ANN001
        box = self._box_rect
        metrics = QFontMetricsF(self.font())
        if self.parent_node is not None and self.node.rule:
            text_width = metrics.horizontalAdvance(self.node.rule)
            text_x = box.center().x() - text_width / 2
            painter.setPen(QPen(QColor("#4b5563")))
            painter.drawText(QPointF(text_x, metrics.ascent() + 1), self.node.rule)
        painter.save()
        if self.isSelected():
            painter.setBrush(QBrush(SELECTION_HIGHLIGHT))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawRoundedRect(box.adjusted(-4, -4, 4, 4), 10, 10)
        painter.restore()
        painter.save()
        painter.setBrush(self._background)
        painter.setPen(QPen(QColor("#1d4ed8"), 2.2) if self.isSelected() else QPen(QColor("#334155"), 1.6))
        if self.node.children:
            painter.drawRect(box)
        else:
            painter.drawRoundedRect(box, 4, 4)
        painter.restore()
        painter.save()
        painter.setClipRect(box)
        super().paint(painter, option, widget)
        painter.restore()
        if self.node.children:
            center = QPointF(box.center().x(), box.bottom())
            painter.save()
            painter.setBrush(QBrush(QColor(175, 175, 175)))
            painter.setPen(QPen(Qt.GlobalColor.white, 1))
            painter.drawEllipse(center, 5, 5)
            painter.restore()

    def edge_in_point(self) -> QPointF:
        return self.mapToScene(QPointF(self._box_rect.center().x(), self._box_rect.top()))

    def edge_out_point(self) -> QPointF:
        return self.mapToScene(QPointF(self._box_rect.center().x(), self._box_rect.bottom()))

    def mouseDoubleClickEvent(self, event) -> None:  # noqa: N802, ANN001
        if self.node.children and event.button() == Qt.MouseButton.LeftButton:
            self.screen.toggle_node_expansion(self.node)
            event.accept()
            return
        super().mouseDoubleClickEvent(event)


class _TreeEdgeItem(QGraphicsLineItem):
    def __init__(self, parent_item: _TreeNodeItem, child_item: _TreeNodeItem) -> None:
        super().__init__()
        self.parent_item = parent_item
        self.child_item = child_item
        self.setZValue(-30)

    def update_geometry(self) -> None:
        start = self.parent_item.edge_out_point()
        end = self.child_item.edge_in_point()
        self.setLine(start.x(), start.y(), end.x(), end.y())


class _TreeGraphicsView(QGraphicsView):
    def __init__(self, scene: QGraphicsScene, parent: QWidget | None = None) -> None:
        super().__init__(scene, parent)
        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.setRenderHint(QPainter.RenderHint.TextAntialiasing)
        self.setViewportUpdateMode(QGraphicsView.ViewportUpdateMode.FullViewportUpdate)
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorViewCenter)
        self.setStyleSheet(
            "QGraphicsView { background: #fffdf9; border: 1px solid #e7dfd4; }"
            "QToolTip { padding: 3px; border: 1px solid #c0c0c0; }"
        )

    def wheelEvent(self, event) -> None:  # noqa: ANN001
        if event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            factor = 1.12 if event.angleDelta().y() > 0 else 1 / 1.12
            self.scale(factor, factor)
            event.accept()
            return
        super().wheelEvent(event)


class TreeViewerScreen(QWidget, WorkflowNodeScreenSupport):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._init_workflow_node_support()
        self._builder = GeneratedDatasetService()
        self._dataset: DatasetHandle | None = None
        self._root: TreeNodeData | None = None
        self._selected_dataset: DatasetHandle | None = None
        self._annotated_dataset: DatasetHandle | None = None
        self._pending_rows: list[int] = []
        self._collapsed_node_ids: set[int] = set()
        self._node_items: dict[int, _TreeNodeItem] = {}
        self._edge_items: list[_TreeEdgeItem] = []
        self._selection_sync_in_progress = False
        self._current_class_labels: list[str] = []
        self._canvas = QWidget(self)
        self._canvas._leaf_labels = {}  # type: ignore[attr-defined]
        self._canvas._node_colors = {}  # type: ignore[attr-defined]

        root = QHBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(12)

        self._sidebar = QWidget(self)
        self._sidebar.setFixedWidth(320)
        sidebar_layout = QVBoxLayout(self._sidebar)
        sidebar_layout.setContentsMargins(0, 0, 0, 0)
        sidebar_layout.setSpacing(10)
        root.addWidget(self._sidebar, 0)

        tree_box = QGroupBox(i18n.t("Tree"))
        tree_layout = QVBoxLayout(tree_box)
        tree_layout.setContentsMargins(10, 10, 10, 10)
        tree_layout.setSpacing(6)
        self._infolabel = QLabel(i18n.t("No tree."))
        self._infolabel.setWordWrap(True)
        self._status_label = self._infolabel
        tree_layout.addWidget(self._infolabel)
        sidebar_layout.addWidget(tree_box)

        display_box = QGroupBox(i18n.t("Display"))
        self._display_layout = QFormLayout(display_box)
        self._display_layout.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)
        self._display_layout.setContentsMargins(10, 10, 10, 10)
        self._display_layout.setSpacing(8)
        self._zoom_slider = self._make_slider(1, 10, 5)
        self._zoom_slider.valueChanged.connect(self._apply_zoom)
        self._display_layout.addRow(i18n.t("Zoom:"), self._zoom_slider)
        self._width_slider = self._make_slider(50, 200, 150)
        self._width_slider.valueChanged.connect(self._handle_tree_geometry_changed)
        self._display_layout.addRow(i18n.t("Width:"), self._width_slider)
        self._depth_combo = QComboBox()
        self._depth_combo.addItems([i18n.t(item) for item in TREE_LEVEL_OPTIONS])
        self._depth_combo.currentIndexChanged.connect(self._handle_tree_geometry_changed)
        self._display_layout.addRow(i18n.t("Depth:"), self._depth_combo)
        self._edge_width_combo = QComboBox()
        self._edge_width_combo.addItems([i18n.t(item) for item in EDGE_WIDTH_OPTIONS])
        self._edge_width_combo.currentIndexChanged.connect(self._handle_tree_geometry_changed)
        self._display_layout.addRow(i18n.t("Edge width:"), self._edge_width_combo)
        self._color_label = QLabel(i18n.t("Target class:"))
        self._color_combo = QComboBox()
        self._color_combo.currentIndexChanged.connect(self._refresh_tree)
        self._display_layout.addRow(self._color_label, self._color_combo)
        self._label_combo = QComboBox()
        self._label_combo.currentIndexChanged.connect(self._refresh_tree)
        self._display_layout.addRow(i18n.t("Node labels:"), self._label_combo)
        self._details_cb = QCheckBox(i18n.t("Show details in non-leaves"))
        self._details_cb.setChecked(True)
        self._details_cb.toggled.connect(self._refresh_tree)
        self._display_layout.addRow(QWidget(), self._details_cb)
        sidebar_layout.addWidget(display_box)
        sidebar_layout.addStretch(1)

        self._scene = QGraphicsScene(self)
        self._scene.selectionChanged.connect(self._handle_scene_selection_changed)
        self._scene_view = _TreeGraphicsView(self._scene, self)
        root.addWidget(self._scene_view, 1)

    def sizeHint(self) -> QSize:  # noqa: N802
        return QSize(1180, 820)

    @staticmethod
    def _make_slider(minimum: int, maximum: int, value: int) -> QSlider:
        slider = QSlider(Qt.Orientation.Horizontal)
        slider.setRange(minimum, maximum)
        slider.setValue(value)
        slider.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        return slider

    def set_input_payload(self, payload) -> None:
        self._dataset, self._root = self._extract_tree_payload(payload)
        self._refresh_tree()

    def set_tree_data(self, tree: dict[str, object] | TreeNodeData, dataset: DatasetHandle | None = None) -> None:
        self._dataset = dataset
        self._root = self._coerce_tree_root(tree)
        self._refresh_tree()

    def current_output_dataset(self) -> DatasetHandle | None:
        return self._selected_dataset

    def current_output_datasets(self) -> dict[str, DatasetHandle | None] | None:
        return {
            "Selected Data": self._selected_dataset,
            "Annotated Data": self._annotated_dataset,
        }

    def serialize_node_state(self) -> dict[str, object]:
        return {
            "zoom": self._zoom_slider.value(),
            "width": self._width_slider.value(),
            "depth": self._depth_combo.currentIndex(),
            "edge_width": self._edge_width_combo.currentIndex(),
            "show_details": self._details_cb.isChecked(),
            "label_column": self._label_combo.currentText(),
            "color_by": self._color_combo.currentText(),
            "selected_rows": list(self._pending_rows),
            "collapsed_nodes": list(self._collapsed_node_ids),
        }

    def restore_node_state(self, payload: dict[str, object]) -> None:
        self._zoom_slider.setValue(int(payload.get("zoom", 5)))
        self._width_slider.setValue(int(payload.get("width", 150)))
        self._depth_combo.setCurrentIndex(int(payload.get("depth", 0)))
        self._edge_width_combo.setCurrentIndex(int(payload.get("edge_width", 0)))
        self._details_cb.setChecked(bool(payload.get("show_details", True)))
        self._pending_rows = [
            int(index)
            for index in payload.get("selected_rows", [])
            if isinstance(index, int | float)
        ]
        self._collapsed_node_ids = {
            int(index)
            for index in payload.get("collapsed_nodes", [])
            if isinstance(index, int | float)
        }
        self._label_combo.setCurrentText(str(payload.get("label_column", i18n.t("None"))))
        self._color_combo.setCurrentText(str(payload.get("color_by", i18n.t("None"))))

    def documentation_url(self) -> str:
        return "https://orangedatamining.com/widget-catalog/visualize/treeviewer/"

    def toggle_node_expansion(self, node: TreeNodeData) -> None:
        node_id = id(node)
        if node_id in self._collapsed_node_ids:
            self._collapsed_node_ids.remove(node_id)
        else:
            self._collapsed_node_ids.add(node_id)
        self._rebuild_scene()

    def _apply_zoom(self) -> None:
        value = self._zoom_slider.value()
        factor = 0.0028 * (value ** 2) + 0.2583 * value + 1.1389
        self._scene_view.setTransform(QTransform().scale(factor / 2, factor / 2))

    def _handle_tree_geometry_changed(self) -> None:
        self._rebuild_scene()

    def _refresh_tree(self) -> None:
        self._sync_label_combo()
        self._sync_color_combo()
        self._apply_zoom()
        self._rebuild_scene()

    def _rebuild_scene(self) -> None:
        self._selection_sync_in_progress = True
        self._scene.clear()
        self._node_items = {}
        self._edge_items = []
        self._canvas._leaf_labels = {}  # type: ignore[attr-defined]
        self._canvas._node_colors = {}  # type: ignore[attr-defined]
        root = self._root

        if root is None:
            self._infolabel.setText(i18n.t("No tree."))
            self._selected_dataset = None
            self._annotated_dataset = None
            self._notify_output_changed()
            self._selection_sync_in_progress = False
            return

        visible_nodes: list[tuple[TreeNodeData, TreeNodeData | None, int]] = []
        self._collect_visible_nodes(root, None, 0, visible_nodes)
        if not visible_nodes:
            self._selection_sync_in_progress = False
            return

        html_map = {id(node): self._node_html(node) for node, _, _ in visible_nodes}
        width = float(self._width_slider.value())
        leaf_labels = self._leaf_label_mapping()
        for node, parent_node, _depth in visible_nodes:
            item = _TreeNodeItem(self, node, parent_node=parent_node)
            item.configure_html(html_map[id(node)], width=width)
            color = self._node_color(node)
            item.set_background(color)
            item.setToolTip(self._node_tooltip(node))
            self._scene.addItem(item)
            self._node_items[id(node)] = item
            self._canvas._node_colors[id(node)] = color  # type: ignore[attr-defined]
        self._canvas._leaf_labels = leaf_labels  # type: ignore[attr-defined]

        level_heights: dict[int, float] = {}
        for node, _parent, depth in visible_nodes:
            rect = self._node_items[id(node)].boundingRect()
            level_heights[depth] = max(level_heights.get(depth, 0.0), rect.height())

        level_tops: dict[int, float] = {}
        current_top = 18.0
        for depth in range(max(level_heights) + 1):
            level_tops[depth] = current_top
            current_top += level_heights.get(depth, 0.0) + NODE_VERTICAL_SPACING

        self._layout_tree(root, 0, 18.0, level_tops)
        self._create_edges(root, None, 0)
        self._update_edge_widths()
        self._restore_selection()
        nodes = self._count_nodes(root)
        leaves = self._count_leaves(root)
        self._infolabel.setText(i18n.tf("{nodes} nodes, {leaves} leaves", nodes=nodes, leaves=leaves))
        self._scene.setSceneRect(self._scene.itemsBoundingRect().adjusted(-20, -20, 40, 40))
        self._selection_sync_in_progress = False
        if not self._pending_rows:
            self._handle_selection_changed([])

    def _collect_visible_nodes(
        self,
        node: TreeNodeData,
        parent: TreeNodeData | None,
        depth: int,
        result: list[tuple[TreeNodeData, TreeNodeData | None, int]],
    ) -> None:
        result.append((node, parent, depth))
        if self._is_depth_capped(depth) or id(node) in self._collapsed_node_ids:
            return
        for child in node.children:
            self._collect_visible_nodes(child, node, depth + 1, result)

    def _layout_tree(self, node: TreeNodeData, depth: int, next_x: float, level_tops: dict[int, float]) -> tuple[float, float]:
        item = self._node_items[id(node)]
        visible_children = [] if self._is_depth_capped(depth) or id(node) in self._collapsed_node_ids else list(node.children)
        item_width = item.box_rect.width()
        if not visible_children:
            x_center = next_x + item_width / 2
            item.setPos(x_center - item.box_rect.width() / 2, level_tops[depth])
            return next_x + item_width + NODE_HORIZONTAL_SPACING, x_center

        child_centers: list[float] = []
        cursor = next_x
        for child in visible_children:
            cursor, child_center = self._layout_tree(child, depth + 1, cursor, level_tops)
            child_centers.append(child_center)
        x_center = sum(child_centers) / len(child_centers)
        item.setPos(x_center - item.box_rect.width() / 2, level_tops[depth])
        return cursor, x_center

    def _create_edges(self, node: TreeNodeData, parent: TreeNodeData | None, depth: int) -> None:
        if parent is not None:
            edge = _TreeEdgeItem(self._node_items[id(parent)], self._node_items[id(node)])
            edge.update_geometry()
            self._scene.addItem(edge)
            self._edge_items.append(edge)
        if self._is_depth_capped(depth) or id(node) in self._collapsed_node_ids:
            return
        for child in node.children:
            self._create_edges(child, node, depth + 1)

    def _is_depth_capped(self, depth: int) -> bool:
        index = self._depth_combo.currentIndex()
        if index == 0:
            return False
        visible_levels = index + 1
        return depth >= visible_levels - 1

    def _update_edge_widths(self) -> None:
        mode = self._edge_width_combo.currentIndex()
        root_count = max(1, self._node_sample_count(self._root))
        for edge in self._edge_items:
            child_count = max(1, self._node_sample_count(edge.child_item.node))
            if mode == 1:
                width = 8.0 * child_count / root_count + 0.20
            elif mode == 2:
                parent_count = max(1, self._node_sample_count(edge.parent_item.node))
                width = 8.0 * child_count / parent_count + 0.20
            else:
                width = 1.6
            edge.setPen(QPen(QColor("#6b7280"), width, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))

    def _restore_selection(self) -> None:
        wanted = set(self._pending_rows)
        if not wanted:
            return
        exact_items = [
            item
            for item in self._node_items.values()
            if set(item.node.descendant_rows()) == wanted
        ]
        if exact_items:
            exact_items[0].setSelected(True)
            return
        covered: set[int] = set()
        for item in sorted(self._node_items.values(), key=lambda candidate: len(candidate.node.descendant_rows())):
            rows = set(item.node.descendant_rows())
            if rows and rows <= wanted and not rows & covered:
                item.setSelected(True)
                covered.update(rows)

    def _handle_scene_selection_changed(self) -> None:
        if self._selection_sync_in_progress:
            return
        selected_nodes = [
            item.node
            for item in self._scene.selectedItems()
            if isinstance(item, _TreeNodeItem)
        ]
        rows = sorted({index for node in selected_nodes for index in node.descendant_rows()})
        self._handle_selection_changed(rows)

    def _extract_tree_payload(self, payload) -> tuple[DatasetHandle | None, TreeNodeData | None]:
        if payload is None:
            return None, None
        dataset = payload.dataset
        tree_value = getattr(payload, "value", None)
        if isinstance(tree_value, dict):
            embedded_dataset = tree_value.get("dataset") or tree_value.get("data")
            if isinstance(embedded_dataset, DatasetHandle):
                dataset = embedded_dataset
            tree_value = tree_value.get("tree") or tree_value.get("root") or tree_value.get("model") or tree_value
        elif isinstance(tree_value, (list, tuple)) and len(tree_value) == 2:
            left, right = tree_value
            if isinstance(left, DatasetHandle):
                dataset = left
                tree_value = right
            elif isinstance(right, DatasetHandle):
                dataset = right
                tree_value = left
        if tree_value is None and dataset is not None:
            annotations = dataset.annotations
            tree_value = annotations.get("tree_model") or annotations.get("tree_viewer")
        return dataset, self._coerce_tree_root(tree_value)

    @staticmethod
    def _coerce_tree_root(tree_value: object) -> TreeNodeData | None:
        if tree_value is None:
            return None
        if hasattr(tree_value, "tree"):
            node = TreeViewerScreen._coerce_tree_root(getattr(tree_value, "tree"))
            if node is not None:
                return node
        if hasattr(tree_value, "root"):
            node = TreeViewerScreen._coerce_tree_root(getattr(tree_value, "root"))
            if node is not None:
                return node
        return TreeNodeData.from_object(tree_value)

    def _sync_label_combo(self) -> None:
        current = self._label_combo.currentText()
        self._label_combo.blockSignals(True)
        self._label_combo.clear()
        self._label_combo.addItem(i18n.t("None"))
        if self._dataset is not None:
            for column in self._dataset.domain.meta_columns + self._dataset.domain.columns:
                self._label_combo.addItem(column.name)
        preferred = current if current and current != i18n.t("None") else self._default_label_name()
        index = self._label_combo.findText(preferred)
        self._label_combo.setCurrentIndex(index if index >= 0 else 0)
        self._label_combo.blockSignals(False)

    def _default_label_name(self) -> str:
        dataset = self._dataset
        if dataset is None:
            return i18n.t("None")
        best_name = i18n.t("None")
        best_unique = -1
        threshold = max(1, int(dataset.row_count * 0.8))
        for column in dataset.domain.meta_columns + dataset.domain.columns:
            if column.name not in dataset.dataframe.columns:
                continue
            if column.logical_type not in {"string", "text"}:
                continue
            if column.unique_count_hint > best_unique:
                best_unique = column.unique_count_hint
                best_name = column.name
        return best_name if best_unique >= threshold else i18n.t("None")

    def _sync_color_combo(self) -> None:
        current = self._color_combo.currentText()
        self._color_combo.blockSignals(True)
        self._color_combo.clear()
        if self._is_classification_tree():
            self._color_label.setText(i18n.t("Target class:"))
            self._current_class_labels = self._classification_labels()
            self._color_combo.addItem(i18n.t("None"))
            for label in self._current_class_labels:
                self._color_combo.addItem(label)
            preferred = current if current in {i18n.t("None"), *self._current_class_labels} else i18n.t("None")
        else:
            self._color_label.setText(i18n.t("Color by:"))
            options = [DEFAULT_COLOR_MODE, INSTANCE_COLOR_MODE, MEAN_COLOR_MODE, VARIANCE_COLOR_MODE]
            self._color_combo.addItems([i18n.t(item) for item in options])
            translated_options = {i18n.t(item) for item in options}
            preferred = current if current in translated_options else i18n.t(DEFAULT_COLOR_MODE)
        index = self._color_combo.findText(preferred)
        self._color_combo.setCurrentIndex(index if index >= 0 else 0)
        self._color_combo.blockSignals(False)

    def _is_classification_tree(self) -> bool:
        dataset = self._dataset
        if dataset is not None and dataset.domain.target_columns:
            target = dataset.domain.target_columns[0]
            if target.logical_type in {"categorical", "boolean", "string", "text"}:
                return True
            if target.logical_type == "numeric":
                return False
        distributions = self._all_distributions()
        if any(len(distribution) > 2 for distribution in distributions):
            return True
        predictions = [node.prediction for node in self._iter_nodes(self._root) if node.prediction]
        if predictions and any(_safe_float(prediction) is None for prediction in predictions):
            return True
        return not self._target_is_numeric()

    def _target_is_numeric(self) -> bool:
        dataset = self._dataset
        return bool(dataset is not None and dataset.domain.target_columns and dataset.domain.target_columns[0].logical_type == "numeric")

    def _classification_labels(self) -> list[str]:
        dataset = self._dataset
        labels: list[str] = []
        if dataset is not None and dataset.domain.target_columns:
            target_name = dataset.domain.target_columns[0].name
            if target_name in dataset.dataframe.columns:
                labels.extend(
                    str(value)
                    for value in dataset.dataframe.get_column(target_name).drop_nulls().unique().to_list()
                )
        if not labels:
            for node in self._iter_nodes(self._root):
                if node.prediction:
                    labels.append(node.prediction)
                if node.distribution:
                    labels.extend(f"Class {index + 1}" for index in range(len(node.distribution)))
        return list(dict.fromkeys(labels))

    def _all_distributions(self) -> list[tuple[float, ...]]:
        return [node.distribution for node in self._iter_nodes(self._root) if node.distribution]

    def _iter_nodes(self, root: TreeNodeData | None) -> list[TreeNodeData]:
        if root is None:
            return []
        nodes: list[TreeNodeData] = []
        queue = [root]
        while queue:
            node = queue.pop(0)
            nodes.append(node)
            queue.extend(node.children)
        return nodes

    def _node_distribution(self, node: TreeNodeData) -> list[float]:
        if node.distribution:
            return [float(value) for value in node.distribution]
        dataset = self._dataset
        if dataset is None or not dataset.domain.target_columns:
            return []
        target = dataset.domain.target_columns[0]
        if target.name not in dataset.dataframe.columns:
            return []
        rows = [index for index in node.descendant_rows() if 0 <= index < dataset.row_count]
        if not rows:
            return []
        series = dataset.dataframe.get_column(target.name)
        values = [series[index] for index in rows]
        if target.logical_type == "numeric":
            numeric = [_safe_float(value) for value in values]
            numeric = [value for value in numeric if value is not None]
            if not numeric:
                return []
            mean = sum(numeric) / len(numeric)
            variance = sum((value - mean) ** 2 for value in numeric) / len(numeric)
            return [mean, variance]
        labels = self._classification_labels()
        counts = {label: 0.0 for label in labels}
        for value in values:
            if value is None:
                continue
            counts[str(value)] = counts.get(str(value), 0.0) + 1.0
        return [counts[label] for label in labels]

    def _node_sample_count(self, node: TreeNodeData | None) -> int:
        if node is None:
            return 0
        if node.row_indices:
            return len(node.descendant_rows())
        distribution = self._node_distribution(node)
        if self._is_classification_tree():
            return int(sum(distribution))
        sample_count = _sample_count_from_summary(node.summary)
        if sample_count is not None:
            return sample_count
        return len(node.descendant_rows())

    def _node_prediction_label(self, node: TreeNodeData, distribution: list[float]) -> str:
        if node.prediction:
            return str(node.prediction)
        labels = self._classification_labels()
        if distribution and labels:
            index = int(max(range(len(distribution)), key=lambda item: distribution[item]))
            if index < len(labels):
                return labels[index]
        return node.label

    def _node_html(self, node: TreeNodeData) -> str:
        if node.children and not self._details_cb.isChecked():
            text = ""
        elif self._is_classification_tree():
            text = self._classification_node_html(node)
        else:
            text = self._regression_node_html(node)
        split_label = self._split_label(node)
        if split_label:
            if text:
                text += "<hr/>"
            text += split_label
        if not node.children:
            leaf_label = self._leaf_label_text(node)
            if leaf_label:
                text += "<hr/>" if text else ""
                text += leaf_label
        return f"<p style='line-height: 120%; margin-bottom: 0'>{text}</p>"

    def _classification_node_html(self, node: TreeNodeData) -> str:
        distribution = self._node_distribution(node)
        total = sum(distribution) or max(1, self._node_sample_count(node))
        if not distribution:
            return f"{self._node_sample_count(node)} {i18n.t('instances')}"
        if self._color_combo.currentIndex() > 0 and self._color_combo.currentIndex() - 1 < len(distribution):
            proportion = distribution[self._color_combo.currentIndex() - 1] / total
            count = int(round(distribution[self._color_combo.currentIndex() - 1]))
            if proportion > 0.999:
                return f"100%, {count}/{int(total)}"
            return f"{100 * proportion:2.1f}%, {count}/{int(total)}"
        prediction = self._node_prediction_label(node, distribution)
        best = max(distribution) if distribution else 0.0
        proportion = best / total if total else 0.0
        if proportion > 0.999:
            detail = f"100%, {int(best)}/{int(total)}"
        else:
            detail = f"{100 * proportion:2.1f}%, {int(round(best))}/{int(total)}"
        return f"<b>{prediction}</b><br/>{detail}"

    def _regression_node_html(self, node: TreeNodeData) -> str:
        distribution = self._node_distribution(node)
        if len(distribution) >= 2:
            mean, variance = distribution[0], distribution[1]
        else:
            mean = _safe_float(node.prediction) or 0.0
            variance = 0.0
        instances = self._node_sample_count(node)
        return f"<b>{mean:.3g}</b> ± {variance:.3g}<br/>{instances} instances"

    def _split_label(self, node: TreeNodeData) -> str:
        if not node.children:
            return ""
        if node.label and node.label.lower() not in {"root", "node"}:
            return node.label
        return ""

    def _leaf_label_text(self, node: TreeNodeData) -> str:
        dataset = self._dataset
        if dataset is None:
            return ""
        label_name = self._label_combo.currentText()
        if not label_name or label_name == i18n.t("None") or label_name not in dataset.dataframe.columns:
            return ""
        rows = [index for index in node.descendant_rows() if 0 <= index < dataset.row_count]
        if not rows:
            return ""
        values = dataset.dataframe.get_column(label_name).to_list()
        labels = [str(values[index]) for index in rows[:4] if values[index] is not None]
        if not labels:
            return ""
        text = ", ".join(labels)
        if len(rows) > 4:
            text += ", ..."
        return text

    def _leaf_label_mapping(self) -> dict[int, str]:
        root = self._root
        if root is None:
            return {}
        mapping: dict[int, str] = {}
        for node in self._iter_nodes(root):
            if node.children:
                continue
            text = self._leaf_label_text(node)
            if text:
                mapping[id(node)] = text
        return mapping

    def _node_tooltip(self, node: TreeNodeData) -> str:
        sections: list[str] = []
        rules = [ancestor.rule for ancestor in self._node_path(node) if ancestor.rule]
        if rules:
            sections.append("<b>Selection</b><br/>" + "<br/>".join(f"&nbsp;&nbsp;&nbsp;- {rule}" for rule in rules))
        distribution = self._node_distribution(node)
        if self._is_classification_tree() and distribution:
            total = sum(distribution) or 1.0
            labels = self._classification_labels()
            rows = []
            for index, value in enumerate(distribution):
                if value <= 0:
                    continue
                label = labels[index] if index < len(labels) else f"Class {index + 1}"
                color = QColor(PALETTE[index % len(PALETTE)]).name()
                rows.append(
                    "<tr>"
                    f"<td><span style='color: {color}'>■</span> {label}</td>"
                    "<td>&nbsp;&nbsp;</td>"
                    f"<td align='right'>{value:g}</td>"
                    "<td>&nbsp;&nbsp;</td>"
                    f"<td align='right'>{value / total * 100:.1f} %</td>"
                    "</tr>"
                )
            if rows:
                sections.append(f"<b>Distribution</b><br/><table>{''.join(rows)}</table>")
        elif not self._is_classification_tree() and len(distribution) >= 2:
            sections.append(f"{distribution[0]:.3g} ± {distribution[1]:.3g}<br/>({self._node_sample_count(node)} instances)")
        split = self._split_label(node)
        if split:
            sections.append(f"<b>Next split:</b> {split}")
        return "<hr/>".join(sections)

    def _node_path(self, node: TreeNodeData) -> list[TreeNodeData]:
        if self._root is None:
            return []
        path: list[TreeNodeData] = []

        def walk(current: TreeNodeData) -> bool:
            path.append(current)
            if current is node:
                return True
            for child in current.children:
                if walk(child):
                    return True
            path.pop()
            return False

        walk(self._root)
        return path[1:]

    def _node_color(self, node: TreeNodeData) -> QColor:
        return self._classification_node_color(node) if self._is_classification_tree() else self._regression_node_color(node)

    def _classification_node_color(self, node: TreeNodeData) -> QColor:
        distribution = self._node_distribution(node)
        if not distribution:
            return DEFAULT_NODE_BRUSH
        total = sum(distribution) or 1.0
        if self._color_combo.currentIndex() > 0 and self._color_combo.currentIndex() - 1 < len(distribution):
            index = self._color_combo.currentIndex() - 1
            proportion = distribution[index] / total
            base = QColor(PALETTE[index % len(PALETTE)])
            return base.lighter(int(200 - 100 * proportion))
        index = int(max(range(len(distribution)), key=lambda item: distribution[item]))
        proportion = distribution[index] / total
        base = QColor(PALETTE[index % len(PALETTE)])
        return base.lighter(int(300 - 200 * proportion))

    def _regression_node_color(self, node: TreeNodeData) -> QColor:
        mode = self._color_combo.currentText()
        if mode == i18n.t(DEFAULT_COLOR_MODE):
            return DEFAULT_REGRESSION_BRUSH
        if mode == i18n.t(INSTANCE_COLOR_MODE):
            root_count = max(1, self._node_sample_count(self._root))
            node_count = self._node_sample_count(node)
            return QColor(DEFAULT_REGRESSION_BRUSH).lighter(int(120 - 20 * node_count / root_count))
        distribution = self._node_distribution(node)
        if mode == i18n.t(MEAN_COLOR_MODE):
            dataset = self._dataset
            if dataset is None or not dataset.domain.target_columns:
                return DEFAULT_REGRESSION_BRUSH
            target_name = dataset.domain.target_columns[0].name
            if target_name not in dataset.dataframe.columns:
                return DEFAULT_REGRESSION_BRUSH
            values = dataset.dataframe.get_column(target_name).cast(pl.Float64, strict=False).drop_nulls().to_list()
            if not values or not distribution:
                return DEFAULT_REGRESSION_BRUSH
            return _continuous_gradient(distribution[0], min(values), max(values))
        if len(distribution) >= 2:
            variances = [dist[1] for dist in self._all_distributions() if len(dist) >= 2]
            max_variance = max(variances) if variances else distribution[1]
            return QColor(DEFAULT_REGRESSION_BRUSH).lighter(int(120 - 20 * distribution[1] / max(max_variance, 1e-9)))
        return DEFAULT_REGRESSION_BRUSH

    def _handle_selection_changed(self, rows: list[int]) -> None:
        self._pending_rows = sorted({index for index in rows})
        self._selected_dataset, self._annotated_dataset = build_selection_outputs(
            self._dataset,
            self._pending_rows,
            generated_by="tree-viewer",
            service=self._builder,
        )
        self._notify_output_changed()

    @staticmethod
    def _count_nodes(root: TreeNodeData | None) -> int:
        if root is None:
            return 0
        return 1 + sum(TreeViewerScreen._count_nodes(child) for child in root.children)

    @staticmethod
    def _count_leaves(root: TreeNodeData | None) -> int:
        if root is None:
            return 0
        if not root.children:
            return 1
        return sum(TreeViewerScreen._count_leaves(child) for child in root.children)
