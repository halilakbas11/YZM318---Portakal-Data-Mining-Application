from __future__ import annotations

from math import degrees, log, sqrt

import numpy as np

from PySide6.QtCore import QRectF, QSize, Qt
from PySide6.QtGui import QColor, QPainter, QPen
from PySide6.QtWidgets import (
    QComboBox,
    QFrame,
    QGraphicsRectItem,
    QGraphicsScene,
    QGraphicsView,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListView,
    QListWidget,
    QListWidgetItem,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from portakal_app.models import WorkflowPayload
from portakal_app.tree_artifacts import DecisionTreeArtifact, RandomForestArtifact
from portakal_app.ui import i18n
from portakal_app.ui.screens.node_screen import WorkflowNodeScreenSupport
from portakal_app.ui.screens.pythagorean_tree_screen import (
    Point,
    Square,
    _BACKGROUND,
    _BORDER_COLOR,
    _CONT_MID,
    _PythagorasLayout,
    _continuous_color,
    _safe_float_series,
)


class _PreviewGraphicsView(QGraphicsView):
    def __init__(self, scene: QGraphicsScene, parent: QWidget | None = None) -> None:
        super().__init__(scene, parent)
        self.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        self.setFrameShape(QFrame.Shape.NoFrame)
        self.setBackgroundBrush(_BACKGROUND)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

    def fit_content(self) -> None:
        rect = self.scene().itemsBoundingRect()
        if not rect.isValid() or rect.isNull():
            return
        rect = rect.adjusted(-16, -16, 16, 16)
        self.resetTransform()
        self.fitInView(rect, Qt.AspectRatioMode.KeepAspectRatio)


class _ForestPreviewCard(QFrame):
    def __init__(self, tree: DecisionTreeArtifact, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._tree = tree
        self._selected = False
        self._square_items: list[QGraphicsRectItem] = []

        self.setFrameShape(QFrame.Shape.NoFrame)
        self.setObjectName("forestPreviewCard")
        self.setStyleSheet("#forestPreviewCard { background: #fffdf9; }")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(0)

        self._scene = QGraphicsScene(self)
        self._view = _PreviewGraphicsView(self._scene, self)
        layout.addWidget(self._view, 1)

    def set_selected(self, selected: bool) -> None:
        self._selected = bool(selected)
        self.update()

    def update_preview(self, depth_limit: int, target_class_index: int, size_calc_idx: int) -> None:
        self._scene.clear()
        self._square_items = []

        builder = _PythagorasLayout(weight_adjustment=self._weight_adjustment(size_calc_idx))
        root = builder.build(self._tree, self._tree.root, Square(Point(0, 0), 200, -3.141592653589793 / 2))

        def draw(node, depth: int) -> None:
            if depth > depth_limit:
                return
            center, length, angle = node.square
            rect = QRectF(center.x - length / 2, center.y - length / 2, length, length)
            item = QGraphicsRectItem(rect)
            item.setTransformOriginPoint(rect.center())
            item.setRotation(degrees(angle))
            pen = QPen(_BORDER_COLOR, 0.75)
            pen.setCosmetic(True)
            item.setPen(pen)
            item.setBrush(self._node_color(node.node_id, target_class_index))
            self._scene.addItem(item)
            self._square_items.append(item)
            for child in node.children:
                draw(child, depth + 1)

        draw(root, 0)
        self._view.fit_content()

    def square_colors(self) -> list[str]:
        return [item.brush().color().name() for item in self._square_items]

    def square_rects(self) -> list[tuple[float, float, float, float]]:
        rects: list[tuple[float, float, float, float]] = []
        for item in self._square_items:
            rect = item.rect()
            rects.append((rect.x(), rect.y(), rect.width(), rect.height()))
        return rects

    def _weight_adjustment(self, size_calc_idx: int):
        if size_calc_idx == 1:
            return lambda value: sqrt(max(1.0, float(value)))
        if size_calc_idx == 2:
            return lambda value: log(max(1.0, float(value)) + 1.0)
        return lambda value: float(value)

    def _classification_palette(self) -> list[QColor]:
        return [QColor(color) for color in self._tree.class_colors[: max(1, len(self._tree.class_values) or 1)]]

    def _regression_summary(self, node_id: int) -> tuple[float, float]:
        target_values = _safe_float_series(self._tree.instances, self._tree.target_name)
        indices = self._tree.get_indices(node_id)
        subset = target_values[indices] if indices else np.asarray([], dtype=float)
        subset = subset[np.isfinite(subset)]
        if subset.size == 0:
            return float("nan"), float("nan")
        return float(np.mean(subset)), float(np.std(subset))

    def _node_color(self, node_id: int, target_class_index: int) -> QColor:
        if self._tree.has_discrete_class:
            distribution = np.asarray(self._tree.get_distribution(node_id)[0], dtype=float)
            total = float(np.sum(distribution))
            palette = self._classification_palette()
            if distribution.size == 0 or not palette:
                return QColor("white")
            if target_class_index > 0:
                target = target_class_index - 1
                p = float(distribution[target] / total) if total > 0 else 0.0
                return QColor(palette[target % len(palette)]).lighter(int(200 - 100 * p))
            modus = int(np.argmax(distribution))
            p = float(distribution[modus] / (total or 1.0))
            return QColor(palette[modus % len(palette)]).lighter(int(400 - 300 * p))

        values = _safe_float_series(self._tree.instances, self._tree.target_name)
        finite = values[np.isfinite(values)]
        if target_class_index == 0:
            return QColor("#ffffff")
        if finite.size == 0:
            return QColor(_CONT_MID)
        mean, std = self._regression_summary(node_id)
        if target_class_index == 1:
            return _continuous_color(mean, float(np.min(finite)), float(np.max(finite)))
        return _continuous_color(std, 0.0, float(np.std(finite)))

    def paintEvent(self, event) -> None:
        super().paintEvent(event)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        if self._selected:
            painter.setPen(QPen(QColor("#7da2ce"), 1.4))
            painter.setBrush(QColor(217, 232, 252, 70))
        else:
            painter.setPen(QPen(QColor("#ebebeb"), 1.0))
            painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawRoundedRect(self.rect().adjusted(0, 0, -1, -1), 6, 6)


class _ClickToClearListWidget(QListWidget):
    def mousePressEvent(self, event) -> None:
        if self.itemAt(event.position().toPoint()) is None:
            self.clearSelection()
            event.accept()
            return
        super().mousePressEvent(event)


class PythagoreanForestScreen(QWidget, WorkflowNodeScreenSupport):
    def __init__(self) -> None:
        super().__init__()
        self._init_workflow_node_support()
        self._forest: RandomForestArtifact | None = None
        self._selected_tree: DecisionTreeArtifact | None = None
        self._selected_index: int | None = None
        self._cards: list[_ForestPreviewCard] = []

        self._pending_depth_limit = 0
        self._pending_target_class_index = 0
        self._pending_size_calc_idx = 0
        self._pending_zoom = 200
        self._pending_selected_index: int | None = None

        self._build_ui()
        self.clear()

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

        info_box = QGroupBox(i18n.t("Forest"), controls)
        info_layout = QVBoxLayout(info_box)
        self.ui_info = QLabel(info_box)
        self.ui_info.setWordWrap(True)
        info_layout.addWidget(self.ui_info)
        controls_layout.addWidget(info_box)

        display_box = QGroupBox(i18n.t("Display"), controls)
        display_layout = QGridLayout(display_box)
        display_layout.setContentsMargins(10, 10, 10, 10)
        display_layout.setHorizontalSpacing(8)
        display_layout.setVerticalSpacing(10)

        display_layout.addWidget(QLabel(i18n.t("Depth"), display_box), 0, 0)
        self.ui_depth_slider = QSlider(Qt.Orientation.Horizontal, display_box)
        self.ui_depth_slider.setRange(0, 0)
        self.ui_depth_slider.valueChanged.connect(self._on_depth_changed)
        display_layout.addWidget(self.ui_depth_slider, 0, 1)

        self._target_label = QLabel(i18n.t("Target class"), display_box)
        display_layout.addWidget(self._target_label, 1, 0)
        self.ui_target_class_combo = QComboBox(display_box)
        self.ui_target_class_combo.currentIndexChanged.connect(self._on_target_class_changed)
        display_layout.addWidget(self.ui_target_class_combo, 1, 1)

        display_layout.addWidget(QLabel(i18n.t("Size"), display_box), 2, 0)
        self.ui_size_calc_combo = QComboBox(display_box)
        self.ui_size_calc_combo.addItems((i18n.t("Normal"), i18n.t("Square root"), i18n.t("Logarithmic")))
        self.ui_size_calc_combo.currentIndexChanged.connect(self._on_size_changed)
        display_layout.addWidget(self.ui_size_calc_combo, 2, 1)

        display_layout.addWidget(QLabel(i18n.t("Zoom"), display_box), 3, 0)
        self.ui_zoom_slider = QSlider(Qt.Orientation.Horizontal, display_box)
        self.ui_zoom_slider.setRange(100, 400)
        self.ui_zoom_slider.setValue(200)
        self.ui_zoom_slider.valueChanged.connect(self._on_zoom_changed)
        display_layout.addWidget(self.ui_zoom_slider, 3, 1)
        controls_layout.addWidget(display_box)

        self._status_label = QLabel(self)
        self._status_label.setWordWrap(True)
        controls_layout.addWidget(self._status_label)
        controls_layout.addStretch(1)
        root.addWidget(controls, 0)

        self._list_widget = _ClickToClearListWidget(self)
        self._list_widget.setViewMode(QListView.ViewMode.IconMode)
        self._list_widget.setFlow(QListView.Flow.LeftToRight)
        self._list_widget.setResizeMode(QListView.ResizeMode.Adjust)
        self._list_widget.setWrapping(True)
        self._list_widget.setMovement(QListView.Movement.Static)
        self._list_widget.setSelectionMode(QListWidget.SelectionMode.SingleSelection)
        self._list_widget.setSpacing(4)
        self._list_widget.itemSelectionChanged.connect(self._on_selection_changed)
        root.addWidget(self._list_widget, 1)

    def clear(self) -> None:
        self._forest = None
        self._selected_tree = None
        self._selected_index = None
        self._cards = []
        self._list_widget.clear()
        self.ui_info.setText(i18n.t("No forest on input."))
        self._status_label.setText(i18n.t("Connect a random forest to browse its trees."))
        self._set_controls_enabled(False)
        self._populate_target_combo()

    def _set_controls_enabled(self, enabled: bool) -> None:
        self.ui_depth_slider.setEnabled(enabled)
        self.ui_target_class_combo.setEnabled(enabled)
        self.ui_size_calc_combo.setEnabled(enabled)
        self.ui_zoom_slider.setEnabled(enabled)

    def set_input_payload(self, payload: WorkflowPayload | None) -> None:
        if payload is None:
            self.clear()
            self._notify_output_changed()
            return

        artifact = payload.value
        if not isinstance(artifact, RandomForestArtifact):
            self.clear()
            self._status_label.setText(i18n.t("Pythagorean Forest expects a random forest model on input."))
            self._notify_output_changed()
            return

        self._forest = artifact
        self._selected_tree = None
        self._selected_index = None
        self._populate_target_combo()
        self._sync_controls_from_pending_state()
        self._populate_list()
        self.ui_info.setText(i18n.tf("Trees: {count}", count=len(self._forest.trees)))
        self._status_label.setText(i18n.t("Select a tree to send it to downstream widgets."))
        self._set_controls_enabled(True)

    def current_output_dataset(self):
        return self._selected_tree

    def serialize_node_state(self) -> dict[str, object]:
        return {
            "depth_limit": self.ui_depth_slider.value(),
            "target_class_index": self.ui_target_class_combo.currentIndex(),
            "size_calc_idx": self.ui_size_calc_combo.currentIndex(),
            "zoom": self.ui_zoom_slider.value(),
            "selected_index": self._selected_index,
        }

    def restore_node_state(self, payload: dict[str, object]) -> None:
        self._pending_depth_limit = int(payload.get("depth_limit", 0) or 0)
        self._pending_target_class_index = int(payload.get("target_class_index", 0) or 0)
        self._pending_size_calc_idx = int(payload.get("size_calc_idx", 0) or 0)
        self._pending_zoom = int(payload.get("zoom", 200) or 200)
        selected_index = payload.get("selected_index")
        self._pending_selected_index = int(selected_index) if isinstance(selected_index, int) else None
        if self._forest is not None:
            self._sync_controls_from_pending_state()
            self._populate_list()

    def help_text(self) -> str:
        return i18n.t("Browse trees in a random forest and send the selected tree to Pythagorean Tree.")

    def _populate_target_combo(self) -> None:
        self.ui_target_class_combo.blockSignals(True)
        self.ui_target_class_combo.clear()
        if self._forest is None:
            self._target_label.setText(i18n.t("Target class"))
            self.ui_target_class_combo.blockSignals(False)
            return
        if self._forest.has_discrete_class:
            self._target_label.setText(i18n.t("Target class"))
            self.ui_target_class_combo.addItem(i18n.t("None"))
            for value in self._forest.class_values:
                self.ui_target_class_combo.addItem(str(value).title())
        else:
            self._target_label.setText(i18n.t("Node color"))
            self.ui_target_class_combo.addItems((i18n.t("None"), i18n.t("Mean"), i18n.t("Standard deviation")))
        self.ui_target_class_combo.blockSignals(False)

    def _sync_controls_from_pending_state(self) -> None:
        if self._forest is None:
            return
        self.ui_depth_slider.blockSignals(True)
        self.ui_depth_slider.setMaximum(max(0, self._forest.max_depth))
        depth = self._pending_depth_limit if self._pending_depth_limit else self._forest.max_depth
        self.ui_depth_slider.setValue(max(0, min(depth, self._forest.max_depth)))
        self.ui_depth_slider.blockSignals(False)

        self.ui_target_class_combo.blockSignals(True)
        self.ui_target_class_combo.setCurrentIndex(max(0, min(self._pending_target_class_index, self.ui_target_class_combo.count() - 1)))
        self.ui_target_class_combo.blockSignals(False)

        self.ui_size_calc_combo.blockSignals(True)
        self.ui_size_calc_combo.setCurrentIndex(max(0, min(self._pending_size_calc_idx, self.ui_size_calc_combo.count() - 1)))
        self.ui_size_calc_combo.blockSignals(False)

        self.ui_zoom_slider.blockSignals(True)
        self.ui_zoom_slider.setValue(max(self.ui_zoom_slider.minimum(), min(self._pending_zoom, self.ui_zoom_slider.maximum())))
        self.ui_zoom_slider.blockSignals(False)

    def _populate_list(self) -> None:
        self._list_widget.blockSignals(True)
        self._list_widget.clear()
        self._cards = []
        if self._forest is None:
            self._list_widget.blockSignals(False)
            return

        for tree in self._forest.trees:
            item = QListWidgetItem(self._list_widget)
            item.setSizeHint(self._item_size())
            card = _ForestPreviewCard(tree, self._list_widget)
            card.setFixedSize(self._card_size())
            card.update_preview(self.ui_depth_slider.value(), self.ui_target_class_combo.currentIndex(), self.ui_size_calc_combo.currentIndex())
            self._list_widget.addItem(item)
            self._list_widget.setItemWidget(item, card)
            self._cards.append(card)

        self._list_widget.blockSignals(False)
        if self._pending_selected_index is not None and 0 <= self._pending_selected_index < self._list_widget.count():
            self._list_widget.setCurrentRow(self._pending_selected_index)
            self._pending_selected_index = None
        self._sync_card_selection()

    def _item_size(self) -> QSize:
        side = self.ui_zoom_slider.value()
        return QSize(side, side)

    def _card_size(self) -> QSize:
        side = max(84, self.ui_zoom_slider.value() - 10)
        return QSize(side, side)

    def _sync_card_selection(self) -> None:
        selected_rows = {index.row() for index in self._list_widget.selectedIndexes()}
        for idx, card in enumerate(self._cards):
            card.set_selected(idx in selected_rows)

    def _refresh_cards(self) -> None:
        for index, card in enumerate(self._cards):
            item = self._list_widget.item(index)
            if item is not None:
                item.setSizeHint(self._item_size())
            card.setFixedSize(self._card_size())
            card.update_preview(self.ui_depth_slider.value(), self.ui_target_class_combo.currentIndex(), self.ui_size_calc_combo.currentIndex())
        self._sync_card_selection()

    def _on_depth_changed(self, value: int) -> None:
        self._pending_depth_limit = int(value)
        self._refresh_cards()

    def _on_target_class_changed(self, index: int) -> None:
        self._pending_target_class_index = int(index)
        self._refresh_cards()

    def _on_size_changed(self, index: int) -> None:
        self._pending_size_calc_idx = int(index)
        self._refresh_cards()

    def _on_zoom_changed(self, value: int) -> None:
        self._pending_zoom = int(value)
        self._refresh_cards()

    def _on_selection_changed(self) -> None:
        self._sync_card_selection()
        indexes = self._list_widget.selectedIndexes()
        if not indexes or self._forest is None:
            self._selected_index = None
            self._selected_tree = None
            self._notify_output_changed()
            return

        self._selected_index = indexes[0].row()
        tree = self._forest.trees[self._selected_index]
        tree.instances = self._forest.instances
        tree.meta_target_class_index = self.ui_target_class_combo.currentIndex()
        tree.meta_size_calc_idx = self.ui_size_calc_combo.currentIndex()
        tree.meta_depth_limit = self.ui_depth_slider.value()
        self._selected_tree = tree
        self._status_label.setText(i18n.tf("Selected tree: {index}", index=self._selected_index + 1))
        self._notify_output_changed()
