from __future__ import annotations

import random
from collections.abc import Callable
from dataclasses import replace

from PySide6.QtCore import QPoint, QRect, QSize, Qt, Signal
from PySide6.QtGui import QColor, QPainter, QPen
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QScrollArea,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from portakal_app.data.models import PaintDataPoint, PaintDataSnapshot, DatasetHandle
from portakal_app.data.services.paint_data_service import PaintDataService
from portakal_app.ui import i18n
from portakal_app.ui.screens.node_screen import WorkflowNodeScreenSupport


LABEL_COLORS = (
    "#4db7eb",
    "#eb5a46",
    "#4caf50",
    "#8b6ad9",
    "#f4b942",
    "#14b8a6",
)


class PaintDataCanvas(QWidget):
    pointsChanged = Signal()
    selectionChanged = Signal(int)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setMinimumSize(520, 420)
        self._points: list[PaintDataPoint] = []
        self._selected_indexes: set[int] = set()
        self._current_label = "C1"
        self._label_colors: dict[str, str] = {"C1": LABEL_COLORS[0], "C2": LABEL_COLORS[1]}
        self._tool = "brush"
        self._radius = 36
        self._intensity = 45
        self._x_axis_label = "x"
        self._y_axis_label = "y"
        self._dragging = False
        self._selection_origin: QPoint | None = None
        self._selection_rect: QRect | None = None

    def set_points(self, points: tuple[PaintDataPoint, ...], labels: tuple[str, ...]) -> None:
        self._points = list(points)
        self._selected_indexes.clear()
        self._label_colors = {label: LABEL_COLORS[index % len(LABEL_COLORS)] for index, label in enumerate(labels)}
        if labels:
            self._current_label = labels[0]
        self.selectionChanged.emit(0)
        self.update()

    def set_current_label(self, label: str) -> None:
        self._current_label = label

    def set_axis_labels(self, x_label: str, y_label: str) -> None:
        self._x_axis_label = x_label.strip() or "x"
        self._y_axis_label = y_label.strip() or "y"
        self.update()

    def set_tool(self, tool: str) -> None:
        self._tool = tool
        self._selection_rect = None
        self.update()

    def set_radius(self, radius: int) -> None:
        self._radius = radius

    def set_intensity(self, intensity: int) -> None:
        self._intensity = intensity

    def current_points(self) -> tuple[PaintDataPoint, ...]:
        return tuple(self._points)

    def selected_count(self) -> int:
        return len(self._selected_indexes)

    def plot_point_count(self) -> int:
        return len(self._points)

    def mousePressEvent(self, event) -> None:
        if event.button() != Qt.MouseButton.LeftButton:
            return super().mousePressEvent(event)
        self._dragging = True
        if self._tool == "select":
            self._selection_origin = event.position().toPoint()
            self._selection_rect = QRect(self._selection_origin, self._selection_origin)
        else:
            self._apply_tool(event.position().toPoint())
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:
        if not self._dragging:
            return super().mouseMoveEvent(event)
        if self._tool == "select" and self._selection_origin is not None:
            self._selection_rect = QRect(self._selection_origin, event.position().toPoint()).normalized()
            self.update()
        elif self._tool in {"brush", "clear", "magnet", "jitter"}:
            self._apply_tool(event.position().toPoint())
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton and self._tool == "select" and self._selection_rect is not None:
            self._selected_indexes = {
                index
                for index, point in enumerate(self._points)
                if self._selection_rect.contains(self._to_canvas_point(point))
            }
            self.selectionChanged.emit(len(self._selected_indexes))
            self._selection_rect = None
            self.update()
        self._dragging = False
        self._selection_origin = None
        super().mouseReleaseEvent(event)

    def paintEvent(self, event) -> None:
        super().paintEvent(event)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), QColor("#ffffff"))

        plot_rect = self._plot_rect()
        painter.setPen(QPen(QColor("#d8d2c7"), 1.2))
        painter.drawRect(plot_rect)
        painter.setPen(QPen(QColor("#cfc7b7"), 1.0))
        for step in range(6):
            ratio = step / 5
            x = int(plot_rect.left() + (plot_rect.width() * ratio))
            y = int(plot_rect.bottom() - (plot_rect.height() * ratio))
            painter.drawLine(x, plot_rect.bottom(), x, plot_rect.bottom() + 5)
            painter.drawLine(plot_rect.left() - 5, y, plot_rect.left(), y)
            painter.drawText(x - 10, plot_rect.bottom() + 20, f"{ratio:.1f}".rstrip("0").rstrip("."))
            painter.drawText(plot_rect.left() - 34, y + 4, f"{ratio:.1f}".rstrip("0").rstrip("."))

        painter.setPen(QPen(QColor("#3b3127"), 1.0))
        painter.drawText(plot_rect.center().x(), self.height() - 12, self._x_axis_label)
        painter.save()
        painter.translate(18, plot_rect.center().y())
        painter.rotate(-90)
        painter.drawText(0, 0, self._y_axis_label)
        painter.restore()

        for index, point in enumerate(self._points):
            color = QColor(self._label_colors.get(point.label, LABEL_COLORS[0]))
            center = self._to_canvas_point(point)
            pen = QPen(color, 2.6)
            painter.setPen(pen)
            size = 6
            painter.drawLine(center.x() - size, center.y(), center.x() + size, center.y())
            painter.drawLine(center.x(), center.y() - size, center.x(), center.y() + size)
            if index in self._selected_indexes:
                painter.setPen(QPen(QColor("#1f2937"), 1.2))
                painter.drawEllipse(center, 8, 8)

        if self._selection_rect is not None:
            painter.setPen(QPen(QColor("#cf9440"), 1.2, Qt.PenStyle.DashLine))
            painter.fillRect(self._selection_rect, QColor(226, 169, 82, 40))
            painter.drawRect(self._selection_rect)

    def _apply_tool(self, position: QPoint) -> None:
        logical_x, logical_y = self._to_logical(position)
        if logical_x is None or logical_y is None:
            return
        if self._tool == "put":
            self._points.append(PaintDataPoint(x=logical_x, y=logical_y, label=self._current_label))
        elif self._tool == "brush":
            count = max(2, self._intensity // 12)
            radius_x = self._radius / max(1, self._plot_rect().width())
            radius_y = self._radius / max(1, self._plot_rect().height())
            for _index in range(count):
                dx = random.uniform(-radius_x, radius_x)
                dy = random.uniform(-radius_y, radius_y)
                self._points.append(
                    PaintDataPoint(
                        x=max(0.0, min(1.0, logical_x + dx)),
                        y=max(0.0, min(1.0, logical_y + dy)),
                        label=self._current_label,
                    )
                )
        elif self._tool == "clear":
            self._points = [
                point for point in self._points if not self._is_in_radius(point, logical_x, logical_y, self._radius)
            ]
            self._selected_indexes.clear()
        elif self._tool == "magnet":
            points: list[PaintDataPoint] = []
            factor = max(0.08, self._intensity / 160)
            for point in self._points:
                if self._is_in_radius(point, logical_x, logical_y, self._radius):
                    points.append(
                        PaintDataPoint(
                            x=point.x + ((logical_x - point.x) * factor),
                            y=point.y + ((logical_y - point.y) * factor),
                            label=point.label,
                        )
                    )
                else:
                    points.append(point)
            self._points = points
        elif self._tool == "jitter":
            points = []
            factor = max(0.01, self._intensity / 500)
            for point in self._points:
                if self._is_in_radius(point, logical_x, logical_y, self._radius):
                    points.append(
                        PaintDataPoint(
                            x=max(0.0, min(1.0, point.x + random.uniform(-factor, factor))),
                            y=max(0.0, min(1.0, point.y + random.uniform(-factor, factor))),
                            label=point.label,
                        )
                    )
                else:
                    points.append(point)
            self._points = points
        self.pointsChanged.emit()
        self.update()

    def _is_in_radius(self, point: PaintDataPoint, x: float, y: float, radius: int) -> bool:
        center = self._to_canvas_point(point)
        target = self._to_canvas_point(PaintDataPoint(x=x, y=y, label=point.label))
        dx = center.x() - target.x()
        dy = center.y() - target.y()
        return (dx * dx) + (dy * dy) <= radius * radius

    def _plot_rect(self) -> QRect:
        return self.rect().adjusted(54, 18, -26, -48)

    def _to_logical(self, point: QPoint) -> tuple[float | None, float | None]:
        rect = self._plot_rect()
        if not rect.contains(point):
            return None, None
        x = (point.x() - rect.left()) / max(1, rect.width())
        y = 1.0 - ((point.y() - rect.top()) / max(1, rect.height()))
        return max(0.0, min(1.0, x)), max(0.0, min(1.0, y))

    def _to_canvas_point(self, point: PaintDataPoint) -> QPoint:
        rect = self._plot_rect()
        x = rect.left() + int(point.x * rect.width())
        y = rect.bottom() - int(point.y * rect.height())
        return QPoint(x, y)


class PaintDataScreen(QWidget, WorkflowNodeScreenSupport):
    def __init__(self, parent: QWidget | None = None, service: PaintDataService | None = None) -> None:
        super().__init__(parent)
        self._init_workflow_node_support()
        self._service = service or PaintDataService()
        self._dataset: DatasetHandle | None = None
        self._output_dataset: DatasetHandle | None = None
        self._input_snapshot = PaintDataSnapshot()
        self._callbacks: list[Callable[[DatasetHandle], None]] = []
        self._ui_sync_guard = False
        self._canvas = PaintDataCanvas(self)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(12)

        self._sidebar = QFrame(self)
        self._sidebar.setProperty("panel", True)
        self._sidebar.setMinimumWidth(248)
        sidebar_layout = QVBoxLayout(self._sidebar)
        sidebar_layout.setContentsMargins(12, 12, 12, 12)
        sidebar_layout.setSpacing(10)

        sidebar_layout.addWidget(self._build_names_panel())
        sidebar_layout.addWidget(self._build_tools_panel())
        sidebar_layout.addStretch(1)

        self._reset_button = QPushButton("Reset to Input Data")
        self._reset_button.setProperty("secondary", True)
        self._reset_button.clicked.connect(self._reset_to_input)
        sidebar_layout.addWidget(self._reset_button)

        self._auto_send_checkbox = QCheckBox("Send Automatically")
        self._auto_send_checkbox.setChecked(True)
        sidebar_layout.addWidget(self._auto_send_checkbox)

        self._send_button = QPushButton("Send Data")
        self._send_button.setProperty("primary", True)
        self._send_button.clicked.connect(self._emit_dataset)
        sidebar_layout.addWidget(self._send_button)

        self._sidebar_scroll = QScrollArea(self)
        self._sidebar_scroll.setWidgetResizable(True)
        self._sidebar_scroll.setFrameShape(QFrame.Shape.NoFrame)
        self._sidebar_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._sidebar_scroll.setWidget(self._sidebar)
        self._sidebar_scroll.setFixedWidth(276)

        layout.addWidget(self._sidebar_scroll)

        self._canvas.pointsChanged.connect(self._handle_canvas_changed)
        self._canvas.selectionChanged.connect(self._set_selection_status)
        self._labels_list.currentTextChanged.connect(self._canvas.set_current_label)

        right_layout = QVBoxLayout()
        right_layout.setSpacing(8)
        right_layout.addWidget(self._canvas, 1)

        footer = QHBoxLayout()
        self._selection_status = QLabel(i18n.tf("Selected: {count}", count=0))
        self._selection_status.setProperty("muted", True)
        footer.addWidget(self._selection_status)
        footer.addStretch(1)
        self._point_status = QLabel(i18n.tf("Points: {count}", count=0))
        self._point_status.setProperty("muted", True)
        footer.addWidget(self._point_status)
        right_layout.addLayout(footer)
        layout.addLayout(right_layout, 1)

        self._load_snapshot(PaintDataSnapshot())

    def sizeHint(self) -> QSize:
        return QSize(1020, 760)

    def minimumSizeHint(self) -> QSize:
        return QSize(900, 680)

    def _build_names_panel(self) -> QFrame:
        frame = QFrame(self)
        layout = QVBoxLayout(frame)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        self._names_header = QLabel("Names")
        self._names_header.setProperty("sectionTitle", True)
        self._names_header.setStyleSheet("font-size: 12pt;")
        layout.addWidget(self._names_header)

        self._source_columns_title = QLabel("Source Columns")
        self._source_columns_title.setProperty("sectionTitle", True)
        self._source_columns_title.setStyleSheet("font-size: 11pt;")
        layout.addWidget(self._source_columns_title)

        row_x_source = QHBoxLayout()
        self._x_source_label = QLabel("X Column:")
        row_x_source.addWidget(self._x_source_label)
        self._x_source_combo = QComboBox(self)
        self._x_source_combo.setEditable(False)
        self._x_source_combo.currentIndexChanged.connect(self._handle_source_columns_changed)
        row_x_source.addWidget(self._x_source_combo, 1)
        layout.addLayout(row_x_source)

        row_y_source = QHBoxLayout()
        self._y_source_label = QLabel("Y Column:")
        row_y_source.addWidget(self._y_source_label)
        self._y_source_combo = QComboBox(self)
        self._y_source_combo.setEditable(False)
        self._y_source_combo.currentIndexChanged.connect(self._handle_source_columns_changed)
        row_y_source.addWidget(self._y_source_combo, 1)
        layout.addLayout(row_y_source)

        row_label_source = QHBoxLayout()
        self._label_source_label = QLabel("Label Column:")
        row_label_source.addWidget(self._label_source_label)
        self._label_source_combo = QComboBox(self)
        self._label_source_combo.setEditable(False)
        self._label_source_combo.currentIndexChanged.connect(self._handle_source_columns_changed)
        row_label_source.addWidget(self._label_source_combo, 1)
        layout.addLayout(row_label_source)

        self._output_names_title = QLabel("Output Names")
        self._output_names_title.setProperty("sectionTitle", True)
        self._output_names_title.setStyleSheet("font-size: 11pt;")
        layout.addWidget(self._output_names_title)

        row_x = QHBoxLayout()
        self._x_name_label = QLabel("Variable X:")
        row_x.addWidget(self._x_name_label)
        self._x_name_input = QLineEdit("x")
        self._x_name_input.textChanged.connect(lambda text: self._handle_output_name_changed("x", text))
        row_x.addWidget(self._x_name_input, 1)
        layout.addLayout(row_x)

        row_y = QHBoxLayout()
        self._y_name_label = QLabel("Variable Y:")
        row_y.addWidget(self._y_name_label)
        self._y_name_input = QLineEdit("y")
        self._y_name_input.textChanged.connect(lambda text: self._handle_output_name_changed("y", text))
        row_y.addWidget(self._y_name_input, 1)
        layout.addLayout(row_y)

        self._labels_title = QLabel("Labels")
        self._labels_title.setProperty("sectionTitle", True)
        self._labels_title.setStyleSheet("font-size: 11pt;")
        layout.addWidget(self._labels_title)

        self._labels_list = QListWidget(self)
        layout.addWidget(self._labels_list, 1)

        buttons = QHBoxLayout()
        self._add_label_button = QPushButton("+")
        self._add_label_button.setProperty("secondary", True)
        self._add_label_button.clicked.connect(self._add_label)
        buttons.addWidget(self._add_label_button)

        self._remove_label_button = QPushButton("-")
        self._remove_label_button.setProperty("secondary", True)
        self._remove_label_button.clicked.connect(self._remove_selected_label)
        buttons.addWidget(self._remove_label_button)
        layout.addLayout(buttons)
        return frame

    def _build_tools_panel(self) -> QFrame:
        frame = QFrame(self)
        layout = QVBoxLayout(frame)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)

        header = QLabel("Tools")
        header.setProperty("sectionTitle", True)
        header.setStyleSheet("font-size: 12pt;")
        layout.addWidget(header)

        grid = QGridLayout()
        grid.setHorizontalSpacing(8)
        grid.setVerticalSpacing(8)
        self._tool_buttons: dict[str, QPushButton] = {}
        tools = [
            ("brush", "Brush"),
            ("put", "Put"),
            ("select", "Select"),
            ("jitter", "Jitter"),
            ("magnet", "Magnet"),
            ("clear", "Clear"),
        ]
        for index, (tool_id, label) in enumerate(tools):
            button = QPushButton(label)
            button.setCheckable(True)
            button.clicked.connect(lambda checked=False, value=tool_id: self._set_tool(value))
            self._tool_buttons[tool_id] = button
            grid.addWidget(button, index // 3, index % 3)
        layout.addLayout(grid)

        self._radius_slider = self._build_slider(layout, "Radius:", 12, 80, 36, self._canvas.set_radius)
        self._intensity_slider = self._build_slider(layout, "Intensity:", 10, 100, 45, self._canvas.set_intensity)
        self._symbol_slider = self._build_slider(layout, "Symbol:", 1, 10, 5, lambda _value: None)

        self._set_tool("brush")
        return frame

    def _build_slider(self, layout: QVBoxLayout, label_text: str, minimum: int, maximum: int, value: int, callback) -> QSlider:
        row = QHBoxLayout()
        label = QLabel(label_text)
        row.addWidget(label)
        slider = QSlider(Qt.Orientation.Horizontal, self)
        slider.setRange(minimum, maximum)
        slider.setValue(value)
        slider.valueChanged.connect(callback)
        row.addWidget(slider, 1)
        layout.addLayout(row)
        return slider

    def set_paint_data_service(self, service: PaintDataService) -> None:
        self._service = service

    def on_apply_requested(self, callback: Callable[[DatasetHandle], None]) -> None:
        self._callbacks.append(callback)

    def set_dataset(self, dataset: DatasetHandle | str | None) -> None:
        self._dataset = dataset if isinstance(dataset, DatasetHandle) else None
        if self._dataset is None:
            self._output_dataset = None
            self._input_snapshot = PaintDataSnapshot()
            self._load_snapshot(self._input_snapshot)
            return

        if self._dataset.annotations.get("generated_by") == "paint-data":
            self._output_dataset = self._dataset
            return

        self._output_dataset = None
        snapshot = self._service.build_snapshot(
            self._dataset,
            x_source_name=self._current_source_column(self._x_source_combo),
            y_source_name=self._current_source_column(self._y_source_combo),
            label_source_name=self._current_source_column(self._label_source_combo),
            x_output_name=self._preferred_output_name(
                self._x_name_input.text(),
                self._input_snapshot.x_name,
                self._input_snapshot.x_source_name,
            ),
            y_output_name=self._preferred_output_name(
                self._y_name_input.text(),
                self._input_snapshot.y_name,
                self._input_snapshot.y_source_name,
            ),
        )
        self._input_snapshot = snapshot
        self._load_snapshot(snapshot, selected_label=self._selected_label())

    def help_text(self) -> str:
        return (
            "Paint a two-dimensional dataset with Orange-like brush, put, select, magnet, jitter, and clear tools, "
            "then send the generated data into the workflow."
        )

    def documentation_url(self) -> str:
        return "https://orangedatamining.com/widget-catalog/data/paint-data/"

    def footer_status_text(self) -> str:
        total = self._canvas.plot_point_count()
        selected = self._canvas.selected_count()
        return f"{selected} | {total}" if selected else str(total)

    def set_input_payload(self, payload) -> None:
        dataset = payload.dataset if payload is not None else None
        self.set_dataset(dataset)

    def current_output_dataset(self) -> DatasetHandle | None:
        return self._output_dataset

    def serialize_node_state(self) -> dict[str, object]:
        snapshot = self._current_snapshot()
        return {
            "snapshot": {
                "x_name": snapshot.x_name,
                "y_name": snapshot.y_name,
                "label_name": snapshot.label_name,
                "x_source_name": snapshot.x_source_name,
                "y_source_name": snapshot.y_source_name,
                "label_source_name": snapshot.label_source_name,
                "labels": list(snapshot.labels),
                "points": [{"x": point.x, "y": point.y, "label": point.label} for point in snapshot.points],
                "source_name": snapshot.source_name,
            },
            "auto_send": self._auto_send_checkbox.isChecked(),
            "output_committed": self._output_dataset is not None,
        }

    def restore_node_state(self, payload: dict[str, object]) -> None:
        snapshot_payload = payload.get("snapshot")
        if isinstance(snapshot_payload, dict):
            snapshot = PaintDataSnapshot(
                x_name=str(snapshot_payload.get("x_name") or "x"),
                y_name=str(snapshot_payload.get("y_name") or "y"),
                label_name=str(snapshot_payload.get("label_name") or "class"),
                x_source_name=self._optional_snapshot_text(snapshot_payload.get("x_source_name")),
                y_source_name=self._optional_snapshot_text(snapshot_payload.get("y_source_name")),
                label_source_name=self._optional_snapshot_text(snapshot_payload.get("label_source_name")),
                labels=tuple(str(item) for item in snapshot_payload.get("labels", ()) if str(item).strip()) or ("C1", "C2"),
                points=tuple(
                    PaintDataPoint(
                        x=float(point.get("x", 0.0)),
                        y=float(point.get("y", 0.0)),
                        label=str(point.get("label") or "C1"),
                    )
                    for point in snapshot_payload.get("points", ())
                    if isinstance(point, dict)
                ),
                source_name=str(snapshot_payload.get("source_name") or "Painted Data"),
            )
            self._input_snapshot = snapshot
            self._load_snapshot(snapshot)
        self._auto_send_checkbox.setChecked(bool(payload.get("auto_send", True)))
        if bool(payload.get("output_committed")) and self._canvas.plot_point_count() > 0:
            self._emit_dataset()

    def report_snapshot(self) -> dict[str, object]:
        return {
            "title": "Paint Data",
            "items": [
                {
                    "title": "Canvas",
                    "timestamp": "Current session",
                    "details": [
                        f"Points: {self._canvas.plot_point_count()}",
                        f"Selected: {self._canvas.selected_count()}",
                        f"X: {self._x_name_input.text().strip() or 'x'}",
                        f"Y: {self._y_name_input.text().strip() or 'y'}",
                    ],
                }
            ],
        }

    def data_preview_snapshot(self) -> dict[str, object]:
        snapshot = self._current_snapshot()
        rows = [[f"{point.x:.4f}", f"{point.y:.4f}", point.label] for point in snapshot.points[:200]]
        return {
            "summary": f"{len(snapshot.points)} painted points",
            "headers": [snapshot.x_name, snapshot.y_name, snapshot.label_name],
            "rows": rows,
        }

    def _set_tool(self, tool_id: str) -> None:
        for current_tool, button in self._tool_buttons.items():
            button.setChecked(current_tool == tool_id)
            if current_tool == tool_id:
                button.setStyleSheet(
                    "background-color: #f0c98a; border: 1px solid #cf9440; border-radius: 10px; padding: 10px 8px;"
                )
            else:
                button.setStyleSheet(
                    "background-color: #fffaf0; border: 1px solid #ddcfbb; border-radius: 10px; padding: 10px 8px;"
                )
        self._canvas.set_tool(tool_id)

    def _load_snapshot(self, snapshot: PaintDataSnapshot, *, selected_label: str | None = None) -> None:
        self._ui_sync_guard = True
        self._x_name_input.setText(snapshot.x_name)
        self._y_name_input.setText(snapshot.y_name)
        self._canvas.set_axis_labels(snapshot.x_name, snapshot.y_name)
        self._populate_source_selectors(snapshot)
        self._labels_list.clear()
        for label in snapshot.labels:
            QListWidgetItem(label, self._labels_list)
        if self._labels_list.count():
            target_row = 0
            if selected_label:
                for index in range(self._labels_list.count()):
                    if self._labels_list.item(index).text() == selected_label:
                        target_row = index
                        break
            self._labels_list.setCurrentRow(target_row)
        self._canvas.set_points(snapshot.points, snapshot.labels)
        self._ui_sync_guard = False
        self._update_status()

    def _current_snapshot(self) -> PaintDataSnapshot:
        labels = tuple(self._labels_list.item(index).text() for index in range(self._labels_list.count())) or ("C1", "C2")
        return PaintDataSnapshot(
            x_name=self._x_name_input.text().strip() or "x",
            y_name=self._y_name_input.text().strip() or "y",
            label_name=self._input_snapshot.label_name or "class",
            labels=labels,
            points=self._canvas.current_points(),
            source_name="Painted Data",
            x_source_name=self._current_source_column(self._x_source_combo),
            y_source_name=self._current_source_column(self._y_source_combo),
            label_source_name=self._current_source_column(self._label_source_combo),
        )

    def _handle_canvas_changed(self) -> None:
        if self._ui_sync_guard:
            return
        self._canvas.set_axis_labels(self._x_name_input.text(), self._y_name_input.text())
        self._update_status()
        if self._auto_send_checkbox.isChecked():
            self._emit_dataset()

    def _handle_output_name_changed(self, axis: str, value: str) -> None:
        if self._ui_sync_guard:
            return

        if self._dataset is not None and self._dataset.annotations.get("generated_by") != "paint-data":
            source_names = set(self._service.numeric_source_columns(self._dataset))
            candidate = value.strip()
            combo = self._x_source_combo if axis == "x" else self._y_source_combo
            current_source = self._current_source_column(combo)
            if candidate and candidate in source_names and candidate != current_source:
                self._set_combo_value(combo, candidate)
                return

        self._handle_canvas_changed()

    def _handle_source_columns_changed(self) -> None:
        if self._ui_sync_guard or self._dataset is None:
            return
        if self._dataset.annotations.get("generated_by") == "paint-data":
            return

        snapshot = self._service.build_snapshot(
            self._dataset,
            x_source_name=self._current_source_column(self._x_source_combo),
            y_source_name=self._current_source_column(self._y_source_combo),
            label_source_name=self._current_source_column(self._label_source_combo),
            x_output_name=self._preferred_output_name(
                self._x_name_input.text(),
                self._input_snapshot.x_name,
                self._input_snapshot.x_source_name,
            ),
            y_output_name=self._preferred_output_name(
                self._y_name_input.text(),
                self._input_snapshot.y_name,
                self._input_snapshot.y_source_name,
            ),
        )
        self._input_snapshot = snapshot
        self._load_snapshot(snapshot, selected_label=self._selected_label())
        if self._auto_send_checkbox.isChecked():
            self._emit_dataset()

    def _update_status(self) -> None:
        self._point_status.setText(i18n.tf("Points: {count}", count=self._canvas.plot_point_count()))
        self._selection_status.setText(i18n.tf("Selected: {count}", count=self._canvas.selected_count()))
        self._send_button.setEnabled(self._canvas.plot_point_count() > 0)

    def _set_selection_status(self, count: int) -> None:
        self._selection_status.setText(i18n.tf("Selected: {count}", count=count))

    def refresh_translations(self) -> None:
        self._names_header.setText(i18n.t("Names"))
        self._source_columns_title.setText(i18n.t("Source Columns"))
        self._x_source_label.setText(i18n.t("X Column:"))
        self._y_source_label.setText(i18n.t("Y Column:"))
        self._label_source_label.setText(i18n.t("Label Column:"))
        self._output_names_title.setText(i18n.t("Output Names"))
        self._x_name_label.setText(i18n.t("Variable X:"))
        self._y_name_label.setText(i18n.t("Variable Y:"))
        self._labels_title.setText(i18n.t("Labels"))
        self._populate_source_selectors(self._current_snapshot())
        self._update_status()

    def _add_label(self) -> None:
        next_name = f"C{self._labels_list.count() + 1}"
        snapshot = self._current_snapshot()
        self._load_snapshot(replace(snapshot, labels=(*snapshot.labels, next_name)), selected_label=next_name)

    def _remove_selected_label(self) -> None:
        current_row = self._labels_list.currentRow()
        if current_row < 0 or self._labels_list.count() <= 1:
            return
        removed_label = self._labels_list.item(current_row).text()
        item = self._labels_list.takeItem(current_row)
        del item
        remaining_labels = tuple(self._labels_list.item(index).text() for index in range(self._labels_list.count()))
        points = tuple(
            replace(point, label=remaining_labels[0] if point.label == removed_label else point.label)
            for point in self._canvas.current_points()
        )
        snapshot = PaintDataSnapshot(
            x_name=self._x_name_input.text().strip() or "x",
            y_name=self._y_name_input.text().strip() or "y",
            label_name=self._input_snapshot.label_name or "class",
            labels=remaining_labels,
            points=points,
            source_name="Painted Data",
            x_source_name=self._current_source_column(self._x_source_combo),
            y_source_name=self._current_source_column(self._y_source_combo),
            label_source_name=self._current_source_column(self._label_source_combo),
        )
        next_label = remaining_labels[min(current_row, len(remaining_labels) - 1)] if remaining_labels else None
        self._load_snapshot(snapshot, selected_label=next_label)

    def _reset_to_input(self) -> None:
        self._load_snapshot(self._input_snapshot, selected_label=self._selected_label())

    def _emit_dataset(self) -> None:
        snapshot = self._current_snapshot()
        if not snapshot.points:
            return
        dataset = self._service.build_dataset(snapshot)
        self._output_dataset = dataset
        for callback in self._callbacks:
            callback(dataset)
        self._notify_output_changed()

    def _selected_label(self) -> str | None:
        current_item = self._labels_list.currentItem()
        if current_item is None:
            return None
        text = current_item.text().strip()
        return text or None

    def _populate_source_selectors(self, snapshot: PaintDataSnapshot) -> None:
        numeric_columns = list(self._service.numeric_source_columns(self._dataset))
        if not numeric_columns:
            numeric_columns = [
                name
                for name in (snapshot.x_source_name or snapshot.x_name, snapshot.y_source_name or snapshot.y_name)
                if name
            ]
        label_columns = list(self._service.label_source_columns(self._dataset))
        if not label_columns and snapshot.label_source_name:
            label_columns = [snapshot.label_source_name]

        self._x_source_combo.clear()
        self._y_source_combo.clear()
        for name in numeric_columns:
            self._x_source_combo.addItem(name, name)
            self._y_source_combo.addItem(name, name)

        self._label_source_combo.clear()
        self._label_source_combo.addItem(i18n.t("None"), None)
        for name in label_columns:
            self._label_source_combo.addItem(name, name)

        self._set_combo_value(self._x_source_combo, snapshot.x_source_name or snapshot.x_name)
        self._set_combo_value(self._y_source_combo, snapshot.y_source_name or snapshot.y_name)
        self._set_combo_value(self._label_source_combo, snapshot.label_source_name)

        numeric_enabled = self._dataset is not None and len(numeric_columns) >= 2
        self._x_source_combo.setEnabled(numeric_enabled)
        self._y_source_combo.setEnabled(numeric_enabled)
        self._label_source_combo.setEnabled(self._dataset is not None and bool(label_columns))

    def _set_combo_value(self, combo: QComboBox, value: str | None) -> None:
        if value is None:
            combo.setCurrentIndex(0 if combo.count() else -1)
            return
        for index in range(combo.count()):
            if combo.itemData(index) == value or combo.itemText(index) == value:
                combo.setCurrentIndex(index)
                return
        combo.addItem(value, value)
        combo.setCurrentIndex(combo.count() - 1)

    def _current_source_column(self, combo: QComboBox) -> str | None:
        current_index = combo.currentIndex()
        if current_index < 0:
            return None
        data = combo.itemData(current_index)
        if data is None:
            return None
        return str(data)

    def _preferred_output_name(self, current_text: str, previous_output_name: str, previous_source_name: str | None) -> str | None:
        value = current_text.strip()
        if not value:
            return None
        previous_source = previous_source_name or previous_output_name
        if value == previous_source and previous_output_name == previous_source:
            return None
        return value

    def _optional_snapshot_text(self, value: object) -> str | None:
        if value is None:
            return None
        text = str(value).strip()
        return text or None
