from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from functools import reduce
from pathlib import Path
from typing import Any
from uuid import uuid4

import polars as pl

from PySide6.QtCore import QPointF, QRectF, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QBrush, QPainter, QPainterPath, QPen, QStandardItem, QStandardItemModel, QTransform
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QRadioButton,
    QSizePolicy,
    QToolTip,
    QVBoxLayout,
    QWidget,
)

from portakal_app.data.errors import DatasetSaveError, UnsupportedFormatError
from portakal_app.data.models import DatasetHandle
from portakal_app.data.services.generated_dataset_service import GeneratedDatasetService
from portakal_app.data.services.save_data_service import SaveDataService
from portakal_app.models import WorkflowPayload
from portakal_app.ui import i18n
from portakal_app.ui.screens.node_screen import WorkflowNodeScreenSupport


IDENTITY_TOKEN = "__identity__"
EQUALITY_TOKEN = "__equality__"
SELECTED_COLUMN_NAME = "Selected"
MAX_INPUTS = 5
INPUT_CHANNELS = tuple(f"Data {index}" for index in range(1, MAX_INPUTS + 1))
OUTPUT_CHANNELS = ("Selected Data", "Annotated Data")
_MISSING = object()

_SET_COLORS = (
    QColor(224, 112, 32, 120),
    QColor(59, 130, 246, 120),
    QColor(34, 197, 94, 120),
    QColor(168, 85, 247, 120),
    QColor(244, 63, 94, 120),
)

_CATEGORY_ANCHORS = (
    ((90, "center", "bottom"),),
    ((180, "right", "middle"), (0, "left", "middle")),
    ((150, "right", "bottom"), (30, "left", "bottom"), (270, "center", "top")),
    (
        (315, "left", "top"),
        (225, "right", "top"),
        (75, "left", "bottom"),
        (105, "right", "bottom"),
    ),
    (
        (85, "center", "bottom"),
        (13, "left", "middle"),
        (301, "left", "top"),
        (229, "right", "top"),
        (157, "right", "middle"),
    ),
)


@dataclass(frozen=True)
class _InputSet:
    slot: str
    dataset: DatasetHandle
    title: str
    items: list[Any]
    unique_count: int
    total_count: int


@dataclass(frozen=True)
class _OutputColumn:
    name: str
    role: str
    source_slot: str
    source_name: str


def _is_missing(value: Any) -> bool:
    return value is None or (isinstance(value, float) and math.isnan(value))


def _normalize_value(value: Any) -> Any:
    if _is_missing(value):
        return _MISSING
    return value


def _values_equal(left: Any, right: Any) -> bool:
    return _normalize_value(left) == _normalize_value(right)


def _display_value(value: Any) -> str:
    return "" if value is _MISSING or _is_missing(value) else str(value)


def _ordered_union(keys_by_slot: dict[str, dict[Any, Any]], slot_order: list[str]) -> list[Any]:
    ordered: list[Any] = []
    seen: set[Any] = set()
    for slot in slot_order:
        for key in keys_by_slot.get(slot, {}):
            if key in seen:
                continue
            seen.add(key)
            ordered.append(key)
    return ordered


def _sorted_input_slots(inputs: dict[str, DatasetHandle]) -> list[str]:
    return sorted(inputs, key=lambda label: INPUT_CHANNELS.index(label) if label in INPUT_CHANNELS else MAX_INPUTS)


def _polar_point(center: QPointF, radius: float, angle_deg: float) -> QPointF:
    angle = math.radians(angle_deg)
    return QPointF(center.x() + radius * math.cos(angle), center.y() - radius * math.sin(angle))


def _anchor_text_rect(text_rect: QRectF, anchor_pos: QPointF, anchor_h: str, anchor_v: str) -> QRectF:
    if anchor_h == "left":
        x = anchor_pos.x()
    elif anchor_h == "center":
        x = anchor_pos.x() - text_rect.width() / 2
    else:
        x = anchor_pos.x() - text_rect.width()

    if anchor_v == "top":
        y = anchor_pos.y()
    elif anchor_v == "middle":
        y = anchor_pos.y() - text_rect.height() / 2
    else:
        y = anchor_pos.y() - text_rect.height()

    return QRectF(x, y, text_rect.width(), text_rect.height())


def _radians(angle: float) -> float:
    return 2 * math.pi * angle / 360


def _unit_point(angle: float, radius: float = 1.0) -> tuple[float, float]:
    rad = _radians(angle)
    return radius * math.cos(rad), radius * math.sin(rad)


def _ellipse_path(center: tuple[float, float], a: float, b: float, rotation: float = 0) -> QPainterPath:
    path = QPainterPath()
    path.addEllipse(QRectF(-a, -b, 2 * a, 2 * b))
    if rotation:
        path = QTransform().rotate(rotation).map(path)
    path.translate(*center)
    return path


def _circle_path(center: tuple[float, float], radius: float = 1.0) -> QPainterPath:
    return _ellipse_path(center, radius, radius, rotation=0)


def _venn_diagram_paths(count: int) -> list[QPainterPath]:
    if count < 1 or count > MAX_INPUTS:
        raise ValueError(f"Unsupported Venn size: {count}")

    if count == 1:
        return [_circle_path((0, 0), 0.5)]
    if count == 2:
        return [_circle_path(_unit_point(angle, 1 / 6), 1 / 3) for angle in (180, 0)]
    if count == 3:
        return [_circle_path(_unit_point(150 - 120 * index, 1 / 6), 1 / 3) for index in range(3)]
    if count == 4:
        return [
            _ellipse_path((0.15, -0.03), 0.35, 0.20, 45),
            _ellipse_path((-0.15, -0.03), 0.35, 0.20, 135),
            _ellipse_path((0.0, 0.07), 0.35, 0.20, 45),
            _ellipse_path((0.0, 0.07), 0.35, 0.20, 134),
        ]

    distance = 0.13
    a, b = 0.48, 0.24
    return [
        _ellipse_path(_unit_point((1 - index) * 72, distance), a, b, rotation=90 - index * 72)
        for index in range(5)
    ]


def _set_key(value: int, bits: int) -> tuple[bool, ...]:
    return tuple(bool(value & (2**index)) for index in range(bits))


def _venn_intersections(paths: list[QPainterPath]) -> dict[tuple[bool, ...], QPainterPath]:
    count = len(paths)
    intersections: dict[tuple[bool, ...], QPainterPath] = {}
    for index in range(2**count):
        key = _set_key(index, count)
        if not any(key):
            intersections[key] = QPainterPath()
            continue
        included = [path for path, include in zip(paths, key) if include]
        excluded = [path for path, include in zip(paths, key) if not include]
        region = reduce(QPainterPath.intersected, included)
        for excluded_path in excluded:
            region = region.subtracted(excluded_path)
        intersections[key] = region
    return intersections


def _disjoint_set_label(index: int, count: int) -> str:
    labels = []
    for position, included in enumerate(_set_key(index, count)):
        name = chr(ord("A") + position)
        labels.append(name if included else f"{name}^c")
    return " & ".join(labels)


def _string_like_column(dataset: DatasetHandle, column_name: str) -> bool:
    if column_name not in dataset.dataframe.columns:
        return False
    dtype_name = str(dataset.dataframe[column_name].dtype).lower()
    if any(token in dtype_name for token in ("string", "str", "utf8", "categorical", "enum")):
        return True
    column = next((col for col in dataset.domain.columns if col.name == column_name), None)
    return bool(column and column.logical_type in {"categorical", "text"} and not dataset.dataframe[column_name].dtype.is_numeric())


class _VennDiagramCanvas(QWidget):
    selectionChanged = Signal(list)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._set_titles: list[str] = []
        self._set_counts: list[str] = []
        self._area_counts: dict[int, int] = {}
        self._area_tooltips: dict[int, str] = {}
        self._selected_areas: set[int] = set()
        self._hovered_area: int | None = None
        self._dirty_layout = True
        self._layout_size = None
        self._set_paths: list[QPainterPath] = []
        self._area_paths: dict[int, QPainterPath] = {}
        self.setMinimumHeight(320)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMouseTracking(True)

    def clear(self) -> None:
        self._set_titles = []
        self._set_counts = []
        self._area_counts = {}
        self._area_tooltips = {}
        self._selected_areas = set()
        self._hovered_area = None
        self._dirty_layout = True
        self.update()

    def set_content(
        self,
        set_titles: list[str],
        set_counts: list[str],
        area_counts: dict[int, int],
        area_tooltips: dict[int, str],
        selected_areas: set[int],
    ) -> None:
        self._set_titles = list(set_titles)
        self._set_counts = list(set_counts)
        self._area_counts = dict(area_counts)
        self._area_tooltips = dict(area_tooltips)
        self._selected_areas = set(selected_areas)
        self._hovered_area = None
        self._dirty_layout = True
        self.update()

    def set_selected_areas(self, selected_areas: set[int]) -> None:
        normalized = set(selected_areas) & set(self._area_counts)
        if normalized == self._selected_areas:
            return
        self._selected_areas = normalized
        self.update()
        self.selectionChanged.emit(sorted(self._selected_areas))

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        self._dirty_layout = True

    def leaveEvent(self, _event) -> None:
        self._hovered_area = None
        QToolTip.hideText()
        self.update()

    def mouseMoveEvent(self, event) -> None:
        self._ensure_layout()
        area = self._area_at(event.position().toPoint())
        if area != self._hovered_area:
            self._hovered_area = area
            self.update()
        if area is not None and area in self._area_tooltips:
            QToolTip.showText(event.globalPosition().toPoint(), self._area_tooltips[area], self)
        else:
            QToolTip.hideText()

    def mousePressEvent(self, event) -> None:
        if event.button() != Qt.MouseButton.LeftButton:
            return super().mousePressEvent(event)
        self._ensure_layout()
        area = self._area_at(event.position().toPoint())
        if area is None:
            return

        selected = set(self._selected_areas)
        modifiers = event.modifiers()
        if modifiers & Qt.KeyboardModifier.AltModifier:
            selected.discard(area)
        elif modifiers & Qt.KeyboardModifier.ControlModifier:
            if area in selected:
                selected.remove(area)
            else:
                selected.add(area)
        elif modifiers & Qt.KeyboardModifier.ShiftModifier:
            selected.add(area)
        else:
            selected = {area}
        self.set_selected_areas(selected)

    def paintEvent(self, _event) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), QColor("#fffdf9"))

        if not self._set_titles:
            painter.setPen(QColor("#8d877d"))
            painter.drawText(
                self.rect(),
                Qt.AlignmentFlag.AlignCenter,
                i18n.t("Connect up to 5 datasets to compare overlaps."),
            )
            return

        self._ensure_layout()

        for index, path in enumerate(self._set_paths):
            color = _SET_COLORS[index % len(_SET_COLORS)]
            painter.setPen(QPen(color.darker(140), 2))
            painter.setBrush(QBrush(color))
            painter.drawPath(path)

        for index, path in self._area_paths.items():
            if path.isEmpty():
                continue
            if index == self._hovered_area and index not in self._selected_areas:
                painter.setPen(QPen(QColor("#2563eb"), 2))
                painter.setBrush(QColor(37, 99, 235, 28))
                painter.drawPath(path)
            if index in self._selected_areas:
                painter.setPen(QPen(QColor("#dc2626"), 2))
                painter.setBrush(QColor(220, 38, 38, 34))
                painter.drawPath(path)

        painter.setPen(QColor("#111827"))
        font = painter.font()
        font.setPointSize(10)
        font.setBold(True)
        painter.setFont(font)

        shape = reduce(QPainterPath.united, self._set_paths)
        center = shape.boundingRect().center()
        radius = max(shape.boundingRect().width(), shape.boundingRect().height()) / 2 + 28
        for index, title in enumerate(self._set_titles):
            label = f"{title}\n{self._set_counts[index]}"
            rect = painter.boundingRect(
                QRectF(0, 0, 160, 48),
                Qt.TextFlag.TextWordWrap | Qt.AlignmentFlag.AlignCenter,
                label,
            )
            anchor_angle, anchor_h, anchor_v = _CATEGORY_ANCHORS[len(self._set_titles) - 1][index]
            anchor = _polar_point(center, radius, anchor_angle)
            target_rect = _anchor_text_rect(rect, anchor, anchor_h, anchor_v)
            painter.drawText(target_rect, Qt.TextFlag.TextWordWrap | Qt.AlignmentFlag.AlignCenter, label)

        count_font = painter.font()
        count_font.setPointSize(9)
        count_font.setBold(True)
        painter.setFont(count_font)
        for index, path in self._area_paths.items():
            if index == 0 or path.isEmpty():
                continue
            count = self._area_counts.get(index, 0)
            center_point = path.boundingRect().center()
            text_rect = QRectF(center_point.x() - 24, center_point.y() - 12, 48, 24)
            painter.drawText(text_rect, Qt.AlignmentFlag.AlignCenter, str(count))

    def _area_at(self, point) -> int | None:
        candidates = sorted(
            self._area_paths.items(),
            key=lambda item: item[1].boundingRect().width() * item[1].boundingRect().height(),
        )
        for index, path in candidates:
            if index == 0 or path.isEmpty():
                continue
            if path.contains(QPointF(point)):
                return index
        return None

    def _ensure_layout(self) -> None:
        if not self._dirty_layout and self._layout_size == self.size():
            return

        self._set_paths = []
        self._area_paths = {}
        count = len(self._set_titles)
        if not count:
            return

        raw_paths = [QTransform().scale(1, -1).map(path) for path in _venn_diagram_paths(count)]
        union_rect = reduce(QRectF.united, (path.boundingRect() for path in raw_paths))

        chart_rect = QRectF(self.rect()).adjusted(132, 72, -132, -108)
        if chart_rect.width() < 40 or chart_rect.height() < 40:
            chart_rect = QRectF(self.rect()).adjusted(48, 48, -48, -72)

        scale = min(chart_rect.width() / max(union_rect.width(), 1.0), chart_rect.height() / max(union_rect.height(), 1.0))
        transform = QTransform()
        transform.translate(chart_rect.center().x(), chart_rect.center().y())
        transform.scale(scale, scale)
        transform.translate(-union_rect.center().x(), -union_rect.center().y())

        self._set_paths = [transform.map(path) for path in raw_paths]
        intersections = _venn_intersections(self._set_paths)
        self._area_paths = {index: intersections[_set_key(index, count)] for index in range(2**count)}
        self._layout_size = self.size()
        self._dirty_layout = False


class VennDiagramScreen(QWidget, WorkflowNodeScreenSupport):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._init_workflow_node_support()
        self._generated_datasets = GeneratedDatasetService()
        self._save_data_service = SaveDataService()
        self._screen_token = uuid4().hex[:8]
        self._inputs: dict[str, DatasetHandle] = {}
        self._title_overrides: dict[str, str] = {}
        self._selected_areas: set[int] = set()
        self._disjoint_items: list[set[Any]] = []
        self._area_slots: list[list[str]] = []
        self._itemsets: list[_InputSet] = []
        self._selected_dataset: DatasetHandle | None = None
        self._annotated_dataset: DatasetHandle | None = None
        self._pending_match_code = IDENTITY_TOKEN
        self._rebuild_timer = QTimer(self)
        self._rebuild_timer.setSingleShot(True)
        self._rebuild_timer.setInterval(0)
        self._rebuild_timer.timeout.connect(self._rebuild)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        header = QLabel(i18n.t("Venn Diagram"))
        header.setProperty("sectionTitle", True)
        layout.addWidget(header)

        description = QLabel(
            i18n.t(
                "Compare overlaps between up to 5 datasets by rows or feature names, then send Selected Data or Annotated Data downstream."
            )
        )
        description.setWordWrap(True)
        description.setProperty("muted", True)
        layout.addWidget(description)

        settings_box = QGroupBox(i18n.t("Settings"))
        settings_layout = QVBoxLayout(settings_box)
        settings_layout.setContentsMargins(10, 10, 10, 10)
        settings_layout.setSpacing(8)

        mode_row = QHBoxLayout()
        self._columns_radio = QRadioButton(i18n.t("Columns (features)"))
        self._rows_radio = QRadioButton(i18n.t("Rows (instances), matched by"))
        self._rows_radio.setChecked(True)
        self._columns_radio.toggled.connect(self._on_configuration_changed)
        self._rows_radio.toggled.connect(self._on_configuration_changed)
        mode_row.addWidget(self._columns_radio)
        mode_row.addWidget(self._rows_radio)
        mode_row.addStretch(1)
        settings_layout.addLayout(mode_row)

        match_row = QHBoxLayout()
        match_row.addWidget(QLabel(i18n.t("Match by:")))
        self._match_combo = QComboBox()
        self._match_combo.currentIndexChanged.connect(self._on_configuration_changed)
        match_row.addWidget(self._match_combo, 1)
        self._duplicates_checkbox = QCheckBox(i18n.t("Output duplicates"))
        self._duplicates_checkbox.stateChanged.connect(self._apply_or_defer)
        match_row.addWidget(self._duplicates_checkbox)
        settings_layout.addLayout(match_row)

        layout.addWidget(settings_box)

        titles_box = QGroupBox(i18n.t("Set Titles"))
        titles_layout = QGridLayout(titles_box)
        titles_layout.setContentsMargins(10, 10, 10, 10)
        titles_layout.setHorizontalSpacing(10)
        titles_layout.setVerticalSpacing(6)
        self._title_inputs: dict[str, QLineEdit] = {}
        self._title_count_labels: dict[str, QLabel] = {}
        for row, slot in enumerate(INPUT_CHANNELS):
            slot_label = QLabel(slot)
            editor = QLineEdit()
            editor.setPlaceholderText(i18n.t("Unused input"))
            editor.textEdited.connect(lambda text, current=slot: self._on_title_edited(current, text))
            count_label = QLabel("-")
            count_label.setProperty("muted", True)
            titles_layout.addWidget(slot_label, row, 0)
            titles_layout.addWidget(editor, row, 1)
            titles_layout.addWidget(count_label, row, 2)
            self._title_inputs[slot] = editor
            self._title_count_labels[slot] = count_label
        layout.addWidget(titles_box)

        diagram_box = QGroupBox(i18n.t("Diagram"))
        diagram_layout = QVBoxLayout(diagram_box)
        diagram_layout.setContentsMargins(6, 6, 6, 6)
        self._diagram = _VennDiagramCanvas()
        self._diagram.selectionChanged.connect(self._on_selection_changed)
        diagram_layout.addWidget(self._diagram)
        layout.addWidget(diagram_box, 1)

        self._status_label = QLabel(i18n.t("Connect up to 5 datasets to compare overlaps."))
        self._status_label.setWordWrap(True)
        self._status_label.setProperty("muted", True)
        layout.addWidget(self._status_label)

        footer = QHBoxLayout()
        self.cb_apply_auto = QCheckBox(i18n.t("Apply Automatically"))
        self.cb_apply_auto.setChecked(True)
        footer.addWidget(self.cb_apply_auto)
        footer.addStretch(1)
        self._apply_button = QPushButton(i18n.t("Apply"))
        self._apply_button.setProperty("primary", True)
        self._apply_button.clicked.connect(self._apply)
        footer.addWidget(self._apply_button)
        layout.addLayout(footer)

    def set_input_payload(self, payload: WorkflowPayload | None) -> None:
        if payload is None:
            self._inputs = {}
        elif payload.dataset is not None:
            slot = payload.port_label if payload.port_label in INPUT_CHANNELS else self._first_available_slot()
            if slot is not None:
                self._inputs[slot] = payload.dataset
        self._rebuild_timer.start()

    def current_output_datasets(self) -> dict[str, DatasetHandle | None] | None:
        return {
            OUTPUT_CHANNELS[0]: self._selected_dataset,
            OUTPUT_CHANNELS[1]: self._annotated_dataset,
        }

    def set_save_data_service(self, service: SaveDataService) -> None:
        self._save_data_service = service

    def documentation_url(self) -> str:
        return "https://orangedatamining.com/widget-catalog/visualize/venndiagram/"

    def help_text(self) -> str:
        return (
            "Compare overlaps between up to 5 datasets. You can compare rows by "
            "instance identity, instance equality, or a shared string column, or "
            "switch to feature mode to compare overlapping columns."
        )

    def footer_status_text(self) -> str:
        annotated_rows = self._annotated_dataset.row_count if self._annotated_dataset is not None else 0
        selected_rows = self._selected_dataset.row_count if self._selected_dataset is not None else 0
        if annotated_rows == 0:
            return "0"
        if selected_rows:
            return f"{selected_rows} | {annotated_rows}"
        return str(annotated_rows)

    def exportable_dataset(self) -> DatasetHandle | None:
        return self._selected_dataset or self._annotated_dataset

    def can_save_export_dataset(self) -> bool:
        return self.exportable_dataset() is not None

    def save_export_dataset(self) -> None:
        dataset = self.exportable_dataset()
        if dataset is None:
            return

        target_path, _selected_filter = QFileDialog.getSaveFileName(
            self,
            i18n.t("Save Data As"),
            str(self._default_export_path(dataset)),
            "Data Files (*.csv *.xlsx *.parquet);;All Files (*.*)",
        )
        if not target_path:
            return
        self._write_export_dataset(dataset, Path(target_path))

    def data_preview_snapshot(self) -> dict[str, object]:
        dataset = self._selected_dataset or self._annotated_dataset
        if dataset is None:
            return {"summary": i18n.t("No preview available."), "headers": [], "rows": []}
        return {
            "summary": self._dataset_summary(dataset),
            "headers": list(dataset.dataframe.columns),
            "rows": self._preview_rows(dataset),
        }

    def detailed_data_snapshot(self) -> dict[str, object]:
        selected = self._selected_dataset
        annotated = self._annotated_dataset
        return {
            "selected_summary": self._dataset_summary(selected) if selected is not None else i18n.t("Selected Data: -"),
            "selected_headers": list(selected.dataframe.columns) if selected is not None else [],
            "selected_rows": self._preview_rows(selected) if selected is not None else [],
            "data_summary": self._dataset_summary(annotated) if annotated is not None else i18n.t("Data: -"),
            "data_headers": list(annotated.dataframe.columns) if annotated is not None else [],
            "data_rows": self._preview_rows(annotated) if annotated is not None else [],
        }

    def serialize_node_state(self) -> dict[str, object]:
        return {
            "rowwise": self._rows_radio.isChecked(),
            "match_code": self._current_match_code(),
            "selection": sorted(self._selected_areas),
            "output_duplicates": self._duplicates_checkbox.isChecked(),
            "auto_apply": self.cb_apply_auto.isChecked(),
            "title_overrides": dict(self._title_overrides),
        }

    def restore_node_state(self, payload: dict[str, object]) -> None:
        self._rows_radio.setChecked(bool(payload.get("rowwise", True)))
        self._columns_radio.setChecked(not self._rows_radio.isChecked())
        self._duplicates_checkbox.setChecked(bool(payload.get("output_duplicates", False)))
        self.cb_apply_auto.setChecked(bool(payload.get("auto_apply", True)))
        overrides = payload.get("title_overrides", {})
        if isinstance(overrides, dict):
            self._title_overrides = {str(key): str(value) for key, value in overrides.items() if str(key) in INPUT_CHANNELS}
        selection = payload.get("selection", [])
        if isinstance(selection, list):
            self._selected_areas = {int(item) for item in selection if isinstance(item, int)}
        self._pending_match_code = str(payload.get("match_code", IDENTITY_TOKEN))
        self._rebuild_timer.start()

    def _first_available_slot(self) -> str | None:
        for slot in INPUT_CHANNELS:
            if slot not in self._inputs:
                return slot
        return None

    def _ordered_slots(self) -> list[str]:
        return _sorted_input_slots(self._inputs)

    def _on_title_edited(self, slot: str, text: str) -> None:
        dataset = self._inputs.get(slot)
        if dataset is None:
            self._title_overrides.pop(slot, None)
            return
        cleaned = text.strip()
        if not cleaned or cleaned == dataset.display_name:
            self._title_overrides.pop(slot, None)
        else:
            self._title_overrides[slot] = cleaned
        self._rebuild()

    def _on_configuration_changed(self) -> None:
        self._pending_match_code = self._current_match_code()
        self._sync_configuration_controls()
        self._rebuild()

    def _on_selection_changed(self, selection: list[int]) -> None:
        self._selected_areas = set(selection)
        self._apply_or_defer()

    def _sync_configuration_controls(self) -> None:
        rowwise = self._rows_radio.isChecked()
        self._match_combo.setEnabled(rowwise)
        self._duplicates_checkbox.setEnabled(rowwise and self._current_match_code() not in {IDENTITY_TOKEN, EQUALITY_TOKEN})

    def _rebuild(self) -> None:
        self._populate_match_options()
        self._sync_title_inputs()
        self._sync_configuration_controls()
        self._build_itemsets_and_diagram()
        self._apply_or_defer()

    def _populate_match_options(self) -> None:
        model = QStandardItemModel(self._match_combo)

        identity_item = QStandardItem(i18n.t("Instance identity"))
        identity_item.setData(IDENTITY_TOKEN, Qt.ItemDataRole.UserRole)
        model.appendRow(identity_item)

        equality_item = QStandardItem(i18n.t("Instance equality"))
        equality_item.setData(EQUALITY_TOKEN, Qt.ItemDataRole.UserRole)
        equality_item.setEnabled(self._same_domains())
        model.appendRow(equality_item)

        for column_name in self._common_match_columns():
            item = QStandardItem(column_name)
            item.setData(column_name, Qt.ItemDataRole.UserRole)
            model.appendRow(item)

        previous = self._pending_match_code or IDENTITY_TOKEN
        self._match_combo.blockSignals(True)
        self._match_combo.setModel(model)
        selected_index = 0
        for index in range(model.rowCount()):
            item = model.item(index)
            if item.data(Qt.ItemDataRole.UserRole) == previous and item.isEnabled():
                selected_index = index
                break
        self._match_combo.setCurrentIndex(selected_index)
        self._match_combo.blockSignals(False)
        self._pending_match_code = self._current_match_code()

    def _sync_title_inputs(self) -> None:
        for slot in INPUT_CHANNELS:
            editor = self._title_inputs[slot]
            count_label = self._title_count_labels[slot]
            dataset = self._inputs.get(slot)
            editor.blockSignals(True)
            if dataset is None:
                editor.setEnabled(False)
                editor.setText("")
                count_label.setText("-")
            else:
                editor.setEnabled(True)
                editor.setText(self._title_overrides.get(slot, dataset.display_name))
                count_label.setText(i18n.tf("{rows} rows x {cols} cols", rows=dataset.row_count, cols=dataset.column_count))
            editor.blockSignals(False)

    def _build_itemsets_and_diagram(self) -> None:
        ordered_slots = self._ordered_slots()
        if not ordered_slots:
            self._itemsets = []
            self._disjoint_items = []
            self._area_slots = []
            self._diagram.clear()
            self._status_label.setText(i18n.t("Connect up to 5 datasets to compare overlaps."))
            return

        self._itemsets = []
        for slot in ordered_slots[:MAX_INPUTS]:
            dataset = self._inputs[slot]
            items = self._items_for_input(slot)
            title = self._title_overrides.get(slot, dataset.display_name)
            self._itemsets.append(
                _InputSet(
                    slot=slot,
                    dataset=dataset,
                    title=title,
                    items=items,
                    unique_count=len(set(items)),
                    total_count=len(items),
                )
            )

        item_sets = [set(itemset.items) for itemset in self._itemsets]
        self._disjoint_items, self._area_slots = self._get_disjoint(item_sets, ordered_slots)
        valid_areas = set(range(1, 2 ** len(self._itemsets)))
        self._selected_areas &= valid_areas

        titles = [itemset.title for itemset in self._itemsets]
        counts = [
            str(itemset.unique_count)
            if itemset.unique_count == itemset.total_count
            else f"{itemset.unique_count} (all: {itemset.total_count})"
            for itemset in self._itemsets
        ]
        area_counts: dict[int, int] = {}
        area_tooltips: dict[int, str] = {}
        show_items = self._rows_radio.isChecked() and self._current_match_code() not in {IDENTITY_TOKEN, EQUALITY_TOKEN}
        if self._columns_radio.isChecked():
            show_items = True

        for index in range(1, 2 ** len(self._itemsets)):
            items = self._disjoint_items[index]
            area_counts[index] = len(items)
            label = _disjoint_set_label(index, len(self._itemsets))
            tooltip_lines = [f"<b>|{label}| = {len(items)}</b>"]
            if show_items and items:
                preview = ", ".join(str(item) for item in list(items)[:32])
                tooltip_lines.append(preview)
                if len(items) > 32:
                    tooltip_lines.append(i18n.tf("... +{count} more", count=len(items) - 32))
            area_tooltips[index] = "<br>".join(tooltip_lines)

        self._diagram.set_content(titles, counts, area_counts, area_tooltips, self._selected_areas)
        self._status_label.setText(self._build_status_message())

    def _build_status_message(self) -> str:
        if not self._itemsets:
            return i18n.t("Connect up to 5 datasets to compare overlaps.")
        if self._columns_radio.isChecked() and not self._column_mode_compatible():
            return i18n.t("Data sets do not contain the same instances.")
        mode = i18n.t("Rows") if self._rows_radio.isChecked() else i18n.t("Columns")
        match_code = self._current_match_code()
        match_text = i18n.t("Instance identity") if match_code == IDENTITY_TOKEN else (
            i18n.t("Instance equality") if match_code == EQUALITY_TOKEN else str(match_code)
        )
        selected_items = sum(len(self._disjoint_items[index]) for index in self._selected_areas if index < len(self._disjoint_items))
        return i18n.tf(
            "{mode} mode | {inputs} inputs | matched by {match} | {selected} selected items",
            mode=mode,
            inputs=len(self._itemsets),
            match=match_text,
            selected=selected_items,
        )

    def _default_export_path(self, dataset: DatasetHandle) -> Path:
        source_path = dataset.source.path
        suffix_by_format = {
            "csv": ".csv",
            "xlsx": ".xlsx",
            "parquet": ".parquet",
        }
        suffix = suffix_by_format.get(dataset.source.format, source_path.suffix.lower() or ".csv")
        return source_path.with_name(f"{source_path.stem}_copy{suffix}")

    def _write_export_dataset(self, dataset: DatasetHandle, target_path: Path) -> None:
        try:
            if target_path.resolve() == dataset.source.path.resolve():
                QMessageBox.information(self, i18n.t("Save Data"), i18n.t("Choose a different output path."))
                return
        except OSError:
            pass

        try:
            self._save_data_service.save(dataset, str(target_path))
        except UnsupportedFormatError as exc:
            QMessageBox.warning(self, i18n.t("Save Data"), str(exc))
            return
        except DatasetSaveError as exc:
            QMessageBox.warning(self, i18n.t("Save Data"), str(exc))
            return

        QMessageBox.information(self, i18n.t("Save Data"), i18n.tf("Dataset saved to:\n{path}", path=target_path))

    def _apply_or_defer(self) -> None:
        self._status_label.setText(self._build_status_message())
        if self.cb_apply_auto.isChecked():
            self._apply()
        else:
            self._notify_output_changed()

    def _apply(self) -> None:
        if not self._itemsets or (self._columns_radio.isChecked() and not self._column_mode_compatible()):
            self._selected_dataset = None
            self._annotated_dataset = None
            self._notify_output_changed()
            return

        self._selected_dataset = self._build_selected_output()
        self._annotated_dataset = self._build_annotated_output()
        self._notify_output_changed()

    def _build_selected_output(self) -> DatasetHandle | None:
        if not self._selected_areas:
            return None
        selected_items = self._selected_items()
        if not selected_items:
            return None

        if self._rows_radio.isChecked():
            relevant_slots = self._selected_slots()
            index_maps = self._row_index_maps(relevant_slots, include_duplicates=self._duplicates_checkbox.isChecked())
            if self._duplicates_checkbox.isChecked() and self._current_match_code() not in {IDENTITY_TOKEN, EQUALITY_TOKEN}:
                return self._build_rowwise_duplicate_dataset(index_maps, selected_items)
            return self._build_rowwise_dataset(index_maps, selected_items, annotate_selection=False)

        return self._build_columnwise_dataset(self._selected_slots(), selected_features=selected_items, annotated=False)

    def _build_annotated_output(self) -> DatasetHandle | None:
        if self._rows_radio.isChecked():
            index_maps = self._row_index_maps(self._ordered_slots(), include_duplicates=False)
            return self._build_rowwise_dataset(index_maps, self._selected_items(), annotate_selection=True)
        return self._build_columnwise_dataset(self._ordered_slots(), selected_features=self._selected_items(), annotated=True)

    def _build_rowwise_dataset(
        self,
        index_maps: dict[str, dict[Any, Any]],
        selected_items: set[Any],
        *,
        annotate_selection: bool,
    ) -> DatasetHandle | None:
        if not index_maps:
            return None

        ordered_slots = [slot for slot in self._ordered_slots() if slot in index_maps]
        ordered_keys = _ordered_union(index_maps, ordered_slots)
        if not annotate_selection:
            ordered_keys = [key for key in ordered_keys if key in selected_items]
        if not ordered_keys:
            return None

        columns = self._merge_rowwise_columns(index_maps, ordered_slots, ordered_keys, duplicate_same_named=False)
        if not columns:
            return None

        data = {column.name: values for column, values in columns}
        role_overrides = {column.name: column.role for column, _values in columns}
        annotations = {
            "row_identity_keys": ordered_keys,
            "venn_mode": "rows",
            "venn_match_code": self._current_match_code(),
        }
        if annotate_selection:
            data[SELECTED_COLUMN_NAME] = [key in selected_items for key in ordered_keys]
            role_overrides[SELECTED_COLUMN_NAME] = "meta"
            annotations["selected_items"] = list(selected_items)

        dataframe = pl.DataFrame(data)
        dataset_id = f"venn-{self._screen_token}-annotated" if annotate_selection else f"venn-{self._screen_token}-selected"
        display_name = i18n.t("Annotated Data") if annotate_selection else i18n.t("Selected Data")
        return self._generated_datasets.build_dataset(
            dataframe,
            dataset_id=dataset_id,
            display_name=display_name,
            file_name=f"{dataset_id}.csv",
            role_overrides=role_overrides,
            annotations=annotations,
        )

    def _build_rowwise_duplicate_dataset(
        self,
        index_maps: dict[str, dict[Any, Any]],
        selected_items: set[Any],
    ) -> DatasetHandle | None:
        ordered_slots = [slot for slot in self._ordered_slots() if slot in index_maps]
        ordered_keys = [key for key in _ordered_union(index_maps, ordered_slots) if key in selected_items]
        if not ordered_keys:
            return None

        output_columns = self._duplicate_rowwise_column_layout(ordered_slots)
        data = {column.name: [] for column in output_columns}
        row_identity_keys: list[Any] = []

        for key in ordered_keys:
            for slot in ordered_slots:
                slot_map = index_maps.get(slot, {})
                if key not in slot_map:
                    continue
                indices = slot_map[key] if isinstance(slot_map[key], list) else [slot_map[key]]
                dataset = self._inputs[slot]
                for index in indices:
                    row_values = dataset.dataframe.row(index, named=True)
                    for column in output_columns:
                        data[column.name].append(row_values.get(column.source_name) if column.source_slot == slot else None)
                    row_identity_keys.append((slot, key, index))

        if not row_identity_keys:
            return None

        dataframe = pl.DataFrame(data)
        role_overrides = {column.name: column.role for column in output_columns}
        return self._generated_datasets.build_dataset(
            dataframe,
            dataset_id=f"venn-{self._screen_token}-selected-duplicates",
            display_name=i18n.t("Selected Data"),
            file_name=f"venn-{self._screen_token}-selected-duplicates.csv",
            role_overrides=role_overrides,
            annotations={
                "row_identity_keys": row_identity_keys,
                "venn_mode": "rows",
                "venn_match_code": self._current_match_code(),
                "duplicates": True,
            },
        )

    def _build_columnwise_dataset(
        self,
        slots: list[str],
        *,
        selected_features: set[Any],
        annotated: bool,
    ) -> DatasetHandle | None:
        if not slots:
            return None

        columns = self._merge_columnwise_columns(slots, None if annotated else selected_features)
        if not columns:
            return None

        dataframe = pl.DataFrame({column.name: values for column, values in columns})
        role_overrides = {column.name: column.role for column, _values in columns}
        dataset_id = f"venn-{self._screen_token}-annotated-columns" if annotated else f"venn-{self._screen_token}-selected-columns"
        display_name = i18n.t("Annotated Data") if annotated else i18n.t("Selected Data")
        return self._generated_datasets.build_dataset(
            dataframe,
            dataset_id=dataset_id,
            display_name=display_name,
            file_name=f"{dataset_id}.csv",
            role_overrides=role_overrides,
            annotations={
                "venn_mode": "columns",
                "selected_features": [str(item) for item in sorted(selected_features, key=str)],
            },
        )

    def _merge_rowwise_columns(
        self,
        index_maps: dict[str, dict[Any, Any]],
        ordered_slots: list[str],
        ordered_keys: list[Any],
        *,
        duplicate_same_named: bool,
    ) -> list[tuple[_OutputColumn, list[Any]]]:
        key_positions = {key: position for position, key in enumerate(ordered_keys)}
        merged: list[dict[str, Any]] = []

        for slot_position, slot in enumerate(ordered_slots, start=1):
            dataset = self._inputs[slot]
            row_map = index_maps[slot]
            role_columns = self._dataset_columns_by_role(dataset)
            for role in ("feature", "target", "meta"):
                for source_name in role_columns[role]:
                    values_by_key = {
                        key: dataset.dataframe[source_name][index]
                        for key, index in row_map.items()
                        if not isinstance(index, list) and key in key_positions
                    }
                    candidate = None if duplicate_same_named else self._find_merge_candidate(
                        merged,
                        role,
                        source_name,
                        values_by_key,
                        key_positions,
                    )
                    if candidate is None:
                        output_name = source_name if not any(col["output"].name == source_name for col in merged) else f"{source_name} ({slot_position})"
                        values = [None] * len(ordered_keys)
                        for key, value in values_by_key.items():
                            values[key_positions[key]] = value
                        merged.append({"output": _OutputColumn(output_name, role, slot, source_name), "values": values})
                    else:
                        for key, value in values_by_key.items():
                            position = key_positions[key]
                            if candidate["values"][position] is None:
                                candidate["values"][position] = value

        return [(entry["output"], entry["values"]) for entry in merged]

    def _duplicate_rowwise_column_layout(self, ordered_slots: list[str]) -> list[_OutputColumn]:
        output_columns: list[_OutputColumn] = []
        for slot_position, slot in enumerate(ordered_slots, start=1):
            dataset = self._inputs[slot]
            role_columns = self._dataset_columns_by_role(dataset)
            for role in ("feature", "target", "meta"):
                for source_name in role_columns[role]:
                    output_name = source_name if len(ordered_slots) == 1 else f"{source_name} ({slot_position})"
                    output_columns.append(_OutputColumn(output_name, role, slot, source_name))
        return output_columns

    def _merge_columnwise_columns(
        self,
        slots: list[str],
        selected_features: set[Any] | None,
    ) -> list[tuple[_OutputColumn, list[Any]]]:
        merged: list[dict[str, Any]] = []
        for slot_position, slot in enumerate(slots, start=1):
            dataset = self._inputs[slot]
            role_columns = self._dataset_columns_by_role(dataset)
            for role in ("feature", "target", "meta"):
                for source_name in role_columns[role]:
                    if selected_features is not None and role == "feature" and source_name not in selected_features:
                        continue
                    values = dataset.dataframe[source_name].to_list()
                    candidate = self._find_column_candidate(merged, role, source_name, values)
                    if candidate is None:
                        output_name = source_name if not any(col["output"].name == source_name for col in merged) else f"{source_name} ({slot_position})"
                        merged.append({"output": _OutputColumn(output_name, role, slot, source_name), "values": list(values)})
        return [(entry["output"], entry["values"]) for entry in merged]

    def _find_merge_candidate(
        self,
        merged: list[dict[str, Any]],
        role: str,
        source_name: str,
        values_by_key: dict[Any, Any],
        key_positions: dict[Any, int],
    ) -> dict[str, Any] | None:
        for entry in merged:
            output: _OutputColumn = entry["output"]
            if output.role != role or output.source_name != source_name:
                continue
            compatible = True
            for key, value in values_by_key.items():
                existing = entry["values"][key_positions[key]]
                if existing is not None and not _values_equal(existing, value):
                    compatible = False
                    break
            if compatible:
                return entry
        return None

    def _find_column_candidate(
        self,
        merged: list[dict[str, Any]],
        role: str,
        source_name: str,
        values: list[Any],
    ) -> dict[str, Any] | None:
        for entry in merged:
            output: _OutputColumn = entry["output"]
            if output.role != role or output.source_name != source_name:
                continue
            if len(entry["values"]) != len(values):
                continue
            if all(_values_equal(left, right) for left, right in zip(entry["values"], values, strict=False)):
                return entry
        return None

    def _dataset_columns_by_role(self, dataset: DatasetHandle) -> dict[str, list[str]]:
        result = {"feature": [], "target": [], "meta": []}
        dataframe_columns = set(dataset.dataframe.columns)
        for column in dataset.domain.columns:
            if column.name not in dataframe_columns:
                continue
            if column.role not in result:
                result["feature"].append(column.name)
            else:
                result[column.role].append(column.name)
        extras = [name for name in dataset.dataframe.columns if name not in set().union(*result.values())]
        result["feature"].extend(extras)
        return result

    def _row_index_maps(self, slots: list[str], *, include_duplicates: bool) -> dict[str, dict[Any, Any]]:
        match_code = self._current_match_code()
        output: dict[str, dict[Any, Any]] = {}
        for slot in slots:
            dataset = self._inputs[slot]
            if match_code == IDENTITY_TOKEN:
                keys = self._row_identity_keys(dataset)
            elif match_code == EQUALITY_TOKEN:
                keys = self._row_equality_keys(dataset)
            else:
                keys = self._row_feature_keys(dataset, str(match_code))

            mapping: dict[Any, Any] = {}
            if include_duplicates and match_code not in {IDENTITY_TOKEN, EQUALITY_TOKEN}:
                temp: dict[Any, list[int]] = defaultdict(list)
                for index, key in enumerate(keys):
                    if key is _MISSING:
                        continue
                    temp[key].append(index)
                mapping = dict(temp)
            else:
                for index, key in enumerate(keys):
                    if key is _MISSING or key in mapping:
                        continue
                    mapping[key] = index
            output[slot] = mapping
        return output

    def _row_identity_keys(self, dataset: DatasetHandle) -> list[Any]:
        explicit = dataset.annotations.get("row_identity_keys")
        if isinstance(explicit, (list, tuple)) and len(explicit) == dataset.row_count:
            return list(explicit)

        source_indices = dataset.annotations.get("source_row_indices")
        if isinstance(source_indices, (list, tuple)) and len(source_indices) == dataset.row_count:
            source_token = str(dataset.source.path)
            return [(source_token, int(index)) for index in source_indices]

        source_token = str(dataset.source.path)
        seen: Counter[Any] = Counter()
        keys: list[Any] = []
        for row in dataset.dataframe.iter_rows(named=False):
            fingerprint = tuple(_normalize_value(value) for value in row)
            occurrence = seen[fingerprint]
            seen[fingerprint] += 1
            keys.append((source_token, fingerprint, occurrence))
        return keys

    def _row_equality_keys(self, dataset: DatasetHandle) -> list[Any]:
        return [tuple(_normalize_value(value) for value in row) for row in dataset.dataframe.iter_rows(named=False)]

    def _row_feature_keys(self, dataset: DatasetHandle, column_name: str) -> list[Any]:
        if column_name not in dataset.dataframe.columns:
            return [_MISSING] * dataset.row_count
        keys: list[Any] = []
        for value in dataset.dataframe[column_name].to_list():
            if _is_missing(value):
                keys.append(_MISSING)
            else:
                keys.append(str(value))
        return keys

    def _column_alignment_keys(self, dataset: DatasetHandle) -> list[Any]:
        explicit = dataset.annotations.get("row_identity_keys")
        if isinstance(explicit, (list, tuple)) and len(explicit) == dataset.row_count:
            return list(explicit)

        source_indices = dataset.annotations.get("source_row_indices")
        if isinstance(source_indices, (list, tuple)) and len(source_indices) == dataset.row_count:
            source_token = str(dataset.source.path)
            return [(source_token, int(index)) for index in source_indices]

        source_token = str(dataset.source.path)
        return [(source_token, index) for index in range(dataset.row_count)]

    def _items_for_input(self, slot: str) -> list[Any]:
        dataset = self._inputs[slot]
        if self._columns_radio.isChecked():
            return list(self._dataset_columns_by_role(dataset)["feature"])
        match_code = self._current_match_code()
        if match_code == IDENTITY_TOKEN:
            return self._row_identity_keys(dataset)
        if match_code == EQUALITY_TOKEN:
            return self._row_equality_keys(dataset)
        return [key for key in self._row_feature_keys(dataset, str(match_code)) if key is not _MISSING]

    def _same_domains(self) -> bool:
        ordered_slots = self._ordered_slots()
        if len(ordered_slots) < 2:
            return True
        first = self._inputs[ordered_slots[0]]
        baseline = [(column.name, column.dtype_repr, column.role) for column in first.domain.columns]
        for slot in ordered_slots[1:]:
            other = [(column.name, column.dtype_repr, column.role) for column in self._inputs[slot].domain.columns]
            if other != baseline:
                return False
        return True

    def _column_mode_compatible(self) -> bool:
        ordered_slots = self._ordered_slots()
        if len(ordered_slots) < 2:
            return True
        baseline = self._column_alignment_keys(self._inputs[ordered_slots[0]])
        for slot in ordered_slots[1:]:
            if self._column_alignment_keys(self._inputs[slot]) != baseline:
                return False
        return True

    def _common_match_columns(self) -> list[str]:
        ordered_slots = self._ordered_slots()
        if not ordered_slots:
            return []
        candidate_sets = []
        for slot in ordered_slots:
            dataset = self._inputs[slot]
            columns = {column.name for column in dataset.domain.columns if _string_like_column(dataset, column.name)}
            candidate_sets.append(columns)
        common = set.intersection(*candidate_sets) if candidate_sets else set()
        first_dataset = self._inputs[ordered_slots[0]]
        return [column.name for column in first_dataset.domain.columns if column.name in common]

    def _current_match_code(self) -> str:
        if self._match_combo.count() == 0:
            return IDENTITY_TOKEN
        data = self._match_combo.currentData(Qt.ItemDataRole.UserRole)
        return str(data) if data is not None else IDENTITY_TOKEN

    def _selected_items(self) -> set[Any]:
        selected: set[Any] = set()
        for index in self._selected_areas:
            if 0 <= index < len(self._disjoint_items):
                selected.update(self._disjoint_items[index])
        return selected

    def _selected_slots(self) -> list[str]:
        slots: list[str] = []
        seen: set[str] = set()
        for index in sorted(self._selected_areas):
            if index >= len(self._area_slots):
                continue
            for slot in self._area_slots[index]:
                if slot not in seen:
                    seen.add(slot)
                    slots.append(slot)
        return slots

    def _get_disjoint(self, sets: list[set[Any]], ordered_slots: list[str]) -> tuple[list[set[Any]], list[list[str]]]:
        disjoint_sets: list[set[Any]] = [set() for _ in range(2 ** len(sets))]
        included_slots: list[list[str]] = [[] for _ in range(2 ** len(sets))]
        for index in range(2 ** len(sets)):
            key = _set_key(index, len(sets))
            included = [item_set for item_set, include in zip(sets, key) if include]
            if included:
                excluded = [item_set for item_set, include in zip(sets, key) if not include]
                current = reduce(set.intersection, included)
                for excluded_set in excluded:
                    current = current.difference(excluded_set)
            else:
                current = set()
            disjoint_sets[index] = current
            included_slots[index] = [slot for slot, include in zip(ordered_slots, key) if include]
        return disjoint_sets, included_slots

    def _dataset_summary(self, dataset: DatasetHandle) -> str:
        return i18n.tf("{name}: {rows} rows x {cols} cols", name=dataset.display_name, rows=dataset.row_count, cols=dataset.column_count)

    def _preview_rows(self, dataset: DatasetHandle | None, limit: int = 200) -> list[list[str]]:
        if dataset is None:
            return []
        rows: list[list[str]] = []
        for row in dataset.dataframe.head(limit).iter_rows(named=False):
            rows.append([_display_value(value) for value in row])
        return rows
