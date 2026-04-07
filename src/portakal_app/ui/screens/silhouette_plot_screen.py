from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
import polars as pl

from PySide6.QtCore import QPoint, QRectF, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QFontMetrics, QPainter, QPen
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QToolTip,
    QVBoxLayout,
    QWidget,
)

from portakal_app.data.errors import DatasetSaveError, UnsupportedFormatError
from portakal_app.data.models import ColumnSchema, DatasetHandle
from portakal_app.data.services.generated_dataset_service import GeneratedDatasetService
from portakal_app.data.services.save_data_service import SaveDataService
from portakal_app.models import WorkflowPayload
from portakal_app.ui import i18n
from portakal_app.ui.screens.node_screen import WorkflowNodeScreenSupport


OUTPUT_CHANNELS = ("Selected Data", "Annotated Data")
SELECTED_COLUMN_NAME = "Selected"
_UNGROUPED_LABEL = "__ungrouped__"
_UNGROUPED_COLOR = QColor(63, 207, 207)
_PALETTE = (
    QColor("#4e79a7"),
    QColor("#f28e2b"),
    QColor("#e15759"),
    QColor("#76b7b2"),
    QColor("#59a14f"),
    QColor("#edc948"),
    QColor("#b07aa1"),
    QColor("#ff9da7"),
    QColor("#9c755f"),
    QColor("#bab0ab"),
)


@dataclass(frozen=True)
class _Metric:
    code: str
    label: str


@dataclass(frozen=True)
class _PlotRow:
    source_index: int
    score: float
    cluster_label: str
    color: QColor
    annotation: str


@dataclass(frozen=True)
class _PlotGroup:
    label: str
    color: QColor
    rows: list[_PlotRow]


def _is_missing(value: Any) -> bool:
    return value is None or (isinstance(value, float) and np.isnan(value))


def _display_value(value: Any) -> str:
    return "" if _is_missing(value) else str(value)


def _preview_rows(dataset: DatasetHandle | None, limit: int = 200) -> list[list[str]]:
    if dataset is None:
        return []
    rows: list[list[str]] = []
    for row in dataset.dataframe.head(limit).iter_rows(named=False):
        rows.append([_display_value(value) for value in row])
    return rows


def _dataset_summary(dataset: DatasetHandle | None) -> str:
    if dataset is None:
        return i18n.t("No data")
    return i18n.tf("{name}: {rows} rows x {cols} cols", name=dataset.display_name, rows=dataset.row_count, cols=dataset.column_count)


def _unique_name(existing: list[str], proposed: str) -> str:
    if proposed not in existing:
        return proposed
    index = 1
    while f"{proposed} ({index})" in existing:
        index += 1
    return f"{proposed} ({index})"


def _role_overrides(dataset: DatasetHandle) -> dict[str, str]:
    return {column.name: column.role for column in dataset.domain.columns}


def _series_to_float_array(series: pl.Series) -> np.ndarray:
    values = series.cast(pl.Float64, strict=False).to_list()
    return np.asarray([float(value) if value is not None else np.nan for value in values], dtype=float)


def _series_to_object_array(series: pl.Series) -> np.ndarray:
    return np.asarray([None if value is None else value for value in series.to_list()], dtype=object)


def _label_values(series: pl.Series) -> list[Any]:
    values: list[Any] = []
    for value in series.to_list():
        if _is_missing(value):
            continue
        if value not in values:
            values.append(value)
    return values


def _encode_labels(series: pl.Series) -> tuple[np.ndarray, np.ndarray, list[str]]:
    values = _label_values(series)
    mapping = {value: index for index, value in enumerate(values)}
    labels = np.full(series.len(), -1, dtype=int)
    missing = np.zeros(series.len(), dtype=bool)
    for index, value in enumerate(series.to_list()):
        if _is_missing(value):
            missing[index] = True
        else:
            labels[index] = mapping[value]
    return labels, missing, [_display_value(value) for value in values]


def _cluster_candidates(dataset: DatasetHandle) -> list[ColumnSchema]:
    candidates: list[ColumnSchema] = []
    for column in dataset.domain.columns:
        if column.logical_type not in {"categorical", "boolean"}:
            continue
        if len(_label_values(dataset.dataframe.get_column(column.name))) >= 2:
            candidates.append(column)
    return candidates


def _annotation_candidates(dataset: DatasetHandle) -> list[ColumnSchema]:
    return [
        column
        for column in dataset.domain.columns
        if column.logical_type in {"categorical", "boolean", "text"}
    ]


def _feature_columns(dataset: DatasetHandle) -> tuple[list[ColumnSchema], list[ColumnSchema], list[ColumnSchema]]:
    numeric: list[ColumnSchema] = []
    discrete: list[ColumnSchema] = []
    ignored: list[ColumnSchema] = []
    for column in dataset.domain.feature_columns:
        if column.logical_type == "numeric":
            numeric.append(column)
        elif column.logical_type in {"categorical", "boolean"}:
            discrete.append(column)
        else:
            ignored.append(column)
    return numeric, discrete, ignored


def _prepare_numeric_matrix(dataset: DatasetHandle, columns: list[ColumnSchema], *, use_median: bool) -> np.ndarray:
    if not columns:
        return np.empty((dataset.row_count, 0), dtype=float)
    arrays: list[np.ndarray] = []
    for column in columns:
        values = _series_to_float_array(dataset.dataframe.get_column(column.name))
        if np.isnan(values).all():
            fill = 0.0
        else:
            fill = float(np.nanmedian(values)) if use_median else float(np.nanmean(values))
        arrays.append(np.nan_to_num(values, nan=fill))
    return np.column_stack(arrays)


def _prepare_discrete_matrix(dataset: DatasetHandle, columns: list[ColumnSchema]) -> np.ndarray:
    if not columns:
        return np.empty((dataset.row_count, 0), dtype=object)
    arrays = [_series_to_object_array(dataset.dataframe.get_column(column.name)) for column in columns]
    return np.column_stack(arrays)


def _euclidean_distances(numeric: np.ndarray, discrete: np.ndarray) -> np.ndarray:
    n_rows = numeric.shape[0] if numeric.size else discrete.shape[0]
    distances = np.zeros((n_rows, n_rows), dtype=float)
    if numeric.size:
        squared = np.sum(numeric * numeric, axis=1, keepdims=True)
        distances += squared - 2.0 * (numeric @ numeric.T) + squared.T
        np.maximum(distances, 0.0, out=distances)
    if discrete.size:
        for column_index in range(discrete.shape[1]):
            values = discrete[:, column_index]
            left = values[:, None]
            right = values[None, :]
            both_missing = np.equal(left, None) & np.equal(right, None)
            different = np.not_equal(left, right)
            different[both_missing] = False
            distances += different.astype(float)
    np.maximum(distances, 0.0, out=distances)
    return np.sqrt(distances)


def _manhattan_distances(numeric: np.ndarray, discrete: np.ndarray) -> np.ndarray:
    n_rows = numeric.shape[0] if numeric.size else discrete.shape[0]
    distances = np.zeros((n_rows, n_rows), dtype=float)
    if numeric.size:
        for column_index in range(numeric.shape[1]):
            values = numeric[:, column_index]
            distances += np.abs(values[:, None] - values[None, :])
    if discrete.size:
        for column_index in range(discrete.shape[1]):
            values = discrete[:, column_index]
            left = values[:, None]
            right = values[None, :]
            both_missing = np.equal(left, None) & np.equal(right, None)
            different = np.not_equal(left, right)
            different[both_missing] = False
            distances += different.astype(float)
    return distances


def _cosine_distances(numeric: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n_rows = numeric.shape[0]
    if numeric.size == 0:
        matrix = np.full((n_rows, n_rows), np.nan, dtype=float)
        return matrix, np.ones(n_rows, dtype=bool)

    norms = np.linalg.norm(numeric, axis=1)
    zero_rows = norms <= 0
    scaled = np.zeros_like(numeric)
    nonzero = ~zero_rows
    if np.any(nonzero):
        scaled[nonzero] = numeric[nonzero] / norms[nonzero, None]
    similarity = scaled @ scaled.T
    np.clip(similarity, -1.0, 1.0, out=similarity)
    distances = 1.0 - similarity
    np.fill_diagonal(distances, 0.0)
    if np.any(zero_rows):
        distances[zero_rows, :] = np.nan
        distances[:, zero_rows] = np.nan
    return distances, zero_rows


def _silhouette_scores(distances: np.ndarray, labels: np.ndarray) -> np.ndarray:
    n_rows = len(labels)
    scores = np.zeros(n_rows, dtype=float)
    unique_labels = sorted(set(int(value) for value in labels.tolist()))
    members = {label: np.flatnonzero(labels == label) for label in unique_labels}

    for row_index in range(n_rows):
        cluster_index = int(labels[row_index])
        cluster_members = members[cluster_index]
        if len(cluster_members) <= 1:
            scores[row_index] = 0.0
            continue

        same_cluster = cluster_members[cluster_members != row_index]
        a_value = float(np.nanmean(distances[row_index, same_cluster])) if len(same_cluster) else 0.0

        b_value = np.inf
        for other_label, other_members in members.items():
            if other_label == cluster_index or len(other_members) == 0:
                continue
            mean_distance = float(np.nanmean(distances[row_index, other_members]))
            if np.isnan(mean_distance):
                continue
            b_value = min(b_value, mean_distance)

        denominator = max(a_value, b_value)
        scores[row_index] = 0.0 if not np.isfinite(denominator) or denominator <= 0 else (b_value - a_value) / denominator

    return scores


class _SilhouetteCanvas(QWidget):
    selectionChanged = Signal(list)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._groups: list[_PlotGroup] = []
        self._bar_height = 3
        self._row_names_visible = False
        self._score_min = -1.0
        self._score_max = 1.0
        self._row_rects: list[tuple[QRectF, _PlotRow]] = []
        self._selected_rows: set[int] = set()
        self._hovered_row: int | None = None
        self._drag_origin: QPoint | None = None
        self._drag_rect = QRectF()
        self._drag_base_selection: set[int] = set()
        self._drag_mode = "replace"
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setMinimumHeight(240)
        self.setMinimumWidth(620)

    def clear(self) -> None:
        self._groups = []
        self._row_rects = []
        self._selected_rows.clear()
        self._hovered_row = None
        self._drag_origin = None
        self._drag_rect = QRectF()
        self._update_height()
        self.update()

    def set_plot(
        self,
        groups: list[_PlotGroup],
        *,
        score_min: float,
        score_max: float,
        selected_rows: set[int] | None = None,
    ) -> None:
        self._groups = groups
        self._row_rects = []
        self._score_min = float(score_min)
        self._score_max = float(score_max)
        available = {row.source_index for group in groups for row in group.rows}
        self._selected_rows = set(selected_rows or set()) & available
        self._hovered_row = None
        self._drag_origin = None
        self._drag_rect = QRectF()
        self._update_height()
        self.update()

    def set_bar_height(self, height: int) -> None:
        self._bar_height = max(1, int(height))
        self._update_height()
        self.update()

    def set_row_names_visible(self, visible: bool) -> None:
        self._row_names_visible = bool(visible)
        self.update()

    def selection(self) -> list[int]:
        return sorted(self._selected_rows)

    def set_selection(self, rows: list[int] | set[int]) -> None:
        available = {row.source_index for group in self._groups for row in group.rows}
        updated = set(int(row) for row in rows) & available
        if updated == self._selected_rows:
            return
        self._selected_rows = updated
        self.selectionChanged.emit(self.selection())
        self.update()

    def _update_height(self) -> None:
        total_rows = sum(len(group.rows) for group in self._groups)
        group_headers = sum(1 for group in self._groups if group.label and group.label != _UNGROUPED_LABEL)
        content_height = 56 + total_rows * (self._bar_height + 1) + group_headers * 22 + max(0, len(self._groups) - 1) * 10
        self.setMinimumHeight(max(240, content_height + 36))
        self.resize(self.width(), self.minimumHeight())

    def _score_to_x(self, value: float, left: float, width: float) -> float:
        span = max(1e-9, self._score_max - self._score_min)
        return left + ((value - self._score_min) / span) * width

    def _row_from_pos(self, pos: QPoint) -> _PlotRow | None:
        for rect, row in self._row_rects:
            if rect.contains(pos):
                return row
        return None

    def _selection_from_rect(self, rect: QRectF) -> set[int]:
        selected: set[int] = set()
        for row_rect, row in self._row_rects:
            if row_rect.intersects(rect):
                selected.add(row.source_index)
        return selected

    def _apply_drag_selection(self, rect: QRectF, finalize: bool) -> None:
        current = self._selection_from_rect(rect)
        base = set(self._drag_base_selection)
        if self._drag_mode == "toggle":
            updated = base.symmetric_difference(current)
        elif self._drag_mode == "add":
            updated = base | current
        elif self._drag_mode == "remove":
            updated = base - current
        else:
            updated = current

        if updated != self._selected_rows:
            self._selected_rows = updated
            self.selectionChanged.emit(self.selection())
        if finalize:
            self._drag_rect = QRectF()
        self.update()

    def mousePressEvent(self, event) -> None:
        if event.button() != Qt.MouseButton.LeftButton:
            return super().mousePressEvent(event)

        modifiers = event.modifiers()
        if modifiers & Qt.KeyboardModifier.ControlModifier:
            self._drag_mode = "toggle"
        elif modifiers & Qt.KeyboardModifier.AltModifier:
            self._drag_mode = "remove"
        elif modifiers & Qt.KeyboardModifier.ShiftModifier:
            self._drag_mode = "add"
        else:
            self._drag_mode = "replace"

        self._drag_origin = event.position().toPoint()
        self._drag_base_selection = set(self._selected_rows if self._drag_mode != "replace" else set())
        self._drag_rect = QRectF(self._drag_origin, self._drag_origin).normalized()

        row = self._row_from_pos(self._drag_origin)
        if row is not None:
            if self._drag_mode == "toggle":
                updated = set(self._selected_rows)
                if row.source_index in updated:
                    updated.remove(row.source_index)
                else:
                    updated.add(row.source_index)
            elif self._drag_mode == "remove":
                updated = set(self._selected_rows)
                updated.discard(row.source_index)
            elif self._drag_mode == "add":
                updated = set(self._selected_rows)
                updated.add(row.source_index)
            else:
                updated = {row.source_index}
            self.set_selection(updated)
        elif self._drag_mode == "replace":
            self.set_selection(set())

        event.accept()

    def mouseMoveEvent(self, event) -> None:
        pos = event.position().toPoint()
        if self._drag_origin is not None and event.buttons() & Qt.MouseButton.LeftButton:
            self._drag_rect = QRectF(self._drag_origin, pos).normalized()
            self._apply_drag_selection(self._drag_rect, finalize=False)
            event.accept()
            return

        row = self._row_from_pos(pos)
        hovered = row.source_index if row is not None else None
        if hovered != self._hovered_row:
            self._hovered_row = hovered
            self.update()
        if row is not None:
            tooltip = f"<b>{row.cluster_label}</b><br>Silhouette: {row.score:.3f}"
            if row.annotation:
                tooltip += f"<br>{row.annotation}"
            QToolTip.showText(event.globalPosition().toPoint(), tooltip, self)
        else:
            QToolTip.hideText()

    def mouseReleaseEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton and self._drag_origin is not None:
            rect = QRectF(self._drag_origin, event.position().toPoint()).normalized()
            self._apply_drag_selection(rect, finalize=True)
            self._drag_origin = None
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def leaveEvent(self, event) -> None:
        self._hovered_row = None
        QToolTip.hideText()
        self.update()
        super().leaveEvent(event)

    def paintEvent(self, _event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), QColor("#fffdf9"))

        if not self._groups:
            painter.setPen(QColor("#8d877d"))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, i18n.t("No silhouette data available."))
            return

        self._row_rects = []

        margin_left = 126
        margin_right = 18
        annotation_width = 210 if self._row_names_visible else 0
        plot_left = margin_left
        plot_right = max(plot_left + 120, self.width() - margin_right - annotation_width)
        plot_width = max(180.0, plot_right - plot_left)
        zero_x = self._score_to_x(0.0, plot_left, plot_width)

        painter.setPen(QPen(QColor("#bbb3a8"), 1))
        painter.drawLine(int(plot_left), 26, int(plot_right), 26)
        painter.drawLine(int(plot_left), self.height() - 26, int(plot_right), self.height() - 26)
        painter.setPen(QPen(QColor("#c9c1b7"), 1, Qt.PenStyle.DashLine))
        painter.drawLine(int(zero_x), 26, int(zero_x), self.height() - 26)

        tick_values = [self._score_min, min(0.0, self._score_max), self._score_max]
        tick_values = sorted(set(round(value, 6) for value in tick_values))
        painter.setPen(QColor("#5f5649"))
        font_metrics = QFontMetrics(painter.font())
        for value in tick_values:
            x = self._score_to_x(value, plot_left, plot_width)
            painter.drawLine(int(x), 22, int(x), 30)
            painter.drawLine(int(x), self.height() - 30, int(x), self.height() - 22)
            label = f"{value:.2f}"
            text_width = font_metrics.horizontalAdvance(label)
            painter.drawText(int(x - text_width / 2), 18, label)
            painter.drawText(int(x - text_width / 2), self.height() - 8, label)

        y = 38
        row_spacing = 1
        for group in self._groups:
            group_label = group.label if group.label != _UNGROUPED_LABEL else ""
            if group_label:
                mean_score = np.mean([row.score for row in group.rows]) if group.rows else float("nan")
                painter.setPen(QColor("#463c2f"))
                painter.drawText(8, y + 14, 112, 18, Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter, f"{group_label} ({mean_score:.3f})")
                y += 20

            for row in group.rows:
                bar_y = y
                value_x = self._score_to_x(row.score, plot_left, plot_width)
                left = min(zero_x, value_x)
                right = max(zero_x, value_x)
                rect = QRectF(left, bar_y, max(1.0, right - left), self._bar_height)

                fill = QColor(row.color)
                fill.setAlpha(190 if row.source_index in self._selected_rows else 150)
                painter.setBrush(fill)
                if row.source_index == self._hovered_row:
                    painter.setPen(QPen(QColor("#666666"), 1))
                elif row.source_index in self._selected_rows:
                    painter.setPen(QPen(QColor("#1f1f1f"), 1))
                else:
                    painter.setPen(Qt.PenStyle.NoPen)
                painter.drawRect(rect)

                if self._row_names_visible and row.annotation:
                    painter.setPen(QColor("#594f43"))
                    painter.drawText(
                        plot_right + 8,
                        int(bar_y - 1),
                        annotation_width - 8,
                        self._bar_height + 4,
                        Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
                        font_metrics.elidedText(row.annotation, Qt.TextElideMode.ElideRight, annotation_width - 16),
                    )

                self._row_rects.append((QRectF(plot_left, bar_y - 1, plot_width, self._bar_height + 2), row))
                y += self._bar_height + row_spacing

            y += 10

        if self._drag_origin is not None and self._drag_rect.isValid():
            painter.setBrush(QColor(78, 121, 167, 40))
            painter.setPen(QPen(QColor(78, 121, 167), 1, Qt.PenStyle.DashLine))
            painter.drawRect(self._drag_rect)

        painter.end()


class SilhouettePlotScreen(QWidget, WorkflowNodeScreenSupport):
    METRICS = (
        _Metric("euclidean", "Euclidean"),
        _Metric("manhattan", "Manhattan"),
        _Metric("cosine", "Cosine"),
    )

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._init_workflow_node_support()
        self._generated_datasets = GeneratedDatasetService()
        self._save_data_service = SaveDataService()
        self._screen_token = uuid4().hex[:8]

        self._dataset: DatasetHandle | None = None
        self._selected_dataset: DatasetHandle | None = None
        self._annotated_dataset: DatasetHandle | None = None

        self._score_column_name = "Silhouette"
        self._full_scores: np.ndarray | None = None
        self._selected_mask: np.ndarray | None = None
        self._valid_mask: np.ndarray | None = None
        self._selected_source_indices: set[int] = set()
        self._cluster_labels: list[str] = []
        self._error_message = ""
        self._warning_messages: list[str] = []
        self._pending_cluster_name: str | None = None
        self._pending_annotation_name: str | None = None
        self._pending_metric_code = self.METRICS[0].code
        self._pending_selection: list[int] = []

        self._refresh_timer = QTimer(self)
        self._refresh_timer.setSingleShot(True)
        self._refresh_timer.setInterval(0)
        self._refresh_timer.timeout.connect(self._recompute_plot)

        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        header = QLabel(i18n.t("Silhouette Plot"))
        header.setProperty("sectionTitle", True)
        layout.addWidget(header)

        description = QLabel(
            i18n.t(
                "Assess cluster quality with silhouette scores, inspect per-instance bars, "
                "and send selected rows or annotated data downstream."
            )
        )
        description.setWordWrap(True)
        description.setProperty("muted", True)
        layout.addWidget(description)

        controls = QGroupBox(i18n.t("Settings"))
        controls_layout = QGridLayout(controls)
        controls_layout.setContentsMargins(8, 8, 8, 8)
        controls_layout.setHorizontalSpacing(12)
        controls_layout.setVerticalSpacing(6)

        controls_layout.addWidget(QLabel(i18n.t("Distance:")), 0, 0)
        self._metric_combo = QComboBox()
        for metric in self.METRICS:
            self._metric_combo.addItem(metric.label, metric.code)
        self._metric_combo.currentIndexChanged.connect(self._schedule_refresh)
        controls_layout.addWidget(self._metric_combo, 0, 1)

        controls_layout.addWidget(QLabel(i18n.t("Cluster variable:")), 0, 2)
        self._cluster_combo = QComboBox()
        self._cluster_combo.currentIndexChanged.connect(self._schedule_refresh)
        controls_layout.addWidget(self._cluster_combo, 0, 3)

        self._group_checkbox = QCheckBox(i18n.t("Show in groups"))
        self._group_checkbox.setChecked(True)
        self._group_checkbox.toggled.connect(self._schedule_refresh)
        controls_layout.addWidget(self._group_checkbox, 1, 0, 1, 2)

        controls_layout.addWidget(QLabel(i18n.t("Bar width:")), 1, 2)
        self._bar_slider = QSlider(Qt.Orientation.Horizontal)
        self._bar_slider.setRange(1, 10)
        self._bar_slider.setValue(3)
        self._bar_slider.valueChanged.connect(self._on_bar_size_changed)
        controls_layout.addWidget(self._bar_slider, 1, 3)

        controls_layout.addWidget(QLabel(i18n.t("Annotations:")), 2, 0)
        self._annotation_combo = QComboBox()
        self._annotation_combo.currentIndexChanged.connect(self._schedule_refresh)
        controls_layout.addWidget(self._annotation_combo, 2, 1)

        self._annotation_warning = QLabel(i18n.t("(increase the width to show)"))
        self._annotation_warning.setProperty("muted", True)
        self._annotation_warning.setVisible(False)
        controls_layout.addWidget(self._annotation_warning, 2, 2, 1, 2)
        layout.addWidget(controls)

        plot_box = QGroupBox(i18n.t("Plot"))
        plot_layout = QVBoxLayout(plot_box)
        plot_layout.setContentsMargins(6, 6, 6, 6)
        self._canvas = _SilhouetteCanvas()
        self._canvas.selectionChanged.connect(self._on_selection_changed)
        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._scroll.setFrameShape(QFrame.Shape.NoFrame)
        self._scroll.setWidget(self._canvas)
        plot_layout.addWidget(self._scroll)
        layout.addWidget(plot_box, 1)

        self._status_label = QLabel(i18n.t("Connect a dataset with feature columns and cluster labels."))
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

    def documentation_url(self) -> str:
        return "https://orangedatamining.com/widget-catalog/visualize/silhouetteplot/"

    def help_text(self) -> str:
        return (
            "Compute silhouette scores from the input data, inspect cluster cohesion "
            "and separation, and output Selected Data or Annotated Data."
        )

    def footer_status_text(self) -> str:
        annotated_rows = self._annotated_dataset.row_count if self._annotated_dataset is not None else 0
        selected_rows = self._selected_dataset.row_count if self._selected_dataset is not None else 0
        if annotated_rows == 0:
            return "0"
        if selected_rows:
            return f"{selected_rows} | {annotated_rows}"
        return str(annotated_rows)

    def set_save_data_service(self, service: SaveDataService) -> None:
        self._save_data_service = service

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

    def set_input_payload(self, payload: WorkflowPayload | None) -> None:
        self._dataset = payload.dataset if payload is not None else None
        self._selected_dataset = None
        self._annotated_dataset = None
        self._selected_source_indices = set()
        self._full_scores = None
        self._selected_mask = None
        self._valid_mask = None
        self._score_column_name = "Silhouette"
        self._populate_column_controls()
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
            "metric_code": self._current_metric_code(),
            "cluster_var": self._current_cluster_name(),
            "annotation_var": self._current_annotation_name(),
            "group_by_cluster": self._group_checkbox.isChecked(),
            "bar_size": self._bar_slider.value(),
            "selection": sorted(self._selected_source_indices),
            "auto_apply": self.cb_apply_auto.isChecked(),
        }

    def restore_node_state(self, payload: dict[str, object]) -> None:
        metric_code = str(payload.get("metric_code", self.METRICS[0].code))
        metric_index = 0
        for index in range(self._metric_combo.count()):
            if self._metric_combo.itemData(index) == metric_code:
                metric_index = index
                break
        self._metric_combo.setCurrentIndex(metric_index)
        self._pending_metric_code = metric_code
        self._pending_cluster_name = str(payload.get("cluster_var")) if payload.get("cluster_var") is not None else None
        self._pending_annotation_name = str(payload.get("annotation_var")) if payload.get("annotation_var") is not None else None
        self._group_checkbox.setChecked(bool(payload.get("group_by_cluster", True)))
        self._bar_slider.setValue(int(payload.get("bar_size", 3)))
        self.cb_apply_auto.setChecked(bool(payload.get("auto_apply", True)))
        selection = payload.get("selection", [])
        if isinstance(selection, list):
            self._pending_selection = [int(value) for value in selection if isinstance(value, int)]
        self._populate_column_controls()
        self._refresh_timer.start()

    def _current_metric_code(self) -> str:
        return str(self._metric_combo.currentData() or self.METRICS[0].code)

    def _current_cluster_name(self) -> str | None:
        value = self._cluster_combo.currentData()
        return str(value) if value else None

    def _current_annotation_name(self) -> str | None:
        value = self._annotation_combo.currentData()
        return str(value) if value else None

    def _set_status(self, text: str) -> None:
        self._status_label.setText(text)

    def _schedule_refresh(self) -> None:
        self._refresh_timer.start()

    def _on_bar_size_changed(self, _value: int) -> None:
        self._canvas.set_bar_height(self._bar_slider.value())
        self._annotation_warning.setVisible(self._bar_slider.value() < 5 and self._current_annotation_name() is not None)
        self._schedule_refresh()

    def _on_selection_changed(self, selection: list[int]) -> None:
        self._selected_source_indices = set(selection)
        if self.cb_apply_auto.isChecked():
            self._apply()
        else:
            self._notify_output_changed()

    def _populate_column_controls(self) -> None:
        self._cluster_combo.blockSignals(True)
        self._annotation_combo.blockSignals(True)
        self._cluster_combo.clear()
        self._annotation_combo.clear()
        self._annotation_combo.addItem(i18n.t("(None)"), None)

        if self._dataset is not None:
            candidates = _cluster_candidates(self._dataset)
            for column in candidates:
                self._cluster_combo.addItem(column.name, column.name)
            annotations = _annotation_candidates(self._dataset)
            for column in annotations:
                self._annotation_combo.addItem(column.name, column.name)

            cluster_index = self._find_combo_index(self._cluster_combo, self._pending_cluster_name)
            if cluster_index == -1 and candidates:
                target_names = {column.name for column in self._dataset.domain.target_columns}
                preferred = next((column.name for column in candidates if column.name in target_names), candidates[0].name)
                cluster_index = self._find_combo_index(self._cluster_combo, preferred)
            if cluster_index != -1:
                self._cluster_combo.setCurrentIndex(cluster_index)

            annotation_index = self._find_combo_index(self._annotation_combo, self._pending_annotation_name)
            self._annotation_combo.setCurrentIndex(0 if annotation_index == -1 else annotation_index)

        self._cluster_combo.blockSignals(False)
        self._annotation_combo.blockSignals(False)

    def _find_combo_index(self, combo: QComboBox, value: str | None) -> int:
        if value is None:
            return -1
        for index in range(combo.count()):
            if combo.itemData(index) == value:
                return index
        return -1

    def _recompute_plot(self) -> None:
        self._pending_cluster_name = self._current_cluster_name()
        self._pending_annotation_name = self._current_annotation_name()
        self._pending_metric_code = self._current_metric_code()

        self._error_message = ""
        self._warning_messages = []
        self._full_scores = None
        self._selected_mask = None
        self._valid_mask = None
        self._cluster_labels = []
        self._score_column_name = "Silhouette"

        if self._dataset is None:
            self._canvas.clear()
            self._set_status(i18n.t("Connect a dataset with feature columns and cluster labels."))
            if self.cb_apply_auto.isChecked():
                self._apply()
            else:
                self._notify_output_changed()
            return

        cluster_name = self._current_cluster_name()
        if not cluster_name:
            self._canvas.clear()
            self._error_message = i18n.t("Input does not have any suitable labels.")
            self._set_status(self._error_message)
            if self.cb_apply_auto.isChecked():
                self._apply()
            else:
                self._notify_output_changed()
            return

        try:
            scores, selected_mask, valid_mask, cluster_labels, plot_groups, score_name = self._build_plot_model(cluster_name)
        except ValueError as error:
            self._canvas.clear()
            self._error_message = str(error)
            self._set_status(self._compose_status())
            if self.cb_apply_auto.isChecked():
                self._apply()
            else:
                self._notify_output_changed()
            return

        self._full_scores = scores
        self._selected_mask = selected_mask
        self._valid_mask = valid_mask
        self._cluster_labels = cluster_labels
        self._score_column_name = score_name

        if self._pending_selection:
            self._selected_source_indices = set(self._pending_selection)
            self._pending_selection = []

        valid_selected = self._selected_source_indices & {index for index, valid in enumerate(valid_mask) if valid}
        score_min = float(np.nanmin(scores[valid_mask])) if np.any(valid_mask) else -1.0
        score_max = float(np.nanmax(scores[valid_mask])) if np.any(valid_mask) else 1.0
        score_min = min(score_min, 0.0)
        if not np.isfinite(score_max):
            score_max = 1.0
        if abs(score_max - score_min) < 1e-9:
            score_max = score_min + 1.0

        self._canvas.set_bar_height(self._bar_slider.value())
        self._canvas.set_row_names_visible(self._bar_slider.value() >= 5 and self._current_annotation_name() is not None)
        self._annotation_warning.setVisible(self._bar_slider.value() < 5 and self._current_annotation_name() is not None)
        self._canvas.set_plot(plot_groups, score_min=score_min, score_max=score_max, selected_rows=valid_selected)
        self._selected_source_indices = set(self._canvas.selection())
        self._set_status(self._compose_status())

        if self.cb_apply_auto.isChecked():
            self._apply()
        else:
            self._notify_output_changed()

    def _build_plot_model(
        self,
        cluster_name: str,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str], list[_PlotGroup], str]:
        assert self._dataset is not None
        dataset = self._dataset
        cluster_series = dataset.dataframe.get_column(cluster_name)
        labels, cluster_missing, cluster_labels = _encode_labels(cluster_series)
        if len(cluster_labels) < 2:
            raise ValueError(i18n.t("Need at least two non-empty clusters."))

        distances, distance_invalid = self._distance_matrix(dataset, self._current_metric_code())
        valid_mask = ~(cluster_missing | distance_invalid)
        valid_labels = labels[valid_mask]
        if len(valid_labels) == 0 or len(set(valid_labels.tolist())) < 2:
            raise ValueError(i18n.t("Need at least two non-empty clusters."))
        if len(set(valid_labels.tolist())) == len(valid_labels):
            raise ValueError(i18n.t("All clusters are singletons."))

        valid_distances = distances[np.ix_(valid_mask, valid_mask)]
        valid_scores = _silhouette_scores(valid_distances, valid_labels)

        full_scores = np.full(dataset.row_count, np.nan, dtype=float)
        full_scores[valid_mask] = valid_scores
        selected_mask = np.zeros(dataset.row_count, dtype=bool)
        for index in self._selected_source_indices:
            if 0 <= index < dataset.row_count and valid_mask[index]:
                selected_mask[index] = True

        missing_count = int(np.count_nonzero(cluster_missing))
        if missing_count:
            noun = i18n.t("instance") if missing_count == 1 else i18n.t("instances")
            self._warning_messages.append(i18n.tf("{count} {noun} omitted (missing cluster assignment)", count=missing_count, noun=noun))
        invalid_count = int(np.count_nonzero(distance_invalid))
        if invalid_count:
            noun = i18n.t("instance") if invalid_count == 1 else i18n.t("instances")
            self._warning_messages.append(i18n.tf("{count} {noun} omitted (undefined distances)", count=invalid_count, noun=noun))

        annotation_name = self._current_annotation_name()
        annotations = []
        if annotation_name:
            annotations = [_display_value(value) for value in dataset.dataframe.get_column(annotation_name).to_list()]

        valid_rows: list[_PlotRow] = []
        for offset, source_index in enumerate(np.flatnonzero(valid_mask)):
            cluster_index = int(valid_labels[offset])
            valid_rows.append(
                _PlotRow(
                    source_index=int(source_index),
                    score=float(valid_scores[offset]),
                    cluster_label=cluster_labels[cluster_index],
                    color=_PALETTE[cluster_index % len(_PALETTE)],
                    annotation=annotations[int(source_index)] if annotations else "",
                )
            )

        if self._group_checkbox.isChecked():
            plot_groups = self._grouped_rows(valid_rows, cluster_labels)
        else:
            plot_groups = [
                _PlotGroup(
                    label=_UNGROUPED_LABEL,
                    color=_UNGROUPED_COLOR,
                    rows=sorted(
                        [
                            _PlotRow(
                                source_index=row.source_index,
                                score=row.score,
                                cluster_label=row.cluster_label,
                                color=_UNGROUPED_COLOR,
                                annotation=row.annotation,
                            )
                            for row in valid_rows
                        ],
                        key=lambda row: row.score,
                        reverse=True,
                    ),
                )
            ]

        score_name = _unique_name(list(dataset.dataframe.columns), f"Silhouette ({cluster_name})")
        return full_scores, selected_mask, valid_mask, cluster_labels, plot_groups, score_name

    def _grouped_rows(self, rows: list[_PlotRow], cluster_labels: list[str]) -> list[_PlotGroup]:
        groups: list[_PlotGroup] = []
        for cluster_index, cluster_label in enumerate(cluster_labels):
            cluster_rows = [row for row in rows if row.cluster_label == cluster_label]
            if not cluster_rows:
                continue
            cluster_rows.sort(key=lambda row: row.score, reverse=True)
            groups.append(
                _PlotGroup(
                    label=cluster_label,
                    color=_PALETTE[cluster_index % len(_PALETTE)],
                    rows=cluster_rows,
                )
            )
        return groups

    def _distance_matrix(self, dataset: DatasetHandle, metric_code: str) -> tuple[np.ndarray, np.ndarray]:
        numeric_columns, discrete_columns, _ignored_columns = _feature_columns(dataset)

        if metric_code == "cosine" and discrete_columns:
            self._warning_messages.append(i18n.t("Ignoring categorical features."))
            discrete_columns = []

        numeric = _prepare_numeric_matrix(dataset, numeric_columns, use_median=metric_code == "manhattan")
        discrete = _prepare_discrete_matrix(dataset, discrete_columns) if metric_code != "cosine" else np.empty((dataset.row_count, 0), dtype=object)

        if metric_code == "cosine":
            if numeric.shape[1] == 0:
                raise ValueError(i18n.t("Cosine distance requires at least one numeric feature."))
            distances, invalid = _cosine_distances(numeric)
            return distances, invalid

        if numeric.shape[1] == 0 and discrete.shape[1] == 0:
            raise ValueError(i18n.t("Input data does not have suitable feature columns for silhouette computation."))

        if metric_code == "euclidean":
            return _euclidean_distances(numeric, discrete), np.zeros(dataset.row_count, dtype=bool)
        if metric_code == "manhattan":
            return _manhattan_distances(numeric, discrete), np.zeros(dataset.row_count, dtype=bool)
        raise ValueError(i18n.t("Unsupported distance metric."))

    def _compose_status(self) -> str:
        if self._error_message:
            return self._error_message

        dataset_name = self._dataset.display_name if self._dataset is not None else i18n.t("No data")
        selected_count = len(self._selected_source_indices)
        valid_count = int(np.count_nonzero(self._valid_mask)) if self._valid_mask is not None else 0
        parts = [
            i18n.tf("{name}: {count} scored instances", name=dataset_name, count=valid_count),
            i18n.tf("{count} selected", count=selected_count),
            i18n.tf("metric: {metric}", metric=self._metric_combo.currentText()),
        ]
        if self._warning_messages:
            parts.extend(self._warning_messages)
        return " | ".join(parts)

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

    def _apply(self) -> None:
        if self._dataset is None or self._full_scores is None:
            self._selected_dataset = None
            self._annotated_dataset = None
            self._notify_output_changed()
            return

        dataset = self._dataset
        full_scores = self._full_scores
        selected_rows = sorted(index for index in self._selected_source_indices if 0 <= index < dataset.row_count)
        selected_mask = np.zeros(dataset.row_count, dtype=bool)
        for index in selected_rows:
            selected_mask[index] = True

        role_overrides = _role_overrides(dataset)
        role_overrides[self._score_column_name] = "meta"
        role_overrides[SELECTED_COLUMN_NAME] = "meta"

        annotated_frame = dataset.dataframe.with_columns(
            [
                pl.Series(self._score_column_name, [None if np.isnan(value) else float(value) for value in full_scores]),
                pl.Series(SELECTED_COLUMN_NAME, selected_mask.tolist()),
            ]
        )
        self._annotated_dataset = self._generated_datasets.build_dataset(
            annotated_frame,
            dataset_id=f"silhouette-{self._screen_token}-annotated",
            display_name=i18n.t("Annotated Data"),
            file_name=f"silhouette-{self._screen_token}-annotated.csv",
            role_overrides=role_overrides,
            annotations={
                "source_row_indices": list(range(dataset.row_count)),
                "selected_row_indices": selected_rows,
                "cluster_variable": self._current_cluster_name(),
                "silhouette_column": self._score_column_name,
            },
        )

        if not selected_rows:
            self._selected_dataset = None
            self._notify_output_changed()
            return

        selected_frame = dataset.dataframe.filter(pl.Series("mask", selected_mask.tolist())).with_columns(
            [pl.Series(self._score_column_name, [float(full_scores[index]) for index in selected_rows])]
        )
        selected_role_overrides = _role_overrides(dataset)
        selected_role_overrides[self._score_column_name] = "meta"
        self._selected_dataset = self._generated_datasets.build_dataset(
            selected_frame,
            dataset_id=f"silhouette-{self._screen_token}-selected",
            display_name=i18n.t("Selected Data"),
            file_name=f"silhouette-{self._screen_token}-selected.csv",
            role_overrides=selected_role_overrides,
            annotations={
                "source_row_indices": selected_rows,
                "cluster_variable": self._current_cluster_name(),
                "silhouette_column": self._score_column_name,
            },
        )
        self._notify_output_changed()
