from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl
from PySide6.QtCore import QPoint, QRect, QSize, Qt, Signal
from PySide6.QtGui import QColor, QPainter, QPen, QPolygon
from PySide6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QToolButton,
    QToolTip,
    QVBoxLayout,
    QWidget,
)

from portakal_app.data.models import DatasetHandle
from portakal_app.data.services.generated_dataset_service import GeneratedDatasetService
from portakal_app.ui import i18n
from portakal_app.ui.icons import get_toolbar_icon
from portakal_app.ui.screens.node_screen import WorkflowNodeScreenSupport
from portakal_app.ui.screens.visualize_common import build_selection_outputs, discrete_columns, nice_ticks
from portakal_app.ui.shared.type_icons import type_badge_icon


MAX_FEATURES = 200
SEL_MAX_INSTANCES = 10000

ORANGE_DISCRETE_COLORS = (
    (70, 190, 250),
    (237, 70, 47),
    (170, 242, 43),
    (245, 174, 50),
    (255, 255, 0),
    (255, 0, 255),
    (0, 255, 255),
    (128, 0, 255),
)

SELECT_MODE = "select"
PAN_MODE = "pan"
ZOOM_MODE = "zoom"


@dataclass(frozen=True)
class _LineGroupState:
    label: str
    color: QColor
    row_indices: np.ndarray
    values: np.ndarray
    subset_mask: np.ndarray


def _ccw(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    return (c[..., 1] - a[..., 1]) * (b[..., 0] - a[..., 0]) > (b[..., 1] - a[..., 1]) * (c[..., 0] - a[..., 0])


def _segments_intersect(p1: np.ndarray, p2: np.ndarray, q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    return np.logical_and(_ccw(p1, q1, q2) != _ccw(p2, q1, q2), _ccw(p1, p2, q1) != _ccw(p1, p2, q2))


class _LinePlotCanvas(QWidget):
    selectionChanged = Signal(list)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._groups: list[_LineGroupState] = []
        self._aggregates: list[_LineGroupState] = []
        self._feature_names: list[str] = []
        self._selected_rows: set[int] = set()
        self._show_profiles = False
        self._show_range = True
        self._show_mean = True
        self._show_error = False
        self._selection_enabled = True
        self._graph_mode = SELECT_MODE
        self._y_domain = (0.0, 1.0)
        self._view_x = (0.5, 1.5)
        self._view_y = (0.0, 1.0)
        self._drag_start: QPoint | None = None
        self._drag_end: QPoint | None = None
        self._pan_origin: tuple[float, float, tuple[float, float], tuple[float, float]] | None = None
        self._profile_hit_areas: list[tuple[QRect, str]] = []
        self._subset_present = False
        self.setMinimumHeight(320)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMouseTracking(True)

    def set_state(
        self,
        groups: list[_LineGroupState],
        feature_names: list[str],
        *,
        show_profiles: bool,
        show_range: bool,
        show_mean: bool,
        show_error: bool,
        selection_enabled: bool,
    ) -> None:
        self._groups = groups
        self._aggregates = groups
        self._feature_names = feature_names
        self._show_profiles = show_profiles
        self._show_range = show_range
        self._show_mean = show_mean
        self._show_error = show_error
        self._selection_enabled = selection_enabled
        self._subset_present = any(bool(np.any(group.subset_mask)) for group in groups)
        self._profile_hit_areas = []
        self._reset_view()
        self.update()

    def set_selected_rows(self, rows: list[int]) -> None:
        self._selected_rows = set(rows)
        self.update()

    def set_graph_mode(self, mode: str) -> None:
        self._graph_mode = mode

    def clear_selection(self) -> None:
        self._drag_start = None
        self._drag_end = None
        if self._selected_rows:
            self._selected_rows.clear()
            self.selectionChanged.emit([])
        self.update()

    def reset_view(self) -> None:
        self._reset_view()
        self.update()

    def wheelEvent(self, event) -> None:
        if not self._feature_names:
            return super().wheelEvent(event)
        chart = self._chart_rect()
        if not chart.contains(event.position().toPoint()):
            return super().wheelEvent(event)
        factor = 0.85 if event.angleDelta().y() > 0 else 1.18
        center = event.position().toPoint()
        wx, wy = self._screen_to_world(center, chart)
        self._zoom_view(factor, factor, wx, wy)
        self.update()
        event.accept()

    def mousePressEvent(self, event) -> None:
        if event.button() != Qt.MouseButton.LeftButton:
            return super().mousePressEvent(event)
        chart = self._chart_rect()
        if not chart.contains(event.position().toPoint()):
            return super().mousePressEvent(event)
        self._drag_start = event.position().toPoint()
        self._drag_end = self._drag_start
        if self._graph_mode == PAN_MODE:
            world_x, world_y = self._screen_to_world(self._drag_start, chart)
            self._pan_origin = (world_x, world_y, self._view_x, self._view_y)
        self.update()

    def mouseMoveEvent(self, event) -> None:
        pos = event.position().toPoint()
        if self._drag_start is not None:
            self._drag_end = pos
            if self._graph_mode == PAN_MODE and self._pan_origin is not None:
                chart = self._chart_rect()
                world_x, world_y = self._screen_to_world(pos, chart)
                anchor_x, anchor_y, original_x, original_y = self._pan_origin
                dx = anchor_x - world_x
                dy = anchor_y - world_y
                self._view_x = (original_x[0] + dx, original_x[1] + dx)
                self._view_y = (original_y[0] + dy, original_y[1] + dy)
            self.update()
            return
        for rect, tooltip in self._profile_hit_areas:
            if rect.contains(pos):
                QToolTip.showText(event.globalPosition().toPoint(), tooltip, self)
                return
        QToolTip.hideText()

    def mouseReleaseEvent(self, event) -> None:
        if event.button() != Qt.MouseButton.LeftButton or self._drag_start is None:
            return super().mouseReleaseEvent(event)
        chart = self._chart_rect()
        self._drag_end = event.position().toPoint()
        if self._graph_mode == SELECT_MODE:
            if self._selection_enabled:
                incoming = set(self._intersected_rows(chart))
                modifiers = event.modifiers()
                if modifiers & Qt.KeyboardModifier.AltModifier:
                    self._selected_rows.difference_update(incoming)
                elif modifiers & (Qt.KeyboardModifier.ShiftModifier | Qt.KeyboardModifier.ControlModifier):
                    self._selected_rows.update(incoming)
                else:
                    self._selected_rows = set(incoming)
                self.selectionChanged.emit(sorted(self._selected_rows))
        elif self._graph_mode == ZOOM_MODE:
            rect = QRect(self._drag_start, self._drag_end).normalized()
            if rect.width() >= 10 and rect.height() >= 10:
                x0, y1 = self._screen_to_world(rect.bottomLeft(), chart)
                x1, y0 = self._screen_to_world(rect.topRight(), chart)
                self._view_x = (min(x0, x1), max(x0, x1))
                self._view_y = (min(y0, y1), max(y0, y1))
        self._drag_start = None
        self._drag_end = None
        self._pan_origin = None
        self.update()

    def leaveEvent(self, _event) -> None:
        QToolTip.hideText()

    def paintEvent(self, _event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), QColor("#ffffff"))
        self._profile_hit_areas = []

        if not self._feature_names or not self._groups:
            painter.setPen(QColor("#8d877d"))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, i18n.t("Load data with at least one numeric feature."))
            return

        chart = self._chart_rect()
        painter.setPen(QPen(QColor("#222222"), 1.0))
        painter.drawLine(chart.left(), chart.bottom(), chart.right(), chart.bottom())
        painter.drawLine(chart.left(), chart.top(), chart.left(), chart.bottom())

        x_ticks = self._visible_feature_ticks()
        y_ticks = nice_ticks(self._view_y[0], self._view_y[1], count=6)
        painter.setPen(QColor("#222222"))
        for y_tick in y_ticks:
            py = self._world_to_screen_y(y_tick, chart)
            painter.drawLine(chart.left() - 5, py, chart.left(), py)
            painter.drawText(QRect(2, py - 8, chart.left() - 10, 16), Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter, f"{y_tick:.3g}")
        for feature_index, name in x_ticks:
            px = self._world_to_screen_x(feature_index + 1, chart)
            painter.drawLine(px, chart.bottom(), px, chart.bottom() + 5)
            painter.drawText(
                QRect(px - 56, chart.bottom() + 8, 112, 40),
                Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop | Qt.TextFlag.TextWordWrap,
                name,
            )

        if self._show_range:
            for group in self._groups:
                self._draw_range(painter, chart, group, selected=False)
            for group in self._groups:
                self._draw_range(painter, chart, group, selected=True)

        if self._show_profiles or self._show_range:
            for group in self._groups:
                self._draw_subset_profiles(painter, chart, group)

        if self._show_profiles:
            for group in self._groups:
                self._draw_profiles(painter, chart, group)
            for group in self._groups:
                self._draw_selected_profiles(painter, chart, group)

        if self._show_mean:
            for group in self._groups:
                self._draw_mean(painter, chart, group)

        if self._show_error:
            for group in self._groups:
                self._draw_error(painter, chart, group)

        if len(self._groups) > 1:
            self._draw_legend(painter, chart)

        if self._drag_start is not None and self._drag_end is not None:
            if self._graph_mode == SELECT_MODE:
                painter.setPen(QPen(QColor("#111111"), 2.0))
                painter.drawLine(self._drag_start, self._drag_end)
            elif self._graph_mode == ZOOM_MODE:
                painter.setPen(QPen(QColor("#2563eb"), 1.5, Qt.PenStyle.DashLine))
                painter.drawRect(QRect(self._drag_start, self._drag_end).normalized())

    def _draw_profiles(self, painter: QPainter, chart: QRect, group: _LineGroupState) -> None:
        selected_mask = np.isin(group.row_indices, np.asarray(sorted(self._selected_rows), dtype=int))
        has_focus = bool(self._selected_rows) or self._subset_present
        for row_index, values, subset, selected in zip(group.row_indices, group.values, group.subset_mask, selected_mask):
            if subset or selected:
                continue
            self._draw_profile_segments(
                painter,
                chart,
                int(row_index),
                values,
                group.color,
                alpha=50 if has_focus else 100,
                width=1.0,
                dashed=False,
                tooltip_label=group.label,
            )
            self._draw_missing_profile(
                painter,
                chart,
                int(row_index),
                values,
                group.color,
                alpha=50 if has_focus else 100,
                width=1.0,
                tooltip_label=group.label,
            )

    def _draw_subset_profiles(self, painter: QPainter, chart: QRect, group: _LineGroupState) -> None:
        for row_index, values, subset in zip(group.row_indices, group.values, group.subset_mask):
            if not subset:
                continue
            self._draw_profile_segments(
                painter,
                chart,
                int(row_index),
                values,
                group.color,
                alpha=170,
                width=3.0,
                dashed=False,
                tooltip_label=f"{group.label} ({i18n.t('subset')})",
            )
            self._draw_missing_profile(
                painter,
                chart,
                int(row_index),
                values,
                group.color,
                alpha=170,
                width=3.0,
                tooltip_label=f"{group.label} ({i18n.t('subset')})",
            )

    def _draw_selected_profiles(self, painter: QPainter, chart: QRect, group: _LineGroupState) -> None:
        selected_mask = np.isin(group.row_indices, np.asarray(sorted(self._selected_rows), dtype=int))
        selected_color = QColor(Qt.GlobalColor.black) if self._subset_present else QColor(group.color)
        for row_index, values, selected in zip(group.row_indices, group.values, selected_mask):
            if not selected:
                continue
            self._draw_profile_segments(
                painter,
                chart,
                int(row_index),
                values,
                selected_color,
                alpha=170,
                width=3.0,
                dashed=False,
                tooltip_label=f"{group.label} ({i18n.t('selected')})",
            )
            self._draw_missing_profile(
                painter,
                chart,
                int(row_index),
                values,
                selected_color,
                alpha=170,
                width=3.0,
                tooltip_label=f"{group.label} ({i18n.t('selected')})",
            )

    def _draw_range(self, painter: QPainter, chart: QRect, group: _LineGroupState, *, selected: bool) -> None:
        values = self._selected_group_values(group) if selected else group.values
        if values.size == 0:
            return
        with np.errstate(all="ignore"):
            low = np.nanmin(values, axis=0)
            high = np.nanmax(values, axis=0)
        if not np.isfinite(low).any() or not np.isfinite(high).any():
            return
        fill = QColor(group.color)
        fill.setAlpha(50 if selected else 25)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(fill)
        for run in self._band_runs(low, high):
            polygon = QPolygon(
                [QPoint(self._world_to_screen_x(index + 1, chart), self._world_to_screen_y(float(low[index]), chart)) for index in run]
                + [QPoint(self._world_to_screen_x(index + 1, chart), self._world_to_screen_y(float(high[index]), chart)) for index in run[::-1]]
            )
            painter.drawPolygon(polygon)
        painter.setBrush(Qt.BrushStyle.NoBrush)

    def _draw_mean(self, painter: QPainter, chart: QRect, group: _LineGroupState) -> None:
        with np.errstate(all="ignore"):
            mean = np.nanmean(group.values, axis=0)
        self._draw_stat_curve(painter, chart, mean, group.color.darker(110), 6.0)

    def _draw_error(self, painter: QPainter, chart: QRect, group: _LineGroupState) -> None:
        with np.errstate(all="ignore"):
            mean = np.nanmean(group.values, axis=0)
            std = np.nanstd(group.values, axis=0)
        painter.setPen(QPen(group.color, 1.2))
        for index, (mean_value, std_value) in enumerate(zip(mean, std), start=1):
            if not np.isfinite(mean_value) or not np.isfinite(std_value):
                continue
            x = self._world_to_screen_x(index, chart)
            y0 = self._world_to_screen_y(float(mean_value - std_value), chart)
            y1 = self._world_to_screen_y(float(mean_value + std_value), chart)
            painter.drawLine(x, y0, x, y1)
            painter.drawLine(x - 4, y0, x + 4, y0)
            painter.drawLine(x - 4, y1, x + 4, y1)

    def _draw_stat_curve(self, painter: QPainter, chart: QRect, values: np.ndarray, color: QColor, width: float) -> None:
        segments = self._profile_runs(values)
        pen = QPen(color, width)
        pen.setCosmetic(True)
        painter.setPen(pen)
        for run in segments:
            points = [
                QPoint(self._world_to_screen_x(index + 1, chart), self._world_to_screen_y(float(values[index]), chart))
                for index in run
            ]
            for start, end in zip(points[:-1], points[1:]):
                painter.drawLine(start, end)

    def _draw_profile_segments(
        self,
        painter: QPainter,
        chart: QRect,
        row_index: int,
        values: np.ndarray,
        color: QColor,
        *,
        alpha: int,
        width: float,
        dashed: bool,
        tooltip_label: str,
    ) -> None:
        pen = QPen(color, width, Qt.PenStyle.DashLine if dashed else Qt.PenStyle.SolidLine)
        pen.setCosmetic(True)
        draw_color = QColor(color)
        draw_color.setAlpha(alpha)
        pen.setColor(draw_color)
        painter.setPen(pen)
        for run in self._profile_runs(values):
            points = [
                QPoint(self._world_to_screen_x(index + 1, chart), self._world_to_screen_y(float(values[index]), chart))
                for index in run
            ]
            for start, end in zip(points[:-1], points[1:]):
                painter.drawLine(start, end)
            if points:
                bounds = QRect(
                    min(point.x() for point in points) - 4,
                    min(point.y() for point in points) - 4,
                    max(point.x() for point in points) - min(point.x() for point in points) + 8,
                    max(point.y() for point in points) - min(point.y() for point in points) + 8,
                )
                self._profile_hit_areas.append((bounds, f"Row: {row_index}\nGroup: {tooltip_label}"))

    def _draw_missing_profile(
        self,
        painter: QPainter,
        chart: QRect,
        row_index: int,
        values: np.ndarray,
        color: QColor,
        *,
        alpha: int,
        width: float,
        tooltip_label: str,
    ) -> None:
        gaps = self._missing_gap_runs(values)
        if not gaps:
            return
        pen = QPen(color, width, Qt.PenStyle.DashLine)
        pen.setCosmetic(True)
        dashed = QColor(color)
        dashed.setAlpha(alpha)
        pen.setColor(dashed)
        painter.setPen(pen)
        for left_index, right_index in gaps:
            left = QPoint(self._world_to_screen_x(left_index + 1, chart), self._world_to_screen_y(float(values[left_index]), chart))
            right = QPoint(self._world_to_screen_x(right_index + 1, chart), self._world_to_screen_y(float(values[right_index]), chart))
            painter.drawLine(left, right)
            bounds = QRect(
                min(left.x(), right.x()) - 4,
                min(left.y(), right.y()) - 4,
                abs(left.x() - right.x()) + 8,
                abs(left.y() - right.y()) + 8,
            )
            self._profile_hit_areas.append((bounds, f"Row: {row_index}\nGroup: {tooltip_label}\nMissing values"))

    def _draw_legend(self, painter: QPainter, chart: QRect) -> None:
        legend_width = 156
        legend_height = min(max(40, 16 + 22 * len(self._groups)), 180)
        rect = QRect(chart.right() - legend_width - 8, chart.top() + 8, legend_width, legend_height)
        painter.setPen(QPen(QColor("#c9c9c9"), 1.0))
        painter.setBrush(QColor("#ffffff"))
        painter.drawRect(rect)
        y = rect.top() + 8
        for group in self._groups[:8]:
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(group.color)
            painter.drawRect(rect.left() + 8, y + 3, 10, 10)
            painter.setPen(QColor("#222222"))
            painter.drawText(QRect(rect.left() + 24, y, rect.width() - 28, 16), Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, group.label)
            y += 22

    def _intersected_rows(self, chart: QRect) -> list[int]:
        if self._drag_start is None or self._drag_end is None:
            return []
        segment_start = np.asarray([self._drag_start.x(), self._drag_start.y()], dtype=float)
        segment_end = np.asarray([self._drag_end.x(), self._drag_end.y()], dtype=float)
        selected: list[int] = []
        for group in self._groups:
            for row_index, values in zip(group.row_indices, group.values):
                screen_segments = self._screen_segments(values, chart)
                if not screen_segments:
                    continue
                for q1, q2 in screen_segments:
                    intersects = _segments_intersect(
                        np.broadcast_to(segment_start, q1.shape),
                        np.broadcast_to(segment_end, q1.shape),
                        q1,
                        q2,
                    )
                    if bool(np.any(intersects)):
                        selected.append(int(row_index))
                        break
        return selected

    def _screen_segments(self, values: np.ndarray, chart: QRect) -> list[tuple[np.ndarray, np.ndarray]]:
        segments: list[tuple[np.ndarray, np.ndarray]] = []
        for run in self._profile_runs(values):
            if len(run) < 2:
                continue
            points = np.array(
                [
                    [self._world_to_screen_x(index + 1, chart), self._world_to_screen_y(float(values[index]), chart)]
                    for index in run
                ],
                dtype=float,
            )
            segments.append((points[:-1], points[1:]))
        return segments

    @staticmethod
    def _profile_runs(values: np.ndarray) -> list[list[int]]:
        runs: list[list[int]] = []
        current: list[int] = []
        for index, value in enumerate(values):
            if np.isfinite(value):
                current.append(index)
            elif current:
                runs.append(current)
                current = []
        if current:
            runs.append(current)
        return runs

    @staticmethod
    def _missing_gap_runs(values: np.ndarray) -> list[tuple[int, int]]:
        finite = np.isfinite(values)
        indices = np.flatnonzero(finite)
        if len(indices) < 2:
            return []
        gaps: list[tuple[int, int]] = []
        for left, right in zip(indices[:-1], indices[1:]):
            if right - left > 1:
                gaps.append((int(left), int(right)))
        return gaps

    @staticmethod
    def _band_runs(low: np.ndarray, high: np.ndarray) -> list[list[int]]:
        runs: list[list[int]] = []
        current: list[int] = []
        for index, (lo, hi) in enumerate(zip(low, high)):
            if np.isfinite(lo) and np.isfinite(hi):
                current.append(index)
            elif current:
                runs.append(current)
                current = []
        if current:
            runs.append(current)
        return runs

    def _selected_group_values(self, group: _LineGroupState) -> np.ndarray:
        if not self._selected_rows:
            return np.empty((0, len(self._feature_names)), dtype=float)
        mask = np.isin(group.row_indices, np.asarray(sorted(self._selected_rows), dtype=int))
        return group.values[mask]

    def _visible_feature_ticks(self) -> list[tuple[int, str]]:
        ticks: list[tuple[int, str]] = []
        start = max(1, int(np.floor(self._view_x[0])))
        end = min(len(self._feature_names), int(np.ceil(self._view_x[1])))
        for feature_index in range(start, end + 1):
            ticks.append((feature_index - 1, self._feature_names[feature_index - 1]))
        return ticks

    def _chart_rect(self) -> QRect:
        return QRect(82, 18, max(10, self.width() - 108), max(10, self.height() - 86))

    def _reset_view(self) -> None:
        if not self._feature_names:
            self._view_x = (0.5, 1.5)
            self._view_y = (0.0, 1.0)
            return
        y_values = [group.values[np.isfinite(group.values)] for group in self._groups]
        if any(values.size for values in y_values):
            data = np.concatenate([values for values in y_values if values.size])
            low = float(np.min(data))
            high = float(np.max(data))
            pad = (high - low) * 0.05 if high > low else 0.5
            self._y_domain = (low - pad, high + pad)
        else:
            self._y_domain = (0.0, 1.0)
        self._view_x = (0.5, len(self._feature_names) + 0.5)
        self._view_y = self._y_domain

    def _world_to_screen_x(self, x: float, chart: QRect) -> int:
        span = max(self._view_x[1] - self._view_x[0], 1e-9)
        return chart.left() + int((x - self._view_x[0]) / span * chart.width())

    def _world_to_screen_y(self, y: float, chart: QRect) -> int:
        span = max(self._view_y[1] - self._view_y[0], 1e-9)
        return chart.bottom() - int((y - self._view_y[0]) / span * chart.height())

    def _screen_to_world(self, point: QPoint, chart: QRect) -> tuple[float, float]:
        x = self._view_x[0] + (point.x() - chart.left()) / max(chart.width(), 1) * (self._view_x[1] - self._view_x[0])
        y = self._view_y[0] + (chart.bottom() - point.y()) / max(chart.height(), 1) * (self._view_y[1] - self._view_y[0])
        return x, y

    def _zoom_view(self, scale_x: float, scale_y: float, center_x: float, center_y: float) -> None:
        span_x = max((self._view_x[1] - self._view_x[0]) * scale_x, 0.5)
        span_y = max((self._view_y[1] - self._view_y[0]) * scale_y, 1e-6)
        self._view_x = (center_x - span_x / 2, center_x + span_x / 2)
        self._view_y = (center_y - span_y / 2, center_y + span_y / 2)


class LinePlotScreen(QWidget, WorkflowNodeScreenSupport):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._init_workflow_node_support()
        self._builder = GeneratedDatasetService()
        self._dataset: DatasetHandle | None = None
        self._subset: DatasetHandle | None = None
        self._selected_dataset: DatasetHandle | None = None
        self._annotated_dataset: DatasetHandle | None = None
        self._pending_selected_rows: list[int] = []
        self._pending_group_name = ""
        self._dirty = False

        root = QHBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(12)

        self._sidebar_scroll = QScrollArea(self)
        self._sidebar_scroll.setWidgetResizable(True)
        self._sidebar_scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        self._sidebar_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._sidebar_scroll.setMinimumWidth(300)
        self._sidebar_scroll.setMaximumWidth(340)
        root.addWidget(self._sidebar_scroll, 0)

        sidebar_host = QWidget(self._sidebar_scroll)
        self._sidebar_scroll.setWidget(sidebar_host)
        sidebar = QVBoxLayout(sidebar_host)
        sidebar.setContentsMargins(4, 4, 4, 4)
        sidebar.setSpacing(12)

        display_box = QGroupBox(i18n.t("Display"))
        display_layout = QVBoxLayout(display_box)
        self._profiles_cb = QCheckBox(i18n.t("Lines"))
        self._range_cb = QCheckBox(i18n.t("Range"))
        self._mean_cb = QCheckBox(i18n.t("Mean"))
        self._error_cb = QCheckBox(i18n.t("Error bars"))
        self._profiles_cb.setChecked(False)
        self._range_cb.setChecked(True)
        self._mean_cb.setChecked(True)
        self._error_cb.setChecked(False)
        display_layout.addWidget(self._profiles_cb)
        display_layout.addWidget(self._range_cb)
        display_layout.addWidget(self._mean_cb)
        display_layout.addWidget(self._error_cb)
        sidebar.addWidget(display_box)

        group_box = QGroupBox(i18n.t("Group by"))
        group_layout = QVBoxLayout(group_box)
        self._group_filter_edit = QLineEdit()
        self._group_filter_edit.setPlaceholderText(i18n.t("Filter..."))
        self._group_list = QListWidget()
        self._group_list.setMinimumHeight(118)
        group_layout.addWidget(self._group_filter_edit)
        group_layout.addWidget(self._group_list)
        sidebar.addWidget(group_box)

        tools_box = QGroupBox(i18n.t("Tools"))
        tools_layout = QHBoxLayout(tools_box)
        self._select_button = QToolButton()
        self._select_button.setIcon(get_toolbar_icon("arrow"))
        self._select_button.setToolTip(i18n.t("Select"))
        self._select_button.setCheckable(True)
        self._pan_button = QToolButton()
        self._pan_button.setIcon(get_toolbar_icon("pan"))
        self._pan_button.setToolTip(i18n.t("Pan"))
        self._pan_button.setCheckable(True)
        self._zoom_button = QToolButton()
        self._zoom_button.setIcon(get_toolbar_icon("zoom_in"))
        self._zoom_button.setToolTip(i18n.t("Zoom"))
        self._zoom_button.setCheckable(True)
        self._reset_view_button = QToolButton()
        self._reset_view_button.setIcon(get_toolbar_icon("reset"))
        self._reset_view_button.setToolTip(i18n.t("Reset Zoom"))
        self._mode_buttons = QButtonGroup(self)
        self._mode_buttons.setExclusive(True)
        for button in (self._select_button, self._pan_button, self._zoom_button):
            self._mode_buttons.addButton(button)
            tools_layout.addWidget(button)
        tools_layout.addWidget(self._reset_view_button)
        self._select_button.setChecked(True)
        sidebar.addWidget(tools_box)

        action_row = QHBoxLayout()
        self._auto_apply_cb = QCheckBox(i18n.t("Apply Automatically"))
        self._auto_apply_cb.setChecked(True)
        self._apply_button = QPushButton(i18n.t("Apply"))
        self._apply_button.setEnabled(False)
        action_row.addWidget(self._auto_apply_cb)
        action_row.addWidget(self._apply_button)
        sidebar.addLayout(action_row)

        self._dataset_label = QLabel(i18n.t("Dataset: none"))
        self._status_label = QLabel(i18n.t("Load data with at least one numeric feature."))
        self._status_label.setWordWrap(True)
        sidebar.addWidget(self._dataset_label)
        sidebar.addWidget(self._status_label)

        self._clear_button = QPushButton(i18n.t("Clear Selection"))
        sidebar.addWidget(self._clear_button)
        self._selection_label = QLabel(i18n.t("Selected: 0"))
        sidebar.addWidget(self._selection_label)
        sidebar.addStretch(1)

        self._canvas = _LinePlotCanvas(self)
        root.addWidget(self._canvas, 1)

        self._group_filter_edit.textChanged.connect(self._filter_group_list)
        self._group_list.currentItemChanged.connect(self._control_changed)
        self._profiles_cb.toggled.connect(self._control_changed)
        self._range_cb.toggled.connect(self._control_changed)
        self._mean_cb.toggled.connect(self._control_changed)
        self._error_cb.toggled.connect(self._control_changed)
        self._auto_apply_cb.toggled.connect(self._handle_auto_apply_changed)
        self._apply_button.clicked.connect(self._apply_controls)
        self._clear_button.clicked.connect(self._clear_selection)
        self._canvas.selectionChanged.connect(self._handle_selection_changed)
        self._mode_buttons.buttonClicked.connect(self._handle_graph_mode_changed)
        self._reset_view_button.clicked.connect(self._canvas.reset_view)

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
        self._apply_controls(force=True)

    def current_output_dataset(self) -> DatasetHandle | None:
        return self._selected_dataset

    def current_output_datasets(self) -> dict[str, DatasetHandle | None] | None:
        return {
            "Selected Data": self._selected_dataset,
            "Annotated Data": self._annotated_dataset,
        }

    def serialize_node_state(self) -> dict[str, object]:
        return {
            "group": self._current_group_name(),
            "show_profiles": self._profiles_cb.isChecked(),
            "show_range": self._range_cb.isChecked(),
            "show_mean": self._mean_cb.isChecked(),
            "show_error": self._error_cb.isChecked(),
            "auto_apply": self._auto_apply_cb.isChecked(),
            "graph_mode": self._current_graph_mode(),
            "selected_rows": list(self._pending_selected_rows),
        }

    def restore_node_state(self, payload: dict[str, object]) -> None:
        self._pending_selected_rows = [
            int(index)
            for index in payload.get("selected_rows", [])
            if isinstance(index, int | float)
        ]
        self._pending_group_name = str(payload.get("group", i18n.t("None")))
        self._profiles_cb.setChecked(bool(payload.get("show_profiles", False)))
        self._range_cb.setChecked(bool(payload.get("show_range", True)))
        self._mean_cb.setChecked(bool(payload.get("show_mean", True)))
        self._error_cb.setChecked(bool(payload.get("show_error", False)))
        self._auto_apply_cb.setChecked(bool(payload.get("auto_apply", True)))
        self._set_graph_mode(str(payload.get("graph_mode", SELECT_MODE)))

    def documentation_url(self) -> str:
        return "https://orangedatamining.com/widget-catalog/visualize/lineplot/"

    def _sync_controls(self) -> None:
        dataset = self._dataset
        current_group = self._pending_group_name or self._current_group_name()
        self._group_list.blockSignals(True)
        self._group_list.clear()
        self._group_list.addItem(QListWidgetItem(i18n.t("None")))
        groups = discrete_columns(dataset)
        for name in groups:
            column = next((column for column in dataset.domain.columns if column.name == name), None) if dataset is not None else None
            item = QListWidgetItem(type_badge_icon(column.logical_type), name) if column is not None else QListWidgetItem(name)
            self._group_list.addItem(item)
        self._group_list.blockSignals(False)
        effective_group = (
            current_group
            if current_group and current_group != i18n.t("None")
            else self._default_group_name(groups)
        )
        self._select_group_name(effective_group)
        self._pending_group_name = ""
        self._filter_group_list(self._group_filter_edit.text())
        self._group_list.setEnabled(bool(groups))

    def _control_changed(self, *_args) -> None:
        if self._auto_apply_cb.isChecked():
            self._apply_controls()
        else:
            self._dirty = True
            self._apply_button.setEnabled(True)

    def _handle_auto_apply_changed(self, checked: bool) -> None:
        self._apply_button.setEnabled(not checked and self._dirty)
        if checked and self._dirty:
            self._apply_controls()

    def _apply_controls(self, force: bool = False) -> None:
        self._dirty = False
        self._apply_button.setEnabled(False)
        self._refresh_plot(force=force)

    def _refresh_plot(self, *, force: bool = False) -> None:
        dataset = self._dataset
        self._selected_dataset = None
        self._annotated_dataset = None
        if dataset is None:
            self._dataset_label.setText(i18n.t("Dataset: none"))
            self._status_label.setText(i18n.t("Load data with at least one numeric feature."))
            self._canvas.set_state(
                [],
                [],
                show_profiles=self._profiles_cb.isChecked(),
                show_range=self._range_cb.isChecked(),
                show_mean=self._mean_cb.isChecked(),
                show_error=self._error_cb.isChecked(),
                selection_enabled=False,
            )
            self._handle_selection_changed([])
            return

        feature_names = self._numeric_feature_names(dataset)
        self._dataset_label.setText(i18n.tf("Dataset: {name}", name=dataset.display_name))
        if not feature_names:
            self._status_label.setText(i18n.t("Need at least one numeric feature."))
            self._canvas.set_state(
                [],
                [],
                show_profiles=self._profiles_cb.isChecked(),
                show_range=self._range_cb.isChecked(),
                show_mean=self._mean_cb.isChecked(),
                show_error=self._error_cb.isChecked(),
                selection_enabled=False,
            )
            self._handle_selection_changed([])
            return

        groups = self._build_groups(feature_names)
        if not groups:
            self._status_label.setText(i18n.t("No valid groups to display."))
            self._canvas.set_state(
                [],
                feature_names,
                show_profiles=self._profiles_cb.isChecked(),
                show_range=self._range_cb.isChecked(),
                show_mean=self._mean_cb.isChecked(),
                show_error=self._error_cb.isChecked(),
                selection_enabled=False,
            )
            self._handle_selection_changed([])
            return

        selection_enabled = self._selection_is_enabled(dataset.row_count)
        self._select_button.setEnabled(selection_enabled)
        if not selection_enabled and self._current_graph_mode() == SELECT_MODE:
            self._set_graph_mode(PAN_MODE if dataset.row_count >= SEL_MAX_INSTANCES else SELECT_MODE)
        self._status_label.setText(self._status_text(dataset, feature_names, selection_enabled))
        self._canvas.set_state(
            groups,
            feature_names,
            show_profiles=self._profiles_cb.isChecked(),
            show_range=self._range_cb.isChecked(),
            show_mean=self._mean_cb.isChecked(),
            show_error=self._error_cb.isChecked(),
            selection_enabled=selection_enabled,
        )
        self._handle_selection_changed(self._pending_selected_rows)

    def _selection_is_enabled(self, row_count: int) -> bool:
        display_has_selectable = self._profiles_cb.isChecked() or self._range_cb.isChecked()
        return display_has_selectable and row_count < SEL_MAX_INSTANCES

    def _status_text(self, dataset: DatasetHandle, feature_names: list[str], selection_enabled: bool) -> str:
        parts = [i18n.tf("Features: {count}", count=len(feature_names))]
        if len(self._numeric_feature_names(dataset, limit=False)) > MAX_FEATURES:
            parts.append(i18n.tf("Only first {count} shown", count=MAX_FEATURES))
        if not (self._profiles_cb.isChecked() or self._range_cb.isChecked() or self._mean_cb.isChecked()):
            parts.append(i18n.t("No display option is selected."))
        elif not selection_enabled and dataset.row_count >= SEL_MAX_INSTANCES:
            parts.append(i18n.tf("Selection disabled above {count} rows", count=SEL_MAX_INSTANCES))
        return " | ".join(parts)

    def _numeric_feature_names(self, dataset: DatasetHandle, *, limit: bool = True) -> list[str]:
        names = [column.name for column in dataset.domain.columns if column.logical_type == "numeric"]
        return names[:MAX_FEATURES] if limit else names

    def _build_groups(self, feature_names: list[str]) -> list[_LineGroupState]:
        dataset = self._dataset
        if dataset is None:
            return []
        matrix = np.column_stack(
            [
                dataset.dataframe.get_column(name).cast(pl.Float64, strict=False).to_numpy()
                for name in feature_names
            ]
        ).astype(float, copy=False)
        subset_rows = self._subset_rows()
        row_indices = np.arange(dataset.row_count, dtype=int)
        group_name = self._current_group_name()
        if not group_name or group_name == i18n.t("None"):
            return [
                _LineGroupState(
                    label=i18n.t("All data"),
                    color=QColor(Qt.GlobalColor.darkGray),
                    row_indices=row_indices,
                    values=matrix,
                    subset_mask=np.isin(row_indices, np.asarray(sorted(subset_rows), dtype=int)),
                )
            ]

        raw_values = dataset.dataframe.get_column(group_name).to_list()
        ordered_labels = []
        for value in raw_values:
            if value is None:
                continue
            label = str(value)
            if label not in ordered_labels:
                ordered_labels.append(label)
        groups: list[_LineGroupState] = []
        for index, label in enumerate(ordered_labels):
            mask = np.asarray([value is not None and str(value) == label for value in raw_values], dtype=bool)
            if not np.any(mask):
                continue
            group_rows = row_indices[mask]
            groups.append(
                _LineGroupState(
                    label=label,
                    color=QColor(*ORANGE_DISCRETE_COLORS[index % len(ORANGE_DISCRETE_COLORS)]),
                    row_indices=group_rows,
                    values=matrix[mask],
                    subset_mask=np.isin(group_rows, np.asarray(sorted(subset_rows), dtype=int)),
                )
            )
        return groups

    def _subset_rows(self) -> set[int]:
        subset = self._subset
        dataset = self._dataset
        if subset is None or dataset is None:
            return set()
        indexed = dataset.dataframe.with_row_index("__row__")
        shared = [column for column in dataset.dataframe.columns if column in subset.dataframe.columns]
        if not shared:
            return set()
        matches = indexed.join(subset.dataframe.select(shared).unique(), on=shared, how="semi")
        return {int(value) for value in matches.get_column("__row__").to_list()}

    def _handle_selection_changed(self, rows: list[int]) -> None:
        self._pending_selected_rows = sorted({index for index in rows})
        self._canvas.set_selected_rows(self._pending_selected_rows)
        self._selection_label.setText(i18n.tf("Selected: {count}", count=len(self._pending_selected_rows)))
        self._selected_dataset, self._annotated_dataset = build_selection_outputs(
            self._dataset,
            self._pending_selected_rows,
            generated_by="line-plot",
            service=self._builder,
        )
        self._notify_output_changed()

    def _clear_selection(self) -> None:
        self._canvas.clear_selection()

    def _filter_group_list(self, text: str) -> None:
        needle = text.strip().lower()
        first_visible = None
        for row in range(self._group_list.count()):
            item = self._group_list.item(row)
            visible = row == 0 or not needle or needle in item.text().lower()
            item.setHidden(not visible)
            if visible and first_visible is None:
                first_visible = item
        current = self._group_list.currentItem()
        if current is None or current.isHidden():
            if first_visible is not None:
                self._group_list.setCurrentItem(first_visible)

    def _current_group_name(self) -> str:
        item = self._group_list.currentItem()
        return item.text() if item is not None else i18n.t("None")

    def _default_group_name(self, groups: list[str]) -> str:
        dataset = self._dataset
        if dataset is None:
            return i18n.t("None")
        for column in dataset.domain.target_columns:
            if column.name in groups and column.logical_type in {"categorical", "boolean"}:
                return column.name
        return i18n.t("None")

    def _select_group_name(self, name: str) -> None:
        wanted = name or i18n.t("None")
        for row in range(self._group_list.count()):
            item = self._group_list.item(row)
            if item.text() == wanted:
                self._group_list.setCurrentRow(row)
                self._group_list.setCurrentItem(item)
                return
        if self._group_list.count():
            self._group_list.setCurrentRow(0)

    def _handle_graph_mode_changed(self, button) -> None:
        if button is self._select_button:
            self._canvas.set_graph_mode(SELECT_MODE)
        elif button is self._pan_button:
            self._canvas.set_graph_mode(PAN_MODE)
        elif button is self._zoom_button:
            self._canvas.set_graph_mode(ZOOM_MODE)

    def _set_graph_mode(self, mode: str) -> None:
        if mode == PAN_MODE:
            self._pan_button.setChecked(True)
        elif mode == ZOOM_MODE:
            self._zoom_button.setChecked(True)
        else:
            self._select_button.setChecked(True)
            mode = SELECT_MODE
        self._canvas.set_graph_mode(mode)

    def _current_graph_mode(self) -> str:
        if self._pan_button.isChecked():
            return PAN_MODE
        if self._zoom_button.isChecked():
            return ZOOM_MODE
        return SELECT_MODE
