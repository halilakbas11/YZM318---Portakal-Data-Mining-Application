from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import polars as pl

from PySide6.QtCore import QPoint, QRectF, QSize, Qt, Signal
from PySide6.QtGui import QColor, QFont, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import (
    QComboBox,
    QFrame,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QScrollArea,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from portakal_app.data.models import DatasetHandle
from portakal_app.data.services.generated_dataset_service import GeneratedDatasetService
from portakal_app.logistic_regression_artifacts import (
    LogisticRegressionClassifierArtifact,
    LogisticRegressionFeatureArtifact,
)
from portakal_app.models import WorkflowPayload
from portakal_app.ui import i18n
from portakal_app.ui.screens.node_screen import WorkflowNodeScreenSupport


SORT_LABELS = (
    "Original order",
    "Alphabetically",
    "Absolute importance",
    "Positive influence",
    "Negative influence",
)
DISPLAY_LABELS = ("All features", "Best ranked")
SCALE_LABELS = ("Points", "Log odds ratios")
NUMERIC_LABELS = ("1D projection", "2D curve")

_BG = QColor("#fffdf9")
_LINE = QColor("#5a4a36")
_MUTED = QColor("#8c806f")
_ACCENT = QColor("#4e79a7")
_GRID = QColor("#ddd6cb")


def _format_number(value: float) -> str:
    if not np.isfinite(value):
        return "0"
    magnitude = abs(value)
    if magnitude >= 100:
        return f"{value:.0f}"
    if magnitude >= 10:
        return f"{value:.1f}"
    if magnitude >= 1:
        return f"{value:.2f}"
    return f"{value:.3f}".rstrip("0").rstrip(".")


def _ruler_values(start: float, stop: float, max_width: float, round_to_nearest: bool = True) -> np.ndarray:
    if max_width <= 0:
        return np.asarray([0.0], dtype=float)
    diff = float(np.nan_to_num((stop - start) / max_width))
    if diff <= 0:
        return np.asarray([0.0], dtype=float)
    decimals = int(np.floor(np.log10(diff)))
    if diff > 4 * pow(10, decimals):
        step = 5 * pow(10, decimals + 2)
    elif diff > 2 * pow(10, decimals):
        step = 2 * pow(10, decimals + 2)
    elif diff > 1 * pow(10, decimals):
        step = 1 * pow(10, decimals + 2)
    else:
        step = 5 * pow(10, decimals + 1)
    remainder = start % step
    if not round_to_nearest:
        values = np.arange(start + step, stop + remainder, step) - remainder
        start_value = np.floor(start * 100.0) / 100.0
        stop_value = np.ceil(stop * 100.0) / 100.0
        return np.round(np.hstack((start_value, values, stop_value)), 2)
    round_by = int(-np.floor(np.log10(step)))
    return np.round(np.arange(start, stop + remainder + step, step) - remainder, round_by)


@dataclass(frozen=True)
class _FeatureState:
    feature: LogisticRegressionFeatureArtifact
    class_index: int
    scale_factor: float
    scale_min: float
    scale_max: float
    numeric_mode: int
    current_value: Any

    @property
    def raw_values(self) -> np.ndarray:
        values, _ = self.feature.contribution_values(self.class_index)
        return values

    @property
    def raw_contributions(self) -> np.ndarray:
        _, contributions = self.feature.contribution_values(self.class_index)
        return contributions

    @property
    def display_contributions(self) -> np.ndarray:
        return self.raw_contributions * self.scale_factor

    @property
    def current_raw_contribution(self) -> float:
        return float(self.feature.contribution(self.class_index, self.current_value))

    @property
    def current_display_contribution(self) -> float:
        return self.current_raw_contribution * self.scale_factor

    @property
    def labels(self) -> tuple[str, ...]:
        if self.feature.is_discrete:
            return tuple(str(value) for value in self.feature.values)
        return tuple(_format_number(float(value)) for value in self.raw_values)


@dataclass(frozen=True)
class _FooterState:
    total_min_raw: float
    total_max_raw: float
    total_current_raw: float
    current_probability: float
    scale_factor: float
    point_mode: bool

    @property
    def total_min_display(self) -> float:
        return self.total_min_raw * self.scale_factor

    @property
    def total_max_display(self) -> float:
        return self.total_max_raw * self.scale_factor

    @property
    def total_current_display(self) -> float:
        return self.total_current_raw * self.scale_factor


class _ScaleWidgetMixin:
    label_width = 180
    content_margin = 16

    def _content_rect(self) -> QRectF:
        left = self.label_width + self.content_margin
        right = max(left + 40.0, float(self.width()) - 20.0)
        return QRectF(left, 0.0, max(40.0, right - left), float(self.height()))

    @staticmethod
    def _map_value(value: float, start: float, stop: float, rect: QRectF) -> float:
        if abs(stop - start) <= 1e-12:
            return rect.center().x()
        ratio = (value - start) / (stop - start)
        return rect.left() + ratio * rect.width()

    @staticmethod
    def _unmap_value(x: float, start: float, stop: float, rect: QRectF) -> float:
        if rect.width() <= 1e-12:
            return start
        ratio = (x - rect.left()) / rect.width()
        return start + ratio * (stop - start)


class _NomogramHeaderWidget(QWidget, _ScaleWidgetMixin):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._scale_min = -1.0
        self._scale_max = 1.0
        self._point_mode = True
        self.setMinimumHeight(54)

    def set_state(self, scale_min: float, scale_max: float, point_mode: bool) -> None:
        self._scale_min = scale_min
        self._scale_max = scale_max
        self._point_mode = point_mode
        self.update()

    def sizeHint(self) -> QSize:
        return QSize(640, 54)

    def paintEvent(self, _event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), _BG)

        title_rect = QRectF(12.0, 0.0, self.label_width - 24.0, float(self.height()))
        font = QFont(self.font())
        font.setBold(True)
        painter.setFont(font)
        painter.setPen(_LINE)
        painter.drawText(
            title_rect,
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
            i18n.t("Points") if self._point_mode else i18n.t("Log odds ratios"),
        )

        content = self._content_rect()
        baseline_y = content.center().y() + 6.0
        painter.setPen(QPen(_LINE, 1.4))
        painter.drawLine(content.left(), baseline_y, content.right(), baseline_y)

        tick_values = _ruler_values(self._scale_min, self._scale_max, content.width())
        tick_font = QFont(self.font())
        tick_font.setPointSize(max(8, tick_font.pointSize() - 1))
        painter.setFont(tick_font)
        painter.setPen(QPen(_GRID, 1.0))
        for value in tick_values:
            x = self._map_value(float(value), self._scale_min, self._scale_max, content)
            painter.drawLine(x, baseline_y - 6.0, x, baseline_y + 6.0)
            text_rect = QRectF(x - 40.0, baseline_y - 26.0, 80.0, 16.0)
            painter.setPen(_MUTED)
            painter.drawText(text_rect, Qt.AlignmentFlag.AlignCenter, _format_number(float(value)))
            painter.setPen(QPen(_GRID, 1.0))


class _NomogramFeatureRow(QWidget, _ScaleWidgetMixin):
    valueChanged = Signal(str, object)

    def __init__(self, state: _FeatureState, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._state = state
        self._dragging = False
        self._set_row_height()
        self.setMouseTracking(True)

    def set_state(self, state: _FeatureState) -> None:
        self._state = state
        self._set_row_height()
        self.update()

    def sizeHint(self) -> QSize:
        return QSize(640, self.minimumHeight())

    def _set_row_height(self) -> None:
        height = 68
        if not self._state.feature.is_discrete and self._state.numeric_mode == 1:
            height = 82
        self.setMinimumHeight(height)

    def paintEvent(self, _event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), _BG)

        title_font = QFont(self.font())
        title_font.setBold(True)
        painter.setFont(title_font)
        painter.setPen(_LINE)
        label_rect = QRectF(12.0, 2.0, self.label_width - 24.0, 20.0)
        painter.drawText(label_rect, Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, self._state.feature.name)

        painter.setFont(self.font())
        painter.setPen(_MUTED)
        subtitle = str(self._state.current_value) if self._state.feature.is_discrete else _format_number(float(self._state.current_value))
        painter.drawText(
            QRectF(12.0, 22.0, self.label_width - 24.0, 18.0),
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
            subtitle,
        )

        content = self._content_rect()
        baseline_y = 34.0 if self._state.feature.is_discrete or self._state.numeric_mode == 0 else 40.0
        painter.setPen(QPen(_LINE, 1.2))

        if self._state.feature.is_discrete:
            contributions = self._state.display_contributions
            xs = [self._map_value(float(value), self._state.scale_min, self._state.scale_max, content) for value in contributions]
            if xs:
                painter.drawLine(min(xs), baseline_y, max(xs), baseline_y)
            for index, (label, x) in enumerate(zip(self._state.labels, xs)):
                painter.setPen(QPen(_GRID, 1.0))
                painter.drawLine(x, baseline_y - 6.0, x, baseline_y + 6.0)
                painter.setPen(_MUTED)
                text_rect = QRectF(x - 44.0, baseline_y + 10.0 + (index % 2) * 12.0, 88.0, 16.0)
                painter.drawText(text_rect, Qt.AlignmentFlag.AlignCenter, label)
        else:
            values = self._state.display_contributions
            x1 = self._map_value(float(values[0]), self._state.scale_min, self._state.scale_max, content)
            x2 = self._map_value(float(values[-1]), self._state.scale_min, self._state.scale_max, content)
            if self._state.numeric_mode == 0:
                painter.drawLine(x1, baseline_y, x2, baseline_y)
            else:
                path = QPainterPath()
                path.moveTo(x1, baseline_y + 10.0)
                ctrl_dx = (x2 - x1) * 0.35
                path.cubicTo(x1 + ctrl_dx, baseline_y - 16.0, x2 - ctrl_dx, baseline_y + 22.0, x2, baseline_y - 4.0)
                painter.drawPath(path)
            tick_values = _ruler_values(float(self._state.raw_values[0]), float(self._state.raw_values[-1]), content.width(), False)
            for raw_value in tick_values:
                contribution = self._state.feature.contribution(self._state.class_index, float(raw_value)) * self._state.scale_factor
                x = self._map_value(contribution, self._state.scale_min, self._state.scale_max, content)
                painter.setPen(QPen(_GRID, 1.0))
                painter.drawLine(x, baseline_y - 6.0, x, baseline_y + 6.0)
                painter.setPen(_MUTED)
                painter.drawText(QRectF(x - 34.0, baseline_y + 12.0, 68.0, 16.0), Qt.AlignmentFlag.AlignCenter, _format_number(float(raw_value)))

        dot_x = self._map_value(self._state.current_display_contribution, self._state.scale_min, self._state.scale_max, content)
        dot_y = baseline_y
        if not self._state.feature.is_discrete and self._state.numeric_mode == 1:
            x1 = self._map_value(float(self._state.display_contributions[0]), self._state.scale_min, self._state.scale_max, content)
            x2 = self._map_value(float(self._state.display_contributions[-1]), self._state.scale_min, self._state.scale_max, content)
            progress = 0.5 if abs(x2 - x1) <= 1e-12 else (dot_x - x1) / (x2 - x1)
            dot_y = (baseline_y + 10.0) + (-14.0) * np.sin(np.clip(progress, 0.0, 1.0) * np.pi)
        painter.setPen(QPen(QColor("white"), 1.2))
        painter.setBrush(_ACCENT)
        painter.drawEllipse(QRectF(dot_x - 6.0, dot_y - 6.0, 12.0, 12.0))

        painter.setPen(_MUTED)
        painter.drawText(
            QRectF(float(self.width()) - 108.0, 2.0, 96.0, 18.0),
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
            _format_number(self._state.current_display_contribution),
        )

    def mousePressEvent(self, event) -> None:
        if event.button() != Qt.MouseButton.LeftButton:
            return super().mousePressEvent(event)
        self._dragging = True
        self._update_value_from_position(event.position().x())
        event.accept()

    def mouseMoveEvent(self, event) -> None:
        self._show_tooltip(event.pos())
        if self._dragging:
            self._update_value_from_position(event.position().x())
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:
        self._dragging = False
        super().mouseReleaseEvent(event)

    def leaveEvent(self, event) -> None:
        self.setToolTip("")
        super().leaveEvent(event)

    def _update_value_from_position(self, x: float) -> None:
        content = self._content_rect()
        clamped_x = max(content.left(), min(float(x), content.right()))
        display_value = self._unmap_value(clamped_x, self._state.scale_min, self._state.scale_max, content)
        raw_contribution = display_value / self._state.scale_factor if abs(self._state.scale_factor) > 1e-12 else 0.0
        raw_value = self._state.feature.raw_value_from_contribution(self._state.class_index, raw_contribution)
        self._state = _FeatureState(
            feature=self._state.feature,
            class_index=self._state.class_index,
            scale_factor=self._state.scale_factor,
            scale_min=self._state.scale_min,
            scale_max=self._state.scale_max,
            numeric_mode=self._state.numeric_mode,
            current_value=raw_value,
        )
        self.valueChanged.emit(self._state.feature.name, raw_value)
        self.update()

    def _show_tooltip(self, pos: QPoint) -> None:
        content = self._content_rect()
        if not content.adjusted(-8.0, -10.0, 8.0, 16.0).contains(pos):
            return
        display_value = self._unmap_value(pos.x(), self._state.scale_min, self._state.scale_max, content)
        raw_contribution = display_value / self._state.scale_factor if abs(self._state.scale_factor) > 1e-12 else 0.0
        raw_value = self._state.feature.raw_value_from_contribution(self._state.class_index, raw_contribution)
        raw_text = str(raw_value) if self._state.feature.is_discrete else _format_number(float(raw_value))
        self.setToolTip(
            f"{self._state.feature.name}\n"
            f"{i18n.t('Value')}: {raw_text}\n"
            f"{i18n.t('Points')}: {_format_number(display_value)}"
        )


class _NomogramFooterWidget(QWidget, _ScaleWidgetMixin):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._state = _FooterState(-1.0, 1.0, 0.0, 0.5, 1.0, True)
        self.setMinimumHeight(92)

    def set_state(self, state: _FooterState) -> None:
        self._state = state
        self.update()

    def sizeHint(self) -> QSize:
        return QSize(640, 92)

    def paintEvent(self, _event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), _BG)

        content = self._content_rect()
        total_y = 28.0
        prob_y = 62.0

        title_font = QFont(self.font())
        title_font.setBold(True)
        painter.setFont(title_font)
        painter.setPen(_LINE)
        painter.drawText(
            QRectF(12.0, 12.0, self.label_width - 24.0, 20.0),
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
            i18n.t("Total Points") if self._state.point_mode else i18n.t("Total Log Odds"),
        )
        painter.drawText(
            QRectF(12.0, 46.0, self.label_width - 24.0, 20.0),
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
            i18n.t("Probability (%)"),
        )

        painter.setFont(self.font())
        painter.setPen(QPen(_LINE, 1.2))
        painter.drawLine(content.left(), total_y, content.right(), total_y)
        painter.drawLine(content.left(), prob_y, content.right(), prob_y)

        total_ticks = _ruler_values(self._state.total_min_display, self._state.total_max_display, content.width())
        painter.setPen(QPen(_GRID, 1.0))
        for value in total_ticks:
            x = self._map_value(float(value), self._state.total_min_display, self._state.total_max_display, content)
            painter.drawLine(x, total_y - 5.0, x, total_y + 5.0)
            painter.setPen(_MUTED)
            painter.drawText(QRectF(x - 36.0, total_y - 24.0, 72.0, 16.0), Qt.AlignmentFlag.AlignCenter, _format_number(float(value)))
            painter.setPen(QPen(_GRID, 1.0))

        probabilities = np.arange(0.1, 1.0, 0.1)
        for probability in probabilities:
            logit = np.log(probability / (1.0 - probability))
            display = logit * self._state.scale_factor
            if display < self._state.total_min_display or display > self._state.total_max_display:
                continue
            x = self._map_value(display, self._state.total_min_display, self._state.total_max_display, content)
            painter.drawLine(x, prob_y - 5.0, x, prob_y + 5.0)
            painter.setPen(_MUTED)
            painter.drawText(QRectF(x - 22.0, prob_y + 8.0, 44.0, 16.0), Qt.AlignmentFlag.AlignCenter, str(int(round(probability * 100))))
            painter.setPen(QPen(_GRID, 1.0))

        total_x = self._map_value(self._state.total_current_display, self._state.total_min_display, self._state.total_max_display, content)
        painter.setPen(QPen(QColor("white"), 1.2))
        painter.setBrush(_ACCENT)
        painter.drawEllipse(QRectF(total_x - 6.0, total_y - 6.0, 12.0, 12.0))
        painter.setBrush(QColor("#59a14f"))
        painter.drawEllipse(QRectF(total_x - 6.0, prob_y - 6.0, 12.0, 12.0))

        painter.setPen(_MUTED)
        painter.drawText(
            QRectF(float(self.width()) - 126.0, 10.0, 114.0, 18.0),
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
            _format_number(self._state.total_current_display),
        )
        painter.drawText(
            QRectF(float(self.width()) - 126.0, 44.0, 114.0, 18.0),
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
            f"{self._state.current_probability * 100:.1f}%",
        )


class NomogramScreen(QWidget, WorkflowNodeScreenSupport):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._init_workflow_node_support()
        self._generated_datasets = GeneratedDatasetService()
        self._classifier: LogisticRegressionClassifierArtifact | None = None
        self._input_data: DatasetHandle | None = None
        self._marker_values: dict[str, Any] = {}
        self._feature_rows: dict[str, _NomogramFeatureRow] = {}
        self._visible_states: list[_FeatureState] = []
        self._features_dataset: DatasetHandle | None = None
        self._build_ui()
        self._clear_view(i18n.t("Connect a Logistic Regression classifier to display a nomogram."))

    def _build_ui(self) -> None:
        root = QHBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        controls = QFrame(self)
        controls.setFrameShape(QFrame.Shape.StyledPanel)
        controls.setMinimumWidth(280)
        controls.setMaximumWidth(340)
        controls_layout = QVBoxLayout(controls)
        controls_layout.setContentsMargins(14, 14, 14, 14)
        controls_layout.setSpacing(12)

        info_box = QGroupBox(i18n.t("Classifier"), controls)
        info_layout = QVBoxLayout(info_box)
        self._info_label = QLabel(info_box)
        self._info_label.setWordWrap(True)
        info_layout.addWidget(self._info_label)
        controls_layout.addWidget(info_box)

        target_box = QGroupBox(i18n.t("Target Class"), controls)
        target_layout = QFormLayout(target_box)
        self._target_combo = QComboBox(target_box)
        self._target_combo.currentIndexChanged.connect(self._refresh_view)
        target_layout.addRow(i18n.t("Class"), self._target_combo)
        self._normalize_label = QLabel(i18n.t("Normalize probabilities is available only for multiclass models."))
        self._normalize_label.setWordWrap(True)
        self._normalize_label.setProperty("muted", True)
        target_layout.addRow("", self._normalize_label)
        controls_layout.addWidget(target_box)

        display_box = QGroupBox(i18n.t("Display"), controls)
        display_layout = QGridLayout(display_box)
        display_layout.setContentsMargins(10, 10, 10, 10)
        display_layout.setHorizontalSpacing(8)
        display_layout.setVerticalSpacing(10)

        display_layout.addWidget(QLabel(i18n.t("Scale"), display_box), 0, 0)
        self._scale_combo = QComboBox(display_box)
        self._scale_combo.addItems(tuple(i18n.t(text) for text in SCALE_LABELS))
        self._scale_combo.currentIndexChanged.connect(self._refresh_view)
        display_layout.addWidget(self._scale_combo, 0, 1)

        display_layout.addWidget(QLabel(i18n.t("Displayed features"), display_box), 1, 0)
        self._display_combo = QComboBox(display_box)
        self._display_combo.addItems(tuple(i18n.t(text) for text in DISPLAY_LABELS))
        self._display_combo.currentIndexChanged.connect(self._on_display_mode_changed)
        display_layout.addWidget(self._display_combo, 1, 1)

        display_layout.addWidget(QLabel(i18n.t("Best ranked"), display_box), 2, 0)
        self._best_n_spin = QSpinBox(display_box)
        self._best_n_spin.setRange(1, 1000)
        self._best_n_spin.setValue(10)
        self._best_n_spin.valueChanged.connect(self._refresh_view)
        display_layout.addWidget(self._best_n_spin, 2, 1)

        display_layout.addWidget(QLabel(i18n.t("Order"), display_box), 3, 0)
        self._sort_combo = QComboBox(display_box)
        self._sort_combo.addItems(tuple(i18n.t(text) for text in SORT_LABELS))
        self._sort_combo.currentIndexChanged.connect(self._refresh_view)
        display_layout.addWidget(self._sort_combo, 3, 1)

        display_layout.addWidget(QLabel(i18n.t("Numeric features"), display_box), 4, 0)
        self._numeric_combo = QComboBox(display_box)
        self._numeric_combo.addItems(tuple(i18n.t(text) for text in NUMERIC_LABELS))
        self._numeric_combo.currentIndexChanged.connect(self._refresh_view)
        display_layout.addWidget(self._numeric_combo, 4, 1)
        controls_layout.addWidget(display_box)

        self._status_label = QLabel(self)
        self._status_label.setWordWrap(True)
        self._status_label.setProperty("muted", True)
        controls_layout.addWidget(self._status_label)
        controls_layout.addStretch(1)
        root.addWidget(controls, 0)

        plot_host = QFrame(self)
        plot_layout = QVBoxLayout(plot_host)
        plot_layout.setContentsMargins(0, 0, 0, 0)
        plot_layout.setSpacing(0)

        self._scroll_area = QScrollArea(plot_host)
        self._scroll_area.setWidgetResizable(True)
        self._scroll_area.setFrameShape(QFrame.Shape.NoFrame)
        self._scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        plot_layout.addWidget(self._scroll_area, 1)

        self._plot_container = QWidget(self._scroll_area)
        self._plot_layout = QVBoxLayout(self._plot_container)
        self._plot_layout.setContentsMargins(16, 16, 16, 16)
        self._plot_layout.setSpacing(4)
        self._plot_layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        self._header_widget = _NomogramHeaderWidget(self._plot_container)
        self._plot_layout.addWidget(self._header_widget)

        self._rows_container = QWidget(self._plot_container)
        self._rows_layout = QVBoxLayout(self._rows_container)
        self._rows_layout.setContentsMargins(0, 0, 0, 0)
        self._rows_layout.setSpacing(2)
        self._plot_layout.addWidget(self._rows_container)

        self._footer_widget = _NomogramFooterWidget(self._plot_container)
        self._plot_layout.addWidget(self._footer_widget)
        self._plot_layout.addStretch(1)
        self._scroll_area.setWidget(self._plot_container)
        root.addWidget(plot_host, 1)

    def sizeHint(self) -> QSize:
        return QSize(960, 620)

    def set_input_payload(self, payload: WorkflowPayload | None) -> None:
        if payload is None:
            self._classifier = None
            self._input_data = None
            self._marker_values = {}
            self._features_dataset = None
            self._clear_view(i18n.t("Connect a Logistic Regression classifier to display a nomogram."))
            self._notify_output_changed()
            return

        if payload.port_label == "Classifier" and isinstance(payload.value, LogisticRegressionClassifierArtifact):
            self._classifier = payload.value
            self._seed_markers(reset_existing=True)
        elif payload.port_label == "Data" and isinstance(payload.dataset, DatasetHandle):
            self._input_data = payload.dataset
            self._seed_markers(reset_existing=False)
        self._refresh_view()

    def current_output_dataset(self):
        return self._features_dataset

    def documentation_url(self) -> str:
        return "https://orangedatamining.com/widget-catalog/visualize/nomogram/"

    def help_text(self) -> str:
        return i18n.t("Inspect logistic-regression feature contributions and the resulting class probability.")

    def footer_status_text(self) -> str:
        return str(len(self._visible_states))

    def serialize_node_state(self) -> dict[str, object]:
        return {
            "target_index": self._target_combo.currentIndex(),
            "scale_index": self._scale_combo.currentIndex(),
            "display_index": self._display_combo.currentIndex(),
            "best_n": self._best_n_spin.value(),
            "sort_index": self._sort_combo.currentIndex(),
            "numeric_index": self._numeric_combo.currentIndex(),
        }

    def restore_node_state(self, payload: dict[str, object]) -> None:
        self._target_combo.setCurrentIndex(int(payload.get("target_index", 0) or 0))
        self._scale_combo.setCurrentIndex(int(payload.get("scale_index", 0) or 0))
        self._display_combo.setCurrentIndex(int(payload.get("display_index", 0) or 0))
        self._best_n_spin.setValue(int(payload.get("best_n", 10) or 10))
        self._sort_combo.setCurrentIndex(int(payload.get("sort_index", 0) or 0))
        self._numeric_combo.setCurrentIndex(int(payload.get("numeric_index", 0) or 0))

    def _clear_layout(self, layout: QVBoxLayout) -> None:
        while layout.count():
            item = layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

    def _clear_view(self, status: str) -> None:
        self._info_label.setText(i18n.t("No classifier"))
        self._status_label.setText(status)
        self._target_combo.blockSignals(True)
        self._target_combo.clear()
        self._target_combo.blockSignals(False)
        self._clear_layout(self._rows_layout)
        self._header_widget.set_state(-1.0, 1.0, True)
        self._footer_widget.set_state(_FooterState(-1.0, 1.0, 0.0, 0.5, 1.0, True))
        self._visible_states = []
        self._feature_rows = {}
        self._best_n_spin.setEnabled(self._display_combo.currentIndex() == 1)

    def _seed_markers(self, *, reset_existing: bool) -> None:
        if self._classifier is None:
            self._marker_values = {}
            return
        if reset_existing or not self._marker_values:
            self._marker_values = {
                feature.name: feature.default_marker_value()
                for feature in self._classifier.features
            }
        if self._input_data is None or not self._classifier.can_apply_to(self._input_data) or self._input_data.row_count == 0:
            return
        row = self._input_data.dataframe.row(0, named=True)
        for feature in self._classifier.features:
            if feature.name in row and row[feature.name] is not None:
                self._marker_values[feature.name] = row[feature.name]

    def _refresh_target_combo(self) -> None:
        self._target_combo.blockSignals(True)
        previous = self._target_combo.currentText()
        self._target_combo.clear()
        if self._classifier is not None:
            for value in self._classifier.class_values:
                self._target_combo.addItem(str(value))
        index = self._target_combo.findText(previous) if previous else -1
        self._target_combo.setCurrentIndex(max(0, index))
        self._target_combo.blockSignals(False)

    def _refresh_view(self) -> None:
        if self._classifier is None:
            self._features_dataset = None
            self._clear_view(i18n.t("Connect a Logistic Regression classifier to display a nomogram."))
            self._notify_output_changed()
            return

        self._refresh_target_combo()
        target_index = max(0, self._target_combo.currentIndex())
        point_mode = self._scale_combo.currentIndex() == 0
        numeric_mode = self._numeric_combo.currentIndex()
        self._best_n_spin.setEnabled(self._display_combo.currentIndex() == 1)

        feature_order = self._ordered_features(target_index)
        if self._display_combo.currentIndex() == 1:
            feature_order = feature_order[: self._best_n_spin.value()]

        if not feature_order:
            self._features_dataset = None
            self._clear_view(i18n.t("Classifier has no supported features for Nomogram."))
            self._notify_output_changed()
            return

        max_abs = 0.0
        for feature in feature_order:
            values = feature.contribution_values(target_index)[1]
            if values.size:
                max_abs = max(max_abs, float(np.max(np.abs(values))))
        scale_factor = 100.0 / max_abs if point_mode and max_abs > 1e-12 else 1.0

        display_min = float("inf")
        display_max = float("-inf")
        visible_states: list[_FeatureState] = []
        for feature in feature_order:
            state = _FeatureState(
                feature=feature,
                class_index=target_index,
                scale_factor=scale_factor,
                scale_min=0.0,
                scale_max=0.0,
                numeric_mode=numeric_mode,
                current_value=self._marker_values.get(feature.name, feature.default_marker_value()),
            )
            contributions = state.display_contributions
            if contributions.size:
                display_min = min(display_min, float(np.min(contributions)))
                display_max = max(display_max, float(np.max(contributions)))
            visible_states.append(state)

        if not np.isfinite(display_min) or not np.isfinite(display_max) or abs(display_max - display_min) <= 1e-12:
            display_min -= 1.0
            display_max += 1.0

        self._visible_states = [
            _FeatureState(
                feature=state.feature,
                class_index=state.class_index,
                scale_factor=state.scale_factor,
                scale_min=display_min,
                scale_max=display_max,
                numeric_mode=state.numeric_mode,
                current_value=state.current_value,
            )
            for state in visible_states
        ]

        self._info_label.setText(
            i18n.tf(
                "{name}\nTarget: {target}\nFeatures: {count}",
                name=self._classifier.display_name,
                target=self._classifier.target_name,
                count=len(self._classifier.features),
            )
        )
        if self._input_data is not None and not self._classifier.can_apply_to(self._input_data):
            self._status_label.setText(i18n.t("Incoming data does not match classifier features; defaults are used for marker positions."))
        else:
            self._status_label.setText(i18n.t("Drag points along the rulers to inspect how the probability changes."))

        self._header_widget.set_state(display_min, display_max, point_mode)
        self._rebuild_rows()
        self._update_footer()
        self._update_features_output(target_index, point_mode)
        self._notify_output_changed()

    def _ordered_features(self, class_index: int) -> list[LogisticRegressionFeatureArtifact]:
        if self._classifier is None:
            return []
        features = list(self._classifier.features)
        sort_index = self._sort_combo.currentIndex()
        if sort_index == 0:
            return features
        if sort_index == 1:
            return sorted(features, key=lambda feature: feature.name.lower())
        if sort_index == 2:
            return sorted(features, key=lambda feature: (-feature.importance(class_index), feature.name.lower()))
        if sort_index == 3:
            return sorted(
                features,
                key=lambda feature: (-float(np.max(feature.contribution_values(class_index)[1])) if feature.contribution_values(class_index)[1].size else 0.0, feature.name.lower()),
            )
        return sorted(
            features,
            key=lambda feature: (float(np.min(feature.contribution_values(class_index)[1])) if feature.contribution_values(class_index)[1].size else 0.0, feature.name.lower()),
        )

    def _rebuild_rows(self) -> None:
        self._clear_layout(self._rows_layout)
        self._feature_rows = {}
        for state in self._visible_states:
            row = _NomogramFeatureRow(state, self._rows_container)
            row.valueChanged.connect(self._on_row_value_changed)
            self._rows_layout.addWidget(row)
            self._feature_rows[state.feature.name] = row
        self._rows_layout.addStretch(1)

    def _on_row_value_changed(self, feature_name: str, value: Any) -> None:
        self._marker_values[feature_name] = value
        self._update_footer()

    def _update_footer(self) -> None:
        if self._classifier is None or not self._visible_states:
            self._footer_widget.set_state(_FooterState(-1.0, 1.0, 0.0, 0.5, 1.0, True))
            return
        target_index = self._visible_states[0].class_index
        point_mode = self._scale_combo.currentIndex() == 0
        scale_factor = self._visible_states[0].scale_factor

        total_min = float(self._classifier.intercepts[target_index])
        total_max = float(self._classifier.intercepts[target_index])
        for state in self._visible_states:
            contributions = state.raw_contributions
            if contributions.size:
                total_min += float(np.min(contributions))
                total_max += float(np.max(contributions))
        current_logits = self._classifier.logits(self._marker_values)
        probabilities = self._classifier.probabilities(self._marker_values)
        self._footer_widget.set_state(
            _FooterState(
                total_min_raw=total_min,
                total_max_raw=total_max,
                total_current_raw=float(current_logits[target_index]),
                current_probability=float(probabilities[target_index]),
                scale_factor=scale_factor,
                point_mode=point_mode,
            )
        )

    def _update_features_output(self, class_index: int, point_mode: bool) -> None:
        if self._classifier is None or not self._visible_states:
            self._features_dataset = None
            return
        scale_text = i18n.t("Points") if point_mode else i18n.t("Log odds ratios")
        dataframe = pl.DataFrame(
            {
                "Feature": [state.feature.name for state in self._visible_states],
                "Type": [state.feature.logical_type for state in self._visible_states],
                f"Min ({scale_text})": [_format_number(float(np.min(state.display_contributions))) for state in self._visible_states],
                f"Max ({scale_text})": [_format_number(float(np.max(state.display_contributions))) for state in self._visible_states],
                "Importance": [_format_number(state.feature.importance(class_index)) for state in self._visible_states],
            }
        )
        self._features_dataset = self._generated_datasets.build_dataset(
            dataframe,
            dataset_id=f"{self._classifier.classifier_id}-nomogram-features",
            display_name="Nomogram Features",
            file_name="nomogram-features.csv",
            role_overrides={column: "feature" for column in dataframe.columns},
            annotations={"target_class": self._classifier.class_values[class_index], "scale": scale_text},
        )

    def _on_display_mode_changed(self) -> None:
        self._best_n_spin.setEnabled(self._display_combo.currentIndex() == 1)
        self._refresh_view()
