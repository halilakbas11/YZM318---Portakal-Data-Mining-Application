from __future__ import annotations

import random

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QPainter, QPen
from PySide6.QtWidgets import (
    QComboBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QSizePolicy,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from portakal_app.data.models import DatasetHandle
from portakal_app.ui.screens.node_screen import WorkflowNodeScreenSupport

_PALETTE = [
    QColor("#e07020"), QColor("#3b82f6"), QColor("#22c55e"),
    QColor("#a855f7"), QColor("#f43f5e"), QColor("#0ea5e9"),
    QColor("#f59e0b"), QColor("#10b981"),
]


class _ScatterWidget(QWidget):
    """Generic 2D scatter plot used for Linear Projection."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        # (x_norm, y_norm, class_idx) all in [0, 1]
        self._points: list[tuple[float, float, int]] = []
        self._x_label = "X"
        self._y_label = "Y"
        self._class_labels: list[str] = []
        self.setMinimumHeight(250)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

    def set_data(
        self,
        points: list[tuple[float, float, int]],
        x_label: str,
        y_label: str,
        class_labels: list[str],
    ) -> None:
        self._points = points
        self._x_label = x_label
        self._y_label = y_label
        self._class_labels = class_labels
        self.update()

    def paintEvent(self, _event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), QColor("#fffdf9"))

        if not self._points:
            painter.setPen(QColor("#8d877d"))
            painter.drawText(
                self.rect(),
                Qt.AlignmentFlag.AlignCenter,
                "No data.\nLoad a dataset with at least 2 numeric columns.",
            )
            return

        w, h = self.width(), self.height()
        margin_l, margin_r = 50, 16
        margin_t, margin_b = 12, 48

        cw = w - margin_l - margin_r
        ch = h - margin_t - margin_b
        if cw < 10 or ch < 10:
            return

        # Draw grid
        painter.setPen(QPen(QColor("#e0ddd6"), 1, Qt.PenStyle.DotLine))
        for i in range(1, 5):
            gx = margin_l + int(cw * i / 4)
            gy = margin_t + int(ch * i / 4)
            painter.drawLine(gx, margin_t, gx, margin_t + ch)
            painter.drawLine(margin_l, gy, margin_l + cw, gy)

        # Draw axes
        painter.setPen(QPen(QColor("#9b9488"), 1))
        painter.drawRect(margin_l, margin_t, cw, ch)

        # Draw points
        for px, py, ci in self._points:
            sx = margin_l + int(px * cw)
            sy = margin_t + int((1 - py) * ch)
            color = _PALETTE[ci % len(_PALETTE)]
            dot = QColor(color)
            dot.setAlpha(180)
            painter.setBrush(dot)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawEllipse(sx - 3, sy - 3, 6, 6)

        # Axis labels
        painter.setPen(QColor("#534b40"))
        painter.drawText(margin_l, h - margin_b + 4, cw, 16,
                         Qt.AlignmentFlag.AlignCenter, self._x_label)

        # Y label (rotated)
        painter.save()
        painter.translate(12, margin_t + ch / 2)
        painter.rotate(-90)
        painter.drawText(-40, -6, 80, 14, Qt.AlignmentFlag.AlignCenter, self._y_label)
        painter.restore()

        # Legend
        if self._class_labels:
            for i, lbl in enumerate(self._class_labels):
                lx = margin_l + 6 + i * 100
                if lx + 90 > w:
                    break
                color = _PALETTE[i % len(_PALETTE)]
                painter.setBrush(color)
                painter.setPen(Qt.PenStyle.NoPen)
                painter.drawEllipse(lx, h - 18, 10, 10)
                painter.setPen(QColor("#534b40"))
                painter.drawText(lx + 14, h - 20, 84, 14, Qt.AlignmentFlag.AlignVCenter, lbl[:10])

        painter.end()


class LinearProjectionScreen(QWidget, WorkflowNodeScreenSupport):
    """Linear Projection – scatter plot of two user-selected numeric features."""

    MAX_POINTS = 1000

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._init_workflow_node_support()
        self._dataset: DatasetHandle | None = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        header = QLabel("Linear Projection")
        header.setProperty("sectionTitle", True)
        layout.addWidget(header)

        desc = QLabel(
            "Project data onto two chosen numeric axes. "
            "Select X and Y features below to explore relationships between them."
        )
        desc.setWordWrap(True)
        desc.setProperty("muted", True)
        layout.addWidget(desc)

        controls_group = QGroupBox("Axes")
        ctrl = QHBoxLayout(controls_group)
        ctrl.addWidget(QLabel("X axis:"))
        self._x_combo = QComboBox()
        self._x_combo.currentTextChanged.connect(self._refresh)
        ctrl.addWidget(self._x_combo, 1)
        ctrl.addWidget(QLabel("Y axis:"))
        self._y_combo = QComboBox()
        self._y_combo.currentTextChanged.connect(self._refresh)
        ctrl.addWidget(self._y_combo, 1)
        ctrl.addWidget(QLabel("Color by:"))
        self._class_combo = QComboBox()
        self._class_combo.currentTextChanged.connect(self._refresh)
        ctrl.addWidget(self._class_combo, 1)
        ctrl.addWidget(QLabel("Max pts:"))
        self._max_spin = QSpinBox()
        self._max_spin.setRange(50, 5000)
        self._max_spin.setValue(self.MAX_POINTS)
        self._max_spin.valueChanged.connect(self._refresh)
        ctrl.addWidget(self._max_spin)
        layout.addWidget(controls_group)

        chart_group = QGroupBox("Projection")
        chart_layout = QVBoxLayout(chart_group)
        self._chart = _ScatterWidget()
        chart_layout.addWidget(self._chart)
        layout.addWidget(chart_group, 1)

        self._status_label = QLabel("Load a dataset with numeric columns.")
        self._status_label.setProperty("muted", True)
        layout.addWidget(self._status_label)

    def set_input_payload(self, payload) -> None:
        dataset = payload.dataset if payload is not None else None
        self._dataset = dataset
        for combo in (self._x_combo, self._y_combo, self._class_combo):
            combo.blockSignals(True)
            combo.clear()
        if dataset is not None:
            df = dataset.dataframe
            numeric_cols = [col.name for col in dataset.domain.columns if df[col.name].dtype.is_numeric()]
            all_cols = [col.name for col in dataset.domain.columns]
            self._x_combo.addItems(numeric_cols)
            self._y_combo.addItems(numeric_cols)
            self._class_combo.addItem("(none)")
            self._class_combo.addItems(all_cols)
            if len(numeric_cols) >= 2:
                self._y_combo.setCurrentIndex(1)
            # Prefer target column for colour
            for col in dataset.domain.target_columns:
                idx = self._class_combo.findText(col.name)
                if idx >= 0:
                    self._class_combo.setCurrentIndex(idx)
                    break
        for combo in (self._x_combo, self._y_combo, self._class_combo):
            combo.blockSignals(False)
        self._refresh()

    def documentation_url(self) -> str:
        return "https://orangedatamining.com/widget-catalog/visualize/linearprojection/"

    def _refresh(self) -> None:
        if self._dataset is None:
            self._chart.set_data([], "X", "Y", [])
            self._status_label.setText("Load a dataset with numeric columns.")
            return

        x_col = self._x_combo.currentText()
        y_col = self._y_combo.currentText()
        if not x_col or not y_col:
            self._chart.set_data([], "X", "Y", [])
            self._status_label.setText("Select X and Y columns.")
            return

        df = self._dataset.dataframe
        try:
            x_vals = df[x_col].to_list()
            y_vals = df[y_col].to_list()
        except Exception:
            self._chart.set_data([], x_col, y_col, [])
            self._status_label.setText("Column not found.")
            return

        class_col = self._class_combo.currentText()
        class_series = None
        if class_col and class_col != "(none)" and class_col in df.columns:
            class_series = [str(v) if v is not None else "(missing)" for v in df[class_col].to_list()]

        max_pts = self._max_spin.value()
        n = len(df)
        indices = list(range(n))
        if n > max_pts:
            random.seed(42)
            indices = random.sample(indices, max_pts)
            indices.sort()

        # Build class map
        class_labels: list[str] = []
        class_map: dict[str, int] = {}
        if class_series is not None:
            unique_vals = list(dict.fromkeys(class_series[i] for i in indices))[:8]
            class_labels = unique_vals
            class_map = {v: i for i, v in enumerate(unique_vals)}

        # Normalize X and Y
        x_floats = [float(x_vals[i]) for i in indices if x_vals[i] is not None]
        y_floats = [float(y_vals[i]) for i in indices if y_vals[i] is not None]
        x_min, x_max = (min(x_floats), max(x_floats)) if x_floats else (0.0, 1.0)
        y_min, y_max = (min(y_floats), max(y_floats)) if y_floats else (0.0, 1.0)
        if x_max == x_min:
            x_max = x_min + 1.0
        if y_max == y_min:
            y_max = y_min + 1.0

        points: list[tuple[float, float, int]] = []
        for i in indices:
            xv = x_vals[i]
            yv = y_vals[i]
            if xv is None or yv is None:
                continue
            xn = (float(xv) - x_min) / (x_max - x_min)
            yn = (float(yv) - y_min) / (y_max - y_min)
            ci = class_map.get(class_series[i], 0) if class_series else 0
            points.append((xn, yn, ci))

        self._chart.set_data(points, x_col, y_col, class_labels)
        self._status_label.setText(
            f"X: '{x_col}'  |  Y: '{y_col}'  |  {len(points)} points"
            + (f"  |  colored by '{class_col}'" if class_col and class_col != "(none)" else "")
        )
