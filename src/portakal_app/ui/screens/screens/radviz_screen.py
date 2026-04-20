from __future__ import annotations

import math
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


class _RadVizWidget(QWidget):
    """RadViz visualization widget.

    Feature anchors are evenly distributed on a circle.
    Each data point is placed at the normalised weighted centre of the anchors.
    The weight of each anchor equals the normalised feature value for that point.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._points: list[tuple[float, float, int]] = []
        self._anchors: list[tuple[str, float]] = []  # (name, angle_rad)
        self._class_labels: list[str] = []
        self.setMinimumHeight(250)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

    def set_projection(
        self,
        points: list[tuple[float, float, int]],
        anchors: list[tuple[str, float]],
        class_labels: list[str],
    ) -> None:
        self._points = points
        self._anchors = anchors
        self._class_labels = class_labels
        self.update()

    def paintEvent(self, _event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), QColor("#fffdf9"))

        if not self._points and not self._anchors:
            painter.setPen(QColor("#8d877d"))
            painter.drawText(
                self.rect(),
                Qt.AlignmentFlag.AlignCenter,
                "No data.\nLoad a dataset with numeric columns.",
            )
            return

        w, h = self.width(), self.height()
        margin = 54
        cx = w / 2
        cy = h / 2
        radius = min(cx, cy) - margin

        # Outer circle
        painter.setPen(QPen(QColor("#d0ccc3"), 1))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawEllipse(int(cx - radius), int(cy - radius), int(radius * 2), int(radius * 2))

        # Inner subtle rings
        painter.setPen(QPen(QColor("#ebe8e1"), 1, Qt.PenStyle.DotLine))
        for r_frac in (0.25, 0.5, 0.75):
            r = int(radius * r_frac)
            painter.drawEllipse(int(cx - r), int(cy - r), r * 2, r * 2)

        # Data points
        for px, py, ci in self._points:
            sx = int(cx + px * radius)
            sy = int(cy - py * radius)
            color = _PALETTE[ci % len(_PALETTE)] if self._class_labels else QColor("#3b82f6")
            dot = QColor(color)
            dot.setAlpha(160)
            painter.setBrush(dot)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawEllipse(sx - 3, sy - 3, 6, 6)

        # Anchors
        for name, angle in self._anchors:
            ax = math.cos(angle)
            ay = math.sin(angle)
            sx = int(cx + ax * radius)
            sy = int(cy - ay * radius)

            painter.setPen(QPen(QColor("#e07020"), 1.5))
            painter.drawLine(int(cx), int(cy), sx, sy)

            painter.setBrush(QColor("#e07020"))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawEllipse(sx - 5, sy - 5, 10, 10)

            lx = int(cx + ax * (radius + 16))
            ly = int(cy - ay * (radius + 16))
            painter.setPen(QColor("#3b2a10"))
            label = name if len(name) <= 10 else name[:9] + "…"
            painter.drawText(lx - 40, ly - 8, 80, 16, Qt.AlignmentFlag.AlignCenter, label)

        # Legend
        if self._class_labels:
            legend_y = h - 18 * len(self._class_labels) - 4
            for i, lbl in enumerate(self._class_labels):
                color = _PALETTE[i % len(_PALETTE)]
                painter.setBrush(color)
                painter.setPen(Qt.PenStyle.NoPen)
                painter.drawEllipse(8, legend_y + i * 18, 10, 10)
                painter.setPen(QColor("#534b40"))
                painter.drawText(22, legend_y + i * 18, 120, 12, Qt.AlignmentFlag.AlignVCenter, lbl)

        painter.end()


class RadvizScreen(QWidget, WorkflowNodeScreenSupport):
    """Radviz – radial visualization of multi-dimensional data."""

    MAX_POINTS = 500

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._init_workflow_node_support()
        self._dataset: DatasetHandle | None = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        header = QLabel("Radviz")
        header.setProperty("sectionTitle", True)
        layout.addWidget(header)

        desc = QLabel(
            "Feature anchors (orange) sit on a circle. "
            "Each data point is placed at the normalised centre of gravity "
            "of its feature values. Points near an anchor have high values for that feature."
        )
        desc.setWordWrap(True)
        desc.setProperty("muted", True)
        layout.addWidget(desc)

        controls_group = QGroupBox("Settings")
        ctrl = QHBoxLayout(controls_group)
        ctrl.addWidget(QLabel("Color by:"))
        self._class_combo = QComboBox()
        self._class_combo.currentTextChanged.connect(self._refresh)
        ctrl.addWidget(self._class_combo, 1)
        ctrl.addWidget(QLabel("Max points:"))
        self._max_spin = QSpinBox()
        self._max_spin.setRange(50, 2000)
        self._max_spin.setValue(self.MAX_POINTS)
        self._max_spin.setSuffix(" pts")
        self._max_spin.valueChanged.connect(self._refresh)
        ctrl.addWidget(self._max_spin)
        layout.addWidget(controls_group)

        chart_group = QGroupBox("Projection")
        chart_layout = QVBoxLayout(chart_group)
        self._chart = _RadVizWidget()
        chart_layout.addWidget(self._chart)
        layout.addWidget(chart_group, 1)

        self._status_label = QLabel("Load a dataset with numeric columns.")
        self._status_label.setProperty("muted", True)
        layout.addWidget(self._status_label)

    def set_input_payload(self, payload) -> None:
        dataset = payload.dataset if payload is not None else None
        self._dataset = dataset
        self._class_combo.blockSignals(True)
        self._class_combo.clear()
        if dataset is not None:
            self._class_combo.addItem("(none)")
            for col in dataset.domain.columns:
                self._class_combo.addItem(col.name)
            for col in dataset.domain.target_columns:
                idx = self._class_combo.findText(col.name)
                if idx >= 0:
                    self._class_combo.setCurrentIndex(idx)
                    break
        self._class_combo.blockSignals(False)
        self._refresh()

    def documentation_url(self) -> str:
        return "https://orangedatamining.com/widget-catalog/visualize/radviz/"

    def _refresh(self) -> None:
        if self._dataset is None:
            self._chart.set_projection([], [], [])
            self._status_label.setText("Load a dataset with numeric columns.")
            return

        df = self._dataset.dataframe
        numeric_cols = [
            col.name for col in self._dataset.domain.columns
            if df[col.name].dtype.is_numeric()
        ]
        if len(numeric_cols) < 2:
            self._chart.set_projection([], [], [])
            self._status_label.setText("Need at least 2 numeric columns.")
            return

        class_col = self._class_combo.currentText()
        if class_col == "(none)":
            class_col = None

        max_pts = self._max_spin.value()
        n = len(df)
        indices = list(range(n))
        if n > max_pts:
            random.seed(42)
            indices = random.sample(indices, max_pts)
            indices.sort()

        # Normalise each numeric column to [0,1]
        col_mins: dict[str, float] = {}
        col_maxs: dict[str, float] = {}
        for col in numeric_cols:
            vals = [float(df[col][i]) for i in indices if df[col][i] is not None]
            mn = min(vals) if vals else 0.0
            mx = max(vals) if vals else 1.0
            col_mins[col] = mn
            col_maxs[col] = mx if mx != mn else mn + 1.0

        # Class mapping
        class_labels: list[str] = []
        class_map: dict[str, int] = {}
        class_series: list[str] | None = None
        if class_col and class_col in df.columns:
            class_series = [str(df[class_col][i]) if df[class_col][i] is not None else "(missing)" for i in indices]
            unique_vals = list(dict.fromkeys(class_series))[:8]
            class_labels = unique_vals
            class_map = {v: i for i, v in enumerate(unique_vals)}

        # Anchor positions on unit circle
        n_features = len(numeric_cols)
        anchors: list[tuple[str, float]] = []
        for i, col in enumerate(numeric_cols):
            angle = 2 * math.pi * i / n_features
            anchors.append((col, angle))

        # RadViz projection: weighted centre of anchors
        points: list[tuple[float, float, int]] = []
        for row_i, i in enumerate(indices):
            weights = []
            for col in numeric_cols:
                raw = df[col][i]
                val = float(raw) if raw is not None else col_mins[col]
                norm = (val - col_mins[col]) / (col_maxs[col] - col_mins[col])
                weights.append(norm)

            total = sum(weights) or 1.0
            px = sum((w / total) * math.cos(anchors[j][1]) for j, w in enumerate(weights))
            py = sum((w / total) * math.sin(anchors[j][1]) for j, w in enumerate(weights))
            ci = class_map.get(class_series[row_i], 0) if class_series else 0
            points.append((px, py, ci))

        self._chart.set_projection(points, anchors, class_labels)
        self._status_label.setText(
            f"{len(points)} points · {n_features} anchors"
            + (f" · colored by '{class_col}'" if class_col else "")
        )
