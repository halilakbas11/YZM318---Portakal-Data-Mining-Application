from __future__ import annotations

import math
import random

import numpy as np

from PySide6.QtCore import Qt, QRect
from PySide6.QtGui import QColor, QPainter, QPen, QFont, QFontMetrics
from PySide6.QtWidgets import (
    QComboBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QSizePolicy,
    QSpinBox,
    QToolTip,
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

_POINT_R = 4


# ── Canvas widget ──────────────────────────────────────────────────────────────

class _RadVizWidget(QWidget):
    """
    RadViz canvas (Hoffman et al., 1997).

    • Feature anchors evenly spaced on a unit circle, starting at top (π/2)
    • Point placed at normalised weighted centre of anchors:
          pos = Σ(w_i · anchor_i) / Σw_i
    • Guide rings at 0.25, 0.5, 0.75 radius
    • Per-class colour legend
    • Hover tooltip: top-3 anchor weights and class
    • Font-metric–based label truncation for anchors and legend
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._points: list[tuple[float, float, int]] = []   # (px, py, class_idx) in [-1,1]
        self._anchors: list[tuple[str, float]] = []          # (name, angle_rad)
        self._class_labels: list[str] = []
        self._weights: list[list[float]] = []                # per-point normalised weights
        self._feature_names: list[str] = []
        self._dot_rects: list[tuple[QRect, str]] = []
        self.setMinimumHeight(260)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMouseTracking(True)

    def set_projection(
        self,
        points: list[tuple[float, float, int]],
        anchors: list[tuple[str, float]],
        class_labels: list[str],
        weights: list[list[float]] | None = None,
        feature_names: list[str] | None = None,
    ) -> None:
        self._points = points
        self._anchors = anchors
        self._class_labels = class_labels
        self._weights = weights or []
        self._feature_names = feature_names or []
        self._dot_rects = []
        self.update()

    # ── Tooltip ────────────────────────────────────────────────────────────────

    def mouseMoveEvent(self, event) -> None:
        pos = event.position().toPoint()
        for rect, tip in self._dot_rects:
            if rect.contains(pos):
                QToolTip.showText(event.globalPosition().toPoint(), tip, self)
                return
        QToolTip.hideText()

    def leaveEvent(self, _event) -> None:
        QToolTip.hideText()

    # ── Drawing ────────────────────────────────────────────────────────────────

    def paintEvent(self, _event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), QColor("#fffdf9"))

        if not self._points and not self._anchors:
            painter.setPen(QColor("#8d877d"))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter,
                             "No data.\nLoad a dataset with at least 2 numeric columns.")
            return

        self._dot_rects = []

        w, h = self.width(), self.height()
        legend_w = 130 if self._class_labels else 0
        margin = 62
        cx = (w - legend_w) / 2
        cy = h / 2
        radius = min(cx - margin, cy - margin)

        painter.setFont(QFont(self.font().family(), 8))
        fm = QFontMetrics(painter.font())

        # ── Guide rings ────────────────────────────────────────────────────────
        for frac, alpha in ((0.25, 40), (0.5, 70), (0.75, 100)):
            r = int(radius * frac)
            painter.setPen(QPen(QColor(180, 174, 162, alpha), 1, Qt.PenStyle.DotLine))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawEllipse(int(cx - r), int(cy - r), r * 2, r * 2)

        # ── Outer circle ───────────────────────────────────────────────────────
        painter.setPen(QPen(QColor("#c0bbb2"), 1))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawEllipse(int(cx - radius), int(cy - radius),
                            int(radius * 2), int(radius * 2))

        # ── Data points ────────────────────────────────────────────────────────
        no_class_color = QColor("#3b82f6")
        for idx, (px, py, ci) in enumerate(self._points):
            sx = int(cx + px * radius)
            sy = int(cy - py * radius)
            base_color = _PALETTE[ci % len(_PALETTE)] if self._class_labels else no_class_color
            color = QColor(base_color)
            color.setAlpha(165)
            painter.setBrush(color)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawEllipse(sx - _POINT_R, sy - _POINT_R, _POINT_R * 2, _POINT_R * 2)

            # Tooltip: class + top-3 feature weights
            lbl = self._class_labels[ci] if ci < len(self._class_labels) else str(ci)
            if idx < len(self._weights) and self._feature_names:
                top3 = sorted(
                    zip(self._feature_names, self._weights[idx]),
                    key=lambda x: -x[1],
                )[:3]
                w_str = "<br>".join(f"{n}: {v:.3f}" for n, v in top3)
                tip = f"<b>Class: {lbl}</b><br>Top weights:<br>{w_str}"
            else:
                tip = f"Class: {lbl}<br>x: {px:.3f}  y: {py:.3f}"
            self._dot_rects.append((
                QRect(sx - _POINT_R - 2, sy - _POINT_R - 2,
                      (_POINT_R + 2) * 2, (_POINT_R + 2) * 2),
                tip,
            ))

        # ── Anchors ────────────────────────────────────────────────────────────
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

            # Label — font-metrics truncation, offset outward from circle edge
            lx = int(cx + ax * (radius + 18))
            ly = int(cy - ay * (radius + 18))
            painter.setPen(QColor("#3b2a10"))
            lbl = fm.elidedText(name, Qt.TextElideMode.ElideRight, 88)
            painter.drawText(lx - 44, ly - 8, 88, 16, Qt.AlignmentFlag.AlignCenter, lbl)

        # ── Legend ─────────────────────────────────────────────────────────────
        if self._class_labels:
            lx = w - legend_w + 4
            for i, lbl in enumerate(self._class_labels[:8]):
                ly = 10 + i * 20
                color = _PALETTE[i % len(_PALETTE)]
                painter.setBrush(color)
                painter.setPen(Qt.PenStyle.NoPen)
                painter.drawEllipse(lx, ly + 3, 10, 10)
                painter.setPen(QColor("#534b40"))
                # Font-metrics truncation for legend labels
                lbl_t = fm.elidedText(lbl, Qt.TextElideMode.ElideRight, legend_w - 20)
                painter.drawText(lx + 14, ly, legend_w - 18, 16,
                                 Qt.AlignmentFlag.AlignVCenter, lbl_t)

        painter.end()


# ── Screen widget ──────────────────────────────────────────────────────────────

class RadvizScreen(QWidget, WorkflowNodeScreenSupport):
    """
    RadViz – Radial Visualization (Hoffman et al., 1997).

    Feature anchors are fixed evenly on a circle, starting at top (π/2).
    Each point is positioned at the normalised weighted centre of its anchor values:
        pos = Σ(w_i · a_i) / Σw_i

    Numpy-vectorized data loading and projection for speed and clarity.
    Points near an anchor have high normalised values for that feature.
    """

    MAX_POINTS = 500

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._init_workflow_node_support()
        self._dataset: DatasetHandle | None = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        header = QLabel("Radviz")
        header.setProperty("sectionTitle", True)
        layout.addWidget(header)

        desc = QLabel(
            "Feature anchors (orange) on a circle. "
            "Each point sits at the normalised weighted centre of its feature values. "
            "Points near an anchor have high values for that feature. "
            "Hover for per-feature weights."
        )
        desc.setWordWrap(True)
        desc.setProperty("muted", True)
        layout.addWidget(desc)

        ctrl_box = QGroupBox("Settings")
        ctrl = QHBoxLayout(ctrl_box)
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
        layout.addWidget(ctrl_box)

        chart_box = QGroupBox("Projection")
        chart_layout = QVBoxLayout(chart_box)
        self._chart = _RadVizWidget()
        chart_layout.addWidget(self._chart)
        layout.addWidget(chart_box, 1)

        self._status_label = QLabel("Load a dataset with numeric columns.")
        self._status_label.setProperty("muted", True)
        layout.addWidget(self._status_label)

    # ── WorkflowNodeScreenSupport ─────────────────────────────────────────────

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

    # ── Internal ──────────────────────────────────────────────────────────────

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

        n_pts = len(indices)
        n_feat = len(numeric_cols)

        if n_pts == 0:
            self._chart.set_projection([], [], [])
            self._status_label.setText("No data rows.")
            return

        # ── Build feature matrix with numpy ───────────────────────────────────
        X = np.zeros((n_pts, n_feat), dtype=np.float64)
        for k, col in enumerate(numeric_cols):
            for row_i, i in enumerate(indices):
                v = df[col][i]
                X[row_i, k] = float(v) if v is not None else 0.0

        # Min-max normalise each column to [0, 1]
        col_min = X.min(axis=0)
        col_rng = X.max(axis=0) - col_min
        col_rng[col_rng < 1e-10] = 1.0   # constant columns → avoid div-by-zero
        X = (X - col_min) / col_rng       # (n_pts, n_feat) in [0, 1]

        # ── Row-normalise: w_i = x_i / Σx_i (RadViz formula) ─────────────────
        row_sums = X.sum(axis=1, keepdims=True)
        row_sums[row_sums < 1e-10] = 1.0  # all-zero rows → centre of circle
        W = X / row_sums                   # (n_pts, n_feat)

        # ── Anchor angles: equally spaced starting at π/2 (top) ──────────────
        angles = [math.pi / 2 - 2 * math.pi * i / n_feat for i in range(n_feat)]
        anchors_list: list[tuple[str, float]] = list(zip(numeric_cols, angles))
        anchor_coords = np.array(                   # (n_feat, 2)
            [[math.cos(a), math.sin(a)] for a in angles],
            dtype=np.float64,
        )

        # ── Project: P = W @ anchor_coords  (n_pts, 2) ───────────────────────
        P = W @ anchor_coords

        # ── Class mapping ─────────────────────────────────────────────────────
        class_labels: list[str] = []
        class_map: dict[str, int] = {}
        class_series: list[str] | None = None
        if class_col and class_col in df.columns:
            class_series = [
                str(df[class_col][i]) if df[class_col][i] is not None else "(missing)"
                for i in indices
            ]
            unique = list(dict.fromkeys(class_series))[:8]
            class_labels = unique
            class_map = {v: idx for idx, v in enumerate(unique)}

        # ── Build output lists ────────────────────────────────────────────────
        points: list[tuple[float, float, int]] = [
            (float(P[row_i, 0]),
             float(P[row_i, 1]),
             class_map.get(class_series[row_i], 0) if class_series else 0)
            for row_i in range(n_pts)
        ]
        all_weights: list[list[float]] = [
            W[row_i].tolist() for row_i in range(n_pts)
        ]

        self._chart.set_projection(
            points, anchors_list, class_labels,
            weights=all_weights, feature_names=numeric_cols,
        )

        sampled = f" (sampled {max_pts} of {n})" if n > max_pts else ""
        self._status_label.setText(
            f"{n_pts} points · {n_feat} anchors{sampled}"
            + (f" · colored by '{class_col}'" if class_col else "")
        )
