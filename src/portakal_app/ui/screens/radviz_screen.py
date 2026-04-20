from __future__ import annotations

import math
import random

import numpy as np

from PySide6.QtCore import Qt, QRect, QPoint, Signal
from PySide6.QtGui import QColor, QPainter, QPen, QFont, QFontMetrics
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QSizePolicy,
    QSlider,
    QToolTip,
    QVBoxLayout,
    QWidget,
)

from portakal_app.data.models import DatasetHandle
from portakal_app.ui import i18n
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
    • Scroll-wheel zoom, drag-anchor to move anchor angle, drag-canvas to pan
    """

    # Emitted when user drags an anchor: (anchor_idx, new_angle_rad)
    anchor_moved = Signal(int, float)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._points: list[tuple[float, float, int]] = []   # (px, py, class_idx) in [-1,1]
        self._anchors: list[tuple[str, float]] = []          # (name, angle_rad)
        self._class_labels: list[str] = []
        self._weights: list[list[float]] = []                # per-point normalised weights
        self._feature_names: list[str] = []
        self._dot_rects: list[tuple[QRect, str]] = []

        # View state
        self._zoom: float = 1.0
        self._pan_x: float = 0.0
        self._pan_y: float = 0.0
        self._drag_start: QPoint | None = None
        self._drag_anchor_idx: int | None = None
        self._anchor_screen_pos: list[tuple[int, int]] = []

        # Cached geometry
        self._display_cx: float = 0.0
        self._display_cy: float = 0.0
        self._display_radius: float = 100.0

        # Presentation flags
        self._show_anchors: bool = True
        self._jitter: float = 0.0
        self._point_size: int = 4
        self._opacity: int = 165

        self.setMinimumHeight(360)
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

    def reset_view(self) -> None:
        self._zoom = 1.0
        self._pan_x = 0.0
        self._pan_y = 0.0
        self.update()

    def set_show_anchors(self, show: bool) -> None:
        self._show_anchors = show
        self.update()

    def set_jitter(self, amount: float) -> None:
        self._jitter = amount
        self.update()

    def set_point_size(self, size: int) -> None:
        self._point_size = max(2, size)
        self.update()

    def set_opacity(self, opacity: int) -> None:
        self._opacity = max(30, min(255, opacity))
        self.update()

    # ── Scroll: zoom ──────────────────────────────────────────────────────────

    def wheelEvent(self, event) -> None:
        delta = event.angleDelta().y()
        if delta > 0:
            self._zoom = min(self._zoom * 1.15, 8.0)
        else:
            self._zoom = max(self._zoom / 1.15, 0.2)
        self.update()

    # ── Mouse: anchor drag or pan ─────────────────────────────────────────────

    def _anchor_at(self, pos: QPoint) -> int | None:
        for k, (asx, asy) in enumerate(self._anchor_screen_pos):
            if abs(pos.x() - asx) <= 9 and abs(pos.y() - asy) <= 9:
                return k
        return None

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            pos = event.position().toPoint()
            self._drag_anchor_idx = self._anchor_at(pos)
            if self._drag_anchor_idx is None:
                self._drag_start = pos

    def mouseMoveEvent(self, event) -> None:
        pos = event.position().toPoint()

        if event.buttons() & Qt.MouseButton.LeftButton:
            if self._drag_anchor_idx is not None:
                # Move anchor: compute angle from center
                r = self._display_radius
                if r > 1:
                    dx = (pos.x() - self._display_cx) / r
                    dy = -(pos.y() - self._display_cy) / r
                    new_angle = math.atan2(dy, dx)
                    self.anchor_moved.emit(self._drag_anchor_idx, new_angle)
                QToolTip.hideText()
                return
            if self._drag_start is not None:
                # Pan the view
                self._pan_x += pos.x() - self._drag_start.x()
                self._pan_y += pos.y() - self._drag_start.y()
                self._drag_start = pos
                self.update()
                QToolTip.hideText()
                return

        # Hover tooltip
        for rect, tip in self._dot_rects:
            if rect.contains(pos):
                QToolTip.showText(event.globalPosition().toPoint(), tip, self)
                return
        QToolTip.hideText()

    def mouseReleaseEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag_start = None
            self._drag_anchor_idx = None

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
        self._anchor_screen_pos = []

        canvas_w, canvas_h = self.width(), self.height()
        legend_w = 130 if self._class_labels else 0
        margin = 62
        # Apply pan offset
        cx = (canvas_w - legend_w) / 2 + self._pan_x
        cy = canvas_h / 2 + self._pan_y
        base_radius = min((canvas_w - legend_w) / 2 - margin, canvas_h / 2 - margin)
        radius = base_radius * self._zoom

        # Cache for mouse events
        self._display_cx = cx
        self._display_cy = cy
        self._display_radius = radius

        painter.setFont(QFont(self.font().family(), 8))
        fm = QFontMetrics(painter.font())

        # ── Hint ──────────────────────────────────────────────────────────────
        painter.setPen(QColor(160, 155, 145, 160))
        painter.setFont(QFont(self.font().family(), 7))
        hint = f"scroll=zoom  drag-anchor=move  drag-canvas=pan  {self._zoom:.2f}×"
        painter.drawText(4, canvas_h - 4, hint)
        painter.setFont(QFont(self.font().family(), 8))

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
        r = self._point_size
        rng = random.Random(1)
        for idx, (px, py, ci) in enumerate(self._points):
            # Apply jitter
            jx = px + (rng.random() - 0.5) * self._jitter * 0.15 if self._jitter else px
            jy = py + (rng.random() - 0.5) * self._jitter * 0.15 if self._jitter else py
            sx = int(cx + jx * radius)
            sy = int(cy - jy * radius)
            base_color = _PALETTE[ci % len(_PALETTE)] if self._class_labels else no_class_color
            color = QColor(base_color)
            color.setAlpha(self._opacity)
            painter.setBrush(color)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawEllipse(sx - r, sy - r, r * 2, r * 2)

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
                QRect(sx - r - 2, sy - r - 2, (r + 2) * 2, (r + 2) * 2),
                tip,
            ))

        # ── Anchors ────────────────────────────────────────────────────────────
        for k, (name, angle) in enumerate(self._anchors):
            ax = math.cos(angle)
            ay = math.sin(angle)
            sx = int(cx + ax * radius)
            sy = int(cy - ay * radius)
            self._anchor_screen_pos.append((sx, sy))

            if self._show_anchors:
                painter.setPen(QPen(QColor("#e07020"), 1.5))
                painter.drawLine(int(cx), int(cy), sx, sy)
                painter.setBrush(QColor("#e07020"))
                painter.setPen(Qt.PenStyle.NoPen)
                painter.drawEllipse(sx - 5, sy - 5, 10, 10)
                # Label offset outward
                lx = int(cx + ax * (radius + 18))
                ly = int(cy - ay * (radius + 18))
                painter.setPen(QColor("#3b2a10"))
                lbl = fm.elidedText(name, Qt.TextElideMode.ElideRight, 88)
                painter.drawText(lx - 44, ly - 8, 88, 16, Qt.AlignmentFlag.AlignCenter, lbl)

        # ── Legend ─────────────────────────────────────────────────────────────
        if self._class_labels:
            lx = canvas_w - legend_w + 4
            for i, lbl in enumerate(self._class_labels[:8]):
                ly = 10 + i * 20
                color = _PALETTE[i % len(_PALETTE)]
                painter.setBrush(color)
                painter.setPen(Qt.PenStyle.NoPen)
                painter.drawEllipse(lx, ly + 3, 10, 10)
                painter.setPen(QColor("#534b40"))
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

    Interactions:
    • Scroll wheel         → zoom in / out
    • Drag anchor dot      → move anchor angle around the circle
    • Drag canvas          → pan the view
    """

    MAX_POINTS = 1000

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._init_workflow_node_support()
        self._dataset: DatasetHandle | None = None
        # Cached computed data to support anchor drag without full recompute
        self._W: np.ndarray | None = None              # (n_pts, n_feat) normalized weights
        self._anchor_angles: list[float] = []           # one per feature
        self._class_series: list[str] | None = None
        self._class_map: dict[str, int] = {}
        self._class_labels_cache: list[str] = []
        self._feature_names_cache: list[str] = []
        self._n_pts: int = 0
        self._n_total: int = 0

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
            "Drag an anchor to reposition it · scroll to zoom · hover for weights."
        )
        desc.setWordWrap(True)
        desc.setProperty("muted", True)
        layout.addWidget(desc)

        ctrl_box = QGroupBox("Settings")
        ctrl_vbox = QVBoxLayout(ctrl_box)
        ctrl_vbox.setSpacing(4)

        # Row 1: Color by, Show anchors
        row1 = QHBoxLayout()
        row1.setSpacing(6)
        row1.addWidget(QLabel("Color by:"))
        self._class_combo = QComboBox()
        self._class_combo.currentTextChanged.connect(self._refresh)
        row1.addWidget(self._class_combo, 2)

        self._show_anchors_cb = QCheckBox("Show anchors")
        self._show_anchors_cb.setChecked(True)
        self._show_anchors_cb.stateChanged.connect(
            lambda s: self._chart.set_show_anchors(bool(s)))
        row1.addWidget(self._show_anchors_cb)

        row1.addStretch(1)
        ctrl_vbox.addLayout(row1)

        # Row 2: Jitter, Size, Opacity
        row2 = QHBoxLayout()
        row2.setSpacing(6)

        row2.addWidget(QLabel("Jitter:"))
        self._jitter_slider = QSlider(Qt.Orientation.Horizontal)
        self._jitter_slider.setRange(0, 50)
        self._jitter_slider.setValue(0)
        self._jitter_slider.setMaximumWidth(80)
        self._jitter_slider.setToolTip("Add random jitter to separate overlapping points")
        self._jitter_slider.valueChanged.connect(
            lambda v: self._chart.set_jitter(v / 50.0))
        row2.addWidget(self._jitter_slider)

        row2.addWidget(QLabel("Size:"))
        self._size_slider = QSlider(Qt.Orientation.Horizontal)
        self._size_slider.setRange(2, 12)
        self._size_slider.setValue(4)
        self._size_slider.setMaximumWidth(70)
        row2.addWidget(self._size_slider)

        row2.addWidget(QLabel("Opacity:"))
        self._opacity_slider = QSlider(Qt.Orientation.Horizontal)
        self._opacity_slider.setRange(30, 255)
        self._opacity_slider.setValue(165)
        self._opacity_slider.setMaximumWidth(80)
        row2.addWidget(self._opacity_slider)

        row2.addStretch(1)
        ctrl_vbox.addLayout(row2)
        layout.addWidget(ctrl_box)

        chart_box = QGroupBox("Projection  (scroll=zoom · drag-anchor=move · drag-canvas=pan)")
        chart_layout = QVBoxLayout(chart_box)
        self._chart = _RadVizWidget()
        self._chart.anchor_moved.connect(self._on_anchor_moved)
        # Connect sliders after chart is created
        self._size_slider.valueChanged.connect(self._chart.set_point_size)
        self._opacity_slider.valueChanged.connect(self._chart.set_opacity)
        chart_layout.addWidget(self._chart)
        layout.addWidget(chart_box, 1)

        self._status_label = QLabel(i18n.t("Load a dataset with numeric columns."))
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

    def _on_anchor_moved(self, idx: int, new_angle: float) -> None:
        """Called when user drags an anchor dot. Update angle and redraw."""
        if 0 <= idx < len(self._anchor_angles):
            self._anchor_angles[idx] = new_angle
            self._redraw()

    def _refresh(self) -> None:
        """Full recompute: load data, normalise, set initial anchor angles."""
        if self._dataset is None:
            self._chart.set_projection([], [], [])
            self._status_label.setText(i18n.t("Load a dataset with numeric columns."))
            return

        df = self._dataset.dataframe
        numeric_cols = [
            col.name for col in self._dataset.domain.columns
            if df[col.name].dtype.is_numeric()
        ]
        if len(numeric_cols) < 2:
            self._chart.set_projection([], [], [])
            self._status_label.setText(i18n.t("Need at least 2 numeric columns."))
            return

        class_col = self._class_combo.currentText()
        if class_col == "(none)":
            class_col = None

        n = len(df)
        indices = list(range(n))
        if n > self.MAX_POINTS:
            random.seed(42)
            indices = random.sample(indices, self.MAX_POINTS)
            indices.sort()

        n_pts = len(indices)
        n_feat = len(numeric_cols)

        if n_pts == 0:
            self._chart.set_projection([], [], [])
            self._status_label.setText(i18n.t("No data rows."))
            return

        # ── Build feature matrix with numpy ───────────────────────────────────
        X = np.zeros((n_pts, n_feat), dtype=np.float64)
        for k, col in enumerate(numeric_cols):
            for row_i, i in enumerate(indices):
                v = df[col][i]
                X[row_i, k] = float(v) if v is not None else 0.0

        col_min = X.min(axis=0)
        col_rng = X.max(axis=0) - col_min
        col_rng[col_rng < 1e-10] = 1.0
        X = (X - col_min) / col_rng

        row_sums = X.sum(axis=1, keepdims=True)
        row_sums[row_sums < 1e-10] = 1.0
        W = X / row_sums

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

        # ── Initial anchor angles: equally spaced starting at π/2 (top) ──────
        angles = [math.pi / 2 - 2 * math.pi * i / n_feat for i in range(n_feat)]

        # Cache for anchor-drag redraws
        self._W = W
        self._anchor_angles = angles[:]
        self._class_series = class_series
        self._class_map = class_map
        self._class_labels_cache = class_labels
        self._feature_names_cache = numeric_cols
        self._n_pts = n_pts
        self._n_total = n

        self._redraw()

    def _redraw(self) -> None:
        """Reproject using current anchor angles (called after drag or full refresh)."""
        if self._W is None:
            return

        n_feat = len(self._anchor_angles)
        anchor_coords = np.array(
            [[math.cos(a), math.sin(a)] for a in self._anchor_angles],
            dtype=np.float64,
        )
        P = self._W @ anchor_coords

        anchors_list: list[tuple[str, float]] = list(
            zip(self._feature_names_cache, self._anchor_angles)
        )

        points: list[tuple[float, float, int]] = [
            (float(P[row_i, 0]),
             float(P[row_i, 1]),
             self._class_map.get(self._class_series[row_i], 0)
             if self._class_series else 0)
            for row_i in range(self._n_pts)
        ]
        all_weights: list[list[float]] = [
            self._W[row_i].tolist() for row_i in range(self._n_pts)
        ]

        self._chart.set_projection(
            points, anchors_list, self._class_labels_cache,
            weights=all_weights, feature_names=self._feature_names_cache,
        )

        class_col = self._class_combo.currentText()
        sampled = (f" (sampled {self.MAX_POINTS} of {self._n_total})"
                   if self._n_total > self.MAX_POINTS else "")
        self._status_label.setText(
            f"{self._n_pts} points · {n_feat} anchors{sampled}"
            + (f" · colored by '{class_col}'"
               if class_col and class_col != "(none)" else "")
        )
