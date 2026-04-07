from __future__ import annotations

import math
import random

import numpy as np

from PySide6.QtCore import Qt, QRect, QPoint, QTimer, Signal
from PySide6.QtGui import QColor, QPainter, QPen, QFont, QFontMetrics
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
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


# ── FreeViz optimisation — RadViz projection + force-directed anchors ─────────
#
# KEY: Uses the same weighted-centre projection as RadViz (Hoffman et al., 1997)
#       P = W @ anchors    where W[i,k] = x_ik / Σ_k x_ik  (row-normalised)
# The optimization (Demsar et al., 2005) moves the anchor positions on the unit
# circle to maximise class separation, using the same-class attract / diff-class
# repel physics. This is exactly how Orange's OWFreeViz works.

def _freeviz_optimize(
    W: np.ndarray,        # (n_pts, n_feat)  row-normalised weights, each row sums to 1
    labels: np.ndarray,   # (n_pts,)          int class indices
    anchors: np.ndarray,  # (n_feat, 2)       unit vectors on circle
    n_iter: int = 10,
    learning_rate: float = 0.01,
) -> tuple[np.ndarray, float]:
    """
    FreeViz gradient descent — O(n²) per iteration, fully vectorized.

    Projection:  P = W @ anchors           shape (n_pts, 2)
    Force i→j:   same class  →  F = +dist     (attractive towards each other)
                 diff class  →  F = -1/dist²  (repulsive away from each other)
    Gradient:    grad[k] = W[:, k] · net_force_per_point   shape (n_feat, 2)
    Update:      anchors -= lr · grad
    Normalise:   anchors → unit circle

    Returns (new_anchors, stress).
    """
    n_pts, n_feat = W.shape
    if n_pts < 2 or n_feat < 2:
        return anchors, 0.0

    for _ in range(n_iter):
        P = W @ anchors   # (n_pts, 2)

        # Pairwise differences in projected space
        dP = P[:, np.newaxis, :] - P[np.newaxis, :, :]  # (n, n, 2): P_i - P_j
        dist = np.sqrt((dP ** 2).sum(axis=2))            # (n, n)
        dist = np.maximum(dist, 1e-8)

        same = labels[:, np.newaxis] == labels[np.newaxis, :]  # (n, n) bool

        # Force magnitude (scalar per pair):
        # same class: positive → force vector points along (P_i-P_j) which is attractive
        # diff class: negative → force vector reverses, pulling i toward j (repulsive pushes apart)
        force = np.where(same, dist, -1.0 / (dist * dist))  # (n, n)
        np.fill_diagonal(force, 0.0)

        unit = dP / dist[:, :, np.newaxis]             # (n, n, 2) unit vectors
        force_vec = force[:, :, np.newaxis] * unit     # (n, n, 2)

        # Net force on each point i: Σ_j force_vec[i, j]
        net_force = force_vec.sum(axis=1)              # (n_pts, 2)

        # Gradient for anchor k:  how much anchor k contributes to each point's force
        # ∂pos_i / ∂anchor_k = W[i, k]  →  grad[k] = Σ_i W[i,k] * net_force[i]
        grad = W.T @ net_force                         # (n_feat, 2)

        anchors = anchors - learning_rate * grad

        # Keep anchors on the unit circle
        norms = np.linalg.norm(anchors, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-8)
        anchors = anchors / norms

    # Stress = mean |force| over all pairs (convergence indicator)
    P = W @ anchors
    dP = P[:, np.newaxis, :] - P[np.newaxis, :, :]
    dist = np.sqrt((dP ** 2).sum(axis=2))
    dist = np.maximum(dist, 1e-8)
    same = labels[:, np.newaxis] == labels[np.newaxis, :]
    force = np.where(same, dist, -1.0 / (dist * dist))
    np.fill_diagonal(force, 0.0)
    n_pairs = max(1, n_pts * (n_pts - 1))
    stress = float(np.abs(force).sum()) / n_pairs

    return anchors, stress


# ── Canvas widget ──────────────────────────────────────────────────────────────

class _FreeVizWidget(QWidget):
    """
    FreeViz projection canvas.

    • Points placed at weighted-centre of anchors (RadViz formula) → always INSIDE circle
    • Anchors on unit circle, draggable to explore different projections
    • Scroll to zoom · drag-canvas to pan · drag-anchor to move it

    Matching Orange's OWFreeViz interaction model.
    """

    # Emitted when user drags an anchor: (anchor_idx, new_ax, new_ay) in data coords
    anchor_moved = Signal(int, float, float)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._points: list[tuple[float, float, int]] = []   # (px, py, class_idx) in [-1,1]
        self._anchors: list[tuple[str, float, float]] = []  # (name, ax, ay) unit vectors
        self._class_labels: list[str] = []
        self._weights: list[list[float]] = []               # per-point normalised weights
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
        self._point_size: int = 5
        self._opacity: int = 170

        self.setMinimumHeight(340)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMouseTracking(True)

    def set_projection(
        self,
        points: list[tuple[float, float, int]],
        anchors: list[tuple[str, float, float]],
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

    # ── Scroll zoom ───────────────────────────────────────────────────────────

    def wheelEvent(self, event) -> None:
        delta = event.angleDelta().y()
        self._zoom = min(self._zoom * 1.15, 8.0) if delta > 0 else max(self._zoom / 1.15, 0.2)
        self.update()

    # ── Mouse: anchor drag or pan ─────────────────────────────────────────────

    def _anchor_at(self, pos: QPoint) -> int | None:
        for k, (asx, asy) in enumerate(self._anchor_screen_pos):
            if abs(pos.x() - asx) <= 10 and abs(pos.y() - asy) <= 10:
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
                # Compute new anchor position as unit vector from canvas centre
                r = self._display_radius
                if r > 1:
                    dx = (pos.x() - self._display_cx) / r
                    dy = -(pos.y() - self._display_cy) / r  # y-flip (screen y downward)
                    norm = math.sqrt(dx * dx + dy * dy)
                    if norm > 1e-6:
                        # Emit new unit-vector anchor position
                        self.anchor_moved.emit(self._drag_anchor_idx, dx / norm, dy / norm)
                QToolTip.hideText()
                return
            if self._drag_start is not None:
                # Pan
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

    def mouseDoubleClickEvent(self, _event) -> None:
        self.reset_view()

    def leaveEvent(self, _event) -> None:
        QToolTip.hideText()

    # ── Drawing ────────────────────────────────────────────────────────────────

    def paintEvent(self, _event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), QColor("#fffdf9"))

        if not self._anchors:
            painter.setPen(QColor("#8d877d"))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter,
                             "No data.\nLoad a dataset with at least 2 numeric columns\n"
                             "and a class variable.")
            return

        self._dot_rects = []
        self._anchor_screen_pos = []

        canvas_w, canvas_h = self.width(), self.height()
        legend_w = 130 if self._class_labels else 0
        margin = 62
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
            r_ring = int(radius * frac)
            painter.setPen(QPen(QColor(180, 174, 162, alpha), 1, Qt.PenStyle.DotLine))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawEllipse(int(cx - r_ring), int(cy - r_ring), r_ring * 2, r_ring * 2)

        # ── Outer circle ───────────────────────────────────────────────────────
        painter.setPen(QPen(QColor("#c0bbb2"), 1))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawEllipse(int(cx - radius), int(cy - radius),
                            int(radius * 2), int(radius * 2))

        # ── Data points ────────────────────────────────────────────────────────
        no_class_color = QColor("#3b82f6")
        r = self._point_size
        rng = random.Random(0)
        for idx, (px, py, ci) in enumerate(self._points):
            jx = px + (rng.random() - 0.5) * self._jitter * 0.15 if self._jitter else px
            jy = py + (rng.random() - 0.5) * self._jitter * 0.15 if self._jitter else py
            sx = int(cx + jx * radius)
            sy = int(cy - jy * radius)   # y-flip: data-y up = screen-y down
            base_color = _PALETTE[ci % len(_PALETTE)] if self._class_labels else no_class_color
            color = QColor(base_color)
            color.setAlpha(self._opacity)
            painter.setBrush(color)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawEllipse(sx - r, sy - r, r * 2, r * 2)

            # Tooltip
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
                QRect(sx - r - 2, sy - r - 2, (r + 2) * 2, (r + 2) * 2), tip,
            ))

        # ── Anchors (vectors from centre to unit circle) ───────────────────────
        for k, (name, ax, ay) in enumerate(self._anchors):
            sx = int(cx + ax * radius)
            sy = int(cy - ay * radius)
            self._anchor_screen_pos.append((sx, sy))

            if self._show_anchors:
                # Vector line from centre
                painter.setPen(QPen(QColor("#e07020"), 1.8))
                painter.drawLine(int(cx), int(cy), sx, sy)
                # Dot at anchor tip
                painter.setBrush(QColor("#e07020"))
                painter.setPen(Qt.PenStyle.NoPen)
                painter.drawEllipse(sx - 5, sy - 5, 10, 10)
                # Label beyond tip
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


# ── Screen widget ──────────────────────────────────────────════════════════════

class FreeVizScreen(QWidget, WorkflowNodeScreenSupport):
    """
    FreeViz – Force-directed RadViz (Demsar et al., 2005).

    Projection:  P = W @ anchors  (RadViz weighted-centre formula — points inside circle)
    Optimisation: gradient descent moves anchors to maximise class separation.

    Initialization modes (matching Orange's InitType enum):
    • Circular  — anchors evenly spaced on unit circle starting at π/2 (top)
                  (FreeViz.init_radial in Orange)
    • Random    — anchors placed at random unit-circle angles with seed 0
                  (FreeViz.init_random in Orange)

    Interactions (matching Orange):
    • Scroll wheel       → zoom in / out
    • Drag-anchor        → move that anchor, instantly see new distribution
    • Drag-canvas        → pan the view
    • Optimize / Stop    → start / stop gradient descent
    • Reset              → re-initialise anchors with current init mode
    """

    MAX_POINTS = 300

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._init_workflow_node_support()
        self._dataset: DatasetHandle | None = None

        # Computed data for optimisation
        self._W_np: np.ndarray | None = None        # (n_pts, n_feat) row-normalised weights
        self._X_np: np.ndarray | None = None        # (n_pts, n_feat) original normalised features
        self._labels_np: np.ndarray | None = None   # (n_pts,) int
        self._anchors_np: np.ndarray | None = None  # (n_feat, 2) unit vectors
        self._feature_names: list[str] = []
        self._class_labels: list[str] = []
        self._n_iters_done = 0
        self._last_stress = 0.0
        self._prev_stress = float("inf")

        self._timer = QTimer(self)
        self._timer.setInterval(40)
        self._timer.timeout.connect(self._run_batch)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        header = QLabel("FreeViz")
        header.setProperty("sectionTitle", True)
        layout.addWidget(header)

        desc = QLabel(
            "Force-directed RadViz (Demsar et al., 2005). "
            "Anchors on unit circle are optimised to maximise class separation. "
            "Drag an anchor to explore · scroll to zoom · pan to move."
        )
        desc.setWordWrap(True)
        desc.setProperty("muted", True)
        layout.addWidget(desc)

        ctrl_box = QGroupBox("Settings")
        ctrl_vbox = QVBoxLayout(ctrl_box)
        ctrl_vbox.setSpacing(4)

        # Row 1: Color by, Initialization, LR, Optimize/Reset
        row1 = QHBoxLayout()
        row1.setSpacing(6)
        row1.addWidget(QLabel("Color by:"))
        self._class_combo = QComboBox()
        self._class_combo.currentTextChanged.connect(self._reset_and_refresh)
        row1.addWidget(self._class_combo, 2)

        # Initialization mode — mirrors Orange's InitType.Circular / InitType.Random
        row1.addWidget(QLabel("Initialization:"))
        self._init_combo = QComboBox()
        self._init_combo.addItems(["Circular", "Random"])
        self._init_combo.setCurrentIndex(0)
        self._init_combo.setToolTip(
            "Circular: anchors evenly spaced on unit circle (Orange init_radial)\n"
            "Random: anchors at random unit-circle angles (Orange init_random)"
        )
        self._init_combo.currentIndexChanged.connect(self._reset_and_refresh)
        row1.addWidget(self._init_combo)

        row1.addWidget(QLabel("LR:"))
        self._lr_slider = QSlider(Qt.Orientation.Horizontal)
        self._lr_slider.setRange(1, 50)
        self._lr_slider.setValue(10)
        self._lr_slider.setMaximumWidth(90)
        self._lr_slider.setToolTip("Learning rate × 0.001")
        self._lr_label = QLabel("0.010")
        self._lr_slider.valueChanged.connect(lambda v: self._lr_label.setText(f"{v*0.001:.3f}"))
        row1.addWidget(self._lr_slider)
        row1.addWidget(self._lr_label)

        self._run_btn = QPushButton("Optimize")
        self._run_btn.setToolTip("Run FreeViz optimisation until convergence")
        self._run_btn.clicked.connect(self._toggle)
        row1.addWidget(self._run_btn)

        self._reset_btn = QPushButton("Reset")
        self._reset_btn.setToolTip(
            "Re-initialise anchors using the selected initialization mode"
        )
        self._reset_btn.clicked.connect(self._reset_and_refresh)
        row1.addWidget(self._reset_btn)
        row1.addStretch(1)
        ctrl_vbox.addLayout(row1)

        # Row 2: Visual controls
        row2 = QHBoxLayout()
        row2.setSpacing(6)

        self._show_anchors_cb = QCheckBox("Show anchors")
        self._show_anchors_cb.setChecked(True)
        self._show_anchors_cb.stateChanged.connect(
            lambda s: self._chart.set_show_anchors(bool(s)))
        row2.addWidget(self._show_anchors_cb)

        row2.addWidget(QLabel("Jitter:"))
        self._jitter_slider = QSlider(Qt.Orientation.Horizontal)
        self._jitter_slider.setRange(0, 50)
        self._jitter_slider.setValue(0)
        self._jitter_slider.setMaximumWidth(80)
        self._jitter_slider.setToolTip("Add jitter to separate overlapping points")
        self._jitter_slider.valueChanged.connect(lambda v: self._chart.set_jitter(v / 50.0))
        row2.addWidget(self._jitter_slider)

        row2.addWidget(QLabel("Size:"))
        self._size_slider = QSlider(Qt.Orientation.Horizontal)
        self._size_slider.setRange(2, 12)
        self._size_slider.setValue(5)
        self._size_slider.setMaximumWidth(70)
        row2.addWidget(self._size_slider)

        row2.addWidget(QLabel("Opacity:"))
        self._opacity_slider = QSlider(Qt.Orientation.Horizontal)
        self._opacity_slider.setRange(30, 255)
        self._opacity_slider.setValue(170)
        self._opacity_slider.setMaximumWidth(80)
        row2.addWidget(self._opacity_slider)

        row2.addStretch(1)
        ctrl_vbox.addLayout(row2)
        layout.addWidget(ctrl_box)

        chart_box = QGroupBox(
            "Projection  (drag-anchor=explore · scroll=zoom · drag-canvas=pan)")
        chart_layout = QVBoxLayout(chart_box)
        self._chart = _FreeVizWidget()
        self._chart.anchor_moved.connect(self._on_anchor_moved)
        chart_layout.addWidget(self._chart)
        layout.addWidget(chart_box, 1)

        # Connect sliders AFTER chart is created
        self._size_slider.valueChanged.connect(self._chart.set_point_size)
        self._opacity_slider.valueChanged.connect(self._chart.set_opacity)

        self._status_label = QLabel(i18n.t("Load a dataset with numeric columns and a class variable."))
        self._status_label.setProperty("muted", True)
        layout.addWidget(self._status_label)

    # ── WorkflowNodeScreenSupport ─────────────────────────────────────────────

    def set_input_payload(self, payload) -> None:
        self._timer.stop()
        self._run_btn.setText("Optimize")
        dataset = payload.dataset if payload is not None else None
        self._dataset = dataset
        self._class_combo.blockSignals(True)
        self._class_combo.clear()
        if dataset is not None:
            self._class_combo.addItem("(none)")
            all_cols = ([c.name for c in dataset.domain.columns] +
                        [c.name for c in dataset.domain.target_columns])
            for name in all_cols:
                self._class_combo.addItem(name)
            # Auto-select target
            for col in dataset.domain.target_columns:
                idx = self._class_combo.findText(col.name)
                if idx >= 0:
                    self._class_combo.setCurrentIndex(idx)
                    break
        self._class_combo.blockSignals(False)
        self._reset_and_refresh()

    def documentation_url(self) -> str:
        return "https://orangedatamining.com/widget-catalog/visualize/freeviz/"

    # ── Optimisation control ──────────────────────────────────────────────────

    def _toggle(self) -> None:
        if self._timer.isActive():
            self._timer.stop()
            self._run_btn.setText("Optimize")
        else:
            if self._W_np is None or len(self._W_np) == 0:
                return
            self._prev_stress = float("inf")
            self._timer.start()
            self._run_btn.setText("Stop")

    def _run_batch(self) -> None:
        if self._W_np is None or self._anchors_np is None:
            self._timer.stop()
            return
        lr = self._lr_slider.value() * 0.001
        self._anchors_np, self._last_stress = _freeviz_optimize(
            self._W_np, self._labels_np, self._anchors_np,
            n_iter=5, learning_rate=lr,
        )
        self._n_iters_done += 5
        stress_delta = abs(self._last_stress - self._prev_stress)
        self._prev_stress = self._last_stress
        if stress_delta < 1e-7 and self._n_iters_done > 30:
            self._timer.stop()
            self._run_btn.setText("Optimize")
        self._redraw()

    def _reset_and_refresh(self) -> None:
        self._timer.stop()
        self._run_btn.setText("Optimize")
        self._n_iters_done = 0
        self._last_stress = 0.0
        self._prev_stress = float("inf")
        self._chart.reset_view()
        self._build_data()
        # Quick warm-up: 50 iterations so initial view is already partially optimised
        # (matching Orange's behaviour of showing an optimised layout on open)
        if self._W_np is not None and len(self._W_np) >= 2:
            lr = self._lr_slider.value() * 0.001
            self._anchors_np, self._last_stress = _freeviz_optimize(
                self._W_np, self._labels_np, self._anchors_np,
                n_iter=50, learning_rate=lr,
            )
            self._n_iters_done = 50
        self._redraw()

    def _on_anchor_moved(self, idx: int, ax: float, ay: float) -> None:
        """User dragged an anchor — update its position and instantly redraw."""
        if self._anchors_np is not None and 0 <= idx < len(self._anchors_np):
            self._anchors_np[idx, 0] = ax
            self._anchors_np[idx, 1] = ay
            self._redraw()

    # ── Data preparation ──────────────────────────────────────────────────────

    def _build_data(self) -> None:
        self._W_np = None
        self._X_np = None
        self._labels_np = None
        self._anchors_np = None
        self._class_labels = []
        self._feature_names = []

        if self._dataset is None:
            return

        df = self._dataset.dataframe
        numeric_cols = [
            col.name for col in self._dataset.domain.columns
            if df[col.name].dtype.is_numeric()
        ]
        if len(numeric_cols) < 2:
            return

        class_col = self._class_combo.currentText()
        if class_col == "(none)":
            class_col = None

        n = len(df)
        if n == 0:
            return
        indices = list(range(n))
        if n > self.MAX_POINTS:
            random.seed(42)
            indices = random.sample(indices, self.MAX_POINTS)
            indices.sort()

        # Build feature matrix X (n_pts × n_feat), normalised [0, 1]
        raw = []
        for i in indices:
            row = [float(df[col][i]) if df[col][i] is not None else 0.0
                   for col in numeric_cols]
            raw.append(row)
        if not raw:
            return

        X = np.array(raw, dtype=np.float64)
        col_min = X.min(axis=0)
        col_rng = X.max(axis=0) - col_min
        col_rng[col_rng < 1e-10] = 1.0
        X = (X - col_min) / col_rng   # X ∈ [0, 1]

        # Row-normalise → W (RadViz weight matrix, each row sums to 1)
        row_sums = X.sum(axis=1, keepdims=True)
        row_sums[row_sums < 1e-10] = 1.0
        W = X / row_sums

        # Class labels
        labels = np.zeros(len(indices), dtype=np.int32)
        if class_col and class_col in df.columns:
            raw_classes = [
                str(df[class_col][i]) if df[class_col][i] is not None else "(missing)"
                for i in indices
            ]
            unique = list(dict.fromkeys(raw_classes))[:8]
            self._class_labels = unique
            class_map = {v: i for i, v in enumerate(unique)}
            labels = np.array([class_map.get(c, 0) for c in raw_classes], dtype=np.int32)

        self._X_np = X
        self._W_np = W
        self._labels_np = labels
        self._feature_names = numeric_cols

        n_feat = len(numeric_cols)
        # Initial anchors — mirrors Orange's InitType selection:
        #   Circular (0): FreeViz.init_radial  → evenly spaced starting at π/2
        #   Random   (1): FreeViz.init_random  → random unit-circle angles, seed 0
        if self._init_combo.currentIndex() == 0:
            # Circular: init_radial equivalent
            self._anchors_np = np.array([
                [math.cos(math.pi / 2 - 2 * math.pi * k / n_feat),
                 math.sin(math.pi / 2 - 2 * math.pi * k / n_feat)]
                for k in range(n_feat)
            ], dtype=np.float64)
        else:
            # Random: init_random equivalent — uniform random angles on unit circle
            rng = np.random.RandomState(0)   # seed=0, same reproducibility as Orange
            angles = rng.uniform(0, 2 * math.pi, n_feat)
            self._anchors_np = np.stack(
                [np.cos(angles), np.sin(angles)], axis=1
            ).astype(np.float64)

    # ── Projection & rendering ────────────────────────────────────────────────

    def _redraw(self) -> None:
        if self._W_np is None or self._anchors_np is None:
            self._chart.set_projection([], [], [])
            self._status_label.setText(
                i18n.t("Load a dataset with at least 2 numeric columns.")
            )
            return

        # Projection: RadViz weighted-centre formula → points inside unit circle
        P = self._W_np @ self._anchors_np   # (n_pts, 2)

        n_pts = len(P)
        n_feat = len(self._feature_names)

        points = [
            (float(P[i, 0]), float(P[i, 1]), int(self._labels_np[i]))
            for i in range(n_pts)
        ]

        anchors = [
            (self._feature_names[k],
             float(self._anchors_np[k, 0]),
             float(self._anchors_np[k, 1]))
            for k in range(n_feat)
        ]

        weights = [self._W_np[i].tolist() for i in range(n_pts)]

        self._chart.set_projection(
            points, anchors, self._class_labels,
            weights=weights, feature_names=self._feature_names,
        )

        class_col = self._class_combo.currentText()
        stress_str = f" · stress {self._last_stress:.4f}" if self._n_iters_done > 0 else ""
        self._status_label.setText(
            f"{n_pts} pts · {n_feat} anchors · iter {self._n_iters_done}"
            + stress_str
            + (f" · colored by '{class_col}'" if class_col and class_col != "(none)" else "")
        )
