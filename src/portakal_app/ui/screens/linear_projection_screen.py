from __future__ import annotations

import math
import random

import numpy as np

from PySide6.QtCore import Qt, QRect, QPoint
from PySide6.QtGui import QColor, QPainter, QPen, QFont, QFontMetrics, QPolygon
from PySide6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QRadioButton,
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

MAX_POINTS = 1000

# Placement modes — mirrors Orange's Placement enum
PLACEMENT_CIRCULAR = 0   # CircularPlacement (RadViz-style axes)
PLACEMENT_PCA      = 1   # PCA, n_components=2
PLACEMENT_LDA      = 2   # LDA (class-separation linear projection)


# ── Projections ──────────────────────────────────────────────────────────────

def _circular_placement(n_axes: int) -> np.ndarray:
    """
    Orange's CircularPlacement.get_components() — exact mirror.

    Returns a (2, n_axes) matrix of (cos, sin) unit vectors,
    evenly spaced in [0, 2π).

    Special cases (copied from Orange):
      n_axes == 1  → angle = [0]
      n_axes == 2  → angles = [0, π/2]
      n_axes  ≥ 3  → np.linspace(0, 2π, n_axes, endpoint=False)
    """
    if n_axes == 1:
        angles = np.array([0.0])
    elif n_axes == 2:
        angles = np.array([0.0, math.pi / 2])
    else:
        angles = np.linspace(0, 2 * math.pi, n_axes, endpoint=False)
    return np.vstack([np.cos(angles), np.sin(angles)])  # (2, n_axes)


def _pca_projection(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    PCA, n_components=2, with normalisation (mean-centred, unit std).
    Mirrors Orange's: PCA(n_components=2) + Normalize() preprocessor.

    Returns:
        coords   – (n, 2) projected coordinates, normalised to [-1, 1] span
        loadings – (n_axes, 2) feature–PC matrix (for biplot arrows)
    """
    means = X.mean(axis=0)
    stds  = X.std(axis=0)
    stds[stds < 1e-10] = 1.0
    Xn = (X - means) / stds

    _, S, Vt = np.linalg.svd(Xn, full_matrices=False)
    coords   = Xn @ Vt[:2].T    # (n, 2)
    loadings = Vt[:2].T         # (n_axes, 2)

    # Normalise coords to [-1,1] range per axis (matches Orange's normalise())
    span = coords.max(axis=0) - coords.min(axis=0)
    span[span < 1e-10] = 1.0
    coords = (coords - coords.mean(axis=0)) / span

    return coords, loadings


def _lda_projection(
    X: np.ndarray, labels: np.ndarray, n_classes: int
) -> tuple[np.ndarray, np.ndarray | None]:
    """
    Simple 2-component LDA using between-class / within-class scatter matrices.
    Mirrors Orange's LDA(solver='eigen', n_components=2).

    Returns:
        coords   – (n, 2) projected coordinates, normalised
        W        – (n_axes, 2) LDA axes (for arrows), or None on failure
    """
    n_feat = X.shape[1]
    means = X.mean(axis=0)

    # Within-class scatter (Sw) and Between-class scatter (Sb)
    Sw = np.zeros((n_feat, n_feat))
    Sb = np.zeros((n_feat, n_feat))
    for c in range(n_classes):
        mask = labels == c
        if mask.sum() < 2:
            continue
        Xc   = X[mask]
        mc   = Xc.mean(axis=0)
        Sw  += (Xc - mc).T @ (Xc - mc)
        diff = (mc - means).reshape(-1, 1)
        Sb  += mask.sum() * (diff @ diff.T)

    try:
        # Solve generalised eigenvalue: Sb @ w = λ Sw @ w
        Sw_reg = Sw + np.eye(n_feat) * 1e-6
        Sw_inv = np.linalg.inv(Sw_reg)
        evals, evecs = np.linalg.eigh(Sw_inv @ Sb)
        # Take top-2 eigenvectors (largest eigenvalues last with eigh)
        W = evecs[:, -2:][:, ::-1]   # (n_feat, 2)
    except np.linalg.LinAlgError:
        return _pca_projection(X)     # fallback to PCA

    coords = X @ W   # (n, 2)

    # Normalise
    span = coords.max(axis=0) - coords.min(axis=0)
    span[span < 1e-10] = 1.0
    coords = (coords - coords.mean(axis=0)) / span

    return coords, W


def _nice_ticks(vmin: float, vmax: float, n: int = 5) -> list[float]:
    span = vmax - vmin or 1.0
    raw  = span / n
    exp  = math.floor(math.log10(abs(raw)) if abs(raw) > 1e-15 else 0)
    frac = raw / 10 ** exp
    nice = next((v for v in (1, 2, 2.5, 5, 10) if frac <= v), 10)
    step  = nice * 10 ** exp
    start = math.ceil(vmin / step) * step
    ticks, v = [], start
    while v <= vmax + step * 0.01:
        ticks.append(round(v, 10))
        v += step
    return ticks


# ── Canvas widget ─────────────────────────────────────────────────────────────

class _ProjectionWidget(QWidget):
    """
    Shared scatter canvas for all three placement modes.

    • Data points coloured by class
    • Optional anchor arrows (loading vectors or circular axes)
    • Axis labels; tick grid; zoom/pan; hover tooltips
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._points:       list[tuple[float, float, int]] = []
        self._raw_values:   list[list[float]] = []
        self._anchors:      list[tuple[float, float, str]] = []  # (ax, ay, label)
        self._class_labels: list[str] = []
        self._feature_names: list[str] = []
        self._x_label = "X"
        self._y_label = "Y"
        self._x_min = -1.0; self._x_max = 1.0
        self._y_min = -1.0; self._y_max = 1.0
        self._show_arrows  = True
        self._point_size   = 5
        self._opacity      = 180
        self._jitter       = 0.0
        self._zoom  = 1.0
        self._pan_x = 0.0
        self._pan_y = 0.0
        self._drag_start: QPoint | None = None
        self._dot_rects: list[tuple[QRect, str]] = []

        self.setMinimumHeight(300)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMouseTracking(True)

    # ── Setters ──────────────────────────────────────────────────────────────

    def set_data(
        self,
        points:        list[tuple[float, float, int]],
        raw_values:    list[list[float]],
        anchors:       list[tuple[float, float, str]],
        class_labels:  list[str],
        feature_names: list[str],
        x_min: float, x_max: float,
        y_min: float, y_max: float,
        x_label: str = "X",
        y_label: str = "Y",
    ) -> None:
        self._points        = points
        self._raw_values    = raw_values
        self._anchors       = anchors
        self._class_labels  = class_labels
        self._feature_names = feature_names
        self._x_min = x_min; self._x_max = x_max
        self._y_min = y_min; self._y_max = y_max
        self._x_label = x_label
        self._y_label = y_label
        self._dot_rects = []
        self._zoom = 1.0; self._pan_x = 0.0; self._pan_y = 0.0
        self.update()

    def set_show_arrows(self, show: bool)  -> None: self._show_arrows = show;  self.update()
    def set_point_size(self, size: int)    -> None: self._point_size  = max(2, size); self.update()
    def set_opacity(self, opacity: int)    -> None: self._opacity = max(30, min(255, opacity)); self.update()
    def set_jitter(self, amount: float)    -> None: self._jitter = amount; self.update()

    # ── Zoom / pan ────────────────────────────────────────────────────────────

    def wheelEvent(self, event) -> None:
        self._zoom = min(self._zoom * 1.15, 8.0) if event.angleDelta().y() > 0 \
                     else max(self._zoom / 1.15, 0.2)
        self.update()

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag_start = event.position().toPoint()

    def mouseMoveEvent(self, event) -> None:
        pos = event.position().toPoint()
        if event.buttons() & Qt.MouseButton.LeftButton and self._drag_start is not None:
            self._pan_x += pos.x() - self._drag_start.x()
            self._pan_y += pos.y() - self._drag_start.y()
            self._drag_start = pos
            self.update()
            QToolTip.hideText()
            return
        for rect, tip in self._dot_rects:
            if rect.contains(pos):
                QToolTip.showText(event.globalPosition().toPoint(), tip, self)
                return
        QToolTip.hideText()

    def mouseReleaseEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag_start = None

    def leaveEvent(self, _event) -> None:
        QToolTip.hideText()

    def mouseDoubleClickEvent(self, _event) -> None:
        self._zoom = 1.0; self._pan_x = 0.0; self._pan_y = 0.0
        self.update()

    # ── Drawing ───────────────────────────────────────────────────────────────

    def paintEvent(self, _event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), QColor("#fffdf9"))

        if not self._points:
            painter.setPen(QColor("#8d877d"))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter,
                             "No data.\nLoad a dataset with at least 2 numeric columns.")
            return

        self._dot_rects = []
        w, h = self.width(), self.height()
        legend_w  = 130 if self._class_labels else 0
        margin_l, margin_r = 58, 14 + legend_w
        margin_t, margin_b = 14, 50
        cw = w - margin_l - margin_r
        ch = h - margin_t - margin_b
        if cw < 20 or ch < 20:
            return

        painter.setFont(QFont(self.font().family(), 8))
        fm = QFontMetrics(painter.font())

        # Zoom / pan transform
        x_span = (self._x_max - self._x_min) or 1.0
        y_span = (self._y_max - self._y_min) or 1.0
        eff_x  = x_span / self._zoom
        eff_y  = y_span / self._zoom
        cx_d   = (self._x_min + self._x_max) / 2.0
        cy_d   = (self._y_min + self._y_max) / 2.0
        vx_min = cx_d - self._pan_x / cw * eff_x - eff_x / 2
        vx_max = vx_min + eff_x
        vy_min = cy_d + self._pan_y / ch * eff_y - eff_y / 2
        vy_max = vy_min + eff_y

        def to_px(xv: float, yv: float) -> tuple[int, int]:
            sx = margin_l + int((xv - vx_min) / (vx_max - vx_min) * cw)
            sy = margin_t + ch - int((yv - vy_min) / (vy_max - vy_min) * ch)
            return sx, sy

        # Grid
        x_ticks = _nice_ticks(vx_min, vx_max, 5)
        y_ticks = _nice_ticks(vy_min, vy_max, 5)
        painter.setPen(QPen(QColor("#e0ddd6"), 1, Qt.PenStyle.DotLine))
        painter.setClipRect(margin_l, margin_t, cw, ch)
        for xt in x_ticks:
            px, _ = to_px(xt, 0); painter.drawLine(px, margin_t, px, margin_t + ch)
        for yt in y_ticks:
            _, py = to_px(0, yt); painter.drawLine(margin_l, py, margin_l + cw, py)
        painter.setClipping(False)

        # Origin crosshair
        ox, oy = to_px(0.0, 0.0)
        painter.setPen(QPen(QColor("#c0bcb5"), 1, Qt.PenStyle.DashLine))
        painter.setClipRect(margin_l, margin_t, cw, ch)
        painter.drawLine(ox, margin_t, ox, margin_t + ch)
        painter.drawLine(margin_l, oy, margin_l + cw, oy)
        painter.setClipping(False)

        # Tick labels
        painter.setPen(QColor("#8d877d"))
        for xt in x_ticks:
            px, _ = to_px(xt, 0)
            painter.drawText(px - 24, margin_t + ch + 4, 48, 14,
                             Qt.AlignmentFlag.AlignCenter, f"{xt:.3g}")
        for yt in y_ticks:
            _, py = to_px(0, yt)
            painter.drawText(2, py - 7, margin_l - 6, 14,
                             Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
                             f"{yt:.3g}")

        # Axes box
        painter.setPen(QPen(QColor("#9b9488"), 1))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawRect(margin_l, margin_t, cw, ch)

        # Axis labels
        painter.setPen(QColor("#534b40"))
        painter.setFont(QFont(self.font().family(), 9))
        painter.drawText(margin_l, margin_t + ch + 20, cw, 18,
                         Qt.AlignmentFlag.AlignCenter, self._x_label)
        painter.save()
        painter.translate(12, margin_t + ch // 2)
        painter.rotate(-90)
        painter.drawText(-50, -7, 100, 14, Qt.AlignmentFlag.AlignCenter, self._y_label)
        painter.restore()

        # Hint
        painter.setFont(QFont(self.font().family(), 7))
        painter.setPen(QColor(160, 155, 145, 160))
        painter.drawText(margin_l + 4, h - 4,
                         f"scroll=zoom · drag=pan · dbl-click=reset  {self._zoom:.2f}×")

        # ── Anchor arrows ─────────────────────────────────────────────────────
        if self._show_arrows and self._anchors:
            painter.setFont(QFont(self.font().family(), 7))
            painter.setClipRect(margin_l, margin_t, cw, ch)
            for ax, ay, fname in self._anchors:
                tx, ty = to_px(ax, ay)
                painter.setPen(QPen(QColor(80, 100, 160, 200), 1.5))
                painter.setBrush(Qt.BrushStyle.NoBrush)
                painter.drawLine(ox, oy, tx, ty)
                # Arrowhead
                dx, dy = tx - ox, ty - oy
                length = math.sqrt(dx * dx + dy * dy) or 1.0
                ux, uy = dx / length, dy / length
                px_p, py_p = -uy, ux
                ah = 6
                p1 = QPoint(int(tx - ux * ah + px_p * 3), int(ty - uy * ah + py_p * 3))
                p2 = QPoint(int(tx - ux * ah - px_p * 3), int(ty - uy * ah - py_p * 3))
                painter.setBrush(QColor(80, 100, 160, 200))
                painter.setPen(Qt.PenStyle.NoPen)
                painter.drawPolygon(QPolygon([QPoint(tx, ty), p1, p2]))
                # Label
                lx = tx + (8 if dx >= 0 else -8 - fm.horizontalAdvance(fname))
                ly = ty + (12 if dy >= 0 else -4)
                painter.setPen(QColor(40, 60, 140))
                painter.setFont(QFont(self.font().family(), 7))
                painter.drawText(lx, ly, fname)
            painter.setClipping(False)

        # ── Points ────────────────────────────────────────────────────────────
        rng = random.Random(42)
        r   = self._point_size
        painter.setClipRect(margin_l - r, margin_t - r, cw + r * 2, ch + r * 2)
        for idx, (nx, ny, ci) in enumerate(self._points):
            xv = self._x_min + nx * x_span
            yv = self._y_min + ny * y_span
            if self._jitter:
                xv += (rng.random() - 0.5) * self._jitter * x_span * 0.04
                yv += (rng.random() - 0.5) * self._jitter * y_span * 0.04
            sx, sy = to_px(xv, yv)
            color = QColor(_PALETTE[ci % len(_PALETTE)])
            color.setAlpha(self._opacity)
            painter.setBrush(color)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawEllipse(sx - r, sy - r, r * 2, r * 2)
            if idx < len(self._raw_values):
                feat_lines = "  ".join(
                    f"{name}: {val:.3g}"
                    for name, val in zip(self._feature_names, self._raw_values[idx])
                )
                class_lbl = (self._class_labels[ci]
                             if ci < len(self._class_labels) else str(ci))
                tip = f"Class: {class_lbl}\n{feat_lines}"
            else:
                tip = f"x: {nx:.3f}  y: {ny:.3f}"
            self._dot_rects.append((
                QRect(sx - r - 2, sy - r - 2, (r + 2) * 2, (r + 2) * 2), tip,
            ))
        painter.setClipping(False)

        # ── Legend ────────────────────────────────────────────────────────────
        if self._class_labels:
            lx = w - legend_w + 4
            painter.setFont(QFont(self.font().family(), 8))
            fm_leg = QFontMetrics(painter.font())
            for i, lbl in enumerate(self._class_labels[:8]):
                ly = margin_t + i * 20
                color = _PALETTE[i % len(_PALETTE)]
                painter.setBrush(color)
                painter.setPen(Qt.PenStyle.NoPen)
                painter.drawEllipse(lx, ly + 3, 10, 10)
                painter.setPen(QColor("#534b40"))
                lbl_t = fm_leg.elidedText(lbl, Qt.TextElideMode.ElideRight, legend_w - 20)
                painter.drawText(lx + 14, ly, legend_w - 18, 16,
                                 Qt.AlignmentFlag.AlignVCenter, lbl_t)

        painter.end()


# ── Screen widget ─────────────────────────────────────────────────────────────

class LinearProjectionScreen(QWidget, WorkflowNodeScreenSupport):
    """
    Linear Projection — three placement modes (mirrors Orange exactly).

    Placement.Circular:
        Each numeric feature gets a fixed axis at an evenly-spaced angle.
        Data point = dot product of (normalised) feature vector onto all axes.
        Exactly Orange's CircularPlacement(LinearProjector).

    Placement.PCA:
        PCA(n_components=2) + Normalize() — projects to PC1 × PC2 only.
        No PC axis selector (Orange doesn't have one either).

    Placement.LDA:
        LDA(solver='eigen', n_components=2) — class-separation projection.
        Falls back to PCA when class variable unsuitable.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._init_workflow_node_support()
        self._dataset: DatasetHandle | None = None
        self._placement = PLACEMENT_CIRCULAR   # default matches Orange

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        header = QLabel("Linear Projection")
        header.setProperty("sectionTitle", True)
        layout.addWidget(header)

        desc = QLabel(
            "Multi-axis projection onto a 2-D plane. "
            "Choose placement: Circular (axes per feature), "
            "PCA (PC1 × PC2), or LDA (class-separation). "
            "Scroll to zoom · drag to pan · double-click to reset."
        )
        desc.setWordWrap(True)
        desc.setProperty("muted", True)
        layout.addWidget(desc)

        # ── Settings box ──────────────────────────────────────────────────────
        ctrl_box  = QGroupBox("Settings")
        ctrl_vbox = QVBoxLayout(ctrl_box)
        ctrl_vbox.setSpacing(4)

        # Row 1: Color by, placement radio buttons
        row1 = QHBoxLayout()
        row1.setSpacing(6)
        row1.addWidget(QLabel("Color by:"))
        self._class_combo = QComboBox()
        self._class_combo.currentTextChanged.connect(self._refresh)
        row1.addWidget(self._class_combo, 2)

        # Placement radio buttons — mirrors Orange's Projection_name dict
        row1.addWidget(QLabel("Placement:"))
        self._placement_group = QButtonGroup(self)
        for idx, label in enumerate(["Circular", "PCA", "LDA"]):
            rb = QRadioButton(label)
            rb.setChecked(idx == 0)
            tooltip_map = {
                0: "Circular: fixed evenly-spaced axes per feature (Orange default)",
                1: "PCA: project to first 2 principal components",
                2: "LDA: class-separation projection (needs class variable)",
            }
            rb.setToolTip(tooltip_map[idx])
            self._placement_group.addButton(rb, idx)
            row1.addWidget(rb)
        self._placement_group.idClicked.connect(self._on_placement_changed)
        row1.addStretch(1)
        ctrl_vbox.addLayout(row1)

        # Row 2: Show arrows, Jitter, Size, Opacity
        row2 = QHBoxLayout()
        row2.setSpacing(6)

        self._arrows_cb = QCheckBox("Show arrows")
        self._arrows_cb.setChecked(True)
        self._arrows_cb.setToolTip("Show feature axis / loading arrows")
        self._arrows_cb.stateChanged.connect(
            lambda s: self._chart.set_show_arrows(bool(s))
        )
        row2.addWidget(self._arrows_cb)

        row2.addWidget(QLabel("Jitter:"))
        self._jitter_slider = QSlider(Qt.Orientation.Horizontal)
        self._jitter_slider.setRange(0, 50)
        self._jitter_slider.setValue(0)
        self._jitter_slider.setMaximumWidth(80)
        self._jitter_slider.setToolTip("Add random jitter to separate overlapping points")
        self._jitter_slider.valueChanged.connect(
            lambda v: self._chart.set_jitter(v / 50.0)
        )
        row2.addWidget(self._jitter_slider)

        row2.addWidget(QLabel("Size:"))
        self._size_slider = QSlider(Qt.Orientation.Horizontal)
        self._size_slider.setRange(2, 14)
        self._size_slider.setValue(5)
        self._size_slider.setMaximumWidth(70)
        row2.addWidget(self._size_slider)

        row2.addWidget(QLabel("Opacity:"))
        self._opacity_slider = QSlider(Qt.Orientation.Horizontal)
        self._opacity_slider.setRange(30, 255)
        self._opacity_slider.setValue(180)
        self._opacity_slider.setMaximumWidth(80)
        row2.addWidget(self._opacity_slider)

        row2.addStretch(1)
        ctrl_vbox.addLayout(row2)
        layout.addWidget(ctrl_box)

        # ── Chart ─────────────────────────────────────────────────────────────
        self._chart_box = QGroupBox("Projection  (Circular Placement)")
        chart_layout = QVBoxLayout(self._chart_box)
        self._chart = _ProjectionWidget()
        chart_layout.addWidget(self._chart)
        layout.addWidget(self._chart_box, 1)

        # Connect sliders AFTER chart created
        self._size_slider.valueChanged.connect(self._chart.set_point_size)
        self._opacity_slider.valueChanged.connect(self._chart.set_opacity)

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
            all_cols = ([c.name for c in dataset.domain.columns] +
                        [c.name for c in dataset.domain.target_columns])
            self._class_combo.addItem("(none)")
            self._class_combo.addItems(all_cols)
            for col in dataset.domain.target_columns:
                idx = self._class_combo.findText(col.name)
                if idx >= 0:
                    self._class_combo.setCurrentIndex(idx)
                    break
        self._class_combo.blockSignals(False)
        self._refresh()

    def documentation_url(self) -> str:
        return "https://orangedatamining.com/widget-catalog/visualize/linearprojection/"

    # ── Internal ──────────────────────────────────────────────────────────────

    def _on_placement_changed(self, idx: int) -> None:
        self._placement = idx
        titles = [
            "Projection  (Circular Placement)",
            "Projection  (PCA — PC1 × PC2)",
            "Projection  (LDA)",
        ]
        self._chart_box.setTitle(titles[idx])
        self._refresh()

    def _refresh(self) -> None:
        def clear():
            self._chart.set_data([], [], [], [], [],
                                 -1.0, 1.0, -1.0, 1.0)

        if self._dataset is None:
            clear()
            self._status_label.setText(i18n.t("Load a dataset with numeric columns."))
            return

        df = self._dataset.dataframe
        num_cols = [c.name for c in self._dataset.domain.columns
                    if df[c.name].dtype.is_numeric()]

        if len(num_cols) < 2:
            clear()
            self._status_label.setText(
                i18n.t("Need at least 2 numeric columns for projection.")
            )
            return

        # Class variable
        class_col = self._class_combo.currentText()
        class_raw: list[str] | None = None
        if class_col and class_col != "(none)" and class_col in df.columns:
            class_raw = [
                str(v) if v is not None else "(missing)"
                for v in df[class_col].to_list()
            ]

        # Sample
        n = len(df)
        indices = list(range(n))
        if n > MAX_POINTS:
            random.seed(42)
            indices = random.sample(indices, MAX_POINTS)
            indices.sort()

        col_data = {col: df[col].to_list() for col in num_cols}
        valid_idx = [
            i for i in indices
            if all(col_data[c][i] is not None for c in num_cols)
        ]
        if len(valid_idx) < 2:
            clear()
            self._status_label.setText(i18n.t("No valid (non-null) data points."))
            return

        X = np.array(
            [[float(col_data[c][i]) for c in num_cols] for i in valid_idx],
            dtype=np.float64,
        )

        # ── Class mapping ─────────────────────────────────────────────────────
        class_labels: list[str] = []
        class_map:    dict[str, int] = {}
        int_labels:   np.ndarray | None = None
        n_classes = 0

        if class_raw is not None:
            cl_subset = [class_raw[i] for i in valid_idx]
            unique      = list(dict.fromkeys(cl_subset))[:8]
            class_labels = unique
            class_map    = {v: k for k, v in enumerate(unique)}
            int_labels   = np.array([class_map.get(c, 0) for c in cl_subset],
                                    dtype=np.int32)
            n_classes = len(unique)

        if int_labels is None:
            int_labels = np.zeros(len(valid_idx), dtype=np.int32)

        # ── Projection ────────────────────────────────────────────────────────
        status_extra = ""

        if self._placement == PLACEMENT_CIRCULAR:
            # Orange's CircularPlacement — axes matrix (2, n_feat)
            axes = _circular_placement(len(num_cols))  # (2, n_feat)
            # Normalise X to [0,1] per column (Orange's Normalize preprocessing)
            col_min = X.min(axis=0)
            col_rng = X.max(axis=0) - col_min
            col_rng[col_rng < 1e-10] = 1.0
            Xn = (X - col_min) / col_rng

            coords  = Xn @ axes.T   # (n, 2) — linear combination onto circle axes
            anchors_raw = axes.T    # (n_feat, 2)

            x_label, y_label = "Axis X", "Axis Y"
            status_extra = f"  ·  {len(num_cols)} circular axes"

        elif self._placement == PLACEMENT_PCA:
            try:
                coords, loadings = _pca_projection(X)
                anchors_raw = loadings   # (n_feat, 2)
            except Exception:
                clear()
                self._status_label.setText("PCA computation failed.")
                return
            x_label, y_label = "PC1", "PC2"
            status_extra = "  ·  PCA"

        else:  # LDA
            if n_classes < 2:
                # LDA needs class variable — fall back to PCA
                try:
                    coords, loadings = _pca_projection(X)
                    anchors_raw = loadings
                except Exception:
                    clear()
                    self._status_label.setText("Projection failed.")
                    return
                x_label, y_label = "PC1", "PC2"
                status_extra = "  ·  LDA→PCA (no class var)"
            else:
                try:
                    coords, W = _lda_projection(X, int_labels, n_classes)
                    anchors_raw = W if W is not None else np.zeros((len(num_cols), 2))
                except Exception:
                    clear()
                    self._status_label.setText("LDA computation failed.")
                    return
                x_label, y_label = "LD1", "LD2"
                status_extra = f"  ·  LDA  ({n_classes} classes)"

        # ── Common: normalise coords to [0,1] for canvas ──────────────────────
        x_min, x_max = float(coords[:, 0].min()), float(coords[:, 0].max())
        y_min, y_max = float(coords[:, 1].min()), float(coords[:, 1].max())
        x_pad = (x_max - x_min) * 0.12 or 0.5
        y_pad = (y_max - y_min) * 0.12 or 0.5
        x_min -= x_pad; x_max += x_pad
        y_min -= y_pad; y_max += y_pad
        x_span = (x_max - x_min) or 1.0
        y_span = (y_max - y_min) or 1.0

        norm_pts = [
            (
                (float(coords[j, 0]) - x_min) / x_span,
                (float(coords[j, 1]) - y_min) / y_span,
                int(int_labels[j]),
            )
            for j in range(len(valid_idx))
        ]
        raw_vals = [
            [float(col_data[c][valid_idx[j]]) for c in num_cols]
            for j in range(len(valid_idx))
        ]

        # Scale arrows to ~35% of data range
        data_scale = max(abs(x_max - x_min), abs(y_max - y_min)) * 0.35
        max_norm   = float(np.max(np.linalg.norm(anchors_raw, axis=1))) or 1.0
        anchors = [
            (float(anchors_raw[k, 0]) * data_scale / max_norm,
             float(anchors_raw[k, 1]) * data_scale / max_norm,
             num_cols[k])
            for k in range(len(num_cols))
        ]

        self._chart.set_data(
            points=norm_pts,
            raw_values=raw_vals,
            anchors=anchors,
            class_labels=class_labels,
            feature_names=num_cols,
            x_min=x_min, x_max=x_max,
            y_min=y_min, y_max=y_max,
            x_label=x_label,
            y_label=y_label,
        )
        self._chart.set_show_arrows(self._arrows_cb.isChecked())

        n_valid = len(valid_idx)
        sampled = f" (sampled {MAX_POINTS} of {n})" if n > MAX_POINTS else ""
        color_note = (f"  ·  colored by '{class_col}'"
                      if class_col and class_col != "(none)" else "")
        self._status_label.setText(
            f"{n_valid} points{sampled}  ·  {len(num_cols)} features"
            f"{status_extra}{color_note}"
        )
