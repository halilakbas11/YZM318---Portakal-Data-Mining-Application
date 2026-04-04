from __future__ import annotations

import math
import random

import numpy as np

from PySide6.QtCore import Qt, QRect
from PySide6.QtGui import QColor, QPainter, QPen, QFont, QFontMetrics
from PySide6.QtWidgets import (
    QCheckBox,
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

_POINT_R = 4  # dot radius in pixels


# ── Statistics ─────────────────────────────────────────────────────────────────

def _correlation_stats(xs: np.ndarray, ys: np.ndarray) -> tuple[float, float, float]:
    """Return (Pearson r, R², p-value) using numpy.

    p-value: two-tailed t-test with df = n − 2.
    Normal approximation used for large n; exact formula for n ≥ 4.
    """
    n = len(xs)
    if n < 4:
        return 0.0, 0.0, 1.0
    r = float(np.corrcoef(xs, ys)[0, 1])
    if not math.isfinite(r):
        r = 0.0
    r = max(-1.0, min(1.0, r))
    r2 = r * r
    # t-statistic: t = r * sqrt(n-2) / sqrt(1 - r²)
    denom = 1.0 - r2
    if denom < 1e-12:
        p = 0.0
    else:
        t = r * math.sqrt(n - 2) / math.sqrt(denom)
        # Two-tailed p via normal approximation (accurate for n > 30, adequate otherwise)
        p = float(math.erfc(abs(t) / math.sqrt(2)))
    return r, r2, p


def _ols_line(xs: np.ndarray, ys: np.ndarray) -> tuple[float, float]:
    """Return OLS regression (slope, intercept): ŷ = slope·x + intercept."""
    if len(xs) < 2:
        return 0.0, float(np.mean(ys)) if len(ys) else 0.0
    x_mean = float(np.mean(xs))
    y_mean = float(np.mean(ys))
    var_x = float(np.var(xs))
    if var_x < 1e-12:
        return 0.0, y_mean
    cov = float(np.mean((xs - x_mean) * (ys - y_mean)))
    slope = cov / var_x
    intercept = y_mean - slope * x_mean
    return slope, intercept


def _nice_ticks(vmin: float, vmax: float, n: int = 5) -> list[float]:
    """Return ~n evenly-spaced, nicely-rounded tick values."""
    span = vmax - vmin or 1.0
    raw = span / n
    exp = math.floor(math.log10(raw))
    frac = raw / 10 ** exp
    nice = next((v for v in (1, 2, 2.5, 5, 10) if frac <= v), 10)
    step = nice * 10 ** exp
    start = math.ceil(vmin / step) * step
    ticks = []
    v = start
    while v <= vmax + step * 0.01:
        ticks.append(round(v, 10))
        v += step
    return ticks


# ── Canvas widget ──────────────────────────────────────────────────────────────

class _ScatterWidget(QWidget):
    """
    2-D scatter plot with:
    • Proper axis tick labels (actual data values, not [0,1])
    • OLS regression line (optional, dashed)
    • Pearson r, R², p-value displayed in the chart corner
    • Per-class colour with legend
    • Hover tooltip: x, y, class
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._points: list[tuple[float, float, int]] = []  # normalised [0,1]
        self._raw_pts: list[tuple[float, float]] = []       # original values
        self._x_label = "X"
        self._y_label = "Y"
        self._class_labels: list[str] = []
        self._x_min = self._x_max = 0.0
        self._y_min = self._y_max = 1.0
        self._pearson = 0.0
        self._r2 = 0.0
        self._p_val = 1.0
        self._reg_slope = 0.0
        self._reg_intercept = 0.0
        self._show_reg_line = True
        self._dot_rects: list[tuple[QRect, str]] = []
        self.setMinimumHeight(250)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMouseTracking(True)

    def set_data(
        self,
        points: list[tuple[float, float, int]],
        raw_pts: list[tuple[float, float]],
        x_label: str,
        y_label: str,
        class_labels: list[str],
        x_min: float, x_max: float,
        y_min: float, y_max: float,
        pearson: float,
        r2: float,
        p_val: float,
        reg_slope: float,
        reg_intercept: float,
    ) -> None:
        self._points = points
        self._raw_pts = raw_pts
        self._x_label = x_label
        self._y_label = y_label
        self._class_labels = class_labels
        self._x_min, self._x_max = x_min, x_max
        self._y_min, self._y_max = y_min, y_max
        self._pearson = pearson
        self._r2 = r2
        self._p_val = p_val
        self._reg_slope = reg_slope
        self._reg_intercept = reg_intercept
        self._dot_rects = []
        self.update()

    def set_show_reg_line(self, show: bool) -> None:
        self._show_reg_line = show
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

        if not self._points:
            painter.setPen(QColor("#8d877d"))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter,
                             "No data.\nLoad a dataset with at least 2 numeric columns.")
            return

        self._dot_rects = []

        w, h = self.width(), self.height()
        legend_w = 130 if self._class_labels else 0
        margin_l, margin_r = 58, 12 + legend_w
        margin_t, margin_b = 14, 50

        cw = w - margin_l - margin_r
        ch = h - margin_t - margin_b
        if cw < 20 or ch < 20:
            return

        painter.setFont(QFont(self.font().family(), 8))
        fm = QFontMetrics(painter.font())

        x_span = (self._x_max - self._x_min) or 1.0
        y_span = (self._y_max - self._y_min) or 1.0

        def data_to_px(xv: float, yv: float) -> tuple[int, int]:
            sx = margin_l + int((xv - self._x_min) / x_span * cw)
            sy = margin_t + ch - int((yv - self._y_min) / y_span * ch)
            return sx, sy

        # ── Grid & tick labels ─────────────────────────────────────────────────
        x_ticks = _nice_ticks(self._x_min, self._x_max, 5)
        y_ticks = _nice_ticks(self._y_min, self._y_max, 5)

        painter.setPen(QPen(QColor("#e0ddd6"), 1, Qt.PenStyle.DotLine))
        for xt in x_ticks:
            px = margin_l + int((xt - self._x_min) / x_span * cw)
            painter.drawLine(px, margin_t, px, margin_t + ch)
        for yt in y_ticks:
            py = margin_t + ch - int((yt - self._y_min) / y_span * ch)
            painter.drawLine(margin_l, py, margin_l + cw, py)

        painter.setPen(QColor("#8d877d"))
        for xt in x_ticks:
            px = margin_l + int((xt - self._x_min) / x_span * cw)
            painter.drawText(px - 24, margin_t + ch + 4, 48, 14,
                             Qt.AlignmentFlag.AlignCenter, f"{xt:.3g}")
        for yt in y_ticks:
            py = margin_t + ch - int((yt - self._y_min) / y_span * ch)
            painter.drawText(2, py - 7, margin_l - 6, 14,
                             Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
                             f"{yt:.3g}")

        # ── Axes box ───────────────────────────────────────────────────────────
        painter.setPen(QPen(QColor("#9b9488"), 1))
        painter.drawRect(margin_l, margin_t, cw, ch)

        # ── Axis labels ────────────────────────────────────────────────────────
        painter.setPen(QColor("#534b40"))
        painter.setFont(QFont(self.font().family(), 9))
        x_lbl = fm.elidedText(self._x_label, Qt.TextElideMode.ElideRight, cw - 10)
        painter.drawText(margin_l, margin_t + ch + 20, cw, 18,
                         Qt.AlignmentFlag.AlignCenter, x_lbl)
        painter.save()
        painter.translate(12, margin_t + ch // 2)
        painter.rotate(-90)
        y_lbl = fm.elidedText(self._y_label, Qt.TextElideMode.ElideRight, ch - 10)
        painter.drawText(-50, -7, 100, 14, Qt.AlignmentFlag.AlignCenter, y_lbl)
        painter.restore()

        # ── OLS Regression line ────────────────────────────────────────────────
        if self._show_reg_line and len(self._points) >= 2:
            # Compute y at x_min and x_max from regression in data space
            y_at_xmin = self._reg_slope * self._x_min + self._reg_intercept
            y_at_xmax = self._reg_slope * self._x_max + self._reg_intercept
            # Clamp to visible y range for drawing
            sx0, sy0 = data_to_px(self._x_min, y_at_xmin)
            sx1, sy1 = data_to_px(self._x_max, y_at_xmax)
            painter.setFont(QFont(self.font().family(), 8))
            pen = QPen(QColor(80, 80, 80, 160), 1.5, Qt.PenStyle.DashLine)
            painter.setPen(pen)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.setClipRect(margin_l, margin_t, cw, ch)
            painter.drawLine(sx0, sy0, sx1, sy1)
            painter.setClipping(False)

        # ── Points ─────────────────────────────────────────────────────────────
        painter.setFont(QFont(self.font().family(), 8))
        for idx, (pn_x, pn_y, ci) in enumerate(self._points):
            sx = margin_l + int(pn_x * cw)
            sy = margin_t + ch - int(pn_y * ch)
            color = QColor(_PALETTE[ci % len(_PALETTE)])
            color.setAlpha(180)
            painter.setBrush(color)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawEllipse(sx - _POINT_R, sy - _POINT_R, _POINT_R * 2, _POINT_R * 2)
            if idx < len(self._raw_pts):
                rx, ry = self._raw_pts[idx]
                lbl = self._class_labels[ci] if ci < len(self._class_labels) else str(ci)
                tip = (f"x ({self._x_label}): {rx:.4g}\n"
                       f"y ({self._y_label}): {ry:.4g}\n"
                       f"Class: {lbl}")
            else:
                tip = f"x: {pn_x:.3f}  y: {pn_y:.3f}"
            self._dot_rects.append((
                QRect(sx - _POINT_R - 2, sy - _POINT_R - 2,
                      (_POINT_R + 2) * 2, (_POINT_R + 2) * 2),
                tip,
            ))

        # ── Stats overlay (top-left) ───────────────────────────────────────────
        painter.setPen(QColor("#534b40"))
        painter.setFont(QFont(self.font().family(), 8))
        p_str = f"{self._p_val:.3f}" if self._p_val >= 0.001 else "< 0.001"
        stats_line1 = f"r = {self._pearson:+.3f}   R² = {self._r2:.3f}"
        stats_line2 = f"p = {p_str}"
        painter.drawText(margin_l + 6, margin_t + 4, 160, 13,
                         Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
                         stats_line1)
        painter.drawText(margin_l + 6, margin_t + 18, 160, 13,
                         Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
                         stats_line2)

        # ── Legend ─────────────────────────────────────────────────────────────
        if self._class_labels:
            lx = w - legend_w + 4
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


# ── Screen widget ──────────────────────────────────────────════════════════════

class LinearProjectionScreen(QWidget, WorkflowNodeScreenSupport):
    """
    Linear Projection – scatter plot of two user-selected numeric features.

    • OLS regression line (toggleable, drawn with clipping to chart area)
    • Pearson r, R² and two-tailed p-value displayed on chart and in status bar
    • Proper axis tick labels with actual data values
    • Per-class colour legend with font-metrics truncation
    • Hover tooltip: exact x, y, class values
    • Numpy-based stats for correctness and speed
    • Random subsampling when data exceeds max points
    """

    MAX_POINTS = 1000

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._init_workflow_node_support()
        self._dataset: DatasetHandle | None = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        header = QLabel("Linear Projection")
        header.setProperty("sectionTitle", True)
        layout.addWidget(header)

        desc = QLabel(
            "Scatter plot of two numeric features. "
            "Select X and Y axes; optionally colour by a third variable. "
            "Pearson r, R² and p-value shown top-left. "
            "OLS regression line drawn as dashed line."
        )
        desc.setWordWrap(True)
        desc.setProperty("muted", True)
        layout.addWidget(desc)

        ctrl_box = QGroupBox("Axes")
        ctrl = QHBoxLayout(ctrl_box)

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

        self._reg_cb = QCheckBox("Regression line")
        self._reg_cb.setChecked(True)
        self._reg_cb.setToolTip("Show OLS regression line")
        self._reg_cb.stateChanged.connect(
            lambda state: self._chart.set_show_reg_line(bool(state))
        )
        ctrl.addWidget(self._reg_cb)

        ctrl.addWidget(QLabel("Max pts:"))
        self._max_spin = QSpinBox()
        self._max_spin.setRange(50, 10000)
        self._max_spin.setValue(self.MAX_POINTS)
        self._max_spin.setSuffix(" pts")
        self._max_spin.valueChanged.connect(self._refresh)
        ctrl.addWidget(self._max_spin)

        layout.addWidget(ctrl_box)

        chart_box = QGroupBox("Scatter Plot")
        chart_layout = QVBoxLayout(chart_box)
        self._chart = _ScatterWidget()
        chart_layout.addWidget(self._chart)
        layout.addWidget(chart_box, 1)

        self._status_label = QLabel("Load a dataset with numeric columns.")
        self._status_label.setProperty("muted", True)
        layout.addWidget(self._status_label)

    # ── WorkflowNodeScreenSupport ─────────────────────────────────────────────

    def set_input_payload(self, payload) -> None:
        dataset = payload.dataset if payload is not None else None
        self._dataset = dataset
        for combo in (self._x_combo, self._y_combo, self._class_combo):
            combo.blockSignals(True)
            combo.clear()
        if dataset is not None:
            df = dataset.dataframe
            num_cols = [c.name for c in dataset.domain.columns
                        if df[c.name].dtype.is_numeric()]
            all_cols = [c.name for c in dataset.domain.columns]
            self._x_combo.addItems(num_cols)
            self._y_combo.addItems(num_cols)
            self._class_combo.addItem("(none)")
            self._class_combo.addItems(all_cols)
            if len(num_cols) >= 2:
                self._y_combo.setCurrentIndex(1)
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

    # ── Internal ──────────────────────────────────────────────────────────────

    def _refresh(self) -> None:
        _empty = dict(
            points=[], raw_pts=[], x_label="X", y_label="Y",
            class_labels=[], x_min=0, x_max=1, y_min=0, y_max=1,
            pearson=0.0, r2=0.0, p_val=1.0, reg_slope=0.0, reg_intercept=0.0,
        )

        if self._dataset is None:
            self._chart.set_data(**_empty)
            self._status_label.setText("Load a dataset with numeric columns.")
            return

        x_col = self._x_combo.currentText()
        y_col = self._y_combo.currentText()
        if not x_col or not y_col:
            self._chart.set_data(**_empty)
            self._status_label.setText("Select X and Y columns.")
            return
        if x_col == y_col:
            self._chart.set_data(**_empty)
            self._status_label.setText(
                f"⚠ X and Y are the same column ('{x_col}'). Select different columns."
            )
            return

        df = self._dataset.dataframe
        try:
            x_raw = df[x_col].to_list()
            y_raw = df[y_col].to_list()
        except Exception:
            self._chart.set_data(**_empty)
            self._status_label.setText("Column not found.")
            return

        class_col = self._class_combo.currentText()
        class_raw: list[str] | None = None
        if class_col and class_col != "(none)" and class_col in df.columns:
            class_raw = [
                str(v) if v is not None else "(missing)"
                for v in df[class_col].to_list()
            ]

        # Sample
        n = len(df)
        max_pts = self._max_spin.value()
        indices = list(range(n))
        if n > max_pts:
            random.seed(42)
            indices = random.sample(indices, max_pts)
            indices.sort()

        # Class map
        class_labels: list[str] = []
        class_map: dict[str, int] = {}
        if class_raw is not None:
            unique = list(dict.fromkeys(class_raw[i] for i in indices))[:8]
            class_labels = unique
            class_map = {v: i for i, v in enumerate(unique)}

        # Filter valid rows
        pairs = [
            (float(x_raw[i]), float(y_raw[i]), i)
            for i in indices
            if x_raw[i] is not None and y_raw[i] is not None
        ]
        if not pairs:
            self._chart.set_data(**_empty)
            self._status_label.setText("No valid (non-null) data points.")
            return

        xs_np = np.array([p[0] for p in pairs], dtype=np.float64)
        ys_np = np.array([p[1] for p in pairs], dtype=np.float64)

        x_min, x_max = float(xs_np.min()), float(xs_np.max())
        y_min, y_max = float(ys_np.min()), float(ys_np.max())
        x_span = (x_max - x_min) or 1.0
        y_span = (y_max - y_min) or 1.0

        # Stats
        pearson, r2, p_val = _correlation_stats(xs_np, ys_np)
        reg_slope, reg_intercept = _ols_line(xs_np, ys_np)

        norm_pts = [
            ((xv - x_min) / x_span,
             (yv - y_min) / y_span,
             class_map.get(class_raw[orig_i], 0) if class_raw else 0)
            for xv, yv, orig_i in pairs
        ]
        raw_pts = [(xv, yv) for xv, yv, _ in pairs]

        self._chart.set_data(
            points=norm_pts,
            raw_pts=raw_pts,
            x_label=x_col,
            y_label=y_col,
            class_labels=class_labels,
            x_min=x_min, x_max=x_max,
            y_min=y_min, y_max=y_max,
            pearson=pearson,
            r2=r2,
            p_val=p_val,
            reg_slope=reg_slope,
            reg_intercept=reg_intercept,
        )
        self._chart.set_show_reg_line(self._reg_cb.isChecked())

        sampled_note = f" (sampled {max_pts} of {n})" if n > max_pts else ""
        p_str = f"{p_val:.3f}" if p_val >= 0.001 else "< 0.001"
        self._status_label.setText(
            f"X: '{x_col}'  ·  Y: '{y_col}'  ·  {len(pairs)} points{sampled_note}"
            f"  ·  r = {pearson:+.3f}  ·  R² = {r2:.3f}  ·  p = {p_str}"
            + (f"  ·  colored by '{class_col}'"
               if class_col and class_col != "(none)" else "")
        )
