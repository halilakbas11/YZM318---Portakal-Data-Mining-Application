from __future__ import annotations

import math

from PySide6.QtCore import Qt, QRect
from PySide6.QtGui import QColor, QPainter, QPen, QFont, QFontMetrics
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QSizePolicy,
    QToolTip,
    QVBoxLayout,
    QWidget,
)

from portakal_app.data.models import DatasetHandle
from portakal_app.ui import i18n
from portakal_app.ui.screens.node_screen import WorkflowNodeScreenSupport

# ── Palette for category colouring ────────────────────────────────────────────
_PALETTE = [
    QColor("#e07020"), QColor("#3b82f6"), QColor("#22c55e"),
    QColor("#a855f7"), QColor("#f43f5e"), QColor("#0ea5e9"),
    QColor("#f59e0b"), QColor("#10b981"),
]

# ── Orange-style Pearson residual colour scale (4 discrete levels) ─────────────
# Matches OWMosaicDisplay.BLUE_COLORS / RED_COLORS exactly.
_BLUE_COLORS = [
    QColor(255, 255, 255),   # ind=0  |p| < 2
    QColor(210, 210, 255),   # ind=1  2 ≤ |p| < 4
    QColor(110, 110, 255),   # ind=2  4 ≤ |p| < 8
    QColor(0,   0,   255),   # ind=3  |p| ≥ 8
]
_RED_COLORS = [
    QColor(255, 255, 255),
    QColor(255, 200, 200),
    QColor(255, 100, 100),
    QColor(255,   0,   0),
]


def _residual_color(pearson: float) -> QColor:
    """Orange OWMosaicDisplay formula: ind = int(log2(|p|)), clamped [0,3]."""
    if pearson == 0:
        return QColor(220, 220, 220)
    ind = max(0, min(int(math.log(abs(pearson), 2)), 3))
    return (_BLUE_COLORS if pearson > 0 else _RED_COLORS)[ind]


# ── Statistics ─────────────────────────────────────────────────────────────────

def _chi2_sf(x: float, k: int) -> float:
    """Survival function of chi-square (Wilson-Hilferty approximation)."""
    if x <= 0:
        return 1.0
    z = ((x / k) ** (1 / 3) - (1 - 2 / (9 * k))) / math.sqrt(2 / (9 * k))
    if z > 6:
        return 0.0
    if z < -6:
        return 1.0
    return 0.5 * math.erfc(z / math.sqrt(2))


def _compute_stats(
    joint: dict[str, dict[str, int]],
    col_totals: dict[str, int],
    row_totals: dict[str, int],
    n_total: int,
) -> tuple[float, float, float]:
    """Return (χ², p-value, Cramér's V)."""
    if n_total == 0:
        return 0.0, 1.0, 0.0
    chi2 = 0.0
    for cv, col_dict in joint.items():
        for rv, observed in col_dict.items():
            expected = (col_totals.get(cv, 0) * row_totals.get(rv, 0)) / n_total
            if expected > 0:
                chi2 += (observed - expected) ** 2 / expected
    n_rows = len(row_totals)
    n_cols = len(col_totals)
    dof = max(1, (n_rows - 1) * (n_cols - 1))
    p = _chi2_sf(chi2, dof)
    k = min(n_rows - 1, n_cols - 1)
    v = math.sqrt(chi2 / (n_total * k)) if k > 0 else 0.0
    return chi2, p, v


# ── Canvas widget ──────────────────────────────────────────────────────────────

class _MosaicWidget(QWidget):
    """
    Mosaic Display canvas.

    • Column width ∝ marginal proportion of the X variable
    • Cell height ∝ conditional proportion of Y variable within each X category
    • Colour: either category palette OR Orange-style 4-level Pearson residual scale
    • Hover tooltip: count, %, expected, Pearson residual
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._joint: dict[str, dict[str, int]] = {}   # {col_val: {row_val: count}}
        self._col_totals: dict[str, int] = {}
        self._row_totals: dict[str, int] = {}
        self._col_vals: list[str] = []
        self._row_vals: list[str] = []
        self._col_var = ""
        self._row_var = ""
        self._n_total = 0
        self._color_by_residual = False
        self._cell_rects: list[tuple[QRect, str, str, int, float, float]] = []
        self.setMinimumHeight(200)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMouseTracking(True)

    def set_data(
        self,
        joint: dict[str, dict[str, int]],
        col_totals: dict[str, int],
        row_totals: dict[str, int],
        col_vals: list[str],
        row_vals: list[str],
        col_var: str,
        row_var: str,
        n_total: int,
        color_by_residual: bool = False,
    ) -> None:
        self._joint = joint
        self._col_totals = col_totals
        self._row_totals = row_totals
        self._col_vals = col_vals
        self._row_vals = row_vals
        self._col_var = col_var
        self._row_var = row_var
        self._n_total = n_total
        self._color_by_residual = color_by_residual
        self._cell_rects = []
        self.update()

    # ── Tooltip ────────────────────────────────────────────────────────────────

    def mouseMoveEvent(self, event) -> None:
        pos = event.position().toPoint()
        for rect, cv, rv, count, expected, pearson in self._cell_rects:
            if rect.contains(pos):
                pct = 100.0 * count / self._n_total if self._n_total else 0
                tip = (
                    f"<b>{self._col_var}</b>: {cv}<br>"
                    f"<b>{self._row_var}</b>: {rv}<br>"
                    f"Count: {count} ({pct:.1f}%)<br>"
                    f"Expected: {expected:.2f}<br>"
                    f"Pearson r: {pearson:+.3f}"
                )
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

        if not self._joint:
            painter.setPen(QColor("#8d877d"))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter,
                             "No data.\nSelect two categorical columns.")
            return

        self._cell_rects = []

        w, h = self.width(), self.height()
        margin_l = 14
        margin_t = 46
        margin_r = 14
        margin_b = 54

        chart_w = w - margin_l - margin_r
        chart_h = h - margin_t - margin_b
        if chart_w < 10 or chart_h < 10:
            return

        n_total = self._n_total or 1
        n_cols = len(self._col_vals)
        gap = 3

        # Column widths proportional to marginal frequency
        avail_w = chart_w - gap * max(0, n_cols - 1)
        col_widths = [
            max(4, int(self._col_totals.get(cv, 0) / n_total * avail_w))
            for cv in self._col_vals
        ]
        # Snap last column to avoid rounding drift
        if col_widths:
            col_widths[-1] = max(4, avail_w - sum(col_widths[:-1]))

        painter.setFont(QFont(self.font().family(), 8))
        fm = QFontMetrics(painter.font())

        # ── X variable label (top) ─────────────────────────────────────────────
        painter.setPen(QColor("#3b2a10"))
        painter.setFont(QFont(self.font().family(), 9, QFont.Weight.Bold))
        painter.drawText(margin_l, 2, chart_w, 16, Qt.AlignmentFlag.AlignCenter, self._col_var)

        # ── Y variable label (left, rotated) ───────────────────────────────────
        if self._row_var:
            painter.save()
            painter.translate(8, margin_t + chart_h // 2)
            painter.rotate(-90)
            painter.setPen(QColor("#3b2a10"))
            painter.drawText(-60, -8, 120, 16, Qt.AlignmentFlag.AlignCenter, self._row_var)
            painter.restore()

        painter.setFont(QFont(self.font().family(), 8))
        fm = QFontMetrics(painter.font())

        x = margin_l
        for ci, cv in enumerate(self._col_vals):
            cw = col_widths[ci]
            col_total = self._col_totals.get(cv, 1) or 1

            # Column header — font-metrics truncation
            painter.setPen(QColor("#534b40"))
            lbl = fm.elidedText(cv, Qt.TextElideMode.ElideRight, max(8, cw - 4))
            painter.drawText(x, 18, cw, margin_t - 20,
                             Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignBottom, lbl)

            y = margin_t
            for ri, rv in enumerate(self._row_vals):
                count = self._joint.get(cv, {}).get(rv, 0)
                prop = count / col_total
                cell_h = max(0, int(prop * chart_h))
                if ri == len(self._row_vals) - 1:
                    # Snap last cell to avoid 1-pixel gaps from rounding
                    cell_h = max(0, margin_t + chart_h - y)

                # Pearson residual
                row_total = self._row_totals.get(rv, 0)
                expected = (col_total / n_total) * (row_total / n_total) * n_total
                pearson = (count - expected) / math.sqrt(expected) if expected > 0 else 0.0

                if self._color_by_residual:
                    color = _residual_color(pearson)
                else:
                    color = _PALETTE[ri % len(_PALETTE)]

                painter.setBrush(color)
                painter.setPen(QPen(QColor("#fffdf9"), 1))
                painter.drawRect(x, y, cw, cell_h)

                # Count label inside cell — adaptive text colour
                if cell_h > 14 and cw > 24:
                    lum = (color.red() + color.green() + color.blue()) / 765.0
                    txt_col = QColor("#111111") if lum > 0.6 else QColor("#ffffff")
                    painter.setPen(txt_col)
                    painter.drawText(x + 2, y + 2, cw - 4, cell_h - 4,
                                     Qt.AlignmentFlag.AlignCenter, str(count))

                self._cell_rects.append((
                    QRect(x, y, cw, cell_h), cv, rv, count, expected, pearson,
                ))
                y += cell_h

            x += cw + gap

        # ── Legend ─────────────────────────────────────────────────────────────
        legend_y = h - margin_b + 10

        if self._color_by_residual:
            painter.setPen(QColor("#534b40"))
            painter.drawText(margin_l, legend_y - 2, 130, 14,
                             Qt.AlignmentFlag.AlignVCenter, "Pearson residual:")
            legend_items = [
                ("<-8",   _RED_COLORS[3]),
                ("-8:-4", _RED_COLORS[2]),
                ("-4:-2", _RED_COLORS[1]),
                ("-2:2",  QColor(220, 220, 220)),
                ("2:4",   _BLUE_COLORS[1]),
                ("4:8",   _BLUE_COLORS[2]),
                (">8",    _BLUE_COLORS[3]),
            ]
            lx = margin_l
            for label, color in legend_items:
                if lx + 50 > w:
                    break
                painter.setBrush(color)
                painter.setPen(Qt.PenStyle.NoPen)
                painter.drawRect(lx, legend_y + 16, 12, 12)
                painter.setPen(QColor("#534b40"))
                painter.drawText(lx + 14, legend_y + 16, 44, 14,
                                 Qt.AlignmentFlag.AlignVCenter, label)
                lx += 58
        else:
            painter.setPen(QColor("#534b40"))
            painter.drawText(margin_l, legend_y - 2, w - margin_l, 14,
                             Qt.AlignmentFlag.AlignVCenter, f"{self._row_var}:")
            lx = margin_l
            for ri, rv in enumerate(self._row_vals):
                if lx + 100 > w:
                    break
                color = _PALETTE[ri % len(_PALETTE)]
                painter.setBrush(color)
                painter.setPen(Qt.PenStyle.NoPen)
                painter.drawRect(lx, legend_y + 16, 12, 12)
                painter.setPen(QColor("#534b40"))
                # Font-metrics truncation for legend labels
                lbl = fm.elidedText(rv, Qt.TextElideMode.ElideRight, 88)
                painter.drawText(lx + 16, legend_y + 16, 90, 14,
                                 Qt.AlignmentFlag.AlignVCenter, lbl)
                lx += 110

        painter.end()


# ── Screen widget ──────────────────────────────────────────────────────────────

class MosaicDisplayScreen(QWidget, WorkflowNodeScreenSupport):
    """
    Mosaic Display — proportional representation of two categorical variables.

    • Column width = marginal proportion of X variable
    • Cell height = conditional proportion of Y variable within X category
    • Interior coloring: category palette OR Orange's 4-level Pearson residual scale
    • Shows χ², Cramér's V (effect size), and p-value in status bar
    • Hover tooltip: count, %, expected count, Pearson residual
    • Sort order control for category axes
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._init_workflow_node_support()
        self._dataset: DatasetHandle | None = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        header = QLabel("Mosaic Display")
        header.setProperty("sectionTitle", True)
        layout.addWidget(header)

        desc = QLabel(
            "Column width ∝ first variable frequency. "
            "Cell height ∝ conditional second variable frequency. "
            "Blue = more than expected, Red = less than expected."
        )
        desc.setWordWrap(True)
        desc.setProperty("muted", True)
        layout.addWidget(desc)

        ctrl_box = QGroupBox("Variables")
        ctrl = QHBoxLayout(ctrl_box)

        ctrl.addWidget(QLabel("X (columns):"))
        self._col_combo = QComboBox()
        self._col_combo.currentTextChanged.connect(self._refresh)
        ctrl.addWidget(self._col_combo, 1)

        ctrl.addWidget(QLabel("Y (color):"))
        self._row_combo = QComboBox()
        self._row_combo.currentTextChanged.connect(self._refresh)
        ctrl.addWidget(self._row_combo, 1)

        ctrl.addWidget(QLabel("Sort:"))
        self._sort_combo = QComboBox()
        self._sort_combo.addItems(["By frequency ↓", "By frequency ↑", "Alphabetically"])
        self._sort_combo.currentIndexChanged.connect(self._refresh)
        ctrl.addWidget(self._sort_combo)

        self._residual_cb = QCheckBox("Color by Pearson residual")
        self._residual_cb.setChecked(True)
        self._residual_cb.setToolTip("Blue = more than expected  ·  Red = less than expected")
        self._residual_cb.stateChanged.connect(self._refresh)
        ctrl.addWidget(self._residual_cb)

        layout.addWidget(ctrl_box)

        chart_box = QGroupBox("Mosaic")
        chart_layout = QVBoxLayout(chart_box)
        self._chart = _MosaicWidget()
        chart_layout.addWidget(self._chart)
        layout.addWidget(chart_box, 1)

        self._status_label = QLabel(i18n.t("Load a dataset to display a mosaic plot."))
        self._status_label.setProperty("muted", True)
        layout.addWidget(self._status_label)

    # ── WorkflowNodeScreenSupport ─────────────────────────────────────────────

    def set_input_payload(self, payload) -> None:
        dataset = payload.dataset if payload is not None else None
        self._dataset = dataset
        for combo in (self._col_combo, self._row_combo):
            combo.blockSignals(True)
            combo.clear()
        if dataset is not None:
            cat_cols = self._categorical_columns()
            self._col_combo.addItems(cat_cols)
            self._row_combo.addItems(cat_cols)
            # Auto-select the target (class) column as Y (color) — matches Orange
            target_selected = False
            for col in dataset.domain.target_columns:
                idx = self._row_combo.findText(col.name)
                if idx >= 0:
                    self._row_combo.setCurrentIndex(idx)
                    target_selected = True
                    break
            if not target_selected and len(cat_cols) >= 2:
                self._row_combo.setCurrentIndex(1)
        for combo in (self._col_combo, self._row_combo):
            combo.blockSignals(False)
        self._refresh()

    def documentation_url(self) -> str:
        return "https://orangedatamining.com/widget-catalog/visualize/mosaicdisplay/"

    # ── Internal ──────────────────────────────────────────────────────────────

    def _categorical_columns(self) -> list[str]:
        """Return ALL columns (including target columns) — numeric columns with
        many unique values are auto-binned when building the contingency table."""
        if self._dataset is None:
            return []
        cols = [col.name for col in self._dataset.domain.columns]
        target_cols = [col.name for col in self._dataset.domain.target_columns]
        # Add target columns that aren't already in regular columns
        for t in target_cols:
            if t not in cols:
                cols.append(t)
        return cols

    def _discretize(self, series_vals: list, col_name: str) -> list[str]:
        """
        Convert column values to category labels.
        • Non-numeric / low-cardinality numeric (≤ 15 unique) → str(value)
        • High-cardinality numeric → 5 equal-width bins like "[0.0, 2.0)"
          Matches Orange's auto-discretization style.
        """
        if self._dataset is None:
            return [str(v) if v is not None else "(missing)" for v in series_vals]
        df = self._dataset.dataframe
        try:
            is_num = df[col_name].dtype.is_numeric()
        except Exception:
            is_num = False

        if not is_num:
            return [str(v) if v is not None else "(missing)" for v in series_vals]

        non_null = [v for v in series_vals if v is not None]
        if len(set(non_null)) <= 15:
            return [str(int(v)) if isinstance(v, float) and v == int(v)
                    else (str(v) if v is not None else "(missing)")
                    for v in series_vals]

        try:
            nums = [float(v) for v in non_null]
            lo, hi = min(nums), max(nums)
            if hi == lo:
                return [str(v) if v is not None else "(missing)" for v in series_vals]
            n_bins = 5
            step = (hi - lo) / n_bins

            def _label(v) -> str:
                if v is None:
                    return "(missing)"
                fv = float(v)
                b = min(int((fv - lo) / step), n_bins - 1)
                b_lo = lo + b * step
                b_hi = lo + (b + 1) * step
                return f"[{b_lo:.2g}, {b_hi:.2g})"

            return [_label(v) for v in series_vals]
        except Exception:
            return [str(v) if v is not None else "(missing)" for v in series_vals]

    def _sort_categories(self, counts: dict[str, int], limit: int) -> list[str]:
        """Sort category labels according to the current sort combo selection."""
        mode = self._sort_combo.currentText()
        if mode == "By frequency ↓":
            ordered = sorted(counts.items(), key=lambda x: -x[1])
        elif mode == "By frequency ↑":
            ordered = sorted(counts.items(), key=lambda x: x[1])
        else:  # Alphabetically
            ordered = sorted(counts.items(), key=lambda x: x[0])
        return [k for k, _ in ordered[:limit]]

    def _refresh(self) -> None:
        if self._dataset is None:
            self._chart.set_data({}, {}, {}, [], [], "", "", 0)
            self._status_label.setText(i18n.t("Load a dataset to display a mosaic plot."))
            return

        col_name = self._col_combo.currentText()
        row_name = self._row_combo.currentText()

        if not col_name or not row_name or col_name == row_name:
            self._chart.set_data({}, {}, {}, [], [], "", "", 0)
            self._status_label.setText(i18n.t("Select two different columns."))
            return

        df = self._dataset.dataframe
        try:
            col_raw = df[col_name].to_list()
            row_raw = df[row_name].to_list()
            # Auto-bin numeric columns with many unique values
            col_series = self._discretize(col_raw, col_name)
            row_series = self._discretize(row_raw, row_name)
        except Exception:
            self._chart.set_data({}, {}, {}, [], [], "", "", 0)
            self._status_label.setText(i18n.t("Column not found."))
            return

        n_total = len(col_series)
        col_counts: dict[str, int] = {}
        row_counts: dict[str, int] = {}
        joint: dict[str, dict[str, int]] = {}

        for ck, rk in zip(col_series, row_series):
            col_counts[ck] = col_counts.get(ck, 0) + 1
            row_counts[rk] = row_counts.get(rk, 0) + 1
            joint.setdefault(ck, {})
            joint[ck][rk] = joint[ck].get(rk, 0) + 1

        col_vals = self._sort_categories(col_counts, 12)
        row_vals = self._sort_categories(row_counts, 8)

        chi2, p_val, v = _compute_stats(joint, col_counts, row_counts, n_total)

        self._chart.set_data(
            joint, col_counts, row_counts,
            col_vals, row_vals,
            col_name, row_name, n_total,
            color_by_residual=self._residual_cb.isChecked(),
        )

        p_str = f"{p_val:.4f}" if p_val >= 0.0001 else "< 0.0001"
        self._status_label.setText(
            f"X: '{col_name}' ({len(col_vals)} cats)  ·  "
            f"Y: '{row_name}' ({len(row_vals)} cats)  ·  "
            f"N = {n_total}  ·  χ² = {chi2:.2f}  ·  V = {v:.3f}  ·  p = {p_str}"
        )
