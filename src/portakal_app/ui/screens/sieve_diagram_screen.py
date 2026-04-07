from __future__ import annotations

import math

from PySide6.QtCore import Qt, QRect, QPoint, QLineF
from PySide6.QtGui import QColor, QPainter, QPen, QFont, QFontMetrics
from PySide6.QtWidgets import (
    QComboBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QToolTip,
    QVBoxLayout,
    QWidget,
)

from portakal_app.data.models import DatasetHandle
from portakal_app.ui import i18n
from portakal_app.ui.screens.node_screen import WorkflowNodeScreenSupport


# ── Statistics ─────────────────────────────────────────────────────────────────

def _pearson_residual(observed: int, expected: float) -> float:
    """r = (O − E) / √E  (Orange's exact formula from owsieve.py)."""
    if expected <= 0:
        return 0.0
    return (observed - expected) / math.sqrt(expected)


def _chi_square_and_p(
    joint: dict[str, dict[str, int]],
    row_totals: dict[str, int],
    col_totals: dict[str, int],
    n_total: int,
) -> tuple[float, float, bool]:
    """Return (χ², p-value, cochran_ok).

    Cochran's rule: all expected counts ≥ 5 (warn otherwise).
    p-value computed from chi-square CDF (Wilson–Hilferty approximation).
    """
    chi2 = 0.0
    cochran_ok = True
    for rv, row_dict in joint.items():
        for cv, observed in row_dict.items():
            expected = (row_totals.get(rv, 0) * col_totals.get(cv, 0)) / max(1, n_total)
            if expected < 5:
                cochran_ok = False
            if expected > 0:
                chi2 += (observed - expected) ** 2 / expected

    n_rows = len(row_totals)
    n_cols = len(col_totals)
    dof = max(1, (n_rows - 1) * (n_cols - 1))
    p = _chi2_sf(chi2, dof)
    return chi2, p, cochran_ok


def _cramers_v(chi2: float, n_total: int, n_rows: int, n_cols: int) -> float:
    """Cramér's V — effect size for chi-square.

    V = √( χ² / (N · min(r−1, c−1)) )
    Range [0, 1]: 0.1 small · 0.3 medium · 0.5+ large.
    """
    k = min(n_rows - 1, n_cols - 1)
    if k <= 0 or n_total <= 0:
        return 0.0
    return math.sqrt(chi2 / (n_total * k))


def _chi2_sf(x: float, k: int) -> float:
    """Survival function of chi-square distribution (Wilson-Hilferty approx)."""
    if x <= 0:
        return 1.0
    z = ((x / k) ** (1 / 3) - (1 - 2 / (9 * k))) / math.sqrt(2 / (9 * k))
    return _normal_sf(z)


def _normal_sf(z: float) -> float:
    """Survival function of standard normal (Abramowitz & Stegun 26.2.17)."""
    if z > 6:
        return 0.0
    if z < -6:
        return 1.0
    return 0.5 * math.erfc(z / math.sqrt(2))


# ── Color helpers (Orange owsieve.py exact formulas) ───────────────────────────

def _sieve_fill_color(pearson: float) -> QColor:
    """
    Orange's exact sieve fill colour from show_pearson():
        if pearson > 0: r=g=max(int(255 - 20*pearson), 55), b=255  → blue tones
        if pearson < 0: b=g=max(int(255 + 20*pearson), 55), r=255  → red tones
        if pearson == 0: grey
    """
    if pearson > 0:
        rg = max(int(255 - 20 * pearson), 55)
        return QColor(rg, rg, 255)
    elif pearson < 0:
        bg = max(int(255 + 20 * pearson), 55)
        return QColor(255, bg, bg)
    return QColor(200, 200, 200)


def _sieve_hatch_spacing(pearson: float) -> float:
    """
    Orange's hatch-line spacing formula from show_pearson():
        positive: dist = 20 - 1.6 * pearson   → denser as pearson increases
        negative: dist = 20 - 8  * pearson     → sparser as |pearson| increases
                                                   (spacing grows beyond cell, painter clips)
    Lower-clamp to 3 only — no upper clamp, matching Orange (painter setClipRect handles overflow).
    """
    if pearson >= 0:
        dist = 20 - 1.6 * pearson
    else:
        dist = 20 - 8 * pearson   # pearson<0 → dist>20, sparser; painter clips to inner rect
    return max(3.0, dist)


# ── Canvas widget ──────────────────────────────────────────────────────────────

class _SieveWidget(QWidget):
    """
    Sieve Diagram (Riedwyl & Schüpbach, 1983).

    • Rectangle SIZE ∝ √(expected)  → area ∝ expected
    • Colour:  Orange's exact formula — blue for r > 0, red for r < 0
    • Hatch lines:
        pearson > 0 → / direction (lower-left to upper-right)
        pearson < 0 → \\ direction (upper-left to lower-right)
        density ∝ |pearson| (Orange's exact formula)
    • Hover tooltip showing O, E, r, χ² contribution
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        # {row_val: {col_val: (actual, expected, pearson_r)}}
        self._data: dict[str, dict[str, tuple[int, float, float]]] = {}
        self._row_vals: list[str] = []
        self._col_vals: list[str] = []
        self._row_var = ""
        self._col_var = ""
        # For tooltip hit-testing
        self._cell_rects: list[tuple[QRect, str, str, int, float, float]] = []
        self.setMinimumHeight(200)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMouseTracking(True)

    def set_data(
        self,
        data: dict[str, dict[str, tuple[int, float, float]]],
        row_vals: list[str],
        col_vals: list[str],
        row_var: str = "",
        col_var: str = "",
    ) -> None:
        self._data = data
        self._row_vals = row_vals
        self._col_vals = col_vals
        self._row_var = row_var
        self._col_var = col_var
        self._cell_rects = []
        self.update()

    # ── Tooltip ────────────────────────────────────────────────────────────────

    def mouseMoveEvent(self, event) -> None:
        pos = event.position().toPoint()
        for rect, rv, cv, actual, expected, r in self._cell_rects:
            if rect.contains(pos):
                chi2_contrib = (actual - expected) ** 2 / expected if expected > 0 else 0
                tip = (
                    f"<b>{self._row_var}</b>: {rv}<br>"
                    f"<b>{self._col_var}</b>: {cv}<br>"
                    f"Observed: {actual}<br>"
                    f"Expected: {expected:.2f}<br>"
                    f"Pearson r: {r:+.3f}<br>"
                    f"χ² contribution: {chi2_contrib:.3f}"
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

        if not self._data:
            painter.setPen(QColor("#8d877d"))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter,
                             "No data.\nSelect two categorical columns.")
            return

        self._cell_rects = []

        margin_l = 96
        margin_t = 48
        margin_r = 12
        margin_b = 60

        w, h = self.width(), self.height()
        chart_w = w - margin_l - margin_r
        chart_h = h - margin_t - margin_b
        if chart_w < 10 or chart_h < 10:
            return

        n_rows = len(self._row_vals)
        n_cols = len(self._col_vals)
        if not n_rows or not n_cols:
            return

        cell_w = chart_w / n_cols
        cell_h = chart_h / n_rows

        max_expected = max(
            (v[1] for row in self._data.values() for v in row.values() if v[1] > 0),
            default=1.0
        )

        painter.setFont(QFont(self.font().family(), 8))
        fm = QFontMetrics(painter.font())

        # ── Variable axis labels ───────────────────────────────────────────────
        painter.setPen(QColor("#3b2a10"))
        if self._col_var:
            painter.setFont(QFont(self.font().family(), 9, QFont.Weight.Bold))
            painter.drawText(margin_l, 2, chart_w, 16,
                             Qt.AlignmentFlag.AlignCenter, self._col_var)
        if self._row_var:
            painter.save()
            painter.translate(10, margin_t + chart_h // 2)
            painter.rotate(-90)
            painter.setFont(QFont(self.font().family(), 9, QFont.Weight.Bold))
            painter.drawText(-60, -8, 120, 16, Qt.AlignmentFlag.AlignCenter, self._row_var)
            painter.restore()

        painter.setFont(QFont(self.font().family(), 8))
        fm = QFontMetrics(painter.font())

        # ── Column headers — font-metrics truncation ───────────────────────────
        col_lbl_w = max(8, int(cell_w) - 4)
        for ci, cv in enumerate(self._col_vals):
            lbl = fm.elidedText(cv, Qt.TextElideMode.ElideRight, col_lbl_w * 3)
            painter.setPen(QColor("#534b40"))
            painter.drawText(
                int(margin_l + ci * cell_w), 18,
                int(cell_w), margin_t - 20,
                Qt.AlignmentFlag.AlignCenter, lbl,
            )

        # ── Row headers — font-metrics truncation ──────────────────────────────
        row_lbl_w = margin_l - 26
        for ri, rv in enumerate(self._row_vals):
            lbl = fm.elidedText(rv, Qt.TextElideMode.ElideRight, row_lbl_w)
            painter.setPen(QColor("#534b40"))
            painter.drawText(
                22, int(margin_t + ri * cell_h),
                row_lbl_w, int(cell_h),
                Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignRight, lbl,
            )

        # ── Cells ──────────────────────────────────────────────────────────────
        for ri, rv in enumerate(self._row_vals):
            for ci, cv in enumerate(self._col_vals):
                actual, expected, pearson_r = self._data.get(rv, {}).get(cv, (0, 0.0, 0.0))

                cx = int(margin_l + ci * cell_w)
                cy = int(margin_t + ri * cell_h)
                cw = int(cell_w)
                ch = int(cell_h)

                # Cell background
                painter.setBrush(QColor("#f0ede7"))
                painter.setPen(QPen(QColor("#d0ccc3"), 1))
                painter.drawRect(cx, cy, cw, ch)

                if expected > 0:
                    # Inner rectangle: area ∝ expected  →  side ∝ √expected
                    scale = math.sqrt(expected / max_expected)
                    inner_w = max(4, int(cw * 0.88 * scale))
                    inner_h = max(4, int(ch * 0.88 * scale))
                    ix = cx + (cw - inner_w) // 2
                    iy = cy + (ch - inner_h) // 2

                    # Fill colour (Orange formula)
                    fill = _sieve_fill_color(pearson_r)
                    painter.setBrush(fill)
                    painter.setPen(Qt.PenStyle.NoPen)
                    painter.drawRect(ix, iy, inner_w, inner_h)

                    # ── Hatch lines (Orange formula) ───────────────────────────
                    # pearson > 0 → / direction (lower-left to upper-right)
                    # pearson < 0 → \ direction (upper-left to lower-right)
                    spacing = _sieve_hatch_spacing(pearson_r)
                    painter.setPen(QPen(QColor(100, 100, 100, 120), 0.8))
                    painter.setClipRect(ix, iy, inner_w, inner_h)

                    if pearson_r >= 0:
                        # / direction: from (ix, iy+offset) → (ix+offset, iy)
                        # Mirrors Orange: drawLine(rect.x(), rect.y()+i, rect.x()+i+w, rect.y())
                        offset = 0.0
                        while offset < inner_w + inner_h:
                            x1 = float(ix)
                            y1 = float(iy + offset)
                            x2 = float(ix + offset + inner_w)
                            y2 = float(iy)
                            painter.drawLine(QLineF(x1, y1, x2, y2))
                            offset += spacing
                    else:
                        # \ direction: from (ix+offset, iy+inner_h) → (ix, iy+inner_h-offset)
                        # Mirrors Orange: drawLine(rect.x()+i, rect.y()+h, rect.x(), rect.y()+h-i)
                        offset = 0.0
                        while offset < inner_w + inner_h:
                            x1 = float(ix + offset)
                            y1 = float(iy + inner_h)
                            x2 = float(ix)
                            y2 = float(iy + inner_h - offset)
                            painter.drawLine(QLineF(x1, y1, x2, y2))
                            offset += spacing

                    painter.setClipping(False)

                    # Count label — show when cell is large enough
                    if cw > 32 and ch > 20:
                        painter.setPen(QColor("#1a1a1a"))
                        painter.drawText(
                            cx + 2, cy + 2, cw - 4, ch - 4,
                            Qt.AlignmentFlag.AlignCenter,
                            f"{actual}\n({pearson_r:+.2f})",
                        )

                # Track rect for tooltip
                self._cell_rects.append((
                    QRect(cx, cy, cw, ch),
                    rv, cv, actual, expected, pearson_r,
                ))

        # ── Legend ─────────────────────────────────────────────────────────────
        legend_y = h - margin_b + 10
        painter.setFont(QFont(self.font().family(), 8))
        items = [
            ("r > 0: more than expected  (/)", _sieve_fill_color(4.0)),
            ("r < 0: less than expected  (\\)", _sieve_fill_color(-4.0)),
        ]
        for li, (label, color) in enumerate(items):
            lx = margin_l + li * 220
            painter.setBrush(color)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawRect(lx, legend_y, 14, 12)
            painter.setPen(QColor("#534b40"))
            painter.drawText(lx + 18, legend_y, 190, 14,
                             Qt.AlignmentFlag.AlignVCenter, label)

        painter.end()


# ── Screen widget ──────────────────────────────────────────────────────────────

class SieveDiagramScreen(QWidget, WorkflowNodeScreenSupport):
    """
    Sieve Diagram (Riedwyl & Schüpbach, 1983) — Orange-faithful implementation.

    • Rectangle area ∝ expected frequency under independence
    • Colour: Blue = more than expected (r > 0), Red = less than expected (r < 0)
    • Hatch lines:
        / direction for r > 0 (positive association)
        \\ direction for r < 0 (negative association)
        density ∝ |Pearson residual| (exact Orange formula)
    • Shows χ², Cramér's V (effect size), and p-value
    • Warns on Cochran's rule violation (E < 5)
    • "Score Combinations" finds the pair with highest χ²
    • Sort order control for category labels
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._init_workflow_node_support()
        self._dataset: DatasetHandle | None = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        header = QLabel("Sieve Diagram")
        header.setProperty("sectionTitle", True)
        layout.addWidget(header)

        desc = QLabel(
            "Rectangle area ∝ expected frequency (independence). "
            "Colour = Pearson residual r = (O−E)/√E: "
            "Blue /→ more frequent · Red \\→ less frequent. "
            "Hatch-line density increases with |r|."
        )
        desc.setWordWrap(True)
        desc.setProperty("muted", True)
        layout.addWidget(desc)

        ctrl_box = QGroupBox("Variables")
        ctrl = QHBoxLayout(ctrl_box)
        ctrl.addWidget(QLabel("Row variable:"))
        self._row_combo = QComboBox()
        self._row_combo.currentTextChanged.connect(self._refresh)
        ctrl.addWidget(self._row_combo, 1)

        ctrl.addWidget(QLabel("Column variable:"))
        self._col_combo = QComboBox()
        self._col_combo.currentTextChanged.connect(self._refresh)
        ctrl.addWidget(self._col_combo, 1)

        ctrl.addWidget(QLabel("Sort:"))
        self._sort_combo = QComboBox()
        self._sort_combo.addItems(["By frequency ↓", "By frequency ↑", "Alphabetically"])
        self._sort_combo.currentIndexChanged.connect(self._refresh)
        ctrl.addWidget(self._sort_combo)

        self._best_btn = QPushButton("Score Combinations")
        self._best_btn.setToolTip("Find the pair with the highest χ² statistic")
        self._best_btn.clicked.connect(self._find_best_pair)
        ctrl.addWidget(self._best_btn)
        layout.addWidget(ctrl_box)

        chart_box = QGroupBox("Diagram")
        chart_layout = QVBoxLayout(chart_box)
        self._chart = _SieveWidget()
        chart_layout.addWidget(self._chart)
        layout.addWidget(chart_box, 1)

        self._status_label = QLabel(i18n.t("Load a dataset to visualise associations."))
        self._status_label.setProperty("muted", True)
        layout.addWidget(self._status_label)

    # ── WorkflowNodeScreenSupport ─────────────────────────────────────────────

    def set_input_payload(self, payload) -> None:
        dataset = payload.dataset if payload is not None else None
        self._dataset = dataset
        for combo in (self._row_combo, self._col_combo):
            combo.blockSignals(True)
            combo.clear()
        if dataset is not None:
            cat_cols = self._categorical_columns()
            self._row_combo.addItems(cat_cols)
            self._col_combo.addItems(cat_cols)
            if len(cat_cols) >= 2:
                self._col_combo.setCurrentIndex(1)
        for combo in (self._row_combo, self._col_combo):
            combo.blockSignals(False)
        self._refresh()

    def documentation_url(self) -> str:
        return "https://orangedatamining.com/widget-catalog/visualize/sievediagram/"

    # ── Internal ──────────────────────────────────────────────────────────────

    def _categorical_columns(self) -> list[str]:
        """Return ALL columns — numeric columns with many unique values are
        auto-binned when building the contingency table."""
        if self._dataset is None:
            return []
        return [col.name for col in self._dataset.domain.columns]

    def _discretize(self, series_vals: list, col_name: str) -> list[str]:
        """
        Convert a column's values to string category labels.
        • Non-numeric / low-cardinality numeric → str(value)
        • High-cardinality numeric (> 15 unique) → equal-width bins (5 bins)
          Labels like "[0.00, 2.00)", matching Orange's auto-discretize style.
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

        # Count unique non-null values
        non_null = [v for v in series_vals if v is not None]
        if len(set(non_null)) <= 15:
            # Low cardinality — keep as-is
            return [str(int(v)) if isinstance(v, float) and v == int(v)
                    else (str(v) if v is not None else "(missing)")
                    for v in series_vals]

        # High cardinality — 5 equal-width bins
        try:
            import math as _math
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

    def _build_contingency(self, row_col: str, col_col: str):
        df = self._dataset.dataframe
        row_raw = df[row_col].to_list()
        col_raw = df[col_col].to_list()
        # Auto-bin numeric columns with many unique values
        row_series = self._discretize(row_raw, row_col)
        col_series = self._discretize(col_raw, col_col)
        joint: dict[str, dict[str, int]] = {}
        row_totals: dict[str, int] = {}
        col_totals: dict[str, int] = {}
        for rk, ck in zip(row_series, col_series):
            row_totals[rk] = row_totals.get(rk, 0) + 1
            col_totals[ck] = col_totals.get(ck, 0) + 1
            joint.setdefault(rk, {})
            joint[rk][ck] = joint[rk].get(ck, 0) + 1
        return joint, row_totals, col_totals, len(df)

    def _sort_categories(
        self,
        totals: dict[str, int],
        limit: int,
    ) -> list[str]:
        """Sort category labels according to the current sort combo selection."""
        mode = self._sort_combo.currentText()
        if mode == "By frequency ↓":
            ordered = sorted(totals.items(), key=lambda x: -x[1])
        elif mode == "By frequency ↑":
            ordered = sorted(totals.items(), key=lambda x: x[1])
        else:  # Alphabetically
            ordered = sorted(totals.items(), key=lambda x: x[0])
        return [k for k, _ in ordered[:limit]]

    def _find_best_pair(self) -> None:
        if self._dataset is None:
            return
        cat_cols = self._categorical_columns()
        if len(cat_cols) < 2:
            return
        best_chi2 = -1.0
        best_pair = (cat_cols[0], cat_cols[1])
        for i, rc in enumerate(cat_cols):
            for cc in cat_cols[i + 1:]:
                try:
                    joint, row_totals, col_totals, n = self._build_contingency(rc, cc)
                    chi2, _, _ = _chi_square_and_p(joint, row_totals, col_totals, n)
                    if chi2 > best_chi2:
                        best_chi2 = chi2
                        best_pair = (rc, cc)
                except Exception:
                    continue
        self._row_combo.setCurrentText(best_pair[0])
        self._col_combo.setCurrentText(best_pair[1])

    def _refresh(self) -> None:
        if self._dataset is None:
            self._chart.set_data({}, [], [])
            self._status_label.setText(i18n.t("Load a dataset to visualise associations."))
            return

        row_col = self._row_combo.currentText()
        col_col = self._col_combo.currentText()

        if not row_col or not col_col or row_col == col_col:
            self._chart.set_data({}, [], [])
            self._status_label.setText(i18n.t("Select two different columns."))
            return

        try:
            joint, row_totals, col_totals, n_total = self._build_contingency(row_col, col_col)
        except Exception as exc:
            self._chart.set_data({}, [], [])
            self._status_label.setText(f"Error: {exc}")
            return

        # Sort and limit to top 10
        row_vals = self._sort_categories(row_totals, 10)
        col_vals = self._sort_categories(col_totals, 10)

        # Build visible-only sub-table for correct stats
        # (chi2/V/p must be computed on the displayed cells only)
        vis_joint: dict[str, dict[str, int]] = {}
        vis_row_totals: dict[str, int] = {}
        vis_col_totals: dict[str, int] = {}
        vis_n = 0
        for rv in row_vals:
            for cv in col_vals:
                actual = joint.get(rv, {}).get(cv, 0)
                vis_joint.setdefault(rv, {})[cv] = actual
                vis_row_totals[rv] = vis_row_totals.get(rv, 0) + actual
                vis_col_totals[cv] = vis_col_totals.get(cv, 0) + actual
                vis_n += actual

        # Build sieve data matrix using visible-subset marginals
        data: dict[str, dict[str, tuple[int, float, float]]] = {}
        for rv in row_vals:
            data[rv] = {}
            for cv in col_vals:
                actual = vis_joint.get(rv, {}).get(cv, 0)
                expected = (vis_row_totals.get(rv, 0) * vis_col_totals.get(cv, 0)) / max(1, vis_n)
                r = _pearson_residual(actual, expected)
                data[rv][cv] = (actual, expected, r)

        chi2, p_val, cochran_ok = _chi_square_and_p(
            vis_joint, vis_row_totals, vis_col_totals, vis_n
        )
        v = _cramers_v(chi2, vis_n, len(row_vals), len(col_vals))

        self._chart.set_data(data, row_vals, col_vals, row_col, col_col)

        p_str = f"{p_val:.4f}" if p_val >= 0.0001 else "< 0.0001"
        cochran_warn = "  ⚠ Cochran: some E < 5" if not cochran_ok else ""
        hidden_note = (
            f"  (showing {vis_n}/{n_total})"
            if vis_n < n_total else ""
        )
        self._status_label.setText(
            f"Rows: '{row_col}' ({len(row_vals)})  ·  "
            f"Columns: '{col_col}' ({len(col_vals)})  ·  "
            f"N = {vis_n}{hidden_note}  ·  χ² = {chi2:.2f}  ·  "
            f"V = {v:.3f}  ·  p = {p_str}"
            + cochran_warn
        )
