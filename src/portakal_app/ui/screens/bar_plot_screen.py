from __future__ import annotations

import math

from PySide6.QtCore import Qt, QPoint, QRect
from PySide6.QtGui import QColor, QPainter, QPen, QFont, QFontMetrics
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QScrollArea,
    QSizePolicy,
    QToolTip,
    QVBoxLayout,
    QWidget,
)

from portakal_app.data.models import DatasetHandle
from portakal_app.ui.screens.node_screen import WorkflowNodeScreenSupport

# ── Colour palette (matches rest of Portakal app) ─────────────────────────────
_PALETTE = [
    QColor("#e07020"), QColor("#3b82f6"), QColor("#22c55e"),
    QColor("#a855f7"), QColor("#f43f5e"), QColor("#0ea5e9"),
    QColor("#f59e0b"), QColor("#10b981"),
]

# Minimum pixels per bar before we give each one room to breathe
_MIN_BAR_PX = 6


class _BarChartCanvas(QWidget):
    """
    QPainter-based bar chart canvas. Features:

    - Zero-baseline (y = 0 is always visible; negative values extend downward)
    - Grouped side-by-side bars with thin outline for readability
    - Error bars (±1σ or Poisson ±√n) drawn as I-beams
    - Per-group legend (top-right corner)
    - Hover tooltips showing exact values
    - Expands horizontally so the parent QScrollArea can scroll when there are
      many bars
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        # (category_label, [(group_label, value), ...])
        self._groups: list[tuple[str, list[tuple[str, float]]]] = []
        # Parallel to _groups: (category_label, [(group_label, error_magnitude), ...])
        # error_magnitude > 0 → draw ±error I-beam on that bar
        self._errors: list[tuple[str, list[tuple[str, float]]]] = []
        self._y_label = "Value"
        self._show_values = True
        self._show_error_bars = True
        # group_label → palette index
        self._group_color_map: dict[str, int] = {}
        # For tooltip hit-testing: list of (QRect, bar_label, group_label, value, error)
        self._bar_rects: list[tuple[QRect, str, str, float, float]] = []

        self.setMinimumHeight(220)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMouseTracking(True)

    # ── Public API ─────────────────────────────────────────────────────────────

    def set_data(
        self,
        groups: list[tuple[str, list[tuple[str, float]]]],
        y_label: str = "Value",
        errors: list[tuple[str, list[tuple[str, float]]]] | None = None,
    ) -> None:
        self._groups = groups
        self._errors = errors or []
        self._y_label = y_label
        # Build stable group → palette index mapping
        seen: dict[str, int] = {}
        for _, sub in groups:
            for group_label, _ in sub:
                if group_label not in seen:
                    seen[group_label] = len(seen)
        self._group_color_map = seen
        self._bar_rects = []
        # Compute preferred width so scroll area knows how wide to be
        self._update_preferred_width()
        self.update()

    def set_show_values(self, show: bool) -> None:
        self._show_values = show
        self.update()

    def set_show_error_bars(self, show: bool) -> None:
        self._show_error_bars = show
        self.update()

    # ── Mouse → tooltip ────────────────────────────────────────────────────────

    def mouseMoveEvent(self, event) -> None:
        pos = event.position().toPoint()
        for rect, bar_lbl, grp_lbl, val, err in self._bar_rects:
            if rect.contains(pos):
                lines = [f"Instance: {bar_lbl}", f"Value: {val:.4g}"]
                if grp_lbl:
                    lines.insert(1, f"Group: {grp_lbl}")
                if err > 0:
                    lines.append(f"±{err:.4g}")
                QToolTip.showText(event.globalPosition().toPoint(), "\n".join(lines), self)
                return
        QToolTip.hideText()

    def leaveEvent(self, _event) -> None:
        QToolTip.hideText()

    # ── Drawing ────────────────────────────────────────────────────────────────

    def paintEvent(self, _event) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), QColor("#fffdf9"))

        if not self._groups:
            painter.setPen(QColor("#8d877d"))
            painter.drawText(
                self.rect(), Qt.AlignmentFlag.AlignCenter,
                "No data to display.\nLoad a dataset and select a column.",
            )
            return

        self._bar_rects = []

        w, h = self.width(), self.height()

        # Reserve space for legend on the right when there are named groups
        unique_groups = list(self._group_color_map.keys())
        has_legend = bool(unique_groups) and unique_groups != [""]
        legend_w = 0
        if has_legend:
            legend_w = max(len(g) for g in unique_groups) * 7 + 28
            legend_w = min(legend_w, 160)

        margin_l = 58
        margin_r = legend_w + 16
        margin_t = 20
        margin_b = 52
        chart_w = w - margin_l - margin_r
        chart_h = h - margin_t - margin_b
        if chart_w < 20 or chart_h < 20:
            return

        # Value range — always include 0 so baseline is visible.
        # Also extend range to accommodate error bars above/below.
        all_vals = [v for _, sub in self._groups for _, v in sub]
        if not all_vals:
            return

        err_lookup: dict[tuple[int, int], float] = {}
        for ci, (_, esub) in enumerate(self._errors):
            for gi, (_, err) in enumerate(esub):
                err_lookup[(ci, gi)] = err

        y_min_raw = min(0.0, min(all_vals))
        y_max_raw = max(0.0, max(all_vals))
        if err_lookup and self._show_error_bars:
            for ci, (_, sub) in enumerate(self._groups):
                for gi, (_, v) in enumerate(sub):
                    e = err_lookup.get((ci, gi), 0.0)
                    y_min_raw = min(y_min_raw, v - e)
                    y_max_raw = max(y_max_raw, v + e)

        # Add 5% padding above/below for readability
        raw_span = y_max_raw - y_min_raw
        pad = raw_span * 0.05 if raw_span > 0 else 0.5
        y_min = y_min_raw - pad
        y_max = y_max_raw + pad
        span = (y_max - y_min) or 1.0

        def val_to_y(v: float) -> int:
            """Convert data value to pixel y coordinate (top-left origin)."""
            return margin_t + int(chart_h * (1.0 - (v - y_min) / span))

        baseline_y = val_to_y(0.0)

        # ── Grid & Y axis ticks ────────────────────────────────────────────────
        grid_steps = 5
        nice_step = _nice_step((y_max_raw - y_min_raw) / grid_steps)
        tick_val = math.floor(y_min_raw / nice_step) * nice_step
        painter.setFont(QFont(self.font().family(), 8))
        fm = QFontMetrics(painter.font())
        while tick_val <= y_max_raw + nice_step * 0.01:
            py = val_to_y(tick_val)
            if margin_t <= py <= margin_t + chart_h:
                # Grid line
                painter.setPen(QPen(QColor("#e0ddd6"), 1, Qt.PenStyle.DotLine))
                painter.drawLine(margin_l, py, margin_l + chart_w, py)
                # Tick label
                painter.setPen(QColor("#8d877d"))
                label = _format_tick(tick_val)
                painter.drawText(2, py - 6, margin_l - 6, 14,
                                 Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
                                 label)
            tick_val = round(tick_val + nice_step, 10)

        # ── Axes ──────────────────────────────────────────────────────────────
        painter.setPen(QPen(QColor("#9b9488"), 1))
        painter.drawLine(margin_l, margin_t, margin_l, margin_t + chart_h)
        painter.drawLine(margin_l, margin_t + chart_h, margin_l + chart_w, margin_t + chart_h)
        # Zero baseline (only if data has negative values)
        if y_min_raw < 0:
            painter.setPen(QPen(QColor("#534b40"), 1))
            painter.drawLine(margin_l, baseline_y, margin_l + chart_w, baseline_y)

        # ── Y-axis label ──────────────────────────────────────────────────────
        painter.save()
        painter.translate(12, margin_t + chart_h // 2)
        painter.rotate(-90)
        painter.setPen(QColor("#534b40"))
        painter.setFont(QFont(self.font().family(), 9))
        painter.drawText(-50, -6, 100, 14, Qt.AlignmentFlag.AlignCenter, self._y_label)
        painter.restore()

        # ── Bars ──────────────────────────────────────────────────────────────
        painter.setFont(QFont(self.font().family(), 8))
        fm = QFontMetrics(painter.font())

        n_cats = len(self._groups)
        n_groups = max(len(sub) for _, sub in self._groups) if self._groups else 1
        cat_area = chart_w / max(1, n_cats)
        bar_w = max(_MIN_BAR_PX, int(cat_area * 0.80 / max(1, n_groups)))
        total_bar_w = bar_w * n_groups
        cat_gap = cat_area - total_bar_w

        for ci, (cat_label, sub) in enumerate(self._groups):
            cat_x = margin_l + int(ci * cat_area) + max(0, int(cat_gap / 2))

            for gi, (group_label, value) in enumerate(sub):
                bx = cat_x + gi * bar_w
                top_y = val_to_y(max(0.0, value))
                bot_y = val_to_y(min(0.0, value))
                bar_h = max(1, bot_y - top_y)

                color_idx = self._group_color_map.get(group_label, gi)
                color = _PALETTE[color_idx % len(_PALETTE)]

                # Bar fill
                painter.setBrush(color)
                painter.setPen(Qt.PenStyle.NoPen)
                painter.drawRoundedRect(bx, top_y, bar_w, bar_h, 2, 2)

                # Bar outline for readability
                outline = QColor(color)
                outline.setAlpha(120)
                painter.setPen(QPen(outline, 0.8))
                painter.setBrush(Qt.BrushStyle.NoBrush)
                painter.drawRoundedRect(bx, top_y, bar_w, bar_h, 2, 2)

                # ── Error bar ─────────────────────────────────────────────────
                err_val = err_lookup.get((ci, gi), 0.0)
                if self._show_error_bars and err_val > 0:
                    err_x = bx + bar_w // 2
                    err_top = val_to_y(value + err_val)
                    err_bot = val_to_y(value - err_val)
                    cap_half = max(3, bar_w // 5)
                    err_pen = QPen(QColor("#222"), 1.5)
                    err_pen.setCapStyle(Qt.PenCapStyle.RoundCap)
                    painter.setPen(err_pen)
                    painter.setBrush(Qt.BrushStyle.NoBrush)
                    painter.drawLine(err_x, err_top, err_x, err_bot)
                    painter.drawLine(err_x - cap_half, err_top, err_x + cap_half, err_top)
                    painter.drawLine(err_x - cap_half, err_bot, err_x + cap_half, err_bot)

                # Track rect for tooltip hit-testing
                self._bar_rects.append((
                    QRect(bx, top_y, bar_w, bar_h),
                    cat_label, group_label if group_label else "", value, err_val,
                ))

                # Value label inside bar (only when bar is tall & wide enough)
                if self._show_values and bar_h > 16 and bar_w > 20:
                    painter.setPen(QColor("#ffffff"))
                    txt = _format_tick(value)
                    painter.drawText(bx, top_y + 2, bar_w, 14,
                                     Qt.AlignmentFlag.AlignCenter, txt)

            # ── X tick label ──────────────────────────────────────────────────
            # Use font metrics for proper truncation
            max_label_px = max(8, int(cat_area) - 4)
            label = fm.elidedText(cat_label, Qt.TextElideMode.ElideRight, max(8, int(total_bar_w)))
            painter.setPen(QColor("#534b40"))
            # Rotate label if bars are narrow
            if cat_area < 40 and len(cat_label) > 3:
                painter.save()
                painter.translate(int(cat_x + total_bar_w / 2),
                                  margin_t + chart_h + 6)
                painter.rotate(40)
                painter.drawText(0, 0, 80, 14, Qt.AlignmentFlag.AlignLeft, label)
                painter.restore()
            else:
                painter.drawText(
                    int(cat_x), margin_t + chart_h + 4,
                    int(total_bar_w), 22,
                    Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop,
                    label,
                )

        # ── Legend ────────────────────────────────────────────────────────────
        if has_legend:
            lx = w - legend_w - 4
            ly = margin_t + 4
            row_h = 18
            box_h = len(unique_groups) * row_h + 10
            painter.setBrush(QColor(255, 255, 255, 200))
            painter.setPen(QPen(QColor("#ccc8c0"), 1))
            painter.drawRoundedRect(lx, ly, legend_w, box_h, 4, 4)
            painter.setFont(QFont(self.font().family(), 8))
            for li, grp in enumerate(unique_groups):
                color_idx = self._group_color_map.get(grp, li)
                color = _PALETTE[color_idx % len(_PALETTE)]
                iy = ly + 5 + li * row_h
                painter.setBrush(color)
                painter.setPen(Qt.PenStyle.NoPen)
                painter.drawRoundedRect(lx + 6, iy + 3, 10, 10, 2, 2)
                painter.setPen(QColor("#534b40"))
                glabel = fm.elidedText(grp, Qt.TextElideMode.ElideRight, legend_w - 24)
                painter.drawText(lx + 20, iy, legend_w - 24, row_h,
                                 Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft,
                                 glabel)

        painter.end()

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _update_preferred_width(self) -> None:
        n_cats = len(self._groups)
        n_groups = max((len(sub) for _, sub in self._groups), default=1)
        # Account for legend width — same calculation as paintEvent
        unique_groups = list(self._group_color_map.keys())
        has_legend = bool(unique_groups) and unique_groups != [""]
        legend_w = 0
        if has_legend:
            legend_w = max(len(g) for g in unique_groups) * 7 + 28
            legend_w = min(legend_w, 160)
        preferred = max(self.minimumWidth(),
                        n_cats * n_groups * max(_MIN_BAR_PX, 14) + 80 + legend_w + 16)
        self.setMinimumWidth(preferred)


# ── Helper functions ───────────────────────────────────────────────────────────

def _nice_step(raw: float) -> float:
    """Round to a human-friendly tick step."""
    if raw <= 0:
        return 1.0
    exp = math.floor(math.log10(raw))
    frac = raw / (10 ** exp)
    for nice in (1, 2, 2.5, 5, 10):
        if frac <= nice:
            return nice * (10 ** exp)
    return 10 ** (exp + 1)


def _format_tick(val: float) -> str:
    """Format a tick/bar value concisely."""
    if val == int(val) and abs(val) < 1e6:
        return str(int(val))
    if abs(val) >= 1e4 or (abs(val) < 0.01 and val != 0):
        return f"{val:.2e}"
    return f"{val:.2g}"


# ══════════════════════════════════════════════════════════════════════════════
#  Screen widget
# ══════════════════════════════════════════════════════════════════════════════

class BarPlotScreen(QWidget, WorkflowNodeScreenSupport):
    """
    Bar Plot – mirrors Orange's Bar Plot widget behaviour.

    Numeric Y variable
    ──────────────────
    Shows one bar per data instance (raw value), up to MAX_INSTANCES = 200.
    Matches Orange's owbarplot.py design: ContinuousVariable on Y axis,
    optional color/group variable, optional annotation variable for X labels.
    Bars extend down for negative values; y = 0 is always the baseline.

    Categorical Y variable
    ──────────────────────
    Falls back to frequency-count bars, optionally grouped side-by-side by a
    second categorical variable.  Poisson error bars (±√count) are shown.

    Additional features (beyond Orange's widget)
    ────────────────────────────────────────────
    • Error bars: ±√count (Poisson) for frequency bars
    • Bar outline for contrast on similar-coloured bars
    • Per-group legend (top-right)
    • Hover tooltips with instance / group / value / error
    • Horizontal scrolling when bars are too many to fit
    • Rotated X labels for dense charts
    • Font-metric–based label truncation (no hardcoded char limit)
    """

    MAX_INSTANCES = 200

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._init_workflow_node_support()
        self._dataset: DatasetHandle | None = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        header = QLabel("Bar Plot")
        header.setProperty("sectionTitle", True)
        layout.addWidget(header)

        # ── Controls ──────────────────────────────────────────────────────────
        ctrl_box = QGroupBox("Settings")
        ctrl = QHBoxLayout(ctrl_box)
        ctrl.setSpacing(6)

        ctrl.addWidget(QLabel("Column:"))
        self._column_combo = QComboBox()
        self._column_combo.setMinimumWidth(100)
        self._column_combo.currentTextChanged.connect(self._refresh_chart)
        ctrl.addWidget(self._column_combo, 2)

        ctrl.addWidget(QLabel("Group / Color:"))
        self._group_combo = QComboBox()
        self._group_combo.setMinimumWidth(100)
        self._group_combo.currentTextChanged.connect(self._refresh_chart)
        ctrl.addWidget(self._group_combo, 2)

        ctrl.addWidget(QLabel("Label by:"))
        self._label_combo = QComboBox()
        self._label_combo.setMinimumWidth(100)
        self._label_combo.currentTextChanged.connect(self._refresh_chart)
        ctrl.addWidget(self._label_combo, 2)

        ctrl.addWidget(QLabel("Sort:"))
        self._sort_combo = QComboBox()
        self._sort_combo.addItems(["Dataset order", "By value ↓", "By value ↑", "Alphabetically"])
        self._sort_combo.currentIndexChanged.connect(self._refresh_chart)
        ctrl.addWidget(self._sort_combo)

        self._show_values_cb = QCheckBox("Show values")
        self._show_values_cb.setChecked(True)
        self._show_values_cb.stateChanged.connect(self._on_show_values_changed)
        ctrl.addWidget(self._show_values_cb)

        self._show_errors_cb = QCheckBox("Error bars")
        self._show_errors_cb.setChecked(True)
        self._show_errors_cb.stateChanged.connect(self._on_show_errors_changed)
        ctrl.addWidget(self._show_errors_cb)

        layout.addWidget(ctrl_box)

        # ── Chart inside a horizontal scroll area ─────────────────────────────
        self._chart = _BarChartCanvas()
        scroll = QScrollArea()
        scroll.setWidget(self._chart)
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)

        chart_box = QGroupBox("Chart")
        chart_layout = QVBoxLayout(chart_box)
        chart_layout.setContentsMargins(2, 2, 2, 2)
        chart_layout.addWidget(scroll)
        layout.addWidget(chart_box, 1)

        self._status_label = QLabel("Load a dataset to see the bar plot.")
        self._status_label.setProperty("muted", True)
        layout.addWidget(self._status_label)

    # ── WorkflowNodeScreenSupport ─────────────────────────────────────────────

    def set_input_payload(self, payload) -> None:
        dataset = payload.dataset if payload is not None else None
        self._dataset = dataset

        for combo in (self._column_combo, self._group_combo, self._label_combo):
            combo.blockSignals(True)
            combo.clear()

        if dataset is not None:
            cols = [col.name for col in dataset.domain.columns]
            self._column_combo.addItems(cols)

            self._group_combo.addItem("(none)")
            self._group_combo.addItems(cols)

            self._label_combo.addItem("(index)")
            self._label_combo.addItems(cols)

            # Auto-select target column as default group/color
            for col in dataset.domain.target_columns:
                idx = self._group_combo.findText(col.name)
                if idx >= 0:
                    self._group_combo.setCurrentIndex(idx)
                    break

        for combo in (self._column_combo, self._group_combo, self._label_combo):
            combo.blockSignals(False)

        self._refresh_chart()

    def documentation_url(self) -> str:
        return "https://orangedatamining.com/widget-catalog/visualize/barplot/"

    # ── Internal ──────────────────────────────────────────────────────────────

    def _on_show_values_changed(self, state: int) -> None:
        self._chart.set_show_values(bool(state))

    def _on_show_errors_changed(self, state: int) -> None:
        self._chart.set_show_error_bars(bool(state))

    def _refresh_chart(self) -> None:
        if self._dataset is None:
            self._chart.set_data([], "Value")
            self._status_label.setText("Load a dataset to see the bar plot.")
            return

        col_name = self._column_combo.currentText()
        if not col_name:
            self._chart.set_data([], "Value")
            self._status_label.setText("Select a column.")
            return

        group_name = self._group_combo.currentText()
        if group_name == "(none)":
            group_name = None

        label_name = self._label_combo.currentText()
        if label_name == "(index)":
            label_name = None

        df = self._dataset.dataframe
        try:
            series = df[col_name]
        except Exception:
            self._chart.set_data([], "Value")
            self._status_label.setText(f"Column '{col_name}' not found.")
            return

        is_numeric = series.dtype.is_numeric()
        sort_mode = self._sort_combo.currentText()

        if is_numeric:
            groups, errors, y_label = self._build_numeric_bars(
                series, df, group_name, label_name, sort_mode
            )
        else:
            groups, errors, y_label = self._build_categorical_bars(
                series, df, group_name, sort_mode
            )

        self._chart.set_data(groups, y_label, errors=errors)
        self._chart.set_show_values(self._show_values_cb.isChecked())
        self._chart.set_show_error_bars(self._show_errors_cb.isChecked())

        n_null = series.null_count()
        grp_info = f" · colored by '{group_name}'" if group_name else ""
        lbl_info = f" · labeled by '{label_name}'" if label_name else ""
        self._status_label.setText(
            f"Column '{col_name}' · {series.dtype} · {n_null} missing{grp_info}{lbl_info}"
        )

    # ── Data builders ─────────────────────────────────────────────────────────

    def _build_numeric_bars(
        self,
        series,
        df,
        group_name: str | None,
        label_name: str | None,
        sort_mode: str,
    ) -> tuple[list, list, str]:
        """
        One bar per instance (Orange-style for numeric Y variable).
        Each bar height = raw value. Color = group variable value.
        X label = label variable value (or row index).
        Returns at most MAX_INSTANCES bars.
        No error bars for individual-instance bars (they show raw data, not stats).
        """
        values = series.to_list()
        n = len(values)

        group_vals: list[str] = (
            [str(v) if v is not None else "(missing)"
             for v in df[group_name].to_list()]
            if group_name and group_name in df.columns
            else ["" for _ in range(n)]
        )

        label_vals: list[str] = (
            [str(v) if v is not None else ""
             for v in df[label_name].to_list()]
            if label_name and label_name in df.columns
            else [str(i) for i in range(n)]
        )

        # Zip and filter out rows with None Y values
        rows = [
            (lbl, v, g)
            for lbl, v, g in zip(label_vals, values, group_vals)
            if v is not None
        ]

        # Sort
        if sort_mode == "By value ↓":
            rows.sort(key=lambda x: -x[1])
        elif sort_mode == "By value ↑":
            rows.sort(key=lambda x: x[1])
        elif sort_mode == "Alphabetically":
            rows.sort(key=lambda x: x[0])   # x[0] = annotation label (X-axis text)
        # else "Dataset order" → keep as-is

        rows = rows[: self.MAX_INSTANCES]

        groups = [(lbl, [(g, float(v))]) for lbl, v, g in rows]
        # No error bars for individual-instance bars
        errors: list = []
        return groups, errors, series.name or "Value"

    def _build_categorical_bars(
        self,
        series,
        df,
        group_name: str | None,
        sort_mode: str,
    ) -> tuple[list, list, str]:
        """
        Frequency-count bars for a categorical column.
        With a group variable: side-by-side bars per group value.
        Error bars = Poisson ±√count (standard for count data).
        """
        cat_vals = [str(v) if v is not None else "(missing)"
                    for v in series.to_list()]

        if not group_name or group_name not in df.columns:
            # Simple frequency
            counts: dict[str, int] = {}
            for v in cat_vals:
                counts[v] = counts.get(v, 0) + 1
            items = _sort_items(list(counts.items()), sort_mode)[:30]
            groups = [(k, [("", float(c))]) for k, c in items]
            # Poisson error bars: ±√count
            errors = [(k, [("", math.sqrt(max(1, c)))]) for k, c in items]
            return groups, errors, "Count"

        # Grouped frequency
        grp_vals = [str(v) if v is not None else "(missing)"
                    for v in df[group_name].to_list()]

        # Top-8 groups by overall count
        grp_counts: dict[str, int] = {}
        for g in grp_vals:
            grp_counts[g] = grp_counts.get(g, 0) + 1
        top_groups = [k for k, _ in sorted(grp_counts.items(),
                                           key=lambda x: -x[1])[:8]]

        # (cat, group) → count
        joint: dict[str, dict[str, int]] = {}
        for cv, gv in zip(cat_vals, grp_vals):
            if cv not in joint:
                joint[cv] = {}
            joint[cv][gv] = joint[cv].get(gv, 0) + 1

        # Sort categories
        cat_totals = {cv: sum(gd.values()) for cv, gd in joint.items()}
        items = _sort_items(list(cat_totals.items()), sort_mode)[:20]

        groups = []
        errors = []
        for cv, _ in items:
            sub_counts = [(g, joint.get(cv, {}).get(g, 0)) for g in top_groups]
            groups.append((cv, [(g, float(c)) for g, c in sub_counts]))
            errors.append((cv, [(g, math.sqrt(max(1, c))) for g, c in sub_counts]))
        return groups, errors, "Count"


def _sort_items(
    items: list[tuple[str, float]],
    sort_mode: str,
) -> list[tuple[str, float]]:
    """Sort (label, value) pairs according to the chosen mode."""
    if sort_mode == "By value ↓":
        return sorted(items, key=lambda x: -x[1])
    if sort_mode == "By value ↑":
        return sorted(items, key=lambda x: x[1])
    if sort_mode == "Alphabetically":
        return sorted(items, key=lambda x: x[0])
    return items  # "Dataset order" → keep as-is
