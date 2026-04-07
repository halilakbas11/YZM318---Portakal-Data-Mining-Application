from __future__ import annotations

import math
from collections import OrderedDict

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
from portakal_app.ui import i18n
from portakal_app.ui.screens.node_screen import WorkflowNodeScreenSupport

# ── Colour palette ────────────────────────────────────────────────────────────
_PALETTE = [
    QColor("#e07020"), QColor("#3b82f6"), QColor("#22c55e"),
    QColor("#a855f7"), QColor("#f43f5e"), QColor("#0ea5e9"),
    QColor("#f59e0b"), QColor("#10b981"),
]

_MIN_BAR_PX = 4   # minimum bar width in pixels


class _BarChartCanvas(QWidget):
    """
    QPainter bar chart canvas – Orange-style layout:

    • One bar per data instance (numeric mode) — NOT aggregated
    • Bars are arranged in sections: all instances of the same group are
      consecutive, separated by a thin vertical divider line, with the group
      name centred below that section (matches Orange's owbarplot.py)
    • Bars within a section are sorted according to the sort control
    • Each bar coloured by its group value (colour key = group value string)
    • X-axis shows individual annotation labels only when bars are wide enough;
      otherwise only the section (group) label is shown
    • Error bars supported (Poisson ±√count for frequency mode)
    • Horizontal scrolling when many bars don't fit
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        # List of (x_label, value, color_key) — one entry per bar
        self._bars: list[tuple[str, float, str]] = []
        # Sections: (section_label, start_idx, end_idx) — inclusive end
        self._sections: list[tuple[str, int, int]] = []
        # color_key → palette index
        self._color_map: dict[str, int] = {}
        self._y_label = "Value"
        self._show_values = True
        self._show_error_bars = True
        # Errors: one per bar (same length as _bars), 0.0 means no error bar
        self._errors: list[float] = []
        # Hit-test rects: (QRect, x_label, color_key, value, error)
        self._bar_rects: list[tuple[QRect, str, str, float, float]] = []

        self.setMinimumHeight(220)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMouseTracking(True)

    # ── Public API ────────────────────────────────────────────────────────────

    def set_data(
        self,
        bars: list[tuple[str, float, str]],
        sections: list[tuple[str, int, int]],
        y_label: str = "Value",
        errors: list[float] | None = None,
    ) -> None:
        """
        bars     – [(x_label, value, color_key), ...]  one per instance
        sections – [(section_label, start_idx, end_idx), ...]
        errors   – parallel to bars; 0.0 = no error bar
        """
        self._bars = bars
        self._sections = sections
        self._y_label = y_label
        self._errors = errors if errors is not None else [0.0] * len(bars)

        # Build stable colour map
        seen: dict[str, int] = {}
        for _, _, key in bars:
            if key not in seen:
                seen[key] = len(seen)
        self._color_map = seen

        self._bar_rects = []
        self._update_preferred_width()
        self.update()

    def set_show_values(self, show: bool) -> None:
        self._show_values = show
        self.update()

    def set_show_error_bars(self, show: bool) -> None:
        self._show_error_bars = show
        self.update()

    # ── Tooltip ───────────────────────────────────────────────────────────────

    def mouseMoveEvent(self, event) -> None:
        pos = event.position().toPoint()
        for rect, x_lbl, color_key, val, err in self._bar_rects:
            if rect.contains(pos):
                lines = []
                if x_lbl:
                    lines.append(f"Instance: {x_lbl}")
                lines.append(f"Value: {val:.4g}")
                if color_key:
                    lines.append(f"Group: {color_key}")
                if err > 0:
                    lines.append(f"±{err:.4g}")
                QToolTip.showText(event.globalPosition().toPoint(),
                                  "\n".join(lines), self)
                return
        QToolTip.hideText()

    def leaveEvent(self, _event) -> None:
        QToolTip.hideText()

    # ── Drawing ───────────────────────────────────────────────────────────────

    def paintEvent(self, _event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), QColor("#fffdf9"))

        if not self._bars:
            painter.setPen(QColor("#8d877d"))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter,
                             "No data.\nLoad a dataset and select a numeric column.")
            return

        self._bar_rects = []

        w, h = self.width(), self.height()

        # Legend on right when there are multiple colour groups
        unique_keys = list(self._color_map.keys())
        has_legend = len(unique_keys) > 1
        legend_w = 0
        if has_legend:
            legend_w = min(160, max(len(k) for k in unique_keys) * 7 + 28)

        margin_l = 58
        margin_r = legend_w + 16
        margin_t = 20
        margin_b = 70  # room for rotated annotation labels + section labels
        chart_w = w - margin_l - margin_r
        chart_h = h - margin_t - margin_b
        if chart_w < 20 or chart_h < 20:
            return

        # ── Value range ───────────────────────────────────────────────────────
        all_vals = [v for _, v, _ in self._bars]
        y_min_raw = min(0.0, min(all_vals))
        y_max_raw = max(0.0, max(all_vals))
        if self._show_error_bars and self._errors:
            for (_, v, _), e in zip(self._bars, self._errors):
                if e > 0:
                    y_min_raw = min(y_min_raw, v - e)
                    y_max_raw = max(y_max_raw, v + e)

        raw_span = y_max_raw - y_min_raw
        pad = raw_span * 0.05 if raw_span > 0 else 0.5
        y_min = y_min_raw - pad
        y_max = y_max_raw + pad
        span = (y_max - y_min) or 1.0

        def val_to_y(v: float) -> int:
            return margin_t + int(chart_h * (1.0 - (v - y_min) / span))

        baseline_y = val_to_y(0.0)

        # ── Grid & Y ticks ────────────────────────────────────────────────────
        painter.setFont(QFont(self.font().family(), 8))
        fm = QFontMetrics(painter.font())
        grid_steps = 5
        nice_step = _nice_step((y_max_raw - y_min_raw) / grid_steps)
        tick_val = math.floor(y_min_raw / nice_step) * nice_step
        while tick_val <= y_max_raw + nice_step * 0.01:
            py = val_to_y(tick_val)
            if margin_t <= py <= margin_t + chart_h:
                painter.setPen(QPen(QColor("#e0ddd6"), 1, Qt.PenStyle.DotLine))
                painter.drawLine(margin_l, py, margin_l + chart_w, py)
                painter.setPen(QColor("#8d877d"))
                painter.drawText(2, py - 6, margin_l - 6, 14,
                                 Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
                                 _format_tick(tick_val))
            tick_val = round(tick_val + nice_step, 10)

        # ── Axes ──────────────────────────────────────────────────────────────
        painter.setPen(QPen(QColor("#9b9488"), 1))
        painter.drawLine(margin_l, margin_t, margin_l, margin_t + chart_h)
        painter.drawLine(margin_l, margin_t + chart_h, margin_l + chart_w, margin_t + chart_h)
        if y_min_raw < 0:
            painter.setPen(QPen(QColor("#534b40"), 1))
            painter.drawLine(margin_l, baseline_y, margin_l + chart_w, baseline_y)

        # ── Y-axis label ──────────────────────────────────────────────────────
        painter.save()
        painter.translate(12, margin_t + chart_h // 2)
        painter.rotate(-90)
        painter.setPen(QColor("#534b40"))
        painter.setFont(QFont(self.font().family(), 9))
        y_lbl_t = fm.elidedText(self._y_label, Qt.TextElideMode.ElideRight, chart_h - 10)
        painter.drawText(-50, -6, 100, 14, Qt.AlignmentFlag.AlignCenter, y_lbl_t)
        painter.restore()

        # ── Bar geometry ──────────────────────────────────────────────────────
        n = len(self._bars)
        bar_w = max(_MIN_BAR_PX, int(chart_w / max(1, n)) - 1)
        total_bars_px = n * (bar_w + 1)

        painter.setFont(QFont(self.font().family(), 8))
        fm = QFontMetrics(painter.font())

        # ── Bars ──────────────────────────────────────────────────────────────
        for idx, (x_lbl, value, color_key) in enumerate(self._bars):
            bx = margin_l + idx * (bar_w + 1)
            top_y = val_to_y(max(0.0, value))
            bot_y = val_to_y(min(0.0, value))
            bar_h = max(1, bot_y - top_y)

            color_idx = self._color_map.get(color_key, 0)
            color = _PALETTE[color_idx % len(_PALETTE)]

            painter.setBrush(color)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawRect(bx, top_y, bar_w, bar_h)

            # Error bar
            err = self._errors[idx] if idx < len(self._errors) else 0.0
            if self._show_error_bars and err > 0:
                err_x = bx + bar_w // 2
                err_top = val_to_y(value + err)
                err_bot = val_to_y(value - err)
                cap = max(2, bar_w // 3)
                painter.setPen(QPen(QColor("#222"), 1.2))
                painter.drawLine(err_x, err_top, err_x, err_bot)
                painter.drawLine(err_x - cap, err_top, err_x + cap, err_top)
                painter.drawLine(err_x - cap, err_bot, err_x + cap, err_bot)

            # Value label (only when bar is tall & wide enough)
            if self._show_values and bar_h > 14 and bar_w > 18:
                painter.setPen(QColor("#ffffff"))
                painter.drawText(bx, top_y + 2, bar_w, 12,
                                 Qt.AlignmentFlag.AlignCenter,
                                 _format_tick(value))

            # X annotation label — always show, skip every N to prevent overlap
            if x_lbl:
                # How many labels fit? each label footprint = ~42px at 45° rotation
                label_footprint = 42
                max_labels = max(1, chart_w // label_footprint)
                skip = max(1, (n + max_labels - 1) // max_labels)
                if idx % skip == 0:
                    painter.setPen(QColor("#8d877d"))
                    font_sz = 8 if bar_w >= 10 else 7
                    painter.setFont(QFont(self.font().family(), font_sz))
                    fm_ann = QFontMetrics(painter.font())
                    lbl = fm_ann.elidedText(x_lbl, Qt.TextElideMode.ElideRight, 72)
                    painter.save()
                    painter.translate(bx + bar_w // 2, margin_t + chart_h + 4)
                    painter.rotate(45)
                    painter.drawText(0, 0, 72, 12, Qt.AlignmentFlag.AlignLeft, lbl)
                    painter.restore()
                    # restore font
                    painter.setFont(QFont(self.font().family(), 8))
                    fm = QFontMetrics(painter.font())

            self._bar_rects.append((
                QRect(bx, top_y, bar_w, bar_h),
                x_lbl, color_key, value, err,
            ))

        # ── Section separators & labels ───────────────────────────────────────
        if self._sections:
            for sec_label, start_i, end_i in self._sections:
                sec_start_x = margin_l + start_i * (bar_w + 1)
                sec_end_x   = margin_l + (end_i + 1) * (bar_w + 1)

                # Separator line (thin gold/tan, like Orange)
                if start_i > 0:
                    sep_x = sec_start_x - 1
                    painter.setPen(QPen(QColor("#c8b860"), 1))
                    painter.drawLine(sep_x, margin_t, sep_x, margin_t + chart_h)

                # Section label below the annotation area
                painter.setPen(QColor("#534b40"))
                painter.setFont(QFont(self.font().family(), 9, QFont.Weight.Bold))
                sec_w = sec_end_x - sec_start_x
                fm_sec = QFontMetrics(painter.font())
                lbl = fm_sec.elidedText(sec_label, Qt.TextElideMode.ElideRight, max(sec_w - 4, 20))
                painter.drawText(sec_start_x, margin_t + chart_h + 48,
                                 sec_w, 18,
                                 Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop,
                                 lbl)
                painter.setFont(QFont(self.font().family(), 8))
        elif self._bars:
            # No sections — single x-axis label centred (the column name from y_label)
            pass

        # ── Legend ────────────────────────────────────────────────────────────
        if has_legend:
            lx = w - legend_w - 4
            ly = margin_t + 4
            row_h = 18
            box_h = len(unique_keys) * row_h + 10
            painter.setFont(QFont(self.font().family(), 8))
            fm_leg = QFontMetrics(painter.font())
            painter.setBrush(QColor(255, 255, 255, 200))
            painter.setPen(QPen(QColor("#ccc8c0"), 1))
            painter.drawRoundedRect(lx, ly, legend_w, box_h, 4, 4)
            for li, key in enumerate(unique_keys[:8]):
                color_idx = self._color_map.get(key, li)
                color = _PALETTE[color_idx % len(_PALETTE)]
                iy = ly + 5 + li * row_h
                painter.setBrush(color)
                painter.setPen(Qt.PenStyle.NoPen)
                painter.drawRoundedRect(lx + 6, iy + 3, 10, 10, 2, 2)
                painter.setPen(QColor("#534b40"))
                glbl = fm_leg.elidedText(key, Qt.TextElideMode.ElideRight, legend_w - 24)
                painter.drawText(lx + 20, iy, legend_w - 24, row_h,
                                 Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft,
                                 glbl)

        painter.end()

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _update_preferred_width(self) -> None:
        n = len(self._bars)
        unique_keys = list(self._color_map.keys())
        has_legend = len(unique_keys) > 1
        legend_w = 0
        if has_legend:
            legend_w = min(160, max((len(k) for k in unique_keys), default=0) * 7 + 28)
        preferred = max(self.minimumWidth(),
                        n * (max(_MIN_BAR_PX, 6) + 1) + 80 + legend_w + 16)
        self.setMinimumWidth(preferred)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _nice_step(raw: float) -> float:
    if raw <= 0:
        return 1.0
    exp = math.floor(math.log10(raw))
    frac = raw / (10 ** exp)
    for nice in (1, 2, 2.5, 5, 10):
        if frac <= nice:
            return nice * (10 ** exp)
    return 10 ** (exp + 1)


def _format_tick(val: float) -> str:
    if val == int(val) and abs(val) < 1e6:
        return str(int(val))
    if abs(val) >= 1e4 or (abs(val) < 0.01 and val != 0):
        return f"{val:.2e}"
    return f"{val:.2g}"


def _sort_rows(rows, sort_mode: str):
    """Sort (x_label, value, color_key) rows by mode, IN-PLACE."""
    if sort_mode == "By value ↓":
        rows.sort(key=lambda r: -r[1])
    elif sort_mode == "By value ↑":
        rows.sort(key=lambda r: r[1])
    elif sort_mode == "Alphabetically":
        rows.sort(key=lambda r: r[0])
    # "Dataset order" → no sort


# ══════════════════════════════════════════════════════════════════════════════
#  Screen widget
# ══════════════════════════════════════════════════════════════════════════════

class BarPlotScreen(QWidget, WorkflowNodeScreenSupport):
    """
    Bar Plot – matches Orange's owbarplot.py behaviour:

    Numeric Values column
    ─────────────────────
    One bar per data instance (raw value), capped at MAX_INSTANCES = 200.
    When "Group by" is set all instances belonging to the same group are drawn
    consecutively (like Orange), separated by a gold divider line, with the
    group name centred below that section.  Bars are coloured by group value.

    Categorical Values column
    ─────────────────────────
    Falls back to frequency-count bars (±√count Poisson error bars).
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

        ctrl.addWidget(QLabel("Values:"))
        self._column_combo = QComboBox()
        self._column_combo.setMinimumWidth(100)
        self._column_combo.currentTextChanged.connect(self._refresh_chart)
        ctrl.addWidget(self._column_combo, 2)

        ctrl.addWidget(QLabel("Group by:"))
        self._group_combo = QComboBox()
        self._group_combo.setMinimumWidth(100)
        self._group_combo.currentTextChanged.connect(self._refresh_chart)
        ctrl.addWidget(self._group_combo, 2)

        ctrl.addWidget(QLabel("Annotations:"))
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

        self._status_label = QLabel(i18n.t("Load a dataset to see the bar plot."))
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
            df = dataset.dataframe
            cols = [col.name for col in dataset.domain.columns]
            num_cols = [c for c in cols if df[c].dtype.is_numeric()]

            # Values: prefer numeric columns (like Orange)
            self._column_combo.addItems(num_cols if num_cols else cols)

            self._group_combo.addItem("(none)")
            self._group_combo.addItems(cols)

            self._label_combo.addItem("Enumerate")
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

    # ── Callbacks ─────────────────────────────────────────────────────────────

    def _on_show_values_changed(self, state: int) -> None:
        self._chart.set_show_values(bool(state))

    def _on_show_errors_changed(self, state: int) -> None:
        self._chart.set_show_error_bars(bool(state))

    # ── Refresh ───────────────────────────────────────────────────────────────

    def _refresh_chart(self) -> None:
        if self._dataset is None:
            self._chart.set_data([], [], "Value")
            self._status_label.setText(i18n.t("Load a dataset to see the bar plot."))
            return

        col_name = self._column_combo.currentText()
        if not col_name:
            self._chart.set_data([], [], "Value")
            self._status_label.setText(i18n.t("Select a column."))
            return

        group_name = self._group_combo.currentText()
        if group_name == "(none)":
            group_name = None

        label_name = self._label_combo.currentText()
        if label_name == "Enumerate":
            label_name = None

        df = self._dataset.dataframe
        try:
            series = df[col_name]
        except Exception:
            self._chart.set_data([], [], "Value")
            self._status_label.setText(f"Column '{col_name}' not found.")
            return

        sort_mode = self._sort_combo.currentText()

        if series.dtype.is_numeric():
            bars, sections, errors, y_label = self._build_numeric_bars(
                series, df, group_name, label_name, sort_mode
            )
        else:
            bars, sections, errors, y_label = self._build_categorical_bars(
                series, df, group_name, sort_mode
            )

        self._chart.set_data(bars, sections, y_label, errors)
        self._chart.set_show_values(self._show_values_cb.isChecked())
        self._chart.set_show_error_bars(self._show_errors_cb.isChecked())

        n_null = series.null_count()
        grp_note = f"  ·  colored by '{group_name}'" if group_name else ""
        self._status_label.setText(
            f"Column '{col_name}'  ·  {series.dtype}  ·  {n_null} missing{grp_note}"
        )

    # ── Data builders ─────────────────────────────────────────────────────────

    def _build_numeric_bars(
        self,
        series,
        df,
        group_name: str | None,
        label_name: str | None,
        sort_mode: str,
    ) -> tuple[list, list, list, str]:
        """
        One bar per instance (Orange-style).

        When group_name is set, instances are grouped together section-by-section
        (all Female first, all Male second, etc.) with the group name shown as
        the x-axis section label — exactly mirroring Orange's owbarplot.py.
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
            else [str(i + 1) for i in range(n)]  # 1-based enumerate like Orange
        )

        # (x_label, value, color_key) — filter nulls
        rows = [
            (label_vals[i], float(values[i]), group_vals[i])
            for i in range(n)
            if values[i] is not None
        ]

        if group_name:
            # ── Orange layout: group all same-group instances together ─────────
            # Build ordered dict preserving first-occurrence order of groups
            by_group: OrderedDict[str, list] = OrderedDict()
            for row in rows:
                gk = row[2]
                if gk not in by_group:
                    by_group[gk] = []
                by_group[gk].append(row)

            # Sort within each group according to sort_mode, cap per group
            max_per_group = max(1, self.MAX_INSTANCES // max(1, len(by_group)))
            for gk in by_group:
                _sort_rows(by_group[gk], sort_mode)
                by_group[gk] = by_group[gk][:max_per_group]

            # Flatten into bars + sections
            bars: list[tuple[str, float, str]] = []
            sections: list[tuple[str, int, int]] = []
            for gk, g_rows in by_group.items():
                start_idx = len(bars)
                bars.extend(g_rows)
                end_idx = len(bars) - 1
                sections.append((gk, start_idx, end_idx))
        else:
            # No grouping — just sort and cap
            _sort_rows(rows, sort_mode)
            rows = rows[: self.MAX_INSTANCES]
            bars = rows
            sections = []

        errors = [0.0] * len(bars)
        return bars, sections, errors, series.name or "Value"

    def _build_categorical_bars(
        self,
        series,
        df,
        group_name: str | None,
        sort_mode: str,
    ) -> tuple[list, list, list, str]:
        """
        Frequency count bars for categorical Values column.
        With group_name: side-by-side sections per category value.
        Error bars = Poisson ±√count.
        """
        cat_vals = [str(v) if v is not None else "(missing)"
                    for v in series.to_list()]

        if not group_name or group_name not in df.columns:
            counts: dict[str, int] = {}
            for v in cat_vals:
                counts[v] = counts.get(v, 0) + 1
            items = _sort_kv(list(counts.items()), sort_mode)[:30]
            bars = [(k, float(c), k) for k, c in items]
            errors = [math.sqrt(max(1, c)) for _, c in items]
            sections = [(k, i, i) for i, (k, _) in enumerate(items)]
            return bars, sections, errors, "Count"

        grp_vals = [str(v) if v is not None else "(missing)"
                    for v in df[group_name].to_list()]

        # Top-8 group values
        grp_counts: dict[str, int] = {}
        for g in grp_vals:
            grp_counts[g] = grp_counts.get(g, 0) + 1
        top_groups = [k for k, _ in sorted(grp_counts.items(), key=lambda x: -x[1])[:8]]

        joint: dict[str, dict[str, int]] = {}
        for cv, gv in zip(cat_vals, grp_vals):
            if cv not in joint:
                joint[cv] = {}
            joint[cv][gv] = joint[cv].get(gv, 0) + 1

        cat_totals = {cv: sum(gd.values()) for cv, gd in joint.items()}
        sorted_cats = [k for k, _ in _sort_kv(list(cat_totals.items()), sort_mode)[:20]]

        bars = []
        errors = []
        sections = []
        for cv in sorted_cats:
            start_idx = len(bars)
            for gv in top_groups:
                cnt = joint.get(cv, {}).get(gv, 0)
                bars.append((gv, float(cnt), gv))
                errors.append(math.sqrt(max(1, cnt)))
            sections.append((cv, start_idx, len(bars) - 1))

        return bars, sections, errors, "Count"


def _sort_kv(items: list[tuple[str, float]], sort_mode: str) -> list[tuple[str, float]]:
    if sort_mode == "By value ↓":
        return sorted(items, key=lambda x: -x[1])
    if sort_mode == "By value ↑":
        return sorted(items, key=lambda x: x[1])
    if sort_mode == "Alphabetically":
        return sorted(items, key=lambda x: x[0])
    return items
