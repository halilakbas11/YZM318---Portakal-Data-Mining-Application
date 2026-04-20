from __future__ import annotations

import math

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QPainter, QPen
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QSizePolicy,
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


class _BarChartWidget(QWidget):
    """
    Bar chart supporting:
    - Grouped bars (one group per category, coloured by a second variable)
    - Numeric mode: shows mean ± implicit range
    - Categorical mode: shows frequency counts
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        # List of (category_label, [(group_label, value), ...])
        self._groups: list[tuple[str, list[tuple[str, float]]]] = []
        self._y_label = "Count"
        self._show_values = True
        self.setMinimumHeight(200)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

    def set_data(
        self,
        groups: list[tuple[str, list[tuple[str, float]]]],
        y_label: str = "Count",
    ) -> None:
        self._groups = groups
        self._y_label = y_label
        self.update()

    def set_show_values(self, show: bool) -> None:
        self._show_values = show
        self.update()

    def paintEvent(self, _event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), QColor("#fffdf9"))

        if not self._groups:
            painter.setPen(QColor("#8d877d"))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter,
                             "No data to display.\nLoad a dataset and select a column.")
            return

        margin_l, margin_r = 52, 12
        margin_t, margin_b = 16, 48
        w, h = self.width(), self.height()
        chart_w = w - margin_l - margin_r
        chart_h = h - margin_t - margin_b
        if chart_w < 20 or chart_h < 20:
            return

        # Max value across all groups
        max_val = max(v for _, sub in self._groups for _, v in sub) or 1.0
        n_cats = len(self._groups)
        n_groups = max(len(sub) for _, sub in self._groups) if self._groups else 1

        cat_area = chart_w / max(1, n_cats)
        bar_w = max(4, int(cat_area * 0.8 / max(1, n_groups)))
        total_bar_w = bar_w * n_groups
        cat_gap = cat_area - total_bar_w

        # Grid lines
        grid_steps = 4
        painter.setPen(QPen(QColor("#e0ddd6"), 1, Qt.PenStyle.DotLine))
        for i in range(1, grid_steps + 1):
            y = margin_t + chart_h - int(chart_h * i / grid_steps)
            painter.drawLine(margin_l, y, margin_l + chart_w, y)
            val_label = f"{max_val * i / grid_steps:.1f}"
            painter.setPen(QColor("#8d877d"))
            painter.drawText(2, y + 4, margin_l - 6, 12, Qt.AlignmentFlag.AlignRight, val_label)
            painter.setPen(QPen(QColor("#e0ddd6"), 1, Qt.PenStyle.DotLine))

        # Axes
        painter.setPen(QPen(QColor("#9b9488"), 1))
        painter.drawLine(margin_l, margin_t, margin_l, margin_t + chart_h)
        painter.drawLine(margin_l, margin_t + chart_h, margin_l + chart_w, margin_t + chart_h)

        # Y-axis label
        painter.save()
        painter.translate(12, margin_t + chart_h // 2)
        painter.rotate(-90)
        painter.setPen(QColor("#534b40"))
        painter.drawText(-40, -6, 80, 14, Qt.AlignmentFlag.AlignCenter, self._y_label)
        painter.restore()

        # Bars
        for ci, (cat_label, sub) in enumerate(self._groups):
            cat_x = margin_l + int(ci * cat_area) + int(cat_gap / 2)

            for gi, (group_label, value) in enumerate(sub):
                bar_h = max(0, int((value / max_val) * chart_h))
                bx = cat_x + gi * bar_w
                by = margin_t + chart_h - bar_h

                color = _PALETTE[gi % len(_PALETTE)]
                painter.setBrush(color)
                painter.setPen(Qt.PenStyle.NoPen)
                painter.drawRoundedRect(bx, by, bar_w, bar_h, 2, 2)

                if self._show_values and bar_h > 14 and bar_w > 18:
                    painter.setPen(QColor("#ffffff"))
                    painter.drawText(bx, by + 2, bar_w, 14,
                                     Qt.AlignmentFlag.AlignCenter, f"{value:.1f}")

            # Category label
            label = cat_label if len(cat_label) <= 10 else cat_label[:9] + "…"
            painter.setPen(QColor("#534b40"))
            painter.drawText(
                int(cat_x), margin_t + chart_h + 4,
                int(total_bar_w), 22,
                Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop,
                label,
            )

        painter.end()


class BarPlotScreen(QWidget, WorkflowNodeScreenSupport):
    """
    Bar Plot – frequency or mean bars, optionally grouped by a second variable.

    - If the selected column is categorical: shows frequency counts
    - If numeric: shows mean per category (when grouped), or histogram bins
    - Supports a grouping variable for side-by-side grouped bars
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._init_workflow_node_support()
        self._dataset: DatasetHandle | None = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        header = QLabel("Bar Plot")
        header.setProperty("sectionTitle", True)
        layout.addWidget(header)

        controls_group = QGroupBox("Settings")
        ctrl = QHBoxLayout(controls_group)

        ctrl.addWidget(QLabel("Column:"))
        self._column_combo = QComboBox()
        self._column_combo.currentTextChanged.connect(self._refresh_chart)
        ctrl.addWidget(self._column_combo, 1)

        ctrl.addWidget(QLabel("Group by:"))
        self._group_combo = QComboBox()
        self._group_combo.currentTextChanged.connect(self._refresh_chart)
        ctrl.addWidget(self._group_combo, 1)

        ctrl.addWidget(QLabel("Sort:"))
        self._sort_combo = QComboBox()
        self._sort_combo.addItems(["By frequency", "Alphabetically", "As is"])
        self._sort_combo.currentIndexChanged.connect(self._refresh_chart)
        ctrl.addWidget(self._sort_combo)

        self._show_values_cb = QCheckBox("Show values")
        self._show_values_cb.setChecked(True)
        self._show_values_cb.stateChanged.connect(self._on_show_values_changed)
        ctrl.addWidget(self._show_values_cb)

        layout.addWidget(controls_group)

        chart_group = QGroupBox("Chart")
        chart_layout = QVBoxLayout(chart_group)
        self._chart = _BarChartWidget()
        chart_layout.addWidget(self._chart)
        layout.addWidget(chart_group, 1)

        self._status_label = QLabel("Load a dataset to see the bar plot.")
        self._status_label.setProperty("muted", True)
        layout.addWidget(self._status_label)

    # ── WorkflowNodeScreenSupport ──────────────────────────────────────

    def set_input_payload(self, payload) -> None:
        dataset = payload.dataset if payload is not None else None
        self._dataset = dataset
        for combo in (self._column_combo, self._group_combo):
            combo.blockSignals(True)
            combo.clear()
        if dataset is not None:
            cols = [col.name for col in dataset.domain.columns]
            self._column_combo.addItems(cols)
            self._group_combo.addItem("(none)")
            self._group_combo.addItems(cols)
            # Default grouping: target column if available
            for col in dataset.domain.target_columns:
                idx = self._group_combo.findText(col.name)
                if idx >= 0:
                    self._group_combo.setCurrentIndex(idx)
                    break
        for combo in (self._column_combo, self._group_combo):
            combo.blockSignals(False)
        self._refresh_chart()

    def documentation_url(self) -> str:
        return "https://orangedatamining.com/widget-catalog/visualize/barplot/"

    # ── Internal ──────────────────────────────────────────────────────

    def _on_show_values_changed(self, state: int) -> None:
        self._chart.set_show_values(bool(state))

    def _refresh_chart(self) -> None:
        if self._dataset is None:
            self._chart.set_data([], "Count")
            self._status_label.setText("Load a dataset to see the bar plot.")
            return

        col_name = self._column_combo.currentText()
        group_name = self._group_combo.currentText()
        if group_name == "(none)":
            group_name = None

        if not col_name:
            self._chart.set_data([], "Count")
            self._status_label.setText("Select a column.")
            return

        df = self._dataset.dataframe
        try:
            series = df[col_name]
        except Exception:
            self._chart.set_data([], "Count")
            self._status_label.setText(f"Column '{col_name}' not found.")
            return

        is_numeric = series.dtype.is_numeric()
        sort_mode = self._sort_combo.currentText()

        if group_name and group_name in df.columns:
            # Grouped mode: one sub-bar per group value
            groups = self._build_grouped_bars(series, df[group_name], is_numeric, sort_mode)
            y_label = "Mean" if is_numeric else "Count"
        elif is_numeric:
            # Histogram bins
            bins = self._build_histogram_bars(series.drop_nulls().to_list())
            groups = [(label, [("", val)]) for label, val in bins]
            y_label = "Count"
        else:
            # Simple frequency
            counts: dict[str, int] = {}
            for val in series.to_list():
                key = str(val) if val is not None else "(missing)"
                counts[key] = counts.get(key, 0) + 1
            if sort_mode == "By frequency":
                items = sorted(counts.items(), key=lambda x: -x[1])[:30]
            elif sort_mode == "Alphabetically":
                items = sorted(counts.items(), key=lambda x: x[0])[:30]
            else:
                items = list(counts.items())[:30]
            groups = [(k, [("", float(v))]) for k, v in items]
            y_label = "Count"

        self._chart.set_data(groups, y_label)
        self._chart.set_show_values(self._show_values_cb.isChecked())
        n_null = series.null_count()
        dtype_str = str(series.dtype)
        grp_info = f" · grouped by '{group_name}'" if group_name else ""
        self._status_label.setText(
            f"Column '{col_name}' · {dtype_str} · {n_null} missing{grp_info}"
        )

    def _build_grouped_bars(self, series, group_series, is_numeric: bool, sort_mode: str):
        """Build grouped bars: each category in `series` has one bar per group value."""
        cat_vals = [str(v) if v is not None else "(missing)" for v in series.to_list()]
        grp_vals = [str(v) if v is not None else "(missing)" for v in group_series.to_list()]

        # Collect unique groups (top 8)
        grp_counts: dict[str, int] = {}
        for g in grp_vals:
            grp_counts[g] = grp_counts.get(g, 0) + 1
        unique_groups = [k for k, _ in sorted(grp_counts.items(), key=lambda x: -x[1])[:8]]

        # Per category per group: count or mean
        cat_grp_data: dict[str, dict[str, list[float]]] = {}
        for cv, gv, sv in zip(cat_vals, grp_vals, series.to_list()):
            if cv not in cat_grp_data:
                cat_grp_data[cv] = {}
            if gv not in cat_grp_data[cv]:
                cat_grp_data[cv][gv] = []
            if sv is not None:
                cat_grp_data[cv][gv].append(float(sv))

        # Compute value per (cat, group)
        cat_total: dict[str, float] = {}
        for cv, grp_dict in cat_grp_data.items():
            total = sum(
                (sum(vals) / len(vals) if is_numeric and vals else len(vals))
                for g, vals in grp_dict.items()
                if g in unique_groups
            )
            cat_total[cv] = total

        # Sort categories
        if sort_mode == "By frequency":
            cats = sorted(cat_grp_data.keys(), key=lambda x: -cat_total.get(x, 0))[:20]
        elif sort_mode == "Alphabetically":
            cats = sorted(cat_grp_data.keys())[:20]
        else:
            cats = list(cat_grp_data.keys())[:20]

        result = []
        for cv in cats:
            sub = []
            for g in unique_groups:
                vals = cat_grp_data.get(cv, {}).get(g, [])
                if is_numeric:
                    val = sum(vals) / len(vals) if vals else 0.0
                else:
                    val = float(len(vals))
                sub.append((g, val))
            result.append((cv, sub))
        return result

    def _build_histogram_bars(self, values: list) -> list[tuple[str, float]]:
        if not values:
            return []
        try:
            float_vals = [float(v) for v in values]
        except (TypeError, ValueError):
            return []
        if not float_vals:
            return []
        mn, mx = min(float_vals), max(float_vals)
        if mn == mx:
            return [(f"{mn:.2f}", float(len(float_vals)))]
        n_bins = min(20, max(5, int(math.sqrt(len(float_vals)))))
        bin_w = (mx - mn) / n_bins
        counts = [0] * n_bins
        for v in float_vals:
            idx = min(n_bins - 1, int((v - mn) / bin_w))
            counts[idx] += 1
        return [(f"{mn + i * bin_w:.2f}", float(c)) for i, c in enumerate(counts)]
