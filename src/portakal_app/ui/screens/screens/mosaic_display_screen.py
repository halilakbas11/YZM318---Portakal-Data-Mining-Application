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

# Palette for categories
_PALETTE = [
    QColor("#e07020"), QColor("#3b82f6"), QColor("#22c55e"),
    QColor("#a855f7"), QColor("#f43f5e"), QColor("#0ea5e9"),
    QColor("#f59e0b"), QColor("#10b981"),
]


class _MosaicWidget(QWidget):
    """Custom QPainter mosaic plot.

    Column width = proportion of first variable.
    Within each column, cell height = conditional proportion of second variable.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        # {col_val: {row_val: count}}
        self._joint: dict[str, dict[str, int]] = {}
        self._col_totals: dict[str, int] = {}
        self._col_vals: list[str] = []
        self._row_vals: list[str] = []
        self._col_var: str = ""
        self._row_var: str = ""
        self._color_by_residual: bool = False
        self.setMinimumHeight(200)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

    def set_data(
        self,
        joint: dict[str, dict[str, int]],
        col_totals: dict[str, int],
        col_vals: list[str],
        row_vals: list[str],
        col_var: str,
        row_var: str,
        color_by_residual: bool = False,
    ) -> None:
        self._joint = joint
        self._col_totals = col_totals
        self._col_vals = col_vals
        self._row_vals = row_vals
        self._col_var = col_var
        self._row_var = row_var
        self._color_by_residual = color_by_residual
        self.update()

    def paintEvent(self, _event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), QColor("#fffdf9"))

        if not self._joint:
            painter.setPen(QColor("#8d877d"))
            painter.drawText(
                self.rect(),
                Qt.AlignmentFlag.AlignCenter,
                "No data.\nSelect two categorical columns.",
            )
            return

        margin_left = 12
        margin_top = 44
        margin_right = 12
        margin_bottom = 50

        w = self.width()
        h = self.height()
        chart_w = w - margin_left - margin_right
        chart_h = h - margin_top - margin_bottom
        if chart_w < 10 or chart_h < 10:
            return

        n_total = sum(self._col_totals.values()) or 1
        gap = 3  # gap between mosaic columns

        # Compute x positions for each column based on its total proportion
        col_widths: list[int] = []
        for cv in self._col_vals:
            prop = self._col_totals.get(cv, 0) / n_total
            col_widths.append(max(4, int(prop * (chart_w - gap * (len(self._col_vals) - 1)))))

        # Draw mosaic
        x = margin_left
        for ci, cv in enumerate(self._col_vals):
            cw = col_widths[ci]
            col_total = self._col_totals.get(cv, 1) or 1

            # Column header
            painter.setPen(QColor("#3b2a10"))
            label = cv if len(cv) <= 12 else cv[:11] + "…"
            painter.drawText(x, 0, cw, margin_top - 4, Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignBottom, label)

            # Draw stacked cells within this column
            y = margin_top
            row_total_for_col = sum(self._col_totals.values()) or 1
            for ri, rv in enumerate(self._row_vals):
                count = self._joint.get(cv, {}).get(rv, 0)
                prop_within = count / col_total
                cell_h = max(0, int(prop_within * chart_h))

                if self._color_by_residual:
                    # Pearson residual coloring: (O-E)/sqrt(E)
                    row_total = sum(self._joint.get(rv2, {}).get(cv, 0) for rv2 in self._row_vals)
                    expected = (self._col_totals.get(cv, 0) / row_total_for_col) * (row_total / row_total_for_col) * row_total_for_col
                    if expected > 0:
                        r = (count - expected) / math.sqrt(expected)
                        intensity = min(1.0, abs(r) / 4.0)
                        if r > 0:
                            color = QColor(int(59 + (255 - 59) * (1 - intensity)), int(130 * (1 - intensity)), 246, 220)
                        else:
                            color = QColor(220, int(38 + (255 - 38) * (1 - intensity)), int(38 + (255 - 38) * (1 - intensity)), 220)
                    else:
                        color = QColor("#cccccc")
                else:
                    color = _PALETTE[ri % len(_PALETTE)]

                painter.setBrush(color)
                painter.setPen(QPen(QColor("#fffdf9"), 1))
                painter.drawRect(x, y, cw, cell_h)

                # Label inside cell if large enough
                if cell_h > 14 and cw > 20:
                    painter.setPen(QColor("#ffffff"))
                    painter.drawText(x + 2, y + 1, cw - 4, cell_h - 2,
                                     Qt.AlignmentFlag.AlignCenter,
                                     f"{count}")
                y += cell_h

            x += cw + gap

        # Legend for row values
        legend_x = margin_left
        legend_y = h - margin_bottom + 8
        painter.setPen(QColor("#534b40"))
        painter.drawText(0, legend_y - 2, w, 14, Qt.AlignmentFlag.AlignLeft, f"{self._row_var}:")
        for ri, rv in enumerate(self._row_vals):
            lx = legend_x + ri * 110
            if lx + 100 > w:
                break
            color = _PALETTE[ri % len(_PALETTE)]
            painter.setBrush(color)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawRect(lx, legend_y + 16, 12, 12)
            painter.setPen(QColor("#534b40"))
            lbl = rv if len(rv) <= 10 else rv[:9] + "…"
            painter.drawText(lx + 16, legend_y + 16, 90, 14, Qt.AlignmentFlag.AlignVCenter, lbl)

        # X-axis label
        painter.drawText(
            0, h - 14, w, 14,
            Qt.AlignmentFlag.AlignCenter,
            self._col_var,
        )

        painter.end()


class MosaicDisplayScreen(QWidget, WorkflowNodeScreenSupport):
    """Mosaic Display – proportional representation of two categorical variables."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._init_workflow_node_support()
        self._dataset: DatasetHandle | None = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        header = QLabel("Mosaic Display")
        header.setProperty("sectionTitle", True)
        layout.addWidget(header)

        desc = QLabel(
            "Column width = proportion of first variable. "
            "Cell height = conditional proportion of second variable within that column."
        )
        desc.setWordWrap(True)
        desc.setProperty("muted", True)
        layout.addWidget(desc)

        controls_group = QGroupBox("Variables")
        ctrl = QHBoxLayout(controls_group)
        ctrl.addWidget(QLabel("X (column):"))
        self._col_combo = QComboBox()
        self._col_combo.currentTextChanged.connect(self._refresh)
        ctrl.addWidget(self._col_combo, 1)
        ctrl.addWidget(QLabel("Y (color):"))
        self._row_combo = QComboBox()
        self._row_combo.currentTextChanged.connect(self._refresh)
        ctrl.addWidget(self._row_combo, 1)
        self._residual_cb = QCheckBox("Color by Pearson residual")
        self._residual_cb.setToolTip("Blue = more than expected, Red = less than expected")
        self._residual_cb.stateChanged.connect(self._refresh)
        ctrl.addWidget(self._residual_cb)
        layout.addWidget(controls_group)

        chart_group = QGroupBox("Mosaic")
        chart_layout = QVBoxLayout(chart_group)
        self._chart = _MosaicWidget()
        chart_layout.addWidget(self._chart)
        layout.addWidget(chart_group, 1)

        self._status_label = QLabel("Load a dataset to display a mosaic plot.")
        self._status_label.setProperty("muted", True)
        layout.addWidget(self._status_label)

    def set_input_payload(self, payload) -> None:
        dataset = payload.dataset if payload is not None else None
        self._dataset = dataset
        for combo in (self._col_combo, self._row_combo):
            combo.blockSignals(True)
            combo.clear()
        if dataset is not None:
            cat_cols = [
                col.name
                for col in dataset.domain.columns
                if col.logical_type in ("categorical", "string") or col.unique_count_hint <= 30
            ]
            if not cat_cols:
                cat_cols = [col.name for col in dataset.domain.columns]
            self._col_combo.addItems(cat_cols)
            self._row_combo.addItems(cat_cols)
            if len(cat_cols) >= 2:
                self._row_combo.setCurrentIndex(1)
        for combo in (self._col_combo, self._row_combo):
            combo.blockSignals(False)
        self._refresh()

    def documentation_url(self) -> str:
        return "https://orangedatamining.com/widget-catalog/visualize/mosaicdisplay/"

    def _refresh(self) -> None:
        if self._dataset is None:
            self._chart.set_data({}, {}, [], [], "", "")
            self._status_label.setText("Load a dataset to display a mosaic plot.")
            return

        col_name = self._col_combo.currentText()
        row_name = self._row_combo.currentText()

        if not col_name or not row_name or col_name == row_name:
            self._chart.set_data({}, {}, [], [], "", "")
            self._status_label.setText("Select two different categorical columns.")
            return

        df = self._dataset.dataframe
        try:
            col_series = df[col_name].to_list()
            row_series = df[row_name].to_list()
        except Exception:
            self._chart.set_data({}, {}, [], [], "", "")
            self._status_label.setText("Column not found.")
            return

        col_counts: dict[str, int] = {}
        joint: dict[str, dict[str, int]] = {}

        for cv, rv in zip(col_series, row_series):
            ck = str(cv) if cv is not None else "(missing)"
            rk = str(rv) if rv is not None else "(missing)"
            col_counts[ck] = col_counts.get(ck, 0) + 1
            if ck not in joint:
                joint[ck] = {}
            joint[ck][rk] = joint[ck].get(rk, 0) + 1

        row_totals: dict[str, int] = {}
        for rv in row_series:
            rk = str(rv) if rv is not None else "(missing)"
            row_totals[rk] = row_totals.get(rk, 0) + 1

        col_vals = [k for k, _ in sorted(col_counts.items(), key=lambda x: -x[1])[:10]]
        row_vals = [k for k, _ in sorted(row_totals.items(), key=lambda x: -x[1])[:8]]

        self._chart.set_data(joint, col_counts, col_vals, row_vals, col_name, row_name,
                             color_by_residual=self._residual_cb.isChecked())
        self._status_label.setText(
            f"X: '{col_name}' ({len(col_vals)} categories)  |  "
            f"Color: '{row_name}' ({len(row_vals)} categories)  |  N = {len(df)}"
        )
