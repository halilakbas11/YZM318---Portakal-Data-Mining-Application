from __future__ import annotations

import math

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QPainter, QPen
from PySide6.QtWidgets import (
    QComboBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from portakal_app.data.models import DatasetHandle
from portakal_app.ui.screens.node_screen import WorkflowNodeScreenSupport


def _pearson_residual(observed: int, expected: float) -> float:
    """Pearson residual: r = (O - E) / sqrt(E)  (Orange formula)."""
    if expected <= 0:
        return 0.0
    return (observed - expected) / math.sqrt(expected)


def _chi_square(joint: dict[str, dict[str, int]], row_totals: dict[str, int],
                col_totals: dict[str, int], n_total: int) -> float:
    """Pearson chi-square statistic for a contingency table."""
    chi2 = 0.0
    for rv, row_dict in joint.items():
        for cv, observed in row_dict.items():
            expected = (row_totals.get(rv, 0) * col_totals.get(cv, 0)) / max(1, n_total)
            if expected > 0:
                chi2 += (observed - expected) ** 2 / expected
    return chi2


class _SieveWidget(QWidget):
    """
    Sieve Diagram after Riedwyl & Schüpbach (1983) / Orange implementation.

    - Rectangle SIZE ∝ sqrt(expected frequency)  → area ∝ expected
    - Colour: Pearson residual r = (O-E)/sqrt(E)
        r > 0  → blue  (more frequent than expected)
        r < 0  → red   (less frequent than expected)
    - |r| drives colour intensity
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._data: dict[str, dict[str, tuple[int, float, float]]] = {}
        # {row_val: {col_val: (actual, expected, pearson_r)}}
        self._row_vals: list[str] = []
        self._col_vals: list[str] = []
        self.setMinimumHeight(200)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

    def set_data(
        self,
        data: dict[str, dict[str, tuple[int, float, float]]],
        row_vals: list[str],
        col_vals: list[str],
    ) -> None:
        self._data = data
        self._row_vals = row_vals
        self._col_vals = col_vals
        self.update()

    def paintEvent(self, _event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), QColor("#fffdf9"))

        if not self._data:
            painter.setPen(QColor("#8d877d"))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter,
                             "No data.\nSelect two categorical columns.")
            return

        margin_left = 90
        margin_top = 44
        margin_right = 12
        margin_bottom = 62

        w, h = self.width(), self.height()
        chart_w = w - margin_left - margin_right
        chart_h = h - margin_top - margin_bottom
        if chart_w < 10 or chart_h < 10:
            return

        n_rows, n_cols = len(self._row_vals), len(self._col_vals)
        if n_rows == 0 or n_cols == 0:
            return

        cell_w = chart_w / n_cols
        cell_h = chart_h / n_rows

        # Max expected (for scaling rectangle sizes)
        max_expected = max(
            (v[1] for row in self._data.values() for v in row.values()),
            default=1.0
        ) or 1.0

        # Column headers
        painter.setPen(QColor("#3b2a10"))
        for ci, cv in enumerate(self._col_vals):
            label = cv if len(cv) <= 10 else cv[:9] + "…"
            painter.drawText(int(margin_left + ci * cell_w), 0, int(cell_w), margin_top,
                             Qt.AlignmentFlag.AlignCenter, label)

        # Row headers
        for ri, rv in enumerate(self._row_vals):
            label = rv if len(rv) <= 14 else rv[:13] + "…"
            painter.drawText(0, int(margin_top + ri * cell_h), margin_left - 4, int(cell_h),
                             Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignRight, label)

        # Cells
        for ri, rv in enumerate(self._row_vals):
            for ci, cv in enumerate(self._col_vals):
                actual, expected, pearson_r = self._data.get(rv, {}).get(cv, (0, 0.0, 0.0))

                cx = int(margin_left + ci * cell_w)
                cy = int(margin_top + ri * cell_h)

                # Cell outline (grey background)
                painter.setBrush(QColor("#f0ede7"))
                painter.setPen(QPen(QColor("#d0ccc3"), 1))
                painter.drawRect(cx, cy, int(cell_w), int(cell_h))

                if expected > 0:
                    # Inner rectangle size ∝ sqrt(expected)  (area ∝ expected)
                    scale = math.sqrt(expected / max_expected)
                    inner_w = max(4, int(cell_w * 0.88 * scale))
                    inner_h = max(4, int(cell_h * 0.88 * scale))
                    ix = cx + (int(cell_w) - inner_w) // 2
                    iy = cy + (int(cell_h) - inner_h) // 2

                    # Colour intensity from |pearson_r|, clamped to [0, 4]
                    intensity = min(1.0, abs(pearson_r) / 4.0)
                    if pearson_r > 0:
                        # Blue: more than expected
                        fill = QColor(
                            int(255 - 196 * intensity),
                            int(255 - 130 * intensity),
                            255,
                            200,
                        )
                    elif pearson_r < 0:
                        # Red: less than expected
                        fill = QColor(
                            255,
                            int(255 - 196 * intensity),
                            int(255 - 196 * intensity),
                            200,
                        )
                    else:
                        fill = QColor(200, 200, 200, 120)

                    painter.setBrush(fill)
                    painter.setPen(Qt.PenStyle.NoPen)
                    painter.drawRect(ix, iy, inner_w, inner_h)

                    # Count + residual label
                    if cell_w > 36 and cell_h > 22:
                        painter.setPen(QColor("#1a1a1a"))
                        painter.drawText(cx + 2, cy + 2, int(cell_w) - 4, int(cell_h) - 4,
                                         Qt.AlignmentFlag.AlignCenter,
                                         f"{actual}\n({pearson_r:+.1f})")

        # Legend
        legend_y = h - margin_bottom + 8
        for lx, label, color in [
            (margin_left, "r > 0: more than expected", QColor(59, 130, 246, 200)),
            (margin_left + 180, "r < 0: less than expected", QColor(220, 38, 38, 200)),
        ]:
            painter.setBrush(color)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawRect(lx, legend_y, 14, 12)
            painter.setPen(QColor("#534b40"))
            painter.drawText(lx + 18, legend_y, 160, 14, Qt.AlignmentFlag.AlignVCenter, label)

        painter.end()


class SieveDiagramScreen(QWidget, WorkflowNodeScreenSupport):
    """
    Sieve Diagram (Riedwyl & Schüpbach 1983).

    Visualises association between two categorical variables.
    Rectangle area ∝ expected frequency; colour encodes Pearson residual.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._init_workflow_node_support()
        self._dataset: DatasetHandle | None = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        header = QLabel("Sieve Diagram")
        header.setProperty("sectionTitle", True)
        layout.addWidget(header)

        desc = QLabel(
            "Rectangle area ∝ expected frequency (under independence). "
            "Colour = Pearson residual r = (O−E)/√E: "
            "Blue → more frequent than expected, Red → less frequent."
        )
        desc.setWordWrap(True)
        desc.setProperty("muted", True)
        layout.addWidget(desc)

        controls_group = QGroupBox("Variables")
        ctrl = QHBoxLayout(controls_group)
        ctrl.addWidget(QLabel("Row variable:"))
        self._row_combo = QComboBox()
        self._row_combo.currentTextChanged.connect(self._refresh)
        ctrl.addWidget(self._row_combo, 1)
        ctrl.addWidget(QLabel("Column variable:"))
        self._col_combo = QComboBox()
        self._col_combo.currentTextChanged.connect(self._refresh)
        ctrl.addWidget(self._col_combo, 1)
        self._best_btn = QPushButton("Score Combinations")
        self._best_btn.setToolTip("Find the pair with the highest chi-square statistic")
        self._best_btn.clicked.connect(self._find_best_pair)
        ctrl.addWidget(self._best_btn)
        layout.addWidget(controls_group)

        chart_group = QGroupBox("Diagram")
        chart_layout = QVBoxLayout(chart_group)
        self._chart = _SieveWidget()
        chart_layout.addWidget(self._chart)
        layout.addWidget(chart_group, 1)

        self._status_label = QLabel("Load a dataset to visualise associations.")
        self._status_label.setProperty("muted", True)
        layout.addWidget(self._status_label)

    # ── WorkflowNodeScreenSupport ──────────────────────────────────────

    def set_input_payload(self, payload) -> None:
        dataset = payload.dataset if payload is not None else None
        self._dataset = dataset
        for combo in (self._row_combo, self._col_combo):
            combo.blockSignals(True)
            combo.clear()
        if dataset is not None:
            cat_cols = self._get_categorical_columns()
            self._row_combo.addItems(cat_cols)
            self._col_combo.addItems(cat_cols)
            if len(cat_cols) >= 2:
                self._col_combo.setCurrentIndex(1)
        for combo in (self._row_combo, self._col_combo):
            combo.blockSignals(False)
        self._refresh()

    def documentation_url(self) -> str:
        return "https://orangedatamining.com/widget-catalog/visualize/sievediagram/"

    # ── Internal ──────────────────────────────────────────────────────

    def _get_categorical_columns(self) -> list[str]:
        if self._dataset is None:
            return []
        cols = [
            col.name for col in self._dataset.domain.columns
            if col.logical_type in ("categorical", "string") or col.unique_count_hint <= 30
        ]
        return cols or [col.name for col in self._dataset.domain.columns]

    def _build_contingency(self, row_col: str, col_col: str):
        """Return (joint, row_totals, col_totals, n_total)."""
        df = self._dataset.dataframe
        row_series = df[row_col].to_list()
        col_series = df[col_col].to_list()
        joint: dict[str, dict[str, int]] = {}
        row_totals: dict[str, int] = {}
        col_totals: dict[str, int] = {}
        for rv, cv in zip(row_series, col_series):
            rk = str(rv) if rv is not None else "(missing)"
            ck = str(cv) if cv is not None else "(missing)"
            row_totals[rk] = row_totals.get(rk, 0) + 1
            col_totals[ck] = col_totals.get(ck, 0) + 1
            if rk not in joint:
                joint[rk] = {}
            joint[rk][ck] = joint[rk].get(ck, 0) + 1
        n_total = len(df)
        return joint, row_totals, col_totals, n_total

    def _find_best_pair(self) -> None:
        if self._dataset is None:
            return
        cat_cols = self._get_categorical_columns()
        if len(cat_cols) < 2:
            return
        best_chi2 = -1.0
        best_pair = (cat_cols[0], cat_cols[1])
        for i, rc in enumerate(cat_cols):
            for cc in cat_cols[i + 1:]:
                try:
                    joint, row_totals, col_totals, n_total = self._build_contingency(rc, cc)
                    chi2 = _chi_square(joint, row_totals, col_totals, n_total)
                    if chi2 > best_chi2:
                        best_chi2 = chi2
                        best_pair = (rc, cc)
                except Exception:
                    continue
        self._row_combo.setCurrentText(best_pair[0])
        self._col_combo.setCurrentText(best_pair[1])
        self._refresh()
        self._status_label.setText(
            self._status_label.text() + f"  |  Best pair by χ² = {best_chi2:.1f}"
        )

    def _refresh(self) -> None:
        if self._dataset is None:
            self._chart.set_data({}, [], [])
            self._status_label.setText("Load a dataset to visualise associations.")
            return

        row_col = self._row_combo.currentText()
        col_col = self._col_combo.currentText()

        if not row_col or not col_col or row_col == col_col:
            self._chart.set_data({}, [], [])
            self._status_label.setText("Select two different categorical columns.")
            return

        try:
            joint, row_totals, col_totals, n_total = self._build_contingency(row_col, col_col)
        except Exception as e:
            self._chart.set_data({}, [], [])
            self._status_label.setText(f"Error: {e}")
            return

        # Top values for readability
        row_vals = [k for k, _ in sorted(row_totals.items(), key=lambda x: -x[1])[:8]]
        col_vals = [k for k, _ in sorted(col_totals.items(), key=lambda x: -x[1])[:8]]

        # Build data with Pearson residuals
        data: dict[str, dict[str, tuple[int, float, float]]] = {}
        for rv in row_vals:
            data[rv] = {}
            for cv in col_vals:
                actual = joint.get(rv, {}).get(cv, 0)
                expected = (row_totals[rv] / n_total) * (col_totals.get(cv, 0) / n_total) * n_total
                r = _pearson_residual(actual, expected)
                data[rv][cv] = (actual, expected, r)

        chi2 = _chi_square(joint, row_totals, col_totals, n_total)
        self._chart.set_data(data, row_vals, col_vals)
        self._status_label.setText(
            f"Rows: '{row_col}' ({len(row_vals)})  |  "
            f"Columns: '{col_col}' ({len(col_vals)})  |  "
            f"N = {n_total}  |  χ² = {chi2:.2f}"
        )
