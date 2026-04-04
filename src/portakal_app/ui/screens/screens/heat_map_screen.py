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
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from portakal_app.data.models import DatasetHandle
from portakal_app.ui.screens.node_screen import WorkflowNodeScreenSupport


# ── Colour palette (Orange default: blue-black-yellow) ─────────────────
def _heat_color(value: float, vmin: float, vmax: float) -> QColor:
    """Map value to Orange-style blue→black→yellow diverging palette."""
    if vmax == vmin:
        t = 0.5
    else:
        t = (value - vmin) / (vmax - vmin)
    t = max(0.0, min(1.0, t))

    if t < 0.5:
        # blue (0,0,1) → black (0,0,0)
        s = t * 2  # 0..1
        r = 0
        g = 0
        b = int(255 * (1 - s))
    else:
        # black (0,0,0) → yellow (1,1,0)
        s = (t - 0.5) * 2  # 0..1
        r = int(255 * s)
        g = int(255 * s)
        b = 0
    return QColor(r, g, b)


# ── Minimal Ward-linkage hierarchical clustering ────────────────────────
def _ward_cluster_order(matrix: list[list[float]]) -> list[int]:
    """
    Return a row-ordering using a simplified Ward-linkage agglomerative
    clustering (complete-linkage approximation for speed).
    Returns the leaf order that makes similar rows adjacent.
    """
    n = len(matrix)
    if n <= 1:
        return list(range(n))

    # Distance matrix: Euclidean between rows
    def row_dist(i: int, j: int) -> float:
        return math.sqrt(sum((matrix[i][k] - matrix[j][k]) ** 2 for k in range(len(matrix[0]))))

    # Start: each row is its own cluster (represented as list of indices)
    clusters: list[list[int]] = [[i] for i in range(n)]
    dist: dict[tuple[int, int], float] = {}
    for i in range(n):
        for j in range(i + 1, n):
            dist[(i, j)] = row_dist(i, j)

    # Agglomerate
    while len(clusters) > 1:
        # Find closest pair of clusters
        min_d = float("inf")
        best = (0, 1)
        c_ids = list(range(len(clusters)))
        for a in range(len(clusters)):
            for b in range(a + 1, len(clusters)):
                # Use average linkage: mean distance between all member pairs
                total = 0.0
                count = 0
                for i in clusters[a]:
                    for j in clusters[b]:
                        key = (min(i, j), max(i, j))
                        total += dist.get(key, row_dist(i, j))
                        count += 1
                avg = total / max(1, count)
                if avg < min_d:
                    min_d = avg
                    best = (a, b)
        a, b = best
        merged = clusters[a] + clusters[b]
        new_clusters = [clusters[i] for i in range(len(clusters)) if i not in (a, b)]
        new_clusters.append(merged)
        clusters = new_clusters

    return clusters[0] if clusters else list(range(n))


# ── Widget ─────────────────────────────────────────────────────────────
class _HeatMapWidget(QWidget):
    """Custom QPainter heat map with clustering-based row/col ordering."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._matrix: list[list[float]] = []
        self._labels: list[str] = []
        self._row_order: list[int] = []
        self._col_order: list[int] = []
        self._vmin = 0.0
        self._vmax = 1.0
        self._show_values = True
        self.setMinimumHeight(200)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

    def set_matrix(
        self,
        matrix: list[list[float]],
        labels: list[str],
        row_order: list[int],
        col_order: list[int],
        vmin: float,
        vmax: float,
    ) -> None:
        self._matrix = matrix
        self._labels = labels
        self._row_order = row_order
        self._col_order = col_order
        self._vmin = vmin
        self._vmax = vmax
        self.update()

    def set_show_values(self, show: bool) -> None:
        self._show_values = show
        self.update()

    def paintEvent(self, _event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), QColor("#fffdf9"))

        if not self._matrix:
            painter.setPen(QColor("#8d877d"))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter,
                             "No data.\nLoad a dataset with at least 2 numeric columns.")
            return

        n = len(self._labels)
        w, h = self.width(), self.height()
        margin_l = 88
        margin_t = 88
        margin_r = 20
        margin_b = 36

        cell_size = min(
            (w - margin_l - margin_r) // max(1, n),
            (h - margin_t - margin_b) // max(1, n),
        )
        cell_size = max(14, cell_size)

        # Column headers (rotated 45°)
        for ci_idx, ci in enumerate(self._col_order):
            label = self._labels[ci]
            lbl = label if len(label) <= 10 else label[:9] + "…"
            cx = margin_l + ci_idx * cell_size + cell_size // 2
            painter.save()
            painter.translate(cx, margin_t - 4)
            painter.rotate(-45)
            painter.setPen(QColor("#3b2a10"))
            painter.drawText(-30, -12, 60, 14, Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter, lbl)
            painter.restore()

        # Row headers
        for ri_idx, ri in enumerate(self._row_order):
            label = self._labels[ri]
            lbl = label if len(label) <= 14 else label[:13] + "…"
            ry = margin_t + ri_idx * cell_size + cell_size // 2
            painter.setPen(QColor("#3b2a10"))
            painter.drawText(0, ry - 8, margin_l - 4, 16,
                             Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignRight, lbl)

        # Cells
        for ri_idx, ri in enumerate(self._row_order):
            for ci_idx, ci in enumerate(self._col_order):
                value = self._matrix[ri][ci]
                color = _heat_color(value, self._vmin, self._vmax)
                cx = margin_l + ci_idx * cell_size
                cy = margin_t + ri_idx * cell_size

                painter.setBrush(color)
                painter.setPen(QPen(QColor("#fffdf9"), 0.5))
                painter.drawRect(cx, cy, cell_size, cell_size)

                if self._show_values and cell_size >= 30:
                    # White text on dark cells, dark on bright
                    brightness = (color.red() + color.green() + color.blue()) / 765.0
                    text_color = QColor("#ffffff") if brightness < 0.45 else QColor("#1a1a1a")
                    painter.setPen(text_color)
                    painter.drawText(cx + 1, cy + 1, cell_size - 2, cell_size - 2,
                                     Qt.AlignmentFlag.AlignCenter, f"{value:.2f}")

        # Colour scale legend
        legend_x = margin_l
        legend_y = h - margin_b + 6
        legend_w = min(cell_size * n, 200)
        legend_h = 10
        if legend_y + legend_h + 18 <= h:
            steps = 24
            for s in range(steps):
                t = s / (steps - 1)
                val = self._vmin + t * (self._vmax - self._vmin)
                color = _heat_color(val, self._vmin, self._vmax)
                painter.setBrush(color)
                painter.setPen(Qt.PenStyle.NoPen)
                painter.drawRect(legend_x + s * (legend_w // steps), legend_y, legend_w // steps + 1, legend_h)
            painter.setPen(QColor("#534b40"))
            painter.drawText(legend_x, legend_y + legend_h + 2, 60, 14, Qt.AlignmentFlag.AlignLeft, f"{self._vmin:.2f}")
            painter.drawText(legend_x + legend_w - 40, legend_y + legend_h + 2, 60, 14, Qt.AlignmentFlag.AlignLeft, f"{self._vmax:.2f}")
            # Label "low" / "high"
            painter.drawText(legend_x, legend_y - 14, 60, 14, Qt.AlignmentFlag.AlignLeft, "Low")
            painter.drawText(legend_x + legend_w - 30, legend_y - 14, 60, 14, Qt.AlignmentFlag.AlignLeft, "High")

        painter.end()


class HeatMapScreen(QWidget, WorkflowNodeScreenSupport):
    """
    Heat Map – numeric matrix visualisation with hierarchical clustering.

    Colour palette: Blue (low) → Black → Yellow (high)  (Orange default).
    Clustering: average-linkage agglomerative (approximates Ward linkage).
    Modes: raw values  OR  Pearson correlation matrix.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._init_workflow_node_support()
        self._dataset: DatasetHandle | None = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        header = QLabel("Heat Map")
        header.setProperty("sectionTitle", True)
        layout.addWidget(header)

        desc = QLabel(
            "Blue → Black → Yellow colour scale (Orange default). "
            "Rows and columns ordered by hierarchical clustering. "
            "Supports raw values or Pearson correlation matrix."
        )
        desc.setWordWrap(True)
        desc.setProperty("muted", True)
        layout.addWidget(desc)

        controls_group = QGroupBox("Settings")
        ctrl = QHBoxLayout(controls_group)

        ctrl.addWidget(QLabel("Mode:"))
        self._mode_combo = QComboBox()
        self._mode_combo.addItems(["Pearson correlation", "Raw values (normalised)"])
        self._mode_combo.currentIndexChanged.connect(self._refresh)
        ctrl.addWidget(self._mode_combo)

        ctrl.addWidget(QLabel("Max cols:"))
        self._max_cols_combo = QComboBox()
        self._max_cols_combo.addItems(["10", "15", "20", "All"])
        self._max_cols_combo.currentTextChanged.connect(self._refresh)
        ctrl.addWidget(self._max_cols_combo)

        self._cluster_cb = QCheckBox("Cluster rows/cols")
        self._cluster_cb.setChecked(True)
        self._cluster_cb.stateChanged.connect(self._refresh)
        ctrl.addWidget(self._cluster_cb)

        self._show_values_cb = QCheckBox("Show values")
        self._show_values_cb.setChecked(True)
        self._show_values_cb.stateChanged.connect(self._on_show_values_changed)
        ctrl.addWidget(self._show_values_cb)

        ctrl.addStretch(1)
        layout.addWidget(controls_group)

        chart_group = QGroupBox("Matrix")
        chart_layout = QVBoxLayout(chart_group)
        self._chart = _HeatMapWidget()
        chart_layout.addWidget(self._chart)
        layout.addWidget(chart_group, 1)

        self._status_label = QLabel("Load a dataset with numeric columns.")
        self._status_label.setProperty("muted", True)
        layout.addWidget(self._status_label)

    def set_input_payload(self, payload) -> None:
        dataset = payload.dataset if payload is not None else None
        self._dataset = dataset
        self._refresh()

    def documentation_url(self) -> str:
        return "https://orangedatamining.com/widget-catalog/visualize/heatmap/"

    def _on_show_values_changed(self, state: int) -> None:
        self._chart.set_show_values(bool(state))

    def _refresh(self) -> None:
        if self._dataset is None:
            self._chart.set_matrix([], [], [], [], 0.0, 1.0)
            self._status_label.setText("Load a dataset with numeric columns.")
            return

        df = self._dataset.dataframe
        numeric_cols = [col.name for col in self._dataset.domain.columns if df[col.name].dtype.is_numeric()]

        max_cols_text = self._max_cols_combo.currentText()
        if max_cols_text != "All":
            numeric_cols = numeric_cols[:int(max_cols_text)]

        if len(numeric_cols) < 2:
            self._chart.set_matrix([], [], [], [], 0.0, 1.0)
            self._status_label.setText("Need at least 2 numeric columns.")
            return

        n = len(numeric_cols)
        n_rows = len(df)
        mode = self._mode_combo.currentText()

        # Build matrix
        if mode.startswith("Pearson"):
            matrix = self._build_corr_matrix(df, numeric_cols, n_rows)
            vmin, vmax = -1.0, 1.0
            mode_str = "Pearson correlation"
        else:
            matrix = self._build_normalised_matrix(df, numeric_cols, n_rows)
            vmin, vmax = 0.0, 1.0
            mode_str = "normalised raw values"

        # Clustering order
        if self._cluster_cb.isChecked() and n <= 20:
            row_order = _ward_cluster_order(matrix)
            # For columns: transpose matrix and cluster
            transposed = [[matrix[r][c] for r in range(n)] for c in range(n)]
            col_order = _ward_cluster_order(transposed)
        else:
            row_order = list(range(n))
            col_order = list(range(n))

        self._chart.set_matrix(matrix, numeric_cols, row_order, col_order, vmin, vmax)
        self._chart.set_show_values(self._show_values_cb.isChecked())
        self._status_label.setText(
            f"{n} columns · {n_rows} rows · {mode_str}"
            + (" · clustered" if self._cluster_cb.isChecked() else "")
        )

    def _build_corr_matrix(self, df, numeric_cols: list[str], n_rows: int) -> list[list[float]]:
        """Pearson correlation matrix."""
        col_data: dict[str, list[float]] = {}
        col_mean: dict[str, float] = {}
        col_std: dict[str, float] = {}
        for col in numeric_cols:
            vals = [float(df[col][i]) for i in range(n_rows) if df[col][i] is not None]
            vals = vals or [0.0]
            mn = sum(vals) / len(vals)
            var = sum((v - mn) ** 2 for v in vals) / max(1, len(vals))
            col_data[col] = vals
            col_mean[col] = mn
            col_std[col] = math.sqrt(var) if var > 0 else 1.0

        matrix: list[list[float]] = []
        for rc in numeric_cols:
            row = []
            for cc in numeric_cols:
                if rc == cc:
                    row.append(1.0)
                    continue
                va, vb = col_data[rc], col_data[cc]
                n_common = min(len(va), len(vb))
                ma, mb = col_mean[rc], col_mean[cc]
                sa, sb = col_std[rc], col_std[cc]
                cov = sum((va[k] - ma) * (vb[k] - mb) for k in range(n_common)) / n_common
                r = max(-1.0, min(1.0, cov / (sa * sb)))
                row.append(r)
            matrix.append(row)
        return matrix

    def _build_normalised_matrix(self, df, numeric_cols: list[str], n_rows: int) -> list[list[float]]:
        """
        Row-normalised matrix: each column's values normalised to [0,1].
        Displayed as column × column matrix of mean normalised values
        (diagonal = 1, off-diagonal = mean absolute correlation proxy).
        """
        col_norm: dict[str, list[float]] = {}
        for col in numeric_cols:
            vals = [float(df[col][i]) if df[col][i] is not None else 0.0 for i in range(n_rows)]
            mn, mx = min(vals), max(vals)
            rng = mx - mn if mx != mn else 1.0
            col_norm[col] = [(v - mn) / rng for v in vals]

        # Return mean per (row_col, col_col) pair as a heat value
        n = len(numeric_cols)
        matrix: list[list[float]] = []
        for ri, rc in enumerate(numeric_cols):
            row = []
            for ci, cc in enumerate(numeric_cols):
                if ri == ci:
                    row.append(1.0)
                else:
                    combined = [(a + b) / 2 for a, b in zip(col_norm[rc], col_norm[cc])]
                    row.append(sum(combined) / max(1, len(combined)))
            matrix.append(row)
        return matrix
