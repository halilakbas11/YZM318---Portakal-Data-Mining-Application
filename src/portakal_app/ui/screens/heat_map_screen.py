from __future__ import annotations

import math

import numpy as np

from PySide6.QtCore import Qt, QRect
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


# ── Colour palettes ─────────────────────────────────────────────────────────────

_PALETTES = [
    "Blue-Black-Yellow",   # Orange default
    "Blue-White-Red",      # standard diverging (good for correlation)
    "Green-Black-Red",     # diverging (green=positive, red=negative)
    "Grayscale",           # simple
    "Black-White-Orange",  # Portakal branded
]

# Annotation strip palette (categorical)
_ANN_PALETTE = [
    QColor("#e07020"), QColor("#3b82f6"), QColor("#22c55e"),
    QColor("#a855f7"), QColor("#f43f5e"), QColor("#0ea5e9"),
    QColor("#f59e0b"), QColor("#10b981"), QColor("#64748b"),
]


def _heat_color(t: float, palette: str = "Blue-Black-Yellow") -> QColor:
    """
    Map t ∈ [0, 1] to a colour according to the selected palette.
    """
    t = max(0.0, min(1.0, t))
    if palette == "Blue-Black-Yellow":
        # Orange default: 0=blue, 0.5=black, 1=yellow
        if t < 0.5:
            b = int(255 * (1.0 - 2.0 * t))
            return QColor(0, 0, b)
        else:
            rg = int(255 * (2.0 * t - 1.0))
            return QColor(rg, rg, 0)
    elif palette == "Blue-White-Red":
        # Standard diverging: 0=blue, 0.5=white, 1=red
        if t < 0.5:
            v = int(255 * 2.0 * t)
            return QColor(v, v, 255)
        else:
            v = int(255 * (2.0 * t - 1.0))
            return QColor(255, 255 - v, 255 - v)
    elif palette == "Green-Black-Red":
        if t < 0.5:
            g = int(255 * (1.0 - 2.0 * t))
            return QColor(0, g, 0)
        else:
            r = int(255 * (2.0 * t - 1.0))
            return QColor(r, 0, 0)
    elif palette == "Grayscale":
        v = int(255 * t)
        return QColor(v, v, v)
    elif palette == "Black-White-Orange":
        if t < 0.5:
            v = int(255 * 2.0 * t)
            return QColor(v, v, v)
        else:
            f = 2.0 * t - 1.0
            r = int(224 + (255 - 224) * f)
            g = int(f * 112)
            b = int(f * 32)
            return QColor(r, g, b)
    # fallback
    v = int(255 * t)
    return QColor(v, v, v)


# ── Clustering ─────────────────────────────────────────────────────────────────

def _average_linkage_order(matrix: list[list[float]]) -> list[int]:
    """
    Average-linkage hierarchical clustering leaf order.
    Returns row indices sorted so similar rows are adjacent.
    O(n³) — only called for n ≤ 30 to keep UI responsive.
    """
    n = len(matrix)
    if n <= 1:
        return list(range(n))

    ncols = len(matrix[0]) if matrix else 1

    def dist(i: int, j: int) -> float:
        return math.sqrt(sum((matrix[i][k] - matrix[j][k]) ** 2 for k in range(ncols)))

    # Pre-compute pairwise distances
    d: dict[tuple[int, int], float] = {}
    for i in range(n):
        for j in range(i + 1, n):
            d[(i, j)] = dist(i, j)

    clusters: list[list[int]] = [[i] for i in range(n)]

    while len(clusters) > 1:
        best_d = math.inf
        best = (0, 1)
        for a in range(len(clusters)):
            for b in range(a + 1, len(clusters)):
                total, count = 0.0, 0
                for i in clusters[a]:
                    for j in clusters[b]:
                        key = (min(i, j), max(i, j))
                        total += d.get(key, dist(i, j))
                        count += 1
                avg = total / max(1, count)
                if avg < best_d:
                    best_d = avg
                    best = (a, b)
        a, b = best
        clusters.append(clusters[a] + clusters[b])
        clusters = [c for k, c in enumerate(clusters) if k not in (a, b)]

    return clusters[0] if clusters else list(range(n))


# ── Canvas widget ──────────────────────────────────────────────────────────────

class _HeatMapWidget(QWidget):
    """
    Heat map canvas.

    • Colour: Orange's blue→black→yellow scale
    • Row/column labels: column headers rotated 45°, font-metrics truncation
    • Hover tooltip: row name, column name, value
    • Colour-scale legend bar with min / midpoint / max labels
    • Show-values option (text inside cells when large enough)
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._matrix: list[list[float]] = []
        self._row_labels: list[str] = []
        self._col_labels: list[str] = []
        self._row_order: list[int] = []
        self._col_order: list[int] = []
        self._vmin = -1.0
        self._vmax = 1.0
        self._show_values = True
        self._show_grid = True
        self._palette = "Blue-Black-Yellow"
        # Annotation strip: list of (str_value) parallel to row_labels original order
        self._row_annotation: list[str] = []
        self._ann_unique: list[str] = []       # sorted unique annotation values
        self._cell_rects: list[tuple[QRect, str]] = []
        self.setMinimumHeight(200)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMouseTracking(True)

    def set_matrix(
        self,
        matrix: list[list[float]],
        row_labels: list[str],
        col_labels: list[str],
        row_order: list[int],
        col_order: list[int],
        vmin: float,
        vmax: float,
        row_annotation: list[str] | None = None,
    ) -> None:
        self._matrix = matrix
        self._row_labels = row_labels
        self._col_labels = col_labels
        self._row_order = row_order
        self._col_order = col_order
        self._vmin = vmin
        self._vmax = vmax
        self._row_annotation = row_annotation or []
        self._ann_unique = sorted(set(self._row_annotation)) if self._row_annotation else []
        self._cell_rects = []
        # Resize preferred width so scroll area reacts
        n_cols = len(col_order)
        ann_w = 14 if self._row_annotation else 0
        self.setMinimumWidth(88 + n_cols * 16 + 20 + ann_w)
        self.update()

    def set_show_values(self, show: bool) -> None:
        self._show_values = show
        self.update()

    def set_show_grid(self, show: bool) -> None:
        self._show_grid = show
        self.update()

    def set_palette(self, palette: str) -> None:
        self._palette = palette
        self.update()

    # ── Tooltip ────────────────────────────────────────────────────────────────

    def mouseMoveEvent(self, event) -> None:
        pos = event.position().toPoint()
        for rect, tip in self._cell_rects:
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

        if not self._matrix:
            painter.setPen(QColor("#8d877d"))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter,
                             "No data.\nLoad a dataset with at least 2 numeric columns.")
            return

        self._cell_rects = []

        n_rows = len(self._row_order)
        n_cols = len(self._col_order)
        w, h = self.width(), self.height()

        # Annotation strip on the left
        ann_strip_w = 14 if self._row_annotation else 0

        margin_l = 96 + ann_strip_w
        margin_t = 96
        margin_r = 20
        margin_b = 40

        avail_w = w - margin_l - margin_r
        avail_h = h - margin_t - margin_b
        cell_w = max(14, avail_w // max(1, n_cols))
        cell_h = max(14, avail_h // max(1, n_rows))
        # Square cells for correlation matrices
        if n_rows == n_cols:
            cell_w = cell_h = min(cell_w, cell_h)

        span = (self._vmax - self._vmin) or 1.0

        painter.setFont(QFont(self.font().family(), 8))
        fm = QFontMetrics(painter.font())

        # ── Column headers (rotated 45°) ───────────────────────────────────────
        for ci_idx, ci in enumerate(self._col_order):
            lbl = fm.elidedText(
                self._col_labels[ci], Qt.TextElideMode.ElideRight, 64
            )
            cx = margin_l + ci_idx * cell_w + cell_w // 2
            painter.save()
            painter.translate(cx, margin_t - 6)
            painter.rotate(-45)
            painter.setPen(QColor("#3b2a10"))
            painter.drawText(-32, -12, 64, 14,
                             Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
                             lbl)
            painter.restore()

        # ── Row headers & annotation strip ────────────────────────────────────
        for ri_idx, ri in enumerate(self._row_order):
            lbl = fm.elidedText(
                self._row_labels[ri], Qt.TextElideMode.ElideRight, 96 - 8 - ann_strip_w
            )
            ry = margin_t + ri_idx * cell_h + cell_h // 2
            painter.setPen(QColor("#3b2a10"))
            painter.drawText(ann_strip_w, ry - 8, 96 - 4 - ann_strip_w, 16,
                             Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignRight,
                             lbl)

            # Annotation colour strip
            if self._row_annotation and ri < len(self._row_annotation):
                ann_val = self._row_annotation[ri]
                ann_idx = self._ann_unique.index(ann_val) if ann_val in self._ann_unique else 0
                ann_color = _ANN_PALETTE[ann_idx % len(_ANN_PALETTE)]
                painter.setBrush(ann_color)
                painter.setPen(Qt.PenStyle.NoPen)
                painter.drawRect(96, margin_t + ri_idx * cell_h, ann_strip_w - 2, cell_h)

        # ── Annotation legend ──────────────────────────────────────────────────
        if self._ann_unique:
            painter.setFont(QFont(self.font().family(), 7))
            fm7 = QFontMetrics(painter.font())
            for ai, av in enumerate(self._ann_unique[:9]):
                lx = w - margin_r - 100
                ly = margin_t + ai * 16
                ac = _ANN_PALETTE[ai % len(_ANN_PALETTE)]
                painter.setBrush(ac)
                painter.setPen(Qt.PenStyle.NoPen)
                painter.drawRect(lx, ly + 2, 10, 10)
                painter.setPen(QColor("#3b2a10"))
                painter.drawText(lx + 14, ly, 80, 14,
                                 Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft,
                                 fm7.elidedText(av, Qt.TextElideMode.ElideRight, 76))
            painter.setFont(QFont(self.font().family(), 8))
            fm = QFontMetrics(painter.font())

        # ── Cells ──────────────────────────────────────────────────────────────
        for ri_idx, ri in enumerate(self._row_order):
            for ci_idx, ci in enumerate(self._col_order):
                if ri >= len(self._matrix) or ci >= len(self._matrix[ri]):
                    continue
                value = self._matrix[ri][ci]
                t = (value - self._vmin) / span
                color = _heat_color(t, self._palette)

                cx = margin_l + ci_idx * cell_w
                cy = margin_t + ri_idx * cell_h

                painter.setBrush(color)
                pen_color = QColor("#fffdf9") if self._show_grid else color
                painter.setPen(QPen(pen_color, 0.5))
                painter.drawRect(cx, cy, cell_w, cell_h)

                # Value text inside cell when large enough
                if self._show_values and cell_w >= 28 and cell_h >= 18:
                    brightness = (color.red() + color.green() + color.blue()) / 765.0
                    txt_col = QColor("#ffffff") if brightness < 0.4 else QColor("#111111")
                    painter.setPen(txt_col)
                    painter.setFont(QFont(self.font().family(), 7))
                    painter.drawText(cx + 1, cy + 1, cell_w - 2, cell_h - 2,
                                     Qt.AlignmentFlag.AlignCenter, f"{value:.2f}")
                    painter.setFont(QFont(self.font().family(), 8))

                # Tooltip (include annotation if present)
                row_lbl = self._row_labels[ri]
                col_lbl = self._col_labels[ci]
                ann_tip = ""
                if self._row_annotation and ri < len(self._row_annotation):
                    ann_tip = f"<br>Annotation: {self._row_annotation[ri]}"
                self._cell_rects.append((
                    QRect(cx, cy, cell_w, cell_h),
                    f"<b>{row_lbl}</b> × <b>{col_lbl}</b><br>Value: {value:.4f}{ann_tip}",
                ))

        # ── Colour scale legend bar ────────────────────────────────────────────
        legend_y = margin_t + n_rows * cell_h + 8
        legend_x = margin_l
        legend_w_px = min(n_cols * cell_w, 240)
        legend_h = 12
        if legend_y + legend_h + 20 <= h:
            steps = 40
            step_w = max(1, legend_w_px // steps)
            for s in range(steps):
                t = s / (steps - 1)
                color = _heat_color(t, self._palette)
                painter.setBrush(color)
                painter.setPen(Qt.PenStyle.NoPen)
                painter.drawRect(legend_x + s * step_w, legend_y, step_w + 1, legend_h)

            painter.setPen(QColor("#534b40"))
            # Min label
            painter.drawText(legend_x, legend_y + legend_h + 2, 60, 14,
                             Qt.AlignmentFlag.AlignLeft, f"{self._vmin:.2f}")
            # Midpoint label (important for correlation: mid = 0)
            mid_val = (self._vmin + self._vmax) / 2
            painter.drawText(
                legend_x + legend_w_px // 2 - 20, legend_y + legend_h + 2, 40, 14,
                Qt.AlignmentFlag.AlignCenter, f"{mid_val:.2f}",
            )
            # Max label
            painter.drawText(legend_x + legend_w_px - 40, legend_y + legend_h + 2, 60, 14,
                             Qt.AlignmentFlag.AlignRight, f"{self._vmax:.2f}")

        painter.end()


# ── Screen widget ──────────────────────────────────────────────────────────────

class HeatMapScreen(QWidget, WorkflowNodeScreenSupport):
    """
    Heat Map – numeric matrix visualisation.

    Modes
    ─────
    • Pearson correlation: n_cols × n_cols matrix via numpy (np.corrcoef with
      pairwise complete observations — NaN-safe, correct row alignment).
    • Raw values: data matrix (rows = instances, cols = features),
      per-column normalised to [0, 1] for comparable colour mapping.

    Colour: Orange's blue→black→yellow diverging palette.
    Clustering: average-linkage hierarchical (rows, and cols for square matrices),
                applied when n ≤ 30 for performance.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._init_workflow_node_support()
        self._dataset: DatasetHandle | None = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        header = QLabel("Heat Map")
        header.setProperty("sectionTitle", True)
        layout.addWidget(header)

        desc = QLabel(
            "Blue → Black → Yellow colour scale (Orange default). "
            "Pearson correlation mode shows feature–feature correlations. "
            "Raw values mode shows the data matrix (per-column normalised)."
        )
        desc.setWordWrap(True)
        desc.setProperty("muted", True)
        layout.addWidget(desc)

        ctrl_box = QGroupBox("Settings")
        ctrl_layout = QVBoxLayout(ctrl_box)
        ctrl_layout.setSpacing(4)

        # Row 1: Mode, Color palette, Max cols, Cluster
        row1 = QHBoxLayout()
        row1.setSpacing(6)
        row1.addWidget(QLabel("Mode:"))
        self._mode_combo = QComboBox()
        self._mode_combo.addItems(["Pearson correlation", "Raw values"])
        self._mode_combo.currentIndexChanged.connect(self._refresh)
        row1.addWidget(self._mode_combo)

        row1.addWidget(QLabel("Color:"))
        self._palette_combo = QComboBox()
        self._palette_combo.addItems(_PALETTES)
        self._palette_combo.currentTextChanged.connect(self._on_palette_changed)
        row1.addWidget(self._palette_combo)

        row1.addWidget(QLabel("Max cols:"))
        self._max_cols_combo = QComboBox()
        self._max_cols_combo.addItems(["10", "15", "20", "30", "All"])
        self._max_cols_combo.currentTextChanged.connect(self._refresh)
        row1.addWidget(self._max_cols_combo)


        row1.addStretch(1)
        ctrl_layout.addLayout(row1)

        # Row 2: Annotate rows, Merge rows by, Show values, Show grid
        row2 = QHBoxLayout()
        row2.setSpacing(6)

        row2.addWidget(QLabel("Annotate rows:"))
        self._row_annot_combo = QComboBox()
        self._row_annot_combo.setMinimumWidth(110)
        self._row_annot_combo.currentTextChanged.connect(self._refresh)
        row2.addWidget(self._row_annot_combo)

        row2.addWidget(QLabel("Merge rows by:"))
        self._merge_combo = QComboBox()
        self._merge_combo.setMinimumWidth(110)
        self._merge_combo.currentTextChanged.connect(self._refresh)
        row2.addWidget(self._merge_combo)

        row2.addWidget(QLabel("Sort rows:"))
        self._sort_rows_combo = QComboBox()
        self._sort_rows_combo.addItems(["Cluster", "Alphabetical", "Dataset order"])
        self._sort_rows_combo.currentIndexChanged.connect(self._refresh)
        row2.addWidget(self._sort_rows_combo)

        self._show_values_cb = QCheckBox("Show values")
        self._show_values_cb.setChecked(True)
        self._show_values_cb.stateChanged.connect(self._on_show_values_changed)
        row2.addWidget(self._show_values_cb)

        self._show_grid_cb = QCheckBox("Show grid")
        self._show_grid_cb.setChecked(True)
        self._show_grid_cb.stateChanged.connect(self._on_show_grid_changed)
        row2.addWidget(self._show_grid_cb)

        row2.addStretch(1)
        ctrl_layout.addLayout(row2)

        layout.addWidget(ctrl_box)

        # Scroll area so large matrices stay usable
        self._chart = _HeatMapWidget()
        scroll = QScrollArea()
        scroll.setWidget(self._chart)
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)

        chart_box = QGroupBox("Matrix")
        chart_layout = QVBoxLayout(chart_box)
        chart_layout.setContentsMargins(2, 2, 2, 2)
        chart_layout.addWidget(scroll)
        layout.addWidget(chart_box, 1)

        self._status_label = QLabel(i18n.t("Load a dataset with numeric columns."))
        self._status_label.setProperty("muted", True)
        layout.addWidget(self._status_label)

    # ── WorkflowNodeScreenSupport ─────────────────────────────────────────────

    def set_input_payload(self, payload) -> None:
        dataset = payload.dataset if payload is not None else None
        self._dataset = dataset

        # Populate annotation & merge combos
        for combo in (self._row_annot_combo, self._merge_combo):
            combo.blockSignals(True)
            combo.clear()
            combo.addItem("(none)")
            if dataset is not None:
                for col in dataset.domain.columns:
                    combo.addItem(col.name)
            combo.blockSignals(False)

        self._refresh()

    def documentation_url(self) -> str:
        return "https://orangedatamining.com/widget-catalog/visualize/heatmap/"

    # ── Internal ──────────────────────────────────────────────────────────────

    def _on_show_values_changed(self, state: int) -> None:
        self._chart.set_show_values(bool(state))

    def _on_show_grid_changed(self, state: int) -> None:
        self._chart.set_show_grid(bool(state))

    def _on_palette_changed(self, palette: str) -> None:
        self._chart.set_palette(palette)
        # Also redraw the legend — easiest to do a full refresh
        self._chart.update()

    def _refresh(self) -> None:
        if self._dataset is None:
            self._chart.set_matrix([], [], [], [], [], -1.0, 1.0)
            self._status_label.setText(i18n.t("Load a dataset with numeric columns."))
            return

        df = self._dataset.dataframe
        num_cols = [c.name for c in self._dataset.domain.columns
                    if df[c.name].dtype.is_numeric()]

        max_text = self._max_cols_combo.currentText()
        if max_text != "All":
            try:
                num_cols = num_cols[:int(max_text)]
            except (ValueError, TypeError):
                pass  # invalid value → use all columns

        if len(num_cols) < 2:
            self._chart.set_matrix([], [], [], [], [], -1.0, 1.0)
            self._status_label.setText(i18n.t("Need at least 2 numeric columns."))
            return

        n_rows_df = len(df)
        mode = self._mode_combo.currentText()

        # ── Merge rows by categorical column ─────────────────────────────────
        merge_col = self._merge_combo.currentText()
        if merge_col and merge_col != "(none)" and merge_col in df.columns:
            df, n_rows_df = self._merge_rows(df, merge_col, num_cols)

        if mode == "Pearson correlation":
            matrix, row_lbl, col_lbl, vmin, vmax = self._build_corr_matrix(
                df, num_cols, n_rows_df
            )
            mode_str = "Pearson correlation"
        else:
            matrix, row_lbl, col_lbl, vmin, vmax = self._build_raw_matrix(
                df, num_cols, n_rows_df
            )
            mode_str = "raw values (normalised)"

        n = len(row_lbl)

        # ── Row annotation strip ──────────────────────────────────────────────
        row_annot_col = self._row_annot_combo.currentText()
        row_annotation: list[str] = []
        if row_annot_col and row_annot_col != "(none)" and row_annot_col in df.columns and mode != "Pearson correlation":
            raw_ann = df[row_annot_col].to_list()
            # Only use sampled rows matching row_lbl length in raw mode
            sample_step = max(1, n_rows_df // 200)
            sample_idx = list(range(0, n_rows_df, sample_step))[:200]
            row_annotation = [
                str(raw_ann[i]) if i < len(raw_ann) and raw_ann[i] is not None else ""
                for i in sample_idx[:n]
            ]

        # ── Row ordering ──────────────────────────────────────────────────────
        sort_mode = self._sort_rows_combo.currentText()
        cluster = (sort_mode == "Cluster") and n <= 30

        if cluster:
            row_order = _average_linkage_order(matrix)
            if row_lbl == col_lbl:  # square matrix → cluster cols too
                col_order = _average_linkage_order(
                    [[matrix[c][r] for c in range(n)] for r in range(n)]
                )
            else:
                col_order = list(range(len(col_lbl)))
        elif sort_mode == "Alphabetical":
            row_order = sorted(range(n), key=lambda i: row_lbl[i])
            col_order = sorted(range(len(col_lbl)), key=lambda i: col_lbl[i])
        else:
            row_order = list(range(n))
            col_order = list(range(len(col_lbl)))

        self._chart.set_matrix(matrix, row_lbl, col_lbl, row_order, col_order,
                                vmin, vmax, row_annotation)
        self._chart.set_show_values(self._show_values_cb.isChecked())
        self._chart.set_show_grid(self._show_grid_cb.isChecked())
        self._chart.set_palette(self._palette_combo.currentText())
        self._status_label.setText(
            f"{len(col_lbl)} columns · {n_rows_df} rows · {mode_str}"
            + (" · clustered" if cluster else "")
            + (f" · annotated by '{row_annot_col}'" if row_annotation else "")
        )

    def _merge_rows(self, df, merge_col: str, num_cols: list[str]):
        """
        Group rows by merge_col (categorical), compute column means per group.
        Returns a new in-memory dict-like object that supports df[col][i] access.
        Uses a simple dict-based approach compatible with the rest of the code.
        """
        import polars as pl
        try:
            # Attempt a polars groupby mean
            agg_exprs = [pl.col(c).mean().alias(c) for c in num_cols]
            merged = df.group_by(merge_col).agg(agg_exprs).sort(merge_col)
            return merged, len(merged)
        except Exception:
            return df, len(df)

    def _build_corr_matrix(self, df, cols: list[str], n_rows: int):
        """
        Pearson correlation matrix (n_cols × n_cols) via numpy.

        Uses pairwise complete observations: for each (col_a, col_b) pair,
        only rows where BOTH columns have valid values are used.
        This is NaN-safe and correctly aligned — unlike the previous pure-Python
        implementation which silently mismatched rows with different missing patterns.
        """
        n = len(cols)
        if n_rows == 0:
            return np.eye(n).tolist(), cols, cols, -1.0, 1.0
        # Build full matrix with NaN for missing values
        X = np.full((n_rows, n), np.nan, dtype=np.float64)
        for k, col in enumerate(cols):
            for i in range(n_rows):
                v = df[col][i]
                if v is not None:
                    X[i, k] = float(v)

        # Pairwise correlation with complete observations
        corr = np.eye(n, dtype=np.float64)
        for a in range(n):
            for b in range(a + 1, n):
                mask = ~(np.isnan(X[:, a]) | np.isnan(X[:, b]))
                if mask.sum() < 2:
                    corr[a, b] = corr[b, a] = 0.0
                    continue
                va = X[mask, a]
                vb = X[mask, b]
                r = float(np.corrcoef(va, vb)[0, 1])
                r = 0.0 if not math.isfinite(r) else max(-1.0, min(1.0, r))
                corr[a, b] = corr[b, a] = r

        matrix = corr.tolist()
        return matrix, cols, cols, -1.0, 1.0

    def _build_raw_matrix(self, df, cols: list[str], n_rows: int):
        """
        Actual data matrix: rows = instances (up to 200), cols = features.
        Each column is normalised to [0, 1] for colour comparability via numpy.
        """
        if n_rows == 0:
            return [], [], cols, 0.0, 1.0

        max_rows = 200
        step = max(1, n_rows // max_rows)
        sample_indices = list(range(0, n_rows, step))[:max_rows]
        n_pts = len(sample_indices)

        if n_pts == 0:
            return [], [], cols, 0.0, 1.0

        # Build matrix with numpy
        X = np.zeros((n_pts, len(cols)), dtype=np.float64)
        for k, col in enumerate(cols):
            for row_i, i in enumerate(sample_indices):
                v = df[col][i]
                X[row_i, k] = float(v) if v is not None else 0.0

        # Per-column min-max normalisation
        col_min = X.min(axis=0)
        col_rng = X.max(axis=0) - col_min
        col_rng[col_rng < 1e-10] = 1.0
        X_norm = (X - col_min) / col_rng

        matrix = X_norm.tolist()
        row_labels = [f"#{sample_indices[i]}" for i in range(n_pts)]
        return matrix, row_labels, cols, 0.0, 1.0
