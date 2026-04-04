from __future__ import annotations

import math
import random

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QColor, QPainter, QPen
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QSlider,
    QSpinBox,
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


def _freeviz_optimize(
    X: list[list[float]],
    labels: list[int],
    anchors: list[list[float]],
    n_iter: int = 10,
    learning_rate: float = 0.01,
) -> list[list[float]]:
    """
    FreeViz gradient descent optimization (Orange algorithm).

    Physics model (Demsar et al., 2005):
    - Project each point: p_i = Σ_k (x_ik * a_k)   (unnormalized)
    - Force between points i and j:
        same class  → attractive  F = ||p_i - p_j||
        diff class  → repulsive   F = -1 / ||p_i - p_j||^2
    - Gradient on anchor k: ∂E/∂a_k = Σ_ij F_ij * x_ik * (p_i - p_j)
    - Update: a_k -= lr * gradient_k
    - Normalize: a_k /= ||a_k||  (keep on unit circle)
    """
    n_pts = len(X)
    n_feat = len(anchors)
    if n_pts < 2 or n_feat < 2:
        return anchors

    for _ in range(n_iter):
        # Project all points: p_i = Σ_k x_ik * a_k
        projected = []
        for row in X:
            px = sum(row[k] * anchors[k][0] for k in range(n_feat))
            py = sum(row[k] * anchors[k][1] for k in range(n_feat))
            projected.append([px, py])

        # Accumulate gradient for each anchor
        grad = [[0.0, 0.0] for _ in range(n_feat)]
        for i in range(n_pts):
            for j in range(i + 1, n_pts):
                dx = projected[i][0] - projected[j][0]
                dy = projected[i][1] - projected[j][1]
                dist = math.sqrt(dx * dx + dy * dy)
                if dist < 1e-8:
                    continue
                # Force magnitude
                if labels[i] == labels[j]:
                    force = dist          # attractive
                else:
                    force = -1.0 / (dist * dist)  # repulsive
                # Direction
                ux, uy = dx / dist, dy / dist
                # Apply to anchors
                for k in range(n_feat):
                    gx = force * X[i][k] * ux - force * X[j][k] * ux
                    gy = force * X[i][k] * uy - force * X[j][k] * uy
                    grad[k][0] += gx
                    grad[k][1] += gy

        # Update and normalise anchors
        new_anchors = []
        for k in range(n_feat):
            ax = anchors[k][0] - learning_rate * grad[k][0]
            ay = anchors[k][1] - learning_rate * grad[k][1]
            length = math.sqrt(ax * ax + ay * ay)
            if length < 1e-8:
                ax, ay = anchors[k]  # keep if degenerate
            else:
                ax /= length
                ay /= length
            new_anchors.append([ax, ay])
        anchors = new_anchors

    return anchors


class _FreeVizWidget(QWidget):
    """FreeViz projection canvas."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._points: list[tuple[float, float, int]] = []
        self._anchors: list[tuple[str, float, float]] = []  # (name, ax, ay) — unit circle coords
        self._class_labels: list[str] = []
        self.setMinimumHeight(250)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

    def set_projection(
        self,
        points: list[tuple[float, float, int]],
        anchors: list[tuple[str, float, float]],
        class_labels: list[str],
    ) -> None:
        self._points = points
        self._anchors = anchors
        self._class_labels = class_labels
        self.update()

    def paintEvent(self, _event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), QColor("#fffdf9"))

        if not self._anchors:
            painter.setPen(QColor("#8d877d"))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter,
                             "No data.\nLoad a dataset with numeric columns.")
            return

        w, h = self.width(), self.height()
        margin = 56
        cx, cy = w / 2, h / 2
        radius = min(cx, cy) - margin

        # Outer unit circle
        painter.setPen(QPen(QColor("#d0ccc3"), 1))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawEllipse(int(cx - radius), int(cy - radius), int(radius * 2), int(radius * 2))

        # Data points (projected)
        for px, py, ci in self._points:
            sx = int(cx + px * radius)
            sy = int(cy - py * radius)
            color = QColor(_PALETTE[ci % len(_PALETTE)])
            color.setAlpha(160)
            painter.setBrush(color)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawEllipse(sx - 3, sy - 3, 6, 6)

        # Anchors (lines from centre to unit circle)
        for name, ax, ay in self._anchors:
            sx = int(cx + ax * radius)
            sy = int(cy - ay * radius)
            painter.setPen(QPen(QColor("#e07020"), 1.8))
            painter.drawLine(int(cx), int(cy), sx, sy)
            painter.setBrush(QColor("#e07020"))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawEllipse(sx - 5, sy - 5, 10, 10)
            # Label
            lx = int(cx + ax * (radius + 16))
            ly = int(cy - ay * (radius + 16))
            painter.setPen(QColor("#3b2a10"))
            label = name if len(name) <= 10 else name[:9] + "…"
            painter.drawText(lx - 40, ly - 8, 80, 16, Qt.AlignmentFlag.AlignCenter, label)

        # Legend
        if self._class_labels:
            base_y = h - 18 * len(self._class_labels) - 4
            for i, lbl in enumerate(self._class_labels[:8]):
                color = _PALETTE[i % len(_PALETTE)]
                painter.setBrush(color)
                painter.setPen(Qt.PenStyle.NoPen)
                painter.drawEllipse(8, base_y + i * 18, 10, 10)
                painter.setPen(QColor("#534b40"))
                painter.drawText(22, base_y + i * 18, 120, 12, Qt.AlignmentFlag.AlignVCenter, lbl)

        painter.end()


class FreeVizScreen(QWidget, WorkflowNodeScreenSupport):
    """
    FreeViz – Force-directed linear projection (Demsar et al., 2005).

    Same-class points attract each other; different-class points repel.
    Forces act on attribute anchors (unit vectors on a circle).
    Gradient descent finds equilibrium.
    """

    MAX_POINTS = 300  # keep optimization fast

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._init_workflow_node_support()
        self._dataset: DatasetHandle | None = None
        self._anchors_xy: list[list[float]] = []   # optimized anchors
        self._feature_names: list[str] = []
        self._X: list[list[float]] = []
        self._labels: list[int] = []
        self._class_labels: list[str] = []
        self._n_iters_done = 0
        self._optimize_timer = QTimer(self)
        self._optimize_timer.setInterval(50)
        self._optimize_timer.timeout.connect(self._run_one_batch)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        header = QLabel("FreeViz")
        header.setProperty("sectionTitle", True)
        layout.addWidget(header)

        desc = QLabel(
            "Force-directed projection (Demsar et al., 2005). "
            "Same-class points attract; different-class points repel. "
            "Anchors (orange) converge to equilibrium via gradient descent."
        )
        desc.setWordWrap(True)
        desc.setProperty("muted", True)
        layout.addWidget(desc)

        controls_group = QGroupBox("Settings")
        ctrl = QHBoxLayout(controls_group)
        ctrl.addWidget(QLabel("Color by:"))
        self._class_combo = QComboBox()
        self._class_combo.currentTextChanged.connect(self._reset_and_refresh)
        ctrl.addWidget(self._class_combo, 1)
        ctrl.addWidget(QLabel("Iterations:"))
        self._iter_spin = QSpinBox()
        self._iter_spin.setRange(0, 500)
        self._iter_spin.setValue(50)
        ctrl.addWidget(self._iter_spin)
        ctrl.addWidget(QLabel("LR:"))
        self._lr_slider = QSlider(Qt.Orientation.Horizontal)
        self._lr_slider.setRange(1, 50)
        self._lr_slider.setValue(10)
        self._lr_slider.setMaximumWidth(80)
        self._lr_slider.setToolTip("Learning rate × 0.001")
        ctrl.addWidget(self._lr_slider)
        self._run_btn = QPushButton("▶ Run")
        self._run_btn.clicked.connect(self._toggle_optimization)
        ctrl.addWidget(self._run_btn)
        self._reset_btn = QPushButton("↺ Reset")
        self._reset_btn.clicked.connect(self._reset_and_refresh)
        ctrl.addWidget(self._reset_btn)
        layout.addWidget(controls_group)

        chart_group = QGroupBox("Projection")
        chart_layout = QVBoxLayout(chart_group)
        self._chart = _FreeVizWidget()
        chart_layout.addWidget(self._chart)
        layout.addWidget(chart_group, 1)

        self._status_label = QLabel("Load a dataset with numeric columns.")
        self._status_label.setProperty("muted", True)
        layout.addWidget(self._status_label)

    # ── WorkflowNodeScreenSupport ──────────────────────────────────────

    def set_input_payload(self, payload) -> None:
        self._optimize_timer.stop()
        dataset = payload.dataset if payload is not None else None
        self._dataset = dataset
        self._class_combo.blockSignals(True)
        self._class_combo.clear()
        if dataset is not None:
            self._class_combo.addItem("(none)")
            for col in dataset.domain.columns:
                self._class_combo.addItem(col.name)
            for col in dataset.domain.target_columns:
                idx = self._class_combo.findText(col.name)
                if idx >= 0:
                    self._class_combo.setCurrentIndex(idx)
                    break
        self._class_combo.blockSignals(False)
        self._reset_and_refresh()

    def documentation_url(self) -> str:
        return "https://orangedatamining.com/widget-catalog/visualize/freeviz/"

    # ── Optimization control ───────────────────────────────────────────

    def _toggle_optimization(self) -> None:
        if self._optimize_timer.isActive():
            self._optimize_timer.stop()
            self._run_btn.setText("▶ Run")
        else:
            if not self._X:
                return
            self._optimize_timer.start()
            self._run_btn.setText("⏸ Pause")

    def _run_one_batch(self) -> None:
        if not self._X:
            self._optimize_timer.stop()
            return
        lr = self._lr_slider.value() * 0.001
        self._anchors_xy = _freeviz_optimize(
            self._X, self._labels, self._anchors_xy,
            n_iter=5, learning_rate=lr
        )
        self._n_iters_done += 5
        n_total = self._iter_spin.value()
        if self._n_iters_done >= n_total:
            self._optimize_timer.stop()
            self._run_btn.setText("▶ Run")
        self._draw_current()

    def _reset_and_refresh(self) -> None:
        self._optimize_timer.stop()
        self._run_btn.setText("▶ Run")
        self._n_iters_done = 0
        self._build_data()
        self._draw_current()

    def _build_data(self) -> None:
        """Prepare feature matrix, class labels, and initial circular anchors."""
        self._X = []
        self._labels = []
        self._class_labels = []
        self._anchors_xy = []
        self._feature_names = []

        if self._dataset is None:
            return

        df = self._dataset.dataframe
        numeric_cols = [col.name for col in self._dataset.domain.columns if df[col.name].dtype.is_numeric()]
        if len(numeric_cols) < 2:
            return

        class_col = self._class_combo.currentText()
        if class_col == "(none)":
            class_col = None

        n = len(df)
        max_pts = self.MAX_POINTS
        indices = list(range(n))
        if n > max_pts:
            random.seed(42)
            indices = random.sample(indices, max_pts)
            indices.sort()

        # Normalise features to [0, 1]
        col_mins: dict[str, float] = {}
        col_maxs: dict[str, float] = {}
        for col in numeric_cols:
            vals = [float(df[col][i]) for i in indices if df[col][i] is not None]
            mn = min(vals) if vals else 0.0
            mx = max(vals) if vals else 1.0
            col_mins[col] = mn
            col_maxs[col] = mx if mx != mn else mn + 1.0

        # Class labels
        class_map: dict[str, int] = {}
        raw_classes: list[str] = []
        if class_col and class_col in df.columns:
            raw_classes = [str(df[class_col][i]) if df[class_col][i] is not None else "(missing)" for i in indices]
            unique = list(dict.fromkeys(raw_classes))[:8]
            self._class_labels = unique
            class_map = {v: i for i, v in enumerate(unique)}

        for row_i, i in enumerate(indices):
            row = []
            for col in numeric_cols:
                raw = df[col][i]
                val = float(raw) if raw is not None else col_mins[col]
                row.append((val - col_mins[col]) / (col_maxs[col] - col_mins[col]))
            self._X.append(row)
            ci = class_map.get(raw_classes[row_i], 0) if raw_classes else 0
            self._labels.append(ci)

        self._feature_names = numeric_cols
        n_feat = len(numeric_cols)
        # Initial anchors: equally spaced on unit circle
        self._anchors_xy = [
            [math.cos(2 * math.pi * k / n_feat), math.sin(2 * math.pi * k / n_feat)]
            for k in range(n_feat)
        ]

    def _draw_current(self) -> None:
        if not self._X or not self._anchors_xy:
            self._chart.set_projection([], [], [])
            self._status_label.setText("Load a dataset with at least 2 numeric columns.")
            return

        n_feat = len(self._anchors_xy)
        # Project each point: p_i = normalised weighted sum of anchors
        points: list[tuple[float, float, int]] = []
        for row_i, row in enumerate(self._X):
            px = sum(row[k] * self._anchors_xy[k][0] for k in range(n_feat))
            py = sum(row[k] * self._anchors_xy[k][1] for k in range(n_feat))
            # Normalise to ~[-1, 1]
            total_w = sum(row) or 1.0
            px /= total_w
            py /= total_w
            points.append((px, py, self._labels[row_i]))

        # Clamp points to unit circle
        max_r = max(math.sqrt(px ** 2 + py ** 2) for px, py, _ in points) or 1.0
        points = [(px / max_r, py / max_r, ci) for px, py, ci in points]

        anchors = [(self._feature_names[k], self._anchors_xy[k][0], self._anchors_xy[k][1])
                   for k in range(n_feat)]

        self._chart.set_projection(points, anchors, self._class_labels)
        class_col = self._class_combo.currentText()
        self._status_label.setText(
            f"{len(points)} points · {n_feat} features · iter {self._n_iters_done}"
            + (f" · colored by '{class_col}'" if class_col and class_col != "(none)" else "")
        )
