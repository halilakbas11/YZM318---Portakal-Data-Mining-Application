from __future__ import annotations

from itertools import chain

from PySide6.QtCore import Qt, QRectF, QPointF, Signal
from PySide6.QtGui import QPainter, QColor, QPen, QRadialGradient, QFont
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QSpinBox,
    QDoubleSpinBox,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from portakal_app.data.services.sklearn_learner_service import SklearnLearnerService
from portakal_app.ui import i18n
from portakal_app.ui.screens.model_base import ModelScreenBase


_ACTIVATIONS = ["identity", "logistic", "tanh", "relu"]
_ACTIVATION_LABELS = ["Identity", "Logistic", "tanh", "ReLU"]
_SOLVERS = ["lbfgs", "sgd", "adam"]
_SOLVER_LABELS = ["L-BFGS-B", "SGD", "Adam"]

_ALPHAS = list(chain(
    [0],
    [x / 10000 for x in range(1, 10)],
    [x / 1000 for x in range(1, 10)],
    [x / 100 for x in range(1, 10)],
    [x / 10 for x in range(1, 10)],
    range(1, 10),
    range(10, 100, 5),
    range(100, 200, 10),
    range(100, 1001, 50),
))

_MAX_DISPLAY_NODES = 8   # max dots drawn per column in the visualizer
_INPUT_DISPLAY_NODES = 6  # fixed representative input nodes


# ── Hidden Layers Editor ──────────────────────────────────────────────────────

class HiddenLayersEditor(QWidget):
    """Dynamic list of hidden layer sizes: one spinbox row per layer."""

    layers_changed = Signal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(4)

        self._rows_container = QWidget()
        self._rows_layout = QVBoxLayout(self._rows_container)
        self._rows_layout.setContentsMargins(0, 0, 0, 0)
        self._rows_layout.setSpacing(2)
        outer.addWidget(self._rows_container)

        add_row = QHBoxLayout()
        self._add_btn = QPushButton("＋  Add Hidden Layer")
        self._add_btn.clicked.connect(lambda: self._add_row(100))
        add_row.addWidget(self._add_btn)
        add_row.addStretch(1)
        outer.addLayout(add_row)

        self._spins: list[QSpinBox] = []
        self._row_widgets: list[QWidget] = []

    def _add_row(self, neurons: int = 100) -> None:
        idx = len(self._spins) + 1

        row = QWidget()
        rl = QHBoxLayout(row)
        rl.setContentsMargins(0, 0, 0, 0)
        rl.setSpacing(6)

        lbl = QLabel(f"H{idx}")
        lbl.setFixedWidth(26)
        lbl.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        lbl.setStyleSheet("color: #6366f1; font-weight: bold; background: transparent;")

        spin = QSpinBox()
        spin.setRange(1, 10000)
        spin.setValue(neurons)
        spin.setAlignment(Qt.AlignmentFlag.AlignRight)
        spin.valueChanged.connect(self.layers_changed)

        neurons_lbl = QLabel("neurons")
        neurons_lbl.setStyleSheet("color: #64748b; background: transparent;")

        del_btn = QToolButton()
        del_btn.setText("✕")
        del_btn.setFixedWidth(24)
        del_btn.setToolTip("Remove this layer")
        del_btn.clicked.connect(lambda _checked=False, r=row: self._remove_row(r))

        rl.addWidget(lbl)
        rl.addWidget(spin, 1)
        rl.addWidget(neurons_lbl)
        rl.addWidget(del_btn)

        self._rows_layout.addWidget(row)
        self._spins.append(spin)
        self._row_widgets.append(row)
        self.layers_changed.emit()

    def _remove_row(self, row: QWidget) -> None:
        if len(self._row_widgets) <= 1:
            return  # always keep at least one hidden layer
        idx = self._row_widgets.index(row)
        self._row_widgets.pop(idx)
        self._spins.pop(idx)
        self._rows_layout.removeWidget(row)
        row.deleteLater()
        self._relabel()
        self.layers_changed.emit()

    def _relabel(self) -> None:
        for i, row in enumerate(self._row_widgets):
            lbl = row.layout().itemAt(0).widget()
            if isinstance(lbl, QLabel):
                lbl.setText(f"H{i + 1}")

    def get_layers(self) -> tuple[int, ...]:
        return tuple(spin.value() for spin in self._spins)

    def set_layers(self, layers: tuple[int, ...]) -> None:
        # Remove all existing rows silently (no signal per deletion)
        for row in list(self._row_widgets):
            self._rows_layout.removeWidget(row)
            row.deleteLater()
        self._spins.clear()
        self._row_widgets.clear()

        for n in (layers if layers else (100,)):
            self._add_row(n)


# ── Visualizer ────────────────────────────────────────────────────────────────

class MLPVisualizer(QWidget):
    """Draws the MLP architecture: one column per layer, nodes as dots."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setMinimumHeight(200)
        self._layers: tuple[int, ...] = (100,)
        self._weights: list | None = None
        self.setStyleSheet(
            "background-color: #1a1c22; border-radius: 10px; border: 1px solid #2d2f3b;"
        )

    def set_layers(self, layers: tuple[int, ...]) -> None:
        if self._layers != layers:
            self._layers = layers
            self._weights = None
            self.update()

    def set_weights(self, weights: list) -> None:
        self._weights = weights
        self.update()

    def paintEvent(self, event) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        W, H = self.width(), self.height()

        # Full layer list: input + hidden... + output
        hidden = self._layers
        all_real = (_INPUT_DISPLAY_NODES,) + hidden + (1,)
        n_cols = len(all_real)

        # Display node count per column (capped for rendering)
        viz = [min(_MAX_DISPLAY_NODES, c) for c in all_real]

        # Adaptive node radius: shrink when many columns
        node_r = max(4, min(8, 18 - n_cols))

        # Vertical: reserve bottom strip for labels
        label_h = 32
        top_pad = 14
        draw_h = H - label_h - top_pad

        # Horizontal: evenly spread columns with padding
        h_pad = max(node_r + 6, int(W * 0.07))
        usable_w = W - 2 * h_pad
        col_xs = [
            h_pad + i * usable_w / (n_cols - 1) if n_cols > 1 else W / 2
            for i in range(n_cols)
        ]

        # Pre-compute node positions
        nodes: list[list[QPointF]] = []
        for col_i, (cnt, cx) in enumerate(zip(viz, col_xs)):
            v_gap = min(node_r * 2.8, draw_h / max(cnt, 1))
            total_v = v_gap * (cnt - 1)
            start_y = top_pad + (draw_h - total_v) / 2
            nodes.append([QPointF(cx, start_y + j * v_gap) for j in range(cnt)])

        # Draw connections
        for col_i in range(n_cols - 1):
            real_rows = all_real[col_i]
            real_cols_next = all_real[col_i + 1]
            layer_w = self._weights[col_i] if (self._weights and col_i < len(self._weights)) else None

            for src_j, p1 in enumerate(nodes[col_i]):
                for dst_k, p2 in enumerate(nodes[col_i + 1]):
                    if layer_w is not None:
                        rj = int(src_j * layer_w.shape[0] / max(len(nodes[col_i]), 1))
                        rk = int(dst_k * layer_w.shape[1] / max(len(nodes[col_i + 1]), 1))
                        rj = min(rj, layer_w.shape[0] - 1)
                        rk = min(rk, layer_w.shape[1] - 1)
                        val = abs(float(layer_w[rj, rk]))
                        width = min(3.5, 0.5 + val * 2.0)
                        opacity = min(190, int(40 + val * 100))
                        negative = float(layer_w[rj, rk]) < 0
                    else:
                        width = 0.7
                        opacity = 25
                        negative = False

                    color = QColor(255, 100, 100, opacity) if negative else QColor(100, 120, 255, opacity)
                    pen = QPen(color)
                    pen.setWidthF(width)
                    painter.setPen(pen)
                    painter.drawLine(p1, p2)

        # Draw nodes
        for col_i, layer_nodes in enumerate(nodes):
            is_hidden = 0 < col_i < n_cols - 1
            base = QColor("#6366f1") if is_hidden else QColor("#94a3b8")
            for pt in layer_nodes:
                grad = QRadialGradient(pt, node_r)
                grad.setColorAt(0, base.lighter(145))
                grad.setColorAt(0.65, base)
                grad.setColorAt(1, base.darker(130))
                painter.setBrush(grad)
                painter.setPen(QPen(base.lighter(160), 0.5))
                painter.drawEllipse(pt, node_r, node_r)

        # Draw "…" when column is capped
        painter.setFont(QFont("Arial", 7))
        for col_i, (real_cnt, viz_cnt, cx) in enumerate(zip(all_real, viz, col_xs)):
            if real_cnt > viz_cnt:
                last_pt = nodes[col_i][-1]
                painter.setPen(QColor("#64748b"))
                painter.drawText(
                    QRectF(cx - 12, last_pt.y() + node_r + 1, 24, 12),
                    Qt.AlignmentFlag.AlignCenter, "…",
                )

        # Layer labels + neuron count at bottom
        painter.setFont(QFont("Arial", 7, QFont.Weight.Bold))
        labels = (
            ["Input"]
            + [f"H{i + 1}" for i in range(len(hidden))]
            + ["Output"]
        )
        for col_i, (label, real_cnt, cx) in enumerate(zip(labels, all_real, col_xs)):
            lx = cx - 24
            # column name
            is_hidden = 0 < col_i < n_cols - 1
            painter.setPen(QColor("#6366f1") if is_hidden else QColor("#94a3b8"))
            painter.drawText(QRectF(lx, H - label_h + 2, 48, 13), Qt.AlignmentFlag.AlignCenter, label)
            # neuron count
            painter.setPen(QColor("#475569"))
            painter.setFont(QFont("Arial", 6))
            painter.drawText(QRectF(lx, H - label_h + 15, 48, 12), Qt.AlignmentFlag.AlignCenter, str(real_cnt))
            painter.setFont(QFont("Arial", 7, QFont.Weight.Bold))


# ── Main Screen ───────────────────────────────────────────────────────────────

class NeuralNetworkScreen(ModelScreenBase):
    """Multi-layer perceptron with backpropagation."""

    _OUTPUT_PORT_LABEL = "Model"

    def __init__(self, parent=None) -> None:
        self._svc = SklearnLearnerService()
        super().__init__(parent)

    def _add_main_layout(self, layout: QVBoxLayout) -> None:
        # ── Architecture ──────────────────────────────────────────────
        arch_box = QGroupBox(i18n.t("Network Architecture"))
        arch_layout = QVBoxLayout(arch_box)

        # Activation row
        act_form = QFormLayout()
        act_form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)
        self._activation_combo = QComboBox()
        self._activation_combo.addItems(_ACTIVATION_LABELS)
        self._activation_combo.setCurrentIndex(3)  # ReLU default
        self._activation_combo.currentIndexChanged.connect(self._settings_changed)
        act_form.addRow(i18n.t("Activation:"), self._activation_combo)
        arch_layout.addLayout(act_form)

        # Hidden layers editor
        layers_lbl = QLabel(i18n.t("Hidden layers:"))
        layers_lbl.setStyleSheet("background: transparent; font-weight: bold;")
        arch_layout.addWidget(layers_lbl)

        self._layers_editor = HiddenLayersEditor()
        self._layers_editor.layers_changed.connect(self._on_layers_changed)
        arch_layout.addWidget(self._layers_editor)

        # Visualizer
        self._visualizer = MLPVisualizer()
        arch_layout.addWidget(self._visualizer)

        layout.addWidget(arch_box)

        # ── Solver ────────────────────────────────────────────────────
        solver_box = QGroupBox(i18n.t("Solver Options"))
        form_solver = QFormLayout(solver_box)
        form_solver.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)

        self._solver_combo = QComboBox()
        self._solver_combo.addItems(_SOLVER_LABELS)
        self._solver_combo.setCurrentIndex(2)  # Adam default
        self._solver_combo.currentIndexChanged.connect(self._settings_changed)
        form_solver.addRow(i18n.t("Solver:"), self._solver_combo)

        self._alpha_label = QLabel()
        self._alpha_slider = QSlider(Qt.Orientation.Horizontal)
        self._alpha_slider.setRange(0, len(_ALPHAS) - 1)
        self._alpha_slider.setValue(1)
        self._alpha_slider.sliderReleased.connect(self._settings_changed)
        self._alpha_slider.valueChanged.connect(self._update_alpha_label)
        self._update_alpha_label()
        form_solver.addRow(self._alpha_label, self._alpha_slider)

        self._max_iter_spin = QSpinBox()
        self._max_iter_spin.setRange(1, 1_000_000)
        self._max_iter_spin.setValue(200)
        self._max_iter_spin.setAlignment(Qt.AlignmentFlag.AlignRight)
        self._max_iter_spin.valueChanged.connect(self._settings_changed)
        form_solver.addRow(i18n.t("Max iterations:"), self._max_iter_spin)

        self._lr_spin = QDoubleSpinBox()
        self._lr_spin.setRange(0.0001, 1.0)
        self._lr_spin.setDecimals(4)
        self._lr_spin.setSingleStep(0.001)
        self._lr_spin.setValue(0.001)
        self._lr_spin.setAlignment(Qt.AlignmentFlag.AlignRight)
        self._lr_spin.valueChanged.connect(self._settings_changed)
        form_solver.addRow(i18n.t("Initial learning rate:"), self._lr_spin)

        self._early_stop_cb = QCheckBox(i18n.t("Early stopping"))
        self._early_stop_cb.stateChanged.connect(self._settings_changed)
        form_solver.addRow(self._early_stop_cb)

        self._val_fract_spin = QDoubleSpinBox()
        self._val_fract_spin.setRange(0.01, 0.99)
        self._val_fract_spin.setValue(0.1)
        self._val_fract_spin.setSingleStep(0.05)
        self._val_fract_spin.setAlignment(Qt.AlignmentFlag.AlignRight)
        self._val_fract_spin.valueChanged.connect(self._settings_changed)
        form_solver.addRow(i18n.t("Validation fraction:"), self._val_fract_spin)

        self._replicable_cb = QCheckBox(i18n.t("Replicable training"))
        self._replicable_cb.setChecked(True)
        self._replicable_cb.stateChanged.connect(self._settings_changed)
        form_solver.addRow(self._replicable_cb)

        layout.addWidget(solver_box)

        # Seed the visualizer with the initial single layer
        self._on_layers_changed()

    # ── Internals ─────────────────────────────────────────────────────────────

    def _update_alpha_label(self) -> None:
        alpha = _ALPHAS[self._alpha_slider.value()]
        self._alpha_label.setText(i18n.tf("Regularization, α={alpha}", alpha=alpha))

    def _on_layers_changed(self) -> None:
        self._visualizer.set_layers(self._layers_editor.get_layers())
        self._settings_changed()

    def _train(self):
        from sklearn.neural_network import MLPClassifier, MLPRegressor

        ds = self._dataset
        target_cols = ds.domain.target_columns
        if not target_cols:
            raise ValueError("No target column.")
        is_clf = target_cols[0].logical_type in {"categorical", "boolean"}

        layers = self._layers_editor.get_layers()
        activation = _ACTIVATIONS[self._activation_combo.currentIndex()]
        solver = _SOLVERS[self._solver_combo.currentIndex()]
        alpha = _ALPHAS[self._alpha_slider.value()]
        max_iter = self._max_iter_spin.value()
        lr = self._lr_spin.value()
        early_stop = self._early_stop_cb.isChecked()
        val_fract = self._val_fract_spin.value()
        random_state = 1 if self._replicable_cb.isChecked() else None

        kw = dict(
            hidden_layer_sizes=layers, activation=activation, solver=solver,
            alpha=alpha, max_iter=max_iter, learning_rate_init=lr,
            early_stopping=early_stop, validation_fraction=val_fract,
            random_state=random_state,
        )
        est = MLPClassifier(**kw) if is_clf else MLPRegressor(**kw)
        params = {
            "hidden_layers": list(layers), "activation": activation,
            "solver": solver, "alpha": alpha, "max_iter": max_iter,
            "learning_rate_init": lr, "early_stopping": early_stop,
            "validation_fraction": val_fract,
        }

        result = self._svc.fit(est, ds, "Neural Network", "neural_network", params)

        if result.trained_model is not None and hasattr(result.trained_model, "coefs_"):
            self._visualizer.set_weights(result.trained_model.coefs_)

        return result

    def serialize_node_state(self) -> dict:
        return {
            **super().serialize_node_state(),
            "layers": list(self._layers_editor.get_layers()),
            "activation": self._activation_combo.currentIndex(),
            "solver": self._solver_combo.currentIndex(),
            "alpha_index": self._alpha_slider.value(),
            "max_iter": self._max_iter_spin.value(),
            "lr": self._lr_spin.value(),
            "early_stopping": self._early_stop_cb.isChecked(),
            "val_fract": self._val_fract_spin.value(),
            "replicable": self._replicable_cb.isChecked(),
        }

    def restore_node_state(self, payload: dict) -> None:
        super().restore_node_state(payload)

        raw = payload.get("layers", [100])
        if isinstance(raw, str):
            # backward compat: old format was "100, 50,"
            import re
            nums = [int(x) for x in re.findall(r"\d+", raw)]
            raw = nums if nums else [100]
        self._layers_editor.set_layers(tuple(int(n) for n in raw))

        self._activation_combo.setCurrentIndex(int(payload.get("activation", 3)))
        self._solver_combo.setCurrentIndex(int(payload.get("solver", 2)))
        self._alpha_slider.setValue(int(payload.get("alpha_index", 1)))
        self._max_iter_spin.setValue(int(payload.get("max_iter", 200)))
        self._lr_spin.setValue(float(payload.get("lr", 0.001)))
        self._early_stop_cb.setChecked(bool(payload.get("early_stopping", False)))
        self._val_fract_spin.setValue(float(payload.get("val_fract", 0.1)))
        self._replicable_cb.setChecked(bool(payload.get("replicable", True)))
        self._update_alpha_label()
        self._on_layers_changed()
