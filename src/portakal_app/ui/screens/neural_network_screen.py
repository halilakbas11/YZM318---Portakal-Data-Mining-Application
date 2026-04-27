from __future__ import annotations

import re
from itertools import chain

from PySide6.QtCore import Qt, QRectF, QPointF
from PySide6.QtGui import QPainter, QColor, QPen, QBrush, QRadialGradient, QFont
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QSlider,
    QSpinBox,
    QDoubleSpinBox,
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


class MLPVisualizer(QWidget):
    """Interactive visualizer for Neural Network architecture with weight support."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setMinimumHeight(220)
        self._layers: tuple[int, ...] = (100,)
        self._weights: list[list[list[float]]] | None = None
        self.setStyleSheet("background-color: #1a1c22; border-radius: 12px; border: 1px solid #333;")

    def set_layers(self, layers: tuple[int, ...]) -> None:
        if self._layers != layers:
            self._layers = layers
            self._weights = None  # Reset weights when architecture changes
            self.update()

    def set_weights(self, weights: list) -> None:
        """Sets the weights to be visualized. Should be list of arrays from sklearn."""
        self._weights = weights
        self.update()

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        w, h = self.width(), self.height()
        
        # We simulate Input and Output layers for visualization
        hidden_counts = self._layers
        viz_layers = [min(12, x) for x in (8,) + hidden_counts + (1,)]
        n_layers = len(viz_layers)
        
        layer_spacing = w / (n_layers + 1)
        
        # Pre-calculate positions
        nodes = []
        for i, count in enumerate(viz_layers):
            layer_nodes = []
            x = layer_spacing * (i + 1)
            node_spacing = min(25, h / (count + 1))
            total_node_h = node_spacing * (count - 1)
            start_y = (h - total_node_h) / 2
            
            for j in range(count):
                y = start_y + j * node_spacing
                layer_nodes.append(QPointF(x, y))
            nodes.append(layer_nodes)

        # Draw connections (weights)
        import random
        for i in range(len(nodes) - 1):
            layer_weights = None
            if self._weights and i < len(self._weights):
                layer_weights = self._weights[i]

            for j, p1 in enumerate(nodes[i]):
                for k, p2 in enumerate(nodes[i+1]):
                    # Determine weight value for thickness
                    if layer_weights is not None:
                        # Map indices if the real layer is larger than viz layer
                        real_j = int(j * (layer_weights.shape[0] / len(nodes[i])))
                        real_k = int(k * (layer_weights.shape[1] / len(nodes[i+1])))
                        val = abs(float(layer_weights[real_j, real_k]))
                        # Normalize weight for visibility (1.0 to 4.0 range)
                        width = min(4.0, 0.5 + val * 2.0)
                        opacity = min(200, int(40 + val * 100))
                    else:
                        # If no real weights, use a subtle default or "pulsing" effect
                        width = 0.8
                        opacity = 30
                    
                    color = QColor(100, 120, 255, opacity)
                    if layer_weights is not None and float(layer_weights[real_j, real_k]) < 0:
                        color = QColor(255, 100, 100, opacity) # Red for negative weights
                    
                    pen = QPen(color)
                    pen.setWidthF(width)
                    painter.setPen(pen)
                    painter.drawLine(p1, p2)

        # Draw neurons (nodes)
        node_radius = 7
        for i, layer in enumerate(nodes):
            is_hidden = 0 < i < len(nodes) - 1
            color_base = QColor("#6366f1") if is_hidden else QColor("#94a3b8")
            
            for p in layer:
                grad = QRadialGradient(p, node_radius)
                grad.setColorAt(0, color_base.lighter(140))
                grad.setColorAt(0.7, color_base)
                grad.setColorAt(1, color_base.darker(120))
                
                painter.setBrush(grad)
                painter.setPen(QPen(color_base.lighter(150), 0.5))
                painter.drawEllipse(p, node_radius, node_radius)

        # Labels
        painter.setPen(QColor("#64748b"))
        painter.setFont(QFont("Inter", 8, QFont.Weight.Bold))
        labels = ["Input"] + [f"H{i+1}" for i in range(len(self._layers))] + ["Output"]
        for i, nodes_list in enumerate(nodes):
            if nodes_list:
                center_p = nodes_list[0]
                painter.drawText(QRectF(center_p.x() - 30, h - 25, 60, 20), 
                                 Qt.AlignmentFlag.AlignCenter, labels[i])


class NeuralNetworkScreen(ModelScreenBase):
    """Multi-layer perceptron with backpropagation — Orange Neural Network equivalent."""

    _OUTPUT_PORT_LABEL = "Model"

    def __init__(self, parent=None) -> None:
        self._svc = SklearnLearnerService()
        super().__init__(parent)

    def _add_main_layout(self, layout: QVBoxLayout) -> None:
        # Architecture Group
        arch_box = QGroupBox(i18n.t("Network Architecture"))
        arch_layout = QVBoxLayout(arch_box)
        
        form_arch = QFormLayout()
        self._layers_edit = QLineEdit("100,")
        self._layers_edit.setPlaceholderText("e.g. 100, 50,")
        self._layers_edit.textChanged.connect(self._on_layers_changed)
        form_arch.addRow(i18n.t("Neurons in hidden layers:"), self._layers_edit)
        
        self._activation_combo = QComboBox()
        self._activation_combo.addItems(_ACTIVATION_LABELS)
        self._activation_combo.setCurrentIndex(3)
        self._activation_combo.currentIndexChanged.connect(self._settings_changed)
        form_arch.addRow(i18n.t("Activation:"), self._activation_combo)
        
        arch_layout.addLayout(form_arch)
        
        # Visualizer
        self._visualizer = MLPVisualizer()
        arch_layout.addWidget(self._visualizer)
        
        layout.addWidget(arch_box)

        # Solver Settings Group
        solver_box = QGroupBox(i18n.t("Solver Options"))
        form_solver = QFormLayout(solver_box)
        form_solver.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)

        self._solver_combo = QComboBox()
        self._solver_combo.addItems(_SOLVER_LABELS)
        self._solver_combo.setCurrentIndex(2)
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

        # Advanced Params
        self._max_iter_spin = QSpinBox()
        self._max_iter_spin.setRange(1, 1000000)
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

    def _update_alpha_label(self) -> None:
        alpha = _ALPHAS[self._alpha_slider.value()]
        self._alpha_label.setText(i18n.tf("Regularization, α={alpha}", alpha=alpha))

    def _on_layers_changed(self) -> None:
        layers = self._get_hidden_layers()
        self._visualizer.set_layers(layers)
        self._settings_changed()

    def _get_hidden_layers(self) -> tuple[int, ...]:
        nums = tuple(int(x) for x in re.findall(r'\d+', self._layers_edit.text()))
        return nums if nums else (100,)

    def _train(self):
        from sklearn.neural_network import MLPClassifier, MLPRegressor

        ds = self._dataset
        target_cols = ds.domain.target_columns
        if not target_cols:
            raise ValueError("No target column.")
        is_clf = target_cols[0].logical_type in {"categorical", "boolean"}

        layers = self._get_hidden_layers()
        activation = _ACTIVATIONS[self._activation_combo.currentIndex()]
        solver = _SOLVERS[self._solver_combo.currentIndex()]
        alpha = _ALPHAS[self._alpha_slider.value()]
        max_iter = self._max_iter_spin.value()
        lr = self._lr_spin.value()
        early_stop = self._early_stop_cb.isChecked()
        val_fract = self._val_fract_spin.value()
        random_state = 1 if self._replicable_cb.isChecked() else None

        kw = dict(hidden_layer_sizes=layers, activation=activation, solver=solver,
                  alpha=alpha, max_iter=max_iter, learning_rate_init=lr,
                  early_stopping=early_stop, validation_fraction=val_fract,
                  random_state=random_state)
        est = MLPClassifier(**kw) if is_clf else MLPRegressor(**kw)
        params = {"hidden_layers": layers, "activation": activation, "solver": solver,
                  "alpha": alpha, "max_iter": max_iter, "learning_rate_init": lr,
                  "early_stopping": early_stop, "validation_fraction": val_fract}
        
        result = self._svc.fit(est, ds, "Neural Network", "neural_network", params)
        
        # After training, extract weights for visualization
        if result.trained_model is not None and hasattr(result.trained_model, "coefs_"):
            self._visualizer.set_weights(result.trained_model.coefs_)
            
        return result

    def serialize_node_state(self) -> dict:
        return {
            **super().serialize_node_state(),
            "layers": self._layers_edit.text(),
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
        self._layers_edit.setText(str(payload.get("layers", "100,")))
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
