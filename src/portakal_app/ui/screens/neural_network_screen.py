from __future__ import annotations

import re
from itertools import chain

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFormLayout,
    QGroupBox,
    QLabel,
    QLineEdit,
    QSlider,
    QSpinBox,
    QVBoxLayout,
)

from portakal_app.data.services.sklearn_learner_service import SklearnLearnerService
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


class NeuralNetworkScreen(ModelScreenBase):
    """Multi-layer perceptron with backpropagation — Orange Neural Network equivalent."""

    _OUTPUT_PORT_LABEL = "Model"

    def __init__(self, parent=None) -> None:
        self._svc = SklearnLearnerService()
        super().__init__(parent)

    def _add_main_layout(self, layout: QVBoxLayout) -> None:
        box = QGroupBox("Neural Network")
        form = QFormLayout(box)
        form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)

        self._layers_edit = QLineEdit("100,")
        self._layers_edit.setPlaceholderText("e.g. 100, 50,")
        self._layers_edit.textChanged.connect(self._settings_changed)
        form.addRow("Neurons in hidden layers:", self._layers_edit)

        self._activation_combo = QComboBox()
        self._activation_combo.addItems(_ACTIVATION_LABELS)
        self._activation_combo.setCurrentIndex(3)
        self._activation_combo.currentIndexChanged.connect(self._settings_changed)
        form.addRow("Activation:", self._activation_combo)

        self._solver_combo = QComboBox()
        self._solver_combo.addItems(_SOLVER_LABELS)
        self._solver_combo.setCurrentIndex(2)
        self._solver_combo.currentIndexChanged.connect(self._settings_changed)
        form.addRow("Solver:", self._solver_combo)

        self._alpha_label = QLabel()
        self._alpha_slider = QSlider(Qt.Orientation.Horizontal)
        self._alpha_slider.setRange(0, len(_ALPHAS) - 1)
        self._alpha_slider.setValue(1)
        self._alpha_slider.sliderReleased.connect(self._settings_changed)
        self._alpha_slider.valueChanged.connect(self._update_alpha_label)
        self._update_alpha_label()
        form.addRow(self._alpha_label, self._alpha_slider)

        self._max_iter_spin = QSpinBox()
        self._max_iter_spin.setRange(10, 1000000)
        self._max_iter_spin.setSingleStep(10)
        self._max_iter_spin.setValue(200)
        self._max_iter_spin.setAlignment(Qt.AlignmentFlag.AlignRight)
        self._max_iter_spin.valueChanged.connect(self._settings_changed)
        form.addRow("Max iterations:", self._max_iter_spin)

        self._replicable_cb = QCheckBox("Replicable training")
        self._replicable_cb.setChecked(True)
        self._replicable_cb.stateChanged.connect(self._settings_changed)
        form.addRow(self._replicable_cb)

        layout.addWidget(box)

    def _update_alpha_label(self) -> None:
        alpha = _ALPHAS[self._alpha_slider.value()]
        self._alpha_label.setText(f"Regularization, α={alpha}:")

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
        random_state = 1 if self._replicable_cb.isChecked() else None

        kw = dict(hidden_layer_sizes=layers, activation=activation, solver=solver,
                  alpha=alpha, max_iter=max_iter, random_state=random_state)
        est = MLPClassifier(**kw) if is_clf else MLPRegressor(**kw)
        params = {"hidden_layers": layers, "activation": activation, "solver": solver,
                  "alpha": alpha, "max_iter": max_iter}
        return self._svc.fit(est, ds, "Neural Network", "neural_network", params)

    def serialize_node_state(self) -> dict:
        return {
            **super().serialize_node_state(),
            "layers": self._layers_edit.text(),
            "activation": self._activation_combo.currentIndex(),
            "solver": self._solver_combo.currentIndex(),
            "alpha_index": self._alpha_slider.value(),
            "max_iter": self._max_iter_spin.value(),
            "replicable": self._replicable_cb.isChecked(),
        }

    def restore_node_state(self, payload: dict) -> None:
        super().restore_node_state(payload)
        self._layers_edit.setText(str(payload.get("layers", "100,")))
        self._activation_combo.setCurrentIndex(int(payload.get("activation", 3)))
        self._solver_combo.setCurrentIndex(int(payload.get("solver", 2)))
        self._alpha_slider.setValue(int(payload.get("alpha_index", 1)))
        self._max_iter_spin.setValue(int(payload.get("max_iter", 200)))
        self._replicable_cb.setChecked(bool(payload.get("replicable", True)))
        self._update_alpha_label()
