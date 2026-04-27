from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QSpinBox,
    QVBoxLayout,
)

from portakal_app.data.services.sklearn_learner_service import SklearnLearnerService
from portakal_app.ui.screens.model_base import ModelScreenBase

_LOSSES = ["linear", "square", "exponential"]
_LOSS_LABELS = ["Linear", "Square", "Exponential"]


class AdaBoostScreen(ModelScreenBase):
    """AdaBoost ensemble — combines weak learners via adaptive boosting.

    Uses decision stumps as default base estimators. Supports classification
    (AdaBoostClassifier) and regression (AdaBoostRegressor) with selectable
    loss functions for the regression variant.
    """

    _OUTPUT_PORT_LABEL = "Model"

    def __init__(self, parent=None) -> None:
        self._svc = SklearnLearnerService()
        super().__init__(parent)

    def _add_main_layout(self, layout: QVBoxLayout) -> None:
        basic = QGroupBox("Basic Properties")
        form = QFormLayout(basic)
        form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)

        self._n_spin = QSpinBox()
        self._n_spin.setRange(1, 10000)
        self._n_spin.setValue(50)
        self._n_spin.setAlignment(Qt.AlignmentFlag.AlignRight)
        self._n_spin.setToolTip("Maximum number of weak learners (boosting rounds).")
        self._n_spin.valueChanged.connect(self._settings_changed)
        form.addRow("Number of estimators:", self._n_spin)

        self._lr_spin = QDoubleSpinBox()
        self._lr_spin.setRange(1e-5, 1.0)
        self._lr_spin.setValue(1.0)
        self._lr_spin.setSingleStep(1e-5)
        self._lr_spin.setDecimals(5)
        self._lr_spin.setAlignment(Qt.AlignmentFlag.AlignRight)
        self._lr_spin.setToolTip("Weight applied to each weak learner. Smaller values require more estimators.")
        self._lr_spin.valueChanged.connect(self._settings_changed)
        form.addRow("Learning rate:", self._lr_spin)

        self._loss_combo = QComboBox()
        self._loss_combo.addItems(_LOSS_LABELS)
        self._loss_combo.currentIndexChanged.connect(self._settings_changed)
        form.addRow("Loss (regression):", self._loss_combo)

        layout.addWidget(basic)

        repro_box = QGroupBox("Reproducibility")
        repro_form = QFormLayout(repro_box)

        self._seed_cb = QCheckBox("Fixed seed for random generator:")
        self._seed_cb.stateChanged.connect(self._on_seed_toggled)
        self._seed_spin = QSpinBox()
        self._seed_spin.setRange(0, 2 ** 31 - 1)
        self._seed_spin.setValue(0)
        self._seed_spin.setAlignment(Qt.AlignmentFlag.AlignRight)
        self._seed_spin.setEnabled(False)  # disabled until checkbox is checked
        self._seed_spin.valueChanged.connect(self._settings_changed)
        repro_form.addRow(self._seed_cb, self._seed_spin)

        layout.addWidget(repro_box)

    def _on_seed_toggled(self) -> None:
        """Enable or disable the seed spin box based on the checkbox state."""
        checked = self._seed_cb.isChecked()
        self._seed_spin.setEnabled(checked)
        self._settings_changed()

    def _train(self):
        from sklearn.ensemble import AdaBoostClassifier, AdaBoostRegressor

        ds = self._dataset
        target_cols = ds.domain.target_columns
        if not target_cols:
            raise ValueError("No target column.")
        is_clf = target_cols[0].logical_type in {"categorical", "boolean"}

        n = self._n_spin.value()
        lr = self._lr_spin.value()
        loss = _LOSSES[self._loss_combo.currentIndex()]
        random_state = self._seed_spin.value() if self._seed_cb.isChecked() else None

        if is_clf:
            est = AdaBoostClassifier(n_estimators=n, learning_rate=lr, random_state=random_state)
        else:
            est = AdaBoostRegressor(n_estimators=n, learning_rate=lr, loss=loss, random_state=random_state)

        params = {
            "n_estimators": n,
            "learning_rate": lr,
            "loss": loss,
            "algorithm": "SAMME" if is_clf else "N/A",
        }
        return self._svc.fit(est, ds, "AdaBoost", "adaboost", params)

    def serialize_node_state(self) -> dict:
        return {
            **super().serialize_node_state(),
            "n_estimators": self._n_spin.value(),
            "learning_rate": self._lr_spin.value(),
            "loss": self._loss_combo.currentIndex(),
            "use_seed": self._seed_cb.isChecked(),
            "seed": self._seed_spin.value(),
        }

    def restore_node_state(self, payload: dict) -> None:
        super().restore_node_state(payload)
        self._n_spin.setValue(int(payload.get("n_estimators", 50)))
        self._lr_spin.setValue(float(payload.get("learning_rate", 1.0)))
        self._loss_combo.setCurrentIndex(int(payload.get("loss", 0)))
        self._seed_cb.setChecked(bool(payload.get("use_seed", False)))
        self._seed_spin.setValue(int(payload.get("seed", 0)))
