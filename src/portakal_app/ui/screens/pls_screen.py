from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QFormLayout,
    QGroupBox,
    QSpinBox,
    QVBoxLayout,
)

from portakal_app.data.services.sklearn_learner_service import SklearnLearnerService
from portakal_app.ui.screens.model_base import ModelScreenBase


class PLSScreen(ModelScreenBase):
    """Partial Least Squares Regression — multivariate latent variable regression."""

    _OUTPUT_PORT_LABEL = "Model"

    def __init__(self, parent=None) -> None:
        self._svc = SklearnLearnerService()
        super().__init__(parent)

    def _add_main_layout(self, layout: QVBoxLayout) -> None:
        box = QGroupBox("Optimization Parameters")
        form = QFormLayout(box)
        form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)

        self._n_components_spin = QSpinBox()
        self._n_components_spin.setRange(1, 50)
        self._n_components_spin.setValue(2)
        self._n_components_spin.setAlignment(Qt.AlignmentFlag.AlignRight)
        self._n_components_spin.valueChanged.connect(self._settings_changed)
        form.addRow("Components:", self._n_components_spin)

        self._max_iter_spin = QSpinBox()
        self._max_iter_spin.setRange(5, 1000000)
        self._max_iter_spin.setSingleStep(50)
        self._max_iter_spin.setValue(500)
        self._max_iter_spin.setAlignment(Qt.AlignmentFlag.AlignRight)
        self._max_iter_spin.valueChanged.connect(self._settings_changed)
        form.addRow("Iteration limit:", self._max_iter_spin)

        self._scale_cb = QCheckBox("Scale features and target")
        self._scale_cb.setChecked(True)
        self._scale_cb.stateChanged.connect(self._settings_changed)
        form.addRow(self._scale_cb)

        layout.addWidget(box)

    def _train(self):
        from sklearn.cross_decomposition import PLSRegression

        ds = self._dataset
        target_cols = ds.domain.target_columns
        if not target_cols:
            raise ValueError("No target column.")
        if target_cols[0].logical_type not in {"numeric"}:
            raise ValueError("PLS Regression requires a numeric target.")

        n_comp = self._n_components_spin.value()
        max_iter = self._max_iter_spin.value()
        scale = self._scale_cb.isChecked()

        est = PLSRegression(n_components=n_comp, max_iter=max_iter, scale=scale)
        params = {"n_components": n_comp, "max_iter": max_iter, "scale": scale}
        return self._svc.fit(est, ds, "PLS", "pls", params)

    def serialize_node_state(self) -> dict:
        return {
            **super().serialize_node_state(),
            "n_components": self._n_components_spin.value(),
            "max_iter": self._max_iter_spin.value(),
            "scale": self._scale_cb.isChecked(),
        }

    def restore_node_state(self, payload: dict) -> None:
        super().restore_node_state(payload)
        self._n_components_spin.setValue(int(payload.get("n_components", 2)))
        self._max_iter_spin.setValue(int(payload.get("max_iter", 500)))
        self._scale_cb.setChecked(bool(payload.get("scale", True)))
