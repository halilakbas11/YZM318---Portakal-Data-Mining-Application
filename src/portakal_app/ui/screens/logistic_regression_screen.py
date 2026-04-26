from __future__ import annotations

from itertools import chain

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QRadioButton,
    QSlider,
    QVBoxLayout,
)

from portakal_app.data.services.logistic_regression_service import (
    LogisticRegressionService,
    LogisticRegressionSettings,
)
from portakal_app.models import WorkflowPayload
from portakal_app.ui.screens.model_base import ModelScreenBase


_C_VALUES = list(
    chain(
        range(1000, 200, -50),
        range(200, 100, -10),
        range(100, 20, -5),
        range(20, 0, -1),
        [x / 10 for x in range(9, 2, -1)],
        [x / 100 for x in range(20, 2, -1)],
        [x / 1000 for x in range(20, 0, -1)],
    )
)
_PENALTIES = ("Lasso (L1)", "Ridge (L2)", "None")
_PENALTY_SHORT = ["l1", "l2", None]


class LogisticRegressionScreen(ModelScreenBase):
    """Logistic Regression with L1/L2/no regularisation — Orange equivalent."""

    _OUTPUT_PORT_LABEL = "Classifier"

    def __init__(self, parent=None) -> None:
        self._svc = LogisticRegressionService()
        self._c_index = 61
        super().__init__(parent)

    def _add_main_layout(self, layout: QVBoxLayout) -> None:
        reg_box = QGroupBox("Regularization")
        reg_layout = QVBoxLayout(reg_box)

        self._penalty_group = QButtonGroup(self)
        for i, label in enumerate(_PENALTIES):
            rb = QRadioButton(label)
            if i == 1:
                rb.setChecked(True)
            self._penalty_group.addButton(rb, i)
            reg_layout.addWidget(rb)
        self._penalty_group.idToggled.connect(self._on_penalty_changed)

        strength_layout = QVBoxLayout()
        strength_layout.setContentsMargins(20, 4, 0, 0)
        lbl = QLabel("Strength:")
        strength_layout.addWidget(lbl)

        slider_row = QHBoxLayout()
        slider_row.addWidget(QLabel("Weak"))
        self._c_slider = QSlider(Qt.Orientation.Horizontal)
        self._c_slider.setRange(0, len(_C_VALUES) - 1)
        self._c_slider.setValue(self._c_index)
        self._c_slider.valueChanged.connect(self._on_c_changed)
        self._c_slider.sliderReleased.connect(self._settings_changed)
        slider_row.addWidget(self._c_slider, 1)
        slider_row.addWidget(QLabel("Strong"))
        strength_layout.addLayout(slider_row)

        self._c_value_label = QLabel()
        self._c_value_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        strength_layout.addWidget(self._c_value_label)
        self._update_c_label()
        reg_layout.addLayout(strength_layout)

        layout.addWidget(reg_box)

        weight_box = QGroupBox()
        weight_layout = QVBoxLayout(weight_box)
        self._balance_cb = QCheckBox("Balance class distribution")
        self._balance_cb.stateChanged.connect(self._settings_changed)
        weight_layout.addWidget(self._balance_cb)
        layout.addWidget(weight_box)

    def _on_c_changed(self, value: int) -> None:
        self._c_index = value
        self._update_c_label()

    def _update_c_label(self) -> None:
        penalty_idx = self._penalty_group.checkedId()
        if penalty_idx == 2:
            self._c_value_label.setText("N/A")
            self._c_slider.setEnabled(False)
        else:
            self._c_slider.setEnabled(True)
            c = _C_VALUES[self._c_index]
            fmt = "C={}" if c >= 1 else "C={:.3f}"
            self._c_value_label.setText(fmt.format(c))

    def _on_penalty_changed(self, _id: int, checked: bool) -> None:
        if checked:
            self._update_c_label()
            self._settings_changed()

    def _train(self):
        c = _C_VALUES[self._c_index]
        penalty_idx = self._penalty_group.checkedId()
        # Map penalty to ridge parameter: L2=1/C, no-penalty=very small ridge
        if penalty_idx == 2:
            ridge = 1e-8
        else:
            ridge = 1.0 / max(float(c), 1e-10)
        settings = LogisticRegressionSettings(
            max_iter=200,
            ridge=ridge,
            tolerance=1e-6,
        )
        return self._svc.fit(self._dataset, settings)

    def current_output_payload(self) -> WorkflowPayload | None:
        if self._model_artifact is None:
            return None
        return WorkflowPayload("Classifier", self._model_artifact)

    def serialize_node_state(self) -> dict:
        return {
            **super().serialize_node_state(),
            "penalty": self._penalty_group.checkedId(),
            "c_index": self._c_slider.value(),
            "balance": self._balance_cb.isChecked(),
        }

    def restore_node_state(self, payload: dict) -> None:
        super().restore_node_state(payload)
        p_id = int(payload.get("penalty", 1))
        btn = self._penalty_group.button(p_id)
        if btn:
            btn.setChecked(True)
        self._c_slider.setValue(int(payload.get("c_index", 61)))
        self._balance_cb.setChecked(bool(payload.get("balance", False)))
        self._update_c_label()
