from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QFormLayout,
    QGroupBox,
    QSpinBox,
    QVBoxLayout,
)

from portakal_app.data.services.cn2_rule_induction_service import (
    CN2RuleInductionService,
    CN2InductionSettings,
)
from portakal_app.models import WorkflowPayload
from portakal_app.ui.screens.model_base import ModelScreenBase


class CN2RuleInductionScreen(ModelScreenBase):
    """CN2 Rule Induction — induce classification rules from data."""

    _OUTPUT_PORT_LABEL = "Classifier"

    def __init__(self, parent=None) -> None:
        self._svc = CN2RuleInductionService()
        super().__init__(parent)

    def _add_main_layout(self, layout: QVBoxLayout) -> None:
        box = QGroupBox("Rule Induction")
        form = QFormLayout(box)
        form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)

        self._max_rules_spin = QSpinBox()
        self._max_rules_spin.setRange(1, 1000)
        self._max_rules_spin.setValue(8)
        self._max_rules_spin.setAlignment(Qt.AlignmentFlag.AlignRight)
        self._max_rules_spin.valueChanged.connect(self._settings_changed)
        form.addRow("Maximum number of rules:", self._max_rules_spin)

        self._min_cov_spin = QSpinBox()
        self._min_cov_spin.setRange(1, 10000)
        self._min_cov_spin.setValue(3)
        self._min_cov_spin.setAlignment(Qt.AlignmentFlag.AlignRight)
        self._min_cov_spin.valueChanged.connect(self._settings_changed)
        form.addRow("Minimum rule coverage:", self._min_cov_spin)

        layout.addWidget(box)

    def _train(self):
        settings = CN2InductionSettings(
            max_rules=self._max_rules_spin.value(),
            min_covered_examples=self._min_cov_spin.value(),
        )
        return self._svc.induce(self._dataset, settings)

    def current_output_payload(self) -> WorkflowPayload | None:
        if self._model_artifact is None:
            return None
        return WorkflowPayload("Classifier", self._model_artifact)

    def serialize_node_state(self) -> dict:
        return {
            **super().serialize_node_state(),
            "max_rules": self._max_rules_spin.value(),
            "min_cov": self._min_cov_spin.value(),
        }

    def restore_node_state(self, payload: dict) -> None:
        super().restore_node_state(payload)
        self._max_rules_spin.setValue(int(payload.get("max_rules", 8)))
        self._min_cov_spin.setValue(int(payload.get("min_cov", 3)))
