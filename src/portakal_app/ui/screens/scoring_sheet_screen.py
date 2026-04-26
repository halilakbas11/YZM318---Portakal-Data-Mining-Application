from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QFormLayout,
    QGroupBox,
    QSpinBox,
    QVBoxLayout,
)

from portakal_app.data.services.scoring_sheet_service import ScoringSheetService, ScoringSheetSettings
from portakal_app.models import WorkflowPayload
from portakal_app.ui.screens.model_base import ModelScreenBase


class ScoringSheetScreen(ModelScreenBase):
    """Scoring Sheet — fast explainable point-based classifier."""

    _OUTPUT_PORT_LABEL = "Classifier"

    def __init__(self, parent=None) -> None:
        self._svc = ScoringSheetService()
        super().__init__(parent)

    def _add_main_layout(self, layout: QVBoxLayout) -> None:
        box = QGroupBox("Model Parameters")
        form = QFormLayout(box)
        form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)

        self._n_rules_spin = QSpinBox()
        self._n_rules_spin.setRange(1, 50)
        self._n_rules_spin.setValue(5)
        self._n_rules_spin.setAlignment(Qt.AlignmentFlag.AlignRight)
        self._n_rules_spin.valueChanged.connect(self._settings_changed)
        form.addRow("Maximum number of decision parameters:", self._n_rules_spin)

        self._max_points_spin = QSpinBox()
        self._max_points_spin.setRange(1, 100)
        self._max_points_spin.setValue(5)
        self._max_points_spin.setAlignment(Qt.AlignmentFlag.AlignRight)
        self._max_points_spin.valueChanged.connect(self._settings_changed)
        form.addRow("Maximum points per decision parameter:", self._max_points_spin)

        layout.addWidget(box)

    def _train(self):
        settings = ScoringSheetSettings(
            max_rules=self._n_rules_spin.value(),
            max_points_per_rule=self._max_points_spin.value(),
        )
        return self._svc.fit(self._dataset, settings)

    def current_output_payload(self) -> WorkflowPayload | None:
        if self._model_artifact is None:
            return None
        return WorkflowPayload("Classifier", self._model_artifact)

    def serialize_node_state(self) -> dict:
        return {
            **super().serialize_node_state(),
            "n_rules": self._n_rules_spin.value(),
            "max_points": self._max_points_spin.value(),
        }

    def restore_node_state(self, payload: dict) -> None:
        super().restore_node_state(payload)
        self._n_rules_spin.setValue(int(payload.get("n_rules", 5)))
        self._max_points_spin.setValue(int(payload.get("max_points", 5)))
