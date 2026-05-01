from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QRadioButton,
    QSpinBox,
    QVBoxLayout,
)

from portakal_app.data.services.cn2_rule_induction_service import (
    CN2RuleInductionService,
    CN2InductionSettings,
)
from portakal_app.models import WorkflowPayload
from portakal_app.ui.screens.model_base import ModelScreenBase

_MEASURE_LABELS = ["Entropy", "Laplace", "WRAcc"]
_MEASURE_VALUES = ["entropy", "laplace", "wracc"]


class CN2RuleInductionScreen(ModelScreenBase):
    """CN2 Rule Induction — induce classification rules from data."""

    _OUTPUT_PORT_LABEL = "Classifier"

    def __init__(self, parent=None) -> None:
        self._svc = CN2RuleInductionService()
        super().__init__(parent)

    def _add_main_layout(self, layout: QVBoxLayout) -> None:
        # ── Rule ordering ─────────────────────────────────────────────
        ordering_box = QGroupBox("Rule ordering")
        ordering_layout = QVBoxLayout(ordering_box)
        self._rb_ordered = QRadioButton("Ordered")
        self._rb_unordered = QRadioButton("Unordered")
        self._rb_ordered.setChecked(True)
        self._rb_ordered.toggled.connect(self._settings_changed)
        ordering_layout.addWidget(self._rb_ordered)
        ordering_layout.addWidget(self._rb_unordered)
        layout.addWidget(ordering_box)

        # ── Covering algorithm ────────────────────────────────────────
        covering_box = QGroupBox("Covering algorithm")
        covering_grid = QGridLayout(covering_box)
        covering_grid.setColumnStretch(0, 1)

        self._rb_exclusive = QRadioButton("Exclusive")
        self._rb_weighted = QRadioButton("Weighted")
        self._rb_exclusive.setChecked(True)
        self._rb_exclusive.toggled.connect(self._settings_changed)

        self._gamma_spin = QDoubleSpinBox()
        self._gamma_spin.setRange(0.0, 1.0)
        self._gamma_spin.setSingleStep(0.05)
        self._gamma_spin.setDecimals(2)
        self._gamma_spin.setValue(0.70)
        self._gamma_spin.setPrefix("γ: ")
        self._gamma_spin.setAlignment(Qt.AlignmentFlag.AlignRight)
        self._gamma_spin.setEnabled(False)
        self._gamma_spin.valueChanged.connect(self._settings_changed)
        self._rb_weighted.toggled.connect(self._gamma_spin.setEnabled)

        covering_grid.addWidget(self._rb_exclusive, 0, 0)
        covering_grid.addWidget(self._rb_weighted, 1, 0)
        covering_grid.addWidget(self._gamma_spin, 1, 1)
        layout.addWidget(covering_box)

        # ── Rule search ───────────────────────────────────────────────
        search_box = QGroupBox("Rule search")
        search_form = QFormLayout(search_box)
        search_form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)

        self._measure_combo = QComboBox()
        self._measure_combo.addItems(_MEASURE_LABELS)
        self._measure_combo.currentIndexChanged.connect(self._settings_changed)
        search_form.addRow("Evaluation measure:", self._measure_combo)

        self._beam_spin = QSpinBox()
        self._beam_spin.setRange(1, 100)
        self._beam_spin.setValue(5)
        self._beam_spin.setAlignment(Qt.AlignmentFlag.AlignRight)
        self._beam_spin.valueChanged.connect(self._settings_changed)
        search_form.addRow("Beam width:", self._beam_spin)
        layout.addWidget(search_box)

        # ── Rule filtering ────────────────────────────────────────────
        filtering_box = QGroupBox("Rule filtering")
        filtering_layout = QVBoxLayout(filtering_box)

        filter_form = QFormLayout()
        filter_form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)

        self._min_cov_spin = QSpinBox()
        self._min_cov_spin.setRange(1, 10000)
        self._min_cov_spin.setValue(1)
        self._min_cov_spin.setAlignment(Qt.AlignmentFlag.AlignRight)
        self._min_cov_spin.valueChanged.connect(self._settings_changed)
        filter_form.addRow("Minimum rule coverage:", self._min_cov_spin)

        self._max_len_spin = QSpinBox()
        self._max_len_spin.setRange(1, 100)
        self._max_len_spin.setValue(5)
        self._max_len_spin.setAlignment(Qt.AlignmentFlag.AlignRight)
        self._max_len_spin.valueChanged.connect(self._settings_changed)
        filter_form.addRow("Maximum rule length:", self._max_len_spin)

        filtering_layout.addLayout(filter_form)

        # Statistical significance row
        stat_row = QHBoxLayout()
        self._cb_stat = QCheckBox("Statistical significance (default α):")
        self._stat_alpha_spin = QDoubleSpinBox()
        self._stat_alpha_spin.setRange(0.0, 1.0)
        self._stat_alpha_spin.setSingleStep(0.05)
        self._stat_alpha_spin.setDecimals(2)
        self._stat_alpha_spin.setValue(1.0)
        self._stat_alpha_spin.setAlignment(Qt.AlignmentFlag.AlignRight)
        self._stat_alpha_spin.setEnabled(False)
        self._cb_stat.toggled.connect(self._stat_alpha_spin.setEnabled)
        self._cb_stat.toggled.connect(self._settings_changed)
        self._stat_alpha_spin.valueChanged.connect(self._settings_changed)
        stat_row.addWidget(self._cb_stat)
        stat_row.addWidget(self._stat_alpha_spin)
        filtering_layout.addLayout(stat_row)

        # Relative significance row
        rel_row = QHBoxLayout()
        self._cb_rel = QCheckBox("Relative significance (parent α):")
        self._rel_alpha_spin = QDoubleSpinBox()
        self._rel_alpha_spin.setRange(0.0, 1.0)
        self._rel_alpha_spin.setSingleStep(0.05)
        self._rel_alpha_spin.setDecimals(2)
        self._rel_alpha_spin.setValue(1.0)
        self._rel_alpha_spin.setAlignment(Qt.AlignmentFlag.AlignRight)
        self._rel_alpha_spin.setEnabled(False)
        self._cb_rel.toggled.connect(self._rel_alpha_spin.setEnabled)
        self._cb_rel.toggled.connect(self._settings_changed)
        self._rel_alpha_spin.valueChanged.connect(self._settings_changed)
        rel_row.addWidget(self._cb_rel)
        rel_row.addWidget(self._rel_alpha_spin)
        filtering_layout.addLayout(rel_row)

        # Restrict categorical checkbox
        self._cb_restrict_cat = QCheckBox(
            "Restrict operator for categorical values to equality"
        )
        self._cb_restrict_cat.toggled.connect(self._settings_changed)
        filtering_layout.addWidget(self._cb_restrict_cat)

        layout.addWidget(filtering_box)

    def _train(self):
        measure_idx = self._measure_combo.currentIndex()
        measure = _MEASURE_VALUES[measure_idx] if 0 <= measure_idx < len(_MEASURE_VALUES) else "entropy"

        settings = CN2InductionSettings(
            rule_ordering="ordered" if self._rb_ordered.isChecked() else "unordered",
            covering_algorithm="exclusive" if self._rb_exclusive.isChecked() else "weighted",
            weighted_gamma=self._gamma_spin.value(),
            evaluation_measure=measure,
            beam_width=self._beam_spin.value(),
            min_covered_examples=self._min_cov_spin.value(),
            max_rule_length=self._max_len_spin.value(),
            statistical_significance=self._cb_stat.isChecked(),
            statistical_alpha=self._stat_alpha_spin.value(),
            relative_significance=self._cb_rel.isChecked(),
            relative_alpha=self._rel_alpha_spin.value(),
            restrict_categorical_to_equality=self._cb_restrict_cat.isChecked(),
        )
        return self._svc.induce(self._dataset, settings)

    def current_output_payload(self) -> WorkflowPayload | None:
        if self._model_artifact is None:
            return None
        return WorkflowPayload("Classifier", self._model_artifact)

    def serialize_node_state(self) -> dict:
        return {
            **super().serialize_node_state(),
            "rule_ordering": "ordered" if self._rb_ordered.isChecked() else "unordered",
            "covering_algorithm": "exclusive" if self._rb_exclusive.isChecked() else "weighted",
            "weighted_gamma": self._gamma_spin.value(),
            "evaluation_measure": self._measure_combo.currentIndex(),
            "beam_width": self._beam_spin.value(),
            "min_cov": self._min_cov_spin.value(),
            "max_rule_length": self._max_len_spin.value(),
            "stat_sig": self._cb_stat.isChecked(),
            "stat_alpha": self._stat_alpha_spin.value(),
            "rel_sig": self._cb_rel.isChecked(),
            "rel_alpha": self._rel_alpha_spin.value(),
            "restrict_cat": self._cb_restrict_cat.isChecked(),
        }

    def restore_node_state(self, payload: dict) -> None:
        super().restore_node_state(payload)
        ordering = payload.get("rule_ordering", "ordered")
        self._rb_ordered.setChecked(ordering == "ordered")
        self._rb_unordered.setChecked(ordering == "unordered")

        covering = payload.get("covering_algorithm", "exclusive")
        self._rb_exclusive.setChecked(covering == "exclusive")
        self._rb_weighted.setChecked(covering == "weighted")
        self._gamma_spin.setValue(float(payload.get("weighted_gamma", 0.70)))
        self._gamma_spin.setEnabled(covering == "weighted")

        measure_idx = int(payload.get("evaluation_measure", 0))
        self._measure_combo.setCurrentIndex(max(0, min(measure_idx, len(_MEASURE_LABELS) - 1)))

        self._beam_spin.setValue(int(payload.get("beam_width", 5)))
        self._min_cov_spin.setValue(int(payload.get("min_cov", 1)))
        self._max_len_spin.setValue(int(payload.get("max_rule_length", 5)))

        stat = bool(payload.get("stat_sig", False))
        self._cb_stat.setChecked(stat)
        self._stat_alpha_spin.setValue(float(payload.get("stat_alpha", 1.0)))
        self._stat_alpha_spin.setEnabled(stat)

        rel = bool(payload.get("rel_sig", False))
        self._cb_rel.setChecked(rel)
        self._rel_alpha_spin.setValue(float(payload.get("rel_alpha", 1.0)))
        self._rel_alpha_spin.setEnabled(rel)

        self._cb_restrict_cat.setChecked(bool(payload.get("restrict_cat", False)))
