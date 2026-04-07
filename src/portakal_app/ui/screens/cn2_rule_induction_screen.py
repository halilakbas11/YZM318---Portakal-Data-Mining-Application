from __future__ import annotations

from PySide6.QtCore import QSize
from PySide6.QtWidgets import (
    QCheckBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from portakal_app.data.models import DatasetHandle
from portakal_app.data.services.cn2_rule_induction_service import CN2InductionSettings, CN2RuleInductionService
from portakal_app.models import WorkflowPayload
from portakal_app.rule_artifacts import CN2RuleClassifierArtifact
from portakal_app.ui import i18n
from portakal_app.ui.screens.node_screen import WorkflowNodeScreenSupport


class CN2RuleInductionScreen(QWidget, WorkflowNodeScreenSupport):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._init_workflow_node_support()
        self._service = CN2RuleInductionService()
        self._dataset: DatasetHandle | None = None
        self._classifier: CN2RuleClassifierArtifact | None = None
        self._build_ui()
        self._update_status(i18n.t("Connect labeled data to induce CN2-style rules."))

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(10)

        title = QLabel(i18n.t("CN2 Rule Induction"))
        title.setProperty("sectionTitle", True)
        layout.addWidget(title)

        description = QLabel(
            i18n.t("Basic CN2-style learner for the Rule Viewer. It induces simple high-precision one-condition rules from categorical targets.")
        )
        description.setWordWrap(True)
        description.setProperty("muted", True)
        layout.addWidget(description)

        info_box = QGroupBox(i18n.t("Data"))
        info_layout = QVBoxLayout(info_box)
        self._dataset_label = QLabel(i18n.t("No data"))
        self._dataset_label.setWordWrap(True)
        info_layout.addWidget(self._dataset_label)
        layout.addWidget(info_box)

        params_box = QGroupBox(i18n.t("Parameters"))
        params_layout = QFormLayout(params_box)
        self._max_rules_spin = QSpinBox(params_box)
        self._max_rules_spin.setRange(1, 50)
        self._max_rules_spin.setValue(8)
        self._max_rules_spin.valueChanged.connect(self._on_parameters_changed)
        params_layout.addRow(i18n.t("Max rules"), self._max_rules_spin)

        self._min_coverage_spin = QSpinBox(params_box)
        self._min_coverage_spin.setRange(1, 100000)
        self._min_coverage_spin.setValue(3)
        self._min_coverage_spin.valueChanged.connect(self._on_parameters_changed)
        params_layout.addRow(i18n.t("Min rule coverage"), self._min_coverage_spin)
        layout.addWidget(params_box)

        self._summary_label = QLabel(self)
        self._summary_label.setWordWrap(True)
        layout.addWidget(self._summary_label)

        self._status_label = QLabel(self)
        self._status_label.setWordWrap(True)
        self._status_label.setProperty("muted", True)
        layout.addWidget(self._status_label)

        footer = QHBoxLayout()
        self.cb_apply_auto = QCheckBox(i18n.t("Apply Automatically"))
        self.cb_apply_auto.setChecked(True)
        footer.addWidget(self.cb_apply_auto)
        footer.addStretch(1)
        self._apply_button = QPushButton(i18n.t("Apply"))
        self._apply_button.setProperty("primary", True)
        self._apply_button.clicked.connect(self._apply)
        footer.addWidget(self._apply_button)
        layout.addLayout(footer)

    def sizeHint(self) -> QSize:
        return QSize(520, 320)

    def set_input_payload(self, payload: WorkflowPayload | None) -> None:
        self._dataset = payload.dataset if payload is not None and isinstance(payload.dataset, DatasetHandle) else None
        if self._dataset is None:
            self._classifier = None
        self._refresh_labels()
        if self.cb_apply_auto.isChecked():
            self._apply()
        else:
            self._notify_output_changed()

    def current_output_dataset(self):
        return self._classifier

    def documentation_url(self) -> str:
        return "https://orangedatamining.com/widget-catalog/model/cn2ruleinduction/"

    def help_text(self) -> str:
        return i18n.t("Induce a lightweight CN2-style rule classifier from labeled data and send it to CN2 Rule Viewer.")

    def footer_status_text(self) -> str:
        if self._classifier is None:
            return "0"
        return str(len(self._classifier.rule_list))

    def serialize_node_state(self) -> dict[str, object]:
        return {
            "auto_apply": self.cb_apply_auto.isChecked(),
            "max_rules": self._max_rules_spin.value(),
            "min_coverage": self._min_coverage_spin.value(),
        }

    def restore_node_state(self, payload: dict[str, object]) -> None:
        self.cb_apply_auto.setChecked(bool(payload.get("auto_apply", True)))
        self._max_rules_spin.setValue(int(payload.get("max_rules", 8)))
        self._min_coverage_spin.setValue(int(payload.get("min_coverage", 3)))

    def _on_parameters_changed(self) -> None:
        if self.cb_apply_auto.isChecked():
            self._apply()

    def _refresh_labels(self) -> None:
        self._dataset_label.setText(
            i18n.t("No data")
            if self._dataset is None
            else i18n.tf("{name}: {rows} rows x {cols} cols", name=self._dataset.display_name, rows=self._dataset.row_count, cols=self._dataset.column_count)
        )

    def _update_status(self, message: str) -> None:
        self._status_label.setText(message)

    def _apply(self) -> None:
        if self._dataset is None:
            self._classifier = None
            self._summary_label.setText("")
            self._update_status(i18n.t("Connect labeled data to induce CN2-style rules."))
            self._notify_output_changed()
            return

        try:
            self._classifier = self._service.induce(
                self._dataset,
                CN2InductionSettings(
                    max_rules=self._max_rules_spin.value(),
                    min_covered_examples=self._min_coverage_spin.value(),
                ),
            )
        except ValueError as exc:
            self._classifier = None
            self._summary_label.setText("")
            self._update_status(str(exc))
            self._notify_output_changed()
            return

        default_rule = self._classifier.rule_list[-1]
        self._summary_label.setText(
            i18n.tf(
                "Generated {count} rules. Default prediction: {prediction}",
                count=len(self._classifier.rule_list),
                prediction=default_rule.prediction,
            )
        )
        self._update_status(i18n.t("Classifier ready. Connect it to CN2 Rule Viewer."))
        self._notify_output_changed()
