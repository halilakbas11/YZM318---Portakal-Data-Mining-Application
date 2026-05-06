from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from portakal_app.data.services.distance_matrix_service import coerce_distance_matrix
from portakal_app.data.services.distance_transformation_service import DistanceTransformationService
from portakal_app.models import WorkflowPayload
from portakal_app.ui import i18n
from portakal_app.ui.screens.node_screen import WorkflowNodeScreenSupport


class DistanceTransformationScreen(QWidget, WorkflowNodeScreenSupport):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._init_workflow_node_support()
        self._service = DistanceTransformationService()
        self._input_value: object | None = None
        self._output_payload: WorkflowPayload | None = None
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(10)

        normalization_group = QGroupBox(i18n.t("Normalization"))
        normalization_form = QFormLayout(normalization_group)
        self._normalization_combo = QComboBox()
        self._normalization_combo.addItem(i18n.t("No normalization"), "none")
        self._normalization_combo.addItem("[0, 1]", "zero_one")
        self._normalization_combo.addItem("[-1, 1]", "minus_one_one")
        self._normalization_combo.addItem(i18n.t("Sigmoid"), "sigmoid")
        normalization_form.addRow(i18n.t("Mode:"), self._normalization_combo)
        layout.addWidget(normalization_group)

        inversion_group = QGroupBox(i18n.t("Inversion"))
        inversion_form = QFormLayout(inversion_group)
        self._inversion_combo = QComboBox()
        self._inversion_combo.addItem(i18n.t("No inversion"), "none")
        self._inversion_combo.addItem("-X", "negate")
        self._inversion_combo.addItem("1-X", "one_minus")
        self._inversion_combo.addItem("max(X)-X", "max_minus")
        self._inversion_combo.addItem("1/X", "reciprocal")
        inversion_form.addRow(i18n.t("Mode:"), self._inversion_combo)
        layout.addWidget(inversion_group)

        self._status_label = QLabel(i18n.t("Distances input is waiting."))
        self._status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._status_label.setWordWrap(True)
        layout.addWidget(self._status_label)
        layout.addStretch(1)

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

        self._normalization_combo.currentIndexChanged.connect(self._schedule_auto_apply)
        self._inversion_combo.currentIndexChanged.connect(self._schedule_auto_apply)
        self.cb_apply_auto.toggled.connect(lambda _checked: self._schedule_auto_apply())

    def set_input_payload(self, payload) -> None:
        self._input_value = None if payload is None else payload.value
        self._output_payload = None
        if self._input_value is None:
            self._status_label.setText(i18n.t("Distances input is waiting."))
            self._notify_output_changed()
            return
        try:
            handle = coerce_distance_matrix(self._input_value)
            self._status_label.setText(
                i18n.tf(
                    "Input: {rows} x {cols}",
                    rows=handle.matrix.shape[0],
                    cols=handle.matrix.shape[1],
                )
            )
        except Exception as exc:
            self._status_label.setText(i18n.tf("Error: {err}", err=exc))
        self._schedule_auto_apply()

    def current_output_payload(self) -> WorkflowPayload | None:
        return self._output_payload

    def serialize_node_state(self) -> dict[str, object]:
        return {
            "normalization": self._normalization_combo.currentData(),
            "inversion": self._inversion_combo.currentData(),
            "auto_apply": self.cb_apply_auto.isChecked(),
        }

    def restore_node_state(self, payload: dict[str, object]) -> None:
        normalization_index = self._normalization_combo.findData(str(payload.get("normalization", "none")))
        self._normalization_combo.setCurrentIndex(max(0, normalization_index))
        inversion_index = self._inversion_combo.findData(str(payload.get("inversion", "none")))
        self._inversion_combo.setCurrentIndex(max(0, inversion_index))
        self.cb_apply_auto.setChecked(bool(payload.get("auto_apply", True)))
        self._schedule_auto_apply()

    def help_text(self) -> str:
        return "Normalize or invert a distance matrix before downstream analysis."

    def documentation_url(self) -> str:
        return "https://orangedatamining.com/widget-catalog/unsupervised/distancetransformation/"

    def _apply(self) -> None:
        self._output_payload = None
        if self._input_value is None:
            self._status_label.setText(i18n.t("Distances input is waiting."))
            self._notify_output_changed()
            return
        try:
            handle = self._service.transform(
                self._input_value,
                normalization=str(self._normalization_combo.currentData()),
                inversion=str(self._inversion_combo.currentData()),
            )
            self._output_payload = WorkflowPayload("Distances", handle)
            self._status_label.setText(
                i18n.tf(
                    "Output: {rows} x {cols}",
                    rows=handle.matrix.shape[0],
                    cols=handle.matrix.shape[1],
                )
            )
        except Exception as exc:
            self._status_label.setText(i18n.tf("Error: {err}", err=exc))
        self._notify_output_changed()

