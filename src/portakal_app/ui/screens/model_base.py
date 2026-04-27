from __future__ import annotations

from typing import Any

from PySide6.QtWidgets import (
    QCheckBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from portakal_app.data.models import DatasetHandle
from portakal_app.models import WorkflowPayload
from portakal_app.ui import i18n
from portakal_app.ui.screens.node_screen import WorkflowNodeScreenSupport


class ModelScreenBase(QWidget, WorkflowNodeScreenSupport):
    """Shared scaffold for all learner screens: data label, status, auto-apply, apply button."""

    _OUTPUT_PORT_LABEL: str = "Model"

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._init_workflow_node_support()
        self._dataset: DatasetHandle | None = None
        self._model_artifact: Any | None = None
        self._extra_inputs: dict[str, Any] = {}

        outer = QVBoxLayout(self)
        outer.setContentsMargins(8, 8, 8, 8)
        outer.setSpacing(10)

        self._dataset_label = QLabel(i18n.t("No data on input."))
        self._dataset_label.setWordWrap(True)
        self._dataset_label.setStyleSheet("font-weight: bold; background: transparent;")
        outer.addWidget(self._dataset_label)

        self._status_label = QLabel("")
        self._status_label.setWordWrap(True)
        self._status_label.setStyleSheet("color: #777; background: transparent;")
        outer.addWidget(self._status_label)

        self._params_widget = QWidget()
        params_layout = QVBoxLayout(self._params_widget)
        params_layout.setContentsMargins(0, 0, 0, 0)
        params_layout.setSpacing(8)
        outer.addWidget(self._params_widget)
        self._add_main_layout(params_layout)

        outer.addStretch(1)

        footer = QHBoxLayout()
        self.cb_apply_auto = QCheckBox(i18n.t("Apply Automatically"))
        self.cb_apply_auto.setChecked(True)
        self.cb_apply_auto.stateChanged.connect(self._on_auto_apply_changed)
        footer.addWidget(self.cb_apply_auto)
        footer.addStretch(1)
        self._apply_button = QPushButton(i18n.t("Apply"))
        self._apply_button.setProperty("primary", True)
        self._apply_button.setEnabled(False)  # disabled while auto-apply is on
        self._apply_button.clicked.connect(self._on_apply_clicked)
        footer.addWidget(self._apply_button)
        outer.addLayout(footer)

    # ── Subclass hooks ────────────────────────────────────────────────

    def _add_main_layout(self, layout: QVBoxLayout) -> None:
        """Override to add parameter controls to `layout`."""

    def _train(self) -> Any | None:
        """Override to train and return the model artifact. Raise ValueError on error."""
        raise NotImplementedError

    # ── Workflow integration ──────────────────────────────────────────

    def set_input_payload(self, payload: WorkflowPayload | None) -> None:
        if payload is None:
            self._dataset = None
            self._extra_inputs = {}
        elif payload.port_label == "Data" and isinstance(payload.value, DatasetHandle):
            self._dataset = payload.value
        else:
            self._extra_inputs[payload.port_label] = payload.value
        self._apply()

    def current_output_payload(self) -> WorkflowPayload | None:
        if self._model_artifact is None:
            return None
        return WorkflowPayload(self._OUTPUT_PORT_LABEL, self._model_artifact)

    def serialize_node_state(self) -> dict[str, object]:
        return {"auto_apply": self.cb_apply_auto.isChecked()}

    def restore_node_state(self, payload: dict[str, object]) -> None:
        self.cb_apply_auto.setChecked(bool(payload.get("auto_apply", True)))

    # ── Internal helpers ──────────────────────────────────────────────

    def _on_auto_apply_changed(self) -> None:
        auto = self.cb_apply_auto.isChecked()
        self._apply_button.setEnabled(not auto)
        if auto:
            self._apply()

    def _on_apply_clicked(self) -> None:
        self._apply_button.setEnabled(False)
        self._apply()

    def _settings_changed(self) -> None:
        if not hasattr(self, "_apply_button"):
            return  # still initializing, button not created yet
        if self._is_auto_apply():
            self._apply()
        else:
            # Mark that there are pending changes the user needs to apply manually
            self._apply_button.setEnabled(True)

    def _apply(self) -> None:
        if self._dataset is None:
            self._model_artifact = None
            self._dataset_label.setText(i18n.t("No data on input."))
            self._status_label.setText("")
            self._notify_output_changed()
            return

        self._dataset_label.setText(
            i18n.tf(
                "Training on: {name} ({rows} rows, {cols} features)",
                name=self._dataset.display_name,
                rows=self._dataset.row_count,
                cols=len(list(self._dataset.domain.feature_columns)),
            )
        )
        try:
            self._model_artifact = self._train()
            self._status_label.setText(i18n.t("Model trained successfully."))
            self._status_label.setStyleSheet("color: #2e7d32; background: transparent;")
        except Exception as exc:  # noqa: BLE001
            self._model_artifact = None
            self._status_label.setText(str(exc))
            self._status_label.setStyleSheet("color: #c62828; background: transparent;")

        self._notify_output_changed()
