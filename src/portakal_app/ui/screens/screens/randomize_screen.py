from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from portakal_app.data.models import DatasetHandle
from portakal_app.data.services.randomize_service import RandomizeService
from portakal_app.ui.screens.node_screen import WorkflowNodeScreenSupport
from portakal_app.ui import i18n


class RandomizeScreen(QWidget, WorkflowNodeScreenSupport):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._init_workflow_node_support()
        self._service = RandomizeService()
        self._dataset_handle: DatasetHandle | None = None
        self._output_dataset: DatasetHandle | None = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(10)

        columns_group = QGroupBox(i18n.t("Shuffled columns"))
        columns_layout = QHBoxLayout(columns_group)
        columns_layout.setContentsMargins(10, 10, 10, 10)
        columns_layout.setSpacing(15)

        self._shuffle_classes = QCheckBox(i18n.t("Classes"))
        self._shuffle_classes.setChecked(True)
        columns_layout.addWidget(self._shuffle_classes)

        self._shuffle_features = QCheckBox(i18n.t("Features"))
        self._shuffle_features.setChecked(False)
        columns_layout.addWidget(self._shuffle_features)

        self._shuffle_metas = QCheckBox(i18n.t("Metas"))
        self._shuffle_metas.setChecked(False)
        columns_layout.addWidget(self._shuffle_metas)
        
        columns_layout.addStretch(1)
        layout.addWidget(columns_group)

        rows_group = QGroupBox(i18n.t("Shuffled rows"))
        rows_layout = QVBoxLayout(rows_group)
        rows_layout.setContentsMargins(10, 10, 10, 10)
        rows_layout.setSpacing(10)

        slider_row = QHBoxLayout()
        slider_row.addWidget(QLabel(i18n.t("None")))
        self._ratio_slider = QSlider(Qt.Orientation.Horizontal)
        self._ratio_slider.setRange(0, 100)
        self._ratio_slider.setValue(100)
        self._ratio_slider.valueChanged.connect(self._on_ratio_changed)
        slider_row.addWidget(self._ratio_slider, 1)
        slider_row.addWidget(QLabel(i18n.t("All")))

        rows_layout.addLayout(slider_row)

        self._ratio_label = QLabel(i18n.tf("{value}%", value=100))
        self._ratio_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        rows_layout.addWidget(self._ratio_label)

        self._replicable = QCheckBox(i18n.t("Replicable shuffling"))
        self._replicable.setChecked(False)
        rows_layout.addWidget(self._replicable)

        layout.addWidget(rows_group)
        layout.addStretch(1)

        self._result_label = QLabel("")
        self._result_label.setWordWrap(True)
        layout.addWidget(self._result_label)

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

    def set_input_payload(self, payload) -> None:
        dataset = payload.dataset if payload is not None else None
        self._dataset_handle = dataset
        self._output_dataset = None
        
        self._apply()

    def current_output_dataset(self) -> DatasetHandle | None:
        return self._output_dataset

    def serialize_node_state(self) -> dict[str, object]:
        return {
            "shuffle_classes": self._shuffle_classes.isChecked(),
            "shuffle_features": self._shuffle_features.isChecked(),
            "shuffle_metas": self._shuffle_metas.isChecked(),
            "ratio": self._ratio_slider.value(),
            "replicable": self._replicable.isChecked(),
            "auto_apply": self.cb_apply_auto.isChecked()
        }

    def restore_node_state(self, payload: dict[str, object]) -> None:
        self._shuffle_classes.setChecked(bool(payload.get("shuffle_classes", True)))
        self._shuffle_features.setChecked(bool(payload.get("shuffle_features", False)))
        self._shuffle_metas.setChecked(bool(payload.get("shuffle_metas", False)))
        
        ratio = int(payload.get("ratio", 100))
        self._ratio_slider.setValue(ratio)
        self._ratio_label.setText(i18n.tf("{value}%", value=ratio))
        
        # Support older saves for 'use_seed'
        rep = bool(payload.get("replicable", payload.get("use_seed", False)))
        self._replicable.setChecked(rep)
        
        self.cb_apply_auto.setChecked(bool(payload.get("auto_apply", True)))

        if self.cb_apply_auto.isChecked():
            self._apply()

    def help_text(self) -> str:
        return "Randomize (shuffle) the order of rows, columns, or values in the dataset."

    def documentation_url(self) -> str:
        return "https://orangedatamining.com/widget-catalog/transform/randomize/"

    def _on_ratio_changed(self, value: int) -> None:
        self._ratio_label.setText(i18n.tf("{value}%", value=value))
        if self.cb_apply_auto.isChecked():
            self._apply()

    def _apply(self) -> None:
        if self._dataset_handle is None:
            self._output_dataset = None
            self._result_label.setText("")
            self._notify_output_changed()
            return

        seed = 42 if self._replicable.isChecked() else None
        
        try:
            self._output_dataset = self._service.randomize(
                self._dataset_handle,
                shuffle_classes=self._shuffle_classes.isChecked(),
                shuffle_features=self._shuffle_features.isChecked(),
                shuffle_metas=self._shuffle_metas.isChecked(),
                shuffle_ratio=self._ratio_slider.value(),
                seed=seed,
            )
            self._result_label.setText(i18n.t("Randomization successful."))
        except Exception as e:
            self._output_dataset = None
            self._result_label.setText(i18n.tf("Error: {err}", err=e))

        self._notify_output_changed()

    def refresh_translations(self) -> None:
        self._ratio_label.setText(i18n.tf("{value}%", value=self._ratio_slider.value()))
