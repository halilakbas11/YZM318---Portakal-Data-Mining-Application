from __future__ import annotations

from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
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
from portakal_app.data.services.outliers_service import OutliersService
from portakal_app.ui import i18n
from portakal_app.ui.screens.node_screen import WorkflowNodeScreenSupport


class OutliersScreen(QWidget, WorkflowNodeScreenSupport):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._init_workflow_node_support()
        self._service = OutliersService()
        self._dataset_handle: DatasetHandle | None = None
        self._outputs: dict[str, DatasetHandle] = {}
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(10)

        settings = QGroupBox(i18n.t("Method"))
        form = QFormLayout(settings)

        self._method_combo = QComboBox()
        self._method_combo.addItem(i18n.t("One Class SVM"), "one_class_svm")
        self._method_combo.addItem(i18n.t("Covariance Estimator"), "covariance")
        self._method_combo.addItem(i18n.t("Local Outlier Factor"), "local_outlier_factor")
        self._method_combo.addItem(i18n.t("Isolation Forest"), "isolation_forest")
        form.addRow(i18n.t("Method:"), self._method_combo)

        self._contamination_spin = QDoubleSpinBox()
        self._contamination_spin.setRange(0.001, 0.499)
        self._contamination_spin.setDecimals(3)
        self._contamination_spin.setSingleStep(0.01)
        self._contamination_spin.setValue(0.10)
        form.addRow(i18n.t("Contamination:"), self._contamination_spin)

        self._neighbors_spin = QSpinBox()
        self._neighbors_spin.setRange(1, 200)
        self._neighbors_spin.setValue(20)
        form.addRow(i18n.t("Neighbors:"), self._neighbors_spin)

        self._metric_combo = QComboBox()
        self._metric_combo.addItem(i18n.t("Euclidean"), "euclidean")
        self._metric_combo.addItem(i18n.t("Manhattan"), "manhattan")
        self._metric_combo.addItem(i18n.t("Maximal"), "chebyshev")
        self._metric_combo.addItem(i18n.t("Mahalanobis"), "mahalanobis")
        form.addRow(i18n.t("Metric:"), self._metric_combo)

        self._nu_spin = QDoubleSpinBox()
        self._nu_spin.setRange(0.001, 0.999)
        self._nu_spin.setDecimals(3)
        self._nu_spin.setSingleStep(0.01)
        self._nu_spin.setValue(0.10)
        form.addRow(i18n.t("Nu:"), self._nu_spin)

        self._gamma_combo = QComboBox()
        self._gamma_combo.addItem("scale")
        self._gamma_combo.addItem("auto")
        form.addRow(i18n.t("Gamma:"), self._gamma_combo)

        self._support_fraction_spin = QDoubleSpinBox()
        self._support_fraction_spin.setRange(0.1, 1.0)
        self._support_fraction_spin.setDecimals(2)
        self._support_fraction_spin.setSingleStep(0.05)
        self._support_fraction_spin.setValue(1.0)
        form.addRow(i18n.t("Support fraction:"), self._support_fraction_spin)

        self._replicable_check = QCheckBox(i18n.t("Replicable training"))
        self._replicable_check.setChecked(True)
        form.addRow("", self._replicable_check)
        layout.addWidget(settings)

        self._status_label = QLabel(i18n.t("No dataset loaded."))
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

        self._method_combo.currentIndexChanged.connect(self._handle_method_changed)
        self._method_combo.currentIndexChanged.connect(self._schedule_auto_apply)
        self._contamination_spin.valueChanged.connect(lambda _value: self._schedule_auto_apply())
        self._neighbors_spin.valueChanged.connect(lambda _value: self._schedule_auto_apply())
        self._metric_combo.currentIndexChanged.connect(self._schedule_auto_apply)
        self._nu_spin.valueChanged.connect(lambda _value: self._schedule_auto_apply())
        self._gamma_combo.currentIndexChanged.connect(self._schedule_auto_apply)
        self._support_fraction_spin.valueChanged.connect(lambda _value: self._schedule_auto_apply())
        self._replicable_check.toggled.connect(lambda _checked: self._schedule_auto_apply())
        self.cb_apply_auto.toggled.connect(lambda _checked: self._schedule_auto_apply())
        self._handle_method_changed()

    def set_input_payload(self, payload) -> None:
        self._dataset_handle = payload.dataset if payload is not None else None
        self._outputs = {}
        self._schedule_auto_apply()

    def current_output_datasets(self) -> dict[str, DatasetHandle | None] | None:
        if not self._outputs:
            return None
        return {
            "Outliers": self._outputs.get("Outliers"),
            "Inliers": self._outputs.get("Inliers"),
            "Data": self._outputs.get("Data"),
        }

    def serialize_node_state(self) -> dict[str, object]:
        return {
            "method": self._method_combo.currentData(),
            "contamination": self._contamination_spin.value(),
            "neighbors": self._neighbors_spin.value(),
            "metric": self._metric_combo.currentData(),
            "nu": self._nu_spin.value(),
            "gamma": self._gamma_combo.currentText(),
            "support_fraction": self._support_fraction_spin.value(),
            "replicable": self._replicable_check.isChecked(),
            "auto_apply": self.cb_apply_auto.isChecked(),
        }

    def restore_node_state(self, payload: dict[str, object]) -> None:
        method_index = self._method_combo.findData(str(payload.get("method", "one_class_svm")))
        self._method_combo.setCurrentIndex(max(0, method_index))
        self._contamination_spin.setValue(float(payload.get("contamination", 0.10)))
        self._neighbors_spin.setValue(int(payload.get("neighbors", 20)))
        metric_index = self._metric_combo.findData(str(payload.get("metric", "euclidean")))
        self._metric_combo.setCurrentIndex(max(0, metric_index))
        self._nu_spin.setValue(float(payload.get("nu", 0.10)))
        gamma_index = self._gamma_combo.findText(str(payload.get("gamma", "scale")))
        self._gamma_combo.setCurrentIndex(max(0, gamma_index))
        self._support_fraction_spin.setValue(float(payload.get("support_fraction", 1.0)))
        self._replicable_check.setChecked(bool(payload.get("replicable", True)))
        self.cb_apply_auto.setChecked(bool(payload.get("auto_apply", True)))
        self._handle_method_changed()
        self._schedule_auto_apply()

    def help_text(self) -> str:
        return "Detect outliers with One-Class SVM, Covariance Estimator, Local Outlier Factor, or Isolation Forest."

    def documentation_url(self) -> str:
        return "https://orangedatamining.com/widget-catalog/unsupervised/outliers/"

    def _handle_method_changed(self) -> None:
        method = str(self._method_combo.currentData())
        self._contamination_spin.setEnabled(method != "one_class_svm")
        self._neighbors_spin.setEnabled(method == "local_outlier_factor")
        self._metric_combo.setEnabled(method == "local_outlier_factor")
        self._nu_spin.setEnabled(method == "one_class_svm")
        self._gamma_combo.setEnabled(method == "one_class_svm")
        self._support_fraction_spin.setEnabled(method == "covariance")
        self._replicable_check.setEnabled(method == "isolation_forest")

    def _apply(self) -> None:
        self._outputs = {}
        if self._dataset_handle is None:
            self._status_label.setText(i18n.t("No dataset loaded."))
            self._notify_output_changed()
            return
        try:
            self._outputs = self._service.detect(
                self._dataset_handle,
                method=str(self._method_combo.currentData()),
                contamination=float(self._contamination_spin.value()),
                neighbors=int(self._neighbors_spin.value()),
                metric=str(self._metric_combo.currentData()),
                nu=float(self._nu_spin.value()),
                gamma=str(self._gamma_combo.currentText()),
                support_fraction=float(self._support_fraction_spin.value()),
                replicable=bool(self._replicable_check.isChecked()),
            )
            outlier_count = self._outputs["Outliers"].row_count
            inlier_count = self._outputs["Inliers"].row_count
            total_count = self._outputs["Data"].row_count
            self._status_label.setText(
                i18n.tf(
                    "{total} instances, {inliers} inliers ({outliers} outliers)",
                    total=total_count,
                    inliers=inlier_count,
                    outliers=outlier_count,
                )
            )
        except Exception as exc:
            self._status_label.setText(i18n.tf("Error: {err}", err=exc))
        self._notify_output_changed()
