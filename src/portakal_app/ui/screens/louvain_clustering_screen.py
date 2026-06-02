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
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)
from PySide6.QtCore import Qt

from portakal_app.data.models import DatasetHandle
from portakal_app.data.services.louvain_clustering_service import LouvainClusteringService
from portakal_app.models import WorkflowPayload
from portakal_app.ui.screens.node_screen import WorkflowNodeScreenSupport
from portakal_app.ui import i18n
from portakal_app.ui.shared.cards import SectionHeader


class LouvainClusteringScreen(QWidget, WorkflowNodeScreenSupport):
    """Louvain community-detection clustering widget."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._init_workflow_node_support()
        self._service = LouvainClusteringService()
        self._dataset_handle: DatasetHandle | None = None
        self._output_dataset: DatasetHandle | None = None

        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(8)

        root.addWidget(SectionHeader(
            i18n.t("Louvain Clustering"),
            i18n.t("Graph-based community detection via the Louvain algorithm"),
        ))

        # ── Info ───────────────────────────────────────────────────────
        info_group = QGroupBox(i18n.t("Info"))
        info_layout = QVBoxLayout(info_group)
        info_layout.setContentsMargins(10, 10, 10, 10)
        self._info_label = QLabel(i18n.t("No data on input."))
        self._info_label.setStyleSheet("font-size: 10pt; color: gray; background: transparent;")
        info_layout.addWidget(self._info_label)
        root.addWidget(info_group)

        # ── Preprocessing ──────────────────────────────────────────────
        pre_group = QGroupBox(i18n.t("Preprocessing"))
        pre_layout = QVBoxLayout(pre_group)
        pre_layout.setContentsMargins(10, 10, 10, 10)
        pre_layout.setSpacing(6)

        self._cb_normalize = QCheckBox(i18n.t("Normalize data"))
        self._cb_normalize.setChecked(True)
        pre_layout.addWidget(self._cb_normalize)

        self._cb_pca = QCheckBox(i18n.t("Apply PCA preprocessing"))
        self._cb_pca.setChecked(True)
        pre_layout.addWidget(self._cb_pca)

        pca_row = QWidget()
        pca_row_layout = QHBoxLayout(pca_row)
        pca_row_layout.setContentsMargins(16, 0, 0, 0)
        pca_row_layout.setSpacing(6)
        pca_row_layout.addWidget(QLabel(i18n.t("PCA Components:")))
        self._slider_pca = QSlider(Qt.Orientation.Horizontal)
        self._slider_pca.setRange(2, 50)
        self._slider_pca.setValue(10)
        self._label_pca_val = QLabel("10")
        self._label_pca_val.setFixedWidth(24)
        self._slider_pca.valueChanged.connect(lambda v: self._label_pca_val.setText(str(v)))
        pca_row_layout.addWidget(self._slider_pca)
        pca_row_layout.addWidget(self._label_pca_val)
        pre_layout.addWidget(pca_row)

        self._cb_pca.stateChanged.connect(lambda s: pca_row.setEnabled(bool(s)))

        root.addWidget(pre_group)

        # ── Graph Parameters ───────────────────────────────────────────
        graph_group = QGroupBox(i18n.t("Graph parameters"))
        graph_form = QFormLayout(graph_group)
        graph_form.setContentsMargins(10, 12, 10, 10)
        graph_form.setSpacing(8)

        self._combo_metric = QComboBox()
        self._combo_metric.addItems(["Euclidean", "Manhattan", "Cosine"])
        graph_form.addRow(i18n.t("Distance metric:"), self._combo_metric)

        self._spin_neighbors = QSpinBox()
        self._spin_neighbors.setRange(2, 200)
        self._spin_neighbors.setValue(30)
        self._spin_neighbors.setToolTip(
            i18n.t("Number of nearest neighbours used to build the k-NN graph.")
        )
        graph_form.addRow(i18n.t("k Neighbors:"), self._spin_neighbors)

        res_row = QWidget()
        res_row_layout = QHBoxLayout(res_row)
        res_row_layout.setContentsMargins(0, 0, 0, 0)
        res_row_layout.setSpacing(6)
        self._slider_resolution = QSlider(Qt.Orientation.Horizontal)
        self._slider_resolution.setRange(1, 50)
        self._slider_resolution.setValue(10)
        self._label_res_val = QLabel("1.0")
        self._label_res_val.setFixedWidth(30)
        self._slider_resolution.valueChanged.connect(
            lambda v: self._label_res_val.setText(f"{v / 10:.1f}")
        )
        res_row_layout.addWidget(self._slider_resolution)
        res_row_layout.addWidget(self._label_res_val)
        graph_form.addRow(i18n.t("Resolution:"), res_row)

        self._spin_seed = QSpinBox()
        self._spin_seed.setRange(0, 999999)
        self._spin_seed.setValue(42)
        graph_form.addRow(i18n.t("Random State:"), self._spin_seed)

        root.addWidget(graph_group)

        # ── Apply ──────────────────────────────────────────────────────
        self.cb_apply_auto = QCheckBox(i18n.t("Apply Automatically"))
        self.cb_apply_auto.setChecked(False)
        self.cb_apply_auto.stateChanged.connect(self._schedule_auto_apply)
        root.addWidget(self.cb_apply_auto)

        self._btn_apply = QPushButton(i18n.t("Apply"))
        self._btn_apply.setEnabled(False)
        self._btn_apply.clicked.connect(self._apply)
        root.addWidget(self._btn_apply)

        self._status_label = QLabel("")
        self._status_label.setStyleSheet("color: gray; background: transparent;")
        self._status_label.setWordWrap(True)
        root.addWidget(self._status_label)

        root.addStretch(1)

        for w in (self._spin_neighbors, self._spin_seed):
            w.valueChanged.connect(self._schedule_auto_apply)
        self._slider_resolution.valueChanged.connect(self._schedule_auto_apply)
        self._slider_pca.valueChanged.connect(self._schedule_auto_apply)
        self._combo_metric.currentIndexChanged.connect(self._schedule_auto_apply)
        for w in (self._cb_normalize, self._cb_pca):
            w.stateChanged.connect(self._schedule_auto_apply)

    # ── WorkflowNodeScreenSupport ──────────────────────────────────────

    def set_input_payload(self, payload: WorkflowPayload | None) -> None:
        if payload is None:
            self._dataset_handle = None
            self._output_dataset = None
            self._info_label.setText(i18n.t("No data on input."))
            self._btn_apply.setEnabled(False)
            self._notify_output_changed()
            return

        self._dataset_handle = payload.dataset
        name = self._dataset_handle.display_name if self._dataset_handle else "–"
        rows = self._dataset_handle.row_count if self._dataset_handle else 0
        cols = self._dataset_handle.column_count if self._dataset_handle else 0
        self._info_label.setText(f"{name}  ({rows} rows × {cols} cols)")
        self._btn_apply.setEnabled(True)
        self._auto_apply_on_input()

    def current_output_dataset(self) -> DatasetHandle | None:
        return self._output_dataset

    def serialize_node_state(self) -> dict[str, object]:
        return {
            "normalize": self._cb_normalize.isChecked(),
            "apply_pca": self._cb_pca.isChecked(),
            "pca_components": self._slider_pca.value(),
            "metric": self._combo_metric.currentText().lower(),
            "n_neighbors": self._spin_neighbors.value(),
            "resolution": self._slider_resolution.value() / 10.0,
            "random_state": self._spin_seed.value(),
            "auto_apply": self.cb_apply_auto.isChecked(),
        }

    def restore_node_state(self, payload: dict[str, object]) -> None:
        self._cb_normalize.setChecked(bool(payload.get("normalize", True)))
        self._cb_pca.setChecked(bool(payload.get("apply_pca", True)))
        self._slider_pca.setValue(int(payload.get("pca_components", 10)))
        metric = str(payload.get("metric", "euclidean")).capitalize()
        idx = self._combo_metric.findText(metric)
        if idx >= 0:
            self._combo_metric.setCurrentIndex(idx)
        self._spin_neighbors.setValue(int(payload.get("n_neighbors", 30)))
        self._slider_resolution.setValue(int(float(payload.get("resolution", 1.0)) * 10))
        self._spin_seed.setValue(int(payload.get("random_state", 42)))
        self.cb_apply_auto.setChecked(bool(payload.get("auto_apply", False)))

    # ── Internal ───────────────────────────────────────────────────────

    def _apply(self) -> None:
        if self._dataset_handle is None:
            return
        try:
            self._status_label.setText(i18n.t("Running Louvain…"))
            self._output_dataset = self._service.run(
                self._dataset_handle,
                n_neighbors=self._spin_neighbors.value(),
                resolution=self._slider_resolution.value() / 10.0,
                random_state=self._spin_seed.value(),
                normalize=self._cb_normalize.isChecked(),
                apply_pca=self._cb_pca.isChecked(),
                pca_components=self._slider_pca.value(),
                metric=self._combo_metric.currentText().lower(),
            )
            n_clusters = self._output_dataset.dataframe["louvain_cluster"].n_unique()
            self._info_label.setText(f"{n_clusters} {i18n.t('clusters found.')}")
            self._status_label.setText(
                f"✓  {n_clusters} clusters  ·  {self._output_dataset.row_count} rows"
            )
        except Exception as exc:
            self._output_dataset = None
            self._status_label.setText(f"✗  {exc}")
        finally:
            self._notify_output_changed()
