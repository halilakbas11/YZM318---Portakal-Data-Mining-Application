from __future__ import annotations

from PySide6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QRadioButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from portakal_app.data.models import DatasetHandle
from portakal_app.data.services.manifold_learning_service import (
    ManifoldLearningService,
    METHODS,
)
from portakal_app.models import WorkflowPayload
from portakal_app.ui.screens.node_screen import WorkflowNodeScreenSupport
from portakal_app.ui import i18n
from portakal_app.ui.shared.cards import SectionHeader

_METRICS = ["Euclidean", "Manhattan", "Cosine"]


class ManifoldLearningScreen(QWidget, WorkflowNodeScreenSupport):
    """Manifold Learning dimensionality reduction widget matching Orange layout."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._init_workflow_node_support()
        self._service = ManifoldLearningService()
        self._dataset_handle: DatasetHandle | None = None
        self._output_dataset: DatasetHandle | None = None

        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(8)

        root.addWidget(SectionHeader(
            i18n.t("Manifold Learning"),
            i18n.t("Non-linear dimensionality reduction"),
        ))

        self._info_label = QLabel(i18n.t("No data connected."))
        self._info_label.setStyleSheet("font-size: 10pt; color: gray; background: transparent;")
        root.addWidget(self._info_label)

        # ── Method Group ───────────────────────────────────────────────
        method_group = QGroupBox(i18n.t("Method"))
        method_form = QFormLayout(method_group)
        method_form.setContentsMargins(10, 12, 10, 10)
        method_form.setSpacing(8)

        self._combo_method = QComboBox()
        self._combo_method.addItems([i18n.t(m) for m in METHODS])
        self._combo_method.setToolTip(i18n.t("Select the manifold learning algorithm."))
        method_form.addRow(i18n.t("Method:"), self._combo_method)

        # Metric — shown for all methods
        self._lbl_metric = QLabel(i18n.t("Metric:"))
        self._combo_metric = QComboBox()
        self._combo_metric.addItems(_METRICS)
        method_form.addRow(self._lbl_metric, self._combo_metric)

        # ── t-SNE specific parameters ──
        self._lbl_perplexity = QLabel(i18n.t("Perplexity:"))
        self._spin_perplexity = QDoubleSpinBox()
        self._spin_perplexity.setRange(1.0, 100.0)
        self._spin_perplexity.setValue(30.0)
        self._spin_perplexity.setSingleStep(1.0)
        method_form.addRow(self._lbl_perplexity, self._spin_perplexity)

        self._lbl_early_exag = QLabel(i18n.t("Early exaggeration:"))
        self._spin_early_exag = QDoubleSpinBox()
        self._spin_early_exag.setRange(1.0, 50.0)
        self._spin_early_exag.setValue(12.0)
        self._spin_early_exag.setSingleStep(1.0)
        method_form.addRow(self._lbl_early_exag, self._spin_early_exag)

        self._lbl_lr = QLabel(i18n.t("Learning rate:"))
        self._spin_lr = QDoubleSpinBox()
        self._spin_lr.setRange(10.0, 1000.0)
        self._spin_lr.setValue(200.0)
        self._spin_lr.setSingleStep(50.0)
        method_form.addRow(self._lbl_lr, self._spin_lr)

        self._lbl_max_iter = QLabel(i18n.t("Max iterations:"))
        self._spin_max_iter = QSpinBox()
        self._spin_max_iter.setRange(100, 5000)
        self._spin_max_iter.setValue(1000)
        self._spin_max_iter.setSingleStep(100)
        method_form.addRow(self._lbl_max_iter, self._spin_max_iter)

        self._lbl_init = QLabel(i18n.t("Initialization:"))
        init_widget = QWidget()
        init_layout = QHBoxLayout(init_widget)
        init_layout.setContentsMargins(0, 0, 0, 0)
        self._radio_pca = QRadioButton(i18n.t("PCA"))
        self._radio_random = QRadioButton(i18n.t("Random"))
        self._radio_pca.setChecked(True)
        self._init_group = QButtonGroup(self)
        self._init_group.addButton(self._radio_pca, 0)
        self._init_group.addButton(self._radio_random, 1)
        init_layout.addWidget(self._radio_pca)
        init_layout.addWidget(self._radio_random)
        self._init_widget = init_widget
        method_form.addRow(self._lbl_init, init_widget)

        # ── Non-t-SNE specific parameters ──
        self._lbl_neighbors = QLabel(i18n.t("Neighbours:"))
        self._spin_neighbors = QSpinBox()
        self._spin_neighbors.setRange(2, 100)
        self._spin_neighbors.setValue(10)
        self._spin_neighbors.setToolTip(i18n.t("Number of nearest neighbours (not used by MDS)."))
        method_form.addRow(self._lbl_neighbors, self._spin_neighbors)

        # Shared
        self._spin_seed = QSpinBox()
        self._spin_seed.setRange(0, 999999)
        self._spin_seed.setValue(42)
        method_form.addRow(i18n.t("Random State:"), self._spin_seed)

        root.addWidget(method_group)

        # ── Output Group ───────────────────────────────────────────────
        output_group = QGroupBox(i18n.t("Output"))
        output_form = QFormLayout(output_group)
        output_form.setContentsMargins(10, 12, 10, 10)
        output_form.setSpacing(8)

        self._spin_components = QSpinBox()
        self._spin_components.setRange(2, 10)
        self._spin_components.setValue(2)
        output_form.addRow(i18n.t("Components:"), self._spin_components)

        root.addWidget(output_group)

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

        # Wire signals
        self._combo_method.currentIndexChanged.connect(self._on_method_changed)
        self._combo_method.currentIndexChanged.connect(self._schedule_auto_apply)
        self._combo_metric.currentIndexChanged.connect(self._schedule_auto_apply)
        for w in (self._spin_perplexity, self._spin_early_exag, self._spin_lr):
            w.valueChanged.connect(self._schedule_auto_apply)
        for w in (self._spin_max_iter, self._spin_neighbors, self._spin_seed, self._spin_components):
            w.valueChanged.connect(self._schedule_auto_apply)
        self._init_group.buttonClicked.connect(self._schedule_auto_apply)

        self._on_method_changed()

    # ── Slots ──────────────────────────────────────────────────────────

    def _set_row_visible(self, label: QLabel, field: QWidget, visible: bool) -> None:
        label.setVisible(visible)
        field.setVisible(visible)

    def _on_method_changed(self) -> None:
        method = METHODS[self._combo_method.currentIndex()]
        is_tsne = method == "t-SNE"
        is_mds = method == "MDS"

        # t-SNE specific rows
        for lbl, fld in (
            (self._lbl_perplexity, self._spin_perplexity),
            (self._lbl_early_exag, self._spin_early_exag),
            (self._lbl_lr, self._spin_lr),
            (self._lbl_max_iter, self._spin_max_iter),
            (self._lbl_init, self._init_widget),
        ):
            self._set_row_visible(lbl, fld, is_tsne)

        # Neighbours row (not for t-SNE, not for MDS)
        self._set_row_visible(
            self._lbl_neighbors, self._spin_neighbors,
            not is_tsne and not is_mds,
        )

    # ── WorkflowNodeScreenSupport ──────────────────────────────────────

    def set_input_payload(self, payload: WorkflowPayload | None) -> None:
        if payload is None:
            self._dataset_handle = None
            self._output_dataset = None
            self._info_label.setText(i18n.t("No data connected."))
            self._btn_apply.setEnabled(False)
            self._notify_output_changed()
            return

        self._dataset_handle = payload.dataset
        if self._dataset_handle:
            name = self._dataset_handle.display_name
            rows = self._dataset_handle.row_count
            cols = self._dataset_handle.column_count
            self._info_label.setText(f"{name}  ({rows} rows × {cols} cols)")
        self._btn_apply.setEnabled(True)
        self._auto_apply_on_input()

    def current_output_dataset(self) -> DatasetHandle | None:
        return self._output_dataset

    def serialize_node_state(self) -> dict[str, object]:
        return {
            "method": METHODS[self._combo_method.currentIndex()],
            "metric": self._combo_metric.currentText().lower(),
            "perplexity": self._spin_perplexity.value(),
            "early_exaggeration": self._spin_early_exag.value(),
            "tsne_learning_rate": self._spin_lr.value(),
            "max_iter": self._spin_max_iter.value(),
            "tsne_init": "pca" if self._radio_pca.isChecked() else "random",
            "n_neighbors": self._spin_neighbors.value(),
            "n_components": self._spin_components.value(),
            "random_state": self._spin_seed.value(),
            "auto_apply": self.cb_apply_auto.isChecked(),
        }

    def restore_node_state(self, payload: dict[str, object]) -> None:
        method = str(payload.get("method", "t-SNE"))
        if method in METHODS:
            self._combo_method.setCurrentIndex(METHODS.index(method))
        metric = str(payload.get("metric", "euclidean")).capitalize()
        idx = self._combo_metric.findText(metric)
        if idx >= 0:
            self._combo_metric.setCurrentIndex(idx)
        self._spin_perplexity.setValue(float(payload.get("perplexity", 30.0)))
        self._spin_early_exag.setValue(float(payload.get("early_exaggeration", 12.0)))
        self._spin_lr.setValue(float(payload.get("tsne_learning_rate", 200.0)))
        self._spin_max_iter.setValue(int(payload.get("max_iter", 1000)))
        init = str(payload.get("tsne_init", "pca"))
        self._radio_pca.setChecked(init == "pca")
        self._radio_random.setChecked(init != "pca")
        self._spin_neighbors.setValue(int(payload.get("n_neighbors", 10)))
        self._spin_components.setValue(int(payload.get("n_components", 2)))
        self._spin_seed.setValue(int(payload.get("random_state", 42)))
        self.cb_apply_auto.setChecked(bool(payload.get("auto_apply", False)))
        self._on_method_changed()

    # ── Internal ───────────────────────────────────────────────────────

    def _apply(self) -> None:
        if self._dataset_handle is None:
            return
        method = METHODS[self._combo_method.currentIndex()]
        try:
            self._status_label.setText(i18n.t(f"Running {method}…"))
            self._output_dataset = self._service.run(
                self._dataset_handle,
                method=method,
                n_components=self._spin_components.value(),
                n_neighbors=self._spin_neighbors.value(),
                random_state=self._spin_seed.value(),
                normalize=True,
                metric=self._combo_metric.currentText().lower(),
                perplexity=self._spin_perplexity.value(),
                early_exaggeration=self._spin_early_exag.value(),
                tsne_learning_rate=self._spin_lr.value(),
                max_iter=self._spin_max_iter.value(),
                tsne_init="pca" if self._radio_pca.isChecked() else "random",
            )
            info = (
                f"{self._output_dataset.row_count} rows × "
                f"{self._output_dataset.column_count} cols"
            )
            self._status_label.setText(f"✓  {info}")
        except Exception as exc:
            self._output_dataset = None
            self._status_label.setText(f"✗  {exc}")
        finally:
            self._notify_output_changed()
