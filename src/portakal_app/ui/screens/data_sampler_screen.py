from __future__ import annotations

from PySide6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QRadioButton,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)
from PySide6.QtCore import Qt

from portakal_app.data.models import DatasetHandle
from portakal_app.data.services.data_sampler_service import DataSamplerService
from portakal_app.ui import i18n
from portakal_app.ui.screens.node_screen import WorkflowNodeScreenSupport


class DataSamplerScreen(QWidget, WorkflowNodeScreenSupport):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._init_workflow_node_support()
        self._service = DataSamplerService()
        self._dataset_handle: DatasetHandle | None = None
        self._output_dataset: DatasetHandle | None = None
        self._remaining_dataset: DatasetHandle | None = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(10)

        # ── Dataset label ─────────────────────────────────────────────
        self._dataset_label = QLabel(i18n.t("Dataset: none"))
        self._dataset_label.setProperty("sectionTitle", True)
        self._dataset_label.setStyleSheet("font-size: 12pt; background: transparent;")
        layout.addWidget(self._dataset_label)

        # ── Target column badge ───────────────────────────────────────
        self._target_label = QLabel("")
        self._target_label.setStyleSheet(
            "font-size: 10pt; font-weight: bold; color: #b35c1e;"
            "background: #fff3e0; border: 1px solid #e0c8a8;"
            "border-radius: 4px; padding: 2px 8px;"
        )
        self._target_label.setVisible(False)
        layout.addWidget(self._target_label)

        # ── Sampling Type ─────────────────────────────────────────────
        mode_group = QGroupBox(i18n.t("Sampling Type"))
        mode_layout = QVBoxLayout(mode_group)
        mode_layout.setContentsMargins(10, 10, 10, 10)
        mode_layout.setSpacing(8)

        self._mode_group = QButtonGroup(self)

        # 1. Fixed proportion
        self._radio_pct = QRadioButton(i18n.t("Fixed proportion of data:"))
        self._radio_pct.setChecked(True)
        self._mode_group.addButton(self._radio_pct, 0)
        mode_layout.addWidget(self._radio_pct)

        pct_layout = QHBoxLayout()
        pct_layout.setContentsMargins(24, 0, 0, 4)
        pct_layout.setSpacing(8)
        self._pct_slider = QSlider(Qt.Orientation.Horizontal)
        self._pct_slider.setRange(1, 99)
        self._pct_slider.setValue(70)
        self._pct_slider.valueChanged.connect(self._on_pct_changed)
        pct_layout.addWidget(self._pct_slider, 1)
        self._pct_label = QLabel(i18n.tf("{value} %", value=70))
        self._pct_label.setMinimumWidth(40)
        pct_layout.addWidget(self._pct_label)
        mode_layout.addLayout(pct_layout)

        # 2. Fixed sample size
        self._radio_fixed = QRadioButton(i18n.t("Fixed sample size"))
        self._mode_group.addButton(self._radio_fixed, 1)
        mode_layout.addWidget(self._radio_fixed)

        fixed_layout = QVBoxLayout()
        fixed_layout.setContentsMargins(24, 0, 0, 4)
        fixed_layout.setSpacing(4)

        fixed_row = QHBoxLayout()
        fixed_row.setContentsMargins(0, 0, 0, 0)
        self._instances_label = QLabel(i18n.t("Instances:"))
        fixed_row.addWidget(self._instances_label)
        self._fixed_spin = QSpinBox()
        self._fixed_spin.setRange(1, 999999)
        self._fixed_spin.setValue(30)
        fixed_row.addWidget(self._fixed_spin)
        fixed_row.addStretch(1)
        fixed_layout.addLayout(fixed_row)

        self._replacement_cb = QCheckBox(i18n.t("Sample with replacement"))
        fixed_layout.addWidget(self._replacement_cb)
        mode_layout.addLayout(fixed_layout)

        # 3. Cross validation
        self._radio_cv = QRadioButton(i18n.t("Cross validation"))
        self._mode_group.addButton(self._radio_cv, 2)
        mode_layout.addWidget(self._radio_cv)

        cv_layout = QVBoxLayout()
        cv_layout.setContentsMargins(24, 0, 0, 4)
        cv_layout.setSpacing(4)

        folds_row = QHBoxLayout()
        folds_row.setContentsMargins(0, 0, 0, 0)
        self._folds_label = QLabel(i18n.t("Number of subsets:"))
        folds_row.addWidget(self._folds_label)
        self._folds_spin = QSpinBox()
        self._folds_spin.setRange(2, 20)
        self._folds_spin.setValue(5)
        self._folds_spin.valueChanged.connect(self._on_folds_changed)
        folds_row.addWidget(self._folds_spin)
        folds_row.addStretch(1)
        cv_layout.addLayout(folds_row)

        fold_row = QHBoxLayout()
        fold_row.setContentsMargins(0, 0, 0, 0)
        self._fold_label = QLabel(i18n.t("Unused subset:"))
        fold_row.addWidget(self._fold_label)
        self._fold_spin = QSpinBox()
        self._fold_spin.setRange(1, 20)
        self._fold_spin.setValue(1)
        fold_row.addWidget(self._fold_spin)
        fold_row.addStretch(1)
        cv_layout.addLayout(fold_row)

        mode_layout.addLayout(cv_layout)

        # 4. Bootstrap
        self._radio_bootstrap = QRadioButton(i18n.t("Bootstrap"))
        self._mode_group.addButton(self._radio_bootstrap, 3)
        mode_layout.addWidget(self._radio_bootstrap)

        layout.addWidget(mode_group)

        # ── Options (Reproducibility & Stratify) ──────────────────────
        options_group = QGroupBox(i18n.t("Options"))
        options_layout = QVBoxLayout(options_group)
        options_layout.setContentsMargins(10, 10, 10, 10)
        options_layout.setSpacing(8)

        self._use_seed = QCheckBox(i18n.t("Replicable (deterministic) sampling"))
        self._use_seed.setChecked(True)
        options_layout.addWidget(self._use_seed)

        self._seed_spin = QSpinBox()
        self._seed_spin.setRange(0, 999999)
        self._seed_spin.setValue(42)
        self._seed_spin.setVisible(False)

        self._stratify_cb = QCheckBox(i18n.t("Stratify sample (when possible)"))
        options_layout.addWidget(self._stratify_cb)

        layout.addWidget(options_group)

        # ── Result info ───────────────────────────────────────────────
        self._result_label = QLabel("")
        self._result_label.setWordWrap(True)
        layout.addWidget(self._result_label)

        # ── Row counts (dynamic) ──────────────────────────────────────
        counts_layout = QHBoxLayout()
        counts_layout.setContentsMargins(0, 0, 0, 0)
        counts_layout.setSpacing(16)
        self._input_count_label = QLabel("")
        self._input_count_label.setStyleSheet("font-size: 9pt; color: #6b5d50; background: transparent;")
        counts_layout.addWidget(self._input_count_label)
        self._sample_count_label = QLabel("")
        self._sample_count_label.setStyleSheet("font-size: 9pt; color: #2e7d32; background: transparent;")
        counts_layout.addWidget(self._sample_count_label)
        self._remaining_count_label = QLabel("")
        self._remaining_count_label.setStyleSheet("font-size: 9pt; color: #c75000; background: transparent;")
        counts_layout.addWidget(self._remaining_count_label)
        counts_layout.addStretch(1)
        layout.addLayout(counts_layout)

        layout.addStretch(1)

        # Auto-apply hooks
        self._mode_group.idClicked.connect(self._check_auto_apply)
        self._pct_slider.valueChanged.connect(self._check_auto_apply)
        self._fixed_spin.valueChanged.connect(self._check_auto_apply)
        self._folds_spin.valueChanged.connect(self._check_auto_apply)
        self._fold_spin.valueChanged.connect(self._check_auto_apply)
        self._replacement_cb.stateChanged.connect(self._check_auto_apply)
        self._stratify_cb.stateChanged.connect(self._check_auto_apply)
        self._use_seed.stateChanged.connect(self._check_auto_apply)
        self._seed_spin.valueChanged.connect(self._check_auto_apply)

        footer = QHBoxLayout()
        self.cb_apply_auto = QCheckBox(i18n.t("Apply Automatically"))
        self.cb_apply_auto.setChecked(True)
        footer.addWidget(self.cb_apply_auto)
        footer.addStretch(1)

        self._apply_button = QPushButton(i18n.t("Sample Data"))
        self._apply_button.setProperty("primary", True)
        self._apply_button.clicked.connect(self._apply)
        footer.addWidget(self._apply_button)
        layout.addLayout(footer)

        # ── Wire mode-change signal ───────────────────────────────────
        self._mode_group.idToggled.connect(self._on_mode_changed)
        # Set initial enable/disable state
        self._on_mode_changed(0, True)

    def _check_auto_apply(self):
        if self.cb_apply_auto.isChecked():
            self._apply()

    # ── input / output ────────────────────────────────────────────────

    def set_input_payload(self, payload) -> None:
        dataset = payload.dataset if payload is not None else None
        self._dataset_handle = dataset
        self._output_dataset = None
        self._remaining_dataset = None
        if dataset:
            self._dataset_label.setText(
                i18n.tf("Dataset: {name}", name=dataset.display_name)
            )
            # Update max for fixed-size spin
            if not self._replacement_cb.isChecked():
                self._fixed_spin.setMaximum(dataset.row_count)
            # Show target column badge
            target_cols = dataset.domain.target_columns if dataset.domain else ()
            if target_cols:
                self._target_label.setText(
                    i18n.tf("Target: {col}", col=target_cols[0].name)
                )
                self._target_label.setVisible(True)
            else:
                self._target_label.setVisible(False)
            # Dynamic input row count
            self._input_count_label.setText(
                i18n.tf("Input: {n} rows", n=dataset.row_count)
            )
            self._sample_count_label.setText("")
            self._remaining_count_label.setText("")
        else:
            self._dataset_label.setText(i18n.t("Dataset: none"))
            self._target_label.setVisible(False)
            self._result_label.setText("")
            self._input_count_label.setText("")
            self._sample_count_label.setText("")
            self._remaining_count_label.setText("")
        self._apply()

    def current_output_dataset(self) -> DatasetHandle | None:
        return self._output_dataset

    def current_output_datasets(self) -> dict[str, DatasetHandle | None] | None:
        return {
            "Data Sample": self._output_dataset,
            "Remaining Data": self._remaining_dataset,
        }

    # ── serialize / restore ───────────────────────────────────────────

    def serialize_node_state(self) -> dict[str, object]:
        return {
            "mode": self._mode_group.checkedId(),
            "percentage": self._pct_slider.value(),
            "fixed_size": self._fixed_spin.value(),
            "folds": self._folds_spin.value(),
            "selected_fold": self._fold_spin.value(),
            "replacement": self._replacement_cb.isChecked(),
            "stratify": self._stratify_cb.isChecked(),
            "use_seed": self._use_seed.isChecked(),
            "seed": self._seed_spin.value(),
            "auto_apply": self.cb_apply_auto.isChecked(),
        }

    def restore_node_state(self, payload: dict[str, object]) -> None:
        mode_id = int(payload.get("mode", 0))
        btn = self._mode_group.button(mode_id)
        if btn:
            btn.setChecked(True)
        self._pct_slider.setValue(int(payload.get("percentage", 70)))
        self._fixed_spin.setValue(int(payload.get("fixed_size", 10)))
        self._folds_spin.setValue(int(payload.get("folds", 10)))
        self._fold_spin.setValue(int(payload.get("selected_fold", 1)))
        self._replacement_cb.setChecked(bool(payload.get("replacement", False)))
        self._stratify_cb.setChecked(bool(payload.get("stratify", False)))
        self._use_seed.setChecked(bool(payload.get("use_seed", True)))
        self._seed_spin.setValue(int(payload.get("seed", 42)))
        self.cb_apply_auto.setChecked(bool(payload.get("auto_apply", True)))
        self._on_mode_changed(mode_id, True)

    def help_text(self) -> str:
        return "Sample a subset of the data using fixed proportion, fixed size, cross-validation, or bootstrap."

    def documentation_url(self) -> str:
        return "https://orangedatamining.com/widget-catalog/transform/datasampler/"

    # ── callbacks ─────────────────────────────────────────────────────

    def _on_pct_changed(self, value: int) -> None:
        self._pct_label.setText(i18n.tf("{value} %", value=value))

    def _on_folds_changed(self, value: int) -> None:
        self._fold_spin.setMaximum(value)
        if self._fold_spin.value() > value:
            self._fold_spin.setValue(value)

    def _on_mode_changed(self, mode_id: int, checked: bool) -> None:
        """Enable / disable controls based on the selected sampling mode."""
        if not checked:
            return

        is_pct = mode_id == 0
        is_fixed = mode_id == 1
        is_cv = mode_id == 2
        is_bootstrap = mode_id == 3

        # Percentage controls
        self._pct_slider.setEnabled(is_pct)
        self._pct_label.setEnabled(is_pct)

        # Fixed-size controls
        self._fixed_spin.setEnabled(is_fixed)
        self._instances_label.setEnabled(is_fixed)
        self._replacement_cb.setEnabled(is_fixed)

        # Cross-validation controls
        self._folds_spin.setEnabled(is_cv)
        self._folds_label.setEnabled(is_cv)
        self._fold_spin.setEnabled(is_cv)
        self._fold_label.setEnabled(is_cv)

        # Stratify makes no sense for bootstrap
        self._stratify_cb.setEnabled(not is_bootstrap)

    def _apply(self) -> None:
        if self._dataset_handle is None:
            self._output_dataset = None
            self._remaining_dataset = None
            self._result_label.setText("")
            self._sample_count_label.setText("")
            self._remaining_count_label.setText("")
            self._notify_output_changed()
            return

        modes = {0: "percentage", 1: "fixed", 2: "cross-validation", 3: "bootstrap"}
        mode = modes.get(self._mode_group.checkedId(), "percentage")
        seed = self._seed_spin.value() if self._use_seed.isChecked() else None

        sample, remaining = self._service.sample(
            self._dataset_handle,
            mode=mode,
            percentage=self._pct_slider.value(),
            fixed_size=self._fixed_spin.value(),
            folds=self._folds_spin.value(),
            selected_fold=self._fold_spin.value(),
            with_replacement=self._replacement_cb.isChecked(),
            stratify=self._stratify_cb.isChecked(),
            seed=seed,
        )

        self._output_dataset = sample
        self._remaining_dataset = remaining

        s_count = sample.row_count if sample else 0
        r_count = remaining.row_count if remaining else 0

        # Dynamic row count labels
        self._sample_count_label.setText(
            i18n.tf("Sample: {n} rows", n=s_count)
        )
        self._remaining_count_label.setText(
            i18n.tf("Remaining: {n} rows", n=r_count)
        )
        self._result_label.setText(
            i18n.tf(
                "Sample: {sample} rows  |  Remaining: {remaining} rows",
                sample=s_count,
                remaining=r_count,
            )
        )
        self._notify_output_changed()

    # ── i18n ──────────────────────────────────────────────────────────

    def refresh_translations(self) -> None:
        if self._dataset_handle:
            self._dataset_label.setText(
                i18n.tf("Dataset: {name}", name=self._dataset_handle.display_name)
            )
            target_cols = (
                self._dataset_handle.domain.target_columns
                if self._dataset_handle.domain
                else ()
            )
            if target_cols:
                self._target_label.setText(
                    i18n.tf("Target: {col}", col=target_cols[0].name)
                )
        else:
            self._dataset_label.setText(i18n.t("Dataset: none"))
        self._pct_label.setText(
            i18n.tf("{value} %", value=self._pct_slider.value())
        )
        if self._output_dataset is not None:
            s_count = self._output_dataset.row_count
            r_count = (
                self._remaining_dataset.row_count if self._remaining_dataset else 0
            )
            self._result_label.setText(
                i18n.tf(
                    "Sample: {sample} rows  |  Remaining: {remaining} rows",
                    sample=s_count,
                    remaining=r_count,
                )
            )
            self._sample_count_label.setText(
                i18n.tf("Sample: {n} rows", n=s_count)
            )
            self._remaining_count_label.setText(
                i18n.tf("Remaining: {n} rows", n=r_count)
            )
