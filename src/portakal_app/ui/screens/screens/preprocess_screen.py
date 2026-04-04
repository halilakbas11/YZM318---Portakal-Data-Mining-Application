from __future__ import annotations

from typing import Any
from PySide6.QtWidgets import (
    QCheckBox,
    QDoubleSpinBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
    QListWidget,
    QRadioButton,
    QButtonGroup,
    QScrollArea,
    QComboBox,
    QSpinBox,
    QFormLayout,
)
from PySide6.QtCore import Qt

from portakal_app.data.models import DatasetHandle
from portakal_app.data.services.preprocess_service import PreprocessService, PreprocessStep
from portakal_app.ui.screens.node_screen import WorkflowNodeScreenSupport
from portakal_app.ui import i18n


class StepEditor(QGroupBox):
    def __init__(self, step_name: str, parent: QWidget | None = None) -> None:
        super().__init__(step_name, parent)
        self.step_name = step_name
        self.layout = QVBoxLayout(self)
        
        # Adding a close button
        header_layout = QHBoxLayout()
        header_layout.addStretch()
        self.btn_remove = QPushButton("X")
        self.btn_remove.setFixedSize(20, 20)
        self.btn_remove.setStyleSheet("color: red; font-weight: bold; border: none;")
        header_layout.addWidget(self.btn_remove)
        self.layout.addLayout(header_layout)

    def parameters(self) -> dict[str, Any]:
        return {}

    def set_parameters(self, params: dict[str, Any]) -> None:
        pass


class ContinuizeEditor(StepEditor):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(i18n.t("Continuize Discrete Variables"), parent)
        self.group = QButtonGroup(self)
        self.methods = [
            "Most frequent is base",
            "One feature per value",
            "Remove non-binary features",
            "Remove categorical features",
            "Treat as ordinal",
            "Divide by number of values"
        ]
        for idx, text in enumerate(self.methods):
            rb = QRadioButton(i18n.t(text))
            if text == "One feature per value":
                rb.setChecked(True)
            self.group.addButton(rb, idx)
            self.layout.addWidget(rb)

    def parameters(self) -> dict[str, Any]:
        return {"method": self.methods[self.group.checkedId()]}

    def set_parameters(self, params: dict[str, Any]) -> None:
        method = params.get("method", "One feature per value")
        for idx, m in enumerate(self.methods):
            if m == method:
                self.group.button(idx).setChecked(True)


class ImputeEditor(StepEditor):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(i18n.t("Impute Missing Values"), parent)
        self.group = QButtonGroup(self)
        self.methods = [
            "Average/Most frequent",
            "Replace with random value",
            "Remove rows with missing values"
        ]
        for idx, text in enumerate(self.methods):
            rb = QRadioButton(i18n.t(text))
            if idx == 0:
                rb.setChecked(True)
            self.group.addButton(rb, idx)
            self.layout.addWidget(rb)

    def parameters(self) -> dict[str, Any]:
        return {"method": self.methods[self.group.checkedId()]}

    def set_parameters(self, params: dict[str, Any]) -> None:
        method = params.get("method", "Average/Most frequent")
        for idx, m in enumerate(self.methods):
            if m == method:
                self.group.button(idx).setChecked(True)


class NormalizeEditor(StepEditor):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(i18n.t("Normalize Features"), parent)
        self.group = QButtonGroup(self)
        self.methods = [
            "Standardize to μ=0, σ²=1",
            "Center to μ=0",
            "Scale to σ²=1",
            "Normalize to interval [-1, 1]",
            "Normalize to interval [0, 1]"
        ]
        for idx, text in enumerate(self.methods):
            rb = QRadioButton(i18n.t(text))
            if idx == 0:
                rb.setChecked(True)
            self.group.addButton(rb, idx)
            self.layout.addWidget(rb)

    def parameters(self) -> dict[str, Any]:
        return {"method": self.methods[self.group.checkedId()]}

    def set_parameters(self, params: dict[str, Any]) -> None:
        method = params.get("method", "Standardize to μ=0, σ²=1")
        for idx, m in enumerate(self.methods):
            if m == method:
                self.group.button(idx).setChecked(True)


class FeatureSelectEditor(StepEditor):
    def __init__(self, parent: QWidget | None = None) -> None:
        # Score, Fixed/proportion
        super().__init__(i18n.t("Select Relevant Features"), parent)
        box = QGroupBox(i18n.t("Score"), flat=True)
        box_lyt = QVBoxLayout(box)
        self.score_cb = QComboBox()
        self.score_cb.addItems(["Information Gain", "Gain Ratio", "Gini Index", "ReliefF", "ANOVA", "Chi2", "Univariate Linear Regression"])
        box_lyt.addWidget(self.score_cb)
        self.layout.addWidget(box)

        box2 = QGroupBox(i18n.t("Number of features"), flat=True)
        form = QFormLayout(box2)
        self.group = QButtonGroup(self)

        self.rb_fixed = QRadioButton(i18n.t("Fixed:"))
        self.rb_fixed.setChecked(True)
        self.spin_fixed = QSpinBox()
        self.spin_fixed.setRange(1, 100000)
        self.spin_fixed.setValue(10)
        form.addRow(self.rb_fixed, self.spin_fixed)

        self.rb_prop = QRadioButton(i18n.t("Proportion:"))
        self.spin_prop = QDoubleSpinBox()
        self.spin_prop.setRange(1, 100)
        self.spin_prop.setValue(75.0)
        self.spin_prop.setSuffix("%")
        form.addRow(self.rb_prop, self.spin_prop)
        
        self.group.addButton(self.rb_fixed, 1)
        self.group.addButton(self.rb_prop, 2)
        self.layout.addWidget(box2)

    def parameters(self) -> dict[str, Any]:
        strategy = "Fixed" if self.rb_fixed.isChecked() else "Proportion"
        k = self.spin_fixed.value() if strategy == "Fixed" else self.spin_prop.value()
        return {"score": self.score_cb.currentText(), "strategy": strategy, "k": k}

    def set_parameters(self, params: dict[str, Any]) -> None:
        self.score_cb.setCurrentText(params.get("score", "Information Gain"))
        strategy = params.get("strategy", "Fixed")
        if strategy == "Fixed":
            self.rb_fixed.setChecked(True)
            self.spin_fixed.setValue(int(params.get("k", 10)))
        else:
            self.rb_prop.setChecked(True)
            self.spin_prop.setValue(float(params.get("k", 75.0)))


class RandomSelectEditor(StepEditor):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(i18n.t("Select Random Features"), parent)
        box2 = QGroupBox(i18n.t("Number of features"), flat=True)
        form = QFormLayout(box2)
        self.group = QButtonGroup(self)

        self.rb_fixed = QRadioButton(i18n.t("Fixed:"))
        self.rb_fixed.setChecked(True)
        self.spin_fixed = QSpinBox()
        self.spin_fixed.setRange(1, 100000)
        self.spin_fixed.setValue(10)
        form.addRow(self.rb_fixed, self.spin_fixed)

        self.rb_prop = QRadioButton(i18n.t("Percentage:"))
        self.spin_prop = QDoubleSpinBox()
        self.spin_prop.setRange(1, 100)
        self.spin_prop.setValue(75.0)
        self.spin_prop.setSuffix("%")
        form.addRow(self.rb_prop, self.spin_prop)
        
        self.group.addButton(self.rb_fixed, 1)
        self.group.addButton(self.rb_prop, 2)
        self.layout.addWidget(box2)

    def parameters(self) -> dict[str, Any]:
        strategy = "Fixed" if self.rb_fixed.isChecked() else "Percentage"
        k = self.spin_fixed.value() if strategy == "Fixed" else self.spin_prop.value()
        return {"strategy": strategy, "k": k}

    def set_parameters(self, params: dict[str, Any]) -> None:
        strategy = params.get("strategy", "Fixed")
        if strategy == "Fixed":
            self.rb_fixed.setChecked(True)
            self.spin_fixed.setValue(int(params.get("k", 10)))
        else:
            self.rb_prop.setChecked(True)
            self.spin_prop.setValue(float(params.get("k", 75.0)))


class RandomizeEditor(StepEditor):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(i18n.t("Randomize"), parent)
        self.cb_classes = QCheckBox(i18n.t("Classes"))
        self.cb_classes.setChecked(True)
        self.cb_features = QCheckBox(i18n.t("Features"))
        self.cb_meta = QCheckBox(i18n.t("Meta data"))
        self.layout.addWidget(self.cb_classes)
        self.layout.addWidget(self.cb_features)
        self.layout.addWidget(self.cb_meta)

    def parameters(self) -> dict[str, Any]:
        return {
            "classes": self.cb_classes.isChecked(),
            "features": self.cb_features.isChecked(),
            "meta": self.cb_meta.isChecked()
        }

    def set_parameters(self, params: dict[str, Any]) -> None:
        self.cb_classes.setChecked(bool(params.get("classes", True)))
        self.cb_features.setChecked(bool(params.get("features", False)))
        self.cb_meta.setChecked(bool(params.get("meta", False)))


class RemoveSparseEditor(StepEditor):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(i18n.t("Remove Sparse Features"), parent)
        self.layout.addWidget(QLabel(i18n.t("Remove features with too many")))
        self.filter_group = QButtonGroup(self)
        self.rb_missing = QRadioButton(i18n.t("missing values"))
        self.rb_missing.setChecked(True)
        self.rb_zeroes = QRadioButton(i18n.t("zeros"))
        self.filter_group.addButton(self.rb_missing, 0)
        self.filter_group.addButton(self.rb_zeroes, 1)
        self.layout.addWidget(self.rb_missing)
        self.layout.addWidget(self.rb_zeroes)

        box2 = QGroupBox(i18n.t("Threshold:"), flat=True)
        form = QFormLayout(box2)
        self.thresh_group = QButtonGroup(self)

        self.rb_perc = QRadioButton(i18n.t("Percentage:"))
        self.rb_perc.setChecked(True)
        self.spin_perc = QSpinBox()
        self.spin_perc.setRange(0, 100)
        self.spin_perc.setValue(5)
        form.addRow(self.rb_perc, self.spin_perc)

        self.rb_fixed = QRadioButton(i18n.t("Fixed:"))
        self.spin_fixed = QSpinBox()
        self.spin_fixed.setRange(0, 100000)
        self.spin_fixed.setValue(50)
        form.addRow(self.rb_fixed, self.spin_fixed)
        
        self.thresh_group.addButton(self.rb_perc, 0)
        self.thresh_group.addButton(self.rb_fixed, 1)
        self.layout.addWidget(box2)

    def parameters(self) -> dict[str, Any]:
        return {
            "filter0": self.filter_group.checkedId(),
            "useFixedThreshold": bool(self.rb_fixed.isChecked()),
            "percThresh": self.spin_perc.value(),
            "fixedThresh": self.spin_fixed.value(),
        }

    def set_parameters(self, params: dict[str, Any]) -> None:
        filter0 = int(params.get("filter0", 0))
        self.rb_missing.setChecked(filter0 == 0)
        self.rb_zeroes.setChecked(filter0 == 1)

        use_fixed = bool(params.get("useFixedThreshold", False))
        self.rb_fixed.setChecked(use_fixed)
        self.rb_perc.setChecked(not use_fixed)
        self.spin_perc.setValue(int(params.get("percThresh", 5)))
        self.spin_fixed.setValue(int(params.get("fixedThresh", 50)))


class PCAEditor(StepEditor):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(i18n.t("Principal Component Analysis"), parent)
        form = QFormLayout()
        self.spin_n = QSpinBox()
        self.spin_n.setRange(1, 10000)
        self.spin_n.setValue(5)
        form.addRow(QLabel(i18n.t("Components:")), self.spin_n)
        self.layout.addLayout(form)

    def parameters(self) -> dict[str, Any]:
        return {"n_components": self.spin_n.value()}

    def set_parameters(self, params: dict[str, Any]) -> None:
        self.spin_n.setValue(int(params.get("n_components", 5)))


class CUREditor(StepEditor):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(i18n.t("CUR Matrix Decomposition"), parent)
        box2 = QGroupBox(i18n.t("Target rank"), flat=True)
        form = QFormLayout(box2)
        self.group = QButtonGroup(self)

        self.rb_fixed = QRadioButton(i18n.t("Fixed:"))
        self.rb_fixed.setChecked(True)
        self.spin_fixed = QSpinBox()
        self.spin_fixed.setRange(1, 100000)
        self.spin_fixed.setValue(10)
        form.addRow(self.rb_fixed, self.spin_fixed)

        self.rb_prop = QRadioButton(i18n.t("Percentage:"))
        self.spin_prop = QDoubleSpinBox()
        self.spin_prop.setRange(1, 100)
        self.spin_prop.setValue(75.0)
        self.spin_prop.setSuffix("%")
        form.addRow(self.rb_prop, self.spin_prop)
        
        self.group.addButton(self.rb_fixed, 1)
        self.group.addButton(self.rb_prop, 2)
        self.layout.addWidget(box2)

        err_form = QFormLayout()
        self.spin_max_error = QDoubleSpinBox()
        self.spin_max_error.setRange(0.01, 100.0)
        self.spin_max_error.setValue(1.0)
        self.spin_max_error.setSingleStep(0.1)
        err_form.addRow(QLabel(i18n.t("Relative error:")), self.spin_max_error)
        self.layout.addLayout(err_form)

    def parameters(self) -> dict[str, Any]:
        strategy = "Fixed" if self.rb_fixed.isChecked() else "Percentage"
        k = self.spin_fixed.value() if strategy == "Fixed" else self.spin_prop.value()
        return {"strategy": strategy, "k": k, "max_error": self.spin_max_error.value()}

    def set_parameters(self, params: dict[str, Any]) -> None:
        strategy = params.get("strategy", "Fixed")
        if strategy == "Fixed":
            self.rb_fixed.setChecked(True)
            self.spin_fixed.setValue(int(params.get("k", 10)))
        else:
            self.rb_prop.setChecked(True)
            self.spin_prop.setValue(float(params.get("k", 75.0)))
        self.spin_max_error.setValue(float(params.get("max_error", 1.0)))


def create_editor(step_name: str) -> StepEditor:
    mapping = {
        "Continuize Discrete Variables": ContinuizeEditor,
        "Impute Missing Values": ImputeEditor,
        "Normalize Features": NormalizeEditor,
        "Select Relevant Features": FeatureSelectEditor,
        "Select Random Features": RandomSelectEditor,
        "Randomize": RandomizeEditor,
        "Remove Sparse Features": RemoveSparseEditor,
        "Principal Component Analysis": PCAEditor,
        "CUR Matrix Decomposition": CUREditor,
    }
    if step_name in mapping:
        return mapping[step_name]()
    return StepEditor(step_name)


class PreprocessScreen(QWidget, WorkflowNodeScreenSupport):
    AVAILABLE_PREPROCESSORS = [
        "Continuize Discrete Variables",
        "Impute Missing Values",
        "Select Relevant Features",
        "Select Random Features",
        "Normalize Features",
        "Randomize",
        "Remove Sparse Features",
        "Principal Component Analysis",
        "CUR Matrix Decomposition",
    ]

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._init_workflow_node_support()
        self._service = PreprocessService()
        self._dataset_handle: DatasetHandle | None = None
        self._output_dataset: DatasetHandle | None = None
        
        self._active_editors: list[StepEditor] = []

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(10)

        self._dataset_label = QLabel(i18n.t("Dataset: none"))
        self._dataset_label.setProperty("sectionTitle", True)
        self._dataset_label.setStyleSheet("font-size: 12pt; background: transparent;")
        layout.addWidget(self._dataset_label)

        content_layout = QHBoxLayout()

        # Left Panel (List of available preprocessors)
        left_panel = QVBoxLayout()
        left_panel.addWidget(QLabel(i18n.t("Preprocessors")))
        self._list_widget = QListWidget()
        self._list_widget.addItems(self.AVAILABLE_PREPROCESSORS)
        self._list_widget.itemDoubleClicked.connect(self._add_preprocessor_from_list)
        left_panel.addWidget(self._list_widget)
        
        btn_add = QPushButton(">")
        btn_add.clicked.connect(self._add_selected_preprocessor)
        
        content_layout.addLayout(left_panel, 1)
        content_layout.addWidget(btn_add)

        # Right Panel (Active pipeline pipeline steps)
        self._scroll_area = QScrollArea()
        self._scroll_area.setWidgetResizable(True)
        self._scroll_content = QWidget()
        self._pipeline_layout = QVBoxLayout(self._scroll_content)
        self._pipeline_layout.setAlignment(Qt.AlignTop)
        self._scroll_area.setWidget(self._scroll_content)
        
        content_layout.addWidget(self._scroll_area, 2)
        layout.addLayout(content_layout)

        self._result_label = QLabel("")
        self._result_label.setWordWrap(True)
        layout.addWidget(self._result_label)

        footer = QHBoxLayout()
        self.cb_apply_auto = QCheckBox(i18n.t("Apply Automatically"))
        self.cb_apply_auto.setChecked(False)
        self.cb_apply_auto.toggled.connect(lambda _: self._check_auto_apply())
        footer.addWidget(self.cb_apply_auto)
        footer.addStretch(1)

        self._apply_button = QPushButton(i18n.t("Apply"))
        self._apply_button.setProperty("primary", True)
        self._apply_button.clicked.connect(self._apply)
        footer.addWidget(self._apply_button)
        layout.addLayout(footer)

    def _add_preprocessor_from_list(self, item) -> None:
        self._append_step(item.text())

    def _add_selected_preprocessor(self) -> None:
        for item in self._list_widget.selectedItems():
            self._append_step(item.text())

    def _append_step(self, step_name: str, params: dict[str, Any] | None = None) -> None:
        editor = create_editor(step_name)
        if params is not None:
            editor.set_parameters(params)
        
        # Listen for close
        editor.btn_remove.clicked.connect(lambda: self._remove_step(editor))
        
        # Listen for param changes for auto apply
        for widget in editor.findChildren(QCheckBox):
            widget.toggled.connect(lambda _: self._check_auto_apply())
        for widget in editor.findChildren(QRadioButton):
            widget.toggled.connect(lambda _: self._check_auto_apply())
        for widget in editor.findChildren(QSpinBox):
            widget.valueChanged.connect(lambda _: self._check_auto_apply())
        for widget in editor.findChildren(QDoubleSpinBox):
            widget.valueChanged.connect(lambda _: self._check_auto_apply())
        for widget in editor.findChildren(QComboBox):
            widget.currentTextChanged.connect(lambda _: self._check_auto_apply())
        
        self._active_editors.append(editor)
        self._pipeline_layout.addWidget(editor)
        self._check_auto_apply()

    def _remove_step(self, editor: StepEditor) -> None:
        if editor in self._active_editors:
            self._active_editors.remove(editor)
            self._pipeline_layout.removeWidget(editor)
            editor.deleteLater()
            self._check_auto_apply()

    def _check_auto_apply(self) -> None:
        if self.cb_apply_auto.isChecked():
            self._apply()

    def set_input_payload(self, payload) -> None:
        dataset = payload.dataset if payload is not None else None
        self._dataset_handle = dataset
        self._output_dataset = None
        if dataset:
            self._dataset_label.setText(i18n.tf("Dataset: {name}", name=dataset.display_name))
        else:
            self._dataset_label.setText(i18n.t("Dataset: none"))
            self._result_label.setText("")
        self._apply()

    def current_output_dataset(self) -> DatasetHandle | None:
        return self._output_dataset

    def serialize_node_state(self) -> dict[str, object]:
        steps = []
        for editor in self._active_editors:
            steps.append({"name": editor.step_name, "params": editor.parameters()})
        return {
            "steps_v2": steps, 
            "auto_apply": self.cb_apply_auto.isChecked()
        }

    def restore_node_state(self, payload: dict[str, object]) -> None:
        # Clear existing
        for editor in list(self._active_editors):
            self._remove_step(editor)
            
        steps = payload.get("steps_v2")
        if steps is not None and isinstance(steps, list):
            for step in steps:
                if isinstance(step, dict) and "name" in step:
                    self._append_step(step["name"], step.get("params", {}))
        else:
            # Fallback for old portakal simple preprocess states
            old_steps = payload.get("steps", [])
            if isinstance(old_steps, list):
                for name in old_steps:
                    self._append_step(name)
                    
        self.cb_apply_auto.setChecked(bool(payload.get("auto_apply", False)))

    def help_text(self) -> str:
        return i18n.t("Build a preprocessing pipeline: remove missing values, constant features, normalize, or standardize.")

    def documentation_url(self) -> str:
        return "https://orangedatamining.com/widget-catalog/transform/preprocess/"

    def _apply(self) -> None:
        if self._dataset_handle is None:
            self._output_dataset = None
            self._result_label.setText("")
            self._notify_output_changed()
            return

        steps = []
        for editor in self._active_editors:
            steps.append(PreprocessStep(name=editor.step_name, params=editor.parameters()))

        import traceback
        try:
            self._output_dataset = self._service.preprocess(
                self._dataset_handle,
                steps=steps,
            )

            before_r = self._dataset_handle.row_count
            before_c = self._dataset_handle.column_count
            after_r = self._output_dataset.row_count
            after_c = self._output_dataset.column_count
            self._result_label.setText(
                i18n.tf(
                    "Successfully transformed.\nBefore: {before_r}r x {before_c}c  ->  After: {after_r}r x {after_c}c",
                    before_r=before_r,
                    before_c=before_c,
                    after_r=after_r,
                    after_c=after_c,
                )
            )
        except Exception as e:
            self._result_label.setText(i18n.tf("Error during preprocessing: {error}", error=e))
            self._output_dataset = None

        self._notify_output_changed()

    def refresh_translations(self) -> None:
        if self._dataset_handle is None:
            self._dataset_label.setText(i18n.t("Dataset: none"))
        else:
            self._dataset_label.setText(i18n.tf("Dataset: {name}", name=self._dataset_handle.display_name))
