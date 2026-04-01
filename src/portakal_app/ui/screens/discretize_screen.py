from __future__ import annotations

from typing import Any
from PySide6.QtWidgets import (
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
    QListWidget,
    QRadioButton,
    QButtonGroup,
    QListWidgetItem,
    QCheckBox,
    QSplitter,
    QSpinBox,
    QLineEdit,
    QComboBox,
)
from PySide6.QtCore import Qt, Signal

from portakal_app.data.models import DatasetHandle
from portakal_app.data.services.discretize_service import (
    METHODS,
    DiscretizeService,
)
from portakal_app.ui.screens.node_screen import WorkflowNodeScreenSupport
from portakal_app.ui import i18n


class DiscretizeConfigurator(QGroupBox):
    param_changed = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(i18n.t("Discretize Settings"), parent)
        self.methods = METHODS
        
        self.preset_method = "Keep numeric"
        self.overrides: dict[str, dict[str, object]] = {}
        
        layout = QHBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        
        # Left Panel (List)
        self.list_widget = QListWidget()
        self.list_widget.currentRowChanged.connect(self._on_list_change)
        layout.addWidget(self.list_widget, 4)
        
        # Right Panel (Radio Buttons + Inputs)
        right_panel = QVBoxLayout()
        self.group = QButtonGroup(self)
        self.radio_buttons: list[QRadioButton] = []
        
        # Mapping index to widgets
        self.widgets: dict[int, dict[str, Any]] = {}

        def add_rb(index: int, method_name: str, has_spinbox=False, spinbox_default=2, has_edit=False, edit_default="", edit_label="", has_time=False):
            row = QHBoxLayout()
            rb = QRadioButton(method_name)
            self.group.addButton(rb, index)
            self.radio_buttons.append(rb)
            row.addWidget(rb)
            
            w_dict = {}
            if has_spinbox:
                sb = QSpinBox()
                sb.setRange(2, 50)
                sb.setValue(spinbox_default)
                sb.setEnabled(False)
                row.addWidget(sb)
                w_dict["spin"] = sb
                
            if has_edit:
                ed = QLineEdit(edit_default)
                ed.setEnabled(False)
                row.addWidget(ed)
                w_dict["edit"] = ed
                if edit_label:
                    row.addWidget(QLabel(edit_label))
                    
            if has_time:
                cb = QComboBox()
                cb.addItems([i18n.t("year(s)"), i18n.t("month(s)"), i18n.t("week(s)"), i18n.t("day(s)"), i18n.t("hour(s)"), i18n.t("minute(s)"), i18n.t("second(s)")])
                cb.setEnabled(False)
                row.addWidget(cb)
                w_dict["combo"] = cb
                
            row.addStretch()
            right_panel.addLayout(row)
            self.widgets[index] = w_dict
            rb.clicked.connect(self._on_radio_change)
            if "spin" in w_dict: w_dict["spin"].valueChanged.connect(self._on_value_change)
            if "edit" in w_dict: w_dict["edit"].textChanged.connect(self._on_value_change)

        # Build UI to exactly match Orange
        add_rb(0, i18n.t("Keep numeric"))
        add_rb(1, i18n.t("Remove"))
        add_rb(2, i18n.t("Natural binning, desired bins:"), has_spinbox=True, spinbox_default=2)
        add_rb(3, i18n.t("Fixed width:"), has_edit=True, edit_default="1.0")
        add_rb(4, i18n.t("Time interval:"), has_edit=True, edit_default="1", has_time=True)
        add_rb(5, i18n.t("Equal frequency, intervals:"), has_spinbox=True, spinbox_default=2)
        add_rb(6, i18n.t("Equal width, intervals:"), has_spinbox=True, spinbox_default=2)
        add_rb(7, i18n.t("Entropy vs. MDL"))
        add_rb(8, i18n.t("Custom:"), has_edit=True, edit_default="", edit_label=i18n.t("e.g. 0.0, 0.5, 1.0"))
        add_rb(9, i18n.t("Use default setting"))
        
        right_panel.addStretch(1)
        layout.addLayout(right_panel, 6)
        
        self.group.buttonClicked.connect(self._update_widget_enabled_states)
        self._populate_list([])

    def _update_widget_enabled_states(self):
        checked_id = self.group.checkedId()
        for idx, controls in self.widgets.items():
            is_enabled = (idx == checked_id)
            if "spin" in controls: controls["spin"].setEnabled(is_enabled)
            if "edit" in controls: controls["edit"].setEnabled(is_enabled)
            if "combo" in controls: controls["combo"].setEnabled(is_enabled)

    def _populate_list(self, cols: list[str], has_target: bool = False) -> None:
        self.list_widget.clear()
        
        # Entropy vs. MDL (Index 7) requires a target variable
        entropy_rb = self.radio_buttons[7]
        entropy_rb.setEnabled(has_target)
        if not has_target:
            entropy_rb.setToolTip(i18n.t("A target variable is required for this method."))
        else:
            entropy_rb.setToolTip("")

        preset_item = QListWidgetItem(i18n.tf("★ Default setting: {method}", method=self.preset_method))
        preset_item.setData(Qt.ItemDataRole.UserRole, "__PRESET__")
        self.list_widget.addItem(preset_item)
        
        for col in cols:
            item = QListWidgetItem(col)
            item.setData(Qt.ItemDataRole.UserRole, col)
            self.list_widget.addItem(item)
            
        self.list_widget.setCurrentRow(0)

    def _on_list_change(self, row: int) -> None:
        if row < 0:
            return
            
        item = self.list_widget.item(row)
        col_id = item.data(Qt.ItemDataRole.UserRole)
        
        if col_id == "__PRESET__":
            self.radio_buttons[-1].setEnabled(False) # 'Use default setting' impossible for default
            method = self.preset_method
            conf = self.overrides.get("__PRESET__", {})
        else:
            self.radio_buttons[-1].setEnabled(True)
            conf = self.overrides.get(col_id, {})
            method = conf.get("method", "Use default setting")
            
        try:
            # Map clean names via lambda
            idx = 0
            for i, rb in enumerate(self.radio_buttons):
                if rb.text().startswith(method) or method.startswith(rb.text().split(",")[0]):
                    idx = i
                    break
            self.radio_buttons[idx].setChecked(True)
            
            # Restore inputs
            w_dict = self.widgets.get(idx, {})
            if "spin" in w_dict and "n_bins" in conf:
                w_dict["spin"].blockSignals(True)
                w_dict["spin"].setValue(int(conf["n_bins"]))
                w_dict["spin"].blockSignals(False)
            if "edit" in w_dict:
                w_dict["edit"].blockSignals(True)
                if idx == 3 and "width" in conf:
                    w_dict["edit"].setText(str(conf["width"]))
                elif idx == 8 and "cuts" in conf:
                    w_dict["edit"].setText(str(conf["cuts"]))
                w_dict["edit"].blockSignals(False)
                
            self._update_widget_enabled_states()
        except ValueError:
            pass

    def _on_radio_change(self) -> None:
        self._on_value_change()

    def _on_value_change(self) -> None:
        row = self.list_widget.currentRow()
        if row < 0:
            return
            
        item = self.list_widget.item(row)
        col_id = item.data(Qt.ItemDataRole.UserRole)
        method_idx = self.group.checkedId()
        if method_idx < 0:
            return
            
        # Extract method string mapping back from UI
        method = METHODS[method_idx]
        
        conf = {"method": method}
        w_dict = self.widgets.get(method_idx, {})
        if "spin" in w_dict:
            conf["n_bins"] = w_dict["spin"].value()
        if "edit" in w_dict:
            if method_idx == 3:
                conf["width"] = w_dict["edit"].text()
            elif method_idx == 8:
                conf["cuts"] = w_dict["edit"].text()
               
        if col_id == "__PRESET__":
            if method == "Use default setting":
                self.radio_buttons[0].setChecked(True) # Force Keep numeric
                self._on_value_change()
                return
            self.preset_method = method
            item.setText(i18n.tf("★ Default setting: {method}", method=method))
            self.overrides["__PRESET__"] = conf
        else:
            if method == "Use default setting":
                if col_id in self.overrides:
                    del self.overrides[col_id]
            else:
                self.overrides[col_id] = conf
                
        self.param_changed.emit()

    def reset_all(self):
        self.overrides.clear()
        self.list_widget.setCurrentRow(0)
        self._on_list_change(0)


class DiscretizeScreen(QWidget, WorkflowNodeScreenSupport):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._init_workflow_node_support()
        self._service = DiscretizeService()
        self._dataset_handle: DatasetHandle | None = None
        self._output_dataset: DatasetHandle | None = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(10)

        self._dataset_label = QLabel(i18n.t("Dataset: none"))
        self._dataset_label.setProperty("sectionTitle", True)
        self._dataset_label.setStyleSheet("font-size: 12pt; background: transparent;")
        layout.addWidget(self._dataset_label)

        self.configurator = DiscretizeConfigurator(self)
        layout.addWidget(self.configurator, 1)

        self._result_label = QLabel("")
        self._result_label.setWordWrap(True)
        layout.addWidget(self._result_label)

        footer = QHBoxLayout()
        btn_reset = QPushButton(i18n.t("Reset All"))
        btn_reset.clicked.connect(self._reset_all)
        footer.addWidget(btn_reset)

        footer.addStretch(1)
        self.cb_apply_auto = QCheckBox(i18n.t("Apply Automatically"))
        self.cb_apply_auto.setChecked(True)
        self.cb_apply_auto.toggled.connect(lambda _: self._check_auto_apply())
        footer.addWidget(self.cb_apply_auto)

        self._apply_button = QPushButton(i18n.t("Apply"))
        self._apply_button.setProperty("primary", True)
        self._apply_button.clicked.connect(self._apply)
        footer.addWidget(self._apply_button)
        layout.addLayout(footer)
        
        self.configurator.param_changed.connect(self._check_auto_apply)

    def _check_auto_apply(self):
        if self.cb_apply_auto.isChecked():
            self._apply()

    def _reset_all(self):
        self.configurator.reset_all()
        if self.cb_apply_auto.isChecked():
            self._apply()

    def set_input_payload(self, payload) -> None:
        dataset = payload.dataset if payload is not None else None
        self._dataset_handle = dataset
        self._output_dataset = None
        
        if dataset:
            self._dataset_label.setText(i18n.tf("Dataset: {name}", name=dataset.display_name))
            df = dataset.dataframe
            num_cols = [c for c in df.columns if df.get_column(c).dtype.is_numeric()]
            # Detect if there's a target column in the data domain
            has_target = any(col.role == "target" for col in dataset.domain.columns)
            
            # Disable Entropy MDL if no target
            if not has_target:
                if self.configurator.preset_method == "Entropy vs. MDL":
                    self.configurator.preset_method = "Keep numeric"
                for cid, cnf in list(self.configurator.overrides.items()):
                    if cnf.get("method") == "Entropy vs. MDL":
                        del self.configurator.overrides[cid]
            
            self.configurator._populate_list(num_cols, has_target=has_target)
        else:
            self._dataset_label.setText(i18n.t("Dataset: none"))
            self._result_label.setText("")
            self.configurator._populate_list([], has_target=False)

        self._apply()

    def current_output_dataset(self) -> DatasetHandle | None:
        return self._output_dataset

    def serialize_node_state(self) -> dict[str, object]:
        return {
            "preset_method": self.configurator.preset_method,
            "overrides": self.configurator.overrides,
            "auto_apply": self.cb_apply_auto.isChecked()
        }

    def restore_node_state(self, payload: dict[str, object]) -> None:
        self.configurator.preset_method = str(payload.get("preset_method", "Keep numeric"))
        self.configurator.overrides = payload.get("overrides", {})
        self.cb_apply_auto.setChecked(bool(payload.get("auto_apply", True)))

        if self.configurator.list_widget.count() > 0:
            self.configurator.list_widget.item(0).setText(i18n.tf("★ Default setting: {method}", method=self.configurator.preset_method))

        self.configurator._on_list_change(self.configurator.list_widget.currentRow())

        if self.cb_apply_auto.isChecked():
            self._apply()

    def help_text(self) -> str:
        return "Discretize numeric columns into categorical bins using equal width, frequency, custom cuts, or entropy."

    def documentation_url(self) -> str:
        return "https://orangedatamining.com/widget-catalog/transform/discretize/"

    def refresh_translations(self) -> None:
        if self._dataset_handle is None:
            self._dataset_label.setText(i18n.t("Dataset: none"))
        else:
            self._dataset_label.setText(i18n.tf("Dataset: {name}", name=self._dataset_handle.display_name))
        # Re-apply result label if output exists
        if self._output_dataset is not None:
            before = self._dataset_handle.column_count if self._dataset_handle else 0
            after = self._output_dataset.column_count
            self._result_label.setText(i18n.tf("Result: {after} columns (was {before})", after=after, before=before))

    def _apply(self) -> None:
        if self._dataset_handle is None:
            self._output_dataset = None
            self._result_label.setText("")
            self._notify_output_changed()
            return
            
        try:
            # Reconstruct default configuration exactly
            preset_conf = self.configurator.overrides.get("__PRESET__", {})
            
            clean_overrides = {k: v for k, v in self.configurator.overrides.items() if k != "__PRESET__"}

            self._output_dataset = self._service.discretize(
                self._dataset_handle,
                default_method=self.configurator.preset_method,
                default_params=preset_conf,
                column_methods=clean_overrides,
            )

            before = self._dataset_handle.column_count
            after = self._output_dataset.column_count
            self._result_label.setText(i18n.tf("Result: {after} columns (was {before})", after=after, before=before))
        except Exception as e:
            self._output_dataset = None
            self._result_label.setText(i18n.tf("Error: {error}", error=e))

        self._notify_output_changed()
