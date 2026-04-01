from __future__ import annotations

from typing import Any
from PySide6.QtWidgets import (
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
    QListWidget,
    QRadioButton,
    QButtonGroup,
    QListWidgetItem,
    QCheckBox,
    QSplitter,
    QAbstractItemView,
    QDialog,
    QDialogButtonBox,
)
from PySide6.QtCore import Qt, Signal
import polars as pl

from portakal_app.data.models import DatasetHandle
from portakal_app.data.services.continuize_service import (
    CONTINUOUS_METHODS,
    DISCRETE_METHODS,
    ContinuizeService,
)
from portakal_app.ui.screens.node_screen import WorkflowNodeScreenSupport
from portakal_app.ui import i18n


class CategoryOrderDialog(QDialog):
    def __init__(self, col_name: str, values: list[str], parent: QWidget | None = None):
        super().__init__(parent)
        self.setWindowTitle(i18n.tf("Order for {col}", col=col_name))
        layout = QVBoxLayout(self)
        
        layout.addWidget(QLabel(i18n.t("Drag and drop to reorder categories (Ordinal logic):")))
        self.list = QListWidget()
        self.list.setDragDropMode(QAbstractItemView.DragDropMode.InternalMove)
        for v in values:
            self.list.addItem(v)
        layout.addWidget(self.list)
        
        bb = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        layout.addWidget(bb)
        
    def get_order(self) -> list[str]:
        return [self.list.item(i).text() for i in range(self.list.count())]


class TypeConfigurator(QGroupBox):
    """Panel for one variable type (discrete / continuous).

    Features matching Orange:
    - Multi-select columns (Extended Selection)
    - Filter / search box
    - Default preset row (★)
    - Per-column overrides via radio buttons
    """
    param_changed = Signal()

    def __init__(
        self,
        title: str,
        methods: tuple[str, ...],
        preset_initial: str,
        is_discrete: bool,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(title, parent)
        self.methods = methods
        self.is_discrete = is_discrete

        self.preset_value = preset_initial
        self.overrides: dict[str, str] = {}
        self.categorical_orders: dict[str, list[str]] = {}
        self._all_cols: list[str] = []
        self._dataset_handle: DatasetHandle | None = None

        layout = QHBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)

        # Left: filter + list
        left_panel = QVBoxLayout()

        self.filter_edit = QLineEdit()
        self.filter_edit.setPlaceholderText(i18n.t("Filter..."))
        self.filter_edit.textChanged.connect(self._apply_filter)
        left_panel.addWidget(self.filter_edit)

        self.list_widget = QListWidget()
        self.list_widget.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.list_widget.itemSelectionChanged.connect(self._on_selection_change)
        left_panel.addWidget(self.list_widget)

        layout.addLayout(left_panel, 1)

        # Right: radio buttons
        right_panel = QVBoxLayout()
        self.group = QButtonGroup(self)
        self.radio_buttons: list[QRadioButton] = []
        for idx, method in enumerate(methods):
            rb = QRadioButton(method)
            self.group.addButton(rb, idx)
            self.radio_buttons.append(rb)
            right_panel.addWidget(rb)
            
            if method == "Normalize to interval [a, b]":
                from PySide6.QtWidgets import QDoubleSpinBox
                self.custom_bounds_widget = QWidget()
                cb_layout = QHBoxLayout(self.custom_bounds_widget)
                cb_layout.setContentsMargins(20, 0, 0, 0)
                
                self.spin_a = QDoubleSpinBox()
                self.spin_a.setRange(-999999.0, 999999.0)
                self.spin_a.setValue(0.0)
                
                self.spin_b = QDoubleSpinBox()
                self.spin_b.setRange(-999999.0, 999999.0)
                self.spin_b.setValue(1.0)
                
                cb_layout.addWidget(QLabel("a:"))
                cb_layout.addWidget(self.spin_a)
                cb_layout.addWidget(QLabel("b:"))
                cb_layout.addWidget(self.spin_b)
                cb_layout.addStretch(1)
                
                right_panel.addWidget(self.custom_bounds_widget)
                self.custom_bounds_widget.setEnabled(False)
                
                self.spin_a.valueChanged.connect(lambda _: self.param_changed.emit())
                self.spin_b.valueChanged.connect(lambda _: self.param_changed.emit())

            rb.clicked.connect(self._on_radio_change)

        right_panel.addStretch(1)
        
        if is_discrete:
            self.btn_order = QPushButton(i18n.t("Values/Order..."))
            self.btn_order.clicked.connect(self._on_order_click)
            self.btn_order.setEnabled(False)
            right_panel.addWidget(self.btn_order)
            
        layout.addLayout(right_panel, 1)

        self._populate_list([])

    def _populate_list(self, cols: list[str]) -> None:
        self._all_cols = list(cols)
        self.list_widget.clear()

        # Preset row (always visible, always first)
        preset_item = QListWidgetItem(f"★ Preset: {self.preset_value}")
        preset_item.setData(Qt.ItemDataRole.UserRole, "__PRESET__")
        self.list_widget.addItem(preset_item)

        for col in cols:
            override = self.overrides.get(col)
            label = f"{col}: {override}" if override else col
            item = QListWidgetItem(label)
            item.setData(Qt.ItemDataRole.UserRole, col)
            self.list_widget.addItem(item)

        self.list_widget.setCurrentRow(0)

    def _apply_filter(self, text: str) -> None:
        text_lower = text.lower()
        for i in range(self.list_widget.count()):
            item = self.list_widget.item(i)
            col_id = item.data(Qt.ItemDataRole.UserRole)
            if col_id == "__PRESET__":
                item.setHidden(False)
                continue
            item.setHidden(text_lower not in col_id.lower())

    def _on_selection_change(self) -> None:
        selected = self.list_widget.selectedItems()
        if not selected:
            return

        # Determine what's selected
        col_ids = [item.data(Qt.ItemDataRole.UserRole) for item in selected]

        if "__PRESET__" in col_ids:
            # Default selected: show preset value, disable "Use preset" radio
            self.radio_buttons[0].setEnabled(False)
            val = self.preset_value
        else:
            self.radio_buttons[0].setEnabled(True)
            # If all selected share the same override, show that; otherwise no check
            methods_set = set()
            for cid in col_ids:
                methods_set.add(self.overrides.get(cid, "Use preset"))
            if len(methods_set) == 1:
                val = methods_set.pop()
            else:
                # Mixed selection – uncheck all radios
                checked = self.group.checkedButton()
                if checked:
                    self.group.setExclusive(False)
                    checked.setChecked(False)
                    self.group.setExclusive(True)
                return

        try:
            idx = self.methods.index(val)
            self.radio_buttons[idx].setChecked(True)
            if hasattr(self, "custom_bounds_widget"):
                self.custom_bounds_widget.setEnabled(val == "Normalize to interval [a, b]")
        except ValueError:
            pass
            
        if self.is_discrete:
            # Enable order button only if exactly one column (not preset) is selected
            self.btn_order.setEnabled(len(col_ids) == 1 and "__PRESET__" not in col_ids)

    def _on_order_click(self) -> None:
        selected = self.list_widget.selectedItems()
        if not selected or not self._dataset_handle: return
        col_id = selected[0].data(Qt.ItemDataRole.UserRole)
        
        # Get current unique values for the column
        series = self._dataset_handle.dataframe.get_column(col_id)
        current_vals = sorted(set(series.cast(pl.Utf8).drop_nulls().to_list()))
        
        # If we already have a manual order, use it as baseline, but sync with current data
        saved_order = self.categorical_orders.get(col_id, [])
        if saved_order:
            # Reconstruct order ensuring all current values are included
            values = [v for v in saved_order if v in current_vals]
            values += [v for v in current_vals if v not in values]
        else:
            values = current_vals
            
        dlg = CategoryOrderDialog(col_id, values, self)
        if dlg.exec():
            self.categorical_orders[col_id] = dlg.get_order()

    def _on_radio_change(self) -> None:
        selected = self.list_widget.selectedItems()
        if not selected:
            return

        method_idx = self.group.checkedId()
        if method_idx < 0:
            return
        method = self.methods[method_idx]
        
        if hasattr(self, "custom_bounds_widget"):
            self.custom_bounds_widget.setEnabled(method == "Normalize to interval [a, b]")

        col_ids = [item.data(Qt.ItemDataRole.UserRole) for item in selected]

        for item in selected:
            col_id = item.data(Qt.ItemDataRole.UserRole)
            if col_id == "__PRESET__":
                # Prevent setting preset to "Use preset"
                if method == "Use preset":
                    self.radio_buttons[1].setChecked(True)
                    method = self.methods[1]
                self.preset_value = method
                item.setText(f"★ Preset: {method}")
            else:
                if method == "Use preset":
                    if col_id in self.overrides:
                        del self.overrides[col_id]
                    item.setText(col_id)
                else:
                    self.overrides[col_id] = method
                    item.setText(f"{col_id}: {method}")

        self.param_changed.emit()

    def reset_all(self):
        self.overrides.clear()
        # Refresh list labels
        for i in range(1, self.list_widget.count()):
            item = self.list_widget.item(i)
            col_id = item.data(Qt.ItemDataRole.UserRole)
            item.setText(col_id)
        self.list_widget.setCurrentRow(0)
        self._on_selection_change()


class ContinuizeScreen(QWidget, WorkflowNodeScreenSupport):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._init_workflow_node_support()
        self._service = ContinuizeService()
        self._dataset_handle: DatasetHandle | None = None
        self._output_dataset: DatasetHandle | None = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(10)

        self._dataset_label = QLabel(i18n.t("Dataset: none"))
        self._dataset_label.setProperty("sectionTitle", True)
        self._dataset_label.setStyleSheet("font-size: 12pt; background: transparent;")
        layout.addWidget(self._dataset_label)

        splitter = QSplitter(Qt.Orientation.Vertical)

        self.cat_config = TypeConfigurator(
            i18n.t("Categorical Variables"), DISCRETE_METHODS, "First value as base", True
        )
        splitter.addWidget(self.cat_config)

        self.num_config = TypeConfigurator(
            i18n.t("Numeric Variables"), CONTINUOUS_METHODS, "Keep as it is", False
        )
        splitter.addWidget(self.num_config)

        layout.addWidget(splitter, 1)

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
        footer.addWidget(self.cb_apply_auto)

        self._apply_button = QPushButton(i18n.t("Apply"))
        self._apply_button.setProperty("primary", True)
        self._apply_button.clicked.connect(self._apply)
        footer.addWidget(self._apply_button)
        layout.addLayout(footer)

        # Connect parameter changes to auto-apply
        self.cat_config.param_changed.connect(self._check_auto_apply)
        self.num_config.param_changed.connect(self._check_auto_apply)
        self.cb_apply_auto.toggled.connect(lambda _: self._check_auto_apply())

    def _check_auto_apply(self):
        if self.cb_apply_auto.isChecked():
            self._apply()

    def _reset_all(self):
        self.cat_config.reset_all()
        self.num_config.reset_all()
        if self.cb_apply_auto.isChecked():
            self._apply()

    def set_input_payload(self, payload) -> None:
        dataset = payload.dataset if payload is not None else None
        self._dataset_handle = dataset
        self._output_dataset = None

        if dataset:
            self._dataset_label.setText(i18n.tf("Dataset: {name}", name=dataset.display_name))
            cat_cols = []
            num_cols = []
            for col in dataset.domain.columns:
                if col.logical_type == "numeric":
                    num_cols.append(col.name)
                elif col.logical_type == "categorical":
                    cat_cols.append(col.name)
            self.cat_config._dataset_handle = dataset
            self.cat_config._populate_list(cat_cols)
            self.num_config._populate_list(num_cols)
        else:
            self._dataset_label.setText(i18n.t("Dataset: none"))
            self._result_label.setText("")
            self.cat_config._populate_list([])
            self.num_config._populate_list([])

        self._apply()

    def current_output_dataset(self) -> DatasetHandle | None:
        return self._output_dataset

    def serialize_node_state(self) -> dict[str, object]:
        bounds = (0.0, 1.0)
        if hasattr(self.num_config, "spin_a"):
            bounds = (self.num_config.spin_a.value(), self.num_config.spin_b.value())
            
        return {
            "discrete_preset": self.cat_config.preset_value,
            "continuous_preset": self.num_config.preset_value,
            "discrete_overrides": self.cat_config.overrides,
            "continuous_overrides": self.num_config.overrides,
            "categorical_orders": self.cat_config.categorical_orders,
            "custom_bounds": bounds,
            "auto_apply": self.cb_apply_auto.isChecked()
        }

    def restore_node_state(self, payload: dict[str, object]) -> None:
        self.cat_config.preset_value = str(payload.get("discrete_preset", "First value as base"))
        self.num_config.preset_value = str(payload.get("continuous_preset", "Keep as it is"))

        self.cat_config.overrides = payload.get("discrete_overrides", {})
        self.num_config.overrides = payload.get("continuous_overrides", {})
        self.cat_config.categorical_orders = payload.get("categorical_orders", {})
        
        bounds_lst = payload.get("custom_bounds", [0.0, 1.0])
        if hasattr(self.num_config, "spin_a"):
            self.num_config.spin_a.setValue(bounds_lst[0])
            self.num_config.spin_b.setValue(bounds_lst[1])

        self.cb_apply_auto.setChecked(bool(payload.get("auto_apply", True)))

        # Update visuals for presets
        if self.cat_config.list_widget.count() > 0:
            self.cat_config.list_widget.item(0).setText(f"★ Preset: {self.cat_config.preset_value}")
        if self.num_config.list_widget.count() > 0:
            self.num_config.list_widget.item(0).setText(f"★ Preset: {self.num_config.preset_value}")

        # Update list labels for overrides
        for config in (self.cat_config, self.num_config):
            for i in range(1, config.list_widget.count()):
                item = config.list_widget.item(i)
                col_id = item.data(Qt.ItemDataRole.UserRole)
                override = config.overrides.get(col_id)
                if override:
                    item.setText(f"{col_id}: {override}")
                else:
                    item.setText(col_id)

        self.cat_config._on_selection_change()
        self.num_config._on_selection_change()

        if self.cb_apply_auto.isChecked():
            self._apply()

    def help_text(self) -> str:
        return "Convert categorical columns to numeric (one-hot, ordinal, etc.) and normalize continuous columns."

    def documentation_url(self) -> str:
        return "https://orangedatamining.com/widget-catalog/transform/continuize/"

    def _apply(self) -> None:
        if self._dataset_handle is None:
            self._output_dataset = None
            self._result_label.setText("")
            self._notify_output_changed()
            return

        all_methods = {}
        all_methods.update(self.cat_config.overrides)
        all_methods.update(self.num_config.overrides)

        a, b = 0.0, 1.0
        if hasattr(self.num_config, "spin_a"):
            a = self.num_config.spin_a.value()
            b = self.num_config.spin_b.value()

        if a >= b:
             self._output_dataset = None
             self._result_label.setText(i18n.t("Error: Lower bound must be smaller than upper bound"))
             self._notify_output_changed()
             return

        try:
            self._output_dataset = self._service.continuize(
                self._dataset_handle,
                discrete_preset=self.cat_config.preset_value,
                continuous_preset=self.num_config.preset_value,
                column_methods=all_methods,
                categorical_orders=self.cat_config.categorical_orders,
                normalize_custom_bounds=(a, b),
            )

            before = self._dataset_handle.column_count
            after = self._output_dataset.column_count
            self._result_label.setText(
                i18n.tf("Result: {after} columns (was {before})", after=after, before=before)
            )
        except Exception as e:
            self._output_dataset = None
            self._result_label.setText(i18n.tf("Error: {error}", error=e))

        self._notify_output_changed()

    def refresh_translations(self) -> None:
        if self._dataset_handle:
            self._dataset_label.setText(
                i18n.tf("Dataset: {name}", name=self._dataset_handle.display_name)
            )
        else:
            self._dataset_label.setText(i18n.t("Dataset: none"))

        self.cat_config.setTitle(i18n.t("Categorical Variables"))
        self.num_config.setTitle(i18n.t("Numeric Variables"))
        self.cat_config.filter_edit.setPlaceholderText(i18n.t("Filter..."))
        self.num_config.filter_edit.setPlaceholderText(i18n.t("Filter..."))
        self._apply_button.setText(i18n.t("Apply"))
        self.cb_apply_auto.setText(i18n.t("Apply Automatically"))
