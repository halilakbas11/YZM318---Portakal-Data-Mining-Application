from __future__ import annotations

from typing import Any

from PySide6.QtWidgets import (
    QComboBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
    QRadioButton,
    QButtonGroup,
    QTableWidget,
    QTableWidgetItem,
    QHeaderView,
    QCheckBox,
)
from PySide6.QtCore import Qt

from portakal_app.data.models import DatasetHandle
from portakal_app.data.services.impute_service import ImputeService
from portakal_app.ui.screens.node_screen import WorkflowNodeScreenSupport
from portakal_app.ui import i18n

METHODS = [
    "Don't impute",
    "Average/Most frequent",
    "As a distinct value",
    "Fixed values",
    "Random values",
    "Remove instances with unknown values",
]

class ImputeScreen(QWidget, WorkflowNodeScreenSupport):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._init_workflow_node_support()
        self._service = ImputeService()
        self._dataset_handle: DatasetHandle | None = None
        self._output_dataset: DatasetHandle | None = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(10)

        # Dataset info label
        self._dataset_label = QLabel(i18n.t("Dataset: none"))
        self._dataset_label.setProperty("sectionTitle", True)
        self._dataset_label.setStyleSheet("font-size: 12pt; background: transparent;")
        layout.addWidget(self._dataset_label)

        # Default Method
        method_group = QGroupBox(i18n.t("Default Method"))
        method_layout = QVBoxLayout(method_group)
        method_layout.setContentsMargins(10, 10, 10, 10)
        
        self.default_group = QButtonGroup(self)
        grid_layout = QVBoxLayout()
        for idx, method in enumerate(METHODS):
            row = QHBoxLayout()
            rb = QRadioButton(i18n.t(method))
            if method == "Average/Most frequent":
                rb.setChecked(True)
            self.default_group.addButton(rb, idx)
            row.addWidget(rb)
            
            if method == "Fixed values":
                self._fixed_edit = QLineEdit("0")
                self._fixed_edit.setFixedWidth(60)
                row.addWidget(QLabel(i18n.t(" num: ")))
                row.addWidget(self._fixed_edit)
                
                self._fixed_edit_cat = QLineEdit("N/A")
                self._fixed_edit_cat.setFixedWidth(60)
                row.addWidget(QLabel(i18n.t(" cat: ")))
                row.addWidget(self._fixed_edit_cat)
            
            if method == "Random values":
                row.addWidget(QLabel(i18n.t(" seed: ")))
                self._seed_spin = QSpinBox()
                self._seed_spin.setRange(0, 999999)
                self._seed_spin.setValue(42)
                row.addWidget(self._seed_spin)
                
            row.addStretch()
            grid_layout.addLayout(row)
            
        method_layout.addLayout(grid_layout)
        layout.addWidget(method_group)

        # Individual Attribute Settings
        attr_group = QGroupBox(i18n.t("Individual Attribute Settings"))
        attr_layout = QVBoxLayout(attr_group)
        
        self.table = QTableWidget(0, 2)
        self.table.setHorizontalHeaderLabels([i18n.t("Attribute"), i18n.t("Imputation Method")])
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.verticalHeader().setVisible(False)
        attr_layout.addWidget(self.table)
        
        btn_restore = QPushButton(i18n.t("Restore All to Default"))
        btn_restore.clicked.connect(self._restore_defaults)
        attr_layout.addWidget(btn_restore)
        
        layout.addWidget(attr_group)

        # Result info
        self._result_label = QLabel("")
        self._result_label.setWordWrap(True)
        layout.addWidget(self._result_label)

        # Auto-apply hooks
        self.default_group.idClicked.connect(self._check_auto_apply)
        self._fixed_edit.textChanged.connect(self._check_auto_apply)
        self._fixed_edit_cat.textChanged.connect(self._check_auto_apply)
        self._seed_spin.valueChanged.connect(self._check_auto_apply)

        # Footer applies
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

    def _check_auto_apply(self):
        if self.cb_apply_auto.isChecked():
            self._apply()

    def _restore_defaults(self):
        for row in range(self.table.rowCount()):
            combo = self.table.cellWidget(row, 1)
            if isinstance(combo, QComboBox):
                combo.setCurrentIndex(0)
        self._check_auto_apply()

    def set_input_payload(self, payload) -> None:
        dataset = payload.dataset if payload is not None else None
        self._dataset_handle = dataset
        self._output_dataset = None

        if dataset:
            self._dataset_label.setText(i18n.tf("Dataset: {name}", name=dataset.display_name))
            self.table.setRowCount(0)
            for c in dataset.domain.columns:
                row = self.table.rowCount()
                self.table.insertRow(row)
                
                item = QTableWidgetItem(c.name)
                item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                
                if getattr(c, "role", "feature") == "target":
                    item.setForeground(Qt.GlobalColor.blue)
                
                self.table.setItem(row, 0, item)
                
                combo = QComboBox()
                combo.addItem(i18n.t("(Default)"))
                combo.addItems([i18n.t(m) for m in METHODS])
                combo.setStyleSheet("QComboBox { color: #2b2b2b; } QComboBox QAbstractItemView { color: #2b2b2b; background: white; }")
                combo.currentIndexChanged.connect(self._check_auto_apply)
                self.table.setCellWidget(row, 1, combo)
        else:
            self._dataset_label.setText(i18n.t("Dataset: none"))
            self.table.setRowCount(0)
            self._result_label.setText("")
            
        self._apply()

    def current_output_dataset(self) -> DatasetHandle | None:
        return self._output_dataset

    def serialize_node_state(self) -> dict[str, object]:
        overrides = {}
        for row in range(self.table.rowCount()):
            col_name = self.table.item(row, 0).text()
            combo = self.table.cellWidget(row, 1)
            if isinstance(combo, QComboBox) and combo.currentIndex() > 0:
                overrides[col_name] = METHODS[combo.currentIndex() - 1]

        return {
            "default_method": METHODS[self.default_group.checkedId()],
            "fixed_value": self._fixed_edit.text(),
            "fixed_value_cat": self._fixed_edit_cat.text(),
            "seed": self._seed_spin.value(),
            "overrides": overrides,
            "auto_apply": self.cb_apply_auto.isChecked()
        }

    def restore_node_state(self, payload: dict[str, object]) -> None:
        m = str(payload.get("default_method", "Average/Most frequent"))
        if m in METHODS:
            self.default_group.button(METHODS.index(m)).setChecked(True)
        else:
            self.default_group.button(1).setChecked(True)
            
        self._fixed_edit.setText(str(payload.get("fixed_value", "0")))
        self._fixed_edit_cat.setText(str(payload.get("fixed_value_cat", "N/A")))
        self._seed_spin.setValue(int(payload.get("seed", 42)))
        self.cb_apply_auto.setChecked(bool(payload.get("auto_apply", True)))
        
        overrides = payload.get("overrides", {})
        if isinstance(overrides, dict):
            for row in range(self.table.rowCount()):
                col_name = self.table.item(row, 0).text()
                if col_name in overrides:
                    combo = self.table.cellWidget(row, 1)
                    if isinstance(combo, QComboBox) and overrides[col_name] in METHODS:
                        combo.setCurrentIndex(METHODS.index(overrides[col_name]) + 1)

    def help_text(self) -> str:
        return "Fill missing values using average, fixed value, random sampling, or drop rows. You can override settings per column."

    def documentation_url(self) -> str:
        return "https://orangedatamining.com/widget-catalog/transform/impute/"

    def _apply(self) -> None:
        if self._dataset_handle is None:
            self._output_dataset = None
            self._result_label.setText("")
            self._notify_output_changed()
            return
            
        default_method = METHODS[self.default_group.checkedId()]
        column_methods = {}
        
        for row in range(self.table.rowCount()):
            col_name = self.table.item(row, 0).text()
            combo = self.table.cellWidget(row, 1)
            if isinstance(combo, QComboBox) and combo.currentIndex() > 0:
                method = METHODS[combo.currentIndex() - 1]
                column_methods[col_name] = {
                    "method": method,
                    "fixed_value": self._fixed_edit.text(),
                    "fixed_value_cat": self._fixed_edit_cat.text()
                }

        try:
            self._output_dataset = self._service.impute(
                self._dataset_handle,
                default_method=default_method,
                default_fixed_value=self._fixed_edit.text(),
                default_fixed_value_cat=self._fixed_edit_cat.text(),
                seed=self._seed_spin.value(),
                column_methods=column_methods,
            )

            remaining = sum(col.null_count for col in self._output_dataset.domain.columns)
            if remaining > 0:
                self._result_label.setText(
                    i18n.tf("Imputed. Warning: Data still contains missing values ({count}) | Rows: {rows}", count=remaining, rows=self._output_dataset.row_count)
                )
                self._result_label.setStyleSheet("color: #d9534f; font-weight: bold;")
            else:
                self._result_label.setText(
                    i18n.tf("Imputed completely. Rows: {rows}", rows=self._output_dataset.row_count)
                )
                self._result_label.setStyleSheet("color: palette(text); font-weight: normal;")
        except Exception as e:
            self._result_label.setText(f"Impute Failed: {e}")
            self._result_label.setStyleSheet("")
            self._output_dataset = None
            
        self._notify_output_changed()
