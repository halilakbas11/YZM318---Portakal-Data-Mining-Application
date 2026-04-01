from __future__ import annotations

from PySide6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QRadioButton,
    QVBoxLayout,
    QWidget,
)

from portakal_app.data.models import DatasetHandle
from portakal_app.data.services.aggregate_columns_service import OPERATIONS, AggregateColumnsService
from portakal_app.ui.screens.node_screen import WorkflowNodeScreenSupport
from portakal_app.ui import i18n
from portakal_app.ui.shared.type_icons import type_badge_icon

# Selection modes (matching Orange3)
_SEL_ALL = 0
_SEL_ALL_META = 1
_SEL_MANUAL = 2


class AggregateColumnsScreen(QWidget, WorkflowNodeScreenSupport):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._init_workflow_node_support()
        self._service = AggregateColumnsService()
        self._dataset_handle: DatasetHandle | None = None
        self._output_dataset: DatasetHandle | None = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(10)

        self._dataset_label = QLabel(i18n.t("Dataset: none"))
        self._dataset_label.setProperty("sectionTitle", True)
        self._dataset_label.setStyleSheet("font-size: 12pt; background: transparent;")
        layout.addWidget(self._dataset_label)

        cols_group = QGroupBox(i18n.t("Select Columns"))
        cols_layout = QVBoxLayout(cols_group)
        cols_layout.setContentsMargins(10, 10, 10, 10)
        cols_layout.setSpacing(8)

        # Radio buttons for selection mode (like Orange3)
        self._sel_group = QButtonGroup(self)
        self._radio_all = QRadioButton(i18n.t("All"))
        self._radio_all_meta = QRadioButton(i18n.t("All, including meta attributes"))
        self._radio_manual = QRadioButton(i18n.t("Selected variables"))
        self._radio_manual.setChecked(True)
        self._sel_group.addButton(self._radio_all, _SEL_ALL)
        self._sel_group.addButton(self._radio_all_meta, _SEL_ALL_META)
        self._sel_group.addButton(self._radio_manual, _SEL_MANUAL)
        cols_layout.addWidget(self._radio_all)
        cols_layout.addWidget(self._radio_all_meta)
        cols_layout.addWidget(self._radio_manual)
        self._sel_group.idClicked.connect(self._on_selection_mode_changed)

        self._column_list = QListWidget()
        self._column_list.setSelectionMode(QListWidget.SelectionMode.MultiSelection)
        cols_layout.addWidget(self._column_list)
        layout.addWidget(cols_group, 1)

        settings_group = QGroupBox(i18n.t("Aggregation"))
        settings_layout = QVBoxLayout(settings_group)
        settings_layout.setContentsMargins(10, 10, 10, 10)
        settings_layout.setSpacing(8)

        op_row = QHBoxLayout()
        op_row.addWidget(QLabel(i18n.t("Operation:")))
        self._op_combo = QComboBox()
        self._op_combo.addItems(list(OPERATIONS.keys()))
        self._op_combo.setCurrentText("Mean")
        op_row.addWidget(self._op_combo, 1)
        settings_layout.addLayout(op_row)

        name_row = QHBoxLayout()
        name_row.addWidget(QLabel(i18n.t("Output column:")))
        self._output_name = QLineEdit("agg")
        name_row.addWidget(self._output_name, 1)
        settings_layout.addLayout(name_row)

        layout.addWidget(settings_group)

        self._result_label = QLabel("")
        self._result_label.setWordWrap(True)
        layout.addWidget(self._result_label)

        # Auto-apply hooks
        self._sel_group.idClicked.connect(self._check_auto_apply)
        self._op_combo.currentIndexChanged.connect(self._check_auto_apply)
        self._output_name.textChanged.connect(self._check_auto_apply)
        self._column_list.itemSelectionChanged.connect(self._check_auto_apply)

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

    def _on_selection_mode_changed(self, mode_id: int) -> None:
        self._column_list.setEnabled(mode_id == _SEL_MANUAL)
        self._check_auto_apply()

    def _check_auto_apply(self):
        if self.cb_apply_auto.isChecked():
            self._apply()

    def _get_selected_columns(self) -> list[str]:
        """Return the list of column names based on current selection mode."""
        if self._dataset_handle is None:
            return []
        mode = self._sel_group.checkedId()
        if mode == _SEL_ALL:
            return [
                col.name for col in self._dataset_handle.domain.columns
                if col.logical_type == "numeric" and col.role != "meta"
            ]
        if mode == _SEL_ALL_META:
            return [
                col.name for col in self._dataset_handle.domain.columns
                if col.logical_type == "numeric"
            ]
        # _SEL_MANUAL
        return [
            self._column_list.item(i).text()
            for i in range(self._column_list.count())
            if self._column_list.item(i).isSelected()
        ]

    def set_input_payload(self, payload) -> None:
        dataset = payload.dataset if payload is not None else None
        if payload is not None and dataset is self._dataset_handle:
            return
        self._dataset_handle = dataset
        self._output_dataset = None
        self._column_list.clear()

        if dataset:
            self._dataset_label.setText(i18n.tf("Dataset: {name}", name=dataset.display_name))
            for col in dataset.domain.columns:
                if col.logical_type == "numeric":
                    item = QListWidgetItem(type_badge_icon(col.logical_type), col.name)
                    item.setSelected(True)
                    self._column_list.addItem(item)
            self._column_list.setEnabled(self._sel_group.checkedId() == _SEL_MANUAL)
        else:
            self._dataset_label.setText(i18n.t("Dataset: none"))
            self._result_label.setText("")
        self._apply()

    def current_output_dataset(self) -> DatasetHandle | None:
        return self._output_dataset

    def serialize_node_state(self) -> dict[str, object]:
        selected = [
            self._column_list.item(i).text()
            for i in range(self._column_list.count())
            if self._column_list.item(i).isSelected()
        ]
        return {
            "columns": selected,
            "selection_mode": self._sel_group.checkedId(),
            "operation": self._op_combo.currentText(),
            "output_name": self._output_name.text(),
            "auto_apply": self.cb_apply_auto.isChecked(),
        }

    def restore_node_state(self, payload: dict[str, object]) -> None:
        mode = int(payload.get("selection_mode", _SEL_MANUAL))
        btn = self._sel_group.button(mode)
        if btn is not None:
            btn.setChecked(True)
        self._column_list.setEnabled(mode == _SEL_MANUAL)

        cols = payload.get("columns", [])
        if isinstance(cols, list):
            for i in range(self._column_list.count()):
                item = self._column_list.item(i)
                item.setSelected(item.text() in cols)
        op = str(payload.get("operation", "Mean"))
        if self._op_combo.findText(op) >= 0:
            self._op_combo.setCurrentText(op)
        self._output_name.setText(str(payload.get("output_name", "agg")))
        self.cb_apply_auto.setChecked(bool(payload.get("auto_apply", True)))

    def help_text(self) -> str:
        return i18n.t("Compute a row-wise aggregation (sum, mean, etc.) over selected numeric columns.")

    def documentation_url(self) -> str:
        return "https://orangedatamining.com/widget-catalog/transform/aggregate-columns/"

    def _apply(self) -> None:
        if self._dataset_handle is None:
            self._output_dataset = None
            self._result_label.setText("")
            self._notify_output_changed()
            return

        selected = self._get_selected_columns()

        self._output_dataset = self._service.aggregate(
            self._dataset_handle,
            columns=selected,
            operation=self._op_combo.currentText(),
            output_name=self._output_name.text() or "agg",
        )

        in_count = self._dataset_handle.row_count
        out_count = self._output_dataset.row_count if self._output_dataset else 0
        self._result_label.setText(
            i18n.tf(
                "{op} of {col_count} column(s) -> '{out_name}'  |  Input: {in_count} rows  |  Output: {out_count} rows",
                op=self._op_combo.currentText(),
                col_count=len(selected),
                out_name=self._output_name.text(),
                in_count=in_count,
                out_count=out_count,
            )
        )
        self._notify_output_changed()

    def refresh_translations(self) -> None:
        if self._dataset_handle is None:
            self._dataset_label.setText(i18n.t("Dataset: none"))
        else:
            self._dataset_label.setText(i18n.tf("Dataset: {name}", name=self._dataset_handle.display_name))
