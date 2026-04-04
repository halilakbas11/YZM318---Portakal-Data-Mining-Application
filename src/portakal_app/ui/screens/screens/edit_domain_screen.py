from __future__ import annotations

from typing import Callable

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QCheckBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from portakal_app.data.models import DatasetHandle, DomainColumnEdit, DomainEditRequest
from portakal_app.data.services.domain_transform_service import DomainTransformService
from portakal_app.data.services.file_import_service import FileImportService
from portakal_app.ui import i18n
from portakal_app.ui.screens.file_screen import RoleCellWidget, TypeCellWidget
from portakal_app.ui.screens.node_screen import WorkflowNodeScreenSupport


class EditDomainScreen(QWidget, WorkflowNodeScreenSupport):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._init_workflow_node_support()
        self._import_service = FileImportService()
        self._domain_transform_service = DomainTransformService()
        self._dataset_handle: DatasetHandle | None = None
        self._output_dataset: DatasetHandle | None = None
        self._apply_callbacks: list[Callable[[DatasetHandle], None]] = []

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(10)

        self._dataset_label = QLabel("Dataset: none")
        self._dataset_label.setProperty("sectionTitle", True)
        self._dataset_label.setStyleSheet("font-size: 12pt; background: transparent;")
        layout.addWidget(self._dataset_label)

        self._summary_label = QLabel("Load a dataset to edit column names, types and roles.")
        self._summary_label.setWordWrap(True)
        layout.addWidget(self._summary_label)

        self._change_summary_label = QLabel("")
        self._change_summary_label.setProperty("muted", True)
        self._change_summary_label.setWordWrap(True)
        layout.addWidget(self._change_summary_label)

        table_group = QGroupBox("Columns")
        table_layout = QVBoxLayout(table_group)
        table_layout.setContentsMargins(10, 10, 10, 10)
        self._columns_table = QTableWidget(0, 4, self)
        self._columns_table.setHorizontalHeaderLabels(["Name", "Type", "Role", "Sample Values"])
        self._columns_table.verticalHeader().setDefaultSectionSize(30)
        self._columns_table.horizontalHeader().setStretchLastSection(True)
        table_layout.addWidget(self._columns_table)
        layout.addWidget(table_group, 1)

        footer = QHBoxLayout()
        self._restore_inferred_button = QPushButton("Restore Inferred")
        self._restore_inferred_button.setProperty("secondary", True)
        self._restore_inferred_button.clicked.connect(self._restore_inferred)
        footer.addWidget(self._restore_inferred_button)

        self._reset_button = QPushButton("Reset")
        self._reset_button.setProperty("secondary", True)
        self._reset_button.clicked.connect(self._reset_to_current)
        footer.addWidget(self._reset_button)

        footer.addStretch(1)

        self._auto_send_checkbox = QCheckBox("Send Automatically")
        self._auto_send_checkbox.setChecked(False)
        footer.addWidget(self._auto_send_checkbox)

        self._apply_button = QPushButton("Apply Domain")
        self._apply_button.setProperty("primary", True)
        self._apply_button.clicked.connect(self._apply_changes)
        footer.addWidget(self._apply_button)
        layout.addLayout(footer)

        self._set_empty_state()

    def on_apply_requested(self, callback: Callable[[DatasetHandle], None]) -> None:
        self._apply_callbacks.append(callback)

    def set_dataset(self, dataset_handle: DatasetHandle | str | None) -> None:
        if isinstance(dataset_handle, str):
            try:
                dataset_handle = self._import_service.load(dataset_handle)
            except Exception:
                dataset_handle = None
        self._dataset_handle = dataset_handle
        self._output_dataset = None
        if dataset_handle is None:
            self._set_empty_state()
            return
        self._dataset_label.setText(i18n.tf("Dataset: {name}", name=dataset_handle.source.path.name))
        self._summary_label.setText(
            i18n.tf(
                "Editing domain for {count} columns. Apply changes to update the workflow dataset.",
                count=dataset_handle.column_count,
            )
        )
        self._change_summary_label.setText("")
        self._populate_from_request(self._domain_transform_service.build_request(dataset_handle))

    def footer_status_text(self) -> str:
        return str(self._columns_table.rowCount()) if self._dataset_handle is not None else "0"

    def set_input_payload(self, payload) -> None:
        dataset = payload.dataset if payload is not None else None
        self.set_dataset(dataset)
        if dataset is not None and getattr(self, "_auto_send_checkbox", None) is not None and self._auto_send_checkbox.isChecked():
            self._apply_changes()

    def current_output_dataset(self) -> DatasetHandle | None:
        return self._output_dataset

    def serialize_node_state(self) -> dict[str, object]:
        request = self._build_request_from_ui() if self._dataset_handle is not None else DomainEditRequest()
        return {
            "request": [
                {
                    "original_name": column.original_name,
                    "new_name": column.new_name,
                    "logical_type": column.logical_type,
                    "role": column.role,
                }
                for column in request.columns
            ],
            "committed": self._output_dataset is not None,
            "auto_send": getattr(self, "_auto_send_checkbox", None) is not None and self._auto_send_checkbox.isChecked(),
        }

    def restore_node_state(self, payload: dict[str, object]) -> None:
        if self._dataset_handle is None:
            return
        request_payload = payload.get("request", [])
        if isinstance(request_payload, list):
            request = DomainEditRequest(
                columns=tuple(
                    DomainColumnEdit(
                        original_name=str(item.get("original_name") or ""),
                        new_name=str(item.get("new_name") or ""),
                        logical_type=str(item.get("logical_type") or "unknown"),
                        role=str(item.get("role") or "feature"),
                    )
                    for item in request_payload
                    if isinstance(item, dict)
                )
            )
            if request.columns:
                self._populate_from_request(request)
        if hasattr(self, "_auto_send_checkbox"):
            self._auto_send_checkbox.setChecked(bool(payload.get("auto_send", True)))
        if bool(payload.get("committed")):
            self._apply_changes()

    def help_text(self) -> str:
        return (
            "Rename columns, apply strict single-target role rules, drop skipped columns, "
            "and fail fast on lossy type conversions."
        )

    def documentation_url(self) -> str:
        return "https://orangedatamining.com/widget-catalog/data/edit-domain/"

    def _populate_from_request(self, request: DomainEditRequest) -> None:
        self._columns_table.setRowCount(len(request.columns))
        for row_index, edit in enumerate(request.columns):
            self._columns_table.setItem(row_index, 0, QTableWidgetItem(edit.new_name))
            type_widget = TypeCellWidget(edit.logical_type, self._columns_table)
            type_widget.changed.connect(lambda _val: self._auto_apply_if_needed())
            self._columns_table.setCellWidget(row_index, 1, type_widget)
            role_widget = RoleCellWidget(edit.role, self._columns_table)
            role_widget.changed.connect(lambda value, source=role_widget: self._enforce_single_target(source, value))
            role_widget.changed.connect(lambda _val: self._auto_apply_if_needed())
            self._columns_table.setCellWidget(row_index, 2, role_widget)
            sample_values = "-"
            if self._dataset_handle is not None:
                original_column = next(column for column in self._dataset_handle.domain.columns if column.name == edit.original_name)
                sample_values = ", ".join(original_column.sample_values) if original_column.sample_values else "-"
            sample_item = QTableWidgetItem(sample_values)
            sample_item.setFlags(sample_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self._columns_table.setItem(row_index, 3, sample_item)
        self._columns_table.resizeColumnsToContents()

    def _build_request_from_ui(self) -> DomainEditRequest:
        assert self._dataset_handle is not None
        columns = []
        for row_index, original_column in enumerate(self._dataset_handle.domain.columns):
            name_item = self._columns_table.item(row_index, 0)
            type_widget = self._columns_table.cellWidget(row_index, 1)
            role_widget = self._columns_table.cellWidget(row_index, 2)
            columns.append(
                DomainColumnEdit(
                    original_name=original_column.name,
                    new_name=(name_item.text().strip() if name_item is not None else original_column.name) or original_column.name,
                    logical_type=type_widget.current_text() if isinstance(type_widget, TypeCellWidget) else original_column.logical_type,
                    role=role_widget.currentText() if isinstance(role_widget, RoleCellWidget) else original_column.role,
                )
            )
        return DomainEditRequest(columns=tuple(columns))

    def _apply_changes(self) -> None:
        if self._dataset_handle is None:
            return
        request = self._build_request_from_ui()
        try:
            updated_dataset = self._domain_transform_service.apply(self._dataset_handle, request)
        except ValueError as exc:
            QMessageBox.warning(self, i18n.t("Edit Domain"), str(exc))
            return
        summary = self._domain_transform_service.summarize_changes(self._dataset_handle, request, updated_dataset)
        self._dataset_handle = updated_dataset
        self._output_dataset = updated_dataset
        self._summary_label.setText("Domain changes applied to the workflow dataset.")
        self._change_summary_label.setText(summary)
        self._populate_from_request(self._domain_transform_service.build_request(updated_dataset))
        for callback in self._apply_callbacks:
            callback(updated_dataset)
        self._notify_output_changed()

    def _restore_inferred(self) -> None:
        if self._dataset_handle is None:
            return
        self._change_summary_label.setText("Restored inferred domain from the current dataset.")
        self._populate_from_request(self._domain_transform_service.build_request(self._dataset_handle))

    def _reset_to_current(self) -> None:
        if self._dataset_handle is None:
            return
        self._change_summary_label.setText("")
        self._populate_from_request(self._domain_transform_service.build_request(self._dataset_handle))

    def _enforce_single_target(self, source_widget: RoleCellWidget, value: str) -> None:
        if value != "target":
            return
        for row_index in range(self._columns_table.rowCount()):
            role_widget = self._columns_table.cellWidget(row_index, 2)
            if not isinstance(role_widget, RoleCellWidget) or role_widget is source_widget:
                continue
            if role_widget.currentText() == "target":
                role_widget.blockSignals(True)
                role_widget.setCurrentText("feature")
                role_widget.blockSignals(False)

    def _auto_apply_if_needed(self) -> None:
        if hasattr(self, "_auto_send_checkbox") and self._auto_send_checkbox.isChecked():
            self._apply_changes()

    def _set_empty_state(self) -> None:
        self._dataset_label.setText(i18n.t("Dataset: none"))
        self._summary_label.setText(i18n.t("Load a dataset to edit column names, types and roles."))
        self._change_summary_label.setText("")
        self._columns_table.setRowCount(0)

    def refresh_translations(self) -> None:
        if self._dataset_handle is None:
            self._dataset_label.setText(i18n.t("Dataset: none"))
            self._summary_label.setText(i18n.t("Load a dataset to edit column names, types and roles."))
            self._change_summary_label.setText("")
            return
        self._dataset_label.setText(i18n.tf("Dataset: {name}", name=self._dataset_handle.source.path.name))
        self._summary_label.setText(
            i18n.tf(
                "Editing domain for {count} columns. Apply changes to update the workflow dataset.",
                count=self._dataset_handle.column_count,
            )
        )
