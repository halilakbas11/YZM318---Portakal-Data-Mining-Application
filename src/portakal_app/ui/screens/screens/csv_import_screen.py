from __future__ import annotations

from typing import Callable

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from portakal_app.data.errors import PortakalDataError
from portakal_app.data.models import CSVImportOptions, DatasetHandle
from portakal_app.data.services.file_import_service import FileImportService
from portakal_app.ui.screens.node_screen import WorkflowNodeScreenSupport


DELIMITER_OPTIONS = {
    "Auto": ",",
    "Comma (,)": ",",
    "Tab (\\t)": "\t",
    "Semicolon (;)": ";",
    "Pipe (|)": "|",
}
ENCODING_OPTIONS = ("Auto", "utf-8-sig", "utf-8", "cp1254", "latin-1")
PREVIEW_ROW_LIMIT = 100


class CSVImportScreen(QWidget, WorkflowNodeScreenSupport):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._init_workflow_node_support()
        self._import_service = FileImportService()
        self._dataset_handle: DatasetHandle | None = None
        self._output_dataset: DatasetHandle | None = None
        self._selected_path: str | None = None
        self._resolved_options: CSVImportOptions | None = None
        self._import_callbacks: list[Callable[[DatasetHandle], None]] = []

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(10)

        layout.addWidget(self._build_source_group())
        layout.addWidget(self._build_options_group())
        layout.addWidget(self._build_info_group())
        layout.addWidget(self._build_preview_group(), 1)
        layout.addLayout(self._build_footer())

        self._set_empty_state()

    def _build_source_group(self) -> QGroupBox:
        group = QGroupBox("Delimited Source")
        layout = QHBoxLayout(group)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)

        self._path_input = QLineEdit(self)
        self._path_input.setPlaceholderText("Select a CSV/TSV/TAB/text file...")
        self._path_input.textChanged.connect(self._handle_path_changed)
        layout.addWidget(self._path_input, 1)

        self._browse_button = QPushButton("...")
        self._browse_button.setObjectName("fileSourceActionButton")
        self._browse_button.clicked.connect(self._handle_browse)
        self._browse_button.setFixedWidth(44)
        layout.addWidget(self._browse_button)

        self._reload_button = QPushButton("Preview")
        self._reload_button.setObjectName("fileSourceActionButton")
        self._reload_button.clicked.connect(self._handle_reload_clicked)
        layout.addWidget(self._reload_button)
        return group

    def _build_options_group(self) -> QGroupBox:
        group = QGroupBox("Import Options")
        layout = QFormLayout(group)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)

        self._delimiter_combo = QComboBox(self)
        self._delimiter_combo.addItems(list(DELIMITER_OPTIONS.keys()))
        self._delimiter_combo.currentTextChanged.connect(lambda _value: self._mark_dirty())
        layout.addRow("Delimiter", self._delimiter_combo)

        self._encoding_combo = QComboBox(self)
        self._encoding_combo.addItems(list(ENCODING_OPTIONS))
        self._encoding_combo.currentTextChanged.connect(lambda _value: self._mark_dirty())
        layout.addRow("Encoding", self._encoding_combo)

        self._skip_rows_spin = QSpinBox(self)
        self._skip_rows_spin.setRange(0, 100000)
        self._skip_rows_spin.valueChanged.connect(lambda _value: self._mark_dirty())
        layout.addRow("Skip first rows", self._skip_rows_spin)

        self._has_header_checkbox = QCheckBox("First parsed row is header")
        self._has_header_checkbox.setChecked(True)
        self._has_header_checkbox.toggled.connect(lambda _checked: self._mark_dirty())
        layout.addRow("", self._has_header_checkbox)
        return group

    def _build_info_group(self) -> QGroupBox:
        group = QGroupBox("Info")
        layout = QVBoxLayout(group)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(4)

        self._dataset_label = QLabel("No imported dataset")
        self._dataset_label.setProperty("sectionTitle", True)
        self._dataset_label.setStyleSheet("font-size: 12pt; background: transparent;")
        layout.addWidget(self._dataset_label)

        self._status_label = QLabel("Choose a delimited file and preview it before importing.")
        self._status_label.setWordWrap(True)
        layout.addWidget(self._status_label)

        self._settings_label = QLabel("")
        self._settings_label.setProperty("muted", True)
        self._settings_label.setWordWrap(True)
        layout.addWidget(self._settings_label)
        return group

    def _build_preview_group(self) -> QGroupBox:
        group = QGroupBox("Preview")
        layout = QVBoxLayout(group)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(6)

        self._preview_table = QTableWidget(0, 0, self)
        self._preview_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._preview_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self._preview_table.horizontalHeader().setStretchLastSection(True)
        self._preview_table.setMinimumHeight(240)
        layout.addWidget(self._preview_table)
        return group

    def _build_footer(self) -> QHBoxLayout:
        layout = QHBoxLayout()
        layout.setSpacing(8)
        self._reset_button = QPushButton("Reset")
        self._reset_button.setProperty("secondary", True)
        self._reset_button.clicked.connect(self._reset_form)
        layout.addWidget(self._reset_button)

        layout.addStretch(1)

        self._auto_send_checkbox = QCheckBox("Send Automatically")
        self._auto_send_checkbox.setChecked(False)
        layout.addWidget(self._auto_send_checkbox)

        self._apply_button = QPushButton("Apply Import")
        self._apply_button.setProperty("primary", True)
        self._apply_button.clicked.connect(self._handle_apply_clicked)
        layout.addWidget(self._apply_button)
        return layout

    def on_import_requested(self, callback: Callable[[DatasetHandle], None]) -> None:
        self._import_callbacks.append(callback)

    def set_dataset(self, dataset_handle: DatasetHandle | str | None) -> None:
        if isinstance(dataset_handle, str):
            try:
                dataset_handle = self._import_service.load(dataset_handle)
            except PortakalDataError:
                dataset_handle = None
        self._dataset_handle = dataset_handle
        self._output_dataset = dataset_handle
        if dataset_handle is None:
            self._set_empty_state()
            return
        self._selected_path = str(dataset_handle.source.path)
        self._path_input.setText(self._selected_path)
        self._populate_from_handle(dataset_handle, imported=False, resolved_options=self._resolved_options)

    def footer_status_text(self) -> str:
        return str(self._dataset_handle.row_count) if self._dataset_handle is not None else "0"

    def set_input_payload(self, payload) -> None:
        _ = payload

    def current_output_dataset(self) -> DatasetHandle | None:
        return self._output_dataset

    def serialize_node_state(self) -> dict[str, object]:
        return {
            "path": self._path_input.text().strip(),
            "delimiter": self._delimiter_combo.currentText(),
            "encoding": self._encoding_combo.currentText(),
            "skip_rows": self._skip_rows_spin.value(),
            "has_header": self._has_header_checkbox.isChecked(),
            "committed": self._output_dataset is not None,
            "auto_send": getattr(self, "_auto_send_checkbox", None) is not None and self._auto_send_checkbox.isChecked(),
        }

    def restore_node_state(self, payload: dict[str, object]) -> None:
        self._path_input.setText(str(payload.get("path") or ""))
        self._delimiter_combo.setCurrentText(str(payload.get("delimiter") or "Auto"))
        self._encoding_combo.setCurrentText(str(payload.get("encoding") or "Auto"))
        self._skip_rows_spin.setValue(int(payload.get("skip_rows") or 0))
        self._has_header_checkbox.setChecked(bool(payload.get("has_header", True)))
        if hasattr(self, "_auto_send_checkbox"):
            self._auto_send_checkbox.setChecked(bool(payload.get("auto_send", True)))
        committed = bool(payload.get("committed"))
        if committed:
            loaded = self._load_dataset_from_controls()
            if loaded is None:
                self._output_dataset = None
                return
            dataset, resolved_options = loaded
            self._output_dataset = dataset
            self._populate_from_handle(dataset, imported=True, resolved_options=resolved_options)
            return
        if self._path_input.text().strip():
            self._handle_reload_clicked()

    def data_preview_snapshot(self) -> dict[str, object]:
        headers = [
            self._preview_table.horizontalHeaderItem(index).text()
            for index in range(self._preview_table.columnCount())
        ]
        rows = []
        for row_index in range(self._preview_table.rowCount()):
            rows.append(
                [
                    self._preview_table.item(row_index, column_index).text()
                    if self._preview_table.item(row_index, column_index) is not None
                    else ""
                    for column_index in range(self._preview_table.columnCount())
                ]
            )
        return {
            "summary": self._status_label.text(),
            "headers": headers,
            "rows": rows,
        }

    def help_text(self) -> str:
        return (
            "Import delimited text data with explicit delimiter, encoding and row skipping controls. "
            "Preview the parsed rows before applying the dataset to the workflow."
        )

    def documentation_url(self) -> str:
        return "https://orangedatamining.com/widget-catalog/data/file/"

    def _handle_browse(self) -> None:
        path, _selected_filter = QFileDialog.getOpenFileName(
            self,
            "Open Delimited File",
            "",
            "Delimited Files (*.csv *.tsv *.tab *.txt);;All Files (*.*)",
        )
        if not path:
            return
        self._path_input.setText(path)
        self._handle_reload_clicked()

    def _handle_path_changed(self, text: str) -> None:
        self._selected_path = text.strip() or None
        self._mark_dirty()

    def _mark_dirty(self) -> None:
        if self._dataset_handle is not None:
            self._status_label.setText("Import options changed. Preview or apply to refresh the parsed dataset.")
            if hasattr(self, "_auto_send_checkbox") and self._auto_send_checkbox.isChecked():
                self._handle_apply_clicked()

    def _handle_reload_clicked(self) -> None:
        loaded = self._load_dataset_from_controls()
        if loaded is None:
            return
        dataset, resolved_options = loaded
        self._populate_from_handle(dataset, imported=False, resolved_options=resolved_options)

    def _handle_apply_clicked(self) -> None:
        loaded = self._load_dataset_from_controls()
        if loaded is None:
            return
        dataset, resolved_options = loaded
        self._populate_from_handle(dataset, imported=True, resolved_options=resolved_options)
        self._output_dataset = dataset
        for callback in self._import_callbacks:
            callback(dataset)
        self._notify_output_changed()

    def _load_dataset_from_controls(self) -> tuple[DatasetHandle, CSVImportOptions] | None:
        path = self._path_input.text().strip()
        if not path:
            self._status_label.setText("Select a delimited file first.")
            return None
        options = CSVImportOptions(
            delimiter=DELIMITER_OPTIONS[self._delimiter_combo.currentText()],
            has_header=self._has_header_checkbox.isChecked(),
            encoding=self._encoding_combo.currentText().lower(),
            skip_rows=self._skip_rows_spin.value(),
            auto_detect_delimiter=self._delimiter_combo.currentText() == "Auto",
        )
        try:
            resolved_options = self._import_service.resolve_delimited_options(path, options)
            dataset = self._import_service.load_delimited_text(path, resolved_options)
        except PortakalDataError as exc:
            self._dataset_handle = None
            self._resolved_options = None
            self._status_label.setText(str(exc))
            return None
        self._resolved_options = resolved_options
        return dataset, resolved_options

    def _populate_from_handle(
        self,
        dataset: DatasetHandle,
        *,
        imported: bool,
        resolved_options: CSVImportOptions | None,
    ) -> None:
        self._dataset_handle = dataset
        self._dataset_label.setText(dataset.display_name or dataset.source.path.name)
        action = "Imported" if imported else "Preview ready for"
        preview_df = dataset.dataframe.head(PREVIEW_ROW_LIMIT)
        self._status_label.setText(
            f"{action} {dataset.row_count} rows and {dataset.column_count} columns from {dataset.source.path.name}. "
            f"Preview shows first {preview_df.height} rows."
        )

        if resolved_options is not None:
            delimiter_label = next(
                (label for label, value in DELIMITER_OPTIONS.items() if label != "Auto" and value == resolved_options.delimiter),
                resolved_options.delimiter,
            )
            self._settings_label.setText(
                "\n".join(
                    [
                        f"Encoding: {resolved_options.encoding}",
                        f"Delimiter: {delimiter_label}",
                        f"Header row: {'Yes' if resolved_options.has_header else 'No'}",
                        f"Skipped rows: {resolved_options.skip_rows}",
                    ]
                )
            )
        else:
            self._settings_label.setText("Current workflow dataset loaded. Import options are not available for this source.")

        self._preview_table.setColumnCount(dataset.column_count)
        self._preview_table.setHorizontalHeaderLabels(list(preview_df.columns))
        self._preview_table.setRowCount(preview_df.height)
        for row_index, row in enumerate(preview_df.rows()):
            for column_index, value in enumerate(row):
                item = QTableWidgetItem("" if value is None else str(value))
                item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                self._preview_table.setItem(row_index, column_index, item)
        self._preview_table.resizeColumnsToContents()

    def _set_empty_state(self) -> None:
        self._dataset_handle = None
        self._output_dataset = None
        self._resolved_options = None
        self._dataset_label.setText("No imported dataset")
        self._status_label.setText("Choose a delimited file and preview it before importing.")
        self._settings_label.setText("")
        self._preview_table.setRowCount(0)
        self._preview_table.setColumnCount(0)

    def _reset_form(self) -> None:
        self._selected_path = None
        self._path_input.clear()
        self._delimiter_combo.setCurrentText("Auto")
        self._encoding_combo.setCurrentText("Auto")
        self._skip_rows_spin.setValue(0)
        self._has_header_checkbox.setChecked(True)
        self._set_empty_state()
