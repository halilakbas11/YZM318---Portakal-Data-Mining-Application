from __future__ import annotations

from dataclasses import replace
from typing import Callable

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QBrush
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
from portakal_app.data.models import (
    CSVImportOptions,
    ColumnSchema,
    DataDomain,
    DatasetHandle,
)
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

# Column type display labels and their mapping to internal logical_type
_TYPE_LABELS = ["Auto", "Numeric", "Categorical", "Text", "DateTime"]
_DISPLAY_TO_LOGICAL = {
    "Auto": None,       # keep inferred
    "Numeric": "numeric",
    "Categorical": "categorical",
    "Text": "text",
    "DateTime": "datetime",
}
_LOGICAL_TO_DISPLAY = {v: k for k, v in _DISPLAY_TO_LOGICAL.items() if v is not None}

# Column role display labels and their mapping to internal role
_ROLE_LABELS = ["Feature", "Target", "Meta", "Skip"]
_DISPLAY_TO_ROLE = {
    "Feature": "feature",
    "Target": "target",
    "Meta": "meta",
    "Skip": "skip",
}
_ROLE_TO_DISPLAY = {v: k.title() for k, v in _DISPLAY_TO_ROLE.items()}

# Row indices in the preview table
_ROW_TYPE = 0
_ROW_ROLE = 1
_DATA_OFFSET = 2

_CFG_BG = QColor(245, 245, 255)   # light blue-gray for config rows
_TARGET_BG = QColor(220, 255, 220) # light green for target role


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
        self._col_type_combos: list[QComboBox] = []
        self._col_role_combos: list[QComboBox] = []

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(10)

        layout.addWidget(self._build_source_group())
        layout.addWidget(self._build_options_group())
        layout.addWidget(self._build_info_group())
        layout.addWidget(self._build_preview_group(), 1)
        layout.addLayout(self._build_footer())

        self._set_empty_state()

    # ── UI builders ────────────────────────────────────────────────────

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
        self._delimiter_combo.currentTextChanged.connect(lambda _: self._mark_dirty())
        layout.addRow("Delimiter", self._delimiter_combo)

        self._encoding_combo = QComboBox(self)
        self._encoding_combo.addItems(list(ENCODING_OPTIONS))
        self._encoding_combo.currentTextChanged.connect(lambda _: self._mark_dirty())
        layout.addRow("Encoding", self._encoding_combo)

        self._skip_rows_spin = QSpinBox(self)
        self._skip_rows_spin.setRange(0, 100000)
        self._skip_rows_spin.valueChanged.connect(lambda _: self._mark_dirty())
        layout.addRow("Skip first rows", self._skip_rows_spin)

        self._has_header_checkbox = QCheckBox("First parsed row is header")
        self._has_header_checkbox.setChecked(True)
        self._has_header_checkbox.toggled.connect(lambda _: self._mark_dirty())
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
        group = QGroupBox("Preview  (Row 1: Type · Row 2: Role — edit before applying)")
        layout = QVBoxLayout(group)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(6)

        self._preview_table = QTableWidget(0, 0, self)
        self._preview_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._preview_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self._preview_table.horizontalHeader().setStretchLastSection(False)
        self._preview_table.setMinimumHeight(280)
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
        self._auto_send_checkbox.stateChanged.connect(self._on_auto_send_changed)
        layout.addWidget(self._auto_send_checkbox)

        self._apply_button = QPushButton("Apply Import")
        self._apply_button.setProperty("primary", True)
        self._apply_button.clicked.connect(self._handle_apply_clicked)
        layout.addWidget(self._apply_button)
        return layout

    # ── Public API ────────────────────────────────────────────────────

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
            "auto_send": self._auto_send_checkbox.isChecked(),
        }

    def restore_node_state(self, payload: dict[str, object]) -> None:
        self._path_input.setText(str(payload.get("path") or ""))
        self._delimiter_combo.setCurrentText(str(payload.get("delimiter") or "Auto"))
        self._encoding_combo.setCurrentText(str(payload.get("encoding") or "Auto"))
        self._skip_rows_spin.setValue(int(payload.get("skip_rows") or 0))
        self._has_header_checkbox.setChecked(bool(payload.get("has_header", True)))
        self._auto_send_checkbox.setChecked(bool(payload.get("auto_send", False)))
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
            self._preview_table.horizontalHeaderItem(i).text()
            for i in range(self._preview_table.columnCount())
        ]
        rows = []
        for row_idx in range(_DATA_OFFSET, self._preview_table.rowCount()):
            rows.append([
                (self._preview_table.item(row_idx, col_idx).text()
                 if self._preview_table.item(row_idx, col_idx) is not None else "")
                for col_idx in range(self._preview_table.columnCount())
            ])
        return {"summary": self._status_label.text(), "headers": headers, "rows": rows}

    def help_text(self) -> str:
        return (
            "Import delimited text data. Set the type and role for each column in "
            "the first two rows of the preview before applying."
        )

    def documentation_url(self) -> str:
        return "https://orangedatamining.com/widget-catalog/data/csvfileimport/"

    # ── Event handlers ────────────────────────────────────────────────

    def _handle_browse(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
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
            self._status_label.setText(
                "Options changed — click Preview to refresh, or Apply Import to commit."
            )

    def _on_auto_send_changed(self) -> None:
        if self._auto_send_checkbox.isChecked() and self._dataset_handle is not None:
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
        # Apply user-specified type/role overrides from the preview header rows
        dataset = self._apply_domain_overrides(dataset)
        self._populate_from_handle(dataset, imported=True, resolved_options=resolved_options)
        self._output_dataset = dataset
        for callback in self._import_callbacks:
            callback(dataset)
        self._notify_output_changed()

    # ── Core logic ────────────────────────────────────────────────────

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

    def _apply_domain_overrides(self, dataset: DatasetHandle) -> DatasetHandle:
        """Rebuild domain from whatever is currently shown in the type/role combo rows."""
        if not self._col_type_combos and not self._col_role_combos:
            return dataset

        new_cols: list[ColumnSchema] = []
        kept_cols: list[ColumnSchema] = []

        for col_idx, col in enumerate(dataset.domain.columns):
            role_display = (
                self._col_role_combos[col_idx].currentText()
                if col_idx < len(self._col_role_combos)
                else "Feature"
            )
            if role_display == "Skip":
                continue  # omit this column from the domain

            type_display = (
                self._col_type_combos[col_idx].currentText()
                if col_idx < len(self._col_type_combos)
                else "Auto"
            )
            logical_type = _DISPLAY_TO_LOGICAL.get(type_display) or col.logical_type
            role = _DISPLAY_TO_ROLE.get(role_display, "feature")

            new_cols.append(replace(col, logical_type=logical_type, role=role))
            kept_cols.append(col)

        # Drop skipped columns from the dataframe too
        kept_names = {c.name for c in new_cols}
        drop_names = [c for c in dataset.dataframe.columns if c not in kept_names]
        df = dataset.dataframe.drop(drop_names) if drop_names else dataset.dataframe

        new_domain = DataDomain(columns=tuple(new_cols))
        return replace(
            dataset,
            domain=new_domain,
            dataframe=df,
            column_count=df.width,
        )

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
        n_data_rows = preview_df.height
        n_cols = dataset.column_count

        self._status_label.setText(
            f"{action} {dataset.row_count} rows × {n_cols} columns "
            f"from {dataset.source.path.name}. "
            f"Showing first {n_data_rows} rows."
        )

        if resolved_options is not None:
            delimiter_label = next(
                (lbl for lbl, val in DELIMITER_OPTIONS.items()
                 if lbl != "Auto" and val == resolved_options.delimiter),
                resolved_options.delimiter,
            )
            self._settings_label.setText(
                f"Encoding: {resolved_options.encoding} · "
                f"Delimiter: {delimiter_label} · "
                f"Header: {'Yes' if resolved_options.has_header else 'No'} · "
                f"Skip rows: {resolved_options.skip_rows}"
            )
        else:
            self._settings_label.setText(
                "Loaded from workflow — import options not available."
            )

        # ── Rebuild preview table ─────────────────────────────────
        self._col_type_combos = []
        self._col_role_combos = []

        self._preview_table.setColumnCount(n_cols)
        self._preview_table.setRowCount(_DATA_OFFSET + n_data_rows)
        self._preview_table.setHorizontalHeaderLabels(list(preview_df.columns))
        self._preview_table.setVerticalHeaderLabels(
            ["Type", "Role"] + [str(i + 1) for i in range(n_data_rows)]
        )

        cfg_brush = QBrush(_CFG_BG)

        for col_idx, col_schema in enumerate(dataset.domain.columns):
            # ── Type row ──
            type_combo = QComboBox()
            type_combo.addItems(_TYPE_LABELS)
            current_type = _LOGICAL_TO_DISPLAY.get(col_schema.logical_type, "Auto")
            type_combo.setCurrentText(current_type)
            self._preview_table.setCellWidget(_ROW_TYPE, col_idx, type_combo)
            self._col_type_combos.append(type_combo)

            # ── Role row ──
            role_combo = QComboBox()
            role_combo.addItems(_ROLE_LABELS)
            current_role = _ROLE_TO_DISPLAY.get(col_schema.role, "Feature")
            role_combo.setCurrentText(current_role)
            role_combo.currentTextChanged.connect(
                lambda text, ci=col_idx: self._on_role_changed(ci, text)
            )
            self._preview_table.setCellWidget(_ROW_ROLE, col_idx, role_combo)
            self._col_role_combos.append(role_combo)

            # Apply background to the two config rows via placeholder items
            for cfg_row in (_ROW_TYPE, _ROW_ROLE):
                if self._preview_table.item(cfg_row, col_idx) is None:
                    placeholder = QTableWidgetItem()
                    placeholder.setBackground(cfg_brush)
                    placeholder.setFlags(Qt.ItemFlag.NoItemFlags)
                    self._preview_table.setItem(cfg_row, col_idx, placeholder)

        # ── Data rows ──
        for row_idx, row in enumerate(preview_df.rows()):
            for col_idx, value in enumerate(row):
                text = "" if value is None else str(value)
                item = QTableWidgetItem(text)
                item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                self._preview_table.setItem(_DATA_OFFSET + row_idx, col_idx, item)

        self._preview_table.resizeColumnsToContents()

        # Highlight target columns after filling
        for col_idx, col_schema in enumerate(dataset.domain.columns):
            self._on_role_changed(col_idx, _ROLE_TO_DISPLAY.get(col_schema.role, "Feature"))

    def _on_role_changed(self, col_idx: int, role_display: str) -> None:
        """Highlight data cells under a column when its role changes."""
        n_rows = self._preview_table.rowCount()
        if role_display == "Target":
            bg = QBrush(QColor(220, 255, 220))   # green
        elif role_display == "Meta":
            bg = QBrush(QColor(255, 245, 220))   # amber
        elif role_display == "Skip":
            bg = QBrush(QColor(230, 230, 230))   # gray
        else:
            bg = QBrush(QColor(255, 255, 255))   # white (feature)

        for row_idx in range(_DATA_OFFSET, n_rows):
            item = self._preview_table.item(row_idx, col_idx)
            if item is not None:
                item.setBackground(bg)

    # ── Empty / reset state ────────────────────────────────────────────

    def _set_empty_state(self) -> None:
        self._dataset_handle = None
        self._output_dataset = None
        self._resolved_options = None
        self._col_type_combos = []
        self._col_role_combos = []
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
