from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable
from urllib.parse import urlparse

from PySide6.QtCore import QEvent, QObject, Qt, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLineEdit,
    QLabel,
    QPushButton,
    QCheckBox,
    QRadioButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from portakal_app.data.models import ColumnSchema, DatasetHandle, DomainEditRequest, DomainColumnEdit
from portakal_app.data.services.domain_transform_service import DomainTransformService
from portakal_app.data.services.file_import_service import FileImportService
from portakal_app.ui import i18n
from portakal_app.ui.icons import get_toolbar_icon
from portakal_app.ui.screens.node_screen import WorkflowNodeScreenSupport


@dataclass
class FileColumnSpec:
    name: str
    type_name: str
    role_name: str
    values_preview: str = ""


TYPE_BADGE_COLORS = {
    "numeric": "#d62828",
    "categorical": "#2ea44f",
    "text": "#6f42c1",
    "boolean": "#0f766e",
    "datetime": "#1d7ed6",
    "date": "#2563eb",
    "time": "#0891b2",
    "duration": "#7c3aed",
    "unknown": "#6f6a62",
}

TYPE_OPTIONS = ["categorical", "numeric", "text", "boolean", "datetime", "date", "time", "duration", "unknown"]
ROLE_OPTIONS = ["feature", "target", "meta", "skip"]
FILE_TYPE_OPTIONS = [
    "Determine type from the file extension",
    "Basket file (*.basket *.bsk)",
    "Comma-separated values (*.csv *.csv.gz *.gz *.csv.bz2 *.bz2 *.csv.xz *.xz)",
    "Microsoft Excel 97-2004 spreadsheet (*.xls)",
    "Microsoft Excel spreadsheet (*.xlsx)",
    "Pickled Orange data (*.pkl *.pickle *.pkl.gz *.pickle.gz *.gz *.pkl.bz2 *.pickle.bz2 *.bz2 *.pkl.xz *.pickle.xz *.xz)",
    "Tab-separated values (*.tab *.tsv *.tab.gz *.tsv.gz *.gz *.tab.bz2 *.tsv.bz2 *.bz2 *.tab.xz *.tsv.xz *.xz)",
]


class TypeCellWidget(QWidget):
    changed = Signal(str)

    def __init__(self, type_name: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        self._badge = QLabel(type_name[:1].upper())
        self._badge.setFixedWidth(18)
        self._badge.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._badge.setObjectName("fileTypeBadge")
        layout.addWidget(self._badge)

        self._combo = QComboBox()
        self._combo.addItems(TYPE_OPTIONS)
        if self._combo.findText(type_name) < 0:
            self._combo.addItem(type_name)
        self._combo.setCurrentText(type_name)
        self._combo.currentTextChanged.connect(self._handle_changed)
        self._combo.view().window().setWindowFlags(Qt.WindowType.Popup | Qt.WindowType.FramelessWindowHint) # Ensure popup draws correctly
        layout.addWidget(self._combo, 1)
        self._apply_badge(type_name)

    def _handle_changed(self, value: str) -> None:
        self._apply_badge(value)
        self.changed.emit(value)

    def _apply_badge(self, type_name: str) -> None:
        color = TYPE_BADGE_COLORS.get(type_name, "#6f6a62")
        self._badge.setText(type_name[:1].upper())
        self._badge.setStyleSheet(
            f"background-color: {color}; color: white; border-radius: 6px; font-weight: 700; padding: 1px 0;"
        )

    def current_text(self) -> str:
        return self._combo.currentText()


class RoleCellWidget(QComboBox):
    changed = Signal(str)

    def __init__(self, role_name: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.addItems(ROLE_OPTIONS)
        if self.findText(role_name) < 0:
            self.addItem(role_name)
        self.setCurrentText(role_name)
        self.currentTextChanged.connect(self.changed.emit)
        self.view().window().setWindowFlags(Qt.WindowType.Popup | Qt.WindowType.FramelessWindowHint)


class FileScreen(QWidget, WorkflowNodeScreenSupport):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._init_workflow_node_support()
        self._import_service = FileImportService()
        self._file_callbacks: list[Callable[[str], None]] = []
        self._reload_callbacks: list[Callable[[str], None]] = []
        self._apply_callbacks: list[Callable[[str], None]] = []
        self._url_callbacks: list[Callable[[str], None]] = []
        self._selected_path: str | None = None
        self._selected_url: str | None = None
        self._dataset_handle: DatasetHandle | None = None
        self._output_dataset: DatasetHandle | None = None
        self._dirty = False
        self._column_specs: list[FileColumnSpec] = []
        self._sample_rows: list[list[str]] = []
        self._sample_headers: list[str] = []
        self._row_count_hint: int | None = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(8)

        layout.addWidget(self._build_source_group())
        layout.addWidget(self._build_file_type_group())
        layout.addWidget(self._build_info_group())
        layout.addWidget(self._build_columns_group(), 1)
        layout.addLayout(self._build_footer())

        self._domain_transform_service = DomainTransformService()
        self._set_source_mode("file")
        self._set_info_placeholder()
        self._set_columns([])
        self._update_buttons()

    def eventFilter(self, watched: QObject, event: QEvent) -> bool:  # type: ignore[name-defined]
        if event.type() in {QEvent.Type.FocusIn, QEvent.Type.MouseButtonPress}:
            if watched in {self._file_combo, self._file_combo.lineEdit()} and not self._file_radio.isChecked():
                self._file_radio.setChecked(True)
            elif watched in {self._url_combo, self._url_combo.lineEdit()} and not self._url_radio.isChecked():
                self._url_radio.setChecked(True)
        return super().eventFilter(watched, event)

    def _build_source_group(self) -> QGroupBox:
        group = QGroupBox("Source")
        layout = QVBoxLayout(group)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(6)

        file_row = QHBoxLayout()
        file_row.setSpacing(8)
        self._file_radio = QRadioButton("File:")
        self._file_radio.setChecked(True)
        self._file_radio.toggled.connect(lambda checked: checked and self._set_source_mode("file"))
        file_row.addWidget(self._file_radio)

        self._file_combo = QComboBox()
        self._file_combo.setEditable(True)
        self._file_combo.setInsertPolicy(QComboBox.InsertPolicy.NoInsert)
        self._file_combo.lineEdit().setPlaceholderText("Select a local dataset...")
        self._file_combo.currentTextChanged.connect(self._handle_file_text_changed)
        self._file_combo.installEventFilter(self)
        self._file_combo.lineEdit().installEventFilter(self)
        file_row.addWidget(self._file_combo, 1)

        self._browse_button = QPushButton("...")
        self._browse_button.setObjectName("fileSourceActionButton")
        self._browse_button.setIcon(get_toolbar_icon("folder"))
        self._browse_button.clicked.connect(self._handle_open_clicked)
        self._browse_button.setFixedWidth(44)
        file_row.addWidget(self._browse_button)

        self._reload_button = QPushButton("Reload")
        self._reload_button.setObjectName("fileSourceActionButton")
        self._reload_button.setIcon(get_toolbar_icon("reload"))
        self._reload_button.clicked.connect(self._handle_reload_clicked)
        file_row.addWidget(self._reload_button)
        layout.addLayout(file_row)

        url_row = QHBoxLayout()
        url_row.setSpacing(8)
        self._url_radio = QRadioButton("URL:")
        self._url_radio.toggled.connect(lambda checked: checked and self._set_source_mode("url"))
        url_row.addWidget(self._url_radio)

        self._url_combo = QComboBox()
        self._url_combo.setEditable(True)
        self._url_combo.setInsertPolicy(QComboBox.InsertPolicy.NoInsert)
        self._url_combo.lineEdit().setPlaceholderText("Paste a remote dataset URL...")
        self._url_combo.currentTextChanged.connect(self._handle_url_text_changed)
        self._url_combo.installEventFilter(self)
        self._url_combo.lineEdit().installEventFilter(self)
        url_row.addWidget(self._url_combo, 1)
        layout.addLayout(url_row)

        kaggle_user_row = QHBoxLayout()
        kaggle_user_row.setSpacing(8)
        self._kaggle_user_label = QLabel("Kaggle User:")
        kaggle_user_row.addWidget(self._kaggle_user_label)

        self._kaggle_username_input = QLineEdit()
        self._kaggle_username_input.setPlaceholderText("Optional: your Kaggle username")
        self._kaggle_username_input.textChanged.connect(lambda _text: self._mark_dirty())
        kaggle_user_row.addWidget(self._kaggle_username_input, 1)
        layout.addLayout(kaggle_user_row)

        kaggle_key_row = QHBoxLayout()
        kaggle_key_row.setSpacing(8)
        self._kaggle_key_label = QLabel("Kaggle API Key:")
        kaggle_key_row.addWidget(self._kaggle_key_label)

        self._kaggle_key_input = QLineEdit()
        self._kaggle_key_input.setEchoMode(QLineEdit.EchoMode.Password)
        self._kaggle_key_input.setPlaceholderText("Optional: paste Kaggle API key")
        self._kaggle_key_input.textChanged.connect(lambda _text: self._mark_dirty())
        kaggle_key_row.addWidget(self._kaggle_key_input, 1)
        layout.addLayout(kaggle_key_row)
        return group

    def _build_file_type_group(self) -> QGroupBox:
        group = QGroupBox("File Type")
        layout = QVBoxLayout(group)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(6)

        self._file_type_combo = QComboBox()
        self._file_type_combo.addItems(FILE_TYPE_OPTIONS)
        self._file_type_combo.currentIndexChanged.connect(lambda _index: self._mark_dirty())
        layout.addWidget(self._file_type_combo)
        return group

    def _build_info_group(self) -> QGroupBox:
        group = QGroupBox("Info")
        layout = QVBoxLayout(group)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(4)

        self._dataset_title = QLabel("No dataset selected")
        self._dataset_title.setProperty("sectionTitle", True)
        self._dataset_title.setStyleSheet("font-size: 12.5pt;")
        layout.addWidget(self._dataset_title)

        self._dataset_description = QLabel("Choose a local file or URL to inspect metadata.")
        self._dataset_description.setWordWrap(True)
        layout.addWidget(self._dataset_description)

        self._dataset_metrics = QLabel("")
        self._dataset_metrics.setTextFormat(Qt.TextFormat.PlainText)
        self._dataset_metrics.setWordWrap(True)
        layout.addWidget(self._dataset_metrics)
        return group

    def _build_columns_group(self) -> QGroupBox:
        group = QGroupBox("Columns (Double click to edit)")
        layout = QVBoxLayout(group)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(6)

        self._columns_table = QTableWidget(0, 4)
        self._columns_table.setHorizontalHeaderLabels(["Name", "Type", "Role", "Values"])
        self._columns_table.verticalHeader().setVisible(True)
        self._columns_table.verticalHeader().setDefaultSectionSize(30)
        self._columns_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self._columns_table.setEditTriggers(
            QTableWidget.EditTrigger.DoubleClicked
            | QTableWidget.EditTrigger.EditKeyPressed
            | QTableWidget.EditTrigger.SelectedClicked
        )
        self._columns_table.itemChanged.connect(self._handle_item_changed)
        self._columns_table.horizontalHeader().setStretchLastSection(True)
        self._columns_table.setMinimumHeight(190)
        layout.addWidget(self._columns_table)
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

        self._apply_button = QPushButton("Apply")
        self._apply_button.setProperty("primary", True)
        self._apply_button.clicked.connect(self._handle_apply_clicked)
        layout.addWidget(self._apply_button)
        return layout

    def on_open_file_requested(self, callback: Callable[[str], None]) -> None:
        self._file_callbacks.append(callback)

    def on_reload_requested(self, callback: Callable[[str], None]) -> None:
        self._reload_callbacks.append(callback)

    def on_apply_requested(self, callback: Callable[[str], None]) -> None:
        self._apply_callbacks.append(callback)

    def on_open_url_requested(self, callback: Callable[[str], None]) -> None:
        self._url_callbacks.append(callback)

    def set_selected_file(self, dataset: DatasetHandle | str | None) -> None:
        if dataset is None:
            self._dataset_handle = None
            self._output_dataset = None
            self._selected_path = None
            self._file_combo.setEditText("")
            self._set_info_placeholder()
            self._set_columns([])
            self._sample_headers = []
            self._sample_rows = []
            self._row_count_hint = None
            self._dirty = False
            self._update_buttons()
            return

        if isinstance(dataset, DatasetHandle):
            self._dataset_handle = dataset
            self._selected_path = str(dataset.source.path)
        else:
            self._selected_path = dataset
            self._dataset_handle = self._load_dataset_handle(dataset)

        if self._selected_path:
            self._file_radio.setChecked(True)
            self._ensure_combo_value(self._file_combo, self._selected_path)

        if self._dataset_handle is not None:
            self._populate_from_handle(self._dataset_handle)
            self._output_dataset = self._dataset_handle
        else:
            self._set_info_placeholder()
            self._set_columns([])
            self._output_dataset = None
        self._dirty = False
        self._update_buttons()
        self._notify_output_changed()

    def set_remote_url(self, url: str | None) -> None:
        self._selected_url = url
        self._dataset_handle = None
        self._output_dataset = None
        if url:
            self._url_radio.setChecked(True)
            self._ensure_combo_value(self._url_combo, url)
            try:
                self._dataset_handle = self._import_service.load_from_url(
                    url,
                    kaggle_username=self._kaggle_username_input.text().strip() or None,
                    kaggle_key=self._kaggle_key_input.text().strip() or None,
                )
                self._populate_from_handle(self._dataset_handle)
                self._output_dataset = self._dataset_handle
            except Exception as e:
                self._set_info_placeholder()
                self._dataset_title.setText(i18n.t("Error loading URL"))
                self._dataset_description.setText(i18n.t(str(e)))
                self._set_columns([])
        else:
            self._url_combo.setEditText("")
            self._set_info_placeholder()
            self._set_columns([])
            
        self._dirty = False
        self._update_buttons()
        self._notify_output_changed()

    def current_source_value(self) -> str | None:
        if self._file_radio.isChecked():
            text = self._file_combo.currentText().strip()
            return text or None
        text = self._url_combo.currentText().strip()
        return text or None

    def documentation_url(self) -> str:
        return "https://orangedatamining.com/widget-catalog/data/file/"

    def help_text(self) -> str:
        return (
            "Load a local or remote dataset, review inferred column types and roles, "
            "then apply the selected source configuration."
        )

    def footer_status_text(self) -> str:
        if self._row_count_hint is not None:
            return str(self._row_count_hint)
        return "0"

    def set_input_payload(self, payload) -> None:
        _ = payload

    def current_output_dataset(self) -> DatasetHandle | None:
        return self._output_dataset

    def serialize_node_state(self) -> dict[str, object]:
        return {
            "source_mode": "file" if self._file_radio.isChecked() else "url",
            "selected_path": self._selected_path or "",
            "selected_url": self._selected_url or "",
            "file_type_index": self._file_type_combo.currentIndex(),
            "auto_send": getattr(self, "_auto_send_checkbox", None) is not None and self._auto_send_checkbox.isChecked(),
        }

    def restore_node_state(self, payload: dict[str, object]) -> None:
        source_mode = str(payload.get("source_mode") or "file")
        self._file_type_combo.setCurrentIndex(int(payload.get("file_type_index") or 0))
        if hasattr(self, "_auto_send_checkbox"):
            self._auto_send_checkbox.setChecked(bool(payload.get("auto_send", True)))
        if source_mode == "url":
            self.set_remote_url(str(payload.get("selected_url") or "") or None)
            return
        self.set_selected_file(str(payload.get("selected_path") or "") or None)

    def report_snapshot(self) -> dict[str, object]:
        source = self.current_source_value() or "No source selected"
        file_format = self._file_type_combo.currentText()
        feature_names = [spec.name for spec in self._column_specs if spec.role_name == "feature"]
        target_names = [spec.name for spec in self._column_specs if spec.role_name == "target"]
        meta_names = [spec.name for spec in self._column_specs if spec.role_name == "meta"]
        return {
            "title": self._dataset_title.text(),
            "items": [
                {
                    "title": "File",
                    "timestamp": "Current session",
                    "details": [
                        f"File name: {source}",
                        f"Format: {file_format}",
                    ],
                },
                {
                    "title": "Data",
                    "timestamp": "",
                    "details": [
                        f"Data instances: {self._row_count_hint or 0}",
                        f"Features: {', '.join(feature_names) if feature_names else 'none'}",
                        f"Target: {', '.join(target_names) if target_names else 'none'}",
                        f"Meta: {', '.join(meta_names) if meta_names else 'none'}",
                    ],
                },
            ],
        }

    def data_preview_snapshot(self) -> dict[str, object]:
        headers, rows = self._full_table_from_current_source()
        return {
            "summary": self._dataset_metrics.text(),
            "headers": headers or self._sample_headers or [spec.name for spec in self._column_specs],
            "rows": rows or self._sample_rows,
        }

    def _handle_file_text_changed(self, text: str) -> None:
        self._selected_path = text.strip() or None
        if self._file_radio.isChecked():
            self._mark_dirty()

    def _handle_url_text_changed(self, text: str) -> None:
        self._selected_url = text.strip() or None
        if self._url_radio.isChecked():
            self._mark_dirty()

    def _set_source_mode(self, mode: str) -> None:
        file_mode = mode == "file"
        url_mode = mode == "url"
        self._browse_button.setEnabled(file_mode)
        self._reload_button.setEnabled(bool(self.current_source_value()))
        for widget in (
            self._kaggle_user_label,
            self._kaggle_username_input,
            self._kaggle_key_label,
            self._kaggle_key_input,
        ):
            widget.setVisible(url_mode)
        self._update_buttons()

    def _handle_open_clicked(self) -> None:
        selected, _selected_filter = QFileDialog.getOpenFileName(
            self,
            "Open Dataset",
            "",
            "Data Files (*.csv *.tsv *.tab *.xlsx *.xls *.parquet *.pkl *.pickle);;All Files (*.*)",
        )
        if not selected:
            return
        if self._file_callbacks:
            for callback in self._file_callbacks:
                callback(selected)
        self.set_selected_file(selected)

    def _handle_reload_clicked(self) -> None:
        source = self.current_source_value()
        if not source:
            return
        callbacks = self._reload_callbacks if self._file_radio.isChecked() else self._url_callbacks
        if callbacks:
            for callback in callbacks:
                callback(source)
        if self._file_radio.isChecked():
            self.set_selected_file(source)
        else:
            self.set_remote_url(source)
        self._dirty = False
        self._update_buttons()

    def _handle_apply_clicked(self) -> None:
        source = self.current_source_value()
        if not source:
            self._dataset_title.setText(i18n.t("No dataset selected"))
            self._dataset_description.setText(i18n.t("Select a file path or URL before applying."))
            return

        if self._file_radio.isChecked() and source != self._selected_path:
            self.set_selected_file(source)
        elif self._url_radio.isChecked():
            # Re-load URL source on each apply so updated credential fields take effect.
            self.set_remote_url(source)

        if self._dataset_handle is not None:
            request = self._build_domain_request()
            try:
                self._output_dataset = self._domain_transform_service.apply(self._dataset_handle, request)
            except Exception as exc:
                self._dataset_title.setText(i18n.t("Apply failed"))
                self._dataset_description.setText(str(exc))
                self._update_buttons()
                return

        callbacks = self._apply_callbacks
        if self._url_radio.isChecked():
            callbacks = self._url_callbacks or self._apply_callbacks
        if callbacks:
            for callback in callbacks:
                callback(source)

        self._dirty = False
        self._update_buttons()
        self._notify_output_changed()

    def _build_domain_request(self) -> DomainEditRequest:
        if self._dataset_handle is None:
            return DomainEditRequest()
        columns = []
        for row_index, original_column in enumerate(self._dataset_handle.domain.columns):
            if row_index >= self._columns_table.rowCount():
                continue
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

    def _reset_form(self) -> None:
        self._selected_path = None
        self._selected_url = None
        self._dataset_handle = None
        self._sample_headers = []
        self._sample_rows = []
        self._row_count_hint = None
        self._file_radio.setChecked(True)
        self._file_combo.setEditText("")
        self._url_combo.setEditText("")
        self._file_type_combo.setCurrentIndex(0)
        self._set_info_placeholder()
        self._set_columns([])
        self._dirty = False
        self._update_buttons()

    def _load_dataset_handle(self, path: str) -> DatasetHandle | None:
        try:
            return self._import_service.load(path)
        except Exception:
            return None

    def _populate_from_handle(self, dataset: DatasetHandle) -> None:
        self._dataset_title.setText(dataset.display_name or i18n.t("Dataset"))
        self._dataset_description.setText(i18n.t("Local file source"))
        self._sync_file_type_from_extension(dataset.source.path.suffix.lower())

        self._sample_headers = list(dataset.dataframe.columns)
        self._sample_rows = self._rows_as_text(dataset.dataframe.head(12))
        self._row_count_hint = dataset.row_count
        metrics = [
            f"{i18n.t('Source path')}: {dataset.source.path}",
            f"{i18n.t('Detected extension')}: {dataset.source.path.suffix.lower() or i18n.t('unknown')}",
            f"{dataset.column_count} {i18n.t('columns discovered')}",
            f"{dataset.row_count} {i18n.t('rows detected')}",
            f"{i18n.t('Cache file')}: {dataset.cache_path.name}",
        ]
        self._dataset_metrics.setText("\n".join(metrics))
        self._set_columns(self._columns_from_domain(dataset))

    def _populate_from_url(self, url: str) -> None:
        pass # Now handled directly via self._import_service.load_from_url

    def _columns_from_domain(self, dataset: DatasetHandle) -> list[FileColumnSpec]:
        return [self._column_from_schema(column) for column in dataset.domain.columns]

    def _column_from_schema(self, column: ColumnSchema) -> FileColumnSpec:
        preview = ", ".join(column.sample_values)
        return FileColumnSpec(
            name=column.name,
            type_name=column.logical_type,
            role_name=column.role,
            values_preview=preview,
        )

    def _full_table_from_current_source(self) -> tuple[list[str], list[list[str]]]:
        dataset = self._output_dataset or self._dataset_handle
        if dataset is None:
            return self._sample_headers, self._sample_rows
        return list(dataset.dataframe.columns), self._rows_as_text(dataset.dataframe)

    def _rows_as_text(self, dataframe) -> list[list[str]]:
        return [[self._cell_to_text(value) for value in row] for row in dataframe.rows()]

    def _cell_to_text(self, value: object) -> str:
        return "" if value is None else str(value)

    def _sync_file_type_from_extension(self, suffix: str) -> None:
        mapping = {
            ".csv": 2,
            ".gz": 2,
            ".xls": 3,
            ".xlsx": 4,
            ".pkl": 5,
            ".pickle": 5,
            ".tab": 6,
            ".tsv": 6,
        }
        self._file_type_combo.setCurrentIndex(mapping.get(suffix, 0))

    def _set_info_placeholder(self) -> None:
        self._dataset_title.setText(i18n.t("No dataset selected"))
        self._dataset_description.setText(i18n.t("Choose a local file or URL to inspect metadata."))
        self._dataset_metrics.setText(
            "\n".join(
                [
                    i18n.t("No source selected yet."),
                    i18n.t("Use Browse, Reload and Apply after the data backend is connected."),
                ]
            )
        )

    def _set_columns(self, columns: list[FileColumnSpec]) -> None:
        self._column_specs = [FileColumnSpec(spec.name, spec.type_name, spec.role_name, spec.values_preview) for spec in columns]
        self._columns_table.blockSignals(True)
        self._columns_table.setRowCount(len(columns))
        for row, spec in enumerate(self._column_specs):
            name_item = QTableWidgetItem(spec.name)
            name_item.setFlags(name_item.flags() | Qt.ItemFlag.ItemIsEditable)
            self._columns_table.setItem(row, 0, name_item)

            type_widget = TypeCellWidget(spec.type_name)
            type_widget.changed.connect(lambda value, row_index=row: self._handle_type_changed(row_index, value))
            self._columns_table.setCellWidget(row, 1, type_widget)

            role_widget = RoleCellWidget(spec.role_name)
            role_widget.changed.connect(lambda value, row_index=row: self._handle_role_changed(row_index, value))
            self._columns_table.setCellWidget(row, 2, role_widget)

            values_item = QTableWidgetItem(spec.values_preview)
            values_item.setFlags(values_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self._columns_table.setItem(row, 3, values_item)

        self._columns_table.blockSignals(False)
        self._columns_table.resizeColumnsToContents()
        self._columns_table.setEnabled(bool(columns))

    def _handle_type_changed(self, row: int, type_name: str) -> None:
        self._column_specs[row].type_name = type_name
        self._mark_dirty()

    def _handle_role_changed(self, row: int, role_name: str) -> None:
        self._column_specs[row].role_name = role_name
        self._mark_dirty()

    def _handle_item_changed(self, item: QTableWidgetItem) -> None:
        if item.column() == 0 and 0 <= item.row() < len(self._column_specs):
            self._column_specs[item.row()].name = item.text()
            self._mark_dirty()

    def _ensure_combo_value(self, combo: QComboBox, value: str) -> None:
        index = combo.findText(value)
        if index < 0:
            combo.insertItem(0, value)
            index = 0
        combo.setCurrentIndex(index)
        combo.setEditText(value)

    def _mark_dirty(self) -> None:
        self._dirty = True
        self._update_buttons()
        if hasattr(self, "_auto_send_checkbox") and self._auto_send_checkbox.isChecked():
            self._handle_apply_clicked()

    def _update_buttons(self) -> None:
        has_source = bool(self.current_source_value())
        self._reload_button.setEnabled(has_source)
        self._apply_button.setEnabled(has_source and self._dirty)
