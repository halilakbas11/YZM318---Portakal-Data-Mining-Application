from __future__ import annotations

from collections.abc import Callable

from PySide6.QtCore import QSize, Qt, QUrl
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QPlainTextEdit,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from portakal_app.data.models import DatasetCatalogEntry, DatasetHandle
from portakal_app.data.services.dataset_catalog_service import DatasetCatalogService
from portakal_app.ui.screens.node_screen import WorkflowNodeScreenSupport
from portakal_app.ui.shared.cards import SectionHeader


class DatasetsScreen(QWidget, WorkflowNodeScreenSupport):
    def __init__(self, parent: QWidget | None = None, service: DatasetCatalogService | None = None) -> None:
        super().__init__(parent)
        self._init_workflow_node_support()
        self._service = service or DatasetCatalogService()
        self._entries = list(self._service.available_datasets())
        self._callbacks: list[Callable[[DatasetHandle], None]] = []
        self._current_dataset: DatasetHandle | None = None
        self._output_dataset: DatasetHandle | None = None
        self._filtered_entries: list[DatasetCatalogEntry] = []
        self._selection_guard = False

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(10)

        layout.addWidget(SectionHeader("Datasets", "Curated downloadable datasets inspired by Orange's online browser."))
        layout.addLayout(self._build_filters())
        layout.addWidget(self._build_table_panel(), 1)
        layout.addWidget(self._build_description_panel(), 1)
        layout.addLayout(self._build_footer())

        self._domain_combo.addItems(self._service.available_domains())
        self._domain_combo.currentTextChanged.connect(self._render_table)
        self._search_input.textChanged.connect(self._render_table)
        self._table.itemSelectionChanged.connect(self._handle_selection_changed)
        self._table.itemDoubleClicked.connect(lambda _item: self._send_selected_dataset())
        self._render_table()

    def sizeHint(self) -> QSize:
        return QSize(980, 720)

    def minimumSizeHint(self) -> QSize:
        return QSize(760, 620)

    def _build_filters(self) -> QHBoxLayout:
        layout = QHBoxLayout()
        layout.setSpacing(10)

        self._search_input = QLineEdit(self)
        self._search_input.setPlaceholderText("Search for data set ...")
        layout.addWidget(self._search_input, 1)

        label = QLabel("Domain:")
        label.setProperty("muted", True)
        layout.addWidget(label)

        self._domain_combo = QComboBox(self)
        self._domain_combo.setMinimumWidth(170)
        layout.addWidget(self._domain_combo)
        return layout

    def _build_table_panel(self) -> QFrame:
        frame = QFrame(self)
        frame.setProperty("panel", True)
        panel_layout = QVBoxLayout(frame)
        panel_layout.setContentsMargins(12, 12, 12, 12)
        panel_layout.setSpacing(8)

        self._summary_label = QLabel("")
        self._summary_label.setProperty("muted", True)
        panel_layout.addWidget(self._summary_label)

        self._table = QTableWidget(0, 6, self)
        self._table.setHorizontalHeaderLabels(["Title", "Size", "Instances", "Variables", "Target", "Tags"])
        self._table.verticalHeader().setVisible(False)
        self._table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self._table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self._table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._table.horizontalHeader().setStretchLastSection(True)
        panel_layout.addWidget(self._table, 1)
        return frame

    def _build_description_panel(self) -> QFrame:
        frame = QFrame(self)
        frame.setProperty("panel", True)
        panel_layout = QVBoxLayout(frame)
        panel_layout.setContentsMargins(12, 12, 12, 12)
        panel_layout.setSpacing(8)

        header = QLabel("Description")
        header.setProperty("sectionTitle", True)
        header.setStyleSheet("font-size: 12pt; background: transparent;")
        panel_layout.addWidget(header)

        self._description_label = QLabel("Select a dataset to review its description and source.")
        self._description_label.setWordWrap(True)
        panel_layout.addWidget(self._description_label)

        self._source_label = QLabel("")
        self._source_label.setProperty("muted", True)
        self._source_label.setWordWrap(True)
        panel_layout.addWidget(self._source_label)

        self._preview = QPlainTextEdit(self)
        self._preview.setReadOnly(True)
        self._preview.setPlaceholderText("Downloaded dataset preview will appear here.")
        self._preview.setMinimumHeight(150)
        panel_layout.addWidget(self._preview, 1)
        return frame

    def _build_footer(self) -> QHBoxLayout:
        layout = QHBoxLayout()
        layout.setSpacing(8)

        self._open_source_button = QPushButton("Open Source")
        self._open_source_button.setProperty("secondary", True)
        self._open_source_button.clicked.connect(self._open_selected_source)
        self._open_source_button.setEnabled(False)
        layout.addWidget(self._open_source_button)

        self._download_button = QPushButton("Download Preview")
        self._download_button.setProperty("secondary", True)
        self._download_button.clicked.connect(self._download_selected_preview)
        self._download_button.setEnabled(False)
        layout.addWidget(self._download_button)

        layout.addStretch(1)

        self._auto_send_checkbox = QCheckBox("Send Automatically")
        self._auto_send_checkbox.setChecked(True)
        layout.addWidget(self._auto_send_checkbox)

        self._send_button = QPushButton("Send Data")
        self._send_button.setProperty("primary", True)
        self._send_button.clicked.connect(self._send_selected_dataset)
        self._send_button.setEnabled(False)
        layout.addWidget(self._send_button)
        return layout

    def set_dataset_catalog_service(self, service: DatasetCatalogService) -> None:
        self._service = service
        self._entries = list(self._service.available_datasets())
        self._domain_combo.blockSignals(True)
        self._domain_combo.clear()
        self._domain_combo.addItems(self._service.available_domains())
        self._domain_combo.blockSignals(False)
        self._render_table()

    def on_dataset_selected(self, callback: Callable[[DatasetHandle], None]) -> None:
        self._callbacks.append(callback)

    def set_dataset(self, dataset: DatasetHandle | None) -> None:
        self._current_dataset = dataset
        if dataset is None:
            self._output_dataset = None

    def help_text(self) -> str:
        return (
            "Search the curated online repository, filter by domain, download a dataset into the local cache, "
            "and send it into the current workflow."
        )

    def documentation_url(self) -> str:
        return "https://orangedatamining.com/widget-catalog/data/datasets/"

    def footer_status_text(self) -> str:
        return f"{self._service.downloaded_count()} / {len(self._entries)}"

    def set_input_payload(self, payload) -> None:
        _ = payload

    def current_output_dataset(self) -> DatasetHandle | None:
        return self._output_dataset

    def serialize_node_state(self) -> dict[str, object]:
        selected_entry = self._selected_entry()
        return {
            "search_text": self._search_input.text(),
            "domain": self._domain_combo.currentText(),
            "selected_dataset_id": selected_entry.dataset_id if selected_entry is not None else "",
            "auto_send": self._auto_send_checkbox.isChecked(),
            "committed_dataset_id": self._output_dataset.dataset_id if self._output_dataset is not None else "",
        }

    def restore_node_state(self, payload: dict[str, object]) -> None:
        self._auto_send_checkbox.setChecked(bool(payload.get("auto_send", True)))
        self._search_input.setText(str(payload.get("search_text") or ""))
        self._domain_combo.setCurrentText(str(payload.get("domain") or "All"))
        selected_dataset_id = str(payload.get("selected_dataset_id") or "")
        committed_dataset_id = str(payload.get("committed_dataset_id") or "")
        if selected_dataset_id:
            self._select_dataset_id(selected_dataset_id)
        if committed_dataset_id and selected_dataset_id == committed_dataset_id:
            self._send_selected_dataset()
        elif selected_dataset_id:
            self._download_selected_preview()

    def report_snapshot(self) -> dict[str, object]:
        entry = self._selected_entry()
        details = [
            f"{len(self._entries)} datasets available",
            f"{self._service.downloaded_count()} cached locally",
        ]
        if entry is not None:
            details.extend(
                [
                    f"Selected: {entry.title}",
                    f"Domain: {entry.domain}",
                    f"Tags: {', '.join(entry.tags)}",
                ]
            )
        return {
            "title": "Datasets",
            "items": [
                {
                    "title": "Catalog",
                    "timestamp": "Current session",
                    "details": details,
                }
            ],
        }

    def data_preview_snapshot(self) -> dict[str, object]:
        if self._current_dataset is None:
            return {"summary": "No downloaded dataset selected.", "headers": [], "rows": []}
        headers = list(self._current_dataset.dataframe.columns)
        rows = [
            ["" if value is None else str(value) for value in row]
            for row in self._current_dataset.dataframe.head(50).rows()
        ]
        return {
            "summary": f"Downloaded dataset: {self._current_dataset.display_name}",
            "headers": headers,
            "rows": rows,
        }

    def _render_table(self) -> None:
        selected_dataset_id = self._selected_entry().dataset_id if self._selected_entry() is not None else None
        query = self._search_input.text().strip().lower()
        domain = self._domain_combo.currentText()
        self._filtered_entries = [
            entry
            for entry in self._entries
            if (domain in {"", "All"} or entry.domain == domain)
            and (
                not query
                or query in entry.title.lower()
                or query in entry.description.lower()
                or query in " ".join(entry.tags).lower()
            )
        ]

        self._table.setRowCount(len(self._filtered_entries))
        self._selection_guard = True
        for row_index, entry in enumerate(self._filtered_entries):
            values = [
                entry.title,
                entry.size_text,
                str(entry.row_count),
                str(entry.column_count),
                entry.target,
                ", ".join(entry.tags),
            ]
            for column_index, text in enumerate(values):
                item = QTableWidgetItem(text)
                item.setData(Qt.ItemDataRole.UserRole, entry.dataset_id)
                if column_index == 0 and self._service.is_downloaded(entry):
                    item.setText(f"{entry.title}  [cached]")
                self._table.setItem(row_index, column_index, item)
        self._table.resizeColumnsToContents()
        self._summary_label.setText(
            f"{len(self._filtered_entries)} datasets shown, {self._service.downloaded_count()} already cached locally."
        )
        if self._filtered_entries:
            target_row = 0
            if selected_dataset_id is not None:
                for row_index, entry in enumerate(self._filtered_entries):
                    if entry.dataset_id == selected_dataset_id:
                        target_row = row_index
                        break
            self._table.selectRow(target_row)
        else:
            self._description_label.setText("No datasets match the current filter.")
            self._source_label.clear()
            self._preview.clear()
            self._open_source_button.setEnabled(False)
            self._download_button.setEnabled(False)
            self._send_button.setEnabled(False)
        self._selection_guard = False
        self._update_selection_details(allow_auto_send=False)

    def _selected_entry(self) -> DatasetCatalogEntry | None:
        indexes = self._table.selectionModel().selectedRows() if self._table.selectionModel() is not None else []
        if not indexes:
            return None
        row = indexes[0].row()
        if not (0 <= row < len(self._filtered_entries)):
            return None
        return self._filtered_entries[row]

    def _handle_selection_changed(self) -> None:
        if self._selection_guard:
            return
        self._update_selection_details(allow_auto_send=True)

    def _update_selection_details(self, *, allow_auto_send: bool) -> None:
        entry = self._selected_entry()
        enabled = entry is not None
        self._open_source_button.setEnabled(enabled)
        self._download_button.setEnabled(enabled)
        self._send_button.setEnabled(enabled)
        if entry is None:
            return

        self._description_label.setText(entry.description)
        self._source_label.setText(
            f"Domain: {entry.domain} | Target: {entry.target} | Tags: {', '.join(entry.tags)}"
        )
        if allow_auto_send and self._auto_send_checkbox.isChecked():
            self._send_selected_dataset()
        elif allow_auto_send and self._service.is_downloaded(entry):
            self._download_selected_preview()
        elif self._service.is_downloaded(entry):
            self._preview.setPlainText("Dataset cached locally. Click Download Preview to inspect or Send Data to load it.")
        else:
            self._preview.setPlainText("This dataset has not been downloaded yet. Click Download Preview or Send Data.")

    def _open_selected_source(self) -> None:
        entry = self._selected_entry()
        if entry is None:
            return
        QDesktopServices.openUrl(QUrl(entry.download_url))

    def _download_selected_preview(self) -> None:
        entry = self._selected_entry()
        if entry is None:
            return
        try:
            dataset = self._service.load_or_download(entry)
        except Exception as exc:
            QMessageBox.warning(self, "Datasets", f"Dataset could not be downloaded.\n\n{exc}")
            return
        self._current_dataset = dataset
        preview_lines = [
            ", ".join(dataset.dataframe.columns),
            *[
                ", ".join("" if value is None else str(value) for value in row)
                for row in dataset.dataframe.head(12).rows()
            ],
        ]
        self._preview.setPlainText("\n".join(preview_lines))
        self._render_table()

    def _send_selected_dataset(self) -> None:
        entry = self._selected_entry()
        if entry is None:
            return
        try:
            dataset = self._service.load_or_download(entry)
        except Exception as exc:
            QMessageBox.warning(self, "Datasets", f"Dataset could not be sent.\n\n{exc}")
            return
        self._current_dataset = dataset
        self._output_dataset = dataset
        self._download_selected_preview()
        for callback in self._callbacks:
            callback(dataset)
        self._notify_output_changed()

    def _select_dataset_id(self, dataset_id: str) -> None:
        for row_index, entry in enumerate(self._filtered_entries):
            if entry.dataset_id != dataset_id:
                continue
            self._selection_guard = True
            self._table.selectRow(row_index)
            self._selection_guard = False
            self._update_selection_details(allow_auto_send=False)
            return
