from __future__ import annotations

import csv
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from PySide6.QtCore import QSize
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from portakal_app.models import WorkflowPayload
from portakal_app.ui.screens.corpus_screen import CorpusDocument, count_words
from portakal_app.ui.screens.node_screen import WorkflowNodeScreenSupport
from portakal_app.ui.shared.cards import SectionHeader


SUPPORTED_DOCUMENT_EXTENSIONS = frozenset({".txt", ".md", ".csv"})
TEXT_ENCODINGS = ("utf-8", "utf-8-sig", "cp1254", "latin-1")
AUTO_ENCODING = "auto"

DUPLICATE_SKIP = "skip"
DUPLICATE_REPLACE = "replace"
DUPLICATE_ALLOW = "allow"
DUPLICATE_POLICIES = (DUPLICATE_SKIP, DUPLICATE_REPLACE, DUPLICATE_ALLOW)


@dataclass(frozen=True)
class CSVColumnMapping:
    title_column: str = ""
    text_column: str = ""
    source_column: str = ""


@dataclass(frozen=True)
class DocumentImportOptions:
    recursive: bool = True
    category_from_subfolders: bool = True
    encoding: str = AUTO_ENCODING
    duplicate_policy: str = DUPLICATE_SKIP
    append_to_existing: bool = False
    csv_mapping: CSVColumnMapping = field(default_factory=CSVColumnMapping)


@dataclass(frozen=True)
class SkippedDocument:
    path: str
    extension: str
    reason: str
    message: str


@dataclass(frozen=True)
class DocumentImportResult:
    documents: tuple[CorpusDocument, ...]
    skipped: tuple[SkippedDocument, ...] = ()
    errors: tuple[str, ...] = ()


@dataclass(frozen=True)
class _ImportPathItem:
    path: Path
    root: Path | None = None


def is_supported_document_path(path: str | Path) -> bool:
    return Path(path).suffix.lower() in SUPPORTED_DOCUMENT_EXTENSIONS


def read_text_file(path: str | Path, encoding: str = AUTO_ENCODING) -> str:
    file_path = Path(path)
    if encoding != AUTO_ENCODING:
        return file_path.read_text(encoding=encoding)

    last_error: UnicodeDecodeError | None = None
    for candidate in TEXT_ENCODINGS:
        try:
            return file_path.read_text(encoding=candidate)
        except UnicodeDecodeError as error:
            last_error = error
    if last_error is not None:
        raise last_error
    return file_path.read_text(encoding="utf-8")


def collect_document_paths(
    paths: Sequence[str | Path],
    *,
    recursive: bool = True,
) -> tuple[tuple[_ImportPathItem, ...], tuple[SkippedDocument, ...]]:
    items: list[_ImportPathItem] = []
    skipped: list[SkippedDocument] = []

    for path in paths:
        file_path = Path(path)
        if file_path.is_dir():
            candidates = file_path.rglob("*") if recursive else file_path.iterdir()
            for candidate in sorted((item for item in candidates if item.is_file()), key=lambda item: item.as_posix()):
                if is_supported_document_path(candidate):
                    items.append(_ImportPathItem(candidate, file_path))
                else:
                    skipped.append(_skip(candidate, "Unsupported file type"))
            continue

        if is_supported_document_path(file_path):
            items.append(_ImportPathItem(file_path, None))
        else:
            skipped.append(_skip(file_path, "Unsupported file type"))

    return tuple(items), tuple(skipped)


def document_from_path(
    path: str | Path,
    *,
    options: DocumentImportOptions | None = None,
    root: Path | None = None,
) -> tuple[CorpusDocument, ...]:
    file_path = Path(path)
    active_options = options or DocumentImportOptions()
    source = _source_for_file(file_path, root, active_options)
    suffix = file_path.suffix.lower()
    if suffix == ".csv":
        return _documents_from_csv(file_path, source, active_options)
    text = read_text_file(file_path, active_options.encoding)
    return (CorpusDocument(file_path.name, text, source),)


def import_documents_from_paths(
    paths: Sequence[str | Path],
    *,
    options: DocumentImportOptions | None = None,
    existing_documents: Sequence[CorpusDocument] = (),
) -> DocumentImportResult:
    active_options = options or DocumentImportOptions()
    documents: list[CorpusDocument] = []
    skipped: list[SkippedDocument] = []

    path_items, path_skips = collect_document_paths(paths, recursive=active_options.recursive)
    skipped.extend(path_skips)

    for item in path_items:
        try:
            documents.extend(document_from_path(item.path, options=active_options, root=item.root))
        except OSError as error:
            skipped.append(_skip(item.path, "Read error", f"Could not read {item.path.name}: {error}"))
        except UnicodeError as error:
            skipped.append(_skip(item.path, "Decode error", f"Could not decode {item.path.name}: {error}"))
        except csv.Error as error:
            skipped.append(_skip(item.path, "CSV parse error", f"Could not parse {item.path.name}: {error}"))

    imported_documents, duplicate_skips = _apply_duplicate_policy(
        documents,
        existing_documents=existing_documents if active_options.append_to_existing else (),
        duplicate_policy=active_options.duplicate_policy,
    )
    skipped.extend(duplicate_skips)
    return _result(imported_documents, skipped)


def _documents_from_csv(
    path: Path,
    source: str,
    options: DocumentImportOptions,
) -> tuple[CorpusDocument, ...]:
    text = read_text_file(path, options.encoding)
    rows = list(csv.reader(text.splitlines()))
    mapping = options.csv_mapping
    if mapping.title_column or mapping.text_column or mapping.source_column:
        return _mapped_documents_from_csv_rows(path, rows, source, mapping)
    return _default_documents_from_csv_rows(path, rows, source)


def _default_documents_from_csv_rows(path: Path, rows: Sequence[Sequence[str]], source: str) -> tuple[CorpusDocument, ...]:
    documents: list[CorpusDocument] = []
    for index, row in enumerate(rows, start=1):
        joined = " ".join(cell.strip() for cell in row if cell.strip())
        if not joined:
            continue
        documents.append(CorpusDocument(f"{path.name} row {index}", joined, source))
    return tuple(documents)


def _mapped_documents_from_csv_rows(
    path: Path,
    rows: Sequence[Sequence[str]],
    default_source: str,
    mapping: CSVColumnMapping,
) -> tuple[CorpusDocument, ...]:
    if not rows:
        return ()

    headers = tuple(cell.strip() for cell in rows[0])
    documents: list[CorpusDocument] = []
    for index, row in enumerate(rows[1:], start=1):
        title = _column_value(row, headers, mapping.title_column).strip()
        text = _column_value(row, headers, mapping.text_column).strip()
        source = _column_value(row, headers, mapping.source_column).strip()
        if not text:
            text = " ".join(cell.strip() for cell in row if cell.strip())
        if not text:
            continue
        documents.append(
            CorpusDocument(
                title or f"{path.name} row {index}",
                text,
                source or default_source,
            )
        )
    return tuple(documents)


def _column_value(row: Sequence[str], headers: Sequence[str], selector: str) -> str:
    normalized = selector.strip()
    if not normalized:
        return ""
    if normalized.isdigit():
        index = int(normalized) - 1
        return row[index] if 0 <= index < len(row) else ""

    header_lookup = {header.casefold(): index for index, header in enumerate(headers)}
    index = header_lookup.get(normalized.casefold())
    if index is None:
        return ""
    return row[index] if index < len(row) else ""


def _source_for_file(path: Path, root: Path | None, options: DocumentImportOptions) -> str:
    if root is None or not options.category_from_subfolders:
        return str(path)
    try:
        relative_parent = path.parent.relative_to(root)
    except ValueError:
        return str(path)
    if str(relative_parent) == ".":
        return root.name
    return relative_parent.as_posix()


def _apply_duplicate_policy(
    documents: Sequence[CorpusDocument],
    *,
    existing_documents: Sequence[CorpusDocument],
    duplicate_policy: str,
) -> tuple[tuple[CorpusDocument, ...], tuple[SkippedDocument, ...]]:
    if duplicate_policy not in DUPLICATE_POLICIES:
        duplicate_policy = DUPLICATE_SKIP
    if duplicate_policy == DUPLICATE_ALLOW:
        return (*existing_documents, *documents), ()

    merged = list(existing_documents)
    positions = {_document_key(document): index for index, document in enumerate(merged)}
    skipped: list[SkippedDocument] = []
    for document in documents:
        key = _document_key(document)
        if key not in positions:
            positions[key] = len(merged)
            merged.append(document)
            continue
        if duplicate_policy == DUPLICATE_REPLACE:
            merged[positions[key]] = document
        else:
            skipped.append(
                SkippedDocument(
                    document.source,
                    Path(document.source).suffix.lower(),
                    "Duplicate document",
                    f"Duplicate document skipped: {document.title}",
                )
            )
    return tuple(merged), tuple(skipped)


def _document_key(document: CorpusDocument) -> tuple[str, str]:
    return (document.source, document.title)


def _skip(path: Path, reason: str, message: str | None = None) -> SkippedDocument:
    return SkippedDocument(
        str(path),
        path.suffix.lower(),
        reason,
        message or f"{reason}: {path.name}",
    )


def _result(documents: Sequence[CorpusDocument], skipped: Sequence[SkippedDocument]) -> DocumentImportResult:
    skipped_items = tuple(skipped)
    return DocumentImportResult(
        tuple(documents),
        skipped_items,
        tuple(item.message for item in skipped_items),
    )


class ImportDocumentsScreen(QWidget, WorkflowNodeScreenSupport):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._init_workflow_node_support()
        self._documents: tuple[CorpusDocument, ...] = ()
        self._skipped: tuple[SkippedDocument, ...] = ()
        self._errors: tuple[str, ...] = ()
        self._last_import_paths: tuple[str | Path, ...] = ()

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(10)

        layout.addWidget(
            SectionHeader(
                "Import Documents",
                "Import local .txt, .md, or .csv files and folders into a lightweight text corpus.",
            )
        )
        layout.addWidget(self._build_actions_panel())
        layout.addWidget(self._build_options_panel())
        layout.addWidget(self._build_table_panel(), 1)

        self._render()

    def sizeHint(self) -> QSize:
        return QSize(1020, 720)

    def minimumSizeHint(self) -> QSize:
        return QSize(820, 560)

    def _build_actions_panel(self) -> QFrame:
        frame = QFrame(self)
        frame.setProperty("panel", True)
        layout = QHBoxLayout(frame)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        self._import_button = QPushButton("Import Files", self)
        self._import_button.setProperty("primary", True)
        self._import_button.clicked.connect(self._choose_files)
        layout.addWidget(self._import_button)

        self._import_folder_button = QPushButton("Import Folder", self)
        self._import_folder_button.setProperty("secondary", True)
        self._import_folder_button.clicked.connect(self._choose_folder)
        layout.addWidget(self._import_folder_button)

        self._reload_button = QPushButton("Reload Last Import", self)
        self._reload_button.setProperty("secondary", True)
        self._reload_button.clicked.connect(self.reload_last_import)
        layout.addWidget(self._reload_button)

        self._remove_button = QPushButton("Remove Selected", self)
        self._remove_button.setProperty("secondary", True)
        self._remove_button.clicked.connect(self.remove_selected_document)
        layout.addWidget(self._remove_button)

        self._clear_button = QPushButton("Clear Corpus", self)
        self._clear_button.setProperty("secondary", True)
        self._clear_button.clicked.connect(self.clear_imported_documents)
        layout.addWidget(self._clear_button)

        self._summary_label = QLabel("", self)
        self._summary_label.setProperty("muted", True)
        self._summary_label.setWordWrap(True)
        layout.addWidget(self._summary_label, 1)
        return frame

    def _build_options_panel(self) -> QFrame:
        frame = QFrame(self)
        frame.setProperty("panel", True)
        layout = QGridLayout(frame)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setHorizontalSpacing(10)
        layout.setVerticalSpacing(8)

        self._recursive_checkbox = QCheckBox("Recursive folders", self)
        self._recursive_checkbox.setChecked(True)
        layout.addWidget(self._recursive_checkbox, 0, 0)

        self._category_checkbox = QCheckBox("Use subfolder as source/category", self)
        self._category_checkbox.setChecked(True)
        layout.addWidget(self._category_checkbox, 0, 1)

        self._append_checkbox = QCheckBox("Append to current corpus", self)
        layout.addWidget(self._append_checkbox, 0, 2)

        layout.addWidget(QLabel("Encoding", self), 1, 0)
        self._encoding_combo = QComboBox(self)
        self._encoding_combo.addItem("Auto", AUTO_ENCODING)
        for encoding in TEXT_ENCODINGS:
            self._encoding_combo.addItem(encoding, encoding)
        layout.addWidget(self._encoding_combo, 1, 1)

        layout.addWidget(QLabel("Duplicates", self), 1, 2)
        self._duplicate_combo = QComboBox(self)
        self._duplicate_combo.addItem("Skip duplicates", DUPLICATE_SKIP)
        self._duplicate_combo.addItem("Replace duplicates", DUPLICATE_REPLACE)
        self._duplicate_combo.addItem("Allow duplicates", DUPLICATE_ALLOW)
        layout.addWidget(self._duplicate_combo, 1, 3)

        layout.addWidget(QLabel("CSV title column", self), 2, 0)
        self._csv_title_input = self._build_mapping_input("title or 1")
        layout.addWidget(self._csv_title_input, 2, 1)

        layout.addWidget(QLabel("CSV text column", self), 2, 2)
        self._csv_text_input = self._build_mapping_input("text/body or 2")
        layout.addWidget(self._csv_text_input, 2, 3)

        layout.addWidget(QLabel("CSV source column", self), 3, 0)
        self._csv_source_input = self._build_mapping_input("category/source or 3")
        layout.addWidget(self._csv_source_input, 3, 1)

        help_label = QLabel(
            "CSV mapping is optional. Leave fields empty to keep the default row-join behavior.",
            self,
        )
        help_label.setProperty("muted", True)
        help_label.setWordWrap(True)
        layout.addWidget(help_label, 3, 2, 1, 2)
        return frame

    def _build_mapping_input(self, placeholder: str) -> QLineEdit:
        input_widget = QLineEdit(self)
        input_widget.setPlaceholderText(placeholder)
        input_widget.setStyleSheet(
            "background: #fffdf9; color: #2b2b2b; border: 1px solid #d1cabf; "
            "border-radius: 8px; padding: 6px 10px;"
        )
        return input_widget

    def _build_table_panel(self) -> QFrame:
        frame = QFrame(self)
        frame.setProperty("panel", True)
        layout = QVBoxLayout(frame)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)

        self._status_label = QLabel("", self)
        self._status_label.setProperty("muted", True)
        self._status_label.setWordWrap(True)
        layout.addWidget(self._status_label)

        self._tabs = QTabWidget(self)
        self._table = QTableWidget(0, 4, self)
        self._table.setHorizontalHeaderLabels(["Title", "Source", "Text Preview", "Words"])
        self._table.verticalHeader().setVisible(False)
        self._table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self._table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self._table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._table.horizontalHeader().setStretchLastSection(True)
        self._table.itemSelectionChanged.connect(self._update_preview)
        self._tabs.addTab(self._table, "Imported Documents")

        self._skipped_table = QTableWidget(0, 3, self)
        self._skipped_table.setHorizontalHeaderLabels(["Path", "Extension", "Reason"])
        self._skipped_table.verticalHeader().setVisible(False)
        self._skipped_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self._skipped_table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self._skipped_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._skipped_table.horizontalHeader().setStretchLastSection(True)
        self._tabs.addTab(self._skipped_table, "Skipped Documents")

        layout.addWidget(self._tabs, 1)

        self._preview_label = QLabel("Select a document to preview its text.", self)
        self._preview_label.setProperty("muted", True)
        self._preview_label.setWordWrap(True)
        layout.addWidget(self._preview_label)
        return frame

    def _choose_files(self) -> None:
        paths, _selected_filter = QFileDialog.getOpenFileNames(
            self,
            "Import Documents",
            "",
            "Text Documents (*.txt *.md *.csv);;All Files (*.*)",
        )
        if not paths:
            self._status_label.setText("No files selected.")
            return
        self.import_paths(paths)

    def _choose_folder(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Import Documents Folder", "")
        if not path:
            self._status_label.setText("No folder selected.")
            return
        self.import_paths((path,))

    def import_paths(
        self,
        paths: Sequence[str | Path],
        *,
        options: DocumentImportOptions | None = None,
    ) -> DocumentImportResult:
        self._last_import_paths = tuple(paths)
        active_options = options or self._current_options()
        result = import_documents_from_paths(
            paths,
            options=active_options,
            existing_documents=self._documents,
        )
        self._documents = result.documents
        self._skipped = result.skipped
        self._errors = result.errors
        self._render()
        self._notify_output_changed()
        return result

    def reload_last_import(self) -> DocumentImportResult | None:
        if not self._last_import_paths:
            self._status_label.setText("No previous import to reload.")
            return None
        return self.import_paths(self._last_import_paths)

    def remove_selected_document(self) -> CorpusDocument | None:
        row = self._table.currentRow()
        if row < 0 or row >= len(self._documents):
            self._status_label.setText("Select a document to remove.")
            return None
        removed = self._documents[row]
        documents = list(self._documents)
        del documents[row]
        self._documents = tuple(documents)
        self._render()
        self._notify_output_changed()
        return removed

    def clear_imported_documents(self) -> None:
        self._documents = ()
        self._skipped = ()
        self._errors = ()
        self._render()
        self._notify_output_changed()

    def current_output_payload(self) -> WorkflowPayload:
        return WorkflowPayload("Corpus", self._documents)

    def _current_options(self) -> DocumentImportOptions:
        return DocumentImportOptions(
            recursive=self._recursive_checkbox.isChecked(),
            category_from_subfolders=self._category_checkbox.isChecked(),
            encoding=self._encoding_combo.currentData() or AUTO_ENCODING,
            duplicate_policy=self._duplicate_combo.currentData() or DUPLICATE_SKIP,
            append_to_existing=self._append_checkbox.isChecked(),
            csv_mapping=CSVColumnMapping(
                title_column=self._csv_title_input.text(),
                text_column=self._csv_text_input.text(),
                source_column=self._csv_source_input.text(),
            ),
        )

    def _render(self) -> None:
        word_count = sum(count_words(document.text) for document in self._documents)
        average = word_count / len(self._documents) if self._documents else 0.0
        file_types = self._file_type_summary(self._documents)
        self._summary_label.setText(
            f"{len(self._documents)} documents imported, {len(self._skipped)} skipped, "
            f"{word_count} total words, {average:.1f} avg words/document"
            + (f", file types: {file_types}." if file_types else ".")
        )
        if self._skipped:
            self._status_label.setText("Import completed with warnings: " + " | ".join(self._errors[:3]))
        elif self._documents:
            self._status_label.setText("Imported documents are ready for the corpus workflow.")
        else:
            self._status_label.setText("No documents imported yet.")

        self._table.setRowCount(len(self._documents))
        for row, document in enumerate(self._documents):
            self._set_item(self._table, row, 0, document.title)
            self._set_item(self._table, row, 1, document.source)
            self._set_item(self._table, row, 2, self._preview_text(document.text))
            self._set_item(self._table, row, 3, str(count_words(document.text)))
        self._table.resizeColumnsToContents()

        self._skipped_table.setRowCount(len(self._skipped))
        for row, item in enumerate(self._skipped):
            self._set_item(self._skipped_table, row, 0, item.path)
            self._set_item(self._skipped_table, row, 1, item.extension or "-")
            self._set_item(self._skipped_table, row, 2, item.message)
        self._skipped_table.resizeColumnsToContents()

        self._tabs.setTabText(0, f"Imported Documents ({len(self._documents)})")
        self._tabs.setTabText(1, f"Skipped Documents ({len(self._skipped)})")
        self._update_preview()

    def _update_preview(self) -> None:
        row = self._table.currentRow()
        if row < 0 or row >= len(self._documents):
            self._preview_label.setText("Select a document to preview its text.")
            return
        document = self._documents[row]
        self._preview_label.setText(f"{document.title}: {self._preview_text(document.text, limit=320)}")

    def _set_item(self, table: QTableWidget, row: int, column: int, text: str) -> None:
        table.setItem(row, column, QTableWidgetItem(text))

    def _preview_text(self, text: str, limit: int = 140) -> str:
        normalized = " ".join(text.split())
        if len(normalized) <= limit:
            return normalized
        return f"{normalized[: limit - 3]}..."

    def _file_type_summary(self, documents: Iterable[CorpusDocument]) -> str:
        counts = Counter(Path(document.source).suffix.lower() or Path(document.title).suffix.lower() for document in documents)
        counts.pop("", None)
        return ", ".join(f"{extension}: {count}" for extension, count in sorted(counts.items()))

    def data_preview_snapshot(self) -> dict[str, object]:
        rows = [
            [document.title, document.source, self._preview_text(document.text), str(count_words(document.text))]
            for document in self._documents
        ]
        word_count = sum(count_words(document.text) for document in self._documents)
        return {
            "summary": (
                f"Imported Documents: {len(self._documents)} documents, "
                f"{len(self._skipped)} skipped, {word_count} total words"
            ),
            "headers": ["Title", "Source", "Text Preview", "Words"],
            "rows": rows,
        }
