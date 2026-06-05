from __future__ import annotations

from collections.abc import Sequence

from PySide6.QtCore import QSize
from PySide6.QtGui import QColor, QPalette
from PySide6.QtWidgets import (
    QComboBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from portakal_app.models import WorkflowPayload
from portakal_app.ui.screens.corpus_screen import CorpusDocument, count_words, summarize_corpus
from portakal_app.ui.screens.node_screen import WorkflowNodeScreenSupport
from portakal_app.ui.shared.cards import SectionHeader


DEFAULT_SOURCE = "Manual"
READABLE_LINE_EDIT_STYLE = """
QLineEdit {
    background-color: #2f2f2f;
    color: #ffffff;
    border: 1px solid #4b4b4b;
    border-radius: 8px;
    padding: 8px 10px;
    selection-background-color: #cf9440;
    selection-color: #ffffff;
}
QLineEdit:focus {
    border-color: #e2a952;
}
"""

BULK_SPLIT_BLANK_LINE = "blank_line"
BULK_SPLIT_DASHES = "dashes"


def create_default_title(index: int) -> str:
    return f"Document {max(1, index)}"


def preview_text(text: str, max_length: int = 140) -> str:
    normalized = " ".join(text.split())
    if len(normalized) <= max_length:
        return normalized
    return f"{normalized[: max_length - 3]}..."


def make_document(title: str, text: str, source: str, index: int = 1) -> CorpusDocument:
    clean_title = title.strip() or create_default_title(index)
    clean_source = source.strip() or DEFAULT_SOURCE
    return CorpusDocument(clean_title, text, clean_source)


def split_bulk_documents(text: str, mode: str = BULK_SPLIT_BLANK_LINE) -> tuple[str, ...]:
    normalized = text.strip()
    if not normalized:
        return ()
    if mode == BULK_SPLIT_DASHES:
        parts = normalized.split("---")
    else:
        parts = normalized.replace("\r\n", "\n").replace("\r", "\n").split("\n\n")
    return tuple(part.strip() for part in parts if part.strip())


def apply_readable_line_edit_style(line_edit: QLineEdit) -> None:
    line_edit.setStyleSheet(READABLE_LINE_EDIT_STYLE)
    palette = line_edit.palette()
    palette.setColor(QPalette.ColorRole.Text, QColor("#ffffff"))
    palette.setColor(QPalette.ColorRole.PlaceholderText, QColor("#d6d6d6"))
    line_edit.setPalette(palette)


class CreateCorpusScreen(QWidget, WorkflowNodeScreenSupport):
    def __init__(self, parent: QWidget | None = None, documents: Sequence[CorpusDocument] | None = None) -> None:
        super().__init__(parent)
        self._init_workflow_node_support()
        self._documents = tuple(documents or ())
        self._selected_index: int | None = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(10)

        layout.addWidget(
            SectionHeader(
                "Create Corpus",
                "Create a small text corpus manually with document titles, sources, and text content.",
            )
        )
        layout.addWidget(self._build_editor_panel())
        layout.addWidget(self._build_bulk_panel())
        layout.addWidget(self._build_metadata_panel())
        layout.addWidget(self._build_table_panel(), 1)

        self._render()

    def sizeHint(self) -> QSize:
        return QSize(920, 700)

    def minimumSizeHint(self) -> QSize:
        return QSize(720, 540)

    def _build_editor_panel(self) -> QFrame:
        frame = QFrame(self)
        frame.setProperty("panel", True)
        layout = QVBoxLayout(frame)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)

        self._title_input = QLineEdit(self)
        self._title_input.setPlaceholderText("Document title")
        apply_readable_line_edit_style(self._title_input)
        layout.addWidget(QLabel("Title", self))
        layout.addWidget(self._title_input)

        self._source_input = QLineEdit(self)
        self._source_input.setPlaceholderText(DEFAULT_SOURCE)
        apply_readable_line_edit_style(self._source_input)
        layout.addWidget(QLabel("Source / Category", self))
        layout.addWidget(self._source_input)

        self._text_input = QPlainTextEdit(self)
        self._text_input.setPlaceholderText("Enter document text...")
        self._text_input.setMinimumHeight(90)
        layout.addWidget(QLabel("Text Content", self))
        layout.addWidget(self._text_input)

        actions_layout = QHBoxLayout()
        actions_layout.setContentsMargins(0, 0, 0, 0)
        actions_layout.setSpacing(8)

        self._add_button = QPushButton("Add Document", self)
        self._add_button.setProperty("primary", True)
        self._add_button.clicked.connect(self._add_from_inputs)
        actions_layout.addWidget(self._add_button)

        self._update_button = QPushButton("Update Selected", self)
        self._update_button.setProperty("secondary", True)
        self._update_button.clicked.connect(self._update_selected_from_inputs)
        actions_layout.addWidget(self._update_button)

        self._new_button = QPushButton("New Document", self)
        self._new_button.setProperty("secondary", True)
        self._new_button.clicked.connect(self._clear_editor_selection)
        actions_layout.addWidget(self._new_button)

        self._clear_button = QPushButton("Clear Corpus", self)
        self._clear_button.setProperty("secondary", True)
        self._clear_button.clicked.connect(self.clear_corpus)
        actions_layout.addWidget(self._clear_button)
        actions_layout.addStretch(1)

        layout.addLayout(actions_layout)
        return frame

    def _build_bulk_panel(self) -> QFrame:
        frame = QFrame(self)
        frame.setProperty("panel", True)
        layout = QVBoxLayout(frame)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)

        header_layout = QHBoxLayout()
        header_layout.setContentsMargins(0, 0, 0, 0)
        header_layout.setSpacing(8)
        header_layout.addWidget(QLabel("Bulk Add Documents", self))

        self._bulk_split_combo = QComboBox(self)
        self._bulk_split_combo.addItem("Split by blank lines", BULK_SPLIT_BLANK_LINE)
        self._bulk_split_combo.addItem("Split by ---", BULK_SPLIT_DASHES)
        header_layout.addWidget(self._bulk_split_combo)

        self._bulk_add_button = QPushButton("Add Bulk Text", self)
        self._bulk_add_button.setProperty("secondary", True)
        self._bulk_add_button.clicked.connect(self._add_bulk_from_inputs)
        header_layout.addWidget(self._bulk_add_button)
        header_layout.addStretch(1)
        layout.addLayout(header_layout)

        self._bulk_text_input = QPlainTextEdit(self)
        self._bulk_text_input.setPlaceholderText("Paste multiple documents here...")
        self._bulk_text_input.setMinimumHeight(70)
        layout.addWidget(self._bulk_text_input)
        return frame

    def _build_metadata_panel(self) -> QFrame:
        frame = QFrame(self)
        frame.setProperty("panel", True)
        layout = QHBoxLayout(frame)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        self._document_count_label = self._build_metric_label("Documents", "0")
        self._total_word_count_label = self._build_metric_label("Total Words", "0")
        self._average_word_count_label = self._build_metric_label("Avg Words / Document", "0.0")
        self._category_count_label = self._build_metric_label("Categories", "0")
        self._empty_document_count_label = self._build_metric_label("Empty Text", "0")

        layout.addWidget(self._document_count_label, 1)
        layout.addWidget(self._total_word_count_label, 1)
        layout.addWidget(self._average_word_count_label, 1)
        layout.addWidget(self._category_count_label, 1)
        layout.addWidget(self._empty_document_count_label, 1)
        return frame

    def _build_metric_label(self, title: str, value: str) -> QLabel:
        label = QLabel(f"{title}\n{value}", self)
        label.setProperty("infoCard", True)
        label.setStyleSheet("padding: 10px;")
        return label

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

        self._table = QTableWidget(0, 4, self)
        self._table.setHorizontalHeaderLabels(["Title", "Source", "Text Preview", "Words"])
        self._table.verticalHeader().setVisible(False)
        self._table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self._table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self._table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._table.horizontalHeader().setStretchLastSection(True)
        self._table.itemSelectionChanged.connect(self._load_selected_document)
        layout.addWidget(self._table, 1)

        actions_layout = QHBoxLayout()
        actions_layout.setContentsMargins(0, 0, 0, 0)
        actions_layout.setSpacing(8)

        self._remove_button = QPushButton("Remove Selected", self)
        self._remove_button.setProperty("secondary", True)
        self._remove_button.clicked.connect(self.remove_selected_document)
        actions_layout.addWidget(self._remove_button)

        self._move_up_button = QPushButton("Move Up", self)
        self._move_up_button.setProperty("secondary", True)
        self._move_up_button.clicked.connect(self.move_selected_up)
        actions_layout.addWidget(self._move_up_button)

        self._move_down_button = QPushButton("Move Down", self)
        self._move_down_button.setProperty("secondary", True)
        self._move_down_button.clicked.connect(self.move_selected_down)
        actions_layout.addWidget(self._move_down_button)
        actions_layout.addStretch(1)
        layout.addLayout(actions_layout)
        return frame

    def _add_from_inputs(self) -> None:
        self.add_document(
            self._title_input.text(),
            self._text_input.toPlainText(),
            self._source_input.text(),
        )
        self._title_input.clear()
        self._text_input.clear()
        self._title_input.setFocus()
        self._selected_index = None
        self._render()

    def _update_selected_from_inputs(self) -> None:
        self.update_selected_document(
            self._title_input.text(),
            self._text_input.toPlainText(),
            self._source_input.text(),
        )

    def _add_bulk_from_inputs(self) -> None:
        self.add_bulk_documents(
            self._bulk_text_input.toPlainText(),
            source=self._source_input.text(),
            mode=self._bulk_split_combo.currentData() or BULK_SPLIT_BLANK_LINE,
        )
        self._bulk_text_input.clear()

    def add_document(self, title: str, text: str, source: str = "") -> CorpusDocument:
        document = make_document(title, text, source, index=len(self._documents) + 1)
        self._documents = (*self._documents, document)
        self._selected_index = len(self._documents) - 1
        self._render()
        self._notify_output_changed()
        return document

    def add_bulk_documents(
        self,
        text: str,
        source: str = "",
        mode: str = BULK_SPLIT_BLANK_LINE,
    ) -> tuple[CorpusDocument, ...]:
        parts = split_bulk_documents(text, mode)
        added: list[CorpusDocument] = []
        for part in parts:
            added.append(
                make_document(
                    "",
                    part,
                    source,
                    index=len(self._documents) + len(added) + 1,
                )
            )
        if not added:
            self._status_label.setText("No bulk documents found.")
            return ()
        self._documents = (*self._documents, *added)
        self._selected_index = len(self._documents) - len(added)
        self._render()
        self._notify_output_changed()
        return tuple(added)

    def update_selected_document(self, title: str, text: str, source: str = "") -> CorpusDocument | None:
        if self._selected_index is None or not (0 <= self._selected_index < len(self._documents)):
            self._status_label.setText("Select a document to update.")
            return None
        document = make_document(title, text, source, index=self._selected_index + 1)
        documents = list(self._documents)
        documents[self._selected_index] = document
        self._documents = tuple(documents)
        self._render()
        self._select_row(self._selected_index)
        self._notify_output_changed()
        return document

    def remove_selected_document(self) -> CorpusDocument | None:
        index = self._current_document_index()
        if index is None:
            self._status_label.setText("Select a document to remove.")
            return None
        documents = list(self._documents)
        removed = documents.pop(index)
        self._documents = tuple(documents)
        self._selected_index = min(index, len(self._documents) - 1) if self._documents else None
        self._render()
        if self._selected_index is not None:
            self._select_row(self._selected_index)
        self._notify_output_changed()
        return removed

    def move_selected_up(self) -> bool:
        index = self._current_document_index()
        if index is None or index <= 0:
            self._status_label.setText("Select a document below the first row to move up.")
            return False
        self._swap_documents(index, index - 1)
        return True

    def move_selected_down(self) -> bool:
        index = self._current_document_index()
        if index is None or index >= len(self._documents) - 1:
            self._status_label.setText("Select a document above the last row to move down.")
            return False
        self._swap_documents(index, index + 1)
        return True

    def clear_corpus(self) -> None:
        self._documents = ()
        self._selected_index = None
        self._clear_inputs()
        self._render()
        self._notify_output_changed()

    def current_output_payload(self) -> WorkflowPayload:
        return WorkflowPayload("Corpus", self._documents)

    def _render(self) -> None:
        summary = summarize_corpus(self._documents)
        self._document_count_label.setText(f"Documents\n{summary.document_count}")
        self._total_word_count_label.setText(f"Total Words\n{summary.total_word_count}")
        self._average_word_count_label.setText(
            f"Avg Words / Document\n{summary.average_words_per_document:.1f}"
        )
        category_count = len({document.source for document in self._documents if document.source})
        empty_count = sum(1 for document in self._documents if not document.text.strip())
        self._category_count_label.setText(f"Categories\n{category_count}")
        self._empty_document_count_label.setText(f"Empty Text\n{empty_count}")
        self._status_label.setText(
            "Manual corpus is ready for later text mining workflow steps."
            if self._documents
            else "No documents created yet. Add a title, optional source, and text content."
        )

        self._table.setRowCount(len(self._documents))
        for row, document in enumerate(self._documents):
            self._set_item(row, 0, document.title)
            self._set_item(row, 1, document.source)
            self._set_item(row, 2, preview_text(document.text))
            self._set_item(row, 3, str(count_words(document.text)))
        self._table.resizeColumnsToContents()
        self._update_action_state()

    def _load_selected_document(self) -> None:
        index = self._current_document_index()
        if index is None:
            self._selected_index = None
            self._update_action_state()
            return
        self._selected_index = index
        document = self._documents[index]
        self._title_input.setText(document.title)
        self._source_input.setText(document.source)
        self._text_input.setPlainText(document.text)
        self._update_action_state()

    def _current_document_index(self) -> int | None:
        row = self._table.currentRow()
        if 0 <= row < len(self._documents):
            return row
        if self._selected_index is not None and 0 <= self._selected_index < len(self._documents):
            return self._selected_index
        return None

    def _swap_documents(self, first: int, second: int) -> None:
        documents = list(self._documents)
        documents[first], documents[second] = documents[second], documents[first]
        self._documents = tuple(documents)
        self._selected_index = second
        self._render()
        self._select_row(second)
        self._notify_output_changed()

    def _select_row(self, row: int) -> None:
        if 0 <= row < self._table.rowCount():
            self._table.selectRow(row)

    def _clear_editor_selection(self) -> None:
        self._selected_index = None
        self._table.clearSelection()
        self._clear_inputs()
        self._update_action_state()

    def _clear_inputs(self) -> None:
        self._title_input.clear()
        self._source_input.clear()
        self._text_input.clear()

    def _update_action_state(self) -> None:
        has_selection = self._current_document_index() is not None
        self._update_button.setEnabled(has_selection)
        self._remove_button.setEnabled(has_selection)
        self._move_up_button.setEnabled(has_selection and (self._current_document_index() or 0) > 0)
        current_index = self._current_document_index()
        self._move_down_button.setEnabled(
            has_selection and current_index is not None and current_index < len(self._documents) - 1
        )

    def _set_item(self, row: int, column: int, text: str) -> None:
        self._table.setItem(row, column, QTableWidgetItem(text))

    def data_preview_snapshot(self) -> dict[str, object]:
        rows = [
            [document.title, document.source, preview_text(document.text), str(count_words(document.text))]
            for document in self._documents
        ]
        summary = summarize_corpus(self._documents)
        return {
            "summary": (
                f"Manual Corpus: {summary.document_count} documents, "
                f"{summary.total_word_count} total words, "
                f"{summary.average_words_per_document:.1f} average words/document, "
                f"{len({document.source for document in self._documents if document.source})} categories"
            ),
            "headers": ["Title", "Source", "Text Preview", "Words"],
            "rows": rows,
        }
