from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from PySide6.QtCore import QSize
from PySide6.QtWidgets import (
    QComboBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from portakal_app.models import WorkflowPayload
from portakal_app.ui.screens.node_screen import WorkflowNodeScreenSupport
from portakal_app.ui.shared.cards import SectionHeader


@dataclass(frozen=True)
class CorpusDocument:
    title: str
    text: str
    source: str = "Sample"
    attributes: tuple[tuple[str, object], ...] = ()


def corpus_document_attributes(document: CorpusDocument) -> dict[str, object]:
    return dict(document.attributes)


def with_corpus_document_attributes(
    document: CorpusDocument,
    attributes: dict[str, object],
) -> CorpusDocument:
    merged = corpus_document_attributes(document)
    merged.update(attributes)
    return CorpusDocument(
        title=document.title,
        text=document.text,
        source=document.source,
        attributes=tuple(sorted(merged.items())),
    )


@dataclass(frozen=True)
class CorpusSummary:
    document_count: int
    total_word_count: int
    average_words_per_document: float


@dataclass(frozen=True)
class SampleCorpusDefinition:
    id: str
    name: str
    description: str
    documents: tuple[CorpusDocument, ...]


SAMPLE_CORPUS: tuple[CorpusDocument, ...] = (
    CorpusDocument(
        "Document 1",
        "Portakal introduces visual workflows for exploring data mining tasks.",
    ),
    CorpusDocument(
        "Document 2",
        "A corpus is a collection of documents prepared for text analysis.",
    ),
    CorpusDocument(
        "Document 3",
        "Simple text features can help compare documents by their vocabulary.",
    ),
    CorpusDocument(
        "Document 4",
        "Preprocessing usually cleans text before models or visual summaries use it.",
    ),
    CorpusDocument(
        "Document 5",
        "Bag of words converts documents into counts that tabular widgets can inspect.",
    ),
)

SAMPLE_CORPORA: tuple[SampleCorpusDefinition, ...] = (
    SampleCorpusDefinition(
        "text-mining-basics",
        "Text Mining Basics",
        "Short documents that describe core text mining workflow steps.",
        SAMPLE_CORPUS,
    ),
    SampleCorpusDefinition(
        "news-snippets",
        "News Snippets",
        "Small news-like documents for testing source/category workflows.",
        (
            CorpusDocument(
                "Local Council Approves Transit Plan",
                "The city council approved a new transit plan after public debate on budget priorities.",
                "News / Local",
            ),
            CorpusDocument(
                "Researchers Publish Climate Report",
                "Researchers published a climate report describing warmer seasons and changing rainfall patterns.",
                "News / Science",
            ),
            CorpusDocument(
                "Startup Releases Data Tool",
                "A software startup released a dashboard tool for teams that analyze customer feedback.",
                "News / Technology",
            ),
            CorpusDocument(
                "Museum Opens Archive Exhibit",
                "The museum opened an archive exhibit featuring letters, photographs, and oral histories.",
                "News / Culture",
            ),
            CorpusDocument(
                "Farmers Monitor Soil Health",
                "Farmers are monitoring soil health with sensors before the spring planting season begins.",
                "News / Agriculture",
            ),
        ),
    ),
    SampleCorpusDefinition(
        "product-reviews",
        "Product Reviews",
        "Compact review documents with mixed sentiment and repeated terms.",
        (
            CorpusDocument(
                "Review 1",
                "The app is fast and useful, but the export screen needs clearer labels.",
                "Reviews",
            ),
            CorpusDocument(
                "Review 2",
                "Search results load quickly and the filter controls make large collections easier to inspect.",
                "Reviews",
            ),
            CorpusDocument(
                "Review 3",
                "The interface looks clean, although dark input fields can hide typed text.",
                "Reviews",
            ),
            CorpusDocument(
                "Review 4",
                "Importing documents works reliably for simple text files and markdown notes.",
                "Reviews",
            ),
            CorpusDocument(
                "Review 5",
                "The word counts are helpful for checking whether preprocessing changed the corpus.",
                "Reviews",
            ),
        ),
    ),
    SampleCorpusDefinition(
        "support-tickets",
        "Support Tickets",
        "Short operational messages for testing preprocessing and bag-of-words output.",
        (
            CorpusDocument(
                "Ticket 1001",
                "User cannot upload a CSV file because the delimiter setting is unclear.",
                "Support",
            ),
            CorpusDocument(
                "Ticket 1002",
                "Workflow output updates after reconnecting the corpus input to preprocessing.",
                "Support",
            ),
            CorpusDocument(
                "Ticket 1003",
                "The table preview should keep document order after multiple inputs are merged.",
                "Support",
            ),
            CorpusDocument(
                "Ticket 1004",
                "Sample data helps new users test the workflow before importing their own documents.",
                "Support",
            ),
            CorpusDocument(
                "Ticket 1005",
                "Bag of words should show repeated tokens and total term frequency without crashing.",
                "Support",
            ),
        ),
    ),
)

DEFAULT_SAMPLE_CORPUS_ID = SAMPLE_CORPORA[0].id


def count_words(text: str) -> int:
    return len(text.split())


def summarize_corpus(documents: Iterable[CorpusDocument]) -> CorpusSummary:
    items = tuple(documents)
    document_count = len(items)
    total_word_count = sum(count_words(document.text) for document in items)
    average = total_word_count / document_count if document_count else 0.0
    return CorpusSummary(document_count, total_word_count, average)


def sample_corpus_by_id(sample_id: str) -> SampleCorpusDefinition:
    for sample in SAMPLE_CORPORA:
        if sample.id == sample_id:
            return sample
    return SAMPLE_CORPORA[0]


def corpus_documents_from_payload(value: object) -> tuple[CorpusDocument, ...] | None:
    if isinstance(value, CorpusDocument):
        return (value,)
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        return None

    documents: list[CorpusDocument] = []
    for item in value:
        if not isinstance(item, CorpusDocument):
            return None
        documents.append(item)
    return tuple(documents)


class CorpusScreen(QWidget, WorkflowNodeScreenSupport):
    def __init__(
        self,
        parent: QWidget | None = None,
        documents: Sequence[CorpusDocument] | None = None,
        sample_id: str = DEFAULT_SAMPLE_CORPUS_ID,
    ) -> None:
        super().__init__(parent)
        self._init_workflow_node_support()
        self._selected_sample_id = sample_corpus_by_id(sample_id).id
        self._documents = tuple(self._selected_sample().documents if documents is None else documents)
        self._using_input_corpus = documents is not None
        self._input_corpus_count = 1 if documents is not None else 0

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(10)

        layout.addWidget(
            SectionHeader(
                "Corpus",
                "A text corpus is a collection of documents prepared for text mining workflows. "
                "Choose a built-in sample corpus, or connect Corpus inputs to override or merge documents.",
            )
        )
        layout.addWidget(self._build_sample_panel())
        layout.addWidget(self._build_metadata_panel())
        layout.addWidget(self._build_table_panel(), 1)

        self._render()

    def sizeHint(self) -> QSize:
        return QSize(860, 620)

    def minimumSizeHint(self) -> QSize:
        return QSize(680, 480)

    def _build_sample_panel(self) -> QFrame:
        frame = QFrame(self)
        frame.setProperty("panel", True)
        layout = QHBoxLayout(frame)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        label = QLabel("Sample Corpus", self)
        label.setProperty("muted", True)
        layout.addWidget(label)

        self._sample_combo = QComboBox(self)
        for sample in SAMPLE_CORPORA:
            self._sample_combo.addItem(sample.name, sample.id)
        self._sample_combo.setCurrentIndex(self._sample_index(self._selected_sample_id))
        self._sample_combo.currentIndexChanged.connect(self._sample_selection_changed)
        layout.addWidget(self._sample_combo, 1)

        self._sample_description_label = QLabel("", self)
        self._sample_description_label.setProperty("muted", True)
        self._sample_description_label.setWordWrap(True)
        layout.addWidget(self._sample_description_label, 3)
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

        layout.addWidget(self._document_count_label, 1)
        layout.addWidget(self._total_word_count_label, 1)
        layout.addWidget(self._average_word_count_label, 1)
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

        self._table = QTableWidget(0, 3, self)
        self._table.setHorizontalHeaderLabels(["Title", "Source", "Text Preview"])
        self._table.verticalHeader().setVisible(False)
        self._table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self._table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self._table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._table.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(self._table, 1)
        return frame

    def set_input_payload(self, payload: WorkflowPayload | None) -> None:
        if payload is None:
            self._documents = tuple(self._selected_sample().documents)
            self._using_input_corpus = False
            self._input_corpus_count = 0
            self._render()
            self._notify_output_changed()
            return

        documents = corpus_documents_from_payload(payload.value)
        self._input_corpus_count += 1
        if documents is None:
            documents = ()
        if self._using_input_corpus:
            self._documents = (*self._documents, *documents)
        else:
            self._documents = documents
            self._using_input_corpus = True
        self._render()
        self._notify_output_changed()

    def current_output_payload(self) -> WorkflowPayload:
        return WorkflowPayload("Corpus", self._documents)

    def _selected_sample(self) -> SampleCorpusDefinition:
        return sample_corpus_by_id(self._selected_sample_id)

    def _sample_index(self, sample_id: str) -> int:
        for index, sample in enumerate(SAMPLE_CORPORA):
            if sample.id == sample_id:
                return index
        return 0

    def _sample_selection_changed(self, _index: int = -1) -> None:
        sample_id = self._sample_combo.currentData()
        if not isinstance(sample_id, str):
            return
        self._selected_sample_id = sample_corpus_by_id(sample_id).id
        if not self._using_input_corpus:
            self._documents = tuple(self._selected_sample().documents)
            self._render()
            self._notify_output_changed()

    def _render(self) -> None:
        summary = summarize_corpus(self._documents)
        self._document_count_label.setText(f"Documents\n{summary.document_count}")
        self._total_word_count_label.setText(f"Total Words\n{summary.total_word_count}")
        self._average_word_count_label.setText(
            f"Avg Words / Document\n{summary.average_words_per_document:.1f}"
        )
        if self._documents and self._using_input_corpus and self._input_corpus_count > 1:
            status = f"Merged {self._input_corpus_count} input corpora and displayed them in connection order."
        elif self._documents and self._using_input_corpus:
            status = "Input corpus is connected and displayed."
        elif self._documents:
            status = f"Built-in sample corpus '{self._selected_sample().name}' is ready."
        elif self._using_input_corpus:
            status = "Input corpus is connected but empty."
        else:
            status = "Corpus is empty. Add documents in a later import or creation step."
        self._status_label.setText(status)
        self._sample_combo.setEnabled(not self._using_input_corpus)
        self._sample_description_label.setText(self._selected_sample().description)

        self._table.setRowCount(len(self._documents))
        for row, document in enumerate(self._documents):
            self._set_item(row, 0, document.title)
            self._set_item(row, 1, document.source)
            self._set_item(row, 2, self._preview_text(document.text))
        self._table.resizeColumnsToContents()

    def _set_item(self, row: int, column: int, text: str) -> None:
        self._table.setItem(row, column, QTableWidgetItem(text))

    def _preview_text(self, text: str, limit: int = 140) -> str:
        normalized = " ".join(text.split())
        if len(normalized) <= limit:
            return normalized
        return f"{normalized[: limit - 3]}..."

    def data_preview_snapshot(self) -> dict[str, object]:
        rows = [
            [document.title, document.source, self._preview_text(document.text)]
            for document in self._documents
        ]
        summary = summarize_corpus(self._documents)
        return {
            "summary": (
                f"Corpus: {summary.document_count} documents, "
                f"{summary.total_word_count} total words, "
                f"{summary.average_words_per_document:.1f} average words/document"
            ),
            "headers": ["Title", "Source", "Text Preview"],
            "rows": rows,
        }
