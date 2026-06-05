from __future__ import annotations

from collections.abc import Mapping
from collections.abc import Sequence

from PySide6.QtCore import QSize
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from portakal_app.models import WorkflowPayload
from portakal_app.ui.screens.corpus_screen import (
    CorpusDocument,
    corpus_documents_from_payload,
    count_words,
    summarize_corpus,
)
from portakal_app.ui.screens.create_corpus_screen import preview_text
from portakal_app.ui.screens.node_screen import WorkflowNodeScreenSupport
from portakal_app.ui.shared.cards import SectionHeader


class CorpusViewerScreen(QWidget, WorkflowNodeScreenSupport):
    def __init__(
        self,
        parent: QWidget | None = None,
        documents: Sequence[CorpusDocument] | None = None,
    ) -> None:
        super().__init__(parent)
        self._init_workflow_node_support()
        self._documents = tuple(() if documents is None else documents)
        self._using_input_corpus = documents is not None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(10)

        layout.addWidget(
            SectionHeader(
                "Corpus Viewer",
                "Inspect corpus documents and pass the same corpus to downstream text mining widgets.",
            )
        )
        layout.addWidget(self._build_metadata_panel())
        layout.addWidget(self._build_table_panel(), 1)

        self._render()

    def sizeHint(self) -> QSize:
        return QSize(920, 640)

    def minimumSizeHint(self) -> QSize:
        return QSize(700, 500)

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

        self._table = QTableWidget(0, 4, self)
        self._table.setHorizontalHeaderLabels(["Title", "Source", "Text Preview", "Words"])
        self._table.verticalHeader().setVisible(False)
        self._table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self._table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self._table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._table.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(self._table, 1)
        return frame

    def set_input_payload(self, payload: WorkflowPayload | None) -> None:
        if payload is None:
            self._documents = ()
            self._using_input_corpus = False
            self._render()
            self._notify_output_changed()
            return

        documents = corpus_documents_from_payload(payload.value)
        self._documents = () if documents is None else documents
        self._using_input_corpus = True
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

        if self._documents and self._using_input_corpus:
            status = "Input corpus is connected and displayed."
        elif self._using_input_corpus:
            status = "Input corpus is connected but empty."
        else:
            status = "Connect a Corpus input to inspect documents."
        self._status_label.setText(status)

        extra_headers = self._sentiment_feature_headers()
        headers = ["Title", "Source", "Text Preview", "Words", *extra_headers]
        self._table.setColumnCount(len(headers))
        self._table.setHorizontalHeaderLabels(headers)
        self._table.setRowCount(len(self._documents))
        for row, document in enumerate(self._documents):
            self._set_item(row, 0, document.title)
            self._set_item(row, 1, document.source)
            self._set_item(row, 2, preview_text(document.text))
            self._set_item(row, 3, str(count_words(document.text)))
            features = self._sentiment_features(document)
            for column, header in enumerate(extra_headers, start=4):
                self._set_item(row, column, str(features.get(header, "")))
        self._table.resizeColumnsToContents()

    def _set_item(self, row: int, column: int, text: str) -> None:
        self._table.setItem(row, column, QTableWidgetItem(text))

    def _sentiment_feature_headers(self) -> list[str]:
        headers: list[str] = []
        for document in self._documents:
            for key in self._sentiment_features(document):
                if key not in headers:
                    headers.append(str(key))
        return headers

    def _sentiment_features(self, document: CorpusDocument) -> Mapping[str, object]:
        features = getattr(document, "sentiment_features", {})
        return features if isinstance(features, Mapping) else {}

    def data_preview_snapshot(self) -> dict[str, object]:
        extra_headers = self._sentiment_feature_headers()
        rows = [
            [
                document.title,
                document.source,
                preview_text(document.text),
                str(count_words(document.text)),
                *(str(self._sentiment_features(document).get(header, "")) for header in extra_headers),
            ]
            for document in self._documents
        ]
        summary = summarize_corpus(self._documents)
        return {
            "summary": (
                f"Corpus Viewer: {summary.document_count} documents, "
                f"{summary.total_word_count} total words, "
                f"{summary.average_words_per_document:.1f} average words/document"
            ),
            "headers": ["Title", "Source", "Text Preview", "Words", *extra_headers],
            "rows": rows,
        }
