from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass

from PySide6.QtCore import QSize
from PySide6.QtWidgets import (
    QFrame,
    QGridLayout,
    QLabel,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from portakal_app.models import WorkflowPayload
from portakal_app.ui.screens.bag_of_words_screen import tokenize_for_bow
from portakal_app.ui.screens.corpus_screen import (
    CorpusDocument,
    corpus_documents_from_payload,
    count_words,
)
from portakal_app.ui.screens.node_screen import WorkflowNodeScreenSupport
from portakal_app.ui.shared.cards import SectionHeader


@dataclass(frozen=True)
class TextStatisticsSummary:
    document_count: int
    total_word_count: int
    average_words_per_document: float
    unique_word_count: int
    longest_document_title: str
    shortest_document_title: str


def word_frequencies(documents: Sequence[CorpusDocument]) -> Counter[str]:
    counter: Counter[str] = Counter()
    for document in documents:
        counter.update(tokenize_for_bow(document.text))
    return counter


def summarize_text_statistics(documents: Sequence[CorpusDocument]) -> TextStatisticsSummary:
    document_count = len(documents)
    word_counts = [(document.title, count_words(document.text)) for document in documents]
    total_word_count = sum(count for _title, count in word_counts)
    average = total_word_count / document_count if document_count else 0.0
    frequencies = word_frequencies(documents)
    longest = max(word_counts, key=lambda item: item[1], default=("-", 0))[0]
    shortest = min(word_counts, key=lambda item: item[1], default=("-", 0))[0]
    return TextStatisticsSummary(
        document_count=document_count,
        total_word_count=total_word_count,
        average_words_per_document=average,
        unique_word_count=len(frequencies),
        longest_document_title=longest,
        shortest_document_title=shortest,
    )


class TextStatisticsScreen(QWidget, WorkflowNodeScreenSupport):
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
                "Statistics",
                "Summarize corpus size, document lengths, vocabulary, and frequent words.",
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
        layout = QGridLayout(frame)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setHorizontalSpacing(10)
        layout.setVerticalSpacing(10)

        self._document_count_label = self._build_metric_label("Documents", "0")
        self._total_word_count_label = self._build_metric_label("Total Words", "0")
        self._average_word_count_label = self._build_metric_label("Avg Words / Document", "0.0")
        self._unique_word_count_label = self._build_metric_label("Unique Words", "0")
        self._longest_document_label = self._build_metric_label("Longest Document", "-")
        self._shortest_document_label = self._build_metric_label("Shortest Document", "-")

        labels = (
            self._document_count_label,
            self._total_word_count_label,
            self._average_word_count_label,
            self._unique_word_count_label,
            self._longest_document_label,
            self._shortest_document_label,
        )
        for index, label in enumerate(labels):
            layout.addWidget(label, index // 3, index % 3)
        return frame

    def _build_metric_label(self, title: str, value: str) -> QLabel:
        label = QLabel(f"{title}\n{value}", self)
        label.setProperty("infoCard", True)
        label.setStyleSheet("padding: 10px;")
        label.setWordWrap(True)
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

        self._table = QTableWidget(0, 2, self)
        self._table.setHorizontalHeaderLabels(["Word", "Frequency"])
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
            return

        documents = corpus_documents_from_payload(payload.value)
        self._documents = () if documents is None else documents
        self._using_input_corpus = True
        self._render()

    def _render(self) -> None:
        summary = summarize_text_statistics(self._documents)
        self._document_count_label.setText(f"Documents\n{summary.document_count}")
        self._total_word_count_label.setText(f"Total Words\n{summary.total_word_count}")
        self._average_word_count_label.setText(
            f"Avg Words / Document\n{summary.average_words_per_document:.1f}"
        )
        self._unique_word_count_label.setText(f"Unique Words\n{summary.unique_word_count}")
        self._longest_document_label.setText(f"Longest Document\n{summary.longest_document_title}")
        self._shortest_document_label.setText(f"Shortest Document\n{summary.shortest_document_title}")

        if self._documents and self._using_input_corpus:
            status = "Input corpus is connected and statistics are ready."
        elif self._using_input_corpus:
            status = "Input corpus is connected but empty."
        else:
            status = "Connect a Corpus input to calculate statistics."
        self._status_label.setText(status)

        rows = sorted(word_frequencies(self._documents).items(), key=lambda item: (-item[1], item[0]))[:25]
        self._table.setRowCount(len(rows))
        for row, (word, frequency) in enumerate(rows):
            self._set_item(row, 0, word)
            self._set_item(row, 1, str(frequency))
        self._table.resizeColumnsToContents()

    def _set_item(self, row: int, column: int, text: str) -> None:
        self._table.setItem(row, column, QTableWidgetItem(text))

    def data_preview_snapshot(self) -> dict[str, object]:
        rows = [
            [word, str(frequency)]
            for word, frequency in sorted(word_frequencies(self._documents).items(), key=lambda item: (-item[1], item[0]))[:25]
        ]
        summary = summarize_text_statistics(self._documents)
        return {
            "summary": (
                f"Statistics: {summary.document_count} documents, "
                f"{summary.total_word_count} total words, "
                f"{summary.unique_word_count} unique words"
            ),
            "headers": ["Word", "Frequency"],
            "rows": rows,
        }
