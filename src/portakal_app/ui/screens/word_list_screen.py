from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass

from PySide6.QtCore import QSize
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from portakal_app.models import WorkflowPayload
from portakal_app.ui.screens.bag_of_words_screen import tokenize_for_bow
from portakal_app.ui.screens.corpus_screen import CorpusDocument, corpus_documents_from_payload
from portakal_app.ui.screens.node_screen import WorkflowNodeScreenSupport
from portakal_app.ui.shared.cards import SectionHeader
from portakal_app.ui.shared.readable_inputs import apply_readable_spin_box_style


@dataclass(frozen=True)
class WordListItem:
    word: str
    frequency: int
    document_count: int


def build_word_list(
    documents: Sequence[CorpusDocument],
    min_frequency: int = 1,
) -> tuple[WordListItem, ...]:
    threshold = max(1, int(min_frequency))
    frequencies: Counter[str] = Counter()
    document_counts: Counter[str] = Counter()

    for document in documents:
        tokens = tokenize_for_bow(document.text)
        frequencies.update(tokens)
        document_counts.update(set(tokens))

    items = [
        WordListItem(word, frequency, document_counts[word])
        for word, frequency in frequencies.items()
        if frequency >= threshold
    ]
    return tuple(sorted(items, key=lambda item: (-item.frequency, item.word)))


class WordListScreen(QWidget, WorkflowNodeScreenSupport):
    def __init__(
        self,
        parent: QWidget | None = None,
        documents: Sequence[CorpusDocument] | None = None,
    ) -> None:
        super().__init__(parent)
        self._init_workflow_node_support()
        self._documents = tuple(() if documents is None else documents)
        self._using_input_corpus = documents is not None
        self._items: tuple[WordListItem, ...] = ()

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(10)

        layout.addWidget(
            SectionHeader(
                "Word List",
                "List corpus terms with total frequency and document frequency.",
            )
        )
        layout.addWidget(self._build_options_panel())
        layout.addWidget(self._build_metadata_panel())
        layout.addWidget(self._build_table_panel(), 1)

        self.apply_filter()

    def sizeHint(self) -> QSize:
        return QSize(860, 640)

    def minimumSizeHint(self) -> QSize:
        return QSize(680, 500)

    def _build_options_panel(self) -> QFrame:
        frame = QFrame(self)
        frame.setProperty("panel", True)
        layout = QHBoxLayout(frame)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        layout.addWidget(QLabel("Minimum frequency", self))
        self._min_frequency_spinbox = QSpinBox(self)
        self._min_frequency_spinbox.setRange(1, 999)
        self._min_frequency_spinbox.setValue(1)
        apply_readable_spin_box_style(self._min_frequency_spinbox)
        layout.addWidget(self._min_frequency_spinbox)

        self._apply_button = QPushButton("Apply Filter", self)
        self._apply_button.setProperty("primary", True)
        self._apply_button.clicked.connect(self.apply_filter)
        layout.addWidget(self._apply_button)
        layout.addStretch(1)
        return frame

    def _build_metadata_panel(self) -> QFrame:
        frame = QFrame(self)
        frame.setProperty("panel", True)
        layout = QHBoxLayout(frame)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        self._document_count_label = self._build_metric_label("Documents", "0")
        self._word_count_label = self._build_metric_label("Words", "0")
        self._total_frequency_label = self._build_metric_label("Total Frequency", "0")

        layout.addWidget(self._document_count_label, 1)
        layout.addWidget(self._word_count_label, 1)
        layout.addWidget(self._total_frequency_label, 1)
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
        self._table.setHorizontalHeaderLabels(["Word", "Frequency", "Documents"])
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
            self.apply_filter()
            return

        documents = corpus_documents_from_payload(payload.value)
        self._documents = () if documents is None else documents
        self._using_input_corpus = True
        self.apply_filter()

    def current_output_payload(self) -> WorkflowPayload:
        rows = [[item.word, item.frequency, item.document_count] for item in self._items]
        return WorkflowPayload("Words", rows)

    def apply_filter(self) -> tuple[WordListItem, ...]:
        self._items = build_word_list(self._documents, self._min_frequency_spinbox.value())
        self._render()
        self._notify_output_changed()
        return self._items

    def _render(self) -> None:
        total_frequency = sum(item.frequency for item in self._items)
        self._document_count_label.setText(f"Documents\n{len(self._documents)}")
        self._word_count_label.setText(f"Words\n{len(self._items)}")
        self._total_frequency_label.setText(f"Total Frequency\n{total_frequency}")

        if self._items and self._using_input_corpus:
            status = "Input corpus is connected and word list is ready."
        elif self._using_input_corpus:
            status = "Input corpus is connected but no words match the current filter."
        else:
            status = "Connect a Corpus input to build a word list."
        self._status_label.setText(status)

        self._table.setRowCount(len(self._items))
        for row, item in enumerate(self._items):
            self._set_item(row, 0, item.word)
            self._set_item(row, 1, str(item.frequency))
            self._set_item(row, 2, str(item.document_count))
        self._table.resizeColumnsToContents()

    def _set_item(self, row: int, column: int, text: str) -> None:
        self._table.setItem(row, column, QTableWidgetItem(text))

    def data_preview_snapshot(self) -> dict[str, object]:
        rows = [[item.word, str(item.frequency), str(item.document_count)] for item in self._items]
        return {
            "summary": (
                f"Word List: {len(self._items)} words, "
                f"{sum(item.frequency for item in self._items)} total frequency"
            ),
            "headers": ["Word", "Frequency", "Documents"],
            "rows": rows,
        }
