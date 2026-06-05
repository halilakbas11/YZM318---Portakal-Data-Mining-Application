from __future__ import annotations

import html
from collections import Counter
from collections.abc import Sequence

from PySide6.QtCore import QSize, Qt
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


WORD_CLOUD_COLORS = ("#7a3f12", "#b05f1a", "#2f5d62", "#5c6f2d", "#874c62", "#3f4f75")


def cloud_word_frequencies(documents: Sequence[CorpusDocument]) -> tuple[tuple[str, int], ...]:
    counter: Counter[str] = Counter()
    for document in documents:
        counter.update(tokenize_for_bow(document.text))
    return tuple(sorted(counter.items(), key=lambda item: (-item[1], item[0])))


class WordCloudScreen(QWidget, WorkflowNodeScreenSupport):
    def __init__(
        self,
        parent: QWidget | None = None,
        documents: Sequence[CorpusDocument] | None = None,
    ) -> None:
        super().__init__(parent)
        self._init_workflow_node_support()
        self._documents = tuple(() if documents is None else documents)
        self._using_input_corpus = documents is not None
        self._frequencies: tuple[tuple[str, int], ...] = ()

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(10)

        layout.addWidget(
            SectionHeader(
                "Word Cloud",
                "Show frequent corpus words with larger text for higher frequency.",
            )
        )
        layout.addWidget(self._build_options_panel())
        layout.addWidget(self._build_cloud_panel(), 2)
        layout.addWidget(self._build_table_panel(), 1)

        self.refresh_cloud()

    def sizeHint(self) -> QSize:
        return QSize(920, 680)

    def minimumSizeHint(self) -> QSize:
        return QSize(700, 520)

    def _build_options_panel(self) -> QFrame:
        frame = QFrame(self)
        frame.setProperty("panel", True)
        layout = QHBoxLayout(frame)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        layout.addWidget(QLabel("Top words", self))
        self._top_words_spinbox = QSpinBox(self)
        self._top_words_spinbox.setRange(5, 100)
        self._top_words_spinbox.setValue(40)
        apply_readable_spin_box_style(self._top_words_spinbox)
        layout.addWidget(self._top_words_spinbox)

        self._refresh_button = QPushButton("Refresh", self)
        self._refresh_button.setProperty("primary", True)
        self._refresh_button.clicked.connect(self.refresh_cloud)
        layout.addWidget(self._refresh_button)

        self._summary_label = QLabel("", self)
        self._summary_label.setProperty("muted", True)
        self._summary_label.setWordWrap(True)
        layout.addWidget(self._summary_label, 1)
        return frame

    def _build_cloud_panel(self) -> QFrame:
        frame = QFrame(self)
        frame.setProperty("panel", True)
        layout = QVBoxLayout(frame)
        layout.setContentsMargins(18, 18, 18, 18)
        layout.setSpacing(8)

        self._cloud_label = QLabel("", self)
        self._cloud_label.setTextFormat(Qt.TextFormat.RichText)
        self._cloud_label.setWordWrap(True)
        self._cloud_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._cloud_label.setMinimumHeight(220)
        layout.addWidget(self._cloud_label, 1)
        return frame

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
            self.refresh_cloud()
            return

        documents = corpus_documents_from_payload(payload.value)
        self._documents = () if documents is None else documents
        self._using_input_corpus = True
        self.refresh_cloud()

    def refresh_cloud(self) -> tuple[tuple[str, int], ...]:
        self._frequencies = cloud_word_frequencies(self._documents)[: self._top_words_spinbox.value()]
        self._render()
        return self._frequencies

    def _render(self) -> None:
        total_frequency = sum(frequency for _word, frequency in self._frequencies)
        self._summary_label.setText(
            f"{len(self._documents)} documents, {len(self._frequencies)} displayed words, {total_frequency} total frequency."
        )

        if self._frequencies and self._using_input_corpus:
            status = "Input corpus is connected and word cloud is ready."
        elif self._using_input_corpus:
            status = "Input corpus is connected but contains no displayable words."
        else:
            status = "Connect a Corpus input to build a word cloud."
        self._status_label.setText(status)

        self._cloud_label.setText(self._cloud_html())
        self._table.setRowCount(len(self._frequencies))
        for row, (word, frequency) in enumerate(self._frequencies):
            self._set_item(row, 0, word)
            self._set_item(row, 1, str(frequency))
        self._table.resizeColumnsToContents()

    def _cloud_html(self) -> str:
        if not self._frequencies:
            return "<span style='color:#7e715e; font-size:16px;'>No words to display.</span>"

        max_frequency = max(frequency for _word, frequency in self._frequencies)
        min_frequency = min(frequency for _word, frequency in self._frequencies)
        spread = max(1, max_frequency - min_frequency)
        spans: list[str] = []
        for index, (word, frequency) in enumerate(self._frequencies):
            size = 14 + int(((frequency - min_frequency) / spread) * 22)
            if max_frequency == min_frequency:
                size = 22
            color = WORD_CLOUD_COLORS[index % len(WORD_CLOUD_COLORS)]
            spans.append(
                "<span style='"
                f"font-size:{size}px; font-weight:700; color:{color}; "
                "padding:4px 7px; line-height:1.9;'>"
                f"{html.escape(word)}</span>"
            )
        return " ".join(spans)

    def _set_item(self, row: int, column: int, text: str) -> None:
        self._table.setItem(row, column, QTableWidgetItem(text))

    def data_preview_snapshot(self) -> dict[str, object]:
        rows = [[word, str(frequency)] for word, frequency in self._frequencies]
        return {
            "summary": (
                f"Word Cloud: {len(self._frequencies)} displayed words, "
                f"{sum(frequency for _word, frequency in self._frequencies)} total frequency"
            ),
            "headers": ["Word", "Frequency"],
            "rows": rows,
        }
