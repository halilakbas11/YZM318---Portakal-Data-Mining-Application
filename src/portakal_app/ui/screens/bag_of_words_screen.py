from __future__ import annotations

import re
from math import log, sqrt
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass

from PySide6.QtCore import QSize
from PySide6.QtWidgets import (
    QComboBox,
    QFrame,
    QGridLayout,
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
from portakal_app.ui.screens.corpus_screen import CorpusDocument, corpus_documents_from_payload
from portakal_app.ui.screens.node_screen import WorkflowNodeScreenSupport
from portakal_app.ui.shared.cards import SectionHeader


TF_COUNT = "count"
TF_BINARY = "binary"
TF_SUBLINEAR = "sublinear"
DF_NONE = "none"
DF_IDF = "idf"
DF_SMOOTH_IDF = "smooth_idf"
NORM_NONE = "none"
NORM_L1 = "l1"
NORM_L2 = "l2"


@dataclass(frozen=True)
class BagOfWordsSummary:
    document_count: int
    vocabulary_size: int
    total_token_count: float
    most_frequent_term: str
    nonzero_entries: int = 0
    density: float = 0.0


class DocumentTermMatrix(tuple):
    def __getitem__(self, index):
        value = super().__getitem__(index)
        if isinstance(index, slice):
            return DocumentTermMatrix(value)
        return value

    @property
    def shape(self) -> tuple[int, int]:
        row_count = len(self)
        column_count = len(self[0]) if row_count else 0
        return row_count, column_count

    def tolist(self) -> list[list[float]]:
        return [list(row) for row in self]


@dataclass(frozen=True)
class BagOfWordsCorpus(Sequence[CorpusDocument]):
    documents: tuple[CorpusDocument, ...]
    vocabulary: tuple[str, ...]
    matrix: tuple[tuple[float, ...], ...]
    local_weighting: str = TF_COUNT
    global_weighting: str = DF_NONE
    normalization: str = NORM_NONE

    def __post_init__(self) -> None:
        matrix = DocumentTermMatrix(tuple(tuple(float(value) for value in row) for row in self.matrix))
        object.__setattr__(self, "matrix", matrix)

    def __len__(self) -> int:
        return len(self.documents)

    def __getitem__(self, index):
        return self.documents[index]

    @property
    def matrix_kind(self) -> str:
        if (
            self.local_weighting == TF_COUNT
            and self.global_weighting == DF_NONE
            and self.normalization == NORM_NONE
        ):
            return "count"
        return "weighted"


def tokenize_for_bow(text: str) -> tuple[str, ...]:
    normalized = re.sub(r"[^\w\s]", " ", text.lower())
    return tuple(normalized.split())


def build_vocabulary(documents: Sequence[CorpusDocument], min_frequency: int = 1) -> tuple[str, ...]:
    threshold = max(1, min_frequency)
    counter: Counter[str] = Counter()
    for document in documents:
        counter.update(tokenize_for_bow(document.text))
    return tuple(sorted(term for term, count in counter.items() if count >= threshold))


def build_weighted_document_term_matrix(
    documents: Sequence[CorpusDocument],
    vocabulary: Sequence[str],
    *,
    local_weighting: str = TF_COUNT,
    global_weighting: str = DF_NONE,
    normalization: str = NORM_NONE,
) -> tuple[tuple[float, ...], ...]:
    raw_matrix = tuple(
        _document_term_counts(document, vocabulary)
        for document in documents
    )
    locally_weighted = tuple(apply_local_weighting(row, local_weighting) for row in raw_matrix)
    globally_weighted = apply_global_weighting(locally_weighted, global_weighting)
    return normalize_matrix(globally_weighted, normalization)


def apply_local_weighting(row: Sequence[float], mode: str = TF_COUNT) -> tuple[float, ...]:
    if mode == TF_BINARY:
        return tuple(1.0 if value > 0 else 0.0 for value in row)
    if mode == TF_SUBLINEAR:
        return tuple(1.0 + log(value) if value > 0 else 0.0 for value in row)
    return tuple(float(value) for value in row)


def document_frequencies(matrix: Sequence[Sequence[float]], vocabulary: Sequence[str]) -> dict[str, int]:
    return {
        term: sum(1 for row in matrix if row[index] > 0)
        for index, term in enumerate(vocabulary)
    }


def apply_global_weighting(
    matrix: Sequence[Sequence[float]],
    mode: str = DF_NONE,
) -> tuple[tuple[float, ...], ...]:
    if not matrix or mode == DF_NONE:
        return tuple(tuple(float(value) for value in row) for row in matrix)
    document_count = len(matrix)
    term_count = len(matrix[0]) if matrix else 0
    dfs = [sum(1 for row in matrix if row[index] > 0) for index in range(term_count)]
    weights: list[float] = []
    for df in dfs:
        if df <= 0:
            weights.append(0.0)
        elif mode == DF_SMOOTH_IDF:
            weights.append(log(1.0 + document_count / df))
        elif mode == DF_IDF:
            weights.append(log(document_count / df))
        else:
            weights.append(1.0)
    return tuple(
        tuple(value * weights[index] for index, value in enumerate(row))
        for row in matrix
    )


def normalize_matrix(
    matrix: Sequence[Sequence[float]],
    mode: str = NORM_NONE,
) -> tuple[tuple[float, ...], ...]:
    if mode == NORM_NONE:
        return tuple(tuple(float(value) for value in row) for row in matrix)
    normalized_rows: list[tuple[float, ...]] = []
    for row in matrix:
        if mode == NORM_L1:
            denominator = sum(abs(value) for value in row)
        elif mode == NORM_L2:
            denominator = sqrt(sum(value * value for value in row))
        else:
            denominator = 0.0
        if denominator == 0:
            normalized_rows.append(tuple(0.0 for _value in row))
        else:
            normalized_rows.append(tuple(value / denominator for value in row))
    return tuple(normalized_rows)


def format_matrix_value(value: float) -> str:
    if abs(value - round(value)) < 1e-9:
        return str(int(round(value)))
    return f"{value:.3f}".rstrip("0").rstrip(".")


def weighting_summary(local_weighting: str, global_weighting: str, normalization: str) -> str:
    local_labels = {
        TF_COUNT: "Count",
        TF_BINARY: "Binary",
        TF_SUBLINEAR: "Sublinear",
    }
    global_labels = {
        DF_NONE: "None",
        DF_IDF: "IDF",
        DF_SMOOTH_IDF: "Smooth IDF",
    }
    norm_labels = {
        NORM_NONE: "None",
        NORM_L1: "L1",
        NORM_L2: "L2",
    }
    return (
        f"TF={local_labels.get(local_weighting, 'Count')}, "
        f"DF={global_labels.get(global_weighting, 'None')}, "
        f"Norm={norm_labels.get(normalization, 'None')}"
    )


def _document_term_counts(
    document: CorpusDocument,
    vocabulary: Sequence[str],
) -> tuple[float, ...]:
    token_counts = Counter(tokenize_for_bow(document.text))
    return tuple(float(token_counts[term]) for term in vocabulary)


def _legacy_document_term_counts(
    document: CorpusDocument,
    vocabulary: Sequence[str],
    binary: bool = False,
) -> tuple[int, ...]:
    token_counts = Counter(tokenize_for_bow(document.text))
    if binary:
        return tuple(1 if token_counts[term] else 0 for term in vocabulary)
    return tuple(token_counts[term] for term in vocabulary)


def _legacy_build_document_term_matrix(
    documents: Sequence[CorpusDocument],
    vocabulary: Sequence[str],
    binary: bool = False,
) -> tuple[tuple[int, ...], ...]:
    return tuple(
        _legacy_document_term_counts(document, vocabulary, binary=binary)
        for document in documents
    )


def build_document_term_matrix(
    documents: Sequence[CorpusDocument],
    vocabulary: Sequence[str],
    binary: bool = False,
) -> tuple[tuple[int, ...], ...]:
    return _legacy_build_document_term_matrix(documents, vocabulary, binary=binary)


def term_frequencies(matrix: Sequence[Sequence[float]], vocabulary: Sequence[str]) -> dict[str, float]:
    return {
        term: sum(row[index] for row in matrix)
        for index, term in enumerate(vocabulary)
    }


def summarize_bow(
    documents: Sequence[CorpusDocument],
    vocabulary: Sequence[str],
    matrix: Sequence[Sequence[float]],
) -> BagOfWordsSummary:
    frequencies = term_frequencies(matrix, vocabulary)
    most_frequent_term = "None"
    most_frequent_count = 0.0
    for term in vocabulary:
        count = frequencies[term]
        if count > most_frequent_count:
            most_frequent_term = term
            most_frequent_count = count
    total_cells = len(documents) * len(vocabulary)
    nonzero_entries = sum(1 for row in matrix for value in row if value != 0)
    density = nonzero_entries / total_cells if total_cells else 0.0

    return BagOfWordsSummary(
        document_count=len(documents),
        vocabulary_size=len(vocabulary),
        total_token_count=sum(sum(row) for row in matrix),
        most_frequent_term=most_frequent_term,
        nonzero_entries=nonzero_entries,
        density=density,
    )


class BagOfWordsScreen(QWidget, WorkflowNodeScreenSupport):
    def __init__(
        self,
        parent: QWidget | None = None,
        documents: Sequence[CorpusDocument] | None = None,
    ) -> None:
        super().__init__(parent)
        self._init_workflow_node_support()
        self._documents = tuple(() if documents is None else documents)
        self._using_input_corpus = documents is not None
        self._vocabulary: tuple[str, ...] = ()
        self._matrix: tuple[tuple[float, ...], ...] = ()

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(10)

        layout.addWidget(
            SectionHeader(
                "Bag of Words",
                "Convert text documents into an Orange-inspired document-term representation.",
            )
        )
        layout.addWidget(self._build_options_panel())
        layout.addWidget(self._build_metadata_panel())
        layout.addWidget(self._build_table_panel(), 1)

        self.apply_options()

    def sizeHint(self) -> QSize:
        return QSize(1020, 680)

    def minimumSizeHint(self) -> QSize:
        return QSize(760, 520)

    def _build_options_panel(self) -> QFrame:
        frame = QFrame(self)
        frame.setProperty("panel", True)
        layout = QGridLayout(frame)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setHorizontalSpacing(14)
        layout.setVerticalSpacing(8)

        self._term_frequency_label = QLabel("Term Frequency", self)
        layout.addWidget(self._term_frequency_label, 0, 0)

        self._term_frequency_combo = QComboBox(self)
        self._term_frequency_combo.addItem("Count", TF_COUNT)
        self._term_frequency_combo.addItem("Binary", TF_BINARY)
        self._term_frequency_combo.addItem("Sublinear", TF_SUBLINEAR)
        layout.addWidget(self._term_frequency_combo, 0, 1)

        self._document_frequency_label = QLabel("Document Frequency", self)
        layout.addWidget(self._document_frequency_label, 0, 2)

        self._document_frequency_combo = QComboBox(self)
        self._document_frequency_combo.addItem("None", DF_NONE)
        self._document_frequency_combo.addItem("IDF", DF_IDF)
        self._document_frequency_combo.addItem("Smooth IDF", DF_SMOOTH_IDF)
        layout.addWidget(self._document_frequency_combo, 0, 3)

        self._normalization_label = QLabel("Normalization", self)
        layout.addWidget(self._normalization_label, 1, 0)

        self._normalization_combo = QComboBox(self)
        self._normalization_combo.addItem("None", NORM_NONE)
        self._normalization_combo.addItem("L1", NORM_L1)
        self._normalization_combo.addItem("L2", NORM_L2)
        layout.addWidget(self._normalization_combo, 1, 1)

        self._min_frequency_label = QLabel("Minimum term frequency", self)
        layout.addWidget(self._min_frequency_label, 1, 2)

        self._min_frequency_spinbox = QSpinBox(self)
        self._min_frequency_spinbox.setRange(1, 999)
        self._min_frequency_spinbox.setValue(1)
        layout.addWidget(self._min_frequency_spinbox, 1, 3)

        self._apply_button = QPushButton("Apply Bag of Words", self)
        self._apply_button.setProperty("primary", True)
        self._apply_button.clicked.connect(self.apply_options)
        layout.addWidget(self._apply_button, 1, 4)
        return frame

    def _build_metadata_panel(self) -> QFrame:
        frame = QFrame(self)
        frame.setProperty("panel", True)
        layout = QHBoxLayout(frame)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        self._document_count_label = self._build_metric_label("Documents", "0")
        self._vocabulary_size_label = self._build_metric_label("Vocabulary Size", "0")
        self._total_token_count_label = self._build_metric_label("Total Tokens", "0")
        self._most_frequent_term_label = self._build_metric_label("Most Frequent Term", "None")
        self._nonzero_entries_label = self._build_metric_label("Non-zero Entries", "0")
        self._density_label = self._build_metric_label("Density", "0.0%")

        layout.addWidget(self._document_count_label, 1)
        layout.addWidget(self._vocabulary_size_label, 1)
        layout.addWidget(self._total_token_count_label, 1)
        layout.addWidget(self._most_frequent_term_label, 1)
        layout.addWidget(self._nonzero_entries_label, 1)
        layout.addWidget(self._density_label, 1)
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

        self._table = QTableWidget(0, 2, self)
        self._table.setHorizontalHeaderLabels(["Document", "Total"])
        self._table.verticalHeader().setVisible(False)
        self._table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self._table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self._table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._table.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(self._table, 1)
        return frame

    def apply_options(self) -> tuple[tuple[float, ...], ...]:
        local_weighting = self._term_frequency_combo.currentData() or TF_COUNT
        global_weighting = self._document_frequency_combo.currentData() or DF_NONE
        normalization = self._normalization_combo.currentData() or NORM_NONE
        self._vocabulary = build_vocabulary(
            self._documents,
            min_frequency=self._min_frequency_spinbox.value(),
        )
        self._matrix = build_weighted_document_term_matrix(
            self._documents,
            self._vocabulary,
            local_weighting=local_weighting,
            global_weighting=global_weighting,
            normalization=normalization,
        )
        self._render()
        self._notify_output_changed()
        return self._matrix

    def set_documents(self, documents: Sequence[CorpusDocument]) -> None:
        self._documents = tuple(documents)
        self._using_input_corpus = True
        self.apply_options()

    def set_input_payload(self, payload: WorkflowPayload | None) -> None:
        if payload is None:
            self._documents = ()
            self._using_input_corpus = False
            self.apply_options()
            return

        documents = corpus_documents_from_payload(payload.value)
        self._documents = () if documents is None else documents
        self._using_input_corpus = True
        self.apply_options()

    def current_output_payload(self) -> WorkflowPayload:
        return WorkflowPayload(
            "Corpus",
            BagOfWordsCorpus(
                documents=self._documents,
                vocabulary=self._vocabulary,
                matrix=self._matrix,
                local_weighting=self._term_frequency_combo.currentData() or TF_COUNT,
                global_weighting=self._document_frequency_combo.currentData() or DF_NONE,
                normalization=self._normalization_combo.currentData() or NORM_NONE,
            ),
        )

    def _render(self) -> None:
        summary = summarize_bow(self._documents, self._vocabulary, self._matrix)
        self._document_count_label.setText(f"Documents\n{summary.document_count}")
        self._vocabulary_size_label.setText(f"Vocabulary Size\n{summary.vocabulary_size}")
        self._total_token_count_label.setText(f"Total Tokens\n{format_matrix_value(summary.total_token_count)}")
        self._most_frequent_term_label.setText(f"Most Frequent Term\n{summary.most_frequent_term}")
        self._nonzero_entries_label.setText(f"Non-zero Entries\n{summary.nonzero_entries}")
        self._density_label.setText(f"Density\n{summary.density:.1%}")
        if self._vocabulary and self._using_input_corpus:
            status = "Input corpus is connected and converted to a document-term matrix."
        elif self._using_input_corpus:
            status = "Input corpus is connected but produced no vocabulary terms."
        else:
            status = "Connect a Corpus input to build a document-term matrix."
        self._status_label.setText(
            status
            + " "
            + weighting_summary(
                self._term_frequency_combo.currentData() or TF_COUNT,
                self._document_frequency_combo.currentData() or DF_NONE,
                self._normalization_combo.currentData() or NORM_NONE,
            )
            + "."
        )

        headers = ["Document", *self._vocabulary, "Total"]
        self._table.setColumnCount(len(headers))
        self._table.setHorizontalHeaderLabels(headers)
        self._table.setRowCount(len(self._documents) + (1 if self._vocabulary else 0))

        for row, (document, counts) in enumerate(zip(self._documents, self._matrix)):
            self._set_item(row, 0, document.title)
            for column, count in enumerate(counts, start=1):
                self._set_item(row, column, format_matrix_value(count))
            self._set_item(row, len(headers) - 1, format_matrix_value(sum(counts)))

        if self._vocabulary:
            totals = term_frequencies(self._matrix, self._vocabulary)
            total_row = len(self._documents)
            self._set_item(total_row, 0, "Total Frequency")
            for column, term in enumerate(self._vocabulary, start=1):
                self._set_item(total_row, column, format_matrix_value(totals[term]))
            self._set_item(total_row, len(headers) - 1, format_matrix_value(sum(totals.values())))

        self._table.resizeColumnsToContents()

    def _set_item(self, row: int, column: int, text: str) -> None:
        self._table.setItem(row, column, QTableWidgetItem(text))

    def data_preview_snapshot(self) -> dict[str, object]:
        headers = ["Document", *self._vocabulary, "Total"]
        rows: list[list[str]] = []
        for document, counts in zip(self._documents, self._matrix):
            rows.append(
                [document.title, *(format_matrix_value(count) for count in counts), format_matrix_value(sum(counts))]
            )

        if self._vocabulary:
            totals = term_frequencies(self._matrix, self._vocabulary)
            rows.append(
                [
                    "Total Frequency",
                    *(format_matrix_value(totals[term]) for term in self._vocabulary),
                    format_matrix_value(sum(totals.values())),
                ]
            )

        summary = summarize_bow(self._documents, self._vocabulary, self._matrix)
        return {
            "summary": (
                f"Bag of Words: {summary.document_count} documents, "
                f"{summary.vocabulary_size} terms, "
                f"{format_matrix_value(summary.total_token_count)} total tokens, "
                f"{summary.nonzero_entries} non-zero entries, "
                f"{summary.density:.1%} density"
            ),
            "headers": headers,
            "rows": rows,
        }
