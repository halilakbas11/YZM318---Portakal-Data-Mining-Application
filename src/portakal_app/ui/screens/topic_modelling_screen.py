from __future__ import annotations

import re
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from PySide6.QtCore import QSize
from PySide6.QtWidgets import (
    QComboBox,
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
from portakal_app.ui.screens.corpus_screen import (
    CorpusDocument,
    corpus_documents_from_payload,
    with_corpus_document_attributes,
)
from portakal_app.ui.screens.node_screen import WorkflowNodeScreenSupport
from portakal_app.ui.shared.cards import SectionHeader
from portakal_app.ui.shared.readable_inputs import apply_readable_spin_box_style


METHOD_LDA = "LDA"
METHOD_LSI = "LSI"
METHOD_NMF = "NMF"
TOPIC_METHODS = (METHOD_LDA, METHOD_LSI, METHOD_NMF)
INPUT_BOW = "Using Bag of Words matrix"
INPUT_PREPROCESSED = "Using preprocessed tokens"
INPUT_RAW = "Fallback: rebuilt from raw text"

TOPIC_STOPWORDS = frozenset(
    {
        "a",
        "about",
        "after",
        "all",
        "also",
        "am",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "been",
        "being",
        "before",
        "because",
        "best",
        "but",
        "by",
        "can",
        "category",
        "could",
        "did",
        "do",
        "document",
        "does",
        "doing",
        "during",
        "else",
        "first",
        "for",
        "from",
        "get",
        "had",
        "has",
        "have",
        "having",
        "he",
        "her",
        "his",
        "i",
        "if",
        "how",
        "in",
        "into",
        "is",
        "it",
        "its",
        "just",
        "last",
        "more",
        "my",
        "new",
        "no",
        "nor",
        "not",
        "now",
        "of",
        "on",
        "only",
        "one",
        "or",
        "other",
        "our",
        "ours",
        "out",
        "over",
        "people",
        "said",
        "say",
        "says",
        "she",
        "should",
        "some",
        "so",
        "than",
        "that",
        "the",
        "their",
        "theirs",
        "them",
        "then",
        "there",
        "these",
        "they",
        "this",
        "those",
        "time",
        "title",
        "to",
        "too",
        "two",
        "under",
        "up",
        "use",
        "used",
        "using",
        "very",
        "was",
        "we",
        "were",
        "what",
        "when",
        "where",
        "which",
        "who",
        "whom",
        "why",
        "will",
        "with",
        "without",
        "would",
        "year",
        "years",
        "you",
        "your",
        "yours",
    }
)


@dataclass(frozen=True)
class TopicSummaryRow:
    topic: str
    top_words: str


@dataclass(frozen=True)
class DocumentTopicRow:
    document: str
    topic: str
    score: float


@dataclass(frozen=True)
class TopicModelResult:
    topics: tuple[TopicSummaryRow, ...] = ()
    document_topics: tuple[DocumentTopicRow, ...] = ()
    status: str = ""
    input_source: str = ""
    vocabulary: tuple[str, ...] = ()
    removed_display_stopwords: int = 0
    method: str = METHOD_LDA


@dataclass(frozen=True)
class TopicInput:
    documents: tuple[CorpusDocument, ...]
    matrix: object | None
    vocabulary: tuple[str, ...]
    texts: tuple[str, ...]
    source: str


def build_topic_model(
    documents: Sequence[CorpusDocument],
    *,
    topic_count: int = 5,
    top_words: int = 8,
    method: str = METHOD_LDA,
    max_features: int = 1000,
    min_document_frequency: int = 0,
    max_document_frequency_percent: int = 90,
) -> TopicModelResult:
    topic_input = build_topic_input(
        documents,
        max_features=max_features,
        min_document_frequency=min_document_frequency,
        max_document_frequency_percent=max_document_frequency_percent,
    )
    clean_documents = topic_input.documents
    if len(clean_documents) < 2:
        return TopicModelResult(
            status="At least 2 non-empty documents are required for topic modelling.",
            input_source=topic_input.source,
            vocabulary=topic_input.vocabulary,
            method=method if method in TOPIC_METHODS else METHOD_LDA,
        )

    try:
        from scipy import sparse
        from sklearn.decomposition import LatentDirichletAllocation, NMF, TruncatedSVD
        from sklearn.feature_extraction.text import CountVectorizer
    except Exception as error:
        return TopicModelResult(status=f"Topic modelling backend is unavailable: {error}")

    selected_method = method if method in TOPIC_METHODS else METHOD_LDA
    matrix = topic_input.matrix
    vocabulary = topic_input.vocabulary

    if matrix is None:
        min_df = _effective_min_df(len(clean_documents), min_document_frequency)
        max_df = _effective_max_df(max_document_frequency_percent)
        vectorizer = CountVectorizer(
            lowercase=True,
            token_pattern=r"(?u)\b[a-zA-Z]{3,}\b",
            stop_words=list(TOPIC_STOPWORDS),
            min_df=min_df,
            max_df=max_df,
            max_features=max(10, int(max_features)),
        )
        try:
            matrix = vectorizer.fit_transform(topic_input.texts)
        except ValueError as error:
            return TopicModelResult(
                status=f"Could not build a document-term matrix: {error}",
                input_source=topic_input.source,
                method=selected_method,
            )
        vocabulary = tuple(str(term) for term in vectorizer.get_feature_names_out())

    matrix = sparse.csr_matrix(matrix, dtype=float)
    if matrix.shape[1] < 2:
        return TopicModelResult(
            status="At least 2 unique terms are required for topic modelling.",
            input_source=topic_input.source,
            vocabulary=tuple(vocabulary),
            method=selected_method,
        )

    safe_topic_count = max(1, min(int(topic_count), len(clean_documents), matrix.shape[1]))
    try:
        if selected_method == METHOD_LSI:
            model = TruncatedSVD(n_components=safe_topic_count, random_state=0)
            document_topic_matrix = model.fit_transform(matrix)
            components = np.abs(model.components_)
        elif selected_method == METHOD_NMF:
            model = NMF(n_components=safe_topic_count, init="nndsvda", random_state=0, max_iter=300)
            document_topic_matrix = model.fit_transform(matrix)
            components = model.components_
        else:
            model = LatentDirichletAllocation(
                n_components=safe_topic_count,
                random_state=0,
                learning_method="batch",
                max_iter=20,
            )
            document_topic_matrix = model.fit_transform(matrix)
            components = model.components_
    except Exception as error:
        return TopicModelResult(
            status=f"Could not fit {selected_method} topic model: {error}",
            input_source=topic_input.source,
            vocabulary=tuple(vocabulary),
            method=selected_method,
        )

    vocabulary_tuple = tuple(vocabulary)
    topic_rows: list[TopicSummaryRow] = []
    removed_display_stopwords = 0
    safe_top_words = max(1, min(int(top_words), len(vocabulary_tuple)))
    for topic_index, weights in enumerate(components, start=1):
        words, removed = _top_display_words(weights, vocabulary_tuple, safe_top_words)
        removed_display_stopwords += removed
        topic_rows.append(TopicSummaryRow(f"Topic {topic_index}", ", ".join(words)))

    document_rows: list[DocumentTopicRow] = []
    for document, scores in zip(clean_documents, document_topic_matrix, strict=False):
        numeric_scores = np.asarray(scores, dtype=float)
        topic_index = int(numeric_scores.argmax())
        score = float(numeric_scores[topic_index])
        if selected_method == METHOD_LSI:
            denominator = float(np.linalg.norm(numeric_scores)) or 1.0
            score = abs(score) / denominator
        document_rows.append(
            DocumentTopicRow(
                document.title,
                f"Topic {topic_index + 1}",
                score,
            )
        )

    status = (
        f"{topic_input.source}. Built {safe_topic_count} {selected_method} topics from "
        f"{len(clean_documents)} documents. Vocabulary: {len(vocabulary_tuple)}. "
        f"Display stopword guard removed {removed_display_stopwords} generic terms."
    )
    return TopicModelResult(
        topics=tuple(topic_rows),
        document_topics=tuple(document_rows),
        status=status,
        input_source=topic_input.source,
        vocabulary=vocabulary_tuple,
        removed_display_stopwords=removed_display_stopwords,
        method=selected_method,
    )


def build_topic_input(
    documents: Sequence[CorpusDocument],
    *,
    max_features: int = 1000,
    min_document_frequency: int = 0,
    max_document_frequency_percent: int = 90,
) -> TopicInput:
    bow_input = _topic_input_from_bow(documents)
    if bow_input is not None:
        return bow_input

    processed_documents: list[CorpusDocument] = []
    processed_texts: list[str] = []
    raw_documents: list[CorpusDocument] = []
    raw_texts: list[str] = []
    for document in documents:
        text = str(document.text or "").strip()
        if not text:
            continue
        if _looks_preprocessed(text):
            processed_documents.append(document)
            processed_texts.append(_clean_token_text(text))
        else:
            raw_documents.append(document)
            raw_texts.append(text)

    if processed_documents:
        return TopicInput(tuple(processed_documents), None, (), tuple(processed_texts), INPUT_PREPROCESSED)

    cleaned_raw_texts = tuple(_clean_token_text(text) for text in raw_texts)
    cleaned_documents = tuple(
        document
        for document, text in zip(raw_documents, cleaned_raw_texts, strict=False)
        if text.strip()
    )
    cleaned_texts = tuple(text for text in cleaned_raw_texts if text.strip())
    return TopicInput(cleaned_documents, None, (), cleaned_texts, INPUT_RAW)


def _topic_input_from_bow(documents: Sequence[CorpusDocument]) -> TopicInput | None:
    rows: list[dict[str, float]] = []
    output_documents: list[CorpusDocument] = []
    vocabulary_set: set[str] = set()
    for document in documents:
        attrs = dict(document.attributes)
        bow_terms: dict[str, float] = {}
        for key, value in attrs.items():
            if not str(key).startswith("bow_") or key == "bow_total":
                continue
            term = str(key)[4:]
            try:
                numeric_value = float(value)
            except (TypeError, ValueError):
                continue
            if numeric_value <= 0:
                continue
            bow_terms[term] = numeric_value
            vocabulary_set.add(term)
        if bow_terms:
            rows.append(bow_terms)
            output_documents.append(document)

    if not rows or not vocabulary_set:
        return None

    from scipy import sparse

    vocabulary = tuple(sorted(vocabulary_set))
    if len(vocabulary) < 2:
        return None
    column_by_term = {term: index for index, term in enumerate(vocabulary)}
    data: list[float] = []
    row_indices: list[int] = []
    column_indices: list[int] = []
    for row_index, row in enumerate(rows):
        for term, value in row.items():
            column_index = column_by_term.get(term)
            if column_index is None:
                continue
            data.append(float(value))
            row_indices.append(row_index)
            column_indices.append(column_index)
    matrix = sparse.csr_matrix(
        (data, (row_indices, column_indices)),
        shape=(len(rows), len(vocabulary)),
        dtype=float,
    )
    return TopicInput(tuple(output_documents), matrix, vocabulary, (), INPUT_BOW)


def _top_display_words(
    weights: Sequence[float],
    vocabulary: Sequence[str],
    top_words: int,
) -> tuple[tuple[str, ...], int]:
    order = np.asarray(weights).argsort()[::-1]
    words: list[str] = []
    removed = 0
    for index in order:
        word = _clean_topic_word(vocabulary[int(index)])
        if not word:
            removed += 1
            continue
        if word in words:
            continue
        words.append(word)
        if len(words) >= top_words:
            break
    return tuple(words), removed


def _clean_topic_word(word: str) -> str:
    normalized = str(word).strip().lower()
    if not re.fullmatch(r"[a-zA-Z]{3,}", normalized):
        return ""
    if normalized in TOPIC_STOPWORDS:
        return ""
    return normalized


def _clean_token_text(text: str) -> str:
    tokens = re.findall(r"[a-zA-Z]{3,}", text.lower())
    return " ".join(token for token in tokens if token not in TOPIC_STOPWORDS)


def _looks_preprocessed(text: str) -> bool:
    return bool(text.strip()) and text == text.lower() and not re.search(r"[^\w\s]", text)


def _effective_min_df(document_count: int, requested: int) -> int:
    if requested > 0:
        return max(1, min(int(requested), max(1, document_count)))
    return 2 if document_count >= 20 else 1


def _effective_max_df(percent: int) -> float:
    return min(1.0, max(0.05, int(percent) / 100))


class TopicModellingScreen(QWidget, WorkflowNodeScreenSupport):
    def __init__(
        self,
        parent: QWidget | None = None,
        documents: Sequence[CorpusDocument] | None = None,
    ) -> None:
        super().__init__(parent)
        self._init_workflow_node_support()
        self._documents = tuple(() if documents is None else documents)
        self._using_input_corpus = documents is not None
        self._result = TopicModelResult()
        self._output_documents: tuple[CorpusDocument, ...] = self._documents

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(10)

        layout.addWidget(
            SectionHeader(
                "Topic Modelling",
                "Discover topics from a Bag of Words matrix or cleaned corpus text.",
            )
        )
        layout.addWidget(self._build_options_panel())
        layout.addWidget(self._build_metadata_panel())
        layout.addWidget(self._build_topics_panel(), 1)
        layout.addWidget(self._build_documents_panel(), 1)

        self.apply_options()

    def sizeHint(self) -> QSize:
        return QSize(980, 720)

    def minimumSizeHint(self) -> QSize:
        return QSize(740, 560)

    def _build_options_panel(self) -> QFrame:
        frame = QFrame(self)
        frame.setProperty("panel", True)
        layout = QHBoxLayout(frame)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        layout.addWidget(QLabel("Method", self))
        self._method_combo = QComboBox(self)
        self._method_combo.addItems(TOPIC_METHODS)
        layout.addWidget(self._method_combo)

        layout.addWidget(QLabel("Topics", self))
        self._topic_count_spinbox = QSpinBox(self)
        self._topic_count_spinbox.setRange(1, 20)
        self._topic_count_spinbox.setValue(5)
        apply_readable_spin_box_style(self._topic_count_spinbox)
        layout.addWidget(self._topic_count_spinbox)

        layout.addWidget(QLabel("Top words", self))
        self._top_words_spinbox = QSpinBox(self)
        self._top_words_spinbox.setRange(2, 20)
        self._top_words_spinbox.setValue(8)
        apply_readable_spin_box_style(self._top_words_spinbox)
        layout.addWidget(self._top_words_spinbox)

        layout.addWidget(QLabel("Max features", self))
        self._max_features_spinbox = QSpinBox(self)
        self._max_features_spinbox.setRange(10, 10000)
        self._max_features_spinbox.setValue(1000)
        apply_readable_spin_box_style(self._max_features_spinbox)
        layout.addWidget(self._max_features_spinbox)

        layout.addWidget(QLabel("Min document frequency", self))
        self._min_df_spinbox = QSpinBox(self)
        self._min_df_spinbox.setRange(0, 999)
        self._min_df_spinbox.setValue(0)
        self._min_df_spinbox.setToolTip("0 = automatic. Uses 2 for corpora with 20 or more documents, otherwise 1.")
        apply_readable_spin_box_style(self._min_df_spinbox)
        layout.addWidget(self._min_df_spinbox)

        layout.addWidget(QLabel("Max document frequency", self))
        self._max_df_spinbox = QSpinBox(self)
        self._max_df_spinbox.setRange(5, 100)
        self._max_df_spinbox.setValue(90)
        self._max_df_spinbox.setSuffix("%")
        apply_readable_spin_box_style(self._max_df_spinbox)
        layout.addWidget(self._max_df_spinbox)

        self._apply_button = QPushButton("Model Topics", self)
        self._apply_button.setProperty("primary", True)
        self._apply_button.clicked.connect(self.apply_options)
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
        self._topic_count_label = self._build_metric_label("Topics", "0")
        self._assignment_count_label = self._build_metric_label("Assignments", "0")

        layout.addWidget(self._document_count_label, 1)
        layout.addWidget(self._topic_count_label, 1)
        layout.addWidget(self._assignment_count_label, 1)
        return frame

    def _build_metric_label(self, title: str, value: str) -> QLabel:
        label = QLabel(f"{title}\n{value}", self)
        label.setProperty("infoCard", True)
        label.setStyleSheet("padding: 10px;")
        return label

    def _build_topics_panel(self) -> QFrame:
        frame = QFrame(self)
        frame.setProperty("panel", True)
        layout = QVBoxLayout(frame)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)

        self._status_label = QLabel("", self)
        self._status_label.setProperty("muted", True)
        self._status_label.setWordWrap(True)
        layout.addWidget(self._status_label)

        self._topics_table = QTableWidget(0, 2, self)
        self._topics_table.setHorizontalHeaderLabels(["Topic", "Top Words"])
        self._topics_table.verticalHeader().setVisible(False)
        self._topics_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._topics_table.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(self._topics_table, 1)
        return frame

    def _build_documents_panel(self) -> QFrame:
        frame = QFrame(self)
        frame.setProperty("panel", True)
        layout = QVBoxLayout(frame)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)

        self._documents_table = QTableWidget(0, 3, self)
        self._documents_table.setHorizontalHeaderLabels(["Document", "Dominant Topic", "Topic Score"])
        self._documents_table.verticalHeader().setVisible(False)
        self._documents_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._documents_table.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(self._documents_table, 1)
        return frame

    def set_input_payload(self, payload: WorkflowPayload | None) -> None:
        if payload is None:
            self._documents = ()
            self._using_input_corpus = False
            self.apply_options()
            return

        documents = corpus_documents_from_payload(payload.value)
        self._documents = () if documents is None else documents
        self._using_input_corpus = True
        print(f"Topic Modelling received corpus: {len(self._documents)} documents")
        self.apply_options()

    def current_output_payload(self) -> WorkflowPayload:
        return WorkflowPayload("Corpus", self._output_documents)

    def current_output_payloads(self) -> dict[str, WorkflowPayload | None]:
        rows = [[row.topic, row.top_words] for row in self._result.topics]
        return {
            "Corpus": WorkflowPayload("Corpus", self._output_documents),
            "Topic": WorkflowPayload("Topic", rows),
        }

    def apply_options(self) -> TopicModelResult:
        self._result = build_topic_model(
            self._documents,
            topic_count=self._topic_count_spinbox.value(),
            top_words=self._top_words_spinbox.value(),
            method=self._method_combo.currentText(),
            max_features=self._max_features_spinbox.value(),
            min_document_frequency=self._min_df_spinbox.value(),
            max_document_frequency_percent=self._max_df_spinbox.value(),
        )
        self._output_documents = self._build_output_documents()
        self._render()
        self._notify_output_changed()
        return self._result

    def _build_output_documents(self) -> tuple[CorpusDocument, ...]:
        by_title = {row.document: row for row in self._result.document_topics}
        return tuple(
            with_corpus_document_attributes(
                document,
                {
                    "topic": by_title[document.title].topic,
                    "topic_score": by_title[document.title].score,
                },
            )
            if document.title in by_title
            else document
            for document in self._documents
        )

    def _render(self) -> None:
        self._document_count_label.setText(f"Documents\n{len(self._documents)}")
        self._topic_count_label.setText(f"Topics\n{len(self._result.topics)}")
        self._assignment_count_label.setText(f"Assignments\n{len(self._result.document_topics)}")
        if not self._using_input_corpus:
            self._status_label.setText("Connect a Corpus input to model topics.")
        else:
            self._status_label.setText(self._result.status or "Input corpus is connected.")

        self._topics_table.setRowCount(len(self._result.topics))
        for row, topic in enumerate(self._result.topics):
            self._topics_table.setItem(row, 0, QTableWidgetItem(topic.topic))
            self._topics_table.setItem(row, 1, QTableWidgetItem(topic.top_words))
        self._topics_table.resizeColumnsToContents()

        self._documents_table.setRowCount(len(self._result.document_topics))
        for row, item in enumerate(self._result.document_topics):
            self._documents_table.setItem(row, 0, QTableWidgetItem(item.document))
            self._documents_table.setItem(row, 1, QTableWidgetItem(item.topic))
            self._documents_table.setItem(row, 2, QTableWidgetItem(f"{item.score:.3f}"))
        self._documents_table.resizeColumnsToContents()

    def data_preview_snapshot(self) -> dict[str, object]:
        rows = [[topic.topic, topic.top_words] for topic in self._result.topics]
        return {
            "summary": f"Topic Modelling: {len(self._result.topics)} topics, {len(self._result.document_topics)} document assignments",
            "headers": ["Topic", "Top Words"],
            "rows": rows,
        }
