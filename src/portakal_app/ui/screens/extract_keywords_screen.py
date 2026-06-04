from __future__ import annotations

import math
import re
import statistics
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

from PySide6.QtCore import QSize, Qt
from PySide6.QtWidgets import (
    QComboBox,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QHeaderView,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from portakal_app.data.models import DatasetHandle
from portakal_app.models import WorkflowPayload
from portakal_app.ui.screens.bag_of_words_screen import tokenize_for_bow
from portakal_app.ui.screens.corpus_screen import CorpusDocument, corpus_documents_from_payload
from portakal_app.ui.screens.node_screen import WorkflowNodeScreenSupport
from portakal_app.ui.shared.cards import SectionHeader
from portakal_app.ui.shared.readable_inputs import apply_readable_spin_box_style


METHOD_TFIDF = "TF-IDF"
METHOD_YAKE = "YAKE"
METHOD_RAKE = "RAKE"
SCORE_METHODS = (METHOD_TFIDF, METHOD_YAKE, METHOD_RAKE)
AGGREGATIONS = ("Mean", "Max", "Median", "Sum")
SELECTION_NONE = "None"
SELECTION_ALL = "All"
SELECTION_MANUAL = "Manual"
SELECTION_TOP_N = "Top words N"
SELECTION_MODES = (SELECTION_ALL, SELECTION_NONE, SELECTION_MANUAL, SELECTION_TOP_N)
DEFAULT_MAX_DF_RATIO = 0.90
AUTO_MIN_DF_THRESHOLD = 20
MIN_KEYWORD_LENGTH = 3
MAX_RAKE_PHRASE_LENGTH = 3

ENGLISH_STOPWORDS = frozenset(
    {
        "a",
        "about",
        "after",
        "again",
        "all",
        "also",
        "an",
        "and",
        "any",
        "are",
        "as",
        "at",
        "be",
        "because",
        "been",
        "before",
        "being",
        "between",
        "both",
        "but",
        "by",
        "can",
        "did",
        "do",
        "does",
        "doing",
        "down",
        "during",
        "each",
        "few",
        "for",
        "from",
        "further",
        "had",
        "has",
        "have",
        "having",
        "he",
        "her",
        "here",
        "hers",
        "him",
        "his",
        "how",
        "i",
        "if",
        "in",
        "into",
        "is",
        "it",
        "its",
        "itself",
        "just",
        "more",
        "most",
        "no",
        "nor",
        "not",
        "of",
        "off",
        "on",
        "once",
        "only",
        "or",
        "other",
        "our",
        "out",
        "over",
        "own",
        "same",
        "she",
        "should",
        "so",
        "some",
        "such",
        "than",
        "that",
        "the",
        "their",
        "them",
        "then",
        "there",
        "these",
        "they",
        "this",
        "those",
        "through",
        "to",
        "too",
        "under",
        "until",
        "up",
        "very",
        "was",
        "we",
        "were",
        "what",
        "when",
        "where",
        "which",
        "while",
        "who",
        "why",
        "will",
        "with",
        "you",
        "your",
    }
)
NEWS_NOISE_WORDS = frozenset(
    {
        "also",
        "category",
        "could",
        "document",
        "mr",
        "mrs",
        "ms",
        "new",
        "one",
        "people",
        "said",
        "says",
        "title",
        "would",
        "year",
        "years",
    }
)
DEFAULT_KEYWORD_STOPWORDS = ENGLISH_STOPWORDS | NEWS_NOISE_WORDS


@dataclass(frozen=True)
class KeywordItem:
    document: str
    keyword: str
    score: float
    frequency: int
    tfidf: float | None = None
    yake: float | None = None
    rake: float | None = None


@dataclass(frozen=True)
class KeywordScores:
    keyword: str
    tfidf: float | None = None
    yake: float | None = None
    rake: float | None = None
    rake_raw: float | None = None
    frequency: int = 0

    def score_for(self, method: str) -> float | None:
        if method == METHOD_TFIDF:
            return self.tfidf
        if method == METHOD_YAKE:
            return self.yake
        if method == METHOD_RAKE:
            return self.rake
        return None


@dataclass(frozen=True)
class KeywordInput:
    documents: tuple[CorpusDocument, ...]
    token_documents: tuple[tuple[str, ...], ...]
    texts: tuple[str, ...]
    source_label: str
    vocabulary: tuple[str, ...] = ()
    # Lightly cleaned, natural text per document (stopwords + word order preserved).
    # RAKE and YAKE consume this view; TF-IDF consumes ``token_documents`` instead.
    light_texts: tuple[str, ...] = ()
    # True when at least one document exposed real raw text (stopwords intact) for
    # the light view; False means the light view was reconstructed from processed
    # tokens, which degrades RAKE/YAKE phrase quality.
    has_natural_text: bool = False


class NumericTableItem(QTableWidgetItem):
    def __init__(self, text: str, value: float | None = None) -> None:
        super().__init__(text)
        self._sort_value = value
        if value is not None:
            self.setData(Qt.ItemDataRole.UserRole, value)

    def __lt__(self, other: QTableWidgetItem) -> bool:
        if isinstance(other, NumericTableItem):
            if self._sort_value is None:
                return False
            if other._sort_value is None:
                return True
            return self._sort_value < other._sort_value
        return super().__lt__(other)


def extract_keywords(
    documents: Sequence[CorpusDocument],
    *,
    top_n: int = 5,
    min_frequency: int = 1,
) -> tuple[KeywordItem, ...]:
    rows = extract_keyword_scores(
        documents,
        top_n=top_n,
        min_frequency=min_frequency,
        aggregation="Mean",
    )[0]
    return tuple(
        KeywordItem(
            "Corpus",
            item.keyword,
            _first_available_score(item),
            item.frequency,
            tfidf=item.tfidf,
            yake=item.yake,
            rake=item.rake,
        )
        for item in rows
    )


def extract_keyword_scores(
    documents: Sequence[CorpusDocument],
    *,
    top_n: int = 50,
    min_frequency: int = 1,
    aggregation: str = "Mean",
    min_document_frequency: int = 0,
    max_document_frequency_ratio: float = DEFAULT_MAX_DF_RATIO,
) -> tuple[tuple[KeywordScores, ...], tuple[str, ...]]:
    keyword_input = _build_keyword_input(documents)
    if not keyword_input.documents:
        return (), ("No usable text was found in the input corpus.",)

    warnings: list[str] = []
    threshold = max(1, int(min_frequency))
    effective_min_df = _effective_min_document_frequency(
        len(keyword_input.documents),
        min_document_frequency,
    )
    effective_max_df = _effective_max_document_frequency(
        len(keyword_input.documents),
        max_document_frequency_ratio,
        effective_min_df,
    )
    aggregation_name = aggregation if aggregation in AGGREGATIONS else "Mean"
    score_maps: dict[str, dict[str, float]] = {
        METHOD_TFIDF: {},
        METHOD_YAKE: {},
        METHOD_RAKE: {},
    }
    rake_raw_scores: dict[str, float] = {}

    tfidf_scores, tfidf_warning = _score_tfidf(
        keyword_input,
        threshold,
        effective_min_df,
        effective_max_df,
        aggregation_name,
    )
    score_maps[METHOD_TFIDF] = tfidf_scores
    if tfidf_warning:
        warnings.append(tfidf_warning)

    yake_scores, yake_warning = _score_yake(keyword_input, aggregation_name)
    score_maps[METHOD_YAKE] = yake_scores
    if yake_warning:
        warnings.append(yake_warning)

    rake_scores, rake_raw_scores = _score_rake(keyword_input, aggregation_name)
    score_maps[METHOD_RAKE] = rake_scores

    candidates = set().union(*(set(scores) for scores in score_maps.values()))
    frequencies = _candidate_frequencies(
        keyword_input.token_documents, candidates, keyword_input.light_texts
    )
    document_frequencies = _candidate_document_frequencies(
        keyword_input.token_documents, candidates, keyword_input.light_texts
    )
    rows: list[KeywordScores] = []
    for keyword in sorted(candidates):
        if frequencies.get(keyword, 0) < threshold:
            continue
        df = document_frequencies.get(keyword, 0)
        if df < effective_min_df or df > effective_max_df:
            continue
        tfidf = score_maps[METHOD_TFIDF].get(keyword)
        yake = score_maps[METHOD_YAKE].get(keyword)
        rake = score_maps[METHOD_RAKE].get(keyword)
        # PART 6: never keep a candidate that no method could score.
        if tfidf is None and yake is None and rake is None:
            continue
        rows.append(
            KeywordScores(
                keyword=keyword,
                tfidf=tfidf,
                yake=yake,
                rake=rake,
                rake_raw=rake_raw_scores.get(keyword),
                frequency=frequencies.get(keyword, 0),
            )
        )

    rows = _sort_results(rows, METHOD_TFIDF)
    return tuple(rows[: max(1, int(top_n))]), tuple(dict.fromkeys(warnings))


def top_keywords_by_method(
    rows: Sequence[KeywordScores],
    method: str,
    top_n: int,
) -> tuple[str, ...]:
    scored = [row for row in rows if row.score_for(method) is not None]
    if method == METHOD_YAKE:
        scored.sort(key=lambda row: (_required_score(row, method), row.keyword))
    else:
        scored.sort(key=lambda row: (-_required_score(row, method), row.keyword))
    return tuple(row.keyword for row in scored[: max(0, int(top_n))])


def documents_from_keyword_payload(value: object) -> tuple[CorpusDocument, ...] | None:
    documents = corpus_documents_from_payload(value)
    if documents is not None:
        return documents
    if isinstance(value, DatasetHandle):
        return _documents_from_dataset(value)
    if isinstance(value, dict):
        document = _document_from_mapping(value, 1)
        return (document,) if document is not None else None
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        return None
    converted: list[CorpusDocument] = []
    for index, item in enumerate(value, start=1):
        document = _coerce_document(item, index)
        if document is None:
            return None
        converted.append(document)
    return tuple(converted)


def keyword_input_source_label(documents: Sequence[CorpusDocument]) -> str:
    return _build_keyword_input(documents).source_label


def _build_keyword_input(documents: Sequence[CorpusDocument]) -> KeywordInput:
    """Resolve the cleanest available text views for keyword scoring.

    Two complementary views are produced per document:

    * ``token_documents`` -- cleaned, stopword-free processed tokens. TF-IDF and
      every corpus-level statistic (frequency, document frequency) use this view.
    * ``light_texts`` -- the most natural text we can recover, with stopwords and
      word order intact. RAKE needs stopwords as phrase delimiters, and YAKE needs
      readable text for its statistical features, so both consume this view.

    Input priority is bag-of-words vocabulary, then preprocessed tokens/text, then
    raw content as a fallback. A ``raw_text`` attribute (set upstream by Preprocess
    Text) is preferred for the light view whenever present.
    """
    usable_documents: list[CorpusDocument] = []
    token_documents: list[tuple[str, ...]] = []
    light_texts: list[str] = []
    vocabulary: set[str] = set()
    source_label = "raw content fallback"
    saw_bow = False
    saw_tokens = False
    saw_natural = False
    all_texts_look_processed = True

    for document in documents:
        attributes = dict(document.attributes)
        raw_text_attr = _first_mapping_text(attributes, ("raw_text", "original_text", "source_text"))
        used_explicit_tokens = False
        bow_terms = _bow_terms_from_attributes(attributes)
        if bow_terms:
            tokens = _tokens_from_bow_terms(bow_terms)
            vocabulary.update(bow_terms)
            saw_bow = True
            used_explicit_tokens = True
        else:
            attribute_tokens = _tokens_from_attributes(attributes)
            if attribute_tokens:
                tokens = attribute_tokens
                saw_tokens = True
                used_explicit_tokens = True
            else:
                raw_text = _document_text(document)
                if not raw_text:
                    continue
                all_texts_look_processed = all_texts_look_processed and _looks_preprocessed(raw_text)
                tokens = tuple(clean_keyword_candidate(token) for token in tokenize_for_bow(raw_text))
        cleaned_tokens = tuple(token for token in tokens if token)
        if not cleaned_tokens:
            continue

        # Decide the light (natural) text view used by RAKE and YAKE.
        if raw_text_attr:
            light = _light_clean_text(raw_text_attr)
            saw_natural = True
        elif not used_explicit_tokens:
            # Raw-content fallback: ``document.text`` is the genuine natural text.
            natural = _document_text(document)
            if natural and not _looks_preprocessed(natural):
                light = _light_clean_text(natural)
                saw_natural = True
            else:
                light = " ".join(cleaned_tokens)
        else:
            # Explicit processed tokens/BoW counts without recoverable raw text.
            # Reconstruct from tokens so RAKE/YAKE never reintroduce raw noise.
            light = " ".join(cleaned_tokens)

        usable_documents.append(document)
        token_documents.append(cleaned_tokens)
        light_texts.append(light)

    if saw_bow:
        source_label = "bag-of-words vocabulary"
    elif saw_tokens or (usable_documents and all_texts_look_processed):
        source_label = "preprocessed tokens"

    cleaned_vocabulary = tuple(sorted(clean_keyword_candidate(term) for term in vocabulary))
    cleaned_vocabulary = tuple(dict.fromkeys(term for term in cleaned_vocabulary if term))
    texts = tuple(" ".join(tokens) for tokens in token_documents)
    return KeywordInput(
        tuple(usable_documents),
        tuple(token_documents),
        texts,
        source_label,
        cleaned_vocabulary,
        tuple(light_texts),
        saw_natural,
    )


def _light_clean_text(text: str) -> str:
    """Lowercase and collapse whitespace while keeping stopwords and word order.

    This is intentionally gentle: RAKE relies on stopwords/punctuation as phrase
    delimiters and YAKE relies on natural word sequences, so we must NOT strip
    stopwords here (unlike the TF-IDF token view).
    """
    lowered = str(text or "").lower()
    # Strip HTML-ish artifacts and URLs but preserve sentence punctuation.
    lowered = re.sub(r"https?://\S+", " ", lowered)
    lowered = re.sub(r"<[^>]+>", " ", lowered)
    lowered = re.sub(r"[ \t\r\f\v]+", " ", lowered)
    return lowered.strip()


def _score_tfidf(
    keyword_input: KeywordInput,
    min_frequency: int,
    min_document_frequency: int,
    max_document_frequency: int,
    aggregation: str,
) -> tuple[dict[str, float], str]:
    """Score terms with per-document l2-normalised TF-IDF, ranked corpus-wide.

    Ranking uses the MEAN over *all* documents (sum of a term's TF-IDF across the
    corpus divided by the document count) rather than the mean over only the
    documents where the term occurs. Averaging over non-zero documents over-promotes
    one-off proper nouns that appear in a single document with a high local weight
    (e.g. ``scottish``/``kenteris``); dividing by the full document count instead
    rewards terms that are characteristic across multiple documents.

    Document-frequency filtering (min_df/max_df) is applied by the caller against
    the shared token statistics, so the vectoriser itself keeps every term. This
    matters because sklearn ignores ``min_df``/``max_df`` when an explicit
    ``vocabulary`` (from Bag of Words) is supplied.
    """
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
    except Exception as exc:
        return {}, f"TF-IDF unavailable: {exc}"

    texts = [" ".join(tokens) for tokens in keyword_input.token_documents]
    if not any(text.strip() for text in texts):
        return {}, ""
    vocabulary = tuple(
        keyword
        for keyword in keyword_input.vocabulary
        if keyword and " " not in keyword
    ) or None
    try:
        vectorizer = TfidfVectorizer(
            lowercase=False,
            analyzer="word",
            token_pattern=r"(?u)\b\w+\b",
            ngram_range=(1, 1),
            min_df=1,
            max_df=1.0,
            norm="l2",
            use_idf=True,
            smooth_idf=True,
            sublinear_tf=False,
            vocabulary=vocabulary,
        )
        matrix = vectorizer.fit_transform(texts)
    except ValueError as exc:
        return {}, f"TF-IDF produced no keyword candidates: {exc}"

    document_count = max(1, matrix.shape[0])
    terms = vectorizer.get_feature_names_out()
    frequencies = _candidate_frequencies(keyword_input.token_documents, terms)
    nonzero_values: dict[str, list[float]] = defaultdict(list)
    term_sums: dict[str, float] = defaultdict(float)
    coo = matrix.tocoo()
    for column, value in zip(coo.col, coo.data, strict=False):
        keyword = clean_keyword_candidate(terms[column])
        if keyword and frequencies.get(keyword, 0) >= min_frequency:
            nonzero_values[keyword].append(float(value))
            term_sums[keyword] += float(value)

    scores: dict[str, float] = {}
    for keyword, values in nonzero_values.items():
        if not values:
            continue
        if aggregation == "Max":
            scores[keyword] = max(values)
        elif aggregation == "Median":
            scores[keyword] = float(statistics.median(values))
        elif aggregation == "Sum":
            scores[keyword] = term_sums[keyword]
        else:  # "Mean" -> characteristic-term mean over the whole corpus
            scores[keyword] = term_sums[keyword] / document_count
    return scores, ""


def _score_yake(
    keyword_input: KeywordInput,
    aggregation: str,
    *,
    ngram: int = 1,
    top_per_document: int = 30,
    dedup_limit: float = 0.9,
) -> tuple[dict[str, float], str]:
    """Score keywords with YAKE on the natural (light) text view.

    YAKE is unsupervised and document-local: it relies on statistical features of
    readable text, so it is run on ``light_texts`` (stopwords/word order intact)
    rather than the stopword-stripped TF-IDF token view. Lower YAKE scores indicate
    stronger candidates. If the optional ``yake`` package is missing we surface a
    single, explicit message instead of silently returning NA for every row.
    """
    try:
        import yake
    except Exception:
        return {}, "YAKE package is not installed. Install with: pip install yake"

    # Prefer the natural text view; fall back to processed text when unavailable.
    texts = keyword_input.light_texts or keyword_input.texts
    if not any(text.strip() for text in texts):
        return {}, ""

    doc_scores: dict[str, list[float]] = defaultdict(list)
    try:
        extractor = yake.KeywordExtractor(
            lan="en",
            n=max(1, int(ngram)),
            dedupLim=float(dedup_limit),
            top=max(1, int(top_per_document)),
        )
        for text in texts:
            if not text.strip():
                continue
            for keyword, score in extractor.extract_keywords(text):
                normalized = clean_keyword_candidate(keyword)
                if normalized and math.isfinite(float(score)):
                    doc_scores[normalized].append(float(score))
    except Exception as exc:
        return {}, f"YAKE scoring failed: {exc}"
    if not doc_scores:
        return {}, "YAKE produced no keyword candidates for this corpus."
    return _aggregate_scores(doc_scores, aggregation), ""


def _score_rake(
    keyword_input: KeywordInput,
    aggregation: str,
) -> tuple[dict[str, float], dict[str, float]]:
    """Score keywords/keyphrases with RAKE on the natural (light) text view.

    IMPORTANT: RAKE is run on ``light_texts`` -- lightly cleaned raw text with
    stopwords and punctuation PRESERVED -- not on the stopword-stripped TF-IDF
    token view. RAKE generates candidate phrases by splitting on stopword and
    punctuation delimiters, so feeding it stopword-free text leaves no delimiters:
    every word degenerates into an identical single-word phrase with the same
    degree/frequency ratio, which previously collapsed every RAKE score to 1.0000
    after normalisation. Keeping stopwords intact restores real phrase boundaries
    and produces varied scores. This is deliberately a different text view from the
    one TF-IDF uses.

    The aggregated RAKE value is returned RAW (higher is better). We intentionally
    do not min-max normalise it, both because normalisation is what collapsed the
    scores before and because RAKE lives on its own scale that is not comparable to
    TF-IDF or YAKE.
    """
    texts = keyword_input.light_texts or keyword_input.texts
    doc_scores: dict[str, list[float]] = defaultdict(list)
    for text in texts:
        for keyword, score in _rake_document_scores(text).items():
            doc_scores[keyword].append(score)
    raw_scores = _aggregate_scores(doc_scores, aggregation)
    # Display the raw RAKE score directly; do not collapse via normalisation.
    return dict(raw_scores), raw_scores


def _rake_phrases_from_text(text: str) -> list[list[str]]:
    """Split natural text into RAKE candidate phrases using stopword/punctuation
    delimiters, exactly as the canonical RAKE algorithm prescribes."""
    phrases: list[list[str]] = []
    current: list[str] = []
    # Emit word tokens and punctuation marks separately so punctuation acts as a
    # delimiter (RAKE breaks candidate phrases on both stopwords and punctuation).
    for token in re.findall(r"\w+|[^\w\s]", str(text).lower(), flags=re.UNICODE):
        if not token[0].isalnum():
            if current:
                phrases.append(current)
                current = []
            continue
        if token.isdigit():
            if current:
                phrases.append(current)
                current = []
            continue
        if token in DEFAULT_KEYWORD_STOPWORDS:
            if current:
                phrases.append(current)
                current = []
            continue
        current.append(token)
    if current:
        phrases.append(current)

    bounded_phrases: list[list[str]] = []
    for phrase in phrases:
        if len(phrase) <= MAX_RAKE_PHRASE_LENGTH:
            bounded_phrases.append(phrase)
        else:
            # Slide a window so over-long phrases still contribute bounded n-grams.
            for index in range(len(phrase) - MAX_RAKE_PHRASE_LENGTH + 1):
                bounded_phrases.append(phrase[index : index + MAX_RAKE_PHRASE_LENGTH])
    return bounded_phrases


def _rake_document_scores(text: str) -> dict[str, float]:
    phrases = _rake_phrases_from_text(text)
    if not phrases:
        return {}

    frequency: Counter[str] = Counter()
    degree: Counter[str] = Counter()
    for phrase in phrases:
        phrase_degree = len(phrase) - 1  # number of co-occurring words in the phrase
        for word in phrase:
            frequency[word] += 1
            degree[word] += phrase_degree
    # Canonical RAKE: degree(word) = co-occurrence degree + frequency(word).
    for word, count in frequency.items():
        degree[word] += count

    word_scores = {
        word: degree[word] / count
        for word, count in frequency.items()
        if count > 0
    }
    scores: dict[str, float] = {}
    for phrase in phrases:
        if not phrase:
            continue
        phrase_text = clean_keyword_candidate(" ".join(phrase))
        if phrase_text:
            phrase_score = sum(word_scores[word] for word in phrase)
            scores[phrase_text] = max(scores.get(phrase_text, 0.0), phrase_score)
        for word in phrase:
            normalized = clean_keyword_candidate(word)
            if normalized:
                scores[normalized] = max(scores.get(normalized, 0.0), word_scores[word])
    return scores


def _aggregate_scores(scores: dict[str, list[float]], aggregation: str) -> dict[str, float]:
    aggregated: dict[str, float] = {}
    for keyword, values in scores.items():
        numeric_values = [value for value in values if math.isfinite(value)]
        if not numeric_values:
            continue
        if aggregation == "Max":
            aggregated[keyword] = max(numeric_values)
        elif aggregation == "Median":
            aggregated[keyword] = float(statistics.median(numeric_values))
        elif aggregation == "Sum":
            aggregated[keyword] = sum(numeric_values)
        else:
            aggregated[keyword] = sum(numeric_values) / len(numeric_values)
    return aggregated


def _sort_results(rows: Sequence[KeywordScores], method: str) -> list[KeywordScores]:
    if method == METHOD_YAKE:
        return sorted(
            rows,
            key=lambda row: (
                row.score_for(method) if row.score_for(method) is not None else math.inf,
                row.keyword,
            ),
        )
    return sorted(
        rows,
        key=lambda row: (
            -(row.score_for(method) if row.score_for(method) is not None else -math.inf),
            row.keyword,
        ),
    )


def _light_word_lists(light_texts: Sequence[str]) -> list[list[str]]:
    return [re.findall(r"\w+", str(text).lower(), flags=re.UNICODE) for text in light_texts]


def _candidate_frequencies(
    token_documents: Sequence[Sequence[str]],
    candidates: Iterable[str],
    light_texts: Sequence[str] = (),
) -> dict[str, int]:
    """Corpus frequency of each candidate across the token view AND the light text.

    YAKE and RAKE discover candidates in the natural (light) text that may not be
    present in the stopword-stripped token view; counting evidence from both views
    lets those method-specific candidates survive frequency/df filtering (PART 6).
    """
    candidate_set = {clean_keyword_candidate(candidate) for candidate in candidates}
    candidate_set.discard("")
    frequencies: dict[str, int] = {}
    token_counters = [Counter(tokens) for tokens in token_documents]
    light_counters = [Counter(words) for words in _light_word_lists(light_texts)]
    token_texts = [f" {' '.join(tokens)} " for tokens in token_documents]
    light_norm = [f" {' '.join(words)} " for words in _light_word_lists(light_texts)]
    n_docs = max(len(token_documents), len(light_counters))
    for candidate in candidate_set:
        if " " in candidate:
            needle = f" {candidate} "
            total = 0
            for index in range(n_docs):
                t = token_texts[index] if index < len(token_texts) else ""
                l = light_norm[index] if index < len(light_norm) else ""
                total += t.count(needle) + l.count(needle)
            frequencies[candidate] = total
        else:
            total = 0
            for index in range(n_docs):
                tc = token_counters[index].get(candidate, 0) if index < len(token_counters) else 0
                lc = light_counters[index].get(candidate, 0) if index < len(light_counters) else 0
                total += max(tc, lc)
            frequencies[candidate] = total
    return frequencies


def _candidate_document_frequencies(
    token_documents: Sequence[Sequence[str]],
    candidates: Iterable[str],
    light_texts: Sequence[str] = (),
) -> dict[str, int]:
    candidate_set = {clean_keyword_candidate(candidate) for candidate in candidates}
    candidate_set.discard("")
    document_frequencies: dict[str, int] = {}
    token_texts = [f" {' '.join(tokens)} " for tokens in token_documents]
    token_sets = [set(tokens) for tokens in token_documents]
    light_word_lists = _light_word_lists(light_texts)
    light_sets = [set(words) for words in light_word_lists]
    light_norm = [f" {' '.join(words)} " for words in light_word_lists]
    n_docs = max(len(token_documents), len(light_sets))
    for candidate in candidate_set:
        count = 0
        if " " in candidate:
            needle = f" {candidate} "
            for index in range(n_docs):
                t = token_texts[index] if index < len(token_texts) else ""
                l = light_norm[index] if index < len(light_norm) else ""
                if needle in t or needle in l:
                    count += 1
        else:
            for index in range(n_docs):
                in_tokens = candidate in token_sets[index] if index < len(token_sets) else False
                in_light = candidate in light_sets[index] if index < len(light_sets) else False
                if in_tokens or in_light:
                    count += 1
        document_frequencies[candidate] = count
    return document_frequencies


def clean_keyword_candidate(keyword: object, custom_stopwords: Iterable[str] = ()) -> str:
    text = re.sub(r"\s+", " ", str(keyword).replace("_", " ").strip().lower())
    text = text.strip(".,;:!?()[]{}\"'")
    if not text or not re.search(r"\w", text, flags=re.UNICODE):
        return ""
    if all(not character.isalnum() for character in text):
        return ""
    if any(character.isdigit() for character in text):
        return ""
    stopwords = DEFAULT_KEYWORD_STOPWORDS | {word.lower() for word in custom_stopwords}
    words = re.findall(r"\b\w+\b", text, flags=re.UNICODE)
    if not words:
        return ""
    if any(len(word) < MIN_KEYWORD_LENGTH for word in words):
        return ""
    if all(word in stopwords for word in words):
        return ""
    if any(word in NEWS_NOISE_WORDS for word in words):
        return ""
    cleaned = " ".join(words)
    return cleaned


def _normalize_keyword(keyword: object) -> str:
    text = clean_keyword_candidate(keyword)
    return text


def _effective_min_document_frequency(document_count: int, requested_min_df: int) -> int:
    if requested_min_df > 0:
        return min(max(1, requested_min_df), max(1, document_count))
    if document_count >= AUTO_MIN_DF_THRESHOLD:
        return 2
    return 1


def _effective_max_document_frequency(
    document_count: int,
    max_df_ratio: float,
    min_document_frequency: int,
) -> int:
    if document_count <= 0:
        return 0
    if document_count < 10:
        return document_count
    ratio = min(1.0, max(0.05, float(max_df_ratio)))
    return max(min_document_frequency, min(document_count, math.floor(document_count * ratio)))


def _document_text(document: CorpusDocument) -> str:
    text = str(document.text or "").strip()
    if text:
        return text
    attributes = dict(document.attributes)
    return _first_mapping_text(attributes, ("text", "content", "body", "document", "tokens")) or ""


def _bow_terms_from_attributes(attributes: dict[str, object]) -> dict[str, float]:
    terms: dict[str, float] = {}
    for key, value in attributes.items():
        if not str(key).startswith("bow_") or key == "bow_total":
            continue
        keyword = clean_keyword_candidate(str(key)[4:])
        if not keyword:
            continue
        try:
            numeric_value = float(value)
        except (TypeError, ValueError):
            continue
        if numeric_value > 0:
            terms[keyword] = numeric_value
    return terms


def _tokens_from_bow_terms(terms: dict[str, float]) -> tuple[str, ...]:
    tokens: list[str] = []
    for term, value in sorted(terms.items()):
        if abs(value - round(value)) < 1e-9 and 0 < value <= 100:
            repeat = int(round(value))
        else:
            repeat = 1
        term_tokens = term.split()
        for _index in range(repeat):
            tokens.extend(term_tokens)
    return tuple(tokens)


def _tokens_from_attributes(attributes: dict[str, object]) -> tuple[str, ...]:
    text = _first_mapping_text(attributes, ("processed_tokens", "tokens", "preprocessed_tokens"))
    if not text:
        text = _first_mapping_text(attributes, ("processed_text", "preprocessed_text", "content", "text"))
    if not text:
        return ()
    return tuple(clean_keyword_candidate(token) for token in tokenize_for_bow(text) if clean_keyword_candidate(token))


def _looks_preprocessed(text: str) -> bool:
    if not text.strip():
        return False
    return text == text.lower() and not re.search(r"[^\w\s]", text, flags=re.UNICODE)


def _coerce_document(item: object, index: int) -> CorpusDocument | None:
    if isinstance(item, CorpusDocument):
        return item
    if isinstance(item, dict):
        return _document_from_mapping(item, index)
    text = _first_text_value(item, ("text", "content", "body", "document", "tokens"))
    if text is None:
        return None
    title = _first_text_value(item, ("title", "name", "category", "source")) or f"Document {index}"
    source = _first_text_value(item, ("source", "category")) or "Input"
    return CorpusDocument(str(title), text, str(source))


def _document_from_mapping(item: dict[Any, Any], index: int) -> CorpusDocument | None:
    text = _first_mapping_text(item, ("text", "content", "body", "document", "tokens"))
    if text is None:
        return None
    title = _first_mapping_text(item, ("title", "name", "category", "source")) or f"Document {index}"
    source = _first_mapping_text(item, ("source", "category")) or "Input"
    return CorpusDocument(str(title), text, str(source))


def _documents_from_dataset(dataset: DatasetHandle) -> tuple[CorpusDocument, ...] | None:
    dataframe = dataset.dataframe
    columns = list(dataframe.columns)
    text_column = _first_existing_column(columns, ("text", "content", "body", "document", "tokens"))
    if text_column is None:
        return None
    title_column = _first_existing_column(columns, ("title", "name", "category"))
    source_column = _first_existing_column(columns, ("source", "category"))
    documents: list[CorpusDocument] = []
    for index, row in enumerate(dataframe.iter_rows(named=True), start=1):
        text = _stringify_text_value(row.get(text_column))
        if not text:
            continue
        title = _stringify_text_value(row.get(title_column)) if title_column else ""
        source = _stringify_text_value(row.get(source_column)) if source_column else ""
        documents.append(CorpusDocument(title or f"Document {index}", text, source or dataset.display_name))
    return tuple(documents)


def _first_existing_column(columns: Sequence[str], names: Sequence[str]) -> str | None:
    by_lower = {column.lower(): column for column in columns}
    for name in names:
        if name.lower() in by_lower:
            return by_lower[name.lower()]
    return None


def _first_text_value(item: object, names: Sequence[str]) -> str | None:
    for name in names:
        if hasattr(item, name):
            value = getattr(item, name)
            text = _stringify_text_value(value)
            if text:
                return text
    return None


def _first_mapping_text(item: dict[Any, Any], names: Sequence[str]) -> str | None:
    by_lower = {str(key).lower(): value for key, value in item.items()}
    for name in names:
        text = _stringify_text_value(by_lower.get(name.lower()))
        if text:
            return text
    return None


def _stringify_text_value(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, (list, tuple, set, frozenset)):
        return " ".join(str(token) for token in value if str(token).strip())
    return str(value).strip()


class ExtractKeywordsScreen(QWidget, WorkflowNodeScreenSupport):
    def __init__(
        self,
        parent: QWidget | None = None,
        documents: Sequence[CorpusDocument] | None = None,
    ) -> None:
        super().__init__(parent)
        self._init_workflow_node_support()
        self._documents = tuple(() if documents is None else documents)
        self._using_input_corpus = documents is not None
        self._keywords: tuple[KeywordItem, ...] = ()
        self._results: tuple[KeywordScores, ...] = ()
        self._visible_results: tuple[KeywordScores, ...] = ()
        self._warnings: tuple[str, ...] = ()
        self._input_source_label = "raw content fallback"

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(10)

        layout.addWidget(
            SectionHeader(
                "Extract Keywords",
                "Score corpus keyword candidates with TF-IDF, YAKE, and RAKE.",
            )
        )
        layout.addWidget(self._build_options_panel())
        layout.addWidget(self._build_metadata_panel())
        layout.addWidget(self._build_table_panel(), 1)

        self.apply_options()

    def sizeHint(self) -> QSize:
        return QSize(1040, 700)

    def minimumSizeHint(self) -> QSize:
        return QSize(780, 540)

    def _build_options_panel(self) -> QFrame:
        frame = QFrame(self)
        frame.setProperty("panel", True)
        layout = QGridLayout(frame)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setHorizontalSpacing(12)
        layout.setVerticalSpacing(8)

        layout.addWidget(QLabel("Max keywords", self), 0, 0)
        self._top_n_spinbox = QSpinBox(self)
        self._top_n_spinbox.setRange(1, 500)
        self._top_n_spinbox.setValue(50)
        apply_readable_spin_box_style(self._top_n_spinbox)
        layout.addWidget(self._top_n_spinbox, 0, 1)

        min_frequency_label = QLabel("Minimum frequency", self)
        min_frequency_label.setToolTip(
            "Minimum number of times a candidate must appear in the corpus (token/candidate frequency, not document frequency)."
        )
        layout.addWidget(min_frequency_label, 0, 2)
        self._min_frequency_spinbox = QSpinBox(self)
        self._min_frequency_spinbox.setRange(1, 999)
        self._min_frequency_spinbox.setValue(1)
        self._min_frequency_spinbox.setToolTip(
            "Minimum token/candidate frequency across the whole corpus."
        )
        apply_readable_spin_box_style(self._min_frequency_spinbox)
        layout.addWidget(self._min_frequency_spinbox, 0, 3)

        layout.addWidget(QLabel("Aggregation", self), 0, 4)
        self._aggregation_combo = QComboBox(self)
        self._aggregation_combo.addItems(AGGREGATIONS)
        self._aggregation_combo.setToolTip("YAKE scores are inverse: lower is better. Max is a numeric maximum, not the best YAKE score.")
        layout.addWidget(self._aggregation_combo, 0, 5)

        layout.addWidget(QLabel("Primary method", self), 1, 0)
        self._primary_method_combo = QComboBox(self)
        self._primary_method_combo.addItems(SCORE_METHODS)
        self._primary_method_combo.setCurrentText(METHOD_TFIDF)
        self._primary_method_combo.currentTextChanged.connect(lambda _text: self._apply_selection_mode())
        layout.addWidget(self._primary_method_combo, 1, 1)

        layout.addWidget(QLabel("Select words", self), 1, 2)
        self._selection_combo = QComboBox(self)
        self._selection_combo.addItems(SELECTION_MODES)
        self._selection_combo.setCurrentText(SELECTION_ALL)
        self._selection_combo.currentTextChanged.connect(lambda _text: self._apply_selection_mode())
        layout.addWidget(self._selection_combo, 1, 3)

        layout.addWidget(QLabel("Top N", self), 1, 4)
        self._selection_count_spinbox = QSpinBox(self)
        self._selection_count_spinbox.setRange(1, 500)
        self._selection_count_spinbox.setValue(20)
        self._selection_count_spinbox.valueChanged.connect(lambda _value: self._apply_selection_mode())
        apply_readable_spin_box_style(self._selection_count_spinbox)
        layout.addWidget(self._selection_count_spinbox, 1, 5)

        layout.addWidget(QLabel("Search", self), 2, 0)
        self._search_input = QLineEdit(self)
        self._search_input.setPlaceholderText("Filter keywords...")
        self._search_input.textChanged.connect(lambda _text: self._render())
        layout.addWidget(self._search_input, 2, 1, 1, 3)

        DF_TOOLTIP = "Filters out candidates that appear in too few or too many documents."
        min_df_label = QLabel("Minimum document frequency", self)
        min_df_label.setToolTip(DF_TOOLTIP)
        layout.addWidget(min_df_label, 3, 0)
        self._min_document_frequency_spinbox = QSpinBox(self)
        self._min_document_frequency_spinbox.setRange(0, 999)
        self._min_document_frequency_spinbox.setValue(0)
        self._min_document_frequency_spinbox.setToolTip(
            "0 = Auto. Uses 2 for corpora with 20 or more documents, otherwise 1. "
            + DF_TOOLTIP
        )
        apply_readable_spin_box_style(self._min_document_frequency_spinbox)
        layout.addWidget(self._min_document_frequency_spinbox, 3, 1)

        max_df_label = QLabel("Maximum document frequency %", self)
        max_df_label.setToolTip(DF_TOOLTIP)
        layout.addWidget(max_df_label, 3, 2)
        self._max_document_frequency_spinbox = QSpinBox(self)
        self._max_document_frequency_spinbox.setRange(5, 100)
        self._max_document_frequency_spinbox.setValue(90)
        self._max_document_frequency_spinbox.setSuffix("%")
        self._max_document_frequency_spinbox.setToolTip(DF_TOOLTIP)
        apply_readable_spin_box_style(self._max_document_frequency_spinbox)
        layout.addWidget(self._max_document_frequency_spinbox, 3, 3)

        self._apply_button = QPushButton("Extract", self)
        self._apply_button.setProperty("primary", True)
        self._apply_button.clicked.connect(self.apply_options)
        layout.addWidget(self._apply_button, 2, 5)
        return frame

    def _build_metadata_panel(self) -> QFrame:
        frame = QFrame(self)
        frame.setProperty("panel", True)
        layout = QHBoxLayout(frame)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        self._document_count_label = self._build_metric_label("Documents", "0")
        self._keyword_count_label = self._build_metric_label("Keywords", "0")
        self._selected_count_label = self._build_metric_label("Selected", "0")
        self._top_keyword_label = self._build_metric_label("Top Keyword", "-")

        layout.addWidget(self._document_count_label, 1)
        layout.addWidget(self._keyword_count_label, 1)
        layout.addWidget(self._selected_count_label, 1)
        layout.addWidget(self._top_keyword_label, 1)
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

        self._help_label = QLabel(
            "TF-IDF and RAKE: higher is better. YAKE: lower is better. RAKE uses its own "
            "raw scale, so its values are not directly comparable to TF-IDF or YAKE. "
            "NA means a method did not score the keyword or is unavailable.",
            self,
        )
        self._help_label.setProperty("muted", True)
        self._help_label.setWordWrap(True)
        layout.addWidget(self._help_label)

        self._status_label = QLabel("", self)
        self._status_label.setProperty("muted", True)
        self._status_label.setWordWrap(True)
        layout.addWidget(self._status_label)

        self._table = QTableWidget(0, 4, self)
        self._table.setHorizontalHeaderLabels(["Keyword", METHOD_TFIDF, METHOD_YAKE, METHOD_RAKE])
        self._table.verticalHeader().setVisible(False)
        self._table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self._table.setSelectionMode(QTableWidget.SelectionMode.ExtendedSelection)
        self._table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._table.setSortingEnabled(True)
        header = self._table.horizontalHeader()
        header.setStretchLastSection(False)
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        for column in range(1, 4):
            header.setSectionResizeMode(column, QHeaderView.ResizeMode.Fixed)
            self._table.setColumnWidth(column, 112)
        self._table.itemSelectionChanged.connect(self._selection_changed)
        layout.addWidget(self._table, 1)
        return frame

    def set_input_payload(self, payload: WorkflowPayload | None) -> None:
        if payload is None:
            self._documents = ()
            self._using_input_corpus = False
            self.apply_options()
            return

        documents = documents_from_keyword_payload(payload.value)
        self._documents = () if documents is None else documents
        self._using_input_corpus = True
        print(f"Extract Keywords received corpus: {len(self._documents)} documents")
        self.apply_options()

    def current_output_payload(self) -> WorkflowPayload:
        return WorkflowPayload("Words", self.selected_keywords())

    def selected_keywords(self) -> tuple[str, ...]:
        mode = self._selection_combo.currentText()
        if mode == SELECTION_NONE:
            return ()
        if mode == SELECTION_TOP_N:
            return top_keywords_by_method(
                self._visible_results,
                self._primary_method_combo.currentText(),
                self._selection_count_spinbox.value(),
            )
        if mode == SELECTION_ALL:
            return tuple(item.keyword for item in self._visible_results)
        selected: list[str] = []
        for index in self._table.selectionModel().selectedRows():
            item = self._table.item(index.row(), 0)
            if item is not None and item.text():
                selected.append(item.text())
        return tuple(dict.fromkeys(selected))

    def apply_options(self) -> tuple[KeywordItem, ...]:
        self._input_source_label = keyword_input_source_label(self._documents)
        self._results, self._warnings = extract_keyword_scores(
            self._documents,
            top_n=self._top_n_spinbox.value(),
            min_frequency=self._min_frequency_spinbox.value(),
            aggregation=self._aggregation_combo.currentText(),
            min_document_frequency=self._min_document_frequency_spinbox.value(),
            max_document_frequency_ratio=self._max_document_frequency_spinbox.value() / 100,
        )
        self._keywords = tuple(
            KeywordItem(
                "Corpus",
                item.keyword,
                _first_available_score(item),
                item.frequency,
                tfidf=item.tfidf,
                yake=item.yake,
                rake=item.rake,
            )
            for item in self._results
        )
        self._render()
        self._notify_output_changed()
        return self._keywords

    def _render(self) -> None:
        query = self._search_input.text().strip().lower()
        self._visible_results = tuple(
            item for item in self._results if not query or query in item.keyword.lower()
        )
        top_keyword = self._visible_results[0].keyword if self._visible_results else "-"
        self._document_count_label.setText(f"Documents\n{len(self._documents)}")
        self._keyword_count_label.setText(f"Keywords\n{len(self._visible_results)} / {len(self._results)}")
        self._top_keyword_label.setText(f"Top Keyword\n{top_keyword}")

        if self._visible_results and self._using_input_corpus:
            status = "Input corpus is connected and keyword scores are ready."
        elif self._using_input_corpus and self._documents:
            status = "Input corpus is connected but no usable keyword candidates match the current settings."
        elif self._using_input_corpus:
            status = "Input is connected but no usable text was found."
        else:
            status = "Connect a Corpus input to extract keywords."
        if self._warnings:
            status = f"{status} {' '.join(self._warnings)}"
        if self._using_input_corpus:
            status = f"{status} Using input: {self._input_source_label}."
        self._status_label.setText(status)

        self._table.setSortingEnabled(False)
        self._table.clearContents()
        self._table.setRowCount(len(self._visible_results))
        for row, item in enumerate(self._visible_results):
            self._set_text_item(row, 0, item.keyword)
            self._set_score_item(row, 1, item.tfidf, decimals=4)
            self._set_score_item(row, 2, item.yake, decimals=6)
            self._set_score_item(row, 3, item.rake, decimals=4)
        self._table.setSortingEnabled(True)
        self._apply_table_column_sizes()
        self._apply_selection_mode()
        self._update_selected_count()

    def _set_text_item(self, row: int, column: int, text: str) -> None:
        self._table.setItem(row, column, QTableWidgetItem(text))

    def _set_score_item(self, row: int, column: int, value: float | None, *, decimals: int) -> None:
        text = "NA" if value is None else f"{value:.{decimals}f}"
        item = NumericTableItem(text, value)
        item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self._table.setItem(row, column, item)

    def _apply_table_column_sizes(self) -> None:
        for column in range(1, 4):
            self._table.setColumnWidth(column, 112)

    def _apply_selection_mode(self) -> None:
        if not hasattr(self, "_table"):
            return
        mode = self._selection_combo.currentText()
        self._table.blockSignals(True)
        self._table.clearSelection()
        if mode == SELECTION_ALL:
            self._table.selectAll()
        elif mode == SELECTION_TOP_N:
            selected_keywords = set(
                top_keywords_by_method(
                    self._visible_results,
                    self._primary_method_combo.currentText(),
                    self._selection_count_spinbox.value(),
                )
            )
            for row in range(self._table.rowCount()):
                item = self._table.item(row, 0)
                if item is not None and item.text() in selected_keywords:
                    self._table.selectRow(row)
        self._table.blockSignals(False)
        self._update_selected_count()
        self._notify_output_changed()

    def _selection_changed(self) -> None:
        if self._selection_combo.currentText() == SELECTION_MANUAL:
            self._update_selected_count()
            self._notify_output_changed()

    def _update_selected_count(self) -> None:
        self._selected_count_label.setText(f"Selected\n{len(self.selected_keywords())}")

    def data_preview_snapshot(self) -> dict[str, object]:
        headers = ["Keyword", METHOD_TFIDF, METHOD_YAKE, METHOD_RAKE]
        rows = [
            [
                item.keyword,
                _format_optional_score(item.tfidf, decimals=4),
                _format_optional_score(item.yake, decimals=6),
                _format_optional_score(item.rake, decimals=4),
            ]
            for item in self._visible_results
        ]
        return {
            "summary": f"Extract Keywords: {len(self._visible_results)} keywords from {len(self._documents)} documents",
            "headers": headers,
            "rows": rows,
        }


def _format_optional_score(value: float | None, *, decimals: int = 4) -> str:
    return "NA" if value is None else f"{value:.{decimals}f}"


def _required_score(row: KeywordScores, method: str) -> float:
    value = row.score_for(method)
    return 0.0 if value is None else value


def _first_available_score(row: KeywordScores) -> float:
    for value in (row.tfidf, row.rake, row.yake):
        if value is not None:
            return value
    return 0.0
