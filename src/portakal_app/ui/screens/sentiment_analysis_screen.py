from __future__ import annotations

import math
import os
import pickle
import re
import urllib.error
import urllib.request
from collections.abc import Mapping, Sequence
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
    QVBoxLayout,
    QWidget,
)

from portakal_app.models import WorkflowPayload
from portakal_app.ui.screens.corpus_screen import CorpusDocument, corpus_documents_from_payload
from portakal_app.ui.screens.create_corpus_screen import preview_text
from portakal_app.ui.screens.node_screen import WorkflowNodeScreenSupport
from portakal_app.ui.shared.cards import SectionHeader
from portakal_app.ui.shared.readable_inputs import apply_readable_line_edit_style


METHOD_LIU_HU = "liu_hu"
METHOD_VADER = "vader"
METHOD_CUSTOM = "custom"
METHOD_MULTILINGUAL = "multilingual"
METHOD_SENTIART = "sentiart"
METHOD_LILAH = "lilah"

METHOD_LABELS = {
    METHOD_LIU_HU: "Liu & Hu",
    METHOD_VADER: "VADER",
    METHOD_CUSTOM: "Custom Dictionary",
    METHOD_MULTILINGUAL: "Multilingual Sentiment",
    METHOD_SENTIART: "SentiART",
    METHOD_LILAH: "LiLaH",
}

SENTIMENT_RESOURCE_ENV = "PORTAKAL_SENTIMENT_RESOURCE_DIR"
SENTIART_SERVER_URL = "https://file.biolab.si/files/sentiart/"
LILAH_SERVER_URL = "https://file.biolab.si/files/sentiment-lilah/"
SENTIART_LANGUAGES = {"en": "EN", "de": "DE"}
LILAH_LANGUAGES = {"hr": "HR", "nl": "NL", "sl": "SL"}
SENTIART_SCORE_NAMES = ("sentiment", "anger", "fear", "disgust", "happiness", "sadness", "surprise")
SENTIART_SCORE_ALIASES = {
    "sentiment": ("AAPZ", "AAPz", "aapz", "sentiment"),
    "anger": ("ang_z", "anger"),
    "fear": ("fear_z", "fear"),
    "disgust": ("disg_z", "disgust"),
    "happiness": ("hap_z", "happiness"),
    "sadness": ("sad_z", "sadness"),
    "surprise": ("surp_z", "surprise"),
}
LILAH_SCORE_NAMES = (
    "Positive",
    "Negative",
    "Anger",
    "Anticipation",
    "Disgust",
    "Fear",
    "Joy",
    "Sadness",
    "Surprise",
    "Trust",
)

POSITIVE_WORDS = frozenset(
    {
        "accurate",
        "amazing",
        "awesome",
        "başarılı",
        "beautiful",
        "best",
        "better",
        "delightful",
        "excellent",
        "fantastic",
        "faydalı",
        "good",
        "great",
        "happy",
        "improved",
        "iyi",
        "love",
        "olumlu",
        "perfect",
        "positive",
        "strong",
        "useful",
        "wonderful",
        "yararlı",
    }
)
NEGATIVE_WORDS = frozenset(
    {
        "awful",
        "bad",
        "başarısız",
        "boring",
        "broken",
        "error",
        "failed",
        "hata",
        "horrible",
        "kötü",
        "negative",
        "olumsuz",
        "poor",
        "problem",
        "sad",
        "slow",
        "terrible",
        "weak",
        "worse",
        "worst",
        "wrong",
        "zararlı",
    }
)

TURKISH_POSITIVE_WORDS = POSITIVE_WORDS | frozenset({"güzel", "harika", "mükemmel", "başarılı", "sevdim"})
TURKISH_NEGATIVE_WORDS = NEGATIVE_WORDS | frozenset({"berbat", "korkunç", "başarısız", "sevmedim", "sorunlu"})

NEGATIONS = frozenset({"not", "no", "never", "none", "hardly", "barely", "değil", "yok"})
INTENSIFIERS = {
    "absolutely": 0.35,
    "extremely": 0.35,
    "really": 0.2,
    "so": 0.15,
    "too": 0.15,
    "very": 0.25,
}
VADER_EXTRA_VALENCE = {
    "awesome": 2.8,
    "excellent": 3.0,
    "amazing": 2.8,
    "good": 1.9,
    "great": 2.5,
    "love": 3.0,
    "perfect": 2.7,
    "terrible": -2.8,
    "awful": -2.7,
    "bad": -1.9,
    "boring": -1.5,
    "horrible": -2.8,
    "worst": -3.0,
}


@dataclass(frozen=True)
class SentimentCorpusDocument(CorpusDocument):
    sentiment_features: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class SentimentResult:
    document: str
    positive_score: float
    negative_score: float
    net_score: float
    label: str
    text_preview: str
    columns: Mapping[str, object] = field(default_factory=dict)


def sentiment_documents_from_payload(value: object) -> tuple[CorpusDocument, ...] | None:
    documents = corpus_documents_from_payload(value)
    if documents is not None:
        return documents
    if isinstance(value, Mapping):
        for key in ("documents", "corpus", "items", "rows", "value"):
            if key in value:
                return sentiment_documents_from_payload(value[key])
        return None
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        return None

    converted: list[CorpusDocument] = []
    for index, item in enumerate(value, start=1):
        document = _document_from_unknown(item, index)
        if document is None:
            return None
        converted.append(document)
    return tuple(converted)


def analyze_sentiment(
    documents: Sequence[CorpusDocument],
    *,
    method: str = METHOD_LIU_HU,
    positive_words: frozenset[str] | None = None,
    negative_words: frozenset[str] | None = None,
    language: str = "en",
    semantic_lexicon: Mapping[str, Mapping[str, float]] | None = None,
) -> tuple[SentimentResult, ...]:
    if method == METHOD_VADER:
        return analyze_vader_sentiment(documents)
    if method == METHOD_CUSTOM:
        return analyze_lexicon_sentiment(
            documents,
            positive_words=positive_words or frozenset(),
            negative_words=negative_words or frozenset(),
            score_prefix="custom_sentiment",
        )
    if method == METHOD_MULTILINGUAL:
        positives, negatives = multilingual_lexicons(language)
        return analyze_lexicon_sentiment(
            documents,
            positive_words=positives,
            negative_words=negatives,
            score_prefix="multilingual",
        )
    if method == METHOD_SENTIART:
        return analyze_sentiart_sentiment(documents, lexicon=semantic_lexicon or {})
    if method == METHOD_LILAH:
        return analyze_lilah_sentiment(documents, lexicon=semantic_lexicon or {})
    return analyze_lexicon_sentiment(
        documents,
        positive_words=positive_words or POSITIVE_WORDS,
        negative_words=negative_words or NEGATIVE_WORDS,
        score_prefix="liu_hu",
    )


def analyze_lexicon_sentiment(
    documents: Sequence[CorpusDocument],
    *,
    positive_words: frozenset[str],
    negative_words: frozenset[str],
    score_prefix: str,
) -> tuple[SentimentResult, ...]:
    rows: list[SentimentResult] = []
    for document in documents:
        tokens = sentiment_tokens(document.text)
        positive = sum(1 for token in tokens if token in positive_words)
        negative = sum(1 for token in tokens if token in negative_words)
        score = ((positive - negative) / len(tokens) * 100) if tokens else 0.0
        label = label_from_signed_score(score)
        rows.append(
            SentimentResult(
                document.title,
                float(positive),
                float(negative),
                float(score),
                label.title(),
                preview_text(document.text),
                {
                    f"{score_prefix}_score": round(float(score), 3),
                    f"{score_prefix}_label": label,
                },
            )
        )
    return tuple(rows)


def analyze_vader_sentiment(documents: Sequence[CorpusDocument]) -> tuple[SentimentResult, ...]:
    rows: list[SentimentResult] = []
    for document in documents:
        scores = vader_scores(document.text)
        label = vader_label(scores["compound"])
        rows.append(
            SentimentResult(
                document.title,
                scores["positive"],
                scores["negative"],
                scores["compound"],
                label.title(),
                preview_text(document.text),
                {
                    "vader_positive": round(scores["positive"], 3),
                    "vader_negative": round(scores["negative"], 3),
                    "vader_neutral": round(scores["neutral"], 3),
                    "vader_compound": round(scores["compound"], 3),
                    "vader_label": label,
                },
            )
        )
    return tuple(rows)


def analyze_sentiart_sentiment(
    documents: Sequence[CorpusDocument],
    *,
    lexicon: Mapping[str, Mapping[str, float]],
) -> tuple[SentimentResult, ...]:
    rows: list[SentimentResult] = []
    for document in documents:
        scores = semantic_sentiment_scores(document.text, lexicon, SENTIART_SCORE_NAMES)
        net_score = scores["sentiment"]
        label = label_from_signed_score(net_score)
        columns = {f"sentiart_{name}": round(scores[name], 3) for name in SENTIART_SCORE_NAMES}
        columns["sentiart_label"] = label
        rows.append(
            SentimentResult(
                document.title,
                max(net_score, 0.0),
                abs(min(net_score, 0.0)),
                net_score,
                label.title(),
                preview_text(document.text),
                columns,
            )
        )
    return tuple(rows)


def analyze_lilah_sentiment(
    documents: Sequence[CorpusDocument],
    *,
    lexicon: Mapping[str, Mapping[str, float]],
) -> tuple[SentimentResult, ...]:
    rows: list[SentimentResult] = []
    for document in documents:
        scores = semantic_sentiment_scores(document.text, lexicon, LILAH_SCORE_NAMES)
        positive = scores["Positive"]
        negative = scores["Negative"]
        net_score = positive - negative
        label = label_from_signed_score(net_score)
        columns = {f"lilah_{name.lower()}": round(scores[name], 3) for name in LILAH_SCORE_NAMES}
        columns["lilah_label"] = label
        rows.append(
            SentimentResult(
                document.title,
                positive,
                negative,
                net_score,
                label.title(),
                preview_text(document.text),
                columns,
            )
        )
    return tuple(rows)


def sentiment_output_documents(
    documents: Sequence[CorpusDocument],
    results: Sequence[SentimentResult],
) -> tuple[SentimentCorpusDocument, ...]:
    output: list[SentimentCorpusDocument] = []
    for document, result in zip(documents, results):
        existing = getattr(document, "sentiment_features", {})
        merged = dict(existing if isinstance(existing, Mapping) else {})
        merged.update(result.columns)
        output.append(SentimentCorpusDocument(document.title, document.text, document.source, merged))
    return tuple(output)


def sentiment_tokens(text: str) -> tuple[str, ...]:
    return tuple(token.lower() for token in re.findall(r"[\w']+", text, flags=re.UNICODE))


def vader_scores(text: str) -> dict[str, float]:
    external = _external_vader_scores(text)
    if external is not None:
        return external

    raw_tokens = re.findall(r"[\w']+|[!?]+", text, flags=re.UNICODE)
    word_tokens = [token for token in raw_tokens if re.search(r"\w", token)]
    valences: list[float] = []
    for index, token in enumerate(word_tokens):
        lower = token.lower()
        valence = vader_valence(lower)
        if valence == 0:
            valences.append(0.0)
            continue
        previous = [word_tokens[pos].lower() for pos in range(max(0, index - 3), index)]
        if any(word in NEGATIONS for word in previous):
            valence *= -0.9
        if index > 0:
            valence += math.copysign(INTENSIFIERS.get(word_tokens[index - 1].lower(), 0.0), valence)
        if token.isupper() and len(token) > 1:
            valence *= 1.15
        valences.append(valence)

    punctuation_boost = min(text.count("!"), 4) * 0.08
    summed = sum(valences)
    if summed > 0:
        summed += punctuation_boost
    elif summed < 0:
        summed -= punctuation_boost
    compound = summed / math.sqrt((summed * summed) + 15) if summed else 0.0

    positive = sum(value for value in valences if value > 0)
    negative = abs(sum(value for value in valences if value < 0))
    neutral = sum(1 for value in valences if value == 0)
    denominator = positive + negative + neutral
    if denominator == 0:
        return {"positive": 0.0, "negative": 0.0, "neutral": 1.0, "compound": 0.0}
    return {
        "positive": positive / denominator,
        "negative": negative / denominator,
        "neutral": neutral / denominator,
        "compound": compound,
    }


def vader_valence(token: str) -> float:
    if token in VADER_EXTRA_VALENCE:
        return VADER_EXTRA_VALENCE[token]
    if token in POSITIVE_WORDS:
        return 1.6
    if token in NEGATIVE_WORDS:
        return -1.6
    return 0.0


def _external_vader_scores(text: str) -> dict[str, float] | None:
    try:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    except Exception:
        return None
    try:
        scores = SentimentIntensityAnalyzer().polarity_scores(text)
    except Exception:
        return None
    return {
        "positive": float(scores.get("pos", 0.0)),
        "negative": float(scores.get("neg", 0.0)),
        "neutral": float(scores.get("neu", 0.0)),
        "compound": float(scores.get("compound", 0.0)),
    }


def vader_label(compound: float) -> str:
    if compound >= 0.05:
        return "positive"
    if compound <= -0.05:
        return "negative"
    return "neutral"


def label_from_signed_score(score: float) -> str:
    if score > 0:
        return "positive"
    if score < 0:
        return "negative"
    return "neutral"


def multilingual_lexicons(language: str) -> tuple[frozenset[str], frozenset[str]]:
    if language == "tr":
        return TURKISH_POSITIVE_WORDS, TURKISH_NEGATIVE_WORDS
    return POSITIVE_WORDS, NEGATIVE_WORDS


def sentiment_resource_filename(method: str, language: str) -> str:
    clean_language = (language or "").lower()
    if method == METHOD_SENTIART:
        if clean_language not in SENTIART_LANGUAGES:
            raise ValueError("SentiART resources support English (en) and German (de).")
        return f"SentiArt_{SENTIART_LANGUAGES[clean_language]}.pickle"
    if method == METHOD_LILAH:
        if clean_language not in LILAH_LANGUAGES:
            raise ValueError("LiLaH resources support Croatian (hr), Dutch (nl), and Slovenian (sl).")
        return f"LiLaH-{LILAH_LANGUAGES[clean_language]}.pickle"
    raise ValueError(f"{method} does not use a semantic sentiment resource.")


def semantic_sentiment_resource_url(method: str, language: str) -> str:
    filename = sentiment_resource_filename(method, language)
    if method == METHOD_SENTIART:
        return f"{SENTIART_SERVER_URL}{filename}"
    if method == METHOD_LILAH:
        return f"{LILAH_SERVER_URL}{filename}"
    raise ValueError(f"{method} does not use a semantic sentiment resource.")


def semantic_score_names(method: str) -> tuple[str, ...]:
    if method == METHOD_SENTIART:
        return SENTIART_SCORE_NAMES
    if method == METHOD_LILAH:
        return LILAH_SCORE_NAMES
    raise ValueError(f"{method} does not use semantic sentiment scores.")


def default_sentiment_resource_dir() -> Path:
    return Path.home() / ".portakal" / "sentiment_lexicons"


def candidate_sentiment_resource_dirs(resource_dir: str | Path | None = None) -> tuple[Path, ...]:
    candidates: list[Path] = []

    def add_candidate(value: str | Path | None) -> None:
        if value is None:
            return
        path = Path(value).expanduser()
        if path not in candidates:
            candidates.append(path)

    add_candidate(resource_dir)
    add_candidate(os.environ.get(SENTIMENT_RESOURCE_ENV))
    add_candidate(Path(__file__).resolve().parents[2] / "resources" / "sentiment_lexicons")
    add_candidate(default_sentiment_resource_dir())
    add_candidate(Path.home() / "Library" / "Application Support" / "Portakal" / "sentiment_lexicons")
    return tuple(candidates)


def load_semantic_sentiment_resource(
    method: str,
    language: str,
    resource_dir: str | Path | None = None,
) -> tuple[dict[str, Mapping[str, float]], Path]:
    filename = sentiment_resource_filename(method, language)
    score_names = semantic_score_names(method)
    searched: list[str] = []
    clean_resource_dir = str(resource_dir).strip() if resource_dir is not None else ""
    candidates = (Path(clean_resource_dir).expanduser(),) if clean_resource_dir else candidate_sentiment_resource_dirs()
    for candidate in candidates:
        searched.append(str(candidate))
        paths = (candidate,) if candidate.is_file() else (candidate / filename,)
        for path in paths:
            if path.exists():
                return load_semantic_sentiment_dictionary(path, score_names), path
    if not clean_resource_dir:
        path = download_semantic_sentiment_resource(method, language)
        return load_semantic_sentiment_dictionary(path, score_names), path
    raise FileNotFoundError(
        f"{filename} was not found. Choose a resource folder or set {SENTIMENT_RESOURCE_ENV}."
        + (f" Searched: {', '.join(searched)}." if searched else "")
    )


def download_semantic_sentiment_resource(
    method: str,
    language: str,
    download_dir: str | Path | None = None,
    *,
    timeout: int = 45,
) -> Path:
    target_dir = Path(download_dir).expanduser() if download_dir is not None else default_sentiment_resource_dir()
    target_dir.mkdir(parents=True, exist_ok=True)
    filename = sentiment_resource_filename(method, language)
    target = target_dir / filename
    if target.exists():
        return target

    url = semantic_sentiment_resource_url(method, language)
    temporary_target = target.with_suffix(target.suffix + ".download")
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            payload = response.read()
    except (OSError, urllib.error.URLError) as error:
        raise FileNotFoundError(f"Could not download {filename} from Orange/Biolab: {error}") from error

    if not payload:
        raise FileNotFoundError(f"Downloaded {filename} from Orange/Biolab, but the file was empty.")
    temporary_target.write_bytes(payload)
    temporary_target.replace(target)
    return target


def load_semantic_sentiment_dictionary(
    path: str | Path,
    score_names: Sequence[str],
) -> dict[str, Mapping[str, float]]:
    resource_path = Path(path).expanduser()
    with resource_path.open("rb") as handle:
        payload = pickle.load(handle)
    if not isinstance(payload, Mapping):
        raise ValueError(f"{resource_path.name} must contain a word-to-score mapping.")

    lexicon: dict[str, Mapping[str, float]] = {}
    for word, entry in payload.items():
        clean_word = str(word).strip().lower()
        if not clean_word:
            continue
        lexicon[clean_word] = {
            score_name: _semantic_entry_value(entry, score_name, index)
            for index, score_name in enumerate(score_names)
        }
    return lexicon


def semantic_sentiment_scores(
    text: str,
    lexicon: Mapping[str, Mapping[str, float]],
    score_names: Sequence[str],
) -> dict[str, float]:
    matched = [lexicon[token] for token in sentiment_tokens(text) if token in lexicon]
    if not matched:
        return {score_name: 0.0 for score_name in score_names}
    return {
        score_name: sum(float(row.get(score_name, 0.0)) for row in matched) / len(matched)
        for score_name in score_names
    }


def _semantic_entry_value(entry: object, score_name: str, index: int) -> float:
    value: object = 0.0
    if isinstance(entry, Mapping):
        aliases = SENTIART_SCORE_ALIASES.get(
            score_name,
            (score_name, score_name.lower(), score_name.upper(), score_name.title()),
        )
        for key in aliases:
            if key in entry:
                value = entry[key]
                break
    elif isinstance(entry, Sequence) and not isinstance(entry, (str, bytes, bytearray)):
        if index < len(entry):
            value = entry[index]
    return _safe_float(value)


def _safe_float(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _format_score(value: object) -> str:
    return f"{_safe_float(value):.3f}"


def load_sentiment_dictionary(path: str) -> frozenset[str]:
    clean_path = path.strip()
    if not clean_path:
        return frozenset()
    dictionary_path = Path(clean_path).expanduser()
    words: set[str] = set()
    with dictionary_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip().lower()
            if not stripped or stripped.startswith("#"):
                continue
            words.add(stripped)
    return frozenset(words)


def _document_from_unknown(item: object, index: int) -> CorpusDocument | None:
    if isinstance(item, CorpusDocument):
        return item
    if isinstance(item, Mapping):
        text = _mapping_text(item)
        if text is None:
            return None
        title = str(item.get("title") or item.get("name") or item.get("path") or f"Document {index}")
        source = str(item.get("source") or item.get("category") or item.get("label") or "Input")
        return CorpusDocument(title, text, source)
    text = _attribute_text(item)
    if text is None:
        return None
    title = str(getattr(item, "title", "") or getattr(item, "name", "") or getattr(item, "path", "") or f"Document {index}")
    source = str(getattr(item, "source", "") or getattr(item, "category", "") or getattr(item, "label", "") or "Input")
    return CorpusDocument(title, text, source)


def _mapping_text(item: Mapping[object, object]) -> str | None:
    for key in ("content", "text", "review", "title", "name"):
        value = item.get(key)
        if value:
            return str(value)
    tokens = item.get("tokens")
    if isinstance(tokens, Sequence) and not isinstance(tokens, (str, bytes, bytearray)):
        return " ".join(str(token) for token in tokens)
    return None


def _attribute_text(item: object) -> str | None:
    for name in ("content", "text", "review", "title", "name"):
        value = getattr(item, name, None)
        if value:
            return str(value)
    tokens = getattr(item, "tokens", None)
    if isinstance(tokens, Sequence) and not isinstance(tokens, (str, bytes, bytearray)):
        return " ".join(str(token) for token in tokens)
    return None


class SentimentAnalysisScreen(QWidget, WorkflowNodeScreenSupport):
    def __init__(
        self,
        parent: QWidget | None = None,
        documents: Sequence[CorpusDocument] | None = None,
    ) -> None:
        super().__init__(parent)
        self._init_workflow_node_support()
        self._documents = tuple(() if documents is None else documents)
        self._using_input_corpus = documents is not None
        self._results: tuple[SentimentResult, ...] = ()
        self._output_documents: tuple[SentimentCorpusDocument, ...] = ()
        self._last_warning = ""
        self._last_resource_path = ""

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(10)

        layout.addWidget(
            SectionHeader(
                "Sentiment Analysis",
                "Score corpus documents with lexicon and VADER-style sentiment methods.",
            )
        )
        layout.addWidget(self._build_options_panel())
        layout.addWidget(self._build_metadata_panel())
        layout.addWidget(self._build_table_panel(), 1)

        self._analyze()

    def sizeHint(self) -> QSize:
        return QSize(1040, 700)

    def minimumSizeHint(self) -> QSize:
        return QSize(760, 540)

    def _build_options_panel(self) -> QFrame:
        frame = QFrame(self)
        frame.setProperty("panel", True)
        layout = QGridLayout(frame)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setHorizontalSpacing(12)
        layout.setVerticalSpacing(8)

        layout.addWidget(QLabel("Method", self), 0, 0)
        self._method_combo = QComboBox(self)
        self._method_combo.addItem(METHOD_LABELS[METHOD_LIU_HU], METHOD_LIU_HU)
        self._method_combo.addItem(METHOD_LABELS[METHOD_VADER], METHOD_VADER)
        self._method_combo.addItem(METHOD_LABELS[METHOD_CUSTOM], METHOD_CUSTOM)
        self._method_combo.addItem(METHOD_LABELS[METHOD_MULTILINGUAL], METHOD_MULTILINGUAL)
        self._method_combo.addItem(METHOD_LABELS[METHOD_SENTIART], METHOD_SENTIART)
        self._method_combo.addItem(METHOD_LABELS[METHOD_LILAH], METHOD_LILAH)
        self._method_combo.currentIndexChanged.connect(self._handle_option_changed)
        layout.addWidget(self._method_combo, 0, 1, 1, 2)

        layout.addWidget(QLabel("Language", self), 0, 3)
        self._language_combo = QComboBox(self)
        self._language_combo.addItem("English", "en")
        self._language_combo.addItem("Turkish", "tr")
        self._language_combo.addItem("German", "de")
        self._language_combo.addItem("Croatian", "hr")
        self._language_combo.addItem("Dutch", "nl")
        self._language_combo.addItem("Slovenian", "sl")
        self._language_combo.currentIndexChanged.connect(self._handle_option_changed)
        layout.addWidget(self._language_combo, 0, 4)

        self._auto_apply_checkbox = QCheckBox("Auto apply", self)
        self._auto_apply_checkbox.setChecked(True)
        layout.addWidget(self._auto_apply_checkbox, 0, 5)

        self._apply_button = QPushButton("Apply Sentiment", self)
        self._apply_button.setProperty("primary", True)
        self._apply_button.clicked.connect(self._analyze)
        layout.addWidget(self._apply_button, 0, 6)

        layout.addWidget(QLabel("Positive dictionary", self), 1, 0)
        self._positive_dictionary_input = QLineEdit(self)
        self._positive_dictionary_input.setPlaceholderText("optional .txt word list")
        apply_readable_line_edit_style(self._positive_dictionary_input)
        self._positive_dictionary_input.textChanged.connect(self._handle_option_changed)
        layout.addWidget(self._positive_dictionary_input, 1, 1, 1, 4)

        self._positive_browse_button = QPushButton("Browse", self)
        self._positive_browse_button.clicked.connect(lambda: self._browse_dictionary(self._positive_dictionary_input))
        layout.addWidget(self._positive_browse_button, 1, 5)

        layout.addWidget(QLabel("Negative dictionary", self), 2, 0)
        self._negative_dictionary_input = QLineEdit(self)
        self._negative_dictionary_input.setPlaceholderText("optional .txt word list")
        apply_readable_line_edit_style(self._negative_dictionary_input)
        self._negative_dictionary_input.textChanged.connect(self._handle_option_changed)
        layout.addWidget(self._negative_dictionary_input, 2, 1, 1, 4)

        self._negative_browse_button = QPushButton("Browse", self)
        self._negative_browse_button.clicked.connect(lambda: self._browse_dictionary(self._negative_dictionary_input))
        layout.addWidget(self._negative_browse_button, 2, 5)

        layout.addWidget(QLabel("Resource folder", self), 3, 0)
        self._resource_directory_input = QLineEdit(self)
        self._resource_directory_input.setPlaceholderText("optional folder with SentiArt_*.pickle / LiLaH-*.pickle")
        apply_readable_line_edit_style(self._resource_directory_input)
        self._resource_directory_input.textChanged.connect(self._handle_option_changed)
        layout.addWidget(self._resource_directory_input, 3, 1, 1, 4)

        self._resource_browse_button = QPushButton("Browse", self)
        self._resource_browse_button.clicked.connect(self._browse_resource_directory)
        layout.addWidget(self._resource_browse_button, 3, 5)

        self._help_label = QLabel(
            "For sentiment analysis, aggressive preprocessing may remove important cues such as negation and punctuation. "
            "SentiART and LiLaH read optional Orange-compatible local pickle resources.",
            self,
        )
        self._help_label.setProperty("muted", True)
        self._help_label.setWordWrap(True)
        layout.addWidget(self._help_label, 4, 0, 1, 7)
        return frame

    def _build_metadata_panel(self) -> QFrame:
        frame = QFrame(self)
        frame.setProperty("panel", True)
        layout = QHBoxLayout(frame)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        self._document_count_label = self._build_metric_label("Documents", "0")
        self._positive_count_label = self._build_metric_label("Positive", "0")
        self._negative_count_label = self._build_metric_label("Negative", "0")
        self._neutral_count_label = self._build_metric_label("Neutral", "0")
        self._method_label = self._build_metric_label("Method", METHOD_LABELS[METHOD_LIU_HU])

        layout.addWidget(self._document_count_label, 1)
        layout.addWidget(self._positive_count_label, 1)
        layout.addWidget(self._negative_count_label, 1)
        layout.addWidget(self._neutral_count_label, 1)
        layout.addWidget(self._method_label, 1)
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

        self._table = QTableWidget(0, 6, self)
        self._table.setHorizontalHeaderLabels(
            ["Document", "Positive", "Negative", "Score", "Label", "Text Preview"]
        )
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
            self._analyze()
            return

        documents = sentiment_documents_from_payload(payload.value)
        self._documents = () if documents is None else documents
        self._using_input_corpus = True
        self._analyze()

    def current_output_payload(self) -> WorkflowPayload:
        return WorkflowPayload("Corpus", self._output_documents)

    def _handle_option_changed(self) -> None:
        if getattr(self, "_auto_apply_checkbox", None) is not None and self._auto_apply_checkbox.isChecked():
            self._analyze()

    def _browse_dictionary(self, target: QLineEdit) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Select sentiment dictionary", "", "Text files (*.txt);;All files (*)")
        if path:
            target.setText(path)
            self._analyze()

    def _browse_resource_directory(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Select sentiment resource folder", "")
        if path:
            self._resource_directory_input.setText(path)
            self._analyze()

    def _analyze(self) -> tuple[SentimentResult, ...]:
        method = self._current_method()
        self._last_warning = ""
        self._last_resource_path = ""
        positive_words: frozenset[str] | None = None
        negative_words: frozenset[str] | None = None
        semantic_lexicon: Mapping[str, Mapping[str, float]] | None = None
        if method == METHOD_CUSTOM:
            positive_words, negative_words = self._custom_dictionaries()
        if method in {METHOD_SENTIART, METHOD_LILAH}:
            semantic_lexicon = self._semantic_resource_lexicon(method, str(self._language_combo.currentData() or "en"))

        self._results = analyze_sentiment(
            self._documents,
            method=method,
            positive_words=positive_words,
            negative_words=negative_words,
            language=self._language_combo.currentData() or "en",
            semantic_lexicon=semantic_lexicon,
        )
        self._output_documents = sentiment_output_documents(self._documents, self._results)
        self._render()
        self._notify_output_changed()
        return self._results

    def _custom_dictionaries(self) -> tuple[frozenset[str], frozenset[str]]:
        try:
            positive_words = load_sentiment_dictionary(self._positive_dictionary_input.text())
            negative_words = load_sentiment_dictionary(self._negative_dictionary_input.text())
        except OSError as error:
            self._last_warning = f"Could not read custom dictionary: {error}"
            return frozenset(), frozenset()
        if not positive_words or not negative_words:
            self._last_warning = "Custom Dictionary requires two .txt files: one positive word list and one negative word list."
        return positive_words, negative_words

    def _semantic_resource_lexicon(self, method: str, language: str) -> Mapping[str, Mapping[str, float]]:
        try:
            lexicon, path = load_semantic_sentiment_resource(
                method,
                language,
                self._resource_directory_input.text(),
            )
        except (OSError, ValueError, pickle.PickleError) as error:
            self._last_warning = f"{METHOD_LABELS.get(method, method)} resource unavailable: {error}"
            return {}
        self._last_resource_path = str(path)
        if not lexicon:
            self._last_warning = f"{METHOD_LABELS.get(method, method)} resource is empty."
        return lexicon

    def _current_method(self) -> str:
        return str(self._method_combo.currentData() or METHOD_LIU_HU)

    def _render(self) -> None:
        positive = sum(1 for result in self._results if result.label == "Positive")
        negative = sum(1 for result in self._results if result.label == "Negative")
        neutral = sum(1 for result in self._results if result.label == "Neutral")
        method = self._current_method()
        method_label = METHOD_LABELS.get(method, METHOD_LABELS[METHOD_LIU_HU])

        self._document_count_label.setText(f"Documents\n{len(self._documents)}")
        self._positive_count_label.setText(f"Positive\n{positive}")
        self._negative_count_label.setText(f"Negative\n{negative}")
        self._neutral_count_label.setText(f"Neutral\n{neutral}")
        self._method_label.setText(f"Method\n{method_label}")

        self._status_label.setText(self._status_text(method))

        headers = self._headers_for_method(method)
        self._table.setColumnCount(len(headers))
        self._table.setHorizontalHeaderLabels(headers)
        self._table.setRowCount(len(self._results))
        for row, result in enumerate(self._results):
            values = self._row_values_for_method(method, result)
            for column, value in enumerate(values):
                self._set_item(row, column, str(value))
        self._table.resizeColumnsToContents()

    def _status_text(self, method: str) -> str:
        if self._last_warning:
            return self._last_warning
        if self._results and self._using_input_corpus:
            base = "Input corpus is connected and sentiment-tagged corpus output is ready."
        elif self._using_input_corpus:
            base = "Input corpus is connected but empty."
        else:
            base = "Connect a Corpus input to analyze sentiment."
        if method == METHOD_VADER:
            return base + " VADER columns: positive, negative, neutral, compound, label."
        if method == METHOD_CUSTOM:
            return base + " Custom Dictionary uses normalized positive minus negative word counts."
        if method == METHOD_MULTILINGUAL:
            return base + " Multilingual Sentiment uses compact English/Turkish built-in lexicons."
        if method == METHOD_SENTIART:
            resource = f" Resource: {self._last_resource_path}." if self._last_resource_path else ""
            return base + " SentiART averages Orange-compatible semantic sentiment scores from local resources." + resource
        if method == METHOD_LILAH:
            resource = f" Resource: {self._last_resource_path}." if self._last_resource_path else ""
            return base + " LiLaH averages Orange-compatible emotion lexicon scores from local resources." + resource
        return base + " Liu & Hu score is normalized: positive > 0, negative < 0, neutral = 0."

    def _headers_for_method(self, method: str) -> list[str]:
        if method == METHOD_VADER:
            return ["Document", "VADER Positive", "VADER Negative", "VADER Neutral", "VADER Compound", "Label", "Text Preview"]
        if method == METHOD_CUSTOM:
            return ["Document", "Positive Matches", "Negative Matches", "Custom Score", "Label", "Text Preview"]
        if method == METHOD_MULTILINGUAL:
            return ["Document", "Positive Matches", "Negative Matches", "Multilingual Score", "Label", "Text Preview"]
        if method == METHOD_SENTIART:
            return [
                "Document",
                "Sentiment",
                "Anger",
                "Fear",
                "Disgust",
                "Happiness",
                "Sadness",
                "Surprise",
                "Label",
                "Text Preview",
            ]
        if method == METHOD_LILAH:
            return [
                "Document",
                "Positive",
                "Negative",
                "Anger",
                "Anticipation",
                "Disgust",
                "Fear",
                "Joy",
                "Sadness",
                "Surprise",
                "Trust",
                "Label",
                "Text Preview",
            ]
        return ["Document", "Positive", "Negative", "Liu & Hu Score", "Label", "Text Preview"]

    def _row_values_for_method(self, method: str, result: SentimentResult) -> list[object]:
        if method == METHOD_VADER:
            return [
                result.document,
                f"{result.columns.get('vader_positive', 0):.3f}",
                f"{result.columns.get('vader_negative', 0):.3f}",
                f"{result.columns.get('vader_neutral', 0):.3f}",
                f"{result.columns.get('vader_compound', 0):.3f}",
                result.label,
                result.text_preview,
            ]
        if method == METHOD_SENTIART:
            return [
                result.document,
                _format_score(result.columns.get("sentiart_sentiment", 0)),
                _format_score(result.columns.get("sentiart_anger", 0)),
                _format_score(result.columns.get("sentiart_fear", 0)),
                _format_score(result.columns.get("sentiart_disgust", 0)),
                _format_score(result.columns.get("sentiart_happiness", 0)),
                _format_score(result.columns.get("sentiart_sadness", 0)),
                _format_score(result.columns.get("sentiart_surprise", 0)),
                result.label,
                result.text_preview,
            ]
        if method == METHOD_LILAH:
            return [
                result.document,
                _format_score(result.columns.get("lilah_positive", 0)),
                _format_score(result.columns.get("lilah_negative", 0)),
                _format_score(result.columns.get("lilah_anger", 0)),
                _format_score(result.columns.get("lilah_anticipation", 0)),
                _format_score(result.columns.get("lilah_disgust", 0)),
                _format_score(result.columns.get("lilah_fear", 0)),
                _format_score(result.columns.get("lilah_joy", 0)),
                _format_score(result.columns.get("lilah_sadness", 0)),
                _format_score(result.columns.get("lilah_surprise", 0)),
                _format_score(result.columns.get("lilah_trust", 0)),
                result.label,
                result.text_preview,
            ]
        return [
            result.document,
            int(result.positive_score),
            int(result.negative_score),
            f"{result.net_score:.3f}",
            result.label,
            result.text_preview,
        ]

    def _set_item(self, row: int, column: int, text: str) -> None:
        self._table.setItem(row, column, QTableWidgetItem(text))

    def data_preview_snapshot(self) -> dict[str, object]:
        method = self._current_method()
        headers = self._headers_for_method(method)
        rows = [[str(value) for value in self._row_values_for_method(method, result)] for result in self._results]
        return {
            "summary": f"Sentiment Analysis: {len(self._results)} documents scored with {METHOD_LABELS.get(method, method)}",
            "headers": headers,
            "rows": rows,
        }
