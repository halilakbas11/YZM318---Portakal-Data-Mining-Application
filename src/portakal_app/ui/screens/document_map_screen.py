"""Document Map widget — geographic country-mention map for a corpus.

Inspired by Orange Text Mining's Document Map: it scans a chosen text field of
every document for explicit country names, counts them, and shades the mentioned
countries on a real, offline world map (light pink → strong red). It is fully
offline (no tiles, no geocoding APIs) and degrades gracefully when input is
missing or no countries are found.
"""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import dataclass

from PySide6.QtCore import QPointF, QRectF, QSize, Qt, Signal
from PySide6.QtGui import (
    QBrush,
    QColor,
    QLinearGradient,
    QPainter,
    QPainterPath,
    QPen,
    QPolygonF,
    QTransform,
)
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QToolTip,
    QVBoxLayout,
    QWidget,
)

from portakal_app.models import WorkflowPayload
from portakal_app.ui.screens.corpus_screen import CorpusDocument, corpus_documents_from_payload
from portakal_app.ui.screens.country_geo import (
    CountryMatcher,
    country_by_iso2,
    load_world_polygons,
)
from portakal_app.ui.screens.node_screen import WorkflowNodeScreenSupport
from portakal_app.ui.shared.cards import SectionHeader


COUNT_DOCUMENTS = "Documents"
COUNT_MENTIONS = "Mentions"
COUNT_MODES = (COUNT_DOCUMENTS, COUNT_MENTIONS)

SCALE_LINEAR = "Linear"
SCALE_LOG = "Log"
SCALE_MODES = (SCALE_LINEAR, SCALE_LOG)

# World is fully implemented; Europe/USA are documented as future work.
MAP_TYPES = ("World",)

_EXAMPLE_LIMIT = 5
_FORM_LIMIT = 6
_COLOR_LOW = QColor("#ffe3ea")     # light pink (low count)
_COLOR_HIGH = QColor("#9b1d2e")    # strong red (high count)
_COLOR_NEUTRAL = QColor("#e9e4d8")  # un-mentioned land
_COLOR_BORDER = QColor("#b3a892")
_COLOR_OCEAN = QColor("#eef3f6")
_COLOR_SELECTED = QColor("#3a2a12")


@dataclass(frozen=True)
class CountryMention:
    name: str
    alpha3: str
    alpha2: str
    region: str
    mention_count: int
    document_count: int
    matched_forms: tuple[str, ...] = ()
    examples: tuple[str, ...] = ()

    def value_for(self, count_mode: str) -> int:
        return self.mention_count if count_mode == COUNT_MENTIONS else self.document_count


@dataclass(frozen=True)
class DocumentMapResult:
    countries: tuple[CountryMention, ...] = ()
    status: str = ""
    attribute_used: str = ""
    count_mode: str = COUNT_DOCUMENTS


def document_field_value(document: CorpusDocument, field_name: str) -> str:
    key = field_name.strip().lower()
    if key in ("content", "text", "body", "document"):
        return _strip_artificial_content_prefix(str(document.text or ""))
    if key in ("title", "name"):
        return str(document.title or "")
    if key in ("source", "category"):
        return str(document.source or "")
    for attr_key, value in dict(document.attributes).items():
        if str(attr_key).lower() == key:
            if isinstance(value, (list, tuple, set, frozenset)):
                return " ".join(str(token) for token in value)
            return str(value or "")
    return ""


def _strip_artificial_content_prefix(text: str) -> str:
    """Remove demo/import metadata prefixes when they were stored in content."""
    return re.sub(
        r"^\s*Title:\s*[^\r\n]*\r?\n\s*Category:\s*[^\r\n]*\r?\n+",
        "",
        text,
        count=1,
        flags=re.IGNORECASE,
    )


def available_text_attributes(documents: Sequence[CorpusDocument]) -> tuple[str, ...]:
    options: list[str] = []

    def add(name: str) -> None:
        if name not in options:
            options.append(name)

    if any(str(d.text or "").strip() for d in documents):
        add("content")
    if any(str(d.title or "").strip() for d in documents):
        add("title")
    if any(str(d.source or "").strip() for d in documents):
        add("source")
    for document in documents:
        for attr_key, value in dict(document.attributes).items():
            key = str(attr_key)
            if key.startswith("bow_") or key == "bow_total":
                continue
            if isinstance(value, (str, list, tuple)) and document_field_value(document, key).strip():
                add(key)
    return tuple(options)


def default_text_attribute(options: Sequence[str]) -> str:
    for preferred in ("content", "text", "title", "name"):
        if preferred in options:
            return preferred
    return options[0] if options else "content"


def build_country_map(
    documents: Sequence[CorpusDocument],
    *,
    attribute: str | None = None,
    count_mode: str = COUNT_DOCUMENTS,
    include_aliases: bool = False,
    include_acronyms: bool = False,
    include_capitals: bool = False,
    matcher: CountryMatcher | None = None,
) -> DocumentMapResult:
    """Detect and count explicit country mentions across the corpus.

    Defaults are conservative (canonical country names only). ``document_count``
    counts each country at most once per document; ``mention_count`` counts every
    occurrence. Only the selected text field is scanned — never title/source/path.
    """
    mode = count_mode if count_mode in COUNT_MODES else COUNT_DOCUMENTS
    if not documents:
        return DocumentMapResult(status="Connect a Corpus input to build a document map.", count_mode=mode)

    options = available_text_attributes(documents)
    if not options:
        return DocumentMapResult(status="No usable text field was found in the input corpus.", count_mode=mode)
    chosen_attribute = attribute if attribute in options else default_text_attribute(options)

    if matcher is None:
        matcher = CountryMatcher(
            include_aliases=include_aliases,
            include_acronyms=include_acronyms,
            include_capitals=include_capitals,
        )

    mention_counter: Counter[str] = Counter()
    document_counter: Counter[str] = Counter()
    examples: dict[str, list[str]] = defaultdict(list)
    forms: dict[str, list[str]] = defaultdict(list)

    scanned_any_text = False
    for index, document in enumerate(documents, start=1):
        text = document_field_value(document, chosen_attribute)
        if text.strip():
            scanned_any_text = True
        detections = matcher.detect_with_forms(text)
        if not detections:
            continue
        per_doc: Counter[str] = Counter()
        for name, surface in detections:
            per_doc[name] += 1
            normalized_form = surface.strip()
            if normalized_form and normalized_form not in forms[name] and len(forms[name]) < _FORM_LIMIT:
                forms[name].append(normalized_form)
        for name, count in per_doc.items():
            mention_counter[name] += count
            document_counter[name] += 1
            title = document.title or f"Document {index}"
            if len(examples[name]) < _EXAMPLE_LIMIT and title not in examples[name]:
                examples[name].append(title)

    if not scanned_any_text:
        return DocumentMapResult(
            status=f"The selected field '{chosen_attribute}' has no text to scan.",
            attribute_used=chosen_attribute,
            count_mode=mode,
        )
    if not mention_counter:
        return DocumentMapResult(
            status=(
                f"No country or region names were detected in '{chosen_attribute}'. "
                "Document Map highlights only explicitly mentioned countries."
            ),
            attribute_used=chosen_attribute,
            count_mode=mode,
        )

    countries: list[CountryMention] = []
    for name, mentions in mention_counter.items():
        country = matcher.country(name)
        if country is None:
            continue
        countries.append(
            CountryMention(
                name=country.name,
                alpha3=country.alpha3,
                alpha2=country.alpha2,
                region=country.region,
                mention_count=mentions,
                document_count=document_counter[name],
                matched_forms=tuple(forms[name]),
                examples=tuple(examples[name]),
            )
        )
    countries.sort(key=lambda c: (c.value_for(mode), c.mention_count, c.name), reverse=True)
    status = (
        f"Detected {len(countries)} countries in '{chosen_attribute}' "
        f"across {len(documents)} documents (count mode: {mode.lower()})."
    )
    return DocumentMapResult(
        countries=tuple(countries),
        status=status,
        attribute_used=chosen_attribute,
        count_mode=mode,
    )


# Backwards-compatible alias for older imports/workflows.
build_document_map = build_country_map


def country_match_debug(
    documents: Sequence[CorpusDocument],
    *,
    attribute: str = "content",
    matcher: CountryMatcher | None = None,
    countries: Sequence[str] | None = None,
    snippet_radius: int = 40,
) -> dict[str, dict]:
    """Per-country diagnostic of *what* matched and *where*.

    Returns, for each detected country, its document_count (unique documents),
    mention_count, a per-surface-form breakdown (document_count + example snippets),
    and the list of document titles that mention it. Used to explain inflated
    counts (e.g. United Kingdom driven by ``UK``/``Britain``/``Scotland``). The
    matcher defaults to an *all-on* configuration so the report exposes every form
    that could match, regardless of the widget's current conservative settings.
    """
    matcher = matcher or CountryMatcher(
        include_aliases=True, include_acronyms=True, include_capitals=True
    )
    report: dict[str, dict] = {}
    for index, document in enumerate(documents, start=1):
        text = document_field_value(document, attribute)
        title = document.title or f"Document {index}"
        detections = matcher.detect_with_forms(text)
        per_doc_names: set[str] = set()
        for name, surface in detections:
            entry = report.setdefault(
                name,
                {"document_count": 0, "mention_count": 0, "documents": [], "forms": {}},
            )
            entry["mention_count"] += 1
            form = entry["forms"].setdefault(surface, {"document_count": 0, "snippets": []})
            form.setdefault("_docs", set())
            idx = text.find(surface)
            if idx >= 0 and len(form["snippets"]) < 3:
                start = max(0, idx - snippet_radius)
                end = min(len(text), idx + len(surface) + snippet_radius)
                form["snippets"].append("…" + text[start:end].replace("\n", " ").strip() + "…")
            if title not in form["_docs"]:
                form["_docs"].add(title)
                form["document_count"] += 1
            per_doc_names.add(name)
        for name in per_doc_names:
            entry = report[name]
            entry["document_count"] += 1
            if title not in entry["documents"]:
                entry["documents"].append(title)
    # Drop the internal helper set so the result is JSON-friendly.
    for entry in report.values():
        for form in entry["forms"].values():
            form.pop("_docs", None)
    if countries is not None:
        report = {name: report.get(name, {"document_count": 0, "mention_count": 0,
                                           "documents": [], "forms": {}})
                  for name in countries}
    return report


def _interpolate_color(ratio: float) -> QColor:
    ratio = max(0.0, min(1.0, ratio))
    r = _COLOR_LOW.red() + (_COLOR_HIGH.red() - _COLOR_LOW.red()) * ratio
    g = _COLOR_LOW.green() + (_COLOR_HIGH.green() - _COLOR_LOW.green()) * ratio
    b = _COLOR_LOW.blue() + (_COLOR_HIGH.blue() - _COLOR_LOW.blue()) * ratio
    return QColor(int(r), int(g), int(b))


def _intensity(value: int, max_value: int, scale: str) -> float:
    if max_value <= 0:
        return 1.0
    if scale == SCALE_LOG:
        return math.log1p(max(0, value)) / (math.log1p(max_value) or 1.0)
    return value / max_value


class WorldMapCanvas(QFrame):
    countryClicked = Signal(str)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setProperty("panel", True)
        self.setMinimumHeight(340)
        self.setMouseTracking(True)
        self._count_mode = COUNT_DOCUMENTS
        self._scale = SCALE_LINEAR
        self._selected: str | None = None
        self._countries: tuple[CountryMention, ...] = ()
        self._value_by_iso2: dict[str, int] = {}
        self._name_by_iso2 = {iso2: c.name for iso2, c in country_by_iso2().items()}
        self._max_value = 0
        self._transform = QTransform()
        self._view_box, polygons = load_world_polygons()
        self._paths: dict[str, QPainterPath] = {}
        for iso2, polys in polygons.items():
            path = QPainterPath()
            for poly in polys:
                if len(poly) < 3:
                    continue
                path.addPolygon(QPolygonF([QPointF(x, y) for x, y in poly]))
            if not path.isEmpty():
                self._paths[iso2] = path

    def has_polygons(self) -> bool:
        return bool(self._paths)

    def configure(self, *, count_mode: str, scale: str) -> None:
        self._count_mode = count_mode
        self._scale = scale
        self._recompute_values()
        self.update()

    def set_countries(self, countries: Sequence[CountryMention]) -> None:
        self._countries = tuple(countries)
        self._recompute_values()
        self.update()

    def set_selected(self, name: str | None) -> None:
        self._selected = name
        self.update()

    def _recompute_values(self) -> None:
        self._value_by_iso2 = {}
        for country in self._countries:
            self._value_by_iso2[country.alpha2.lower()] = country.value_for(self._count_mode)
        self._max_value = max(self._value_by_iso2.values(), default=0)

    def _plot_rect(self) -> QRectF:
        return QRectF(self.rect().adjusted(12, 12, -12, -44))

    def _compute_transform(self, rect: QRectF) -> QTransform:
        _minx, _miny, width, height = self._view_box
        width = width or 1.0
        height = height or 1.0
        scale = min(rect.width() / width, rect.height() / height)
        draw_w = width * scale
        draw_h = height * scale
        ox = rect.left() + (rect.width() - draw_w) / 2.0 - _minx * scale
        oy = rect.top() + (rect.height() - draw_h) / 2.0 - _miny * scale
        transform = QTransform()
        transform.translate(ox, oy)
        transform.scale(scale, scale)
        return transform

    def paintEvent(self, event) -> None:  # noqa: N802
        super().paintEvent(event)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        rect = self._plot_rect()

        painter.setPen(QPen(_COLOR_BORDER, 1))
        painter.setBrush(QBrush(_COLOR_OCEAN))
        painter.drawRect(rect)

        if not self._paths:
            painter.setPen(QColor("#7e715e"))
            painter.drawText(rect, Qt.AlignmentFlag.AlignCenter,
                             "World map asset is unavailable.")
            painter.end()
            return

        self._transform = self._compute_transform(rect)
        painter.save()
        painter.setClipRect(rect)
        painter.setTransform(self._transform, True)

        border_pen = QPen(_COLOR_BORDER)
        border_pen.setCosmetic(True)
        border_pen.setWidthF(0.7)

        # Base layer: every country in neutral land colour.
        painter.setPen(border_pen)
        for iso2, path in self._paths.items():
            if iso2 in self._value_by_iso2:
                continue
            painter.setBrush(QBrush(_COLOR_NEUTRAL))
            painter.drawPath(path)

        # Highlight layer: mentioned countries by intensity.
        for iso2, value in self._value_by_iso2.items():
            path = self._paths.get(iso2)
            if path is None:
                continue
            ratio = _intensity(value, self._max_value, self._scale) if self._max_value > 1 else 1.0
            painter.setBrush(QBrush(_interpolate_color(ratio)))
            painter.setPen(border_pen)
            painter.drawPath(path)

        # Selected outline on top.
        if self._selected:
            iso2 = self._iso2_for_name(self._selected)
            path = self._paths.get(iso2)
            if path is not None:
                sel_pen = QPen(_COLOR_SELECTED)
                sel_pen.setCosmetic(True)
                sel_pen.setWidthF(2.0)
                painter.setPen(sel_pen)
                painter.setBrush(Qt.BrushStyle.NoBrush)
                painter.drawPath(path)
        painter.restore()

        self._draw_legend(painter, rect)
        painter.end()

    def _draw_legend(self, painter: QPainter, rect: QRectF) -> None:
        bar_width = 170.0
        bar_height = 12.0
        x = rect.right() - bar_width
        y = rect.bottom() + 14.0
        gradient = QLinearGradient(x, 0, x + bar_width, 0)
        gradient.setColorAt(0.0, _COLOR_LOW)
        gradient.setColorAt(1.0, _COLOR_HIGH)
        painter.setPen(QPen(_COLOR_BORDER, 1))
        painter.setBrush(QBrush(gradient))
        painter.drawRect(QRectF(x, y, bar_width, bar_height))
        painter.setPen(QColor("#4a4034"))
        label = "mentions" if self._count_mode == COUNT_MENTIONS else "documents"
        painter.drawText(QRectF(x - 170, y - 2, 166, bar_height + 4),
                         Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
                         f"fewer ← {label} → more")
        painter.drawText(QRectF(x, y + bar_height, 40, 16), Qt.AlignmentFlag.AlignLeft,
                         "0" if not self._max_value else "1")
        painter.drawText(QRectF(x + bar_width - 40, y + bar_height, 40, 16),
                         Qt.AlignmentFlag.AlignRight, str(self._max_value))

    def _iso2_for_name(self, name: str) -> str:
        for country in self._countries:
            if country.name == name:
                return country.alpha2.lower()
        return ""

    def _country_at(self, pos: QPointF) -> CountryMention | None:
        inverted, ok = self._transform.inverted()
        if not ok:
            return None
        svg_point = inverted.map(pos)
        for country in self._countries:
            path = self._paths.get(country.alpha2.lower())
            if path is not None and path.contains(svg_point):
                return country
        return None

    def mouseMoveEvent(self, event) -> None:  # noqa: N802
        country = self._country_at(QPointF(event.position()))
        if country is not None:
            QToolTip.showText(
                event.globalPosition().toPoint(),
                f"{country.name} — mentions: {country.mention_count}, "
                f"documents: {country.document_count}",
                self,
            )
        else:
            QToolTip.hideText()
        super().mouseMoveEvent(event)

    def mousePressEvent(self, event) -> None:  # noqa: N802
        country = self._country_at(QPointF(event.position()))
        if country is not None:
            self._selected = country.name
            self.update()
            self.countryClicked.emit(country.name)
        super().mousePressEvent(event)


class DocumentMapScreen(QWidget, WorkflowNodeScreenSupport):
    def __init__(
        self,
        parent: QWidget | None = None,
        documents: Sequence[CorpusDocument] | None = None,
    ) -> None:
        super().__init__(parent)
        self._init_workflow_node_support()
        self._documents = tuple(() if documents is None else documents)
        self._using_input_corpus = documents is not None
        self._result = DocumentMapResult()
        self._countries_by_name: dict[str, CountryMention] = {}
        self._updating_controls = False

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(10)

        layout.addWidget(
            SectionHeader(
                "Document Map",
                "Scan a text field for country names and shade them on an offline world map.",
            )
        )
        layout.addWidget(self._build_controls_panel())

        self._status_label = QLabel("", self)
        self._status_label.setProperty("muted", True)
        self._status_label.setWordWrap(True)
        layout.addWidget(self._status_label)

        self._canvas = WorldMapCanvas(self)
        self._canvas.countryClicked.connect(self._on_country_selected)
        layout.addWidget(self._canvas, 3)

        layout.addWidget(self._build_lower_panel(), 2)

        self._build_map()

    def sizeHint(self) -> QSize:
        return QSize(1040, 780)

    def minimumSizeHint(self) -> QSize:
        return QSize(780, 580)

    def _build_controls_panel(self) -> QFrame:
        frame = QFrame(self)
        frame.setProperty("panel", True)
        layout = QGridLayout(frame)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setHorizontalSpacing(12)
        layout.setVerticalSpacing(8)

        layout.addWidget(QLabel("Region attribute", self), 0, 0)
        self._attribute_combo = QComboBox(self)
        self._attribute_combo.setToolTip("Which text field to scan for country names.")
        self._attribute_combo.currentTextChanged.connect(lambda _t: self._on_controls_changed())
        layout.addWidget(self._attribute_combo, 0, 1)

        layout.addWidget(QLabel("Map type", self), 0, 2)
        self._map_combo = QComboBox(self)
        self._map_combo.addItems(MAP_TYPES)
        self._map_combo.setToolTip("World map. Europe/USA views are planned future work.")
        layout.addWidget(self._map_combo, 0, 3)

        layout.addWidget(QLabel("Count mode", self), 1, 0)
        self._count_combo = QComboBox(self)
        self._count_combo.addItems(COUNT_MODES)
        self._count_combo.setToolTip(
            "Documents: count each country once per document. Mentions: count every occurrence."
        )
        self._count_combo.currentTextChanged.connect(lambda _t: self._on_count_or_scale_changed())
        layout.addWidget(self._count_combo, 1, 1)

        layout.addWidget(QLabel("Color scale", self), 1, 2)
        self._scale_combo = QComboBox(self)
        self._scale_combo.addItems(SCALE_MODES)
        self._scale_combo.currentTextChanged.connect(lambda _t: self._on_count_or_scale_changed())
        layout.addWidget(self._scale_combo, 1, 3)

        self._aliases_checkbox = QCheckBox("Include aliases", self)
        self._aliases_checkbox.setChecked(False)
        self._aliases_checkbox.setToolTip(
            "Match broad aliases like America, Britain, England, Scotland. Off by default — these "
            "inflate counts on news corpora."
        )
        self._aliases_checkbox.toggled.connect(lambda _v: self._on_controls_changed())
        layout.addWidget(self._aliases_checkbox, 2, 0, 1, 2)

        self._acronyms_checkbox = QCheckBox("Include acronyms", self)
        self._acronyms_checkbox.setChecked(False)
        self._acronyms_checkbox.setToolTip(
            "Match country acronyms like US, USA, UK, UAE (case-sensitive). Off by default — US/UK "
            "appear in most news articles and inflate counts."
        )
        self._acronyms_checkbox.toggled.connect(lambda _v: self._on_controls_changed())
        layout.addWidget(self._acronyms_checkbox, 2, 2, 1, 2)

        self._capitals_checkbox = QCheckBox("Include capitals", self)
        self._capitals_checkbox.setChecked(False)
        self._capitals_checkbox.setToolTip("Also match capital city names (e.g. London → United Kingdom). Off by default.")
        self._capitals_checkbox.toggled.connect(lambda _v: self._on_controls_changed())
        layout.addWidget(self._capitals_checkbox, 2, 4, 1, 2)

        self._apply_button = QPushButton("Refresh", self)
        self._apply_button.setProperty("primary", True)
        self._apply_button.clicked.connect(self._build_map)
        layout.addWidget(self._apply_button, 0, 5, 2, 1)

        self._help_label = QLabel(
            "Mentioned countries are shaded by count (redder = more). Detection is conservative by "
            "default: explicit country names only. Enable aliases/acronyms/capitals to widen matching "
            "(this raises counts). Not topic modelling or sentiment.",
            self,
        )
        self._help_label.setProperty("muted", True)
        self._help_label.setWordWrap(True)
        layout.addWidget(self._help_label, 3, 0, 1, 6)
        return frame

    def _build_lower_panel(self) -> QFrame:
        frame = QFrame(self)
        layout = QHBoxLayout(frame)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)
        layout.addWidget(self._build_table_panel(), 3)
        layout.addWidget(self._build_details_panel(), 2)
        return frame

    def _build_table_panel(self) -> QFrame:
        frame = QFrame(self)
        frame.setProperty("panel", True)
        layout = QVBoxLayout(frame)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)
        layout.addWidget(QLabel("Detected countries", self))

        self._table = QTableWidget(0, 6, self)
        self._table.setHorizontalHeaderLabels(
            ["Country", "ISO", "Mentions", "Documents", "Matched forms", "Example documents"]
        )
        self._table.verticalHeader().setVisible(False)
        self._table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self._table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self._table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._table.setSortingEnabled(True)
        header = self._table.horizontalHeader()
        header.setStretchLastSection(True)
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        for column in (1, 2, 3):
            header.setSectionResizeMode(column, QHeaderView.ResizeMode.ResizeToContents)
        self._table.itemSelectionChanged.connect(self._on_table_selection_changed)
        layout.addWidget(self._table, 1)
        return frame

    def _build_details_panel(self) -> QFrame:
        frame = QFrame(self)
        frame.setProperty("panel", True)
        layout = QVBoxLayout(frame)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(6)
        layout.addWidget(QLabel("Country details", self))
        self._details_label = QLabel("Click a country on the map or a table row to see details.", self)
        self._details_label.setWordWrap(True)
        self._details_label.setProperty("muted", True)
        self._details_label.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        layout.addWidget(self._details_label, 1)
        return frame

    # ── Input / control handling ───────────────────────────────────────
    def set_input_payload(self, payload: WorkflowPayload | None) -> None:
        if payload is None:
            self._documents = ()
            self._using_input_corpus = False
            self._build_map()
            return
        documents = corpus_documents_from_payload(payload.value)
        self._documents = () if documents is None else documents
        self._using_input_corpus = True
        print(f"Document Map received corpus: {len(self._documents)} documents")
        self._build_map()

    def _refresh_attribute_options(self) -> None:
        options = available_text_attributes(self._documents)
        current = self._attribute_combo.currentText()
        self._updating_controls = True
        self._attribute_combo.clear()
        self._attribute_combo.addItems(options)
        if current in options:
            self._attribute_combo.setCurrentText(current)
        elif options:
            self._attribute_combo.setCurrentText(default_text_attribute(options))
        self._updating_controls = False

    def _on_controls_changed(self) -> None:
        if not self._updating_controls:
            self._build_map()

    def _on_count_or_scale_changed(self) -> None:
        # Count mode changes the value driving the colour; rebuild so sort/counts follow.
        if not self._updating_controls:
            self._build_map()

    def _build_map(self) -> DocumentMapResult:
        self._refresh_attribute_options()
        attribute = self._attribute_combo.currentText() or None
        self._result = build_country_map(
            self._documents,
            attribute=attribute,
            count_mode=self._count_combo.currentText(),
            include_aliases=self._aliases_checkbox.isChecked(),
            include_acronyms=self._acronyms_checkbox.isChecked(),
            include_capitals=self._capitals_checkbox.isChecked(),
        )
        self._countries_by_name = {c.name: c for c in self._result.countries}
        self._render()
        self._notify_output_changed()
        return self._result

    def _render(self) -> None:
        if not self._using_input_corpus:
            self._status_label.setText("Connect a Corpus input to build a document map.")
        else:
            self._status_label.setText(self._result.status or "Input corpus is connected.")

        self._canvas.configure(count_mode=self._result.count_mode, scale=self._scale_combo.currentText())
        self._canvas.set_countries(self._result.countries)
        self._canvas.set_selected(None)
        self._details_label.setText("Click a country on the map or a table row to see details.")

        self._table.setSortingEnabled(False)
        self._table.setRowCount(len(self._result.countries))
        for row, country in enumerate(self._result.countries):
            self._set_text(row, 0, country.name)
            self._set_text(row, 1, country.alpha3)
            self._set_number(row, 2, country.mention_count)
            self._set_number(row, 3, country.document_count)
            self._set_text(row, 4, ", ".join(country.matched_forms))
            self._set_text(row, 5, ", ".join(country.examples))
        self._table.setSortingEnabled(True)

    def _set_text(self, row: int, column: int, text: str) -> None:
        self._table.setItem(row, column, QTableWidgetItem(text))

    def _set_number(self, row: int, column: int, value: int) -> None:
        item = QTableWidgetItem()
        item.setData(Qt.ItemDataRole.DisplayRole, int(value))
        item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self._table.setItem(row, column, item)

    def _on_country_selected(self, name: str) -> None:
        self._show_details(name)
        for row in range(self._table.rowCount()):
            item = self._table.item(row, 0)
            if item is not None and item.text() == name:
                self._table.blockSignals(True)
                self._table.selectRow(row)
                self._table.blockSignals(False)
                break

    def _on_table_selection_changed(self) -> None:
        model = self._table.selectionModel()
        rows = model.selectedRows() if model else []
        if not rows:
            return
        item = self._table.item(rows[0].row(), 0)
        if item is None:
            return
        name = item.text()
        self._canvas.set_selected(name)
        self._show_details(name)

    def _show_details(self, name: str) -> None:
        country = self._countries_by_name.get(name)
        if country is None:
            self._details_label.setText("Click a country on the map or a table row to see details.")
            return
        forms = ", ".join(country.matched_forms) or "(none)"
        examples = "\n".join(f"  • {title}" for title in country.examples) or "  • (none)"
        self._details_label.setText(
            f"{country.name} ({country.alpha3})\n"
            f"Mentions: {country.mention_count}\n"
            f"Documents: {country.document_count}\n"
            f"Matched forms: {forms}\n"
            f"Example documents:\n{examples}"
        )

    def data_preview_snapshot(self) -> dict[str, object]:
        headers = ["Country", "ISO", "Mentions", "Documents", "Matched forms", "Example documents"]
        rows = [
            [c.name, c.alpha3, str(c.mention_count), str(c.document_count),
             ", ".join(c.matched_forms), ", ".join(c.examples)]
            for c in self._result.countries
        ]
        return {
            "summary": f"Document Map: {len(self._result.countries)} countries detected",
            "headers": headers,
            "rows": rows,
        }

    def country_counts(self) -> tuple[dict[str, object], ...]:
        return tuple(
            {
                "country": c.name,
                "iso_alpha3": c.alpha3,
                "iso_alpha2": c.alpha2,
                "mention_count": c.mention_count,
                "document_count": c.document_count,
                "matched_forms": list(c.matched_forms),
                "example_documents": list(c.examples),
            }
            for c in self._result.countries
        )
