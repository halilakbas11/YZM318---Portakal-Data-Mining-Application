"""Offline country/region detection + world-polygon data for the Document Map.

Two responsibilities live here:

1. A conservative ``CountryMatcher`` that finds explicit country mentions in text
   without the false positives that plague naive matching.
2. A loader for the bundled offline world-countries polygon asset used to draw a
   real, recognisable world map (no tiles, no network, no GIS dependencies).

Conservative matching rules (this is the important fix):

* A word/alias only counts when it appears **capitalised** (its first letter is
  upper-case). Country names are proper nouns, so this single rule rejects the
  lowercase common words that caused over-detection — ``can`` (Canada), ``in``
  (India), ``it`` (Italy), ``us`` (United States), ``china`` (porcelain),
  ``turkey`` (the bird) — while still matching ``Canada``, ``India``, ``Turkey``.
* Short acronyms (``USA``, ``U.S.``, ``U.S.A.``, ``UK``, ``U.K.``, ``UAE``,
  ``U.A.E.``) are OFF by default and only matched when explicitly enabled. When
  enabled they are matched case-sensitively (upper-case only), so ``us`` and
  ``uk`` in ordinary prose never match.
* All matches are word-boundary safe, so ``Indiana`` never triggers ``India`` and
  ``business`` never triggers ``US``.
* Longer names win over shorter aliases at the same position and overlapping
  matches are resolved greedily, so a mention is never double counted.
* Broad aliases and capital-city names are OFF by default. ``America``,
  ``Britain``, ``England``, ``Scotland`` and ``London`` are intentionally opt-in
  because they inflate BBC/news corpus counts.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path


@dataclass(frozen=True)
class Country:
    name: str
    alpha3: str
    alpha2: str
    longitude: float
    latitude: float
    region: str = "World"
    # Curated, low-risk word aliases (matched only when capitalised).
    aliases: tuple[str, ...] = ()
    # Case-sensitive acronyms (upper-case only).
    acronyms: tuple[str, ...] = ()
    # Capital-city names (matched only when "Include capitals" is enabled).
    capitals: tuple[str, ...] = ()


_R_EU = "Europe"
_R_AS = "Asia"
_R_AF = "Africa"
_R_NA = "Americas"
_R_SA = "Americas"
_R_OC = "Oceania"


_COUNTRIES: tuple[Country, ...] = (
    # ── Americas ──
    Country("United States", "USA", "US", -98.5, 39.8, _R_NA,
            aliases=("United States of America", "United States", "America"),
            acronyms=("USA", "U.S.A.", "U.S.", "US", "U.S"),
            capitals=("Washington",)),
    Country("Canada", "CAN", "CA", -106.3, 56.1, _R_NA, aliases=("Canada",), capitals=("Ottawa",)),
    Country("Mexico", "MEX", "MX", -102.5, 23.6, _R_NA, aliases=("Mexico",), capitals=("Mexico City",)),
    Country("Brazil", "BRA", "BR", -51.9, -14.2, _R_SA, aliases=("Brazil",), capitals=("Brasilia",)),
    Country("Argentina", "ARG", "AR", -63.6, -38.4, _R_SA, aliases=("Argentina",), capitals=("Buenos Aires",)),
    Country("Chile", "CHL", "CL", -71.5, -35.7, _R_SA, aliases=("Chile",)),
    Country("Colombia", "COL", "CO", -74.3, 4.6, _R_SA, aliases=("Colombia",)),
    Country("Peru", "PER", "PE", -75.0, -9.2, _R_SA, aliases=("Peru",)),
    Country("Venezuela", "VEN", "VE", -66.6, 6.4, _R_SA, aliases=("Venezuela",)),
    Country("Cuba", "CUB", "CU", -77.8, 21.5, _R_NA, aliases=("Cuba",)),
    # ── Europe ──
    Country("United Kingdom", "GBR", "GB", -1.5, 52.4, _R_EU,
            aliases=("United Kingdom", "Great Britain", "Britain", "England", "Scotland",
                     "Wales", "Northern Ireland"),
            acronyms=("UK", "U.K."),
            capitals=("London",)),
    Country("Ireland", "IRL", "IE", -8.2, 53.4, _R_EU, aliases=("Ireland",), capitals=("Dublin",)),
    Country("France", "FRA", "FR", 2.2, 46.2, _R_EU, aliases=("France",), capitals=("Paris",)),
    Country("Germany", "DEU", "DE", 10.4, 51.2, _R_EU, aliases=("Germany",), capitals=("Berlin",)),
    Country("Italy", "ITA", "IT", 12.6, 41.9, _R_EU, aliases=("Italy",), capitals=("Rome",)),
    Country("Spain", "ESP", "ES", -3.7, 40.5, _R_EU, aliases=("Spain",), capitals=("Madrid",)),
    Country("Portugal", "PRT", "PT", -8.2, 39.4, _R_EU, aliases=("Portugal",), capitals=("Lisbon",)),
    Country("Netherlands", "NLD", "NL", 5.3, 52.1, _R_EU,
            aliases=("Netherlands", "the Netherlands"), capitals=("Amsterdam",)),
    Country("Belgium", "BEL", "BE", 4.5, 50.5, _R_EU, aliases=("Belgium",), capitals=("Brussels",)),
    Country("Switzerland", "CHE", "CH", 8.2, 46.8, _R_EU, aliases=("Switzerland",)),
    Country("Austria", "AUT", "AT", 14.6, 47.5, _R_EU, aliases=("Austria",), capitals=("Vienna",)),
    Country("Sweden", "SWE", "SE", 18.6, 60.1, _R_EU, aliases=("Sweden",), capitals=("Stockholm",)),
    Country("Norway", "NOR", "NO", 8.5, 60.5, _R_EU, aliases=("Norway",), capitals=("Oslo",)),
    Country("Denmark", "DNK", "DK", 9.5, 56.3, _R_EU, aliases=("Denmark",), capitals=("Copenhagen",)),
    Country("Finland", "FIN", "FI", 25.7, 61.9, _R_EU, aliases=("Finland",), capitals=("Helsinki",)),
    Country("Poland", "POL", "PL", 19.1, 51.9, _R_EU, aliases=("Poland",), capitals=("Warsaw",)),
    Country("Greece", "GRC", "GR", 21.8, 39.1, _R_EU, aliases=("Greece",), capitals=("Athens",)),
    Country("Russia", "RUS", "RU", 105.3, 61.5, _R_EU,
            aliases=("Russia", "Russian Federation", "Soviet Union"), capitals=("Moscow",)),
    Country("Ukraine", "UKR", "UA", 31.2, 48.4, _R_EU, aliases=("Ukraine",), capitals=("Kyiv", "Kiev")),
    Country("Turkey", "TUR", "TR", 35.2, 39.0, _R_EU,
            aliases=("Turkey", "Türkiye", "Turkiye"), capitals=("Ankara",)),
    # ── Africa / Middle East ──
    Country("South Africa", "ZAF", "ZA", 24.7, -29.0, _R_AF, aliases=("South Africa",)),
    Country("Egypt", "EGY", "EG", 30.8, 26.8, _R_AF, aliases=("Egypt",), capitals=("Cairo",)),
    Country("Nigeria", "NGA", "NG", 8.7, 9.1, _R_AF, aliases=("Nigeria",)),
    Country("Kenya", "KEN", "KE", 37.9, 0.5, _R_AF, aliases=("Kenya",), capitals=("Nairobi",)),
    Country("Ethiopia", "ETH", "ET", 40.5, 9.1, _R_AF, aliases=("Ethiopia",)),
    Country("Morocco", "MAR", "MA", -7.1, 31.8, _R_AF, aliases=("Morocco",)),
    Country("Algeria", "DZA", "DZ", 1.7, 28.0, _R_AF, aliases=("Algeria",)),
    Country("Ghana", "GHA", "GH", -1.0, 7.9, _R_AF, aliases=("Ghana",)),
    Country("Israel", "ISR", "IL", 34.9, 31.0, _R_AS, aliases=("Israel",)),
    Country("Saudi Arabia", "SAU", "SA", 45.1, 23.9, _R_AS, aliases=("Saudi Arabia",)),
    Country("United Arab Emirates", "ARE", "AE", 53.8, 23.4, _R_AS,
            aliases=("United Arab Emirates",), acronyms=("UAE", "U.A.E.")),
    Country("Iran", "IRN", "IR", 53.7, 32.4, _R_AS, aliases=("Iran",)),
    Country("Iraq", "IRQ", "IQ", 43.7, 33.2, _R_AS, aliases=("Iraq",)),
    Country("Qatar", "QAT", "QA", 51.2, 25.3, _R_AS, aliases=("Qatar",)),
    # ── Asia ──
    Country("China", "CHN", "CN", 104.2, 35.9, _R_AS, aliases=("China",), capitals=("Beijing",)),
    Country("India", "IND", "IN", 79.0, 22.0, _R_AS, aliases=("India",), capitals=("New Delhi", "Delhi")),
    Country("Japan", "JPN", "JP", 138.3, 36.2, _R_AS, aliases=("Japan",), capitals=("Tokyo",)),
    Country("South Korea", "KOR", "KR", 127.8, 36.5, _R_AS,
            aliases=("South Korea", "Republic of Korea", "Korea"), capitals=("Seoul",)),
    Country("North Korea", "PRK", "KP", 127.5, 40.3, _R_AS, aliases=("North Korea",)),
    Country("Indonesia", "IDN", "ID", 113.9, -0.8, _R_AS, aliases=("Indonesia",), capitals=("Jakarta",)),
    Country("Pakistan", "PAK", "PK", 69.3, 30.4, _R_AS, aliases=("Pakistan",)),
    Country("Bangladesh", "BGD", "BD", 90.4, 23.7, _R_AS, aliases=("Bangladesh",)),
    Country("Vietnam", "VNM", "VN", 108.3, 14.1, _R_AS, aliases=("Vietnam",)),
    Country("Thailand", "THA", "TH", 100.9, 15.9, _R_AS, aliases=("Thailand",)),
    Country("Malaysia", "MYS", "MY", 101.9, 4.2, _R_AS, aliases=("Malaysia",)),
    Country("Singapore", "SGP", "SG", 103.8, 1.4, _R_AS, aliases=("Singapore",)),
    Country("Philippines", "PHL", "PH", 122.9, 12.9, _R_AS, aliases=("Philippines", "the Philippines")),
    Country("Afghanistan", "AFG", "AF", 67.7, 33.9, _R_AS, aliases=("Afghanistan",)),
    Country("Sri Lanka", "LKA", "LK", 80.8, 7.9, _R_AS, aliases=("Sri Lanka",)),
    # ── Oceania ──
    Country("Australia", "AUS", "AU", 133.8, -25.3, _R_OC, aliases=("Australia",), capitals=("Canberra",)),
    Country("New Zealand", "NZL", "NZ", 174.0, -41.0, _R_OC, aliases=("New Zealand",), capitals=("Wellington",)),
)


def all_countries() -> tuple[Country, ...]:
    return _COUNTRIES


def regions() -> tuple[str, ...]:
    seen: list[str] = []
    for country in _COUNTRIES:
        if country.region not in seen:
            seen.append(country.region)
    return tuple(seen)


def country_by_iso2() -> dict[str, Country]:
    return {c.alpha2.lower(): c for c in _COUNTRIES}


@dataclass
class _Compiled:
    pattern: re.Pattern[str]
    country: str
    length: int
    require_capital: bool


class CountryMatcher:
    """Compile alias patterns once and detect country mentions in text."""

    def __init__(
        self,
        *,
        include_aliases: bool = False,
        include_acronyms: bool = False,
        include_capitals: bool = False,
        use_pycountry: bool = False,
    ) -> None:
        # Conservative defaults (Orange-like): only explicit canonical country
        # names are matched. Broad aliases (America, Britain, England, Scotland),
        # short acronyms (US, UK, UAE) and capital cities are each opt-in, because
        # on news corpora they appear in almost every article and massively inflate
        # per-country document counts.
        self._by_name: dict[str, Country] = {c.name: c for c in _COUNTRIES}
        self.include_aliases = include_aliases
        self.include_acronyms = include_acronyms
        self.include_capitals = include_capitals

        word_terms: list[tuple[str, str]] = []     # (alias, country) capital-required
        acronym_terms: list[tuple[str, str]] = []  # (acronym, country) case-sensitive

        for country in _COUNTRIES:
            # Canonical name always participates.
            word_terms.append((country.name, country.name))
            if include_aliases:
                for alias in country.aliases:
                    if alias != country.name:
                        word_terms.append((alias, country.name))
            if include_acronyms:
                for acronym in country.acronyms:
                    acronym_terms.append((acronym, country.name))
            if include_capitals:
                for capital in country.capitals:
                    word_terms.append((capital, country.name))

        if include_aliases and use_pycountry:
            self._enrich_with_pycountry(word_terms)

        compiled: list[_Compiled] = []
        seen_word: set[tuple[str, str]] = set()
        for alias, name in word_terms:
            key = (alias.lower(), name)
            if not alias or key in seen_word:
                continue
            seen_word.add(key)
            compiled.append(_Compiled(self._word_pattern(alias), name, len(alias), True))
        seen_ac: set[tuple[str, str]] = set()
        for acronym, name in acronym_terms:
            key = (acronym, name)
            if not acronym or key in seen_ac:
                continue
            seen_ac.add(key)
            compiled.append(_Compiled(self._acronym_pattern(acronym), name, len(acronym), False))
        compiled.sort(key=lambda c: c.length, reverse=True)
        self._compiled = compiled

    @staticmethod
    def _word_pattern(alias: str) -> re.Pattern[str]:
        body = r"\s+".join(re.escape(part) for part in alias.split())
        return re.compile(rf"(?<![0-9A-Za-z]){body}(?![0-9A-Za-z])", re.IGNORECASE)

    @staticmethod
    def _acronym_pattern(acronym: str) -> re.Pattern[str]:
        body = re.escape(acronym).rstrip("\\.")
        return re.compile(rf"(?<![0-9A-Za-z]){body}\.?(?![0-9A-Za-z])")

    def detect_with_forms(self, text: str) -> list[tuple[str, str]]:
        """Return ``(country_name, matched_surface_form)`` for each mention."""
        if not text:
            return []
        spans: list[tuple[int, int, str, str]] = []
        for compiled in self._compiled:
            for match in compiled.pattern.finditer(text):
                surface = match.group()
                if compiled.require_capital and not surface[:1].isupper():
                    continue  # lowercase common word — not a country mention
                spans.append((match.start(), match.end(), compiled.country, surface))
        if not spans:
            return []
        spans.sort(key=lambda s: (s[0], -(s[1] - s[0])))
        chosen: list[tuple[int, int, str, str]] = []
        last_end = -1
        for start, end, name, surface in spans:
            if start >= last_end:
                chosen.append((start, end, name, surface))
                last_end = end
        chosen.sort(key=lambda s: s[0])
        return [(name, surface) for _s, _e, name, surface in chosen]

    def detect(self, text: str) -> list[str]:
        return [name for name, _surface in self.detect_with_forms(text)]

    def country(self, name: str) -> Country | None:
        return self._by_name.get(name)

    def _enrich_with_pycountry(self, word_terms: list[tuple[str, str]]) -> None:
        try:
            import pycountry  # type: ignore
        except Exception:
            return
        for country in _COUNTRIES:
            try:
                record = pycountry.countries.get(alpha_3=country.alpha3)
            except Exception:
                record = None
            if record is None:
                continue
            for attr in ("name", "official_name", "common_name"):
                value = getattr(record, attr, None)
                if value:
                    word_terms.append((value, country.name))


_DEFAULT_MATCHER: CountryMatcher | None = None


def default_matcher() -> CountryMatcher:
    global _DEFAULT_MATCHER
    if _DEFAULT_MATCHER is None:
        _DEFAULT_MATCHER = CountryMatcher()
    return _DEFAULT_MATCHER


def detect_country_mentions(text: str) -> list[str]:
    return default_matcher().detect(text)


# ── Offline world polygon asset ────────────────────────────────────────────
_ASSET_PATH = Path(__file__).resolve().parent.parent / "assets" / "world_countries.json"


@lru_cache(maxsize=1)
def load_world_polygons() -> tuple[tuple[float, float, float, float], dict[str, list[list[tuple[float, float]]]]]:
    """Load the bundled world-countries polygons.

    Returns ``(view_box, polygons_by_iso2)`` where ``view_box`` is
    ``(min_x, min_y, width, height)`` and each value is a list of polygons, each a
    list of ``(x, y)`` points in the asset's projected coordinate space. Returns an
    empty mapping (never raises) if the asset is missing or unreadable.
    """
    try:
        data = json.loads(_ASSET_PATH.read_text(encoding="utf-8"))
    except Exception:
        return (0.0, 0.0, 1.0, 1.0), {}
    vb = data.get("viewBox", [0.0, 0.0, 1.0, 1.0])
    view_box = (float(vb[0]), float(vb[1]), float(vb[2]), float(vb[3]))
    polygons: dict[str, list[list[tuple[float, float]]]] = {}
    for iso2, polys in data.get("countries", {}).items():
        polygons[str(iso2).lower()] = [
            [(float(p[0]), float(p[1])) for p in poly] for poly in polys
        ]
    return view_box, polygons


def world_polygons_available() -> bool:
    _vb, polygons = load_world_polygons()
    return bool(polygons)
