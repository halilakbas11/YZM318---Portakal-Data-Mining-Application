from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import polars as pl


LOW_CARDINALITY_LIMIT = 12


@dataclass(frozen=True)
class SourceInfo:
    path: Path
    format: str
    size_bytes: int
    modified_at: datetime
    cache_path: Path


@dataclass(frozen=True)
class CSVImportOptions:
    delimiter: str = ","
    has_header: bool = True
    encoding: str = "auto"
    skip_rows: int = 0
    auto_detect_delimiter: bool = False


@dataclass(frozen=True)
class ColumnSchema:
    name: str
    dtype_repr: str
    logical_type: str
    role: str
    nullable: bool
    null_count: int
    unique_count_hint: int
    sample_values: tuple[str, ...] = ()


@dataclass(frozen=True)
class DataDomain:
    columns: tuple[ColumnSchema, ...]

    @property
    def feature_columns(self) -> tuple[ColumnSchema, ...]:
        return tuple(column for column in self.columns if column.role == "feature")

    @property
    def target_columns(self) -> tuple[ColumnSchema, ...]:
        return tuple(column for column in self.columns if column.role == "target")

    @property
    def meta_columns(self) -> tuple[ColumnSchema, ...]:
        return tuple(column for column in self.columns if column.role == "meta")


@dataclass(frozen=True)
class DatasetHandle:
    dataset_id: str
    display_name: str
    source: SourceInfo
    domain: DataDomain
    dataframe: pl.DataFrame
    row_count: int
    column_count: int
    cache_path: Path
    annotations: dict[str, object] = field(default_factory=dict)

    @property
    def path(self) -> Path:
        return self.source.path


@dataclass(frozen=True)
class PreviewPage:
    headers: tuple[str, ...] = ()
    rows: tuple[tuple[str, ...], ...] = ()
    offset: int = 0
    limit: int = 0
    total_rows: int = 0


@dataclass(frozen=True)
class ColumnProfile:
    column_name: str
    logical_type: str
    role: str
    null_count: int = 0
    null_ratio: float = 0.0
    unique_count_hint: int = 0
    sample_values: tuple[str, ...] = ()
    summary: str = ""


@dataclass(frozen=True)
class DatasetSummary:
    row_count: int = 0
    column_count: int = 0
    missing_value_count: int = 0
    missing_ratio: float = 0.0
    duplicate_row_count: int = 0
    feature_count: int = 0
    target_count: int = 0
    dtype_counts: dict[str, int] = field(default_factory=dict)
    column_profiles: tuple[ColumnProfile, ...] = ()


@dataclass(frozen=True)
class AnalysisSuggestion:
    title: str
    body: str
    kind: str = "suggestion"
    severity: str = "medium"


@dataclass(frozen=True)
class HistogramBin:
    label: str
    count: int
    fraction: float = 0.0


@dataclass(frozen=True)
class ValueFrequency:
    value: str
    count: int
    ratio: float


@dataclass(frozen=True)
class ColumnStatisticsResult:
    column_name: str
    logical_type: str
    role: str
    row_count: int
    missing_count: int
    missing_ratio: float
    unique_count: int
    metrics: tuple[tuple[str, str], ...] = ()
    warning_tags: tuple[str, ...] = ()
    histogram_bins: tuple[HistogramBin, ...] = ()
    top_values: tuple[ValueFrequency, ...] = ()


@dataclass(frozen=True)
class RankedFeature:
    feature_name: str
    logical_type: str
    score: float
    method: str
    details: str
    target_name: str | None = None


@dataclass(frozen=True)
class DomainColumnEdit:
    original_name: str
    new_name: str
    logical_type: str
    role: str


@dataclass(frozen=True)
class DomainEditRequest:
    columns: tuple[DomainColumnEdit, ...] = ()


@dataclass(frozen=True)
class DatasetCatalogEntry:
    dataset_id: str
    title: str
    description: str
    domain: str
    target: str
    tags: tuple[str, ...]
    download_url: str
    size_text: str
    row_count: int
    column_count: int
    format: str = "csv"


@dataclass(frozen=True)
class PaintDataPoint:
    x: float
    y: float
    label: str


@dataclass(frozen=True)
class PaintDataSnapshot:
    x_name: str = "x"
    y_name: str = "y"
    label_name: str = "class"
    labels: tuple[str, ...] = ("C1", "C2")
    points: tuple[PaintDataPoint, ...] = ()
    source_name: str = "Painted Data"


def build_data_domain(dataframe: pl.DataFrame) -> DataDomain:
    columns: list[ColumnSchema] = []
    for name in dataframe.columns:
        series = dataframe.get_column(name)
        dtype_repr = str(series.dtype)
        logical_type = _infer_logical_type(series)
        null_count = int(series.null_count())
        unique_count = _safe_unique_count(series)
        sample_values = _sample_values(series)
        columns.append(
            ColumnSchema(
                name=name,
                dtype_repr=dtype_repr,
                logical_type=logical_type,
                role="feature",
                nullable=null_count > 0,
                null_count=null_count,
                unique_count_hint=unique_count,
                sample_values=sample_values,
            )
        )

    target_index = _infer_target_index(columns)
    if target_index is not None:
        target_column = columns[target_index]
        columns[target_index] = ColumnSchema(
            name=target_column.name,
            dtype_repr=target_column.dtype_repr,
            logical_type=target_column.logical_type,
            role="target",
            nullable=target_column.nullable,
            null_count=target_column.null_count,
            unique_count_hint=target_column.unique_count_hint,
            sample_values=target_column.sample_values,
        )
    return DataDomain(columns=tuple(columns))


def _infer_target_index(columns: list[ColumnSchema]) -> int | None:
    candidates = [
        index
        for index, column in enumerate(columns)
        if column.logical_type != "numeric" and 0 < column.unique_count_hint <= LOW_CARDINALITY_LIMIT
    ]
    if not candidates:
        return None
    return candidates[-1]


def _infer_logical_type(series: pl.Series) -> str:
    dtype_name = str(series.dtype).lower()
    if "bool" in dtype_name:
        return "boolean"
    if dtype_name == "date":
        return "date"
    if dtype_name == "time":
        return "time"
    if "datetime" in dtype_name:
        return "datetime"
    if "duration" in dtype_name:
        return "duration"
    if any(token in dtype_name for token in ("int", "float", "decimal")):
        return "numeric"
    if dtype_name in {"categorical", "enum"}:
        return "categorical"
    if dtype_name in {"string", "str", "utf8"}:
        return _infer_string_like_type(series)
    return "unknown"


def _infer_string_like_type(series: pl.Series) -> str:
    values = [str(value).strip() for value in series.drop_nulls().head(128).to_list() if str(value).strip()]
    if not values:
        return "text"
    if all(_looks_like_datetime(value) for value in values[:8]):
        return "datetime"
    unique_count = len({value for value in values})
    if unique_count <= max(LOW_CARDINALITY_LIMIT, len(values) // 3):
        return "categorical"
    return "text"


def _looks_like_datetime(value: str) -> bool:
    return any(token in value for token in ("-", "/", ":"))


def _safe_unique_count(series: pl.Series) -> int:
    try:
        value = series.n_unique()
    except Exception:
        return 0
    if value is None:
        return 0
    return int(value)


def _sample_values(series: pl.Series) -> tuple[str, ...]:
    preview: list[str] = []
    for value in series.drop_nulls().head(8).to_list():
        text = str(value).strip()
        if not text or text in preview:
            continue
        preview.append(text)
        if len(preview) == 3:
            break
    return tuple(preview)
