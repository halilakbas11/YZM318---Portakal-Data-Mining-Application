from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Any

import numpy as np
import polars as pl
from PySide6.QtGui import QColor

from portakal_app.data.models import ColumnSchema, DatasetHandle
from portakal_app.data.services.generated_dataset_service import GeneratedDatasetService
from portakal_app.models import WorkflowPayload


PALETTE = (
    QColor("#e07020"),
    QColor("#3b82f6"),
    QColor("#22c55e"),
    QColor("#a855f7"),
    QColor("#f43f5e"),
    QColor("#0ea5e9"),
    QColor("#f59e0b"),
    QColor("#10b981"),
)

GRADIENT_LOW = QColor("#3b82f6")
GRADIENT_MID = QColor("#f8fafc")
GRADIENT_HIGH = QColor("#ef4444")


@dataclass(frozen=True)
class PlotColumn:
    name: str
    logical_type: str
    row_indices: np.ndarray
    values: np.ndarray
    raw_values: tuple[Any, ...]
    is_discrete: bool
    categories: tuple[str, ...] = ()


@dataclass(frozen=True)
class CategoricalView:
    name: str
    labels: tuple[str, ...]
    categories: tuple[str, ...]
    was_discretized: bool = False


@dataclass(frozen=True)
class TreeNodeData:
    label: str
    summary: str = ""
    rule: str = ""
    children: tuple["TreeNodeData", ...] = ()
    row_indices: tuple[int, ...] = ()
    prediction: str = ""
    distribution: tuple[float, ...] = ()

    @classmethod
    def from_object(cls, value: object) -> "TreeNodeData | None":
        if isinstance(value, TreeNodeData):
            return value
        if value is None:
            return None
        if isinstance(value, dict):
            source = value
        else:
            source = {
                key: getattr(value, key)
                for key in (
                    "label",
                    "name",
                    "title",
                    "summary",
                    "description",
                    "rule",
                    "condition",
                    "children",
                    "branches",
                    "row_indices",
                    "rows",
                    "prediction",
                    "value",
                    "distribution",
                    "dist",
                    "n_samples",
                    "samples",
                )
                if hasattr(value, key)
            }
            if not source:
                return None
        children_source = source.get("children", source.get("branches", ()))
        if callable(children_source):
            children_source = children_source()
        children = tuple(
            child
            for item in children_source or ()
            if (child := cls.from_object(item)) is not None
        )
        summary = source.get("summary") or source.get("description") or ""
        if not summary:
            sample_count = source.get("n_samples", source.get("samples", ""))
            if isinstance(sample_count, int | float):
                summary = f"{int(sample_count)} rows"
        return cls(
            label=str(source.get("label") or source.get("name") or source.get("title") or "Node"),
            summary=str(summary),
            rule=str(source.get("rule") or source.get("condition") or ""),
            children=children,
            row_indices=tuple(
                int(index)
                for index in source.get("row_indices", source.get("rows", ()))
                if isinstance(index, int | float)
            ),
            prediction=str(source.get("prediction") or source.get("value") or ""),
            distribution=tuple(
                float(item)
                for item in source.get("distribution", source.get("dist", ()))
                if isinstance(item, int | float)
            ),
        )

    def descendant_rows(self) -> list[int]:
        collected = list(self.row_indices)
        for child in self.children:
            collected.extend(child.descendant_rows())
        seen: set[int] = set()
        ordered: list[int] = []
        for index in collected:
            if index not in seen:
                seen.add(index)
                ordered.append(index)
        return ordered


def source_column(dataset: DatasetHandle | None, name: str | None) -> ColumnSchema | None:
    if dataset is None or not name:
        return None
    return next((column for column in dataset.domain.columns if column.name == name), None)


def numeric_columns(dataset: DatasetHandle | None) -> list[str]:
    if dataset is None:
        return []
    return [column.name for column in dataset.domain.columns if column.logical_type == "numeric"]


def discrete_columns(dataset: DatasetHandle | None) -> list[str]:
    if dataset is None:
        return []
    return [
        column.name
        for column in dataset.domain.columns
        if column.logical_type in {"categorical", "boolean", "string", "text"}
    ]


def primitive_columns(dataset: DatasetHandle | None) -> list[str]:
    if dataset is None:
        return []
    return [
        column.name
        for column in dataset.domain.columns
        if column.logical_type in {"numeric", "categorical", "boolean", "string", "text"}
    ]


def categorical_candidate_columns(dataset: DatasetHandle | None) -> list[str]:
    if dataset is None:
        return []
    result: list[str] = []
    for column in dataset.domain.columns:
        if column.logical_type == "numeric":
            result.append(column.name)
            continue
        if column.logical_type in {"categorical", "boolean"}:
            result.append(column.name)
            continue
        if column.unique_count_hint <= 40:
            result.append(column.name)
    return result or primitive_columns(dataset)


def categorical_view(
    dataset: DatasetHandle | None,
    name: str | None,
    *,
    bins: int = 4,
    discretize_numeric: bool = True,
) -> CategoricalView | None:
    if dataset is None or not name or name not in dataset.dataframe.columns:
        return None
    column = source_column(dataset, name)
    if column is None:
        return None

    series = dataset.dataframe.get_column(name)
    raw_values = series.to_list()
    if column.logical_type == "numeric" and discretize_numeric:
        numeric = series.cast(pl.Float64, strict=False).to_numpy()
        valid = numeric[np.isfinite(numeric)]
        if valid.size:
            labels = _bin_numeric_labels(numeric, bins=max(2, int(bins)))
            categories = _ordered_categories(labels)
            return CategoricalView(
                name=name,
                labels=tuple(labels),
                categories=categories,
                was_discretized=True,
            )

    labels = tuple("(missing)" if value is None else str(value) for value in raw_values)
    return CategoricalView(
        name=name,
        labels=labels,
        categories=_ordered_categories(labels),
        was_discretized=False,
    )


def prepared_column(dataset: DatasetHandle | None, name: str | None) -> PlotColumn | None:
    if dataset is None or not name or name not in dataset.dataframe.columns:
        return None
    column = source_column(dataset, name)
    if column is None:
        return None

    series = dataset.dataframe.get_column(name)
    raw_values = series.to_list()
    if column.logical_type == "numeric":
        numeric = series.cast(pl.Float64, strict=False).to_numpy()
        mask = np.isfinite(numeric)
        return PlotColumn(
            name=name,
            logical_type=column.logical_type,
            row_indices=np.flatnonzero(mask).astype(int),
            values=numeric[mask].astype(float),
            raw_values=tuple(raw_values[index] for index in np.flatnonzero(mask)),
            is_discrete=False,
            categories=(),
        )

    labels: list[str] = []
    for value in raw_values:
        if value is None:
            continue
        text = str(value)
        if text not in labels:
            labels.append(text)
    mapping = {value: index for index, value in enumerate(labels)}

    row_indices: list[int] = []
    encoded: list[float] = []
    filtered_raw: list[Any] = []
    for index, value in enumerate(raw_values):
        if value is None:
            continue
        text = str(value)
        row_indices.append(index)
        encoded.append(float(mapping[text]))
        filtered_raw.append(value)

    return PlotColumn(
        name=name,
        logical_type=column.logical_type,
        row_indices=np.asarray(row_indices, dtype=int),
        values=np.asarray(encoded, dtype=float),
        raw_values=tuple(filtered_raw),
        is_discrete=True,
        categories=tuple(labels),
    )


def shared_numeric_rows(dataset: DatasetHandle | None, columns: list[str]) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    if dataset is None or not columns:
        return np.asarray([], dtype=int), {}

    valid_mask = np.ones(dataset.row_count, dtype=bool)
    values_by_column: dict[str, np.ndarray] = {}
    for name in columns:
        if name not in dataset.dataframe.columns:
            return np.asarray([], dtype=int), {}
        data = dataset.dataframe.get_column(name).cast(pl.Float64, strict=False).to_numpy()
        values_by_column[name] = data.astype(float, copy=False)
        valid_mask &= np.isfinite(data)

    row_indices = np.flatnonzero(valid_mask).astype(int)
    return row_indices, {name: values[row_indices] for name, values in values_by_column.items()}


def discrete_colors(values: np.ndarray) -> list[QColor]:
    colors: list[QColor] = []
    for value in values:
        index = int(value) if math.isfinite(float(value)) else 0
        colors.append(PALETTE[index % len(PALETTE)])
    return colors


def gradient_color(value: float, low: float, high: float) -> QColor:
    if not math.isfinite(value):
        return QColor("#9ca3af")
    span = high - low
    if abs(span) < 1e-12:
        t = 0.5
    else:
        t = max(0.0, min(1.0, (value - low) / span))
    if t < 0.5:
        local = t / 0.5
        return QColor(
            int(GRADIENT_LOW.red() + (GRADIENT_MID.red() - GRADIENT_LOW.red()) * local),
            int(GRADIENT_LOW.green() + (GRADIENT_MID.green() - GRADIENT_LOW.green()) * local),
            int(GRADIENT_LOW.blue() + (GRADIENT_MID.blue() - GRADIENT_LOW.blue()) * local),
        )
    local = (t - 0.5) / 0.5
    return QColor(
        int(GRADIENT_MID.red() + (GRADIENT_HIGH.red() - GRADIENT_MID.red()) * local),
        int(GRADIENT_MID.green() + (GRADIENT_HIGH.green() - GRADIENT_MID.green()) * local),
        int(GRADIENT_MID.blue() + (GRADIENT_HIGH.blue() - GRADIENT_MID.blue()) * local),
    )


def subset_dataframe(dataset: DatasetHandle, rows: list[int]) -> pl.DataFrame:
    if not rows:
        return dataset.dataframe.head(0)
    normalized = sorted({index for index in rows if 0 <= index < dataset.row_count})
    indexed = dataset.dataframe.with_row_index("__portakal_row__")
    return indexed.filter(pl.col("__portakal_row__").is_in(normalized)).drop("__portakal_row__")


def build_selection_outputs(
    dataset: DatasetHandle | None,
    rows: list[int],
    *,
    generated_by: str,
    service: GeneratedDatasetService | None = None,
) -> tuple[DatasetHandle | None, DatasetHandle | None]:
    if dataset is None:
        return None, None

    normalized = sorted({index for index in rows if 0 <= index < dataset.row_count})
    annotations = dict(dataset.annotations)
    annotations.update(
        {
            "generated_by": generated_by,
            "selected_rows": normalized,
        }
    )
    annotated = replace(
        dataset,
        dataset_id=f"{dataset.dataset_id}-{generated_by}-annotated",
        display_name=f"{dataset.display_name} (annotated)",
        annotations=annotations,
    )
    if not normalized:
        return None, annotated

    builder = service or GeneratedDatasetService()
    selected_df = subset_dataframe(dataset, normalized)
    role_overrides = {column.name: column.role for column in dataset.domain.columns}
    selected = builder.build_dataset(
        selected_df,
        dataset_id=f"{dataset.dataset_id}-{generated_by}-selected",
        display_name=f"{dataset.display_name} (selected)",
        file_name=f"{dataset.dataset_id}-{generated_by}-selected.csv",
        role_overrides=role_overrides,
        annotations=annotations,
    )
    return selected, annotated


def subset_row_indices(dataset: DatasetHandle | None, subset: DatasetHandle | None) -> set[int]:
    if dataset is None or subset is None:
        return set()
    indexed = dataset.dataframe.with_row_index("__row__")
    shared = [column for column in dataset.dataframe.columns if column in subset.dataframe.columns]
    if not shared:
        return set()
    matches = indexed.join(subset.dataframe.select(shared).unique(), on=shared, how="semi")
    return {int(value) for value in matches.get_column("__row__").to_list()}


def _ordered_categories(labels: tuple[str, ...] | list[str]) -> tuple[str, ...]:
    seen: set[str] = set()
    ordered: list[str] = []
    for label in labels:
        if label not in seen:
            seen.add(label)
            ordered.append(label)
    return tuple(ordered)


def _bin_numeric_labels(values: np.ndarray, bins: int) -> list[str]:
    valid = values[np.isfinite(values)]
    if valid.size == 0:
        return ["(missing)"] * len(values)
    if np.nanmin(valid) == np.nanmax(valid):
        label = f"{float(valid[0]):.3g}"
        return [label if math.isfinite(value) else "(missing)" for value in values]

    quantiles = np.linspace(0.0, 1.0, bins + 1)
    edges = np.quantile(valid, quantiles)
    edges = np.unique(edges.astype(float))
    if edges.size <= 1:
        label = f"{float(valid[0]):.3g}"
        return [label if math.isfinite(value) else "(missing)" for value in values]

    boundaries = edges[1:-1]
    labels_for_bins: list[str] = []
    for index in range(edges.size - 1):
        low = edges[index]
        high = edges[index + 1]
        if index == 0:
            labels_for_bins.append(f"<= {high:.3g}")
        elif index == edges.size - 2:
            labels_for_bins.append(f"> {low:.3g}")
        else:
            labels_for_bins.append(f"({low:.3g}, {high:.3g}]")

    binned: list[str] = []
    for value in values:
        if not math.isfinite(value):
            binned.append("(missing)")
            continue
        index = int(np.searchsorted(boundaries, float(value), side="right"))
        index = max(0, min(index, len(labels_for_bins) - 1))
        binned.append(labels_for_bins[index])
    return binned


def build_components_dataset(
    names: list[str],
    x_values: list[float],
    y_values: list[float],
    *,
    dataset_id: str,
    display_name: str,
    file_name: str,
    service: GeneratedDatasetService | None = None,
) -> DatasetHandle | None:
    if not names or len(names) != len(x_values) or len(names) != len(y_values):
        return None
    builder = service or GeneratedDatasetService()
    dataframe = pl.DataFrame(
        {
            "feature": names,
            "component_1": [float(value) for value in x_values],
            "component_2": [float(value) for value in y_values],
        }
    )
    return builder.build_dataset(
        dataframe,
        dataset_id=dataset_id,
        display_name=display_name,
        file_name=file_name,
        role_overrides={"feature": "meta", "component_1": "feature", "component_2": "feature"},
        annotations={"generated_by": "components"},
    )


def feature_names_from_payload(payload: WorkflowPayload | None) -> tuple[str, ...]:
    if payload is None:
        return ()
    value = payload.value
    if isinstance(value, (list, tuple)):
        names = [str(item) for item in value if str(item).strip()]
        return tuple(names)
    return ()


def feature_output_payload(names: list[str], *, port_label: str = "Features") -> WorkflowPayload | None:
    normalized = [name for name in names if name]
    if not normalized:
        return None
    return WorkflowPayload(port_label, tuple(normalized))


def nice_ticks(vmin: float, vmax: float, count: int = 5) -> list[float]:
    if not math.isfinite(vmin) or not math.isfinite(vmax):
        return [0.0]
    if abs(vmax - vmin) < 1e-12:
        return [vmin]
    span = abs(vmax - vmin)
    raw_step = span / max(1, count)
    magnitude = 10 ** math.floor(math.log10(raw_step))
    fraction = raw_step / magnitude
    if fraction <= 1:
        step = magnitude
    elif fraction <= 2:
        step = 2 * magnitude
    elif fraction <= 2.5:
        step = 2.5 * magnitude
    elif fraction <= 5:
        step = 5 * magnitude
    else:
        step = 10 * magnitude

    start = math.floor(vmin / step) * step
    end = math.ceil(vmax / step) * step
    ticks: list[float] = []
    value = start
    while value <= end + step * 0.01:
        ticks.append(round(value, 10))
        value += step
    return ticks


def kernel_density(
    values: np.ndarray,
    *,
    points: int = 160,
    kernel: str = "gaussian",
) -> tuple[np.ndarray, np.ndarray]:
    if values.size == 0:
        return np.asarray([]), np.asarray([])
    values = np.asarray(values, dtype=float)
    if values.size == 1 or float(np.nanstd(values)) < 1e-12:
        center = float(values[0])
        support = np.linspace(center - 1.0, center + 1.0, points)
        scaled = (support - center) / 0.15
        density = np.exp(-0.5 * scaled * scaled)
        return support, density / max(float(np.max(density)), 1.0)

    std = float(np.nanstd(values, ddof=1))
    q75, q25 = np.nanpercentile(values, [75, 25])
    iqr = float(q75 - q25)
    sigma = min(std, iqr / 1.34) if iqr > 0 else std
    bandwidth = 0.9 * sigma * (values.size ** (-1.0 / 5.0))
    bandwidth = max(bandwidth, std * 0.1, 1e-3)

    low = float(np.nanmin(values)) - bandwidth * 2.5
    high = float(np.nanmax(values)) + bandwidth * 2.5
    support = np.linspace(low, high, points)
    scaled = (support[:, None] - values[None, :]) / bandwidth
    if kernel == "epanechnikov":
        weights = np.where(np.abs(scaled) <= 1.0, 0.75 * (1.0 - scaled * scaled), 0.0)
        density = weights.sum(axis=1) / max(values.size * bandwidth, 1e-9)
    elif kernel == "linear":
        weights = np.where(np.abs(scaled) <= 1.0, 1.0 - np.abs(scaled), 0.0)
        density = weights.sum(axis=1) / max(values.size * bandwidth, 1e-9)
    else:
        density = np.exp(-0.5 * scaled * scaled).sum(axis=1)
        density /= max(values.size * bandwidth * math.sqrt(2 * math.pi), 1e-9)
    return support, density


def suggest_numeric_features(
    dataset: DatasetHandle | None,
    *,
    class_name: str | None = None,
    limit: int = 4,
) -> list[str]:
    if dataset is None:
        return []
    candidates = numeric_columns(dataset)
    if not candidates:
        return []
    limit = max(1, min(limit, len(candidates)))
    labels = _class_labels(dataset, class_name)
    scores: list[tuple[float, str]] = []
    for name in candidates:
        column = prepared_column(dataset, name)
        if column is None or len(column.row_indices) < 2:
            continue
        if labels is not None:
            aligned = [(float(value), labels.get(int(row))) for row, value in zip(column.row_indices, column.values)]
            grouped: dict[str, list[float]] = {}
            for value, label in aligned:
                if label is None:
                    continue
                grouped.setdefault(label, []).append(value)
            score = _anova_score(grouped) if len(grouped) >= 2 else float(np.nanvar(column.values))
        else:
            score = float(np.nanvar(column.values))
        scores.append((score, name))
    scores.sort(key=lambda item: (-item[0], item[1]))
    return [name for _score, name in scores[:limit]]


def suggest_scatter_pair(
    dataset: DatasetHandle | None,
    *,
    class_name: str | None = None,
) -> tuple[str, str] | None:
    ranked = rank_scatter_pairs(dataset, class_name=class_name, limit=1)
    if not ranked:
        return None
    return ranked[0][:2]


def rank_scatter_pairs(
    dataset: DatasetHandle | None,
    *,
    class_name: str | None = None,
    limit: int | None = None,
) -> list[tuple[str, str, float]]:
    if dataset is None:
        return []
    candidates = numeric_columns(dataset)
    if len(candidates) < 2:
        return []
    labels = _class_labels(dataset, class_name)
    ranked: list[tuple[str, str, float]] = []
    for i, x_name in enumerate(candidates):
        x_col = prepared_column(dataset, x_name)
        if x_col is None:
            continue
        x_lookup = {int(row): float(value) for row, value in zip(x_col.row_indices, x_col.values)}
        for y_name in candidates[i + 1:]:
            y_col = prepared_column(dataset, y_name)
            if y_col is None:
                continue
            y_lookup = {int(row): float(value) for row, value in zip(y_col.row_indices, y_col.values)}
            shared_rows = sorted(set(x_lookup).intersection(y_lookup))
            if len(shared_rows) < 3:
                continue
            points = np.asarray([[x_lookup[row], y_lookup[row]] for row in shared_rows], dtype=float)
            if labels is not None:
                grouped: dict[str, list[np.ndarray]] = {}
                for row, point in zip(shared_rows, points):
                    label = labels.get(int(row))
                    if label is None:
                        continue
                    grouped.setdefault(label, []).append(point)
                score = _pair_class_separation(grouped) if len(grouped) >= 2 else _pair_variance(points)
            else:
                score = _pair_variance(points)
            ranked.append((x_name, y_name, float(score)))
    ranked.sort(key=lambda item: (-item[2], item[0], item[1]))
    if limit is not None:
        return ranked[: max(0, int(limit))]
    return ranked


def _class_labels(dataset: DatasetHandle, class_name: str | None) -> dict[int, str] | None:
    target = class_name
    if not target:
        for column in dataset.domain.target_columns:
            if column.name in dataset.dataframe.columns:
                target = column.name
                break
    if not target or target not in dataset.dataframe.columns:
        return None
    series = dataset.dataframe.get_column(target)
    return {
        index: str(value)
        for index, value in enumerate(series.to_list())
        if value is not None
    }


def _anova_score(grouped: dict[str, list[float]]) -> float:
    samples = [np.asarray(values, dtype=float) for values in grouped.values() if len(values) >= 1]
    if len(samples) < 2:
        return 0.0
    overall = np.concatenate(samples)
    grand_mean = float(np.mean(overall))
    between = sum(len(sample) * (float(np.mean(sample)) - grand_mean) ** 2 for sample in samples)
    within = sum(float(np.sum((sample - float(np.mean(sample))) ** 2)) for sample in samples)
    if within <= 1e-12:
        return between
    return between / within


def _pair_variance(points: np.ndarray) -> float:
    if len(points) < 2:
        return 0.0
    cov = np.cov(points.T)
    if cov.shape != (2, 2) or not np.isfinite(cov).all():
        return 0.0
    return float(np.trace(cov))


def _pair_class_separation(grouped: dict[str, list[np.ndarray]]) -> float:
    samples = [np.asarray(points, dtype=float) for points in grouped.values() if len(points) >= 1]
    if len(samples) < 2:
        return 0.0
    overall = np.vstack(samples)
    grand = np.mean(overall, axis=0)
    between = 0.0
    within = 0.0
    for sample in samples:
        mean = np.mean(sample, axis=0)
        between += len(sample) * float(np.sum((mean - grand) ** 2))
        within += float(np.sum((sample - mean) ** 2))
    if within <= 1e-12:
        return between
    return between / within
