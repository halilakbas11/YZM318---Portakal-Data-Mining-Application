from __future__ import annotations

from collections import Counter
from datetime import datetime
from statistics import mean

from portakal_app.data.models import ColumnStatisticsResult, DatasetHandle, HistogramBin, ValueFrequency


class ColumnStatisticsService:
    HISTOGRAM_BINS = 12

    def describe(self, dataset: DatasetHandle, column_name: str) -> ColumnStatisticsResult:
        schema = next(column for column in dataset.domain.columns if column.name == column_name)
        series = dataset.dataframe.get_column(column_name)
        row_count = dataset.row_count
        missing_count = schema.null_count
        missing_ratio = missing_count / max(1, row_count)
        unique_count = schema.unique_count_hint
        clean_values = [value for value in series.to_list() if value is not None and str(value).strip() != ""]
        warning_tags = list(self._warning_tags(clean_values, row_count, missing_ratio, unique_count))

        metrics = [
            ("Type", schema.logical_type),
            ("Role", schema.role),
            ("Rows", str(row_count)),
            ("Missing", str(missing_count)),
            ("Missing %", f"{missing_ratio * 100:.1f}%"),
            ("Unique", str(unique_count)),
        ]

        histogram_bins: tuple[HistogramBin, ...] = ()
        top_values: tuple[ValueFrequency, ...] = ()

        if schema.logical_type == "numeric":
            numeric_values = [float(value) for value in clean_values]
            if numeric_values:
                q1, q3 = self._quartiles(numeric_values)
                iqr = q3 - q1
                lower = q1 - (1.5 * iqr)
                upper = q3 + (1.5 * iqr)
                outlier_count = sum(1 for value in numeric_values if value < lower or value > upper)
                metrics.extend(
                    [
                        ("Mean", f"{mean(numeric_values):.4f}"),
                        ("Median", f"{self._median(numeric_values):.4f}"),
                        ("Q1", f"{q1:.4f}"),
                        ("Q3", f"{q3:.4f}"),
                        ("IQR", f"{iqr:.4f}"),
                        ("Std", f"{self._std(numeric_values):.4f}"),
                        ("Min", f"{min(numeric_values):.4f}"),
                        ("Max", f"{max(numeric_values):.4f}"),
                        ("Outliers", str(outlier_count)),
                    ]
                )
                histogram_bins = self._build_histogram(numeric_values)
        elif schema.logical_type in {"date", "datetime", "time"} and clean_values:
            min_value = min(clean_values)
            max_value = max(clean_values)
            metrics.extend(
                [
                    ("Min", str(min_value)),
                    ("Max", str(max_value)),
                    ("Span", self._format_span(min_value, max_value)),
                ]
            )
        else:
            top_values = self._build_top_values(clean_values, row_count)
            if clean_values:
                avg_length = sum(len(str(value)) for value in clean_values) / len(clean_values)
                metrics.append(("Avg length", f"{avg_length:.2f}"))
            if top_values:
                top_ratio = top_values[0].ratio
                if top_ratio >= 0.95 and "near constant" not in warning_tags:
                    warning_tags.append("near constant")

        if not top_values:
            top_values = self._build_top_values(clean_values, row_count)

        return ColumnStatisticsResult(
            column_name=column_name,
            logical_type=schema.logical_type,
            role=schema.role,
            row_count=row_count,
            missing_count=missing_count,
            missing_ratio=missing_ratio,
            unique_count=unique_count,
            metrics=tuple(metrics),
            warning_tags=tuple(warning_tags),
            histogram_bins=histogram_bins,
            top_values=top_values,
        )

    def _warning_tags(self, clean_values: list[object], row_count: int, missing_ratio: float, unique_count: int) -> tuple[str, ...]:
        tags: list[str] = []
        if missing_ratio >= 0.3:
            tags.append("high missing")
        if row_count >= 20 and unique_count >= max(12, int(row_count * 0.8)):
            tags.append("high cardinality")
        if clean_values and len({str(value) for value in clean_values}) <= 1:
            tags.append("near constant")
        return tuple(tags)

    def _build_top_values(self, clean_values: list[object], row_count: int) -> tuple[ValueFrequency, ...]:
        counts = Counter(str(value) for value in clean_values)
        total = max(1, row_count)
        return tuple(
            ValueFrequency(value=value, count=count, ratio=count / total)
            for value, count in counts.most_common(8)
        )

    def _build_histogram(self, values: list[float]) -> tuple[HistogramBin, ...]:
        if not values:
            return ()
        minimum = min(values)
        maximum = max(values)
        if minimum == maximum:
            return (HistogramBin(label=f"{minimum:.2f}", count=len(values), fraction=1.0),)
        width = (maximum - minimum) / self.HISTOGRAM_BINS
        counts = [0] * self.HISTOGRAM_BINS
        for value in values:
            index = min(int((value - minimum) / width), self.HISTOGRAM_BINS - 1)
            counts[index] += 1
        total = len(values)
        bins = []
        for index, count in enumerate(counts):
            start = minimum + (index * width)
            end = start + width
            bins.append(HistogramBin(label=f"{start:.1f}-{end:.1f}", count=count, fraction=count / total))
        return tuple(bins)

    def _quartiles(self, values: list[float]) -> tuple[float, float]:
        ordered = sorted(values)
        return self._percentile(ordered, 25), self._percentile(ordered, 75)

    def _median(self, values: list[float]) -> float:
        return self._percentile(sorted(values), 50)

    def _percentile(self, ordered: list[float], percentile: int) -> float:
        if not ordered:
            return 0.0
        if len(ordered) == 1:
            return ordered[0]
        position = (len(ordered) - 1) * (percentile / 100)
        lower = int(position)
        upper = min(lower + 1, len(ordered) - 1)
        fraction = position - lower
        return ordered[lower] + ((ordered[upper] - ordered[lower]) * fraction)

    def _std(self, values: list[float]) -> float:
        if len(values) <= 1:
            return 0.0
        avg = mean(values)
        variance = sum((value - avg) ** 2 for value in values) / (len(values) - 1)
        return variance ** 0.5

    def _format_span(self, min_value: object, max_value: object) -> str:
        if isinstance(min_value, datetime) and isinstance(max_value, datetime):
            return str(max_value - min_value)
        return f"{min_value} -> {max_value}"
