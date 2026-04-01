from __future__ import annotations

from dataclasses import replace

import polars as pl

from portakal_app.data.models import DatasetHandle, build_data_domain


OPERATIONS = {
    "Sum": "sum",
    "Product": "product",
    "Min": "min",
    "Max": "max",
    "Mean": "mean",
    "Variance": "var",
    "Median": "median",
    "Count non-zero": "count_nonzero",
}


class AggregateColumnsService:
    def aggregate(
        self,
        dataset: DatasetHandle,
        *,
        columns: list[str],
        operation: str = "Mean",
        output_name: str = "agg",
    ) -> DatasetHandle:
        df = dataset.dataframe

        if not columns:
            return dataset

        numeric_cols = [c for c in columns if c in df.columns]
        if not numeric_cols:
            return dataset

        subset = df.select(numeric_cols)
        for col_name in subset.columns:
            col = subset.get_column(col_name)
            if not col.dtype.is_numeric():
                try:
                    subset = subset.with_columns(pl.col(col_name).cast(pl.Float64))
                except Exception:
                    subset = subset.with_columns(pl.lit(None).cast(pl.Float64).alias(col_name))

        op = OPERATIONS.get(operation, "mean")

        if op == "sum":
            result = subset.sum_horizontal()
        elif op == "min":
            result = subset.min_horizontal()
        elif op == "max":
            result = subset.max_horizontal()
        elif op == "mean":
            result = subset.mean_horizontal()
        elif op == "product":
            result = _product_horizontal(subset)
        elif op == "var":
            result = _variance_horizontal(subset)
        elif op == "median":
            result = _median_horizontal(subset)
        elif op == "count_nonzero":
            result = _count_nonzero_horizontal(subset)
        else:
            result = subset.mean_horizontal()

        result_df = df.with_columns(result.alias(output_name))

        return replace(
            dataset,
            dataset_id=f"{dataset.dataset_id}-aggregated",
            display_name=f"{dataset.display_name} (aggregated)",
            dataframe=result_df,
            row_count=result_df.height,
            column_count=result_df.width,
            domain=build_data_domain(result_df, source_domain=dataset.domain),
        )


def _product_horizontal(df: pl.DataFrame) -> pl.Series:
    result = pl.Series("product", [1.0] * df.height)
    for col_name in df.columns:
        result = result * df.get_column(col_name).cast(pl.Float64).fill_null(1.0)
    return result


def _variance_horizontal(df: pl.DataFrame) -> pl.Series:
    mean = df.mean_horizontal()
    n = len(df.columns)
    if n <= 1:
        return pl.Series("var", [0.0] * df.height)
    sq_diffs = []
    for col_name in df.columns:
        diff = df.get_column(col_name).cast(pl.Float64) - mean
        sq_diffs.append(diff * diff)
    total = sq_diffs[0]
    for s in sq_diffs[1:]:
        total = total + s
    return total / n


def _median_horizontal(df: pl.DataFrame) -> pl.Series:
    import statistics
    values = []
    for i in range(df.height):
        row_vals = []
        for col_name in df.columns:
            v = df.get_column(col_name)[i]
            if v is not None:
                row_vals.append(float(v))
        values.append(statistics.median(row_vals) if row_vals else None)
    return pl.Series("median", values, dtype=pl.Float64)


def _count_nonzero_horizontal(df: pl.DataFrame) -> pl.Series:
    values = []
    for i in range(df.height):
        count = 0
        for col_name in df.columns:
            v = df.get_column(col_name)[i]
            if v is not None and float(v) != 0.0:
                count += 1
        values.append(count)
    return pl.Series("count_nonzero", values, dtype=pl.Int64)
