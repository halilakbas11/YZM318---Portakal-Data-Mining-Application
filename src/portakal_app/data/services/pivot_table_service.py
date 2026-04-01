from __future__ import annotations

from dataclasses import replace

import polars as pl

from portakal_app.data.models import DatasetHandle, build_data_domain


PIVOT_AGGREGATIONS = ("Count", "Sum", "Mean", "Min", "Max", "Median")


class PivotTableService:
    def pivot(
        self,
        dataset: DatasetHandle,
        *,
        row_column: str,
        col_column: str,
        value_column: str | None = None,
        aggregation: str = "Count",
    ) -> DatasetHandle:
        df = dataset.dataframe

        if row_column not in df.columns or col_column not in df.columns:
            return dataset

        if value_column and value_column not in df.columns:
            value_column = None

        if aggregation == "Count" or value_column is None:
            grouped = df.group_by([row_column, col_column], maintain_order=True).agg(
                pl.len().alias("__count__")
            )
            result = grouped.pivot(
                on=col_column,
                index=row_column,
                values="__count__",
            )
        else:
            agg_expr = _get_agg_expr(value_column, aggregation)
            grouped = df.group_by([row_column, col_column], maintain_order=True).agg(agg_expr)
            alias_name = grouped.columns[-1]
            result = grouped.pivot(
                on=col_column,
                index=row_column,
                values=alias_name,
            )

        result = result.fill_null(0)

        return replace(
            dataset,
            dataset_id=f"{dataset.dataset_id}-pivot",
            display_name=f"{dataset.display_name} (pivot)",
            dataframe=result,
            row_count=result.height,
            column_count=result.width,
            domain=build_data_domain(result, source_domain=dataset.domain),
        )


def _get_agg_expr(col_name: str, agg: str) -> pl.Expr:
    col = pl.col(col_name)
    alias = f"__{agg.lower()}__"
    if agg == "Sum":
        return col.sum().round(10).alias(alias)
    if agg == "Mean":
        return col.mean().round(10).alias(alias)
    if agg == "Min":
        return col.min().alias(alias)
    if agg == "Max":
        return col.max().alias(alias)
    if agg == "Median":
        return col.median().round(10).alias(alias)
    return col.count().alias(alias)
