from __future__ import annotations

from dataclasses import replace

import polars as pl

from portakal_app.data.models import DatasetHandle, build_data_domain


AGGREGATIONS = (
    "Mean", "Median", "Q1", "Q3", "Min. value", "Max. value", "Mode",
    "Standard deviation", "Variance", "Sum", "Concatenate", "Span",
    "First value", "Last value", "Random value", "Count defined", "Count",
    "Proportion defined",
)


class GroupByService:
    def group_by(
        self,
        dataset: DatasetHandle,
        *,
        group_columns: list[str],
        aggregations: dict[str, list[str]],
    ) -> DatasetHandle:
        df = dataset.dataframe

        if not group_columns:
            return dataset

        valid_group_cols = [c for c in group_columns if c in df.columns]
        if not valid_group_cols:
            return dataset

        agg_exprs: list[pl.Expr] = []
        for col_name, agg_list in aggregations.items():
            if col_name not in df.columns or col_name in valid_group_cols:
                continue
            for agg in agg_list:
                expr = _build_agg_expr(col_name, agg)
                if expr is not None:
                    agg_exprs.append(expr)

        if not agg_exprs:
            agg_exprs = [pl.len().alias("count")]

        result = df.group_by(valid_group_cols, maintain_order=True).agg(agg_exprs)

        return replace(
            dataset,
            dataset_id=f"{dataset.dataset_id}-grouped",
            display_name=f"{dataset.display_name} (grouped)",
            dataframe=result,
            row_count=result.height,
            column_count=result.width,
            domain=build_data_domain(result, source_domain=dataset.domain),
        )


def _build_agg_expr(col_name: str, agg: str) -> pl.Expr | None:
    # Use lowercase, replace spaces and dots for alias
    safe_agg_name = agg.lower().replace(" ", "_").replace(".", "")
    alias = f"{col_name}_{safe_agg_name}"
    col = pl.col(col_name)
    
    if agg == "Mean":
        return col.mean().round(10).alias(alias)
    if agg == "Median":
        return col.median().round(10).alias(alias)
    if agg == "Q1":
        return col.quantile(0.25).round(10).alias(alias)
    if agg == "Q3":
        return col.quantile(0.75).round(10).alias(alias)
    if agg == "Min. value":
        return col.min().alias(alias)
    if agg == "Max. value":
        return col.max().alias(alias)
    if agg == "Mode":
        return col.drop_nulls().mode().first().alias(alias)
    if agg == "Standard deviation":
        return col.std().round(10).alias(alias)
    if agg == "Variance":
        return col.var().round(10).alias(alias)
    if agg == "Sum":
        return col.sum().round(10).alias(alias)
    if agg == "Concatenate":
        # Convert all to string, drop nulls, and join with space
        return col.drop_nulls().cast(pl.String).str.join(" ").alias(alias)
    if agg == "Span":
        # Max minus Min
        return (col.max() - col.min()).alias(alias)
    if agg == "First value":
        return col.first().alias(alias)
    if agg == "Last value":
        return col.last().alias(alias)
    if agg == "Random value":
        # Shuffle with a fixed seed and take the first item
        return col.shuffle(seed=0).first().alias(alias)
    if agg == "Count defined":
        # Count non-nulls
        return col.count().alias(alias)
    if agg == "Count":
        # Count all (includes nulls)
        return pl.len().alias(alias)
    if agg == "Proportion defined":
        # Percentage of non-nulls
        return (col.count() / pl.len()).alias(alias)
    
    return None
