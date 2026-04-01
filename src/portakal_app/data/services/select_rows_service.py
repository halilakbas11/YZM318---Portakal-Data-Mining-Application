from __future__ import annotations

from dataclasses import replace

import polars as pl

from portakal_app.data.models import DatasetHandle, build_data_domain


# ── Operator definitions (aligned with Orange Data Mining) ────────────────

OPERATORS_NUMERIC = (
    "equals", "is not",
    "is below", "is at most",
    "is greater than", "is at least",
    "is between", "is outside",
    "is defined", "is not defined",
)

OPERATORS_CATEGORICAL = (
    "is", "is not", "is one of",
    "is defined", "is not defined",
)

OPERATORS_STRING = (
    "equals", "is not",
    "contains", "does not contain",
    "begins with", "ends with",
    "is defined", "is not defined",
)

# Operators that need two value inputs
DUAL_VALUE_OPS = ("is between", "is outside")

# Operators that need no value input
NO_VALUE_OPS = ("is defined", "is not defined")


class SelectRowsService:
    def filter_rows(
        self,
        dataset: DatasetHandle,
        *,
        conditions: list[tuple[str, str, str]],
        conjunction: str = "all",
        purge_attributes: bool = False,
        purge_classes: bool = False,
    ) -> tuple[DatasetHandle | None, DatasetHandle | None]:
        """Filter rows by conditions.

        Parameters
        ----------
        conjunction : "all" (AND) or "any" (OR)
        purge_attributes : remove unused values and constant feature columns
        purge_classes : remove unused values and constant class/target columns
        """
        df = dataset.dataframe

        if not conditions:
            return dataset, None

        condition_masks = []
        for col_name, operator, value in conditions:
            if col_name not in df.columns:
                continue
            condition_masks.append(_apply_condition(df, col_name, operator, value))

        if not condition_masks:
            return dataset, None

        # Combine with AND or OR
        combined = condition_masks[0]
        for m in condition_masks[1:]:
            if conjunction == "any":
                combined = combined | m
            else:
                combined = combined & m

        matching_df = df.filter(combined)
        non_matching_df = df.filter(~combined)

        # ── Purge unused values / constant columns (Orange compatibility) ──
        if purge_attributes or purge_classes:
            matching_df = _purge_domain(
                matching_df, dataset.domain, purge_attributes, purge_classes
            )
            non_matching_df = _purge_domain(
                non_matching_df, dataset.domain, purge_attributes, purge_classes
            )

        matching = None
        if matching_df.height > 0:
            matching = replace(
                dataset,
                dataset_id=f"{dataset.dataset_id}-matching",
                display_name=f"{dataset.display_name} (matching)",
                dataframe=matching_df,
                row_count=matching_df.height,
                domain=build_data_domain(matching_df, source_domain=dataset.domain),
            )

        non_matching = None
        if non_matching_df.height > 0:
            non_matching = replace(
                dataset,
                dataset_id=f"{dataset.dataset_id}-unmatched",
                display_name=f"{dataset.display_name} (unmatched)",
                dataframe=non_matching_df,
                row_count=non_matching_df.height,
                domain=build_data_domain(non_matching_df, source_domain=dataset.domain),
            )

        return matching, non_matching


def _apply_condition(
    df: pl.DataFrame, col_name: str, operator: str, value: str,
) -> pl.Series:
    series = df.get_column(col_name)
    n = df.height

    # ── No-value operators ────────────────────────────────────────────
    if operator == "is defined":
        return series.is_not_null()
    if operator == "is not defined":
        return series.is_null()

    # ── Numeric operators ─────────────────────────────────────────────
    if series.dtype.is_numeric():
        if operator in ("is between", "is outside"):
            return _apply_range_op(series, operator, value, n)

        try:
            num_val = float(value)
        except (ValueError, TypeError):
            return pl.Series("m", [True] * n)

        float_series = series.cast(pl.Float64, strict=False)
        if operator in ("equals", "=", "=="):
            return float_series == num_val
        if operator in ("is not", "!="):
            return float_series != num_val
        if operator in ("is below", "<"):
            return float_series < num_val
        if operator in ("is at most", "<="):
            return float_series <= num_val
        if operator in ("is greater than", ">"):
            return float_series > num_val
        if operator in ("is at least", ">="):
            return float_series >= num_val

    # ── Categorical operators ─────────────────────────────────────────
    str_series = series.cast(pl.Utf8, strict=False).fill_null("")

    if operator in ("is", "equals", "=", "=="):
        return str_series == value
    if operator in ("is not", "not equals", "!="):
        return str_series != value
    if operator == "is one of":
        # Value is comma-separated list
        vals = {v.strip() for v in value.split(",") if v.strip()}
        return str_series.is_in(list(vals))
    if operator == "contains":
        return str_series.str.contains(value, literal=True)
    if operator == "does not contain":
        return ~str_series.str.contains(value, literal=True)
    if operator in ("begins with", "starts with"):
        return str_series.str.starts_with(value)
    if operator in ("ends with",):
        return str_series.str.ends_with(value)

    return pl.Series("m", [True] * n)


def _apply_range_op(
    series: pl.Series, operator: str, value: str, n: int,
) -> pl.Series:
    """Handle 'is between' and 'is outside' with two values separated by ';'."""
    parts = value.split(";")
    if len(parts) != 2:
        return pl.Series("m", [True] * n)
    try:
        lo, hi = float(parts[0].strip()), float(parts[1].strip())
    except (ValueError, TypeError):
        return pl.Series("m", [True] * n)

    float_series = series.cast(pl.Float64, strict=False)
    if operator == "is between":
        return (float_series >= lo) & (float_series <= hi)
    else:  # is outside
        return (float_series < lo) | (float_series > hi)


# ── Purge helpers (matches Orange's Remove preprocessor) ─────────────────


def _purge_domain(
    df: pl.DataFrame,
    domain,
    purge_attributes: bool,
    purge_classes: bool,
) -> pl.DataFrame:
    """Remove unused categorical values and constant columns.

    Mirrors Orange's ``Remove(RemoveConstant | RemoveUnusedValues, ...)``.
    - purge_attributes → applies to feature and meta columns
    - purge_classes    → applies to target columns
    """
    if domain is None or df.height == 0:
        return df

    cols_to_drop: list[str] = []

    for col_schema in domain.columns:
        name = col_schema.name
        if name not in df.columns:
            continue

        role = col_schema.role
        # Decide whether this column is subject to purge
        if role == "target":
            if not purge_classes:
                continue
        else:  # feature or meta
            if not purge_attributes:
                continue

        series = df.get_column(name)

        # Remove constant columns (only 1 unique non-null value or all null)
        n_unique = series.drop_nulls().n_unique()
        if n_unique <= 1:
            cols_to_drop.append(name)
            continue

        # Remove unused categorical values: Polars automatically handles this
        # since Polars categoricals are value-based (no fixed category set).
        # The domain will be rebuilt from actual data via build_data_domain().

    if cols_to_drop:
        df = df.drop(cols_to_drop)

    return df
