from __future__ import annotations

from dataclasses import replace

import polars as pl

from portakal_app.data.models import (
    ColumnSchema,
    DataDomain,
    DatasetHandle,
    build_data_domain,
)


class ApplyDomainService:
    """Apply the domain (column structure, roles, types) from a template
    dataset onto the input data.

    Orange equivalent: ``data.transform(template_data.domain)``

    Behaviour:
    1. Columns present in both template and data are kept, cast to the
       template's dtype when possible.
    2. Columns in the template but missing from data are filled with null
       values of the correct type.
    3. Columns in the data but not in the template are dropped.
    4. Column ordering follows the template.
    5. Roles (feature / target / meta) are copied from the template.
    """

    def apply(
        self,
        data: DatasetHandle,
        template: DatasetHandle,
    ) -> DatasetHandle:
        template_columns = list(template.domain.columns)
        data_df = data.dataframe
        data_col_set = set(data_df.columns)

        result_series: list[pl.Series] = []

        for tcol in template_columns:
            col_name = tcol.name
            if col_name in data_col_set:
                src = data_df.get_column(col_name)
                # Try to cast to template dtype
                target_dtype = _resolve_polars_dtype(tcol.dtype_repr)
                if target_dtype is not None and src.dtype != target_dtype:
                    try:
                        converted = src.cast(target_dtype, strict=False)
                        # If strict=False turned a string column into 100% null (failed parsing),
                        # assume the template used an ordinal encoding, so try fallback.
                        if converted.null_count() == data.row_count and src.dtype in (pl.Utf8, pl.Categorical) and target_dtype in (pl.Int64, pl.Int32, pl.Float64):
                            converted = src.cast(pl.Categorical).to_physical().cast(target_dtype)
                        src = converted
                    except Exception:
                        pass  # keep original dtype if cast fails
                result_series.append(src.alias(col_name))
            else:
                # Column missing in data. 
                # Check if it is a dynamically generated one-hot encoded variable (from Continuize)
                base_name = None
                val = None
                if "=" in col_name:
                    splits = col_name.split("=", 1)
                    if len(splits) == 2 and splits[0] in data_col_set:
                        base_name, val = splits[0], splits[1]

                if base_name is not None and val is not None:
                    src = data_df.get_column(base_name)
                    target_dtype = _resolve_polars_dtype(tcol.dtype_repr)
                    if target_dtype is None:
                        target_dtype = pl.Int64
                    
                    # Compute indicator matching Orange style OHE
                    indicator = (src == val).fill_null(False).cast(target_dtype).alias(col_name)
                    result_series.append(indicator)
                else:
                    # Column missing in data -> null-filled
                    target_dtype = _resolve_polars_dtype(tcol.dtype_repr)
                    if target_dtype is None:
                        target_dtype = pl.Utf8
                    null_series = pl.Series(
                        col_name, [None] * data.row_count, dtype=target_dtype
                    )
                    result_series.append(null_series)

        if not result_series:
            empty = pl.DataFrame()
            return replace(
                data,
                dataset_id=f"{data.dataset_id}-transformed",
                display_name=f"{data.display_name} (transformed)",
                dataframe=empty,
                row_count=0,
                column_count=0,
                domain=build_data_domain(empty),
            )

        result_df = pl.DataFrame(result_series)

        # Build domain preserving template roles
        adjusted_columns = []
        for tcol in template_columns:
            col_name = tcol.name
            if col_name not in result_df.columns:
                continue
            s = result_df.get_column(col_name)
            adjusted_columns.append(
                ColumnSchema(
                    name=col_name,
                    dtype_repr=str(s.dtype),
                    logical_type=tcol.logical_type,
                    role=tcol.role,
                    nullable=s.null_count() > 0,
                    null_count=s.null_count(),
                    unique_count_hint=s.n_unique(),
                    sample_values=tcol.sample_values,
                )
            )

        domain = DataDomain(columns=tuple(adjusted_columns))

        return replace(
            data,
            dataset_id=f"{data.dataset_id}-transformed",
            display_name=f"{data.display_name} (transformed)",
            dataframe=result_df,
            row_count=result_df.height,
            column_count=result_df.width,
            domain=domain,
        )


def _resolve_polars_dtype(dtype_repr: str) -> pl.DataType | None:
    """Best-effort mapping from a dtype string repr back to a Polars dtype."""
    mapping = {
        "Int8": pl.Int8,
        "Int16": pl.Int16,
        "Int32": pl.Int32,
        "Int64": pl.Int64,
        "UInt8": pl.UInt8,
        "UInt16": pl.UInt16,
        "UInt32": pl.UInt32,
        "UInt64": pl.UInt64,
        "Float32": pl.Float32,
        "Float64": pl.Float64,
        "Boolean": pl.Boolean,
        "Utf8": pl.Utf8,
        "String": pl.Utf8,
        "Categorical": pl.Categorical,
        "Date": pl.Date,
        "Datetime": pl.Datetime,
    }
    return mapping.get(dtype_repr)
