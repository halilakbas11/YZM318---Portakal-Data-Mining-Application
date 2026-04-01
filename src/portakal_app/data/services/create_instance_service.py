from __future__ import annotations

import random
from dataclasses import replace

import polars as pl

from portakal_app.data.models import DatasetHandle, build_data_domain


class CreateInstanceService:
    def create(
        self,
        reference: DatasetHandle | None = None,
        data: DatasetHandle | None = None,
        *,
        values: dict[str, object],
        append_to_data: bool = False,
    ) -> DatasetHandle | None:
        
        schema_ds = reference if reference is not None else data
        if schema_ds is None:
            return None

        df_schema = schema_ds.dataframe

        row_data: dict[str, list] = {}
        for col_name in df_schema.columns:
            series = df_schema.get_column(col_name)
            if col_name in values:
                val = values[col_name]
                try:
                    if series.dtype.is_numeric():
                        val = float(val) if val is not None and str(val).strip() != "" else None
                except (ValueError, TypeError):
                    val = None
                row_data[col_name] = [val]
            else:
                row_data[col_name] = [None]

        schema = {}
        for c in df_schema.columns:
            dtype = df_schema.get_column(c).dtype
            # Widen all integer types to Float64 so that float values
            # (e.g. 78.91 from the spin-box) never clash with Int64.
            # Orange3 stores everything as np.float64 internally.
            if dtype.is_integer():
                dtype = pl.Float64
            schema[c] = dtype
        new_row = pl.DataFrame(row_data, schema=schema)

        if append_to_data and data is not None:
            # Cast original data's integer columns to Float64 too
            data_df = data.dataframe
            for c in data_df.columns:
                if data_df.get_column(c).dtype.is_integer():
                    data_df = data_df.with_columns(pl.col(c).cast(pl.Float64))
            try:
                result = pl.concat([data_df, new_row], how="vertical_relaxed")
            except Exception:
                result = new_row
        else:
            result = new_row

        base_id = data.dataset_id if data else schema_ds.dataset_id
        base_name = data.display_name if data else schema_ds.display_name

        return replace(
            schema_ds,
            dataset_id=f"{base_id}-instance",
            display_name=f"{base_name} (instance)",
            dataframe=result,
            row_count=result.height,
            column_count=result.width,
            domain=build_data_domain(result, source_domain=schema_ds.domain),
        )

    def get_defaults(self, dataset: DatasetHandle, stat_type: str = "median") -> dict[str, str]:
        defaults: dict[str, str] = {}
        for col in dataset.domain.columns:
            series = dataset.dataframe.get_column(col.name)
            non_null = series.drop_nulls()
            if non_null.len() == 0:
                defaults[col.name] = ""
                continue
            
            if series.dtype.is_numeric():
                if stat_type == "mean":
                    val = non_null.mean()
                elif stat_type == "random":
                    # Simple random choice or uniform
                    mn, mx = non_null.min(), non_null.max()
                    val = random.uniform(mn, mx) if mn is not None and mx is not None else None
                else:
                    val = non_null.median()
                defaults[col.name] = str(val) if val is not None else ""
            else:
                # Mode for categoricals
                if stat_type == "random":
                    unique_vals = non_null.unique().to_list()
                    val = random.choice(unique_vals) if unique_vals else None
                else:
                    mode_res = non_null.mode()
                    val = mode_res[0] if mode_res.len() > 0 else None
                defaults[col.name] = str(val) if val is not None else ""
                
        return defaults
