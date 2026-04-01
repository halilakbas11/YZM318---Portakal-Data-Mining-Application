from __future__ import annotations

from dataclasses import replace

import polars as pl

from portakal_app.data.models import ColumnSchema, DataDomain, DatasetHandle, build_data_domain


JOIN_TYPES = ("Left Join", "Inner Join", "Outer Join")


class MergeDataService:
    def merge(
        self,
        data: DatasetHandle,
        extra: DatasetHandle,
        *,
        left_on: list[str] | str,
        right_on: list[str] | str,
        join_type: str = "Left Join",
    ) -> DatasetHandle:
        df_left = data.dataframe
        df_right = extra.dataframe

        if isinstance(left_on, str):
            left_on = [left_on]
        if isinstance(right_on, str):
            right_on = [right_on]

        added_row_index = False
        added_instance_id = False

        if "Row index" in left_on and "Row index" not in df_left.columns:
            df_left = df_left.with_row_index("Row index")
            added_row_index = True

        if "Row index" in right_on and "Row index" not in df_right.columns:
            df_right = df_right.with_row_index("Row index")
            added_row_index = True

        if "Instance id" in left_on and "Instance id" not in df_left.columns:
            df_left = df_left.with_row_index("Instance id")
            added_instance_id = True

        if "Instance id" in right_on and "Instance id" not in df_right.columns:
            df_right = df_right.with_row_index("Instance id")
            added_instance_id = True

        for col in left_on:
            if col not in df_left.columns:
                return data
        for col in right_on:
            if col not in df_right.columns:
                return data

        how_map = {
            "Left Join": "left",
            "Inner Join": "inner",
            "Outer Join": "full",
        }
        how = how_map.get(join_type, "left")

        right_cols_to_rename = {}
        overlapping_cols = []
        for col in df_right.columns:
            if col not in right_on and col in df_left.columns:
                right_cols_to_rename[col] = f"{col}_right"
                overlapping_cols.append(col)
        
        if right_cols_to_rename:
            df_right = df_right.rename(right_cols_to_rename)

        for l_col, r_col in zip(left_on, right_on):
            if df_left[l_col].dtype != df_right[r_col].dtype:
                try:
                    df_right = df_right.with_columns(pl.col(r_col).cast(df_left[l_col].dtype, strict=False))
                except Exception:
                    df_left = df_left.with_columns(pl.col(l_col).cast(pl.Utf8))
                    df_right = df_right.with_columns(pl.col(r_col).cast(pl.Utf8))

        left_is_unique = df_left.select(left_on).is_unique().all()
        right_is_unique = df_right.select(right_on).is_unique().all()

        if how == "full":
            if not left_is_unique or not right_is_unique:
                raise ValueError("For Outer Join, row matching combinations must be unique in both datasets.")
        else:
            if not right_is_unique:
                right_dups = df_right.filter(~df_right.select(right_on).is_unique()).select(right_on).unique()
                left_keys = df_left.select(left_on).unique()
                left_keys_renamed = left_keys.rename(dict(zip(left_on, right_on)))
                matches = right_dups.join(left_keys_renamed, on=right_on, how="inner")
                if matches.height > 0:
                    raise ValueError("Matched row combinations in Extra Data appear in multiple rows. Every matched combination may appear at most once.")
            if how == "inner" and not left_is_unique:
                left_dups = df_left.filter(~df_left.select(left_on).is_unique()).select(left_on).unique()
                right_keys = df_right.select(right_on).unique()
                right_keys_renamed = right_keys.rename(dict(zip(right_on, left_on)))
                matches = left_dups.join(right_keys_renamed, on=left_on, how="inner")
                if matches.height > 0:
                    raise ValueError("Matched row combinations in Data appear in multiple rows. Every matched combination may appear at most once.")

        result = df_left.join(
            df_right,
            left_on=left_on,
            right_on=right_on,
            how=how,
            coalesce=True,
        )

        cols_to_drop = []
        for col in overlapping_cols:
            right_col = right_cols_to_rename[col]
            conflict_count = result.select(
                (pl.col(col).is_not_null() & pl.col(right_col).is_not_null() & (pl.col(col) != pl.col(right_col))).sum()
            ).item()

            if conflict_count == 0:
                # No conflicting non-null values, so we can coalesce them and drop the right column.
                # This perfectly mimics Orange's behavior of omitting perfectly identical overlapping columns.
                result = result.with_columns(pl.coalesce([col, right_col]).alias(col))
                cols_to_drop.append(right_col)

        if cols_to_drop:
            result = result.drop(cols_to_drop)

        # Preserve roles and Orange-like layout: Meta -> Target -> Feature
        left_schemas = {col.name: col for col in data.domain.columns}
        right_schemas = {col.name: col for col in extra.domain.columns}

        meta_cols = []
        target_cols = []
        feature_cols = []

        for col_name in result.columns:
            if col_name in ("Row index", "Instance id"):
                schema = ColumnSchema(
                    name=col_name,
                    dtype_repr=str(result.get_column(col_name).dtype),
                    logical_type="numeric",
                    role="meta",
                    nullable=False,
                    null_count=0,
                    unique_count_hint=result.height,
                )
            elif col_name in left_schemas:
                schema = left_schemas[col_name]
            elif col_name in right_schemas:
                schema = right_schemas[col_name]
            elif col_name.endswith("_right") and col_name[:-6] in right_schemas:
                orig_schema = right_schemas[col_name[:-6]]
                schema = replace(orig_schema, name=col_name)
            else:
                s = result.get_column(col_name)
                schema = ColumnSchema(
                    name=col_name,
                    dtype_repr=str(s.dtype),
                    logical_type="unknown",
                    role="feature",
                    nullable=s.null_count() > 0,
                    null_count=s.null_count(),
                    unique_count_hint=s.n_unique(),
                )

            if schema.role == "meta":
                meta_cols.append(schema)
            elif schema.role == "target":
                target_cols.append(schema)
            else:
                feature_cols.append(schema)

        final_ordered_schemas = meta_cols + target_cols + feature_cols

        if added_row_index:
            final_ordered_schemas = [s for s in final_ordered_schemas if s.name != "Row index"]
        if added_instance_id:
            final_ordered_schemas = [s for s in final_ordered_schemas if s.name != "Instance id"]

        ordered_names = [s.name for s in final_ordered_schemas]
        result = result.select(ordered_names)

        return replace(
            data,
            dataset_id=f"{data.dataset_id}-merged",
            display_name=f"{data.display_name} (merged)",
            dataframe=result,
            row_count=result.height,
            column_count=result.width,
            domain=DataDomain(columns=tuple(final_ordered_schemas)),
        )
