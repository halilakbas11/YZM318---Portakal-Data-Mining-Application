from __future__ import annotations

from dataclasses import replace

import polars as pl

from portakal_app.data.models import DatasetHandle, build_data_domain


class SelectByIndexService:
    def select(
        self,
        data: DatasetHandle,
        subset: DatasetHandle,
    ) -> tuple[DatasetHandle | None, DatasetHandle | None]:
        data_df = data.dataframe
        subset_df = subset.dataframe

        # Find common columns to join on (row identity matching)
        common_cols = [c for c in data_df.columns if c in subset_df.columns]
        if not common_cols:
            # No common columns - cannot match, return all as non-matching
            return None, data

        idx_col = "__portakal_idx__"
        dup_col = "__dup_idx__"
        
        data_indexed = data_df.with_row_index(idx_col)

        # Distinguish completely identical duplicate rows by assigning them an occurrence number
        data_keyed = data_indexed.with_columns(
            pl.col(common_cols[0]).cum_count().over(common_cols).alias(dup_col)
        )
        subset_keyed = subset_df.with_columns(
            pl.col(common_cols[0]).cum_count().over(common_cols).alias(dup_col)
        )

        join_on = common_cols + [dup_col]

        # Semi-join: keep rows from data that exist in subset
        matching_indexed = data_keyed.join(
            subset_keyed.select(join_on).unique(),
            on=join_on,
            how="semi",
        ).drop(dup_col)

        # Anti-join: keep rows from data that don't exist in subset
        non_matching_indexed = data_keyed.join(
            subset_keyed.select(join_on).unique(),
            on=join_on,
            how="anti",
        ).drop(dup_col)

        matching_df = matching_indexed.drop(idx_col) if matching_indexed.height > 0 else None
        non_matching_df = non_matching_indexed.drop(idx_col) if non_matching_indexed.height > 0 else None

        matching = None
        if matching_df is not None and matching_df.height > 0:
            matching = replace(
                data,
                dataset_id=f"{data.dataset_id}-matching",
                display_name=f"{data.display_name} (matching)",
                dataframe=matching_df,
                row_count=matching_df.height,
                domain=build_data_domain(matching_df, source_domain=data.domain),
            )

        non_matching = None
        if non_matching_df is not None and non_matching_df.height > 0:
            non_matching = replace(
                data,
                dataset_id=f"{data.dataset_id}-non-matching",
                display_name=f"{data.display_name} (non-matching)",
                dataframe=non_matching_df,
                row_count=non_matching_df.height,
                domain=build_data_domain(non_matching_df, source_domain=data.domain),
            )

        return matching, non_matching
