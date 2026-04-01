from __future__ import annotations

from dataclasses import replace

import polars as pl

from portakal_app.data.models import DatasetHandle, build_data_domain


OUTPUT_TYPES = ("Categorical (No, Yes)", "Numerical (0, 1)", "Counts")


class SplitService:
    def split_column(
        self,
        dataset: DatasetHandle,
        *,
        column_name: str,
        delimiter: str = ";",
        output_type: str = "Numerical (0, 1)",
    ) -> DatasetHandle:
        df = dataset.dataframe

        if column_name not in df.columns:
            return dataset

        series = df.get_column(column_name).cast(pl.Utf8).fill_null("")
        all_values: set[str] = set()
        for val in series.to_list():
            parts = [p.strip() for p in str(val).split(delimiter) if p.strip()]
            all_values.update(parts)

        sorted_values = sorted(all_values)
        if not sorted_values:
            return dataset

        new_columns: list[pl.Series] = []
        rows = series.to_list()

        for sv in sorted_values:
            col_name = f"{column_name} - {sv}"
            if output_type == "Counts":
                values = []
                for row in rows:
                    parts = [p.strip() for p in str(row).split(delimiter)]
                    values.append(parts.count(sv))
                new_columns.append(pl.Series(col_name, values, dtype=pl.Int64))
            elif output_type == "Categorical (No, Yes)":
                values = []
                for row in rows:
                    parts = {p.strip() for p in str(row).split(delimiter)}
                    values.append("Yes" if sv in parts else "No")
                new_columns.append(pl.Series(col_name, values, dtype=pl.Utf8))
            else:
                values = []
                for row in rows:
                    parts = {p.strip() for p in str(row).split(delimiter)}
                    values.append(1 if sv in parts else 0)
                new_columns.append(pl.Series(col_name, values, dtype=pl.Int64))

        result_df = df
        for col in new_columns:
            result_df = result_df.with_columns(col)

        return replace(
            dataset,
            dataset_id=f"{dataset.dataset_id}-split",
            display_name=f"{dataset.display_name} (split)",
            dataframe=result_df,
            row_count=result_df.height,
            column_count=result_df.width,
            domain=build_data_domain(result_df, source_domain=dataset.domain),
        )
