from __future__ import annotations

from dataclasses import replace

import polars as pl

from portakal_app.data.models import DatasetHandle, build_data_domain


class TransposeService:
    def transpose(
        self,
        dataset: DatasetHandle,
        *,
        feature_names_from: str | None = None,
        feature_name_prefix: str = "Feature",
        auto_column_name: str = "column",
        remove_redundant_instance: bool = False,
    ) -> DatasetHandle:
        df = dataset.dataframe

        # Determine feature names and which data to transpose
        if feature_names_from and feature_names_from in df.columns:
            name_series = df.get_column(feature_names_from).cast(pl.Utf8)
            names = name_series.to_list()

            if remove_redundant_instance:
                # Drop the column before transposing so it does NOT
                # appear as a (redundant) row in the output.
                df_to_transpose = df.drop(feature_names_from)
            else:
                # Keep it — it becomes an extra row in the result.
                df_to_transpose = df
        else:
            names = None
            df_to_transpose = df

        transposed = df_to_transpose.transpose(
            include_header=True, header_name=auto_column_name
        )

        if names:
            seen = {}
            new_col_names = [auto_column_name]
            seen[auto_column_name] = 1
            for i, n in enumerate(names):
                base = str(n) if n is not None else f"row_{i}"
                if base not in seen:
                    new_col_names.append(base)
                    seen[base] = 1
                else:
                    count = seen[base]
                    candidate = f"{base} ({count})"
                    while candidate in seen:
                        count += 1
                        candidate = f"{base} ({count})"
                    seen[base] = count + 1
                    seen[candidate] = 1
                    new_col_names.append(candidate)

            if len(new_col_names) == len(transposed.columns):
                transposed.columns = new_col_names
        else:
            new_col_names = [auto_column_name] + [
                f"{feature_name_prefix}{i + 1}"
                for i in range(len(transposed.columns) - 1)
            ]
            if len(new_col_names) == len(transposed.columns):
                transposed.columns = new_col_names

        return replace(
            dataset,
            dataset_id=f"{dataset.dataset_id}-transposed",
            display_name=f"{dataset.display_name} (transposed)",
            dataframe=transposed,
            row_count=transposed.height,
            column_count=transposed.width,
            domain=build_data_domain(transposed, source_domain=dataset.domain),
        )
