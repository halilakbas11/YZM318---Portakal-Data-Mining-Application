from __future__ import annotations

from dataclasses import replace

from portakal_app.data.models import ColumnSchema, DataDomain, DatasetHandle


class SelectColumnsService:
    def select(
        self,
        dataset: DatasetHandle,
        *,
        features: list[str],
        target: list[str],
        metas: list[str],
    ) -> DatasetHandle:
        df = dataset.dataframe
        # Column hierarchy: Target → Meta → Features
        # Ignored columns are physically dropped (not included here)
        all_selected = target + metas + features
        keep_cols = [c for c in all_selected if c in df.columns]

        if not keep_cols:
            return dataset

        result_df = df.select(keep_cols)

        role_map: dict[str, str] = {}
        for name in features:
            role_map[name] = "feature"
        for name in target:
            role_map[name] = "target"
        for name in metas:
            role_map[name] = "meta"

        old_schema_map = {col.name: col for col in dataset.domain.columns}
        new_columns: list[ColumnSchema] = []
        for name in keep_cols:
            col = old_schema_map[name]
            new_columns.append(
                ColumnSchema(
                    name=col.name,
                    dtype_repr=col.dtype_repr,
                    logical_type=col.logical_type,
                    role=role_map[col.name],
                    nullable=col.nullable,
                    null_count=col.null_count,
                    unique_count_hint=col.unique_count_hint,
                    sample_values=col.sample_values,
                )
            )

        return replace(
            dataset,
            dataset_id=f"{dataset.dataset_id}-selected",
            display_name=f"{dataset.display_name} (selected)",
            dataframe=result_df,
            row_count=result_df.height,
            column_count=result_df.width,
            domain=DataDomain(columns=tuple(new_columns)),
        )
