from __future__ import annotations

from collections import Counter

import polars as pl

from portakal_app.data.models import ColumnProfile, ColumnSchema, DatasetHandle, DatasetSummary


class ProfilingService:
    def summarize(self, dataset: DatasetHandle) -> DatasetSummary:
        row_count = dataset.row_count
        column_count = dataset.column_count
        missing_value_count = sum(column.null_count for column in dataset.domain.columns)
        total_cells = row_count * column_count
        missing_ratio = (missing_value_count / total_cells) if total_cells else 0.0
        duplicate_row_count = self._duplicate_row_count(dataset.dataframe)
        dtype_counts = dict(Counter(column.logical_type for column in dataset.domain.columns))
        column_profiles = tuple(self._build_column_profile(column, row_count) for column in dataset.domain.columns)

        return DatasetSummary(
            row_count=row_count,
            column_count=column_count,
            missing_value_count=missing_value_count,
            missing_ratio=missing_ratio,
            duplicate_row_count=duplicate_row_count,
            feature_count=len(dataset.domain.feature_columns),
            target_count=len(dataset.domain.target_columns),
            dtype_counts=dtype_counts,
            column_profiles=column_profiles,
        )

    def _duplicate_row_count(self, dataframe: pl.DataFrame) -> int:
        if dataframe.is_empty():
            return 0
        duplicate_mask = dataframe.is_duplicated()
        if duplicate_mask is None:
            return 0
        return int(duplicate_mask.cast(pl.Int64).sum() or 0)

    def _build_column_profile(self, column: ColumnSchema, row_count: int) -> ColumnProfile:
        null_ratio = (column.null_count / row_count) if row_count else 0.0
        sample_values = column.sample_values
        summary_parts = [
            f"{column.logical_type.title()} column",
            f"role: {column.role}",
            f"unique: {column.unique_count_hint}",
            f"nulls: {column.null_count}",
        ]
        if sample_values:
            summary_parts.append(f"samples: {', '.join(sample_values)}")

        return ColumnProfile(
            column_name=column.name,
            logical_type=column.logical_type,
            role=column.role,
            null_count=column.null_count,
            null_ratio=null_ratio,
            unique_count_hint=column.unique_count_hint,
            sample_values=sample_values,
            summary=" | ".join(summary_parts),
        )
