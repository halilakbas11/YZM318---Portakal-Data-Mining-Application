from __future__ import annotations

from portakal_app.data.models import DatasetHandle, PreviewPage


class PreviewService:
    """Paginated data access for DatasetHandle objects using Polars slicing."""

    def get_page(self, dataset: DatasetHandle, offset: int = 0, limit: int = 100) -> PreviewPage:
        """Return a page of rows from the dataset as string tuples."""
        df = dataset.dataframe
        total = df.height
        offset = max(0, min(offset, total))
        limit = max(0, limit)
        page_df = df.slice(offset, limit)

        headers = tuple(df.columns)
        rows = tuple(
            tuple(self._cell_to_str(value) for value in row)
            for row in page_df.rows()
        )
        return PreviewPage(
            headers=headers,
            rows=rows,
            offset=offset,
            limit=limit,
            total_rows=total,
        )

    def get_total_rows(self, dataset: DatasetHandle) -> int:
        """Return the total number of rows in the dataset."""
        return dataset.dataframe.height

    def get_numeric_ranges(self, dataset: DatasetHandle) -> dict[int, tuple[float, float]]:
        """Compute (min, max) for each numeric column. Returns {column_index: (min, max)}."""
        df = dataset.dataframe
        ranges: dict[int, tuple[float, float]] = {}
        for col_index, col_schema in enumerate(dataset.domain.columns):
            if col_schema.logical_type != "numeric":
                continue
            series = df.get_column(col_schema.name)
            try:
                col_min = series.min()
                col_max = series.max()
            except Exception:
                continue
            if col_min is None or col_max is None:
                continue
            ranges[col_index] = (float(col_min), float(col_max))
        return ranges

    def get_column_specs(self, dataset: DatasetHandle) -> list[dict[str, str]]:
        """Return column metadata from the dataset domain, suitable for the table model."""
        target_index = self._infer_target_index(dataset)
        specs: list[dict[str, str]] = []
        for index, col in enumerate(dataset.domain.columns):
            role = "target" if index == target_index else col.role
            preview = ", ".join(col.sample_values) if col.logical_type == "categorical" else ""
            specs.append({
                "name": col.name,
                "type_name": col.logical_type,
                "role_name": role,
                "values_preview": preview,
            })
        return specs

    def _infer_target_index(self, dataset: DatasetHandle) -> int | None:
        """Find the target column index (last low-cardinality non-numeric column)."""
        candidates: list[int] = []
        for index, col in enumerate(dataset.domain.columns):
            if col.role == "target":
                candidates.append(index)
        return candidates[-1] if candidates else None

    def _cell_to_str(self, value: object) -> str:
        if value is None:
            return ""
        return str(value)
