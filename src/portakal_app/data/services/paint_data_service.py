from __future__ import annotations

import tempfile
from dataclasses import replace
from datetime import datetime
from pathlib import Path

import polars as pl

from portakal_app.data.models import PaintDataPoint, PaintDataSnapshot, DatasetHandle, SourceInfo, build_data_domain


class PaintDataService:
    def build_snapshot(self, dataset: DatasetHandle | None) -> PaintDataSnapshot:
        if dataset is None or dataset.row_count == 0:
            return PaintDataSnapshot()

        numeric_columns = [column for column in dataset.domain.columns if column.logical_type == "numeric"]
        if len(numeric_columns) < 2:
            return PaintDataSnapshot(source_name=dataset.display_name)

        x_column = numeric_columns[0]
        y_column = numeric_columns[1]
        target_column = next((column for column in dataset.domain.columns if column.role == "target"), None)

        x_values = self._numeric_values(dataset.dataframe.get_column(x_column.name))
        y_values = self._numeric_values(dataset.dataframe.get_column(y_column.name))
        label_values = self._label_values(dataset, target_column)
        normalized_x = self._normalize_values(x_values)
        normalized_y = self._normalize_values(y_values)
        points = tuple(
            PaintDataPoint(x=x_value, y=y_value, label=label)
            for x_value, y_value, label in zip(normalized_x, normalized_y, label_values, strict=False)
        )
        labels = tuple(dict.fromkeys(label_values)) or ("C1", "C2")
        label_name = target_column.name if target_column is not None else "class"
        return PaintDataSnapshot(
            x_name=x_column.name,
            y_name=y_column.name,
            label_name=label_name,
            labels=labels,
            points=points,
            source_name=dataset.display_name,
        )

    def build_dataset(self, snapshot: PaintDataSnapshot) -> DatasetHandle:
        dataframe = pl.DataFrame(
            {
                snapshot.x_name: [point.x for point in snapshot.points],
                snapshot.y_name: [point.y for point in snapshot.points],
                snapshot.label_name: [point.label for point in snapshot.points],
            }
        )
        dataset_root = Path(tempfile.gettempdir()) / "portakal-app" / "generated"
        dataset_root.mkdir(parents=True, exist_ok=True)
        source_path = dataset_root / "paint-data.csv"
        cache_path = dataset_root / "paint-data.parquet"
        dataframe.write_csv(source_path)
        dataframe.write_parquet(cache_path)
        stat = source_path.stat()
        source = SourceInfo(
            path=source_path,
            format="csv",
            size_bytes=int(stat.st_size),
            modified_at=datetime.fromtimestamp(stat.st_mtime),
            cache_path=cache_path,
        )
        return DatasetHandle(
            dataset_id="paint-data",
            display_name=snapshot.source_name or "Painted Data",
            source=source,
            domain=build_data_domain(dataframe),
            dataframe=dataframe,
            row_count=dataframe.height,
            column_count=dataframe.width,
            cache_path=cache_path,
            annotations={"generated_by": "paint-data"},
        )

    def replace_labels(self, snapshot: PaintDataSnapshot, labels: tuple[str, ...]) -> PaintDataSnapshot:
        if not labels:
            return snapshot
        valid_labels = tuple(dict.fromkeys(label.strip() for label in labels if label.strip()))
        if not valid_labels:
            return snapshot
        fallback = valid_labels[0]
        points = tuple(
            replace(point, label=point.label if point.label in valid_labels else fallback)
            for point in snapshot.points
        )
        return replace(snapshot, labels=valid_labels, points=points)

    def _numeric_values(self, series: pl.Series) -> list[float]:
        values: list[float] = []
        for value in series.to_list():
            try:
                values.append(float(value))
            except (TypeError, ValueError):
                values.append(0.0)
        return values

    def _label_values(self, dataset: DatasetHandle, target_column) -> list[str]:
        if target_column is None:
            return ["C1"] * dataset.row_count
        values: list[str] = []
        for value in dataset.dataframe.get_column(target_column.name).to_list():
            text = str(value).strip()
            values.append(text or "Unknown")
        return values

    def _normalize_values(self, values: list[float]) -> list[float]:
        if not values:
            return []
        minimum = min(values)
        maximum = max(values)
        if minimum == maximum:
            return [0.5 for _value in values]
        return [(value - minimum) / (maximum - minimum) for value in values]
