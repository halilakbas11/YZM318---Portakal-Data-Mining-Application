from __future__ import annotations

import tempfile
from dataclasses import replace
from datetime import datetime
from pathlib import Path

import polars as pl

from portakal_app.data.models import DataDomain, DatasetHandle, SourceInfo, build_data_domain


class GeneratedDatasetService:
    def build_dataset(
        self,
        dataframe: pl.DataFrame,
        *,
        dataset_id: str,
        display_name: str,
        file_name: str,
        format_name: str = "csv",
        role_overrides: dict[str, str] | None = None,
        annotations: dict[str, object] | None = None,
    ) -> DatasetHandle:
        dataset_root = Path(tempfile.gettempdir()) / "portakal-app" / "generated"
        dataset_root.mkdir(parents=True, exist_ok=True)

        source_path = dataset_root / file_name
        cache_path = dataset_root / f"{Path(file_name).stem}.parquet"
        if format_name in {"csv", "tsv", "tab"}:
            separator = "\t" if format_name in {"tsv", "tab"} else ","
            dataframe.write_csv(source_path, separator=separator)
        elif format_name == "parquet":
            dataframe.write_parquet(source_path)
        else:
            dataframe.write_csv(source_path)
        dataframe.write_parquet(cache_path)

        stat = source_path.stat()
        domain = build_data_domain(dataframe)
        if role_overrides:
            domain = DataDomain(
                columns=tuple(replace(column, role=role_overrides.get(column.name, column.role)) for column in domain.columns)
            )

        source = SourceInfo(
            path=source_path,
            format=format_name,
            size_bytes=int(stat.st_size),
            modified_at=datetime.fromtimestamp(stat.st_mtime),
            cache_path=cache_path,
        )
        return DatasetHandle(
            dataset_id=dataset_id,
            display_name=display_name,
            source=source,
            domain=domain,
            dataframe=dataframe,
            row_count=dataframe.height,
            column_count=dataframe.width,
            cache_path=cache_path,
            annotations=annotations or {},
        )
