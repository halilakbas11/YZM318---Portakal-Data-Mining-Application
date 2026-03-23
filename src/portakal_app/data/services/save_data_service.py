from __future__ import annotations

from pathlib import Path

from portakal_app.data.errors import DatasetSaveError, UnsupportedFormatError
from portakal_app.data.models import DatasetHandle


class SaveDataService:
    def save(self, dataset: DatasetHandle, path: str, format: str | None = None) -> None:
        target_path = Path(path).expanduser().resolve()
        file_format = (format or self._format_from_path(target_path)).lower()
        if file_format not in {"csv", "xlsx", "parquet"}:
            raise UnsupportedFormatError(f"Unsupported export format: {file_format}")

        target_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            if file_format == "csv":
                dataset.dataframe.write_csv(target_path)
            elif file_format == "xlsx":
                dataset.dataframe.write_excel(target_path)
            else:
                dataset.dataframe.write_parquet(target_path)
        except UnsupportedFormatError:
            raise
        except Exception as exc:
            raise DatasetSaveError(f"Dataset could not be saved to {target_path.name}.") from exc

    def _format_from_path(self, target_path: Path) -> str:
        suffix = target_path.suffix.lower()
        mapping = {
            ".csv": "csv",
            ".xlsx": "xlsx",
            ".parquet": "parquet",
        }
        file_format = mapping.get(suffix)
        if file_format is None:
            raise UnsupportedFormatError(f"Unsupported export format: {suffix or 'unknown'}")
        return file_format
