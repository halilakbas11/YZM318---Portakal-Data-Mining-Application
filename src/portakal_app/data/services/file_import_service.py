from __future__ import annotations

import hashlib
import io
import tempfile
from datetime import datetime
from pathlib import Path

import polars as pl

from portakal_app.data.errors import DatasetLoadError, UnsupportedFormatError
from portakal_app.data.models import CSVImportOptions, DatasetHandle, SourceInfo, build_data_domain


class FileImportService:
    DELIMITED_CANDIDATES = (",", "\t", ";", "|")
    AUTO_ENCODINGS = ("utf-8-sig", "utf-8", "cp1254", "latin-1")

    def load(self, path: str) -> DatasetHandle:
        source_path = Path(path).expanduser().resolve()
        if not source_path.exists() or not source_path.is_file():
            raise DatasetLoadError(f"Dataset file could not be found: {source_path}")

        file_format = self._detect_format(source_path)
        cache_path = self._cache_path_for(source_path)

        try:
            dataframe = self._read_dataframe(source_path, file_format)
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            dataframe.write_parquet(cache_path)
        except UnsupportedFormatError:
            raise
        except Exception as exc:
            raise DatasetLoadError(f"Dataset could not be loaded from {source_path.name}.") from exc

        stat = source_path.stat()
        source = SourceInfo(
            path=source_path,
            format=file_format,
            size_bytes=int(stat.st_size),
            modified_at=datetime.fromtimestamp(stat.st_mtime),
            cache_path=cache_path,
        )
        domain = build_data_domain(dataframe)
        dataset_id = cache_path.stem
        return DatasetHandle(
            dataset_id=dataset_id,
            display_name=source_path.stem.replace("_", " ").title() or source_path.name,
            source=source,
            domain=domain,
            dataframe=dataframe,
            row_count=dataframe.height,
            column_count=dataframe.width,
            cache_path=cache_path,
        )

    def load_delimited_text(self, path: str, options: CSVImportOptions | None = None) -> DatasetHandle:
        source_path = Path(path).expanduser().resolve()
        if not source_path.exists() or not source_path.is_file():
            raise DatasetLoadError(f"Dataset file could not be found: {source_path}")

        options = self.resolve_delimited_options(path, options)
        cache_path = self._cache_path_for(source_path)
        try:
            source_text = source_path.read_text(encoding=options.encoding)
            dataframe = pl.read_csv(
                io.StringIO(source_text),
                separator=options.delimiter,
                has_header=options.has_header,
                skip_rows=options.skip_rows,
            )
            if not options.has_header:
                dataframe = dataframe.rename(
                    {column: f"Column {index}" for index, column in enumerate(dataframe.columns, start=1)}
                )
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            dataframe.write_parquet(cache_path)
        except Exception as exc:
            raise DatasetLoadError(f"Dataset could not be imported from {source_path.name}.") from exc

        stat = source_path.stat()
        detected_format = self._format_for_delimited_source(source_path, options.delimiter)
        source = SourceInfo(
            path=source_path,
            format=detected_format,
            size_bytes=int(stat.st_size),
            modified_at=datetime.fromtimestamp(stat.st_mtime),
            cache_path=cache_path,
        )
        domain = build_data_domain(dataframe)
        dataset_id = cache_path.stem
        return DatasetHandle(
            dataset_id=dataset_id,
            display_name=source_path.stem.replace("_", " ").title() or source_path.name,
            source=source,
            domain=domain,
            dataframe=dataframe,
            row_count=dataframe.height,
            column_count=dataframe.width,
            cache_path=cache_path,
        )

    def resolve_delimited_options(self, path: str, options: CSVImportOptions | None = None) -> CSVImportOptions:
        source_path = Path(path).expanduser().resolve()
        if not source_path.exists() or not source_path.is_file():
            raise DatasetLoadError(f"Dataset file could not be found: {source_path}")

        requested = options or CSVImportOptions()
        sample_bytes = source_path.read_bytes()[:65536]
        encoding = self._resolve_encoding(sample_bytes, requested.encoding)
        sample_text = sample_bytes.decode(encoding, errors="replace")
        delimiter = (
            self._detect_delimiter(sample_text, requested.delimiter)
            if requested.auto_detect_delimiter
            else requested.delimiter
        )
        return CSVImportOptions(
            delimiter=delimiter,
            has_header=requested.has_header,
            encoding=encoding,
            skip_rows=max(0, requested.skip_rows),
            auto_detect_delimiter=False,
        )

    def _read_dataframe(self, source_path: Path, file_format: str) -> pl.DataFrame:
        if file_format == "csv":
            return pl.read_csv(source_path)
        if file_format in {"tsv", "tab"}:
            return pl.read_csv(source_path, separator="\t")
        if file_format in {"xlsx", "xls"}:
            return pl.read_excel(source_path, engine="calamine")
        if file_format == "parquet":
            return pl.read_parquet(source_path)
        raise UnsupportedFormatError(f"Unsupported dataset format: {source_path.suffix.lower() or 'unknown'}")

    def _detect_format(self, source_path: Path) -> str:
        suffix = source_path.suffix.lower()
        mapping = {
            ".csv": "csv",
            ".tsv": "tsv",
            ".tab": "tab",
            ".xlsx": "xlsx",
            ".xls": "xls",
            ".parquet": "parquet",
        }
        file_format = mapping.get(suffix)
        if file_format is None:
            raise UnsupportedFormatError(f"Unsupported dataset format: {suffix or 'unknown'}")
        return file_format

    def _cache_path_for(self, source_path: Path) -> Path:
        stat = source_path.stat()
        fingerprint = hashlib.sha1(
            f"{source_path.resolve()}::{stat.st_mtime_ns}::{stat.st_size}".encode("utf-8")
        ).hexdigest()[:12]
        return Path(tempfile.gettempdir()) / "portakal-app" / "datasets" / f"{fingerprint}.parquet"

    def _format_for_delimited_source(self, source_path: Path, delimiter: str) -> str:
        suffix = source_path.suffix.lower()
        if suffix == ".tsv" or delimiter == "\t":
            return "tsv"
        if suffix == ".tab":
            return "tab"
        return "csv"

    def _resolve_encoding(self, sample_bytes: bytes, requested_encoding: str) -> str:
        if requested_encoding and requested_encoding.lower() != "auto":
            return requested_encoding
        for encoding in self.AUTO_ENCODINGS:
            try:
                sample_bytes.decode(encoding)
                return encoding
            except UnicodeDecodeError:
                continue
        return self.AUTO_ENCODINGS[-1]

    def _detect_delimiter(self, sample_text: str, fallback: str) -> str:
        lines = [line for line in sample_text.splitlines()[:12] if line.strip()]
        if not lines:
            return fallback
        best_delimiter = fallback
        best_score = -1
        for delimiter in self.DELIMITED_CANDIDATES:
            counts = [line.count(delimiter) for line in lines]
            if not counts:
                continue
            positive_counts = [count for count in counts if count > 0]
            if not positive_counts:
                continue
            score = min(positive_counts) * len(positive_counts)
            if len(set(positive_counts)) == 1:
                score += 100
            if score > best_score:
                best_score = score
                best_delimiter = delimiter
        return best_delimiter
