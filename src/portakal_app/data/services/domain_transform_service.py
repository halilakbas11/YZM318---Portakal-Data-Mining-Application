from __future__ import annotations

from dataclasses import replace
from datetime import date, datetime, time, timedelta

import polars as pl

from portakal_app.data.models import (
    DataDomain,
    DatasetHandle,
    DomainColumnEdit,
    DomainEditRequest,
    build_data_domain,
)


class DomainTransformService:
    def build_request(self, dataset: DatasetHandle) -> DomainEditRequest:
        return DomainEditRequest(
            columns=tuple(
                DomainColumnEdit(
                    original_name=column.name,
                    new_name=column.name,
                    logical_type=column.logical_type,
                    role=column.role,
                )
                for column in dataset.domain.columns
            )
        )

    def apply(self, dataset: DatasetHandle, request: DomainEditRequest) -> DatasetHandle:
        edits = list(request.columns)
        if not edits:
            return dataset

        target_count = sum(1 for edit in edits if edit.role == "target")
        if target_count > 1:
            raise ValueError("Exactly one target column is allowed.")

        new_columns: list[pl.Series] = []
        role_overrides: dict[str, str] = {}
        used_names: set[str] = set()
        edit_by_name = {edit.original_name: edit for edit in edits}

        for column in dataset.domain.columns:
            edit = edit_by_name.get(column.name)
            if edit is None:
                continue
            if edit.role == "skip":
                continue

            new_name = edit.new_name.strip() or column.name
            if new_name in used_names:
                raise ValueError(f"Duplicate column name is not allowed: {new_name}")
            used_names.add(new_name)

            coerced = self._coerce_series(dataset.dataframe.get_column(column.name), edit.logical_type)
            if coerced.name != new_name:
                coerced = coerced.rename(new_name)
            new_columns.append(coerced)
            role_overrides[new_name] = "feature" if edit.role == "skip" else edit.role

        updated_df = pl.DataFrame(new_columns)
        inferred_domain = build_data_domain(updated_df)
        updated_domain = DataDomain(
            columns=tuple(replace(column, role=role_overrides.get(column.name, column.role)) for column in inferred_domain.columns)
        )
        return replace(
            dataset,
            dataframe=updated_df,
            domain=updated_domain,
            row_count=updated_df.height,
            column_count=updated_df.width,
        )

    def summarize_changes(self, dataset: DatasetHandle, request: DomainEditRequest, updated_dataset: DatasetHandle) -> str:
        renamed = []
        role_changes = []
        type_changes = []
        dropped = []

        original_map = {column.name: column for column in dataset.domain.columns}
        updated_names = {column.name for column in updated_dataset.domain.columns}
        for edit in request.columns:
            original = original_map[edit.original_name]
            if edit.role == "skip":
                dropped.append(edit.original_name)
                continue
            if edit.new_name != edit.original_name:
                renamed.append(f"{edit.original_name} -> {edit.new_name}")
            if edit.role != original.role:
                role_changes.append(f"{edit.new_name}: {original.role} -> {edit.role}")
            if edit.logical_type != original.logical_type:
                type_changes.append(f"{edit.new_name}: {original.logical_type} -> {edit.logical_type}")

        parts: list[str] = []
        if renamed:
            parts.append(f"Renamed {len(renamed)} column(s)")
        if role_changes:
            parts.append(f"Updated roles for {len(role_changes)} column(s)")
        if type_changes:
            parts.append(f"Changed types for {len(type_changes)} column(s)")
        if dropped:
            parts.append(f"Dropped {len(dropped)} skipped column(s)")
        if not parts:
            return "No domain changes applied."
        return ". ".join(parts) + "."

    def _coerce_series(self, series: pl.Series, logical_type: str) -> pl.Series:
        if logical_type in {"text", "categorical", "unknown"}:
            return series.cast(pl.String, strict=False)
        if logical_type == "numeric":
            return self._validate_conversion(series, series.cast(pl.Float64, strict=False), logical_type)
        if logical_type == "boolean":
            return self._validate_conversion(
                series,
                pl.Series(series.name, [self._parse_boolean(value) for value in series.to_list()], dtype=pl.Boolean),
                logical_type,
            )
        if logical_type == "datetime":
            return self._validate_conversion(
                series,
                pl.Series(series.name, [self._parse_datetime(value) for value in series.to_list()], dtype=pl.Datetime),
                logical_type,
            )
        if logical_type == "date":
            return self._validate_conversion(
                series,
                pl.Series(series.name, [self._parse_date(value) for value in series.to_list()], dtype=pl.Date),
                logical_type,
            )
        if logical_type == "time":
            return self._validate_conversion(
                series,
                pl.Series(series.name, [self._parse_time(value) for value in series.to_list()], dtype=pl.Time),
                logical_type,
            )
        if logical_type == "duration":
            return self._validate_conversion(
                series,
                pl.Series(series.name, [self._parse_duration(value) for value in series.to_list()], dtype=pl.Duration),
                logical_type,
            )
        return series

    def _validate_conversion(self, original: pl.Series, converted: pl.Series, logical_type: str) -> pl.Series:
        before_non_null = original.len() - int(original.null_count())
        after_non_null = converted.len() - int(converted.null_count())
        if after_non_null < before_non_null:
            raise ValueError(f"Column '{original.name}' could not be safely converted to {logical_type}.")
        return converted

    def _parse_boolean(self, value: object) -> bool | None:
        if value is None:
            return None
        if isinstance(value, bool):
            return value
        text = str(value).strip().lower()
        if text in {"true", "1", "yes", "y"}:
            return True
        if text in {"false", "0", "no", "n"}:
            return False
        return None

    def _parse_datetime(self, value: object) -> datetime | None:
        if value is None:
            return None
        if isinstance(value, datetime):
            return value
        text = str(value).strip()
        if not text:
            return None
        for parser in (
            lambda: datetime.fromisoformat(text),
            lambda: datetime.strptime(text, "%Y-%m-%d %H:%M"),
            lambda: datetime.strptime(text, "%Y-%m-%d %H:%M:%S"),
            lambda: datetime.strptime(text, "%d.%m.%Y %H:%M"),
            lambda: datetime.strptime(text, "%m/%d/%Y %H:%M"),
        ):
            try:
                return parser()
            except ValueError:
                continue
        return None

    def _parse_date(self, value: object) -> date | None:
        if value is None:
            return None
        if isinstance(value, date) and not isinstance(value, datetime):
            return value
        text = str(value).strip()
        if not text:
            return None
        for parser in (
            lambda: date.fromisoformat(text),
            lambda: datetime.strptime(text, "%Y-%m-%d").date(),
            lambda: datetime.strptime(text, "%d.%m.%Y").date(),
            lambda: datetime.strptime(text, "%m/%d/%Y").date(),
        ):
            try:
                return parser()
            except ValueError:
                continue
        return None

    def _parse_time(self, value: object) -> time | None:
        if value is None:
            return None
        if isinstance(value, time):
            return value
        text = str(value).strip()
        if not text:
            return None
        for parser in (
            lambda: time.fromisoformat(text),
            lambda: datetime.strptime(text, "%H:%M").time(),
            lambda: datetime.strptime(text, "%H:%M:%S").time(),
        ):
            try:
                return parser()
            except ValueError:
                continue
        return None

    def _parse_duration(self, value: object) -> timedelta | None:
        if value is None:
            return None
        if isinstance(value, timedelta):
            return value
        text = str(value).strip()
        if not text:
            return None
        if text.isdigit():
            return timedelta(seconds=int(text))
        try:
            hours, minutes, seconds = [int(part) for part in text.split(":")]
            return timedelta(hours=hours, minutes=minutes, seconds=seconds)
        except Exception:
            return None
