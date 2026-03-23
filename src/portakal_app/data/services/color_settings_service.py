from __future__ import annotations

from dataclasses import replace

import polars as pl

from portakal_app.data.models import DatasetHandle


class ColorSettingsService:
    DISCRETE_COLORS = (
        "#4db7eb",
        "#eb5a46",
        "#4caf50",
        "#f4b942",
        "#8b6ad9",
        "#f08080",
        "#3fb6a8",
        "#7f8c8d",
    )
    GRADIENTS = {
        "Citrus": ("#1d4ed8", "#22c55e", "#fde047"),
        "Sunset": ("#4338ca", "#f97316", "#facc15"),
        "Forest": ("#0f766e", "#65a30d", "#facc15"),
        "Berry": ("#7c3aed", "#ec4899", "#f9a8d4"),
    }

    def build_state(self, dataset: DatasetHandle | None) -> dict[str, object]:
        if dataset is None:
            return {"discrete": {}, "numeric": {}}

        existing = dataset.annotations.get("color_settings")
        existing_discrete = {}
        existing_numeric = {}
        if isinstance(existing, dict):
            existing_discrete = existing.get("discrete", {}) if isinstance(existing.get("discrete"), dict) else {}
            existing_numeric = existing.get("numeric", {}) if isinstance(existing.get("numeric"), dict) else {}

        discrete: dict[str, dict[str, str]] = {}
        numeric: dict[str, str] = {}
        for column in dataset.domain.columns:
            if self._is_discrete(dataset, column.name):
                values = self._discrete_values(dataset.dataframe.get_column(column.name))
                mapping: dict[str, str] = {}
                previous = existing_discrete.get(column.name, {})
                for index, value in enumerate(values):
                    if isinstance(previous, dict) and value in previous:
                        mapping[value] = str(previous[value])
                    else:
                        mapping[value] = self.DISCRETE_COLORS[index % len(self.DISCRETE_COLORS)]
                if mapping:
                    discrete[column.name] = mapping
            elif column.logical_type == "numeric":
                palette_name = existing_numeric.get(column.name)
                numeric[column.name] = palette_name if palette_name in self.GRADIENTS else "Citrus"
        return {"discrete": discrete, "numeric": numeric}

    def apply(self, dataset: DatasetHandle, state: dict[str, object]) -> DatasetHandle:
        annotations = dict(dataset.annotations)
        annotations["color_settings"] = state
        return replace(dataset, annotations=annotations)

    def _is_discrete(self, dataset: DatasetHandle, column_name: str) -> bool:
        column = next((item for item in dataset.domain.columns if item.name == column_name), None)
        if column is None:
            return False
        if column.logical_type in {"categorical", "boolean"}:
            return True
        if column.logical_type == "text":
            values = self._discrete_values(dataset.dataframe.get_column(column.name))
            return 0 < len(values) <= 12
        return False

    def _discrete_values(self, series: pl.Series) -> tuple[str, ...]:
        values: list[str] = []
        for value in series.drop_nulls().cast(pl.String).unique(maintain_order=True).to_list():
            text = str(value).strip()
            if not text:
                continue
            values.append(text)
            if len(values) == 12:
                break
        return tuple(values)
