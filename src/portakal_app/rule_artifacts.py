from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from portakal_app.data.models import DatasetHandle


def _is_missing(value: Any) -> bool:
    return value is None or (isinstance(value, float) and np.isnan(value))


def _display_value(value: Any) -> str:
    if _is_missing(value):
        return ""
    return str(value)


@dataclass(frozen=True)
class RuleConditionArtifact:
    attribute: str
    operator: str
    value: Any

    def as_text(self) -> str:
        op_map = {
            "==": "=",
            "!=": "\u2260",
            "<=": "\u2264",
            ">=": "\u2265",
        }
        return f"{self.attribute}{op_map.get(self.operator, self.operator)}{_display_value(self.value)}"

    def evaluate_values(self, values: list[Any]) -> np.ndarray:
        mask = np.zeros(len(values), dtype=bool)
        for index, value in enumerate(values):
            if _is_missing(value):
                continue
            if self.operator == "==":
                mask[index] = value == self.value
            elif self.operator == "!=":
                mask[index] = value != self.value
            elif self.operator == "<=":
                try:
                    mask[index] = float(value) <= float(self.value)
                except (TypeError, ValueError):
                    mask[index] = False
            elif self.operator == ">=":
                try:
                    mask[index] = float(value) >= float(self.value)
                except (TypeError, ValueError):
                    mask[index] = False
            elif self.operator == "<":
                try:
                    mask[index] = float(value) < float(self.value)
                except (TypeError, ValueError):
                    mask[index] = False
            elif self.operator == ">":
                try:
                    mask[index] = float(value) > float(self.value)
                except (TypeError, ValueError):
                    mask[index] = False
        return mask

    def evaluate_dataset(self, dataset: DatasetHandle) -> np.ndarray:
        if self.attribute not in dataset.dataframe.columns:
            return np.zeros(dataset.row_count, dtype=bool)
        values = dataset.dataframe.get_column(self.attribute).to_list()
        return self.evaluate_values(values)


@dataclass(frozen=True)
class CN2RuleArtifact:
    target_name: str
    prediction: str
    selectors: tuple[RuleConditionArtifact, ...] = ()
    curr_class_dist: tuple[float, ...] = ()
    probabilities: tuple[float, ...] = ()
    quality: float = 0.0
    covered_count: int = 0
    learning_covered_indices: tuple[int, ...] = ()
    is_default: bool = False

    @property
    def length(self) -> int:
        return len(self.selectors)

    def evaluate_data(self, dataset: DatasetHandle) -> np.ndarray:
        if not self.selectors:
            return np.ones(dataset.row_count, dtype=bool)
        mask = np.ones(dataset.row_count, dtype=bool)
        for selector in self.selectors:
            mask &= selector.evaluate_dataset(dataset)
        return mask

    def conditions_text(self, compact_view: bool = False) -> str:
        if not self.selectors:
            return "TRUE"
        delimiter = " AND " if compact_view else " AND\n"
        return delimiter.join(selector.as_text() for selector in self.selectors)

    def prediction_text(self) -> str:
        return f"{self.target_name}={self.prediction}"

    def to_text(self) -> str:
        return f"IF {self.conditions_text(compact_view=True)} THEN {self.prediction_text()}"


@dataclass
class CN2RuleClassifierArtifact:
    classifier_id: str
    display_name: str
    instances: DatasetHandle | None
    original_feature_names: tuple[str, ...]
    target_name: str
    class_values: tuple[str, ...]
    rule_list: list[CN2RuleArtifact]
    params: dict[str, object] = field(default_factory=dict)

    def can_apply_to(self, dataset: DatasetHandle | None) -> bool:
        if dataset is None:
            return False
        feature_names = tuple(column.name for column in dataset.domain.feature_columns)
        return feature_names == self.original_feature_names
