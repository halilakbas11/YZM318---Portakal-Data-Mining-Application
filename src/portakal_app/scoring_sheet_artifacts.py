from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np

from portakal_app.data.models import DatasetHandle


def _is_missing(value: Any) -> bool:
    return value is None or (isinstance(value, float) and np.isnan(value))


@dataclass(frozen=True)
class ScoringSheetRuleArtifact:
    display_name: str
    source_feature: str
    operator: str
    reference_value: Any
    points: int

    def matches_row(self, row: Mapping[str, Any]) -> bool:
        value = row.get(self.source_feature)
        if _is_missing(value):
            return False
        if self.operator == "==":
            return value == self.reference_value
        try:
            numeric_value = float(value)
            reference = float(self.reference_value)
        except (TypeError, ValueError):
            return False
        if self.operator == "<=":
            return numeric_value <= reference
        if self.operator == ">":
            return numeric_value > reference
        return False


@dataclass(frozen=True)
class ScoringSheetClassifierArtifact:
    classifier_id: str
    display_name: str
    instances: DatasetHandle
    target_name: str
    class_values: tuple[str, str]
    rules: tuple[ScoringSheetRuleArtifact, ...]
    logistic_bias: float
    logistic_weight: float
    base_scores: tuple[float, ...]
    base_risks: tuple[float, ...]
    params: dict[str, object] = field(default_factory=dict)

    @property
    def original_feature_names(self) -> tuple[str, ...]:
        return tuple(rule.source_feature for rule in self.rules)

    @property
    def base_coefficients(self) -> tuple[int, ...]:
        return tuple(int(rule.points) for rule in self.rules)

    @property
    def rule_names(self) -> tuple[str, ...]:
        return tuple(rule.display_name for rule in self.rules)

    def can_apply_to(self, dataset: DatasetHandle | None) -> bool:
        if dataset is None:
            return False
        feature_names = {column.name for column in dataset.domain.feature_columns}
        return all(rule.source_feature in feature_names for rule in self.rules)

    def first_row_matches(self, dataset: DatasetHandle | None) -> list[int]:
        if dataset is None or dataset.row_count == 0:
            return [0] * len(self.rules)
        row = dataset.dataframe.row(0, named=True)
        return [1 if rule.matches_row(row) else 0 for rule in self.rules]

    def score_row(self, row: Mapping[str, Any]) -> int:
        return int(sum(rule.points for rule in self.rules if rule.matches_row(row)))

    def probability_for_score(self, score: float) -> float:
        value = float(np.clip(self.logistic_bias + self.logistic_weight * score, -40.0, 40.0))
        return float(1.0 / (1.0 + np.exp(-value)))

    def class_view(self, class_index: int) -> tuple[list[int], list[float], list[float]]:
        if class_index == 1:
            return (
                list(self.base_coefficients),
                list(self.base_scores),
                list(self.base_risks),
            )
        coefficients = [-coef for coef in self.base_coefficients]
        scores = sorted([-score if score != 0 else score for score in self.base_scores])
        risks = sorted([100.0 - risk for risk in self.base_risks])
        return coefficients, scores, risks

    @property
    def is_classifier(self) -> bool:
        return True

    def predict_from_dataset(self, dataset: DatasetHandle) -> np.ndarray:
        """Return integer class-index predictions for each row."""
        class_to_idx = {v: i for i, v in enumerate(self.class_values)}
        preds: list[int] = []
        for row_idx in range(dataset.row_count):
            row = dataset.dataframe.row(row_idx, named=True)
            score = self.score_row(row)
            prob = self.probability_for_score(score)
            pred_class = self.class_values[1] if prob >= 0.5 else self.class_values[0]
            preds.append(class_to_idx.get(pred_class, 0))
        return np.asarray(preds, dtype=int)
