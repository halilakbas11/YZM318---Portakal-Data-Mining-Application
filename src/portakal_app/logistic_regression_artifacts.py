from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from portakal_app.data.models import DatasetHandle


def _is_missing(value: Any) -> bool:
    return value is None or (isinstance(value, float) and np.isnan(value))


@dataclass(frozen=True)
class LogisticRegressionFeatureArtifact:
    name: str
    logical_type: str
    coefficients: tuple[float, float] = (0.0, 0.0)
    values: tuple[Any, ...] = ()
    discrete_points: tuple[tuple[float, ...], tuple[float, ...]] = ()
    mean_value: float | None = None
    std_value: float | None = None
    min_value: float | None = None
    max_value: float | None = None
    default_value: Any = None

    @property
    def is_discrete(self) -> bool:
        return self.logical_type in {"categorical", "boolean"}

    def contribution(self, class_index: int, value: Any) -> float:
        if self.is_discrete:
            if not self.values:
                return 0.0
            try:
                active_index = self.values.index(value)
            except ValueError:
                active_index = self.values.index(self.default_value) if self.default_value in self.values else 0
            if self.discrete_points and class_index < len(self.discrete_points):
                points = self.discrete_points[class_index]
                if active_index < len(points):
                    return float(points[active_index])
            coefficient = self.coefficients[class_index]
            centered = active_index - (len(self.values) - 1) / 2
            return float(coefficient * centered)

        coefficient = self.coefficients[class_index]
        if _is_missing(value):
            value = self.default_value if self.default_value is not None else self.mean_value
        try:
            numeric_value = float(value)
        except (TypeError, ValueError):
            numeric_value = float(self.default_value if self.default_value is not None else self.mean_value or 0.0)
        mean = float(self.mean_value or 0.0)
        std = float(self.std_value or 1.0) or 1.0
        return float(coefficient * ((numeric_value - mean) / std))

    def contribution_values(self, class_index: int) -> tuple[np.ndarray, np.ndarray]:
        if self.is_discrete:
            values = np.asarray(list(range(len(self.values))), dtype=float)
            if self.discrete_points and class_index < len(self.discrete_points):
                contributions = np.asarray(self.discrete_points[class_index], dtype=float)
            else:
                contributions = np.asarray([self.contribution(class_index, value) for value in self.values], dtype=float)
            return values, contributions

        min_value = float(self.min_value if self.min_value is not None else self.mean_value or 0.0)
        max_value = float(self.max_value if self.max_value is not None else self.mean_value or 0.0)
        values = np.asarray([min_value, max_value], dtype=float)
        contributions = np.asarray([self.contribution(class_index, min_value), self.contribution(class_index, max_value)], dtype=float)
        return values, contributions

    def default_marker_value(self) -> Any:
        return self.default_value

    def raw_value_from_contribution(self, class_index: int, contribution: float) -> Any:
        if self.is_discrete:
            _, contributions = self.contribution_values(class_index)
            if contributions.size == 0 or not self.values:
                return self.default_value
            index = int(np.argmin(np.abs(contributions - contribution)))
            return self.values[index]

        coefficient = float(self.coefficients[class_index] or 0.0)
        if abs(coefficient) < 1e-12:
            return self.default_marker_value()
        mean = float(self.mean_value or 0.0)
        std = float(self.std_value or 1.0) or 1.0
        raw_value = mean + (float(contribution) / coefficient) * std
        min_value = float(self.min_value if self.min_value is not None else raw_value)
        max_value = float(self.max_value if self.max_value is not None else raw_value)
        return float(np.clip(raw_value, min(min_value, max_value), max(min_value, max_value)))

    def importance(self, class_index: int) -> float:
        _, contributions = self.contribution_values(class_index)
        if contributions.size == 0:
            return 0.0
        return float(np.max(contributions) - np.min(contributions))


@dataclass(frozen=True)
class LogisticRegressionClassifierArtifact:
    classifier_id: str
    display_name: str
    instances: DatasetHandle
    original_feature_names: tuple[str, ...]
    target_name: str
    class_values: tuple[str, str]
    intercepts: tuple[float, float]
    features: tuple[LogisticRegressionFeatureArtifact, ...]
    feature_names: tuple[str, ...] = ()
    categorical_encoders: dict = field(default_factory=dict)
    numeric_cols: tuple[str, ...] = ()
    target_encoder: dict | None = None
    params: dict[str, object] = field(default_factory=dict)

    def is_classifier(self) -> bool:
        return True

    def can_apply_to(self, dataset: DatasetHandle | None) -> bool:
        if dataset is None:
            return False
        feature_names = {column.name for column in dataset.domain.feature_columns}
        return all(name in feature_names for name in self.original_feature_names)

    def feature_by_name(self, name: str) -> LogisticRegressionFeatureArtifact:
        return next(feature for feature in self.features if feature.name == name)

    def predict(self, X: np.ndarray) -> np.ndarray:
        # Standardize X based on feature artifacts before predicting
        X_proc = X.copy()
        col_idx = 0
        for feature in self.features:
            if not feature.is_discrete:
                mean = feature.mean_value or 0.0
                std = feature.std_value or 1.0
                X_proc[:, col_idx] = (X_proc[:, col_idx] - mean) / std
                col_idx += 1
            else:
                # Skip the one-hot columns for this discrete feature
                width = max(0, len(feature.values) - 1)
                col_idx += width

        # Construct the weight vector
        weights = [self.intercepts[1]]
        for feature in self.features:
            if not feature.is_discrete:
                weights.append(feature.coefficients[1])
            else:
                if feature.discrete_points:
                    weights.extend(feature.discrete_points[1][1:])
                else:
                    weights.append(feature.coefficients[1])
        
        w = np.asarray(weights)
        X_bias = np.hstack([np.ones((X_proc.shape[0], 1)), X_proc])
        
        if X_bias.shape[1] != w.shape[0]:
            return np.zeros(X.shape[0], dtype=int)
            
        scores = X_bias @ w
        return (scores > 0).astype(int)

    def logits(self, marker_values: dict[str, Any]) -> np.ndarray:
        totals = np.asarray(self.intercepts, dtype=float)
        for feature in self.features:
            value = marker_values.get(feature.name, feature.default_marker_value())
            totals[0] += feature.contribution(0, value)
            totals[1] += feature.contribution(1, value)
        return totals

    def probabilities(self, marker_values: dict[str, Any]) -> np.ndarray:
        logits = self.logits(marker_values)
        positive = 1.0 / (1.0 + np.exp(-logits[1]))
        return np.asarray([1.0 - positive, positive], dtype=float)
