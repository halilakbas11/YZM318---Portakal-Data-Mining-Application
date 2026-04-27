from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import polars as pl

from portakal_app.data.models import DatasetHandle
from portakal_app.logistic_regression_artifacts import (
    LogisticRegressionClassifierArtifact,
    LogisticRegressionFeatureArtifact,
)


def _sigmoid(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def _is_missing(value: Any) -> bool:
    return value is None or (isinstance(value, float) and np.isnan(value))


def _safe_unique_values(series: pl.Series) -> list[Any]:
    try:
        values = series.drop_nulls().unique(maintain_order=True).to_list()
    except TypeError:
        values = series.drop_nulls().unique().to_list()
    return list(values)


def _mode_value(series: pl.Series) -> Any:
    values = series.drop_nulls().to_list()
    if not values:
        return None
    counts: dict[Any, int] = {}
    best_value = values[0]
    best_count = 0
    for value in values:
        counts[value] = counts.get(value, 0) + 1
        if counts[value] > best_count:
            best_value = value
            best_count = counts[value]
    return best_value


@dataclass(frozen=True)
class LogisticRegressionSettings:
    max_iter: int = 200
    ridge: float = 1.0
    tolerance: float = 1e-6


@dataclass(frozen=True)
class _EncodedFeature:
    artifact: LogisticRegressionFeatureArtifact
    matrix: np.ndarray
    width: int


class LogisticRegressionService:
    def fit(
        self,
        dataset: DatasetHandle,
        settings: LogisticRegressionSettings | None = None,
    ) -> LogisticRegressionClassifierArtifact:
        config = settings or LogisticRegressionSettings()
        target_columns = dataset.domain.target_columns
        if len(target_columns) != 1:
            raise ValueError("Logistic Regression requires exactly one target column.")

        target_column = target_columns[0]
        if target_column.logical_type not in {"categorical", "boolean"}:
            raise ValueError("Logistic Regression currently supports only binary categorical targets.")

        dataframe = dataset.dataframe
        target_series = dataframe.get_column(target_column.name)
        class_values = [str(value) for value in _safe_unique_values(target_series)]
        if len(class_values) != 2:
            raise ValueError("Logistic Regression currently supports exactly two target classes.")

        feature_columns = list(dataset.domain.feature_columns)
        encoded_features: list[_EncodedFeature] = []
        original_feature_names: list[str] = []
        design_blocks: list[np.ndarray] = []
        
        cat_encoders: dict[str, list] = {}
        num_cols: list[str] = []

        for column in feature_columns:
            series = dataframe.get_column(column.name)
            encoded = self._encode_feature(column.name, column.logical_type, series)
            if encoded is None:
                continue
            encoded_features.append(encoded)
            original_feature_names.append(column.name)
            if encoded.width:
                design_blocks.append(encoded.matrix)
            
            # Track encoders for Confusion Matrix compatibility
            if encoded.artifact.is_discrete:
                cat_encoders[column.name] = list(encoded.artifact.values)
            else:
                num_cols.append(column.name)

        if not encoded_features:
            raise ValueError("No supported feature columns found for Logistic Regression.")

        row_count = dataset.row_count
        matrix = np.concatenate(design_blocks, axis=1) if design_blocks else np.zeros((row_count, 0), dtype=float)
        
        # Target Encoder
        raw_classes = _safe_unique_values(target_series)
        target_encoder = {str(v): i for i, v in enumerate(raw_classes)}
        
        y = self._encode_target(target_series, class_values)
        weights = self._fit_binary(matrix, y, config)

        intercept = float(weights[0])
        offset = 1
        features: list[LogisticRegressionFeatureArtifact] = []
        for encoded in encoded_features:
            artifact = encoded.artifact
            if artifact.is_discrete:
                points = [0.0]
                if encoded.width:
                    points.extend(float(value) for value in weights[offset : offset + encoded.width])
                discrete_positive = tuple(points[: len(artifact.values)])
                if len(discrete_positive) < len(artifact.values):
                    discrete_positive = discrete_positive + (0.0,) * (len(artifact.values) - len(discrete_positive))
                features.append(
                    LogisticRegressionFeatureArtifact(
                        name=artifact.name,
                        logical_type=artifact.logical_type,
                        values=artifact.values,
                        discrete_points=(tuple(-point for point in discrete_positive), discrete_positive),
                        default_value=artifact.default_value,
                    )
                )
            else:
                coefficient = float(weights[offset]) if encoded.width else 0.0
                features.append(
                    LogisticRegressionFeatureArtifact(
                        name=artifact.name,
                        logical_type=artifact.logical_type,
                        coefficients=(-coefficient, coefficient),
                        mean_value=artifact.mean_value,
                        std_value=artifact.std_value,
                        min_value=artifact.min_value,
                        max_value=artifact.max_value,
                        default_value=artifact.default_value,
                    )
                )
            offset += encoded.width

        return LogisticRegressionClassifierArtifact(
            classifier_id=f"{dataset.dataset_id}-logistic-regression",
            display_name="Logistic Regression",
            instances=dataset,
            original_feature_names=tuple(original_feature_names),
            target_name=target_column.name,
            class_values=(class_values[0], class_values[1]),
            intercepts=(-intercept, intercept),
            features=tuple(features),
            feature_names=tuple(original_feature_names),
            categorical_encoders=cat_encoders,
            numeric_cols=tuple(num_cols),
            target_encoder=target_encoder,
            params={
                "max_iter": config.max_iter,
                "ridge": config.ridge,
                "tolerance": config.tolerance,
            },
        )

    def _encode_feature(self, name: str, logical_type: str, series: pl.Series) -> _EncodedFeature | None:
        if logical_type in {"numeric"}:
            values = [float(value) if not _is_missing(value) else float("nan") for value in series.to_list()]
            numeric = np.asarray(values, dtype=float)
            finite = numeric[np.isfinite(numeric)]
            if finite.size == 0:
                return _EncodedFeature(
                    artifact=LogisticRegressionFeatureArtifact(
                        name=name,
                        logical_type=logical_type,
                        mean_value=0.0,
                        std_value=1.0,
                        min_value=0.0,
                        max_value=0.0,
                        default_value=0.0,
                    ),
                    matrix=np.zeros((series.len(), 1), dtype=float),
                    width=1,
                )
            mean = float(np.mean(finite))
            std = float(np.std(finite))
            if std <= 1e-12:
                std = 1.0
            filled = np.where(np.isfinite(numeric), numeric, mean)
            standardized = ((filled - mean) / std).reshape(-1, 1)
            return _EncodedFeature(
                artifact=LogisticRegressionFeatureArtifact(
                    name=name,
                    logical_type=logical_type,
                    mean_value=mean,
                    std_value=std,
                    min_value=float(np.min(finite)),
                    max_value=float(np.max(finite)),
                    default_value=mean,
                ),
                matrix=standardized,
                width=1,
            )

        if logical_type not in {"categorical", "boolean"}:
            return None

        categories = _safe_unique_values(series)
        if not categories:
            categories = [False, True] if logical_type == "boolean" else ["(empty)"]
        default_value = _mode_value(series)
        if default_value is None:
            default_value = categories[0]
        rows = []
        for value in series.to_list():
            resolved = default_value if _is_missing(value) else value
            rows.append([1.0 if resolved == category else 0.0 for category in categories[1:]])
        matrix = np.asarray(rows, dtype=float) if len(categories) > 1 else np.zeros((series.len(), 0), dtype=float)
        return _EncodedFeature(
            artifact=LogisticRegressionFeatureArtifact(
                name=name,
                logical_type=logical_type,
                values=tuple(categories),
                default_value=default_value,
            ),
            matrix=matrix,
            width=max(0, len(categories) - 1),
        )

    def _encode_target(self, series: pl.Series, class_values: list[str]) -> np.ndarray:
        rows = []
        positive = class_values[1]
        fallback = class_values[0]
        for value in series.to_list():
            text = fallback if _is_missing(value) else str(value)
            rows.append(1.0 if text == positive else 0.0)
        return np.asarray(rows, dtype=float)

    def _fit_binary(self, matrix: np.ndarray, y: np.ndarray, settings: LogisticRegressionSettings) -> np.ndarray:
        row_count = matrix.shape[0]
        x = np.concatenate((np.ones((row_count, 1), dtype=float), matrix), axis=1)
        weights = np.zeros(x.shape[1], dtype=float)
        regularizer = np.eye(x.shape[1], dtype=float)
        regularizer[0, 0] = 0.0

        for _ in range(max(1, settings.max_iter)):
            logits = x @ weights
            probabilities = _sigmoid(logits)
            diagonal = np.clip(probabilities * (1.0 - probabilities), 1e-6, None)
            weighted_x = x * diagonal[:, None]
            hessian = x.T @ weighted_x + settings.ridge * regularizer
            gradient = x.T @ (probabilities - y) + settings.ridge * regularizer @ weights
            try:
                step = np.linalg.solve(hessian, gradient)
            except np.linalg.LinAlgError:
                step = np.linalg.pinv(hessian) @ gradient
            weights -= step
            if float(np.max(np.abs(step))) <= settings.tolerance:
                break
        return weights
