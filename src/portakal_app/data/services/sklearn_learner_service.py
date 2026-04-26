from __future__ import annotations

import uuid
from typing import Any

import numpy as np
import polars as pl

from portakal_app.data.models import DatasetHandle
from portakal_app.sklearn_model_artifacts import SklearnModelArtifact


def _safe_unique(series: pl.Series) -> list[Any]:
    try:
        return series.drop_nulls().unique(maintain_order=True).to_list()
    except TypeError:
        return series.drop_nulls().unique().to_list()


def _mode(series: pl.Series) -> Any:
    values = series.drop_nulls().to_list()
    if not values:
        return None
    counts: dict[Any, int] = {}
    best, best_n = values[0], 0
    for v in values:
        counts[v] = counts.get(v, 0) + 1
        if counts[v] > best_n:
            best, best_n = v, counts[v]
    return best


def _is_missing(v: Any) -> bool:
    return v is None or (isinstance(v, float) and np.isnan(v))


class SklearnLearnerService:
    """Encode a DatasetHandle and fit any sklearn estimator."""

    def encode_X(
        self,
        dataset: DatasetHandle,
        feature_names: tuple[str, ...],
        categorical_encoders: dict[str, list],
        numeric_cols: tuple[str, ...],
        numeric_means: dict[str, float],
    ) -> np.ndarray:
        df = dataset.dataframe
        blocks: list[np.ndarray] = []
        for name in feature_names:
            col = df.get_column(name)
            if name in categorical_encoders:
                cats = categorical_encoders[name]
                rows = []
                for v in col.to_list():
                    resolved = _mode(col) if _is_missing(v) else v
                    rows.append([1.0 if resolved == c else 0.0 for c in cats[1:]])
                matrix = np.asarray(rows, dtype=float) if len(cats) > 1 else np.zeros((col.len(), 0))
                blocks.append(matrix)
            else:
                raw = [float(v) if not _is_missing(v) else np.nan for v in col.to_list()]
                arr = np.asarray(raw, dtype=float)
                mean = numeric_means.get(name, 0.0)
                arr = np.where(np.isfinite(arr), arr, mean)
                blocks.append(arr.reshape(-1, 1))
        if not blocks:
            return np.zeros((dataset.row_count, 0), dtype=float)
        return np.concatenate(blocks, axis=1)

    def fit(
        self,
        estimator: object,
        dataset: DatasetHandle,
        display_name: str,
        model_type: str,
        params: dict | None = None,
    ) -> SklearnModelArtifact:
        from sklearn.base import clone

        target_cols = dataset.domain.target_columns
        if not target_cols:
            raise ValueError("No target column. Assign one with Select Columns or Edit Domain.")
        target_col = target_cols[0]
        target_name = target_col.name
        is_classifier = target_col.logical_type in {"categorical", "boolean"}

        df = dataset.dataframe
        feature_columns = [c for c in dataset.domain.feature_columns if c.logical_type in {"numeric", "categorical", "boolean"}]
        if not feature_columns:
            raise ValueError("No supported feature columns (numeric/categorical).")

        feature_names: list[str] = []
        categorical_encoders: dict[str, list] = {}
        numeric_cols: list[str] = []
        numeric_means: dict[str, float] = {}

        for col in feature_columns:
            feature_names.append(col.name)
            series = df.get_column(col.name)
            if col.logical_type in {"categorical", "boolean"}:
                cats = _safe_unique(series)
                if not cats:
                    cats = [False, True] if col.logical_type == "boolean" else ["(empty)"]
                categorical_encoders[col.name] = cats
            else:
                raw = [float(v) if not _is_missing(v) else np.nan for v in series.to_list()]
                arr = np.asarray(raw, dtype=float)
                finite = arr[np.isfinite(arr)]
                numeric_means[col.name] = float(np.mean(finite)) if finite.size else 0.0
                numeric_cols.append(col.name)

        X = self.encode_X(
            dataset,
            tuple(feature_names),
            categorical_encoders,
            tuple(numeric_cols),
            numeric_means,
        )

        target_series = df.get_column(target_name)
        target_encoder: dict[str, int] | None = None
        class_values: tuple[str, ...] | None = None

        if is_classifier:
            raw_classes = _safe_unique(target_series)
            class_values = tuple(str(v) for v in raw_classes)
            target_encoder = {str(v): i for i, v in enumerate(raw_classes)}
            y = np.asarray(
                [target_encoder.get(str(v), 0) if not _is_missing(v) else 0 for v in target_series.to_list()],
                dtype=int,
            )
        else:
            y = np.asarray(
                [float(v) if not _is_missing(v) else np.nan for v in target_series.to_list()],
                dtype=float,
            )
            finite_y = y[np.isfinite(y)]
            mean_y = float(np.mean(finite_y)) if finite_y.size else 0.0
            y = np.where(np.isfinite(y), y, mean_y)

        unfitted = clone(estimator)
        trained = clone(estimator)
        trained.fit(X, y)

        artifact = SklearnModelArtifact(
            model_id=str(uuid.uuid4()),
            display_name=display_name,
            model_type=model_type,
            sklearn_estimator=unfitted,
            trained_model=trained,
            feature_names=tuple(feature_names),
            categorical_encoders=categorical_encoders,
            numeric_cols=tuple(numeric_cols),
            target_name=target_name,
            is_classifier=is_classifier,
            class_values=class_values,
            target_encoder=target_encoder,
            params=params or {},
            training_dataset=dataset,
        )
        return artifact
