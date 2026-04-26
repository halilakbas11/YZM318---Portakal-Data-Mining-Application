from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Any

import numpy as np

from portakal_app.data.models import DatasetHandle
from portakal_app.data.services.sklearn_learner_service import SklearnLearnerService, _safe_unique, _is_missing
from portakal_app.data.services.tree_service import _build_nodes
from portakal_app.tree_artifacts import DecisionTreeArtifact, RandomForestArtifact


@dataclass(frozen=True)
class RandomForestSettings:
    n_estimators: int = 10
    use_max_features: bool = False
    max_features: int = 5
    use_random_state: bool = False
    use_max_depth: bool = False
    max_depth: int = 3
    use_min_samples_split: bool = True
    min_samples_split: int = 5
    class_weight: bool = False


class RandomForestService:
    def __init__(self) -> None:
        self._encoder = SklearnLearnerService()

    def fit(self, dataset: DatasetHandle, settings: RandomForestSettings | None = None) -> RandomForestArtifact:
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

        cfg = settings or RandomForestSettings()

        target_cols = dataset.domain.target_columns
        if not target_cols:
            raise ValueError("No target column.")
        target_col = target_cols[0]
        is_clf = target_col.logical_type in {"categorical", "boolean"}

        df = dataset.dataframe
        feature_columns = [c for c in dataset.domain.feature_columns if c.logical_type in {"numeric", "categorical", "boolean"}]
        if not feature_columns:
            raise ValueError("No supported feature columns.")

        feature_names: list[str] = []
        cat_encoders: dict[str, list] = {}
        num_cols: list[str] = []
        num_means: dict[str, float] = {}

        for col in feature_columns:
            feature_names.append(col.name)
            series = df.get_column(col.name)
            if col.logical_type in {"categorical", "boolean"}:
                cats = _safe_unique(series)
                if not cats:
                    cats = [False, True] if col.logical_type == "boolean" else ["(empty)"]
                cat_encoders[col.name] = cats
            else:
                raw = [float(v) if not _is_missing(v) else np.nan for v in series.to_list()]
                arr = np.asarray(raw, dtype=float)
                finite = arr[np.isfinite(arr)]
                num_means[col.name] = float(np.mean(finite)) if finite.size else 0.0
                num_cols.append(col.name)

        X = self._encoder.encode_X(dataset, tuple(feature_names), cat_encoders, tuple(num_cols), num_means)

        target_series = df.get_column(target_col.name)
        class_values: tuple[str, ...] = ()
        if is_clf:
            raw_classes = _safe_unique(target_series)
            class_values = tuple(str(v) for v in raw_classes)
            target_encoder = {str(v): i for i, v in enumerate(raw_classes)}
            y = np.asarray([target_encoder.get(str(v), 0) if not _is_missing(v) else 0 for v in target_series.to_list()], dtype=int)
        else:
            y_raw = [float(v) if not _is_missing(v) else np.nan for v in target_series.to_list()]
            y_arr = np.asarray(y_raw, dtype=float)
            mean_y = float(np.nanmean(y_arr)) if np.any(np.isfinite(y_arr)) else 0.0
            y = np.where(np.isfinite(y_arr), y_arr, mean_y)

        kwargs: dict[str, Any] = {"n_estimators": cfg.n_estimators}
        if cfg.use_max_features:
            kwargs["max_features"] = cfg.max_features
        if cfg.use_random_state:
            kwargs["random_state"] = 0
        if cfg.use_max_depth:
            kwargs["max_depth"] = cfg.max_depth
        if cfg.use_min_samples_split:
            kwargs["min_samples_split"] = cfg.min_samples_split
        if is_clf and cfg.class_weight:
            kwargs["class_weight"] = "balanced"

        if is_clf:
            forest = RandomForestClassifier(**kwargs)
        else:
            forest = RandomForestRegressor(**kwargs)
        forest.fit(X, y)

        expanded_names: list[str] = []
        for name in feature_names:
            if name in cat_encoders:
                for cat in cat_encoders[name][1:]:
                    expanded_names.append(f"{name}={cat}")
            else:
                expanded_names.append(name)

        trees: list[DecisionTreeArtifact] = []
        for i, estimator in enumerate(forest.estimators_):
            tree_nodes = _build_nodes(estimator, X, y, expanded_names, feature_names, class_values, is_clf)
            trees.append(DecisionTreeArtifact(
                tree_id=f"tree-{i}",
                display_name=f"Tree {i + 1}",
                instances=dataset,
                root_id=0,
                nodes=tree_nodes,
                target_name=target_col.name,
                kind="classification" if is_clf else "regression",
                class_values=class_values,
            ))

        return RandomForestArtifact(
            forest_id=str(uuid.uuid4()),
            display_name="Random Forest",
            instances=dataset,
            trees=trees,
            kind="classification" if is_clf else "regression",
            class_values=class_values,
            trained_model=forest,
            feature_names=tuple(feature_names),
            categorical_encoders=cat_encoders,
            numeric_cols=tuple(num_cols),
            target_name=target_col.name,
            target_encoder=target_encoder if is_clf else None,
        )
