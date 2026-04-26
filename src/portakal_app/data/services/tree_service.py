from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Any

import numpy as np
import polars as pl

from portakal_app.data.models import DatasetHandle
from portakal_app.data.services.sklearn_learner_service import SklearnLearnerService, _safe_unique, _is_missing, _mode
from portakal_app.tree_artifacts import DecisionTreeArtifact, TreeNodeArtifact


@dataclass(frozen=True)
class TreeSettings:
    binary_trees: bool = True
    limit_min_leaf: bool = True
    min_leaf: int = 2
    limit_min_internal: bool = True
    min_internal: int = 5
    limit_depth: bool = True
    max_depth: int = 100
    limit_majority: bool = True
    sufficient_majority: float = 0.95


class TreeService:
    def __init__(self) -> None:
        self._encoder = SklearnLearnerService()

    def fit(self, dataset: DatasetHandle, settings: TreeSettings | None = None) -> DecisionTreeArtifact:
        from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

        cfg = settings or TreeSettings()

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
        target_encoder: dict[str, int] = {}

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

        kwargs: dict[str, Any] = {
            "min_samples_leaf": cfg.min_leaf if cfg.limit_min_leaf else 1,
            "min_samples_split": cfg.min_internal if cfg.limit_min_internal else 2,
            "max_depth": cfg.max_depth if cfg.limit_depth else None,
        }
        if cfg.binary_trees:
            kwargs["splitter"] = "best"
        if is_clf and cfg.limit_majority:
            kwargs["min_impurity_decrease"] = 0.0

        if is_clf:
            clf = DecisionTreeClassifier(**kwargs)
        else:
            clf = DecisionTreeRegressor(**kwargs)

        clf.fit(X, y)

        # Build expanded feature name list for one-hot columns
        expanded_names: list[str] = []
        for name in feature_names:
            if name in cat_encoders:
                cats = cat_encoders[name]
                for cat in cats[1:]:
                    expanded_names.append(f"{name}={cat}")
            else:
                expanded_names.append(name)

        nodes = _build_nodes(clf, X, y, expanded_names, feature_names, class_values, is_clf)

        return DecisionTreeArtifact(
            tree_id=str(uuid.uuid4()),
            display_name="Tree",
            instances=dataset,
            root_id=0,
            nodes=nodes,
            target_name=target_col.name,
            kind="classification" if is_clf else "regression",
            class_values=class_values,
            trained_model=clf,
            feature_names=tuple(feature_names),
            categorical_encoders=cat_encoders,
            numeric_cols=tuple(num_cols),
            target_encoder=target_encoder if is_clf else None,
        )


def _build_nodes(
    clf: Any,
    X: np.ndarray,
    y: np.ndarray,
    expanded_names: list[str],
    feature_names: list[str],
    class_values: tuple[str, ...],
    is_clf: bool,
) -> dict[int, TreeNodeArtifact]:
    tree = clf.tree_
    n_nodes = tree.node_count
    TREE_LEAF = -1

    node_sample_indices: dict[int, list[int]] = {i: [] for i in range(n_nodes)}
    decision_path = clf.decision_path(X)
    for sample_idx in range(X.shape[0]):
        _, node_indices = decision_path[sample_idx].nonzero()
        for node_idx in node_indices:
            node_sample_indices[int(node_idx)].append(sample_idx)

    nodes: dict[int, TreeNodeArtifact] = {}
    for node_id in range(n_nodes):
        left = int(tree.children_left[node_id])
        right = int(tree.children_right[node_id])
        is_leaf = left == TREE_LEAF

        child_ids = [] if is_leaf else [left, right]
        split_attr: str | None = None
        if not is_leaf:
            feat_idx = int(tree.feature[node_id])
            split_attr = expanded_names[feat_idx] if feat_idx < len(expanded_names) else str(feat_idx)

        if is_clf:
            dist = tree.value[node_id][0]
            total = dist.sum()
            class_dist = tuple(float(d) / max(total, 1) for d in dist)
            pred = class_values[int(np.argmax(dist))] if class_values else int(np.argmax(dist))
        else:
            dist = tree.value[node_id][0]
            class_dist = (float(dist[0]),)
            pred = float(dist[0])

        nodes[node_id] = TreeNodeArtifact(
            node_id=node_id,
            sample_indices=node_sample_indices[node_id],
            child_ids=child_ids,
            split_attr=split_attr,
            class_distribution=class_dist,
            prediction=pred,
        )

    return nodes
