from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from portakal_app.data.models import DatasetHandle


@dataclass(frozen=True)
class DistanceMatrixHandle:
    matrix: np.ndarray
    metric: str
    metric_label: str
    axis: str
    axis_label: str
    row_labels: tuple[str, ...]
    feature_names: tuple[str, ...]
    source_dataset: DatasetHandle | None = None
    metadata: dict[str, object] = field(default_factory=dict)


def build_distance_matrix(
    matrix: object,
    *,
    metric: str = "precomputed",
    metric_label: str = "Precomputed",
    axis: str = "rows",
    axis_label: str = "Distances between rows",
    row_labels: tuple[str, ...] | None = None,
    feature_names: tuple[str, ...] | None = None,
    source_dataset: DatasetHandle | None = None,
    metadata: dict[str, object] | None = None,
) -> DistanceMatrixHandle:
    values = np.asarray(matrix, dtype=float)
    if values.ndim != 2 or values.shape[0] != values.shape[1]:
        raise ValueError("Distance Matrix must be square.")
    n_rows = int(values.shape[0])
    labels = row_labels or tuple(str(index + 1) for index in range(n_rows))
    features = feature_names or labels
    return DistanceMatrixHandle(
        matrix=values,
        metric=metric,
        metric_label=metric_label,
        axis=axis,
        axis_label=axis_label,
        row_labels=labels,
        feature_names=features,
        source_dataset=source_dataset,
        metadata=dict(metadata or {}),
    )


def coerce_distance_matrix(value: object) -> DistanceMatrixHandle:
    if isinstance(value, DistanceMatrixHandle):
        return value

    matrix = getattr(value, "matrix", None)
    if matrix is not None:
        return build_distance_matrix(
            matrix,
            metric=str(getattr(value, "metric", "precomputed")),
            metric_label=str(getattr(value, "metric_label", "Precomputed")),
            axis=str(getattr(value, "axis", "rows")),
            axis_label=str(getattr(value, "axis_label", "Distances between rows")),
            row_labels=tuple(str(label) for label in getattr(value, "row_labels", ()) or ()),
            feature_names=tuple(str(label) for label in getattr(value, "feature_names", ()) or ()),
            source_dataset=getattr(value, "source_dataset", None),
            metadata=dict(getattr(value, "metadata", {}) or {}),
        )

    return build_distance_matrix(value)

