from __future__ import annotations

from dataclasses import replace

import numpy as np
import polars as pl
from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM

from portakal_app.data.models import DatasetHandle, build_data_domain


class OutliersService:
    def detect(
        self,
        dataset: DatasetHandle,
        *,
        method: str,
        contamination: float,
        neighbors: int,
        nu: float,
        gamma: str,
        support_fraction: float,
        metric: str = "euclidean",
        replicable: bool = True,
    ) -> dict[str, DatasetHandle]:
        matrix, columns = self._numeric_matrix(dataset)
        predictions, scores = self._predict(
            matrix,
            method=method,
            contamination=contamination,
            neighbors=neighbors,
            nu=nu,
            gamma=gamma,
            support_fraction=support_fraction,
            metric=metric,
            replicable=replicable,
        )

        outlier_mask = predictions == -1
        annotated = dataset.dataframe.with_columns(
            pl.Series("Outlier", outlier_mask.tolist()),
            pl.Series("Outlier score", [float(value) for value in scores.tolist()]),
        )
        annotations = {
            **dataset.annotations,
            "outliers": {
                "method": method,
                "feature_columns": list(columns),
                "contamination": contamination,
                "metric": metric,
                "replicable": replicable,
            },
        }
        annotated_dataset = replace(
            dataset,
            dataset_id=f"{dataset.dataset_id}-outliers",
            display_name=f"{dataset.display_name} (Outliers)",
            dataframe=annotated,
            row_count=annotated.height,
            column_count=annotated.width,
            domain=build_data_domain(annotated, source_domain=dataset.domain),
            annotations=annotations,
        )
        outliers_dataset = self._subset_dataset(
            annotated_dataset,
            outlier_mask,
            dataset_id=f"{dataset.dataset_id}-outliers-only",
            display_name=f"{dataset.display_name} (Outliers only)",
        )
        inliers_dataset = self._subset_dataset(
            annotated_dataset,
            ~outlier_mask,
            dataset_id=f"{dataset.dataset_id}-inliers-only",
            display_name=f"{dataset.display_name} (Inliers only)",
        )
        return {
            "Outliers": outliers_dataset,
            "Inliers": inliers_dataset,
            "Data": annotated_dataset,
        }

    def _numeric_matrix(self, dataset: DatasetHandle) -> tuple[np.ndarray, tuple[str, ...]]:
        columns = tuple(column.name for column in dataset.domain.feature_columns if column.logical_type == "numeric")
        if not columns:
            raise ValueError("Outliers needs at least one numeric feature.")
        matrix = dataset.dataframe.select(columns).to_numpy(allow_copy=True).astype(float)
        if matrix.shape[0] < 2:
            raise ValueError("Outliers needs at least two rows.")
        nan_mask = np.isnan(matrix)
        if nan_mask.any():
            means = np.nanmean(matrix, axis=0)
            means = np.where(np.isnan(means), 0.0, means)
            matrix[nan_mask] = np.take(means, np.where(nan_mask)[1])
        return matrix, columns

    def _predict(
        self,
        matrix: np.ndarray,
        *,
        method: str,
        contamination: float,
        neighbors: int,
        nu: float,
        gamma: str,
        support_fraction: float,
        metric: str,
        replicable: bool,
    ) -> tuple[np.ndarray, np.ndarray]:
        if method == "one_class_svm":
            estimator = OneClassSVM(nu=nu, gamma=gamma)
            predictions = estimator.fit_predict(matrix)
            scores = -estimator.decision_function(matrix)
            return predictions, np.asarray(scores, dtype=float)
        if method == "covariance":
            estimator = EllipticEnvelope(
                contamination=contamination,
                support_fraction=support_fraction,
                random_state=42,
            )
            predictions = estimator.fit_predict(matrix)
            scores = -estimator.decision_function(matrix)
            return predictions, np.asarray(scores, dtype=float)
        if method == "local_outlier_factor":
            effective_neighbors = max(1, min(neighbors, matrix.shape[0] - 1))
            kwargs: dict[str, object] = {"metric": metric}
            if metric == "mahalanobis":
                covariance = np.cov(matrix, rowvar=False)
                covariance = np.atleast_2d(covariance)
                kwargs["metric_params"] = {"VI": np.linalg.pinv(covariance)}
            estimator = LocalOutlierFactor(
                n_neighbors=effective_neighbors,
                contamination=contamination,
                **kwargs,
            )
            predictions = estimator.fit_predict(matrix)
            scores = -np.asarray(estimator.negative_outlier_factor_, dtype=float)
            return predictions, scores
        if method == "isolation_forest":
            estimator = IsolationForest(
                contamination=contamination,
                random_state=42 if replicable else None,
            )
            predictions = estimator.fit_predict(matrix)
            scores = -estimator.decision_function(matrix)
            return predictions, np.asarray(scores, dtype=float)
        raise ValueError(f"Unsupported outlier method: {method}")

    def _subset_dataset(
        self,
        dataset: DatasetHandle,
        mask: np.ndarray,
        *,
        dataset_id: str,
        display_name: str,
    ) -> DatasetHandle:
        frame = dataset.dataframe.filter(pl.Series(mask.tolist()))
        return replace(
            dataset,
            dataset_id=dataset_id,
            display_name=display_name,
            dataframe=frame,
            row_count=frame.height,
            column_count=frame.width,
            domain=build_data_domain(frame, source_domain=dataset.domain),
        )
