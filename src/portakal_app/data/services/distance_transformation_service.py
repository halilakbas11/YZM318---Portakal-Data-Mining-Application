from __future__ import annotations

import numpy as np

from portakal_app.data.services.distance_matrix_service import DistanceMatrixHandle, build_distance_matrix, coerce_distance_matrix


class DistanceTransformationService:
    def transform(
        self,
        value: object,
        *,
        normalization: str,
        inversion: str,
    ) -> DistanceMatrixHandle:
        handle = coerce_distance_matrix(value)
        matrix = np.asarray(handle.matrix, dtype=float).copy()
        matrix = self._normalize(matrix, normalization)
        matrix = self._invert(matrix, inversion)
        np.fill_diagonal(matrix, 0.0)
        metadata = {
            **handle.metadata,
            "transformation": {
                "normalization": normalization,
                "inversion": inversion,
            },
        }
        return build_distance_matrix(
            matrix,
            metric=handle.metric,
            metric_label=handle.metric_label,
            axis=handle.axis,
            axis_label=handle.axis_label,
            row_labels=handle.row_labels,
            feature_names=handle.feature_names,
            source_dataset=handle.source_dataset,
            metadata=metadata,
        )

    def _normalize(self, matrix: np.ndarray, normalization: str) -> np.ndarray:
        if normalization == "none":
            return matrix
        minimum = float(np.min(matrix))
        maximum = float(np.max(matrix))
        span = maximum - minimum
        if span <= 1e-12:
            return np.zeros_like(matrix)
        scaled = (matrix - minimum) / span
        if normalization == "zero_one":
            return scaled
        if normalization == "minus_one_one":
            return scaled * 2.0 - 1.0
        if normalization == "sigmoid":
            centered = matrix - float(np.mean(matrix))
            return 1.0 / (1.0 + np.exp(-centered))
        raise ValueError(f"Unsupported normalization: {normalization}")

    def _invert(self, matrix: np.ndarray, inversion: str) -> np.ndarray:
        if inversion == "none":
            return matrix
        if inversion == "negate":
            return -matrix
        if inversion == "one_minus":
            return 1.0 - matrix
        if inversion == "max_minus":
            return float(np.max(matrix)) - matrix
        if inversion == "reciprocal":
            result = np.zeros_like(matrix)
            nonzero = np.abs(matrix) > 1e-12
            result[nonzero] = 1.0 / matrix[nonzero]
            return result
        raise ValueError(f"Unsupported inversion: {inversion}")

