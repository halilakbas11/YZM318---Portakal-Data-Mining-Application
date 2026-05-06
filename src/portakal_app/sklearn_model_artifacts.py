from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from portakal_app.data.models import DatasetHandle


@dataclass
class SklearnModelArtifact:
    """Trained sklearn model with encoding metadata for prediction."""

    model_id: str
    display_name: str
    model_type: str
    sklearn_estimator: object
    trained_model: object | None
    feature_names: tuple[str, ...]
    categorical_encoders: dict[str, list]
    numeric_cols: tuple[str, ...]
    target_name: str | None
    is_classifier: bool
    class_values: tuple[str, ...] | None
    target_encoder: dict[str, int] | None
    numeric_means: dict[str, float] = field(default_factory=dict)
    params: dict = field(default_factory=dict)
    training_dataset: DatasetHandle | None = None

    def can_apply_to(self, dataset: DatasetHandle | None) -> bool:
        if dataset is None or not self.feature_names:
            return False
        col_names = {col.name for col in dataset.domain.feature_columns}
        return all(name in col_names for name in self.feature_names)

    def predict_proba(self, X: np.ndarray) -> np.ndarray | None:
        if self.trained_model is None:
            return None
        try:
            return self.trained_model.predict_proba(X)
        except AttributeError:
            return None

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.trained_model is None:
            raise RuntimeError("Model has not been trained.")
        return self.trained_model.predict(X)

    def predict_from_dataset(self, dataset: DatasetHandle) -> np.ndarray:
        from portakal_app.data.services.sklearn_learner_service import SklearnLearnerService

        X = SklearnLearnerService().encode_X(
            dataset,
            self.feature_names,
            self.categorical_encoders,
            self.numeric_cols,
            self.numeric_means,
        )
        return self.predict(X)
