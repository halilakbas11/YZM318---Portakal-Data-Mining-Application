from __future__ import annotations

from PySide6.QtWidgets import QLabel, QVBoxLayout

from portakal_app.data.services.sklearn_learner_service import SklearnLearnerService
from portakal_app.ui.screens.model_base import ModelScreenBase


class ConstantScreen(ModelScreenBase):
    """Predicts the most frequent class or mean value — a baseline model."""

    _OUTPUT_PORT_LABEL = "Model"

    def __init__(self, parent=None) -> None:
        self._svc = SklearnLearnerService()
        super().__init__(parent)

    def _add_main_layout(self, layout: QVBoxLayout) -> None:
        lbl = QLabel(
            "This model always predicts the most frequent class (classification) "
            "or the mean target value (regression). No parameters."
        )
        lbl.setWordWrap(True)
        lbl.setStyleSheet("color: #555; background: transparent;")
        layout.addWidget(lbl)

    def _train(self):
        from sklearn.dummy import DummyClassifier, DummyRegressor

        ds = self._dataset
        target_cols = ds.domain.target_columns
        if not target_cols:
            raise ValueError("No target column.")
        is_clf = target_cols[0].logical_type in {"categorical", "boolean"}
        estimator = DummyClassifier(strategy="most_frequent") if is_clf else DummyRegressor(strategy="mean")
        return self._svc.fit(estimator, ds, "Constant", "constant", {})
