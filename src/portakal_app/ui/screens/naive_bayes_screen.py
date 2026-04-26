from __future__ import annotations

from PySide6.QtWidgets import QLabel, QVBoxLayout

from portakal_app.data.services.sklearn_learner_service import SklearnLearnerService
from portakal_app.ui.screens.model_base import ModelScreenBase


class NaiveBayesScreen(ModelScreenBase):
    """Gaussian Naive Bayes — fast probabilistic classifier."""

    _OUTPUT_PORT_LABEL = "Classifier"

    def __init__(self, parent=None) -> None:
        self._svc = SklearnLearnerService()
        super().__init__(parent)

    def _add_main_layout(self, layout: QVBoxLayout) -> None:
        lbl = QLabel(
            "Naive Bayes with the assumption of feature independence. "
            "Supports only classification tasks. No configurable parameters."
        )
        lbl.setWordWrap(True)
        lbl.setStyleSheet("color: #555; background: transparent;")
        layout.addWidget(lbl)

    def _train(self):
        from sklearn.naive_bayes import GaussianNB

        ds = self._dataset
        target_cols = ds.domain.target_columns
        if not target_cols:
            raise ValueError("No target column.")
        if target_cols[0].logical_type not in {"categorical", "boolean"}:
            raise ValueError("Naive Bayes supports only classification targets.")
        return self._svc.fit(GaussianNB(), ds, "Naive Bayes", "naive_bayes", {})
