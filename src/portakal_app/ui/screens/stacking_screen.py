from __future__ import annotations

from PySide6.QtWidgets import (
    QGroupBox,
    QLabel,
    QVBoxLayout,
)

from portakal_app.data.services.sklearn_learner_service import SklearnLearnerService
from portakal_app.models import WorkflowPayload
from portakal_app.ui.screens.model_base import ModelScreenBase


class StackingScreen(ModelScreenBase):
    """Stacking — combine predictions of multiple models via a meta-learner."""

    _OUTPUT_PORT_LABEL = "Model"

    def __init__(self, parent=None) -> None:
        self._svc = SklearnLearnerService()
        self._base_artifacts: dict[str, object] = {}
        self._aggregate_artifact = None
        super().__init__(parent)

    def _add_main_layout(self, layout: QVBoxLayout) -> None:
        info_box = QGroupBox("Base Models")
        info_layout = QVBoxLayout(info_box)
        self._models_label = QLabel("Connect models to the 'Model N' input ports.")
        self._models_label.setWordWrap(True)
        self._models_label.setStyleSheet("font-style: italic; background: transparent;")
        info_layout.addWidget(self._models_label)
        layout.addWidget(info_box)

        agg_box = QGroupBox("Aggregate (Meta-Learner)")
        agg_layout = QVBoxLayout(agg_box)
        self._agg_label = QLabel("No aggregate learner. Using logistic regression by default.")
        self._agg_label.setWordWrap(True)
        self._agg_label.setStyleSheet("font-style: italic; background: transparent;")
        agg_layout.addWidget(self._agg_label)
        layout.addWidget(agg_box)

    def set_input_payload(self, payload: WorkflowPayload | None) -> None:
        from portakal_app.sklearn_model_artifacts import SklearnModelArtifact
        from portakal_app.data.models import DatasetHandle

        if payload is None:
            self._dataset = None
            self._base_artifacts.clear()
            self._aggregate_artifact = None
            self._extra_inputs = {}
        elif payload.port_label == "Data" and isinstance(payload.value, DatasetHandle):
            self._dataset = payload.value
        elif payload.port_label == "Aggregate" and isinstance(payload.value, SklearnModelArtifact):
            self._aggregate_artifact = payload.value
            self._agg_label.setText(f"Aggregate: {payload.value.display_name}")
        elif isinstance(payload.value, SklearnModelArtifact):
            # All base models arrive via the "Model" multi-input channel; key by model_id
            key = getattr(payload.value, "model_id", None) or str(id(payload.value))
            self._base_artifacts[key] = payload.value
        else:
            self._extra_inputs[payload.port_label] = payload.value

        self._update_models_label()
        self._apply()

    def _update_models_label(self) -> None:
        if self._base_artifacts:
            names = ", ".join(a.display_name for a in self._base_artifacts.values())
            self._models_label.setText(f"Base models ({len(self._base_artifacts)}): {names}")
        else:
            self._models_label.setText("Connect models to the 'Model' input port.")

    def _train(self):
        from sklearn.ensemble import StackingClassifier, StackingRegressor
        from sklearn.base import clone

        ds = self._dataset
        target_cols = ds.domain.target_columns
        if not target_cols:
            raise ValueError("No target column.")
        if not self._base_artifacts:
            raise ValueError("Connect at least one base model.")

        is_clf = target_cols[0].logical_type in {"categorical", "boolean"}

        estimators = [
            (f"base_{i}", clone(art.sklearn_estimator))
            for i, (_, art) in enumerate(self._base_artifacts.items())
            if getattr(art, "sklearn_estimator", None) is not None
        ]
        if not estimators:
            raise ValueError("Base models must be sklearn-backed (e.g. from kNN, SVM, Neural Network).")

        final_estimator = None
        if self._aggregate_artifact and getattr(self._aggregate_artifact, "sklearn_estimator", None) is not None:
            final_estimator = clone(self._aggregate_artifact.sklearn_estimator)

        kw: dict = {"estimators": estimators}
        if final_estimator:
            kw["final_estimator"] = final_estimator

        est = StackingClassifier(**kw) if is_clf else StackingRegressor(**kw)
        params = {"base_models": [k for k, _ in estimators]}
        return self._svc.fit(est, ds, "Stacking", "stacking", params)

    def serialize_node_state(self) -> dict:
        return {**super().serialize_node_state()}

    def restore_node_state(self, payload: dict) -> None:
        super().restore_node_state(payload)
