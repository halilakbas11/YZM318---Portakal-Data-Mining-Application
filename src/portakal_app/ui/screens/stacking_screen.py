from __future__ import annotations

from PySide6.QtWidgets import (
    QGroupBox,
    QLabel,
    QVBoxLayout,
)

from portakal_app.data.services.sklearn_learner_service import SklearnLearnerService
from portakal_app.models import WorkflowPayload
from portakal_app.ui.screens.model_base import ModelScreenBase


# Artifact types accepted on the Model / Aggregate ports.
_MODEL_ARTIFACT_TYPES: tuple[type, ...] = ()
try:
    from portakal_app.sklearn_model_artifacts import SklearnModelArtifact
    _MODEL_ARTIFACT_TYPES += (SklearnModelArtifact,)
except ImportError:
    pass
try:
    from portakal_app.tree_artifacts import DecisionTreeArtifact, RandomForestArtifact
    _MODEL_ARTIFACT_TYPES += (DecisionTreeArtifact, RandomForestArtifact)
except ImportError:
    pass


def _is_model_artifact(value: object) -> bool:
    return isinstance(value, _MODEL_ARTIFACT_TYPES)


def _extract_unfitted_estimator(artifact: object) -> object | None:
    """Return an unfitted sklearn estimator clone-able for re-fitting per fold."""
    from sklearn.base import clone

    est = getattr(artifact, "sklearn_estimator", None)
    if est is not None:
        try:
            return clone(est)
        except Exception:
            return est
    trained = getattr(artifact, "trained_model", None)
    if trained is not None:
        try:
            return clone(trained)  # clone resets fit state but keeps hyperparams
        except Exception:
            return None
    return None


class StackingScreen(ModelScreenBase):
    """Stacking — combine predictions of multiple models via a meta-learner."""

    _OUTPUT_PORT_LABEL = "Model"

    def __init__(self, parent=None) -> None:
        self._svc = SklearnLearnerService()
        # Keyed by id(artifact) so multiple instances of the same model type stay distinct.
        self._base_artifacts: dict[int, object] = {}
        self._aggregate_artifact = None
        super().__init__(parent)

    def _add_main_layout(self, layout: QVBoxLayout) -> None:
        info_box = QGroupBox("Base Models")
        info_layout = QVBoxLayout(info_box)
        self._models_label = QLabel("Connect models to the 'Model' input port.")
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
        from portakal_app.data.models import DatasetHandle

        if payload is None:
            self._dataset = None
            self._base_artifacts.clear()
            self._aggregate_artifact = None
            self._agg_label.setText("No aggregate learner. Using logistic regression by default.")
            self._extra_inputs = {}
        elif payload.port_label == "Data" and isinstance(payload.value, DatasetHandle):
            self._dataset = payload.value
        elif payload.port_label == "Aggregate" and _is_model_artifact(payload.value):
            self._aggregate_artifact = payload.value
            display = getattr(payload.value, "display_name", "Aggregate")
            self._agg_label.setText(f"Aggregate: {display}")
        elif _is_model_artifact(payload.value):
            # All base models arrive via the "Model" multi-input channel.
            # Key by id() so 3x same-type widgets (e.g. 3 Naive Bayes) stay distinct.
            self._base_artifacts[id(payload.value)] = payload.value
        else:
            self._extra_inputs[payload.port_label] = payload.value

        self._update_models_label()
        self._apply()

    def _update_models_label(self) -> None:
        if self._base_artifacts:
            names = ", ".join(
                getattr(a, "display_name", type(a).__name__)
                for a in self._base_artifacts.values()
            )
            self._models_label.setText(f"Base models ({len(self._base_artifacts)}): {names}")
        else:
            self._models_label.setText("Connect models to the 'Model' input port.")

    def _train(self):
        from sklearn.ensemble import StackingClassifier, StackingRegressor

        ds = self._dataset
        target_cols = ds.domain.target_columns
        if not target_cols:
            raise ValueError("No target column.")
        if not self._base_artifacts:
            raise ValueError("Connect at least one base model.")

        is_clf = target_cols[0].logical_type in {"categorical", "boolean"}

        estimators: list[tuple[str, object]] = []
        skipped: list[str] = []
        for i, art in enumerate(self._base_artifacts.values()):
            est = _extract_unfitted_estimator(art)
            if est is None:
                skipped.append(getattr(art, "display_name", type(art).__name__))
                continue
            estimators.append((f"base_{i}", est))

        if not estimators:
            raise ValueError(
                "No usable sklearn estimators among the connected base models."
            )

        final_estimator = None
        if self._aggregate_artifact is not None:
            final_estimator = _extract_unfitted_estimator(self._aggregate_artifact)

        kw: dict = {"estimators": estimators}
        if final_estimator is not None:
            kw["final_estimator"] = final_estimator

        est = StackingClassifier(**kw) if is_clf else StackingRegressor(**kw)
        params = {
            "base_models": [
                getattr(a, "display_name", type(a).__name__)
                for a in self._base_artifacts.values()
            ],
        }
        if skipped:
            params["skipped"] = skipped
        return self._svc.fit(est, ds, "Stacking", "stacking", params)

    def serialize_node_state(self) -> dict:
        return {**super().serialize_node_state()}

    def restore_node_state(self, payload: dict) -> None:
        super().restore_node_state(payload)
