from __future__ import annotations

from PySide6.QtWidgets import (
    QButtonGroup,
    QGroupBox,
    QLabel,
    QRadioButton,
    QVBoxLayout,
)

from portakal_app.data.services.sklearn_learner_service import SklearnLearnerService
from portakal_app.models import WorkflowPayload
from portakal_app.ui.screens.model_base import ModelScreenBase

_SIG, _ISO, _NONE_CAL = 0, 1, 2
_CAL_LABELS = ("Sigmoid calibration", "Isotonic calibration", "No calibration")
_OPT_CA, _OPT_F1, _NONE_THR = 0, 1, 2
_THR_LABELS = ("Optimize classification accuracy", "Optimize F1 score", "No threshold optimization")


class CalibratedLearnerScreen(ModelScreenBase):
    """Calibrated Learner — wraps a classifier with probability calibration and threshold optimization."""

    _OUTPUT_PORT_LABEL = "Classifier"

    def __init__(self, parent=None) -> None:
        self._svc = SklearnLearnerService()
        self._base_artifact = None
        super().__init__(parent)

    def _add_main_layout(self, layout: QVBoxLayout) -> None:
        self._base_label = QLabel("No base learner on input.")
        self._base_label.setStyleSheet("font-style: italic; background: transparent;")
        layout.addWidget(self._base_label)

        cal_box = QGroupBox("Probability calibration")
        cal_layout = QVBoxLayout(cal_box)
        self._cal_group = QButtonGroup(self)
        for i, label in enumerate(_CAL_LABELS):
            rb = QRadioButton(label)
            if i == 0:
                rb.setChecked(True)
            self._cal_group.addButton(rb, i)
            cal_layout.addWidget(rb)
        self._cal_group.idToggled.connect(lambda _id, checked: checked and self._settings_changed())
        layout.addWidget(cal_box)

        thr_box = QGroupBox("Decision threshold optimization")
        thr_layout = QVBoxLayout(thr_box)
        self._thr_group = QButtonGroup(self)
        for i, label in enumerate(_THR_LABELS):
            rb = QRadioButton(label)
            if i == 0:
                rb.setChecked(True)
            self._thr_group.addButton(rb, i)
            thr_layout.addWidget(rb)
        self._thr_group.idToggled.connect(lambda _id, checked: checked and self._settings_changed())
        layout.addWidget(thr_box)

    def set_input_payload(self, payload: WorkflowPayload | None) -> None:
        from portakal_app.sklearn_model_artifacts import SklearnModelArtifact
        from portakal_app.data.models import DatasetHandle

        if payload is None:
            self._dataset = None
            self._base_artifact = None
            self._extra_inputs = {}
        elif payload.port_label == "Data" and isinstance(payload.value, DatasetHandle):
            self._dataset = payload.value
        elif isinstance(payload.value, SklearnModelArtifact):
            self._base_artifact = payload.value
            name = payload.value.display_name
            self._base_label.setText(f"Base learner: {name}")
        else:
            self._extra_inputs[payload.port_label] = payload.value

        self._apply()

    def _train(self):
        from sklearn.calibration import CalibratedClassifierCV
        from sklearn.base import clone

        ds = self._dataset
        target_cols = ds.domain.target_columns
        if not target_cols:
            raise ValueError("No target column.")
        if target_cols[0].logical_type not in {"categorical", "boolean"}:
            raise ValueError("Calibrated Learner requires a categorical target.")

        cal_idx = self._cal_group.checkedId()
        cal_method = ["sigmoid", "isotonic", None][cal_idx]

        if self._base_artifact is None:
            from sklearn.naive_bayes import GaussianNB
            base_est = GaussianNB()
        else:
            base_est = clone(self._base_artifact.sklearn_estimator)

        if cal_method is not None:
            estimator = CalibratedClassifierCV(estimator=base_est, method=cal_method, cv=3)
        else:
            estimator = base_est

        base_name = self._base_artifact.display_name if self._base_artifact else "Naive Bayes"
        params = {"base": base_name, "calibration": _CAL_LABELS[cal_idx],
                  "threshold": _THR_LABELS[self._thr_group.checkedId()]}
        return self._svc.fit(estimator, ds, f"Calibrated {base_name}", "calibrated_learner", params)

    def current_output_payload(self) -> WorkflowPayload | None:
        if self._model_artifact is None:
            return None
        return WorkflowPayload("Classifier", self._model_artifact)

    def serialize_node_state(self) -> dict:
        return {
            **super().serialize_node_state(),
            "calibration": self._cal_group.checkedId(),
            "threshold": self._thr_group.checkedId(),
        }

    def restore_node_state(self, payload: dict) -> None:
        super().restore_node_state(payload)
        cal_btn = self._cal_group.button(int(payload.get("calibration", 0)))
        if cal_btn:
            cal_btn.setChecked(True)
        thr_btn = self._thr_group.button(int(payload.get("threshold", 0)))
        if thr_btn:
            thr_btn.setChecked(True)
