from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QComboBox,
    QFormLayout,
    QGroupBox,
    QLabel,
    QLineEdit,
    QSpinBox,
    QVBoxLayout,
)

from portakal_app.data.services.sklearn_learner_service import SklearnLearnerService
from portakal_app.ui.screens.model_base import ModelScreenBase


_METRICS = ["euclidean", "manhattan", "chebyshev", "mahalanobis"]
_METRIC_LABELS = ["Euclidean", "Manhattan", "Maximal", "Mahalanobis"]
_WEIGHTS = ["uniform", "distance"]
_WEIGHT_LABELS = ["Uniform", "Distance"]


class KNNScreen(ModelScreenBase):
    """k-Nearest Neighbours — predict by k closest training instances."""

    _OUTPUT_PORT_LABEL = "Model"

    def __init__(self, parent=None) -> None:
        self._svc = SklearnLearnerService()
        self._learner_name = "kNN"
        self._n_neighbors = 5
        self._metric_index = 0
        self._weight_index = 0
        super().__init__(parent)

    def _add_main_layout(self, layout: QVBoxLayout) -> None:
        self._name_edit = QLineEdit("kNN")
        self._name_edit.textChanged.connect(self._settings_changed)
        layout.addWidget(QLabel("Name"))
        layout.addWidget(self._name_edit)

        box = QGroupBox("Neighbours")
        form = QFormLayout(box)
        form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)

        self._n_spin = QSpinBox()
        self._n_spin.setRange(1, 100)
        self._n_spin.setValue(5)
        self._n_spin.setAlignment(Qt.AlignmentFlag.AlignRight)
        self._n_spin.valueChanged.connect(self._settings_changed)
        form.addRow("Number of neighbors:", self._n_spin)

        self._metric_combo = QComboBox()
        self._metric_combo.addItems(_METRIC_LABELS)
        self._metric_combo.currentIndexChanged.connect(self._settings_changed)
        form.addRow("Metric:", self._metric_combo)

        self._weight_combo = QComboBox()
        self._weight_combo.addItems(_WEIGHT_LABELS)
        self._weight_combo.currentIndexChanged.connect(self._settings_changed)
        form.addRow("Weights:", self._weight_combo)

        layout.addWidget(box)

        note = QLabel(
            "Default preprocessing: remove rows with unknown target, one-hot encode categorical "
            "variables, remove empty columns, impute missing values, then normalize."
        )
        note.setWordWrap(True)
        note.setProperty("muted", True)
        layout.addWidget(note)

    def _train(self):
        from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor

        ds = self._dataset
        target_cols = ds.domain.target_columns
        if not target_cols:
            raise ValueError("No target column.")
        is_clf = target_cols[0].logical_type in {"categorical", "boolean"}

        n = self._n_spin.value()
        metric = _METRICS[self._metric_combo.currentIndex()]
        weight = _WEIGHTS[self._weight_combo.currentIndex()]
        display_name = self._name_edit.text().strip() or "kNN"

        kwargs = {"n_neighbors": n, "weights": weight}
        if metric == "mahalanobis":
            X, _feature_names, _categorical_encoders, _numeric_cols, _numeric_means = self._svc.prepare_features(ds)
            covariance = np.cov(X, rowvar=False)
            covariance = np.atleast_2d(covariance)
            kwargs["metric"] = "mahalanobis"
            kwargs["metric_params"] = {"VI": np.linalg.pinv(covariance)}
        else:
            kwargs["metric"] = metric

        estimator = KNeighborsClassifier(**kwargs) if is_clf else KNeighborsRegressor(**kwargs)
        params = {"n_neighbors": n, "metric": metric, "weights": weight}
        return self._svc.fit(estimator, ds, display_name, "knn", params)

    def serialize_node_state(self) -> dict:
        return {
            **super().serialize_node_state(),
            "learner_name": self._name_edit.text(),
            "n_neighbors": self._n_spin.value(),
            "metric_index": self._metric_combo.currentIndex(),
            "weight_index": self._weight_combo.currentIndex(),
        }

    def restore_node_state(self, payload: dict) -> None:
        super().restore_node_state(payload)
        self._name_edit.setText(str(payload.get("learner_name", "kNN")))
        self._n_spin.setValue(int(payload.get("n_neighbors", 5)))
        self._metric_combo.setCurrentIndex(int(payload.get("metric_index", 0)))
        self._weight_combo.setCurrentIndex(int(payload.get("weight_index", 0)))
