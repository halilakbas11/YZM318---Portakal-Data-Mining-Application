from __future__ import annotations

"""
k-Nearest Neighbors as metric-topological local inference.

Let X = {x_i}_{i=1}^n, x_i in R^p, be a finite sample embedded in a
Minkowski metric space. For a query x, k-nearest neighbors constructs the
index set N_k(x) containing the k observations with smallest

    d_p(x, x_i) = (sum_j |x_j - x_ij|^p)^(1/p).

The special case p=1 is Manhattan geometry: metric balls are cross-polytopes,
and adjacency is biased toward axis-aligned paths. The case p=2 is Euclidean
geometry: metric balls are hyperspheres, and adjacency is rotation-invariant
after feature scaling. Larger p progressively emphasizes the largest coordinate
deviations and approaches Chebyshev-like hypercubic neighborhoods. Thus the
Minkowski exponent is not a cosmetic hyperparameter; it changes the local
topology by changing which samples are open-neighborhood neighbors.

Classification predicts by local voting,

    y_hat(x) = argmax_c sum_{i in N_k(x)} w_i 1[y_i = c],

where uniform weights set w_i=1 and distance weights set
w_i = 1 / (d_p(x, x_i) + eps). Regression replaces the vote with the weighted
local mean. Geometrically, uniform voting treats the selected neighborhood as
a discrete ball with equal mass, while inverse-distance voting creates a
singular radial kernel whose influence decays away from the query point.

Search algorithms alter the computational geometry without changing the
mathematical estimator. brute evaluates all pairwise distances and therefore
preserves the exact finite metric graph. kd_tree recursively partitions the
space by axis-aligned hyperplanes, making it effective when rectangular
bounding boxes can prune the search. ball_tree partitions by hyperspherical
enclosures, which better respects radial neighborhoods in some medium-
dimensional settings. auto delegates this spatial indexing decision to
scikit-learn. All background computation returns through WorkerSignals so the
Qt UI thread remains the only place where widgets are mutated.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np
import polars as pl

from PySide6.QtCore import QObject, QRunnable, QThreadPool, Signal, Slot
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from portakal_app.data.models import DatasetHandle
from portakal_app.data.services.sklearn_learner_service import SklearnLearnerService
from portakal_app.models import WorkflowPayload
from portakal_app.sklearn_model_artifacts import SklearnModelArtifact
from portakal_app.ui import i18n
from portakal_app.ui.screens.node_screen import WorkflowNodeScreenSupport


@dataclass(frozen=True)
class _SupervisedInput:
    values: np.ndarray
    target_values: tuple[Any, ...]
    target_name: str
    target_type: str
    feature_names: tuple[str, ...]
    source_dataset: DatasetHandle | None
    source_name: str
    imputed_count: int


class WorkerSignals(QObject):
    result_ready = Signal(dict)
    error_ready = Signal(dict)


@dataclass(frozen=True)
class KNNResult:
    """
    Immutable-style namespace for kNN prediction output.

    The workflow emits dictionaries and DatasetHandle objects for compatibility
    with the repository pipeline, but this class name is retained for package
    exports. The mathematical result is the vector of leave-one-out predictions
    generated from a Minkowski neighborhood graph.
    """

    predictions: tuple[Any, ...]
    target_name: str
    target_type: str
    feature_names: tuple[str, ...]
    algorithm: str
    weights: str
    metric: str
    p_value: int


class KNNWorker(QRunnable):
    """Background kNN worker; it emits data only and never touches Qt widgets."""

    def __init__(
        self,
        *,
        job_id: int,
        supervised: _SupervisedInput,
        n_neighbors: int,
        algorithm: str,
        weights: str,
        metric: str,
        p_value: int,
        standardize: bool,
    ) -> None:
        super().__init__()
        self.signals = WorkerSignals()
        self._job_id = job_id
        self._supervised = supervised
        self._n_neighbors = n_neighbors
        self._algorithm = algorithm
        self._weights = weights
        self._metric = metric
        self._p_value = p_value
        self._standardize = standardize

    @Slot()
    def run(self) -> None:
        try:
            from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor, NearestNeighbors

            values = self._supervised.values.astype(float, copy=True)
            if values.shape[0] < 2:
                raise ValueError(i18n.t("Need at least two rows for leave-one-out kNN."))
            if self._standardize:
                values = _standardize(values)

            preview_k = max(1, min(self._n_neighbors, values.shape[0] - 1))
            n_query_neighbors = min(max(2, preview_k + 1), values.shape[0])
            query_kwargs = _knn_metric_kwargs(self._metric, self._p_value)
            neighbors = NearestNeighbors(
                n_neighbors=n_query_neighbors,
                algorithm=self._algorithm,
                **query_kwargs,
            )
            neighbors.fit(values)
            distances, indices = neighbors.kneighbors(values, return_distance=True)
            predictions = _leave_one_out_predictions(
                distances=distances,
                indices=indices,
                target_values=self._supervised.target_values,
                target_type=self._supervised.target_type,
                requested_k=preview_k,
                weights=self._weights,
            )
            artifact: SklearnModelArtifact | None = None
            if self._supervised.source_dataset is not None:
                service = SklearnLearnerService()
                fit_kwargs = {
                    "n_neighbors": preview_k,
                    "weights": self._weights,
                    "algorithm": self._algorithm,
                    **query_kwargs,
                }
                base_estimator = (
                    KNeighborsRegressor(**fit_kwargs)
                    if self._supervised.target_type == "numeric"
                    else KNeighborsClassifier(**fit_kwargs)
                )
                if self._standardize:
                    from sklearn.pipeline import make_pipeline
                    from sklearn.preprocessing import StandardScaler

                    estimator = make_pipeline(StandardScaler(), base_estimator)
                else:
                    estimator = base_estimator
                artifact = service.fit(
                    estimator,
                    self._supervised.source_dataset,
                    "kNN",
                    "knn",
                    params={
                        "n_neighbors": preview_k,
                        "algorithm": self._algorithm,
                        "weights": self._weights,
                        "metric": self._metric,
                        "minkowski_p": self._p_value,
                        "p_value": self._p_value,
                        "standardize": self._standardize,
                        "standardized_preview": self._standardize,
                    },
                )
            self.signals.result_ready.emit(
                {
                    "job_id": self._job_id,
                    "predictions": predictions,
                    "model_artifact": artifact,
                    "target_name": self._supervised.target_name,
                    "target_type": self._supervised.target_type,
                    "feature_names": self._supervised.feature_names,
                    "algorithm": self._algorithm,
                    "weights": self._weights,
                    "metric": self._metric,
                    "p_value": self._p_value,
                    "standardized": self._standardize,
                    "imputed_count": self._supervised.imputed_count,
                    "rows": values.shape[0],
                    "n_neighbors": preview_k,
                }
            )
        except Exception as exc:
            self.signals.error_ready.emit({"job_id": self._job_id, "error": str(exc)})


class OWKNN(QWidget, WorkflowNodeScreenSupport):
    """
    Production-grade kNN workflow node.

    The node extracts numeric predictors from DatasetHandle, Polars, pandas-
    like, custom dataframe-holding objects, or array-like matrices. Missing
    numeric coordinates are imputed by column means before conversion to
    numpy. The response variable is taken from the project's target role when
    available; otherwise the last column is used as a pragmatic supervised
    fallback. The worker constructs an exact leave-one-out neighborhood graph
    through scikit-learn's NearestNeighbors and returns predictions by
    WorkerSignals.result_ready. This main-thread widget then populates the
    preview table and emits a SklearnModelArtifact on the Model output for
    downstream prediction and evaluation nodes.

    Geometrically, kNN turns the incoming table into a finite metric space.
    The Minkowski exponent p reshapes metric balls and therefore changes the
    adjacency topology. kd_tree indexes that space by nested axis-aligned
    cells, ball_tree by nested hyperspheres, brute by the complete distance
    graph, and auto selects an implementation according to data geometry.
    Uniform voting treats N_k(x) as an unweighted local chart; distance voting
    imposes a radial kernel that contracts influence around the query.
    """

    output_signal = Signal(object)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._init_workflow_node_support()
        self._thread_pool = QThreadPool.globalInstance()
        self._job_id = 0
        self._active_worker: KNNWorker | None = None
        self._supervised: _SupervisedInput | None = None
        self._model_artifact: SklearnModelArtifact | None = None
        self._last_payload: dict[str, Any] | None = None
        self._n_neighbors = 5
        self._algorithm = "auto"
        self._weights = "uniform"
        self._metric = "euclidean"
        self._p_value = 2
        self._standardize_enabled = True
        self._auto_apply_enabled = True

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        title = QLabel(i18n.t("kNN"))
        title.setProperty("sectionTitle", True)
        layout.addWidget(title)

        self._summary = QLabel(i18n.t("Connect a dataset with numeric features and a target."))
        self._summary.setWordWrap(True)
        self._summary.setProperty("muted", True)
        layout.addWidget(self._summary)

        settings = QGroupBox(i18n.t("Settings"))
        form = QFormLayout(settings)
        self._neighbors_spin = QSpinBox()
        self._neighbors_spin.setRange(1, 100)
        self._neighbors_spin.setValue(5)
        form.addRow(i18n.t("Neighbors:"), self._neighbors_spin)
        self._algorithm_combo = QComboBox()
        for label, algorithm in (
            ("Auto", "auto"),
            ("Ball Tree", "ball_tree"),
            ("KD Tree", "kd_tree"),
            ("Brute Force", "brute"),
        ):
            self._algorithm_combo.addItem(i18n.t(label), algorithm)
        form.addRow(i18n.t("Algorithm:"), self._algorithm_combo)
        self._weights_combo = QComboBox()
        self._weights_combo.addItem(i18n.t("Uniform"), "uniform")
        self._weights_combo.addItem(i18n.t("Distance"), "distance")
        form.addRow(i18n.t("Weights:"), self._weights_combo)
        self._metric_combo = QComboBox()
        for label, metric in (
            ("Euclidean", "euclidean"),
            ("Manhattan", "manhattan"),
            ("Chebyshev", "chebyshev"),
            ("Minkowski", "minkowski"),
        ):
            self._metric_combo.addItem(i18n.t(label), metric)
        form.addRow(i18n.t("Metric:"), self._metric_combo)
        self._p_spin = QSpinBox()
        self._p_spin.setRange(1, 10)
        self._p_spin.setValue(2)
        form.addRow(i18n.t("Minkowski p:"), self._p_spin)
        self._standardize = QCheckBox(i18n.t("Standardize Data"))
        self._standardize.setChecked(True)
        form.addRow("", self._standardize)
        layout.addWidget(settings)

        self._preview_table = QTableWidget(0, 2)
        self._preview_table.setHorizontalHeaderLabels([i18n.t("Row"), i18n.t("Prediction")])
        layout.addWidget(self._preview_table, 1)

        footer = QHBoxLayout()
        self.cb_apply_auto = QCheckBox(i18n.t("Apply Automatically"))
        self.cb_apply_auto.setChecked(True)
        footer.addWidget(self.cb_apply_auto)
        footer.addStretch(1)
        self._progress = QProgressBar()
        self._progress.setRange(0, 0)
        self._progress.setVisible(False)
        footer.addWidget(self._progress)
        self._apply_button = QPushButton(i18n.t("Apply"))
        self._apply_button.setProperty("primary", True)
        self._apply_button.clicked.connect(self._apply)
        footer.addWidget(self._apply_button)
        layout.addLayout(footer)

        self._sync_state_from_controls()
        self._neighbors_spin.valueChanged.connect(self._handle_settings_changed)
        self._algorithm_combo.currentIndexChanged.connect(self._handle_settings_changed)
        self._weights_combo.currentIndexChanged.connect(self._handle_settings_changed)
        self._metric_combo.currentIndexChanged.connect(self._handle_settings_changed)
        self._p_spin.valueChanged.connect(self._handle_settings_changed)
        self._standardize.stateChanged.connect(self._handle_settings_changed)
        self.cb_apply_auto.stateChanged.connect(self._handle_auto_apply_changed)

    @Slot(object)
    def receive_data(self, data: object) -> None:
        self.set_input_payload(data if isinstance(data, WorkflowPayload) else WorkflowPayload("Data", data))

    @Slot(object)
    def set_input_payload(self, payload: WorkflowPayload | None) -> None:
        self._model_artifact = None
        self._last_payload = None
        self._preview_table.setRowCount(0)
        try:
            self._supervised = _extract_supervised_input(payload)
            self._neighbors_spin.setMaximum(max(1, self._supervised.values.shape[0] - 1))
            self._summary.setText(
                i18n.tf(
                    "{rows} rows x {cols} numeric features | target: {target}",
                    rows=self._supervised.values.shape[0],
                    cols=self._supervised.values.shape[1],
                    target=self._supervised.target_name,
                )
            )
        except Exception as exc:
            self._supervised = None
            self._summary.setText(i18n.tf("Error: {err}", err=exc))
            self.output_signal.emit(None)
            self._notify_output_changed()
            return
        self._sync_state_from_controls()
        if self._auto_apply_enabled:
            self._apply()

    def current_output_dataset(self) -> DatasetHandle | None:
        return None

    def current_output_payload(self) -> WorkflowPayload | None:
        return None if self._model_artifact is None else WorkflowPayload("Model", self._model_artifact)

    def serialize_node_state(self) -> dict[str, object]:
        self._sync_state_from_controls()
        return {
            "neighbors": self._n_neighbors,
            "algorithm": self._algorithm,
            "weights": self._weights,
            "metric": self._metric,
            "p_value": self._p_value,
            "standardize": self._standardize_enabled,
            "auto_apply": self._auto_apply_enabled,
        }

    def restore_node_state(self, payload: dict[str, object]) -> None:
        self._neighbors_spin.setValue(int(payload.get("neighbors", 5)))
        algorithm_index = self._algorithm_combo.findData(str(payload.get("algorithm", "auto")))
        self._algorithm_combo.setCurrentIndex(max(0, algorithm_index))
        weights_index = self._weights_combo.findData(str(payload.get("weights", "uniform")))
        self._weights_combo.setCurrentIndex(max(0, weights_index))
        metric_index = self._metric_combo.findData(str(payload.get("metric", "euclidean")))
        self._metric_combo.setCurrentIndex(max(0, metric_index))
        self._p_spin.setValue(int(payload.get("p_value", 2)))
        self._standardize.setChecked(bool(payload.get("standardize", True)))
        self.cb_apply_auto.setChecked(bool(payload.get("auto_apply", True)))
        self._sync_state_from_controls()
        if self._auto_apply_enabled:
            self._apply()

    def _sync_state_from_controls(self) -> None:
        self._n_neighbors = int(self._neighbors_spin.value())
        self._algorithm = str(self._algorithm_combo.currentData())
        self._weights = str(self._weights_combo.currentData())
        self._metric = str(self._metric_combo.currentData())
        self._p_value = int(self._p_spin.value())
        self._standardize_enabled = bool(self._standardize.isChecked())
        self._auto_apply_enabled = bool(self.cb_apply_auto.isChecked())

    def _handle_settings_changed(self, *_args: object) -> None:
        self._sync_state_from_controls()
        if self._auto_apply_enabled:
            self._apply()

    def _handle_auto_apply_changed(self, *_args: object) -> None:
        self._sync_state_from_controls()
        if self._auto_apply_enabled:
            self._apply()

    def _auto_apply(self) -> None:
        self._handle_settings_changed()

    def _apply(self) -> None:
        self._sync_state_from_controls()
        if self._supervised is None:
            self._model_artifact = None
            self.output_signal.emit(None)
            self._notify_output_changed()
            return
        self._job_id += 1
        self._progress.setVisible(True)
        self._apply_button.setEnabled(False)
        worker = KNNWorker(
            job_id=self._job_id,
            supervised=self._supervised,
            n_neighbors=self._n_neighbors,
            algorithm=self._algorithm,
            weights=self._weights,
            metric=self._metric,
            p_value=self._p_value,
            standardize=self._standardize_enabled,
        )
        worker.signals.result_ready.connect(self._on_result_ready)
        worker.signals.error_ready.connect(self._on_error_ready)
        self._active_worker = worker
        self._thread_pool.start(worker)

    @Slot(dict)
    def _on_result_ready(self, payload: dict) -> None:
        if int(payload.get("job_id", -1)) != self._job_id:
            return
        self._progress.setVisible(False)
        self._apply_button.setEnabled(True)
        self._last_payload = payload
        self._populate_preview(payload)
        artifact = payload.get("model_artifact")
        self._model_artifact = artifact if isinstance(artifact, SklearnModelArtifact) else None
        self._summary.setText(
            i18n.tf(
                "Model: k={k} | metric: {metric} | weights: {weights}",
                k=payload["n_neighbors"],
                metric=payload["metric"],
                weights=payload["weights"],
            )
            if self._model_artifact is not None
            else i18n.tf(
                "Preview: {rows} leave-one-out predictions | model requires DatasetHandle input",
                rows=payload["rows"],
            )
        )
        self.output_signal.emit(self.current_output_payload())
        self._notify_output_changed()

    @Slot(dict)
    def _on_error_ready(self, payload: dict) -> None:
        if int(payload.get("job_id", -1)) != self._job_id:
            return
        self._progress.setVisible(False)
        self._apply_button.setEnabled(True)
        self._model_artifact = None
        self._last_payload = None
        self._summary.setText(i18n.tf("Error: {err}", err=payload.get("error", "")))
        self.output_signal.emit(None)
        self._notify_output_changed()

    def _populate_preview(self, payload: dict) -> None:
        predictions = list(payload["predictions"])
        visible = min(12, len(predictions))
        self._preview_table.setRowCount(visible)
        for row in range(visible):
            self._preview_table.setItem(row, 0, QTableWidgetItem(str(row + 1)))
            self._preview_table.setItem(row, 1, QTableWidgetItem(str(predictions[row])))
        self._preview_table.resizeColumnsToContents()


def _unwrap_data(data: object) -> object:
    if data is None:
        raise ValueError(i18n.t("Input is empty."))
    dataset = getattr(data, "dataset", None)
    if dataset is not None:
        return dataset
    value = getattr(data, "value", None)
    if value is not None and value is not data:
        return value
    return data


def _extract_supervised_input(data: object) -> _SupervisedInput:
    source = _unwrap_data(data)
    if isinstance(source, DatasetHandle):
        return _supervised_from_dataset(source)
    frame = getattr(source, "dataframe", source)
    if isinstance(frame, pl.DataFrame):
        return _supervised_from_polars(frame, None, None, "unknown", str(getattr(source, "display_name", "Data")), None)
    if _looks_like_pandas(frame):
        return _supervised_from_pandas(frame, str(getattr(source, "display_name", "Data")))
    return _supervised_from_array(np.asarray(frame, dtype=object), "Array")


def _supervised_from_dataset(dataset: DatasetHandle) -> _SupervisedInput:
    target_column = dataset.domain.target_columns[0] if dataset.domain.target_columns else None
    target_name = target_column.name if target_column is not None else dataset.dataframe.columns[-1]
    target_type = target_column.logical_type if target_column is not None else _infer_target_type(dataset.dataframe.get_column(target_name).to_list())
    feature_names = [
        column.name
        for column in dataset.domain.feature_columns
        if column.logical_type == "numeric" and column.name != target_name
    ]
    if not feature_names:
        feature_names = [
            column.name
            for column in dataset.domain.columns
            if column.logical_type == "numeric" and column.name != target_name
        ]
    return _supervised_from_polars(dataset.dataframe, feature_names, target_name, target_type, dataset.display_name, dataset)


def _supervised_from_polars(
    frame: pl.DataFrame,
    feature_names: list[str] | None,
    target_name: str | None,
    target_type: str,
    source_name: str,
    dataset: DatasetHandle | None,
) -> _SupervisedInput:
    if frame.width < 2:
        raise ValueError(i18n.t("Need at least one feature column and one target column."))
    target = target_name or frame.columns[-1]
    if target not in frame.columns:
        raise ValueError(i18n.t("Target column is missing."))
    names = feature_names or [name for name in frame.columns if name != target and _polars_is_numeric(frame.get_column(name).dtype)]
    values, kept, imputed = _numeric_matrix_from_polars(frame, names)
    target_values = tuple(frame.get_column(target).to_list())
    inferred_type = target_type if target_type != "unknown" else _infer_target_type(target_values)
    return _SupervisedInput(values, target_values, target, inferred_type, kept, dataset, source_name, imputed)


def _supervised_from_pandas(frame: Any, source_name: str) -> _SupervisedInput:
    columns = [str(column) for column in frame.columns]
    if len(columns) < 2:
        raise ValueError(i18n.t("Need at least one feature column and one target column."))
    target = columns[-1]
    numeric = frame.drop(columns=[frame.columns[-1]]).select_dtypes(include=["number", "bool"])
    values, kept, imputed = _numeric_matrix_from_array(np.asarray(numeric.to_numpy(), dtype=float), tuple(str(c) for c in numeric.columns))
    target_values = tuple(frame.iloc[:, -1].tolist())
    return _SupervisedInput(values, target_values, target, _infer_target_type(target_values), kept, None, source_name, imputed)


def _supervised_from_array(values: np.ndarray, source_name: str) -> _SupervisedInput:
    if values.ndim == 1:
        raise ValueError(i18n.t("Input matrix must contain feature columns and a target column."))
    if values.ndim != 2 or values.shape[1] < 2:
        raise ValueError(i18n.t("Need at least one feature column and one target column."))
    matrix, kept, imputed = _numeric_matrix_from_array(values[:, :-1].astype(float), None)
    target_values = tuple(values[:, -1].tolist())
    return _SupervisedInput(matrix, target_values, "target", _infer_target_type(target_values), kept, None, source_name, imputed)


def _numeric_matrix_from_polars(frame: pl.DataFrame, names: list[str]) -> tuple[np.ndarray, tuple[str, ...], int]:
    arrays: list[np.ndarray] = []
    kept: list[str] = []
    imputed = 0
    for name in names:
        if name not in frame.columns:
            continue
        vector = np.asarray(
            [np.nan if value is None else float(value) for value in frame.get_column(name).cast(pl.Float64, strict=False).to_list()],
            dtype=float,
        )
        cleaned, count = _clean_vector(vector)
        if cleaned is None:
            continue
        arrays.append(cleaned)
        kept.append(name)
        imputed += count
    if not arrays:
        raise ValueError(i18n.t("Need at least one numeric feature."))
    return np.column_stack(arrays), tuple(kept), imputed


def _numeric_matrix_from_array(values: np.ndarray, names: tuple[str, ...] | None) -> tuple[np.ndarray, tuple[str, ...], int]:
    if values.ndim == 1:
        values = values.reshape(-1, 1)
    arrays: list[np.ndarray] = []
    kept: list[str] = []
    imputed = 0
    for index in range(values.shape[1]):
        name = names[index] if names and index < len(names) else f"x{index + 1}"
        cleaned, count = _clean_vector(values[:, index])
        if cleaned is None:
            continue
        arrays.append(cleaned)
        kept.append(name)
        imputed += count
    if not arrays:
        raise ValueError(i18n.t("Need at least one numeric feature."))
    return np.column_stack(arrays), tuple(kept), imputed


def _clean_vector(vector: np.ndarray) -> tuple[np.ndarray | None, int]:
    values = np.asarray(vector, dtype=float)
    values[~np.isfinite(values)] = np.nan
    if np.isnan(values).all():
        return None, 0
    missing = np.isnan(values)
    if missing.any():
        values = values.copy()
        values[missing] = float(np.nanmean(values))
    return values, int(missing.sum())


def _leave_one_out_predictions(
    *,
    distances: np.ndarray,
    indices: np.ndarray,
    target_values: tuple[Any, ...],
    target_type: str,
    requested_k: int,
    weights: str,
) -> tuple[Any, ...]:
    predictions: list[Any] = []
    is_regression = target_type == "numeric"
    for row in range(indices.shape[0]):
        neighbor_values: list[Any] = []
        neighbor_distances: list[float] = []
        for distance, index in zip(distances[row], indices[row]):
            if int(index) == row or _is_missing(target_values[int(index)]):
                continue
            neighbor_values.append(target_values[int(index)])
            neighbor_distances.append(float(distance))
            if len(neighbor_values) == requested_k:
                break
        if not neighbor_values:
            predictions.append(None)
        elif is_regression:
            predictions.append(_regression_vote(neighbor_values, np.asarray(neighbor_distances, dtype=float), weights))
        else:
            predictions.append(_classification_vote(neighbor_values, np.asarray(neighbor_distances, dtype=float), weights))
    return tuple(predictions)


def _classification_vote(values: list[Any], distances: np.ndarray, weights: str) -> Any:
    totals: dict[Any, float] = {}
    for value, weight in zip(values, _distance_weights(distances, weights)):
        key = _python_scalar(value)
        totals[key] = totals.get(key, 0.0) + float(weight)
    return sorted(totals.items(), key=lambda item: (-item[1], str(item[0])))[0][0]


def _regression_vote(values: list[Any], distances: np.ndarray, weights: str) -> float | None:
    numeric = np.asarray([np.nan if _is_missing(value) else float(value) for value in values], dtype=float)
    valid = np.isfinite(numeric)
    if not valid.any():
        return None
    local_weights = _distance_weights(distances[valid], weights)
    return float(np.sum(numeric[valid] * local_weights) / np.sum(local_weights))


def _distance_weights(distances: np.ndarray, weights: str) -> np.ndarray:
    if weights == "distance":
        return 1.0 / (distances + 1e-12)
    return np.ones_like(distances, dtype=float)


def _knn_metric_kwargs(metric: str, p_value: int) -> dict[str, object]:
    if metric == "minkowski":
        return {"metric": "minkowski", "p": max(1, int(p_value))}
    if metric in {"euclidean", "manhattan", "chebyshev"}:
        return {"metric": metric}
    raise ValueError(i18n.tf("Unsupported metric: {metric}", metric=metric))


def _standardize(values: np.ndarray) -> np.ndarray:
    mean = values.mean(axis=0, keepdims=True)
    scale = values.std(axis=0, keepdims=True)
    scale[scale < 1e-12] = 1.0
    return (values - mean) / scale


def _infer_target_type(values: Any) -> str:
    nonmissing = [value for value in values if not _is_missing(value)]
    if not nonmissing:
        return "categorical"
    try:
        np.asarray([float(value) for value in nonmissing], dtype=float)
    except (TypeError, ValueError):
        return "categorical"
    return "numeric"


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    try:
        return bool(np.isnan(value))
    except (TypeError, ValueError):
        return False


def _python_scalar(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    return value


def _unique_column_name(dataset: DatasetHandle, base: str) -> str:
    existing = set(dataset.dataframe.columns)
    if base not in existing:
        return base
    index = 1
    while f"{base} ({index})" in existing:
        index += 1
    return f"{base} ({index})"


def _polars_is_numeric(dtype: object) -> bool:
    checker = getattr(dtype, "is_numeric", None)
    return bool(checker()) if callable(checker) else any(token in str(dtype).lower() for token in ("int", "float", "decimal"))


def _looks_like_pandas(value: object) -> bool:
    return hasattr(value, "select_dtypes") and hasattr(value, "to_numpy") and hasattr(value, "columns")


__all__ = ["KNNResult", "OWKNN", "WorkerSignals"]
