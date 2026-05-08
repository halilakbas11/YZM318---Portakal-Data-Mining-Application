from __future__ import annotations

"""
Distance Matrix for unsupervised metric-space construction.

Let X = {x_1, ..., x_n}, x_i in R^p, be the numeric representation extracted
from an incoming data object. This widget computes D in R^{n x n}, where
D_ij = d(x_i, x_j). In Orange-style unsupervised workflows this matrix is not
merely a table of numbers; it is the explicit geometry consumed by clustering,
nearest-neighbor graphs, multidimensional scaling, manifold learning, and
density estimation.

Euclidean distance is the L2 metric

    d_2(x, y) = sqrt(sum_k (x_k - y_k)^2).

Its metric balls are hyperspheres. It is invariant under orthonormal rotations
and induces the standard Euclidean topology, so local neighborhoods are
isotropic when features share comparable scale.

Manhattan distance is the L1 metric

    d_1(x, y) = sum_k |x_k - y_k|.

Its balls are diamonds in R^2 and cross-polytopes in R^p. L1 and L2 are
topologically equivalent in finite-dimensional R^p, but they bend finite
neighborhoods differently: L1 privileges axis-aligned movement and changes
nearest-neighbor relations in sparse or coordinate-separable data.

Cosine distance is the angular dissimilarity

    d_cos(x, y) = 1 - <x, y> / (||x||_2 ||y||_2).

It projects nonzero vectors onto the unit sphere and compares rays rather than
positions. Positive scalar multiples collapse to the same direction. Since
1-cos(theta) is not generally a true metric, the induced geometry should be
read as projective angular topology, not as a normed vector-space topology.

Chebyshev distance is the L-infinity metric

    d_inf(x, y) = max_k |x_k - y_k|.

Its balls are axis-aligned hypercubes. It changes the local geometry by making
the largest single coordinate deviation determine distance, which is useful
when a point is considered far if any one feature exceeds tolerance.

Jaccard distance is

    d_J(A, B) = 1 - |A intersection B| / |A union B|.

For numeric tables the widget binarizes nonzero entries into sets of active
features. The resulting topology is combinatorial rather than Euclidean:
neighborhoods are governed by shared support patterns and ignore magnitude.
"""

from dataclasses import dataclass
from math import isfinite
from typing import Any

import numpy as np
import polars as pl

from PySide6.QtCore import QObject, QRunnable, QThreadPool, Signal, Slot
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from portakal_app.data.models import DatasetHandle
from portakal_app.data.services.distance_matrix_service import DistanceMatrixHandle, build_distance_matrix
from portakal_app.models import WorkflowPayload
from portakal_app.ui import i18n
from portakal_app.ui.screens.node_screen import WorkflowNodeScreenSupport


DistanceMatrixResult = DistanceMatrixHandle


@dataclass(frozen=True)
class _NumericInput:
    values: np.ndarray
    feature_names: tuple[str, ...]
    row_labels: tuple[str, ...]
    imputed_count: int
    source_name: str
    source_dataset: DatasetHandle | None


class WorkerSignals(QObject):
    result_ready = Signal(dict)
    error_ready = Signal(dict)


class DistanceWorker(QRunnable):
    """Background worker that never mutates Qt widgets directly."""

    def __init__(
        self,
        *,
        job_id: int,
        numeric: _NumericInput,
        metric: str,
        metric_label: str,
        axis: str,
        axis_label: str,
        standardize: bool,
    ) -> None:
        super().__init__()
        self.signals = WorkerSignals()
        self._job_id = job_id
        self._numeric = numeric
        self._metric = metric
        self._metric_label = metric_label
        self._axis = axis
        self._axis_label = axis_label
        self._standardize = standardize

    @Slot()
    def run(self) -> None:
        try:
            values = self._numeric.values.astype(float, copy=True)
            if self._standardize and self._metric != "jaccard":
                values = _standardize(values)
            if self._metric == "jaccard":
                values = values != 0
            if self._axis == "columns":
                values = values.T
                matrix_labels = self._numeric.feature_names
                coordinate_labels = self._numeric.row_labels
            else:
                matrix_labels = self._numeric.row_labels
                coordinate_labels = self._numeric.feature_names

            effective_metric = self._metric
            effective_label = self._metric_label
            warnings: list[str] = []
            try:
                matrix = _compute_distance_matrix(values, self._metric)
            except Exception as exc:
                warning = i18n.tf(
                    "{metric} failed ({err}); falling back to Euclidean.",
                    metric=self._metric_label,
                    err=exc,
                )
                print(f"OWDistances warning: {warning}", flush=True)
                warnings.append(warning)
                effective_metric = "euclidean"
                effective_label = i18n.t("Euclidean")
                matrix = _compute_distance_matrix(np.asarray(values, dtype=float), effective_metric)
            matrix = np.asarray(matrix, dtype=float)
            if effective_metric != "jaccard":
                np.fill_diagonal(matrix, 0.0)
            metadata = {
                "source_name": self._numeric.source_name,
                "standardized": self._standardize and self._metric != "jaccard",
                "imputed_count": self._numeric.imputed_count,
                "metric": self._metric,
                "effective_metric": effective_metric,
                "axis": self._axis,
                "shape": tuple(int(v) for v in matrix.shape),
                "jaccard_binarized": self._metric == "jaccard",
                "warnings": warnings,
            }
            result = build_distance_matrix(
                matrix=matrix,
                metric=effective_metric,
                metric_label=effective_label,
                axis=self._axis,
                axis_label=self._axis_label,
                row_labels=matrix_labels,
                feature_names=coordinate_labels,
                source_dataset=self._numeric.source_dataset if self._axis == "rows" else None,
                metadata=metadata,
            )
            self.signals.result_ready.emit({"job_id": self._job_id, "result": result})
        except Exception as exc:
            self.signals.error_ready.emit({"job_id": self._job_id, "error": str(exc)})


class OWDistances(QWidget, WorkflowNodeScreenSupport):
    """
    Production-grade distance matrix widget.

    The widget receives Polars, pandas-like, DatasetHandle, WorkflowPayload, or
    matrix-like objects, isolates numeric coordinates, imputes missing values,
    and launches pairwise-distance computation in a QRunnable. Results return
    through WorkerSignals.result_ready as dictionaries; only the main-thread
    slot populates the QTableWidget, stores the DistanceMatrixResult, emits
    output_signal, and notifies the workflow runtime.

    Geometrically, the metric selector changes the topology observed by
    downstream algorithms. Euclidean creates spherical L2 neighborhoods;
    Manhattan creates axis-aligned L1 cross-polytopes; Cosine turns points into
    angular rays on the unit sphere; Chebyshev creates L-infinity hypercubes;
    Jaccard turns rows into support sets and compares overlap. These choices
    bend the local adjacency relation and therefore alter clustering,
    projection, and neighborhood graph behavior.
    """

    output_signal = Signal(object)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._init_workflow_node_support()
        self._thread_pool = QThreadPool.globalInstance()
        self._job_id = 0
        self._active_worker: DistanceWorker | None = None
        self._numeric: _NumericInput | None = None
        self._result: DistanceMatrixResult | None = None
        self._metric = "euclidean"
        self._metric_label = "Euclidean"
        self._axis = "rows"
        self._axis_label = "Distances between rows"
        self._standardize_enabled = True
        self._auto_apply_enabled = True

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        title = QLabel(i18n.t("Distances"))
        title.setProperty("sectionTitle", True)
        layout.addWidget(title)

        self._summary = QLabel(i18n.t("Load a dataset with numeric columns."))
        self._summary.setWordWrap(True)
        self._summary.setProperty("muted", True)
        layout.addWidget(self._summary)

        settings = QGroupBox(i18n.t("Settings"))
        form = QFormLayout(settings)
        self._axis_combo = QComboBox()
        self._axis_combo.addItem(i18n.t("Distances between rows"), "rows")
        self._axis_combo.addItem(i18n.t("Distances between columns"), "columns")
        form.addRow(i18n.t("Compute:"), self._axis_combo)
        self._metric_combo = QComboBox()
        for label, code in (
            ("Euclidean", "euclidean"),
            ("Manhattan", "manhattan"),
            ("Cosine", "cosine"),
            ("Chebyshev", "chebyshev"),
            ("Mahalanobis", "mahalanobis"),
            ("Hamming", "hamming"),
            ("Pearson correlation", "pearson"),
            ("Pearson absolute correlation", "pearson_abs"),
            ("Spearman correlation", "spearman"),
            ("Spearman absolute correlation", "spearman_abs"),
            ("Jaccard", "jaccard"),
        ):
            self._metric_combo.addItem(i18n.t(label), code)
        form.addRow(i18n.t("Metric:"), self._metric_combo)
        self._standardize = QCheckBox(i18n.t("Standardize Data"))
        self._standardize.setChecked(True)
        form.addRow("", self._standardize)
        layout.addWidget(settings)

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

        self._table = QTableWidget(0, 0)
        self._table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self._table.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        layout.addWidget(self._table, 1)

        self._sync_state_from_controls()
        self._axis_combo.currentIndexChanged.connect(self._handle_settings_changed)
        self._metric_combo.currentIndexChanged.connect(self._handle_settings_changed)
        self._standardize.stateChanged.connect(self._handle_settings_changed)
        self.cb_apply_auto.stateChanged.connect(self._handle_auto_apply_changed)

    @Slot(object)
    def receive_data(self, data: object) -> None:
        self.set_input_payload(data if isinstance(data, WorkflowPayload) else WorkflowPayload("Data", data))

    @Slot(object)
    def set_input_payload(self, payload: WorkflowPayload | None) -> None:
        self._result = None
        self._table.setRowCount(0)
        self._table.setColumnCount(0)
        try:
            source = payload if payload is not None else None
            self._numeric = _extract_numeric_input(source, prefer_feature_columns=True)
            self._summary.setText(
                i18n.tf(
                    "{rows} rows x {cols} numeric features",
                    rows=self._numeric.values.shape[0],
                    cols=self._numeric.values.shape[1],
                )
            )
        except Exception as exc:
            self._numeric = None
            self._summary.setText(i18n.tf("Error: {err}", err=exc))
            self.output_signal.emit(None)
            self._notify_output_changed()
            return
        self._sync_state_from_controls()
        if self._auto_apply_enabled:
            self._apply()

    def current_output_payload(self) -> WorkflowPayload | None:
        return None if self._result is None else WorkflowPayload("Distance Matrix", self._result)

    def serialize_node_state(self) -> dict[str, object]:
        self._sync_state_from_controls()
        return {
            "metric": self._metric,
            "axis": self._axis,
            "standardize": self._standardize_enabled,
            "auto_apply": self._auto_apply_enabled,
        }

    def restore_node_state(self, payload: dict[str, object]) -> None:
        axis_index = self._axis_combo.findData(str(payload.get("axis", "rows")))
        self._axis_combo.setCurrentIndex(max(0, axis_index))
        index = self._metric_combo.findData(str(payload.get("metric", "euclidean")))
        self._metric_combo.setCurrentIndex(max(0, index))
        self._standardize.setChecked(bool(payload.get("standardize", True)))
        self.cb_apply_auto.setChecked(bool(payload.get("auto_apply", True)))
        self._sync_state_from_controls()
        if self._auto_apply_enabled:
            self._apply()

    def _sync_state_from_controls(self) -> None:
        self._axis = str(self._axis_combo.currentData())
        self._axis_label = str(self._axis_combo.currentText())
        self._metric = str(self._metric_combo.currentData())
        self._metric_label = str(self._metric_combo.currentText())
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
        if self._numeric is None:
            self._result = None
            self.output_signal.emit(None)
            self._notify_output_changed()
            return
        self._job_id += 1
        self._progress.setVisible(True)
        self._apply_button.setEnabled(False)
        worker = DistanceWorker(
            job_id=self._job_id,
            numeric=self._numeric,
            metric=self._metric,
            metric_label=self._metric_label,
            axis=self._axis,
            axis_label=self._axis_label,
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
        self._result = payload.get("result")
        if isinstance(self._result, DistanceMatrixResult):
            self._populate_table(self._result)
            self._summary.setText(
                i18n.tf(
                    "{rows} rows x {cols} distance matrix",
                    rows=self._result.matrix.shape[0],
                    cols=self._result.matrix.shape[1],
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
        self._result = None
        self._summary.setText(i18n.tf("Error: {err}", err=payload.get("error", "")))
        self.output_signal.emit(None)
        self._notify_output_changed()

    def _populate_table(self, result: DistanceMatrixResult) -> None:
        count = min(12, result.matrix.shape[0])
        self._table.setRowCount(count)
        self._table.setColumnCount(count)
        labels = list(result.row_labels[:count])
        self._table.setHorizontalHeaderLabels(labels)
        self._table.setVerticalHeaderLabels(labels)
        for row in range(count):
            for col in range(count):
                value = float(result.matrix[row, col])
                text = i18n.t("NaN") if not isfinite(value) else f"{value:.5g}"
                self._table.setItem(row, col, QTableWidgetItem(text))
        self._table.resizeColumnsToContents()
        self._table.resizeRowsToContents()


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


def _extract_numeric_input(data: object, *, prefer_feature_columns: bool) -> _NumericInput:
    source = _unwrap_data(data)
    if isinstance(source, DatasetHandle):
        names = _dataset_numeric_names(source, prefer_feature_columns)
        return _numeric_from_polars(source.dataframe, names, source.display_name, source)
    frame = getattr(source, "dataframe", source)
    if isinstance(frame, pl.DataFrame):
        return _numeric_from_polars(frame, None, str(getattr(source, "display_name", "Data")), None)
    if _looks_like_pandas(frame):
        return _numeric_from_pandas(frame, str(getattr(source, "display_name", "Data")))
    values = np.asarray(frame, dtype=float)
    return _numeric_from_array(values, "Array")


def _dataset_numeric_names(dataset: DatasetHandle, prefer_feature_columns: bool) -> list[str]:
    columns = dataset.domain.feature_columns if prefer_feature_columns else dataset.domain.columns
    names = [column.name for column in columns if column.logical_type == "numeric"]
    if not names:
        names = [column.name for column in dataset.domain.columns if column.logical_type == "numeric"]
    return names


def _numeric_from_polars(
    frame: pl.DataFrame,
    names: list[str] | None,
    source_name: str,
    dataset: DatasetHandle | None,
) -> _NumericInput:
    if names is None:
        names = [name for name in frame.columns if _polars_is_numeric(frame.get_column(name).dtype)]
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
    values = np.column_stack(arrays)
    return _NumericInput(values, tuple(kept), tuple(f"#{i}" for i in range(values.shape[0])), imputed, source_name, dataset)


def _numeric_from_pandas(frame: Any, source_name: str) -> _NumericInput:
    numeric = frame.select_dtypes(include=["number", "bool"])
    return _numeric_from_array(np.asarray(numeric.to_numpy(), dtype=float), source_name, tuple(str(c) for c in numeric.columns))


def _numeric_from_array(values: np.ndarray, source_name: str, names: tuple[str, ...] | None = None) -> _NumericInput:
    if values.ndim == 1:
        values = values.reshape(-1, 1)
    if values.ndim != 2:
        raise ValueError(i18n.t("Input matrix must be two-dimensional."))
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
    matrix = np.column_stack(arrays)
    return _NumericInput(matrix, tuple(kept), tuple(f"#{i}" for i in range(matrix.shape[0])), imputed, source_name, None)


def _clean_vector(vector: np.ndarray) -> tuple[np.ndarray | None, int]:
    values = np.asarray(vector, dtype=float)
    values[~np.isfinite(values)] = np.nan
    if np.isnan(values).all():
        return None, 0
    missing = np.isnan(values)
    count = int(missing.sum())
    if count:
        values = values.copy()
        values[missing] = float(np.nanmean(values))
    return values, count


def _standardize(values: np.ndarray) -> np.ndarray:
    mean = values.mean(axis=0, keepdims=True)
    scale = values.std(axis=0, keepdims=True)
    scale[scale < 1e-12] = 1.0
    return (values - mean) / scale


def _compute_distance_matrix(values: np.ndarray, metric: str) -> np.ndarray:
    if values.ndim != 2:
        raise ValueError(i18n.t("Distance input must be two-dimensional."))
    if values.shape[0] == 0:
        raise ValueError(i18n.t("Distance input is empty."))
    if metric in {"pearson", "pearson_abs", "spearman", "spearman_abs"}:
        if values.shape[1] < 2:
            raise ValueError(i18n.t("Correlation distance needs at least two coordinates."))
        vectors = _rank_rows(values) if metric in {"spearman", "spearman_abs"} else values
        corr = np.corrcoef(vectors)
        corr = np.asarray(corr, dtype=float)
        if corr.shape != (values.shape[0], values.shape[0]):
            raise ValueError(i18n.t("Correlation distance produced an invalid matrix."))
        corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
        corr = np.clip(corr, -1.0, 1.0)
        if metric in {"pearson_abs", "spearman_abs"}:
            corr = np.abs(corr)
        return (1.0 - corr) / 2.0
    from sklearn.metrics import pairwise_distances

    if metric == "mahalanobis":
        covariance = np.cov(values, rowvar=False)
        covariance = np.atleast_2d(covariance)
        inverse_covariance = np.linalg.pinv(covariance)
        return pairwise_distances(values, metric=metric, VI=inverse_covariance)
    return pairwise_distances(values, metric=metric)


def _rank_rows(values: np.ndarray) -> np.ndarray:
    ranked = np.empty_like(values, dtype=float)
    for row_index, row in enumerate(values):
        order = np.argsort(row, kind="mergesort")
        ranks = np.empty(row.shape[0], dtype=float)
        sorted_values = row[order]
        start = 0
        while start < sorted_values.shape[0]:
            end = start + 1
            while end < sorted_values.shape[0] and sorted_values[end] == sorted_values[start]:
                end += 1
            ranks[order[start:end]] = (start + end - 1) / 2.0 + 1.0
            start = end
        ranked[row_index] = ranks
    return ranked


def _polars_is_numeric(dtype: object) -> bool:
    checker = getattr(dtype, "is_numeric", None)
    return bool(checker()) if callable(checker) else any(token in str(dtype).lower() for token in ("int", "float", "decimal"))


def _looks_like_pandas(value: object) -> bool:
    return hasattr(value, "select_dtypes") and hasattr(value, "to_numpy") and hasattr(value, "columns")


__all__ = ["DistanceMatrixHandle", "DistanceMatrixResult", "OWDistances", "WorkerSignals"]
