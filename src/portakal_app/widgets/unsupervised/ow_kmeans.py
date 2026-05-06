from __future__ import annotations

"""
k-Means clustering as Voronoi quantization of Euclidean feature space.

Given X = {x_i}_{i=1}^n in R^p, k-Means searches for centroids
mu_1, ..., mu_k minimizing

    J(mu, c) = sum_i ||x_i - mu_{c_i}||_2^2.

The assignment step constructs a Voronoi tessellation: every point is assigned
to the nearest centroid and cluster boundaries lie on hyperplane fragments
where two centroids are equidistant. The update step replaces each centroid by
the arithmetic mean of its assigned points, the unique minimizer of squared
Euclidean distortion inside that Voronoi cell. Lloyd iteration alternates
between these two operations and monotonically decreases J.

k-means++ initialization samples the first centroid uniformly and subsequent
centroids with probability proportional to squared distance from the nearest
existing centroid. Geometrically, this spreads prototypes across the data
cloud before Lloyd iteration, reducing collapse into the same basin of the
objective landscape. random initialization samples centroids without this
spatial spreading and is therefore more sensitive to local minima. n_init
repeats the process from multiple initial conditions and keeps the lowest
inertia solution.

The silhouette coefficient compares a point's mean intra-cluster distance a
with the nearest alternative-cluster distance b through s=(b-a)/max(a,b).
It evaluates how well the Voronoi-induced topology separates neighborhoods.
"""

from dataclasses import dataclass, replace
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

from portakal_app.data.models import DataDomain, DatasetHandle, build_data_domain
from portakal_app.models import WorkflowPayload
from portakal_app.ui import i18n
from portakal_app.ui.screens.node_screen import WorkflowNodeScreenSupport


@dataclass(frozen=True)
class _NumericInput:
    values: np.ndarray
    feature_names: tuple[str, ...]
    source_dataset: DatasetHandle | None
    source_name: str
    imputed_count: int


class WorkerSignals(QObject):
    result_ready = Signal(dict)
    error_ready = Signal(dict)


class KMeansWorker(QRunnable):
    """Background k-Means worker; all UI mutation happens in widget slots."""

    def __init__(
        self,
        *,
        job_id: int,
        numeric: _NumericInput,
        n_clusters: int,
        optimize_clusters: bool,
        min_clusters: int,
        max_clusters: int,
        init_method: str,
        max_iter: int,
        n_init: int,
        standardize: bool,
    ) -> None:
        super().__init__()
        self.signals = WorkerSignals()
        self._job_id = job_id
        self._numeric = numeric
        self._n_clusters = n_clusters
        self._optimize_clusters = optimize_clusters
        self._min_clusters = min_clusters
        self._max_clusters = max_clusters
        self._init_method = init_method
        self._max_iter = max_iter
        self._n_init = n_init
        self._standardize = standardize

    @Slot()
    def run(self) -> None:
        try:
            from sklearn.cluster import KMeans
            from sklearn.metrics import silhouette_score

            values = self._numeric.values.astype(float, copy=True)
            if self._standardize:
                values = _standardize(values)
            if values.shape[0] < 2:
                raise ValueError(i18n.t("Need at least two rows for k-Means."))
            if self._optimize_clusters:
                start = max(2, self._min_clusters)
                stop = min(self._max_clusters, values.shape[0] - 1)
                candidates = list(range(start, stop + 1)) or [max(2, min(self._min_clusters, values.shape[0]))]
            else:
                candidates = [max(2, min(self._n_clusters, values.shape[0]))]
            best: tuple[float, float, Any, np.ndarray, int] | None = None
            scores: list[dict[str, float | int]] = []
            for n_clusters in candidates:
                model = KMeans(
                    n_clusters=n_clusters,
                    init=self._init_method,
                    max_iter=self._max_iter,
                    n_init=self._n_init,
                    random_state=42,
                )
                labels = model.fit_predict(values)
                silhouette = _safe_silhouette(values, labels, silhouette_score)
                inertia = float(model.inertia_)
                scores.append({"k": int(n_clusters), "silhouette": silhouette, "inertia": inertia})
                comparable = silhouette if np.isfinite(silhouette) else float("-inf")
                if best is None or comparable > best[0] or (comparable == best[0] and inertia < best[1]):
                    best = (comparable, inertia, model, labels, int(n_clusters))
            if best is None:
                raise ValueError(i18n.t("No valid cluster count in the selected range."))
            _, inertia, model, labels, n_clusters = best
            silhouette = next(
                float(item["silhouette"])
                for item in scores
                if int(item["k"]) == n_clusters and float(item["inertia"]) == inertia
            )
            self.signals.result_ready.emit(
                {
                    "job_id": self._job_id,
                    "labels": labels,
                    "centroids": np.asarray(model.cluster_centers_, dtype=float),
                    "inertia": float(model.inertia_),
                    "iterations": int(model.n_iter_),
                    "silhouette": silhouette,
                    "feature_names": self._numeric.feature_names,
                    "standardized": self._standardize,
                    "init_method": self._init_method,
                    "n_init": self._n_init,
                    "n_clusters": n_clusters,
                    "optimized": self._optimize_clusters,
                    "cluster_scores": scores,
                }
            )
        except Exception as exc:
            self.signals.error_ready.emit({"job_id": self._job_id, "error": str(exc)})


class OWKMeans(QWidget, WorkflowNodeScreenSupport):
    """
    Production-grade k-Means workflow node.

    Numeric data are extracted, imputed, converted to numpy, and fitted in a
    QRunnable. WorkerSignals.result_ready returns cluster labels, centroids,
    inertia, iteration count, and silhouette score. The main widget appends
    labels to the outgoing DatasetHandle and populates centroid diagnostics.

    The algorithm imposes a Euclidean Voronoi topology on the data: clusters
    are convex cells around centroids, not arbitrary connected components.
    k-means++ changes initialization geometry by probabilistically spreading
    centroids according to squared distance; random initialization does not.
    Multiple n_init runs sample multiple basins of the nonconvex objective and
    choose the lowest-distortion tessellation.
    """

    output_signal = Signal(object)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._init_workflow_node_support()
        self._thread_pool = QThreadPool.globalInstance()
        self._job_id = 0
        self._active_worker: KMeansWorker | None = None
        self._numeric: _NumericInput | None = None
        self._output_dataset: DatasetHandle | None = None
        self._n_clusters = 3
        self._cluster_mode = "fixed"
        self._min_clusters = 2
        self._max_clusters = 8
        self._init_method = "k-means++"
        self._max_iter = 300
        self._n_init = 10
        self._standardize_enabled = True
        self._auto_apply_enabled = True

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        title = QLabel(i18n.t("k-Means"))
        title.setProperty("sectionTitle", True)
        layout.addWidget(title)

        self._summary = QLabel(i18n.t("Load a dataset with numeric columns."))
        self._summary.setWordWrap(True)
        self._summary.setProperty("muted", True)
        layout.addWidget(self._summary)

        settings = QGroupBox(i18n.t("Settings"))
        form = QFormLayout(settings)
        self._cluster_mode_combo = QComboBox()
        self._cluster_mode_combo.addItem(i18n.t("Fixed number of clusters"), "fixed")
        self._cluster_mode_combo.addItem(i18n.t("Optimize clusters from range"), "range")
        form.addRow(i18n.t("Optimization:"), self._cluster_mode_combo)
        self._clusters_spin = QSpinBox()
        self._clusters_spin.setRange(2, 100)
        self._clusters_spin.setValue(3)
        form.addRow(i18n.t("Clusters:"), self._clusters_spin)
        self._min_clusters_spin = QSpinBox()
        self._min_clusters_spin.setRange(2, 100)
        self._min_clusters_spin.setValue(2)
        form.addRow(i18n.t("From k:"), self._min_clusters_spin)
        self._max_clusters_spin = QSpinBox()
        self._max_clusters_spin.setRange(2, 100)
        self._max_clusters_spin.setValue(8)
        form.addRow(i18n.t("To k:"), self._max_clusters_spin)
        self._init_combo = QComboBox()
        self._init_combo.addItem(i18n.t("k-means++"), "k-means++")
        self._init_combo.addItem(i18n.t("random"), "random")
        form.addRow(i18n.t("Initialization:"), self._init_combo)
        self._max_iter_spin = QSpinBox()
        self._max_iter_spin.setRange(1, 5000)
        self._max_iter_spin.setValue(300)
        form.addRow(i18n.t("Max iterations:"), self._max_iter_spin)
        self._n_init_spin = QSpinBox()
        self._n_init_spin.setRange(1, 100)
        self._n_init_spin.setValue(10)
        form.addRow(i18n.t("Number of runs:"), self._n_init_spin)
        self._standardize = QCheckBox(i18n.t("Standardize Data"))
        self._standardize.setChecked(True)
        form.addRow("", self._standardize)
        layout.addWidget(settings)

        self._centroid_table = QTableWidget(0, 0)
        layout.addWidget(self._centroid_table, 1)

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
        self._cluster_mode_combo.currentIndexChanged.connect(self._handle_settings_changed)
        self._clusters_spin.valueChanged.connect(self._handle_settings_changed)
        self._min_clusters_spin.valueChanged.connect(self._handle_settings_changed)
        self._max_clusters_spin.valueChanged.connect(self._handle_settings_changed)
        self._init_combo.currentIndexChanged.connect(self._handle_settings_changed)
        self._max_iter_spin.valueChanged.connect(self._handle_settings_changed)
        self._n_init_spin.valueChanged.connect(self._handle_settings_changed)
        self._standardize.stateChanged.connect(self._handle_settings_changed)
        self.cb_apply_auto.stateChanged.connect(self._handle_auto_apply_changed)

    @Slot(object)
    def receive_data(self, data: object) -> None:
        self.set_input_payload(data if isinstance(data, WorkflowPayload) else WorkflowPayload("Data", data))

    @Slot(object)
    def set_input_payload(self, payload: WorkflowPayload | None) -> None:
        self._output_dataset = None
        self._centroid_table.setRowCount(0)
        self._centroid_table.setColumnCount(0)
        try:
            self._numeric = _extract_numeric_input(payload, prefer_feature_columns=True)
            max_k = max(2, self._numeric.values.shape[0])
            self._clusters_spin.setMaximum(max_k)
            self._min_clusters_spin.setMaximum(max_k)
            self._max_clusters_spin.setMaximum(max_k)
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

    def current_output_dataset(self) -> DatasetHandle | None:
        return self._output_dataset

    def serialize_node_state(self) -> dict[str, object]:
        self._sync_state_from_controls()
        return {
            "n_clusters": self._n_clusters,
            "cluster_mode": self._cluster_mode,
            "min_clusters": self._min_clusters,
            "max_clusters": self._max_clusters,
            "init": self._init_method,
            "max_iter": self._max_iter,
            "n_init": self._n_init,
            "standardize": self._standardize_enabled,
            "auto_apply": self._auto_apply_enabled,
        }

    def restore_node_state(self, payload: dict[str, object]) -> None:
        mode_index = self._cluster_mode_combo.findData(str(payload.get("cluster_mode", "fixed")))
        self._cluster_mode_combo.setCurrentIndex(max(0, mode_index))
        self._clusters_spin.setValue(int(payload.get("n_clusters", 3)))
        self._min_clusters_spin.setValue(int(payload.get("min_clusters", 2)))
        self._max_clusters_spin.setValue(int(payload.get("max_clusters", 8)))
        index = self._init_combo.findData(str(payload.get("init", "k-means++")))
        self._init_combo.setCurrentIndex(max(0, index))
        self._max_iter_spin.setValue(int(payload.get("max_iter", 300)))
        self._n_init_spin.setValue(int(payload.get("n_init", 10)))
        self._standardize.setChecked(bool(payload.get("standardize", True)))
        self.cb_apply_auto.setChecked(bool(payload.get("auto_apply", True)))
        self._sync_state_from_controls()
        if self._auto_apply_enabled:
            self._apply()

    def _sync_state_from_controls(self) -> None:
        self._n_clusters = int(self._clusters_spin.value())
        self._cluster_mode = str(self._cluster_mode_combo.currentData())
        self._min_clusters = int(self._min_clusters_spin.value())
        self._max_clusters = int(self._max_clusters_spin.value())
        if self._max_clusters < self._min_clusters:
            self._max_clusters = self._min_clusters
            self._max_clusters_spin.blockSignals(True)
            self._max_clusters_spin.setValue(self._max_clusters)
            self._max_clusters_spin.blockSignals(False)
        self._init_method = str(self._init_combo.currentData())
        self._max_iter = int(self._max_iter_spin.value())
        self._n_init = int(self._n_init_spin.value())
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
            self._output_dataset = None
            self.output_signal.emit(None)
            self._notify_output_changed()
            return
        self._job_id += 1
        self._progress.setVisible(True)
        self._apply_button.setEnabled(False)
        worker = KMeansWorker(
            job_id=self._job_id,
            numeric=self._numeric,
            n_clusters=self._n_clusters,
            optimize_clusters=self._cluster_mode == "range",
            min_clusters=self._min_clusters,
            max_clusters=self._max_clusters,
            init_method=self._init_method,
            max_iter=self._max_iter,
            n_init=self._n_init,
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
        self._populate_centroids(payload)
        self._output_dataset = self._build_output_dataset(payload)
        silhouette = payload["silhouette"]
        silhouette_text = "NaN" if not np.isfinite(silhouette) else f"{float(silhouette):.6g}"
        self._summary.setText(
            i18n.tf(
                "Clusters: {count} | Inertia: {inertia} | Silhouette: {silhouette}",
                count=payload["n_clusters"],
                inertia=f"{float(payload['inertia']):.6g}",
                silhouette=silhouette_text,
            )
        )
        self.output_signal.emit(self._output_dataset)
        self._notify_output_changed()

    @Slot(dict)
    def _on_error_ready(self, payload: dict) -> None:
        if int(payload.get("job_id", -1)) != self._job_id:
            return
        self._progress.setVisible(False)
        self._apply_button.setEnabled(True)
        self._output_dataset = None
        self._summary.setText(i18n.tf("Error: {err}", err=payload.get("error", "")))
        self.output_signal.emit(None)
        self._notify_output_changed()

    def _populate_centroids(self, payload: dict) -> None:
        centroids = payload["centroids"]
        names = list(payload["feature_names"])
        self._centroid_table.setRowCount(centroids.shape[0])
        self._centroid_table.setColumnCount(centroids.shape[1])
        self._centroid_table.setHorizontalHeaderLabels(names)
        self._centroid_table.setVerticalHeaderLabels([f"C{i + 1}" for i in range(centroids.shape[0])])
        for row in range(centroids.shape[0]):
            for col in range(centroids.shape[1]):
                self._centroid_table.setItem(row, col, QTableWidgetItem(f"{float(centroids[row, col]):.6g}"))
        self._centroid_table.resizeColumnsToContents()

    def _build_output_dataset(self, payload: dict) -> DatasetHandle | None:
        numeric = self._numeric
        if numeric is None or numeric.source_dataset is None:
            return None
        cluster_name = _unique_column_name(numeric.source_dataset, "Cluster")
        labels = [f"C{int(label) + 1}" for label in payload["labels"]]
        frame = numeric.source_dataset.dataframe.with_columns(pl.Series(cluster_name, labels))
        domain = _domain_with_role_overrides(frame, numeric.source_dataset, {cluster_name: "target"})
        annotations = {
            **numeric.source_dataset.annotations,
            "source_dataset_id": numeric.source_dataset.dataset_id,
            "cluster_column": cluster_name,
            "centroids": payload["centroids"].tolist(),
            "inertia": payload["inertia"],
            "silhouette": payload["silhouette"],
            "iterations": payload["iterations"],
            "init_method": payload["init_method"],
            "n_init": payload["n_init"],
            "optimized": payload["optimized"],
            "cluster_scores": payload["cluster_scores"],
        }
        return replace(
            numeric.source_dataset,
            dataset_id=f"kmeans-{numeric.source_dataset.dataset_id}",
            display_name=f"{numeric.source_dataset.display_name} (k-Means)",
            dataframe=frame,
            row_count=frame.height,
            column_count=frame.width,
            domain=domain,
            annotations=annotations,
        )


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
    return _numeric_from_array(np.asarray(frame, dtype=float), "Array")


def _dataset_numeric_names(dataset: DatasetHandle, prefer_feature_columns: bool) -> list[str]:
    columns = dataset.domain.feature_columns if prefer_feature_columns else dataset.domain.columns
    names = [column.name for column in columns if column.logical_type == "numeric"]
    if not names:
        names = [column.name for column in dataset.domain.columns if column.logical_type == "numeric"]
    return names


def _numeric_from_polars(frame: pl.DataFrame, names: list[str] | None, source_name: str, dataset: DatasetHandle | None) -> _NumericInput:
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
    return _NumericInput(np.column_stack(arrays), tuple(kept), dataset, source_name, imputed)


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
    return _NumericInput(np.column_stack(arrays), tuple(kept), None, source_name, imputed)


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


def _standardize(values: np.ndarray) -> np.ndarray:
    mean = values.mean(axis=0, keepdims=True)
    scale = values.std(axis=0, keepdims=True)
    scale[scale < 1e-12] = 1.0
    return (values - mean) / scale


def _safe_silhouette(values: np.ndarray, labels: np.ndarray, scorer: Any) -> float:
    unique_labels = set(int(v) for v in labels.tolist())
    if not (1 < len(unique_labels) < values.shape[0]):
        return float("nan")
    try:
        return float(scorer(values, labels, metric="euclidean"))
    except Exception as exc:
        print(f"OWKMeans warning: silhouette score failed ({exc}); continuing with NaN.", flush=True)
        return float("nan")


def _unique_column_name(dataset: DatasetHandle, base: str) -> str:
    existing = set(dataset.dataframe.columns)
    if base not in existing:
        return base
    index = 1
    while f"{base} ({index})" in existing:
        index += 1
    return f"{base} ({index})"


def _domain_with_role_overrides(frame: pl.DataFrame, dataset: DatasetHandle, roles: dict[str, str]) -> DataDomain:
    domain = build_data_domain(frame, source_domain=dataset.domain)
    return DataDomain(columns=tuple(replace(column, role=roles.get(column.name, column.role)) for column in domain.columns))


def _polars_is_numeric(dtype: object) -> bool:
    checker = getattr(dtype, "is_numeric", None)
    return bool(checker()) if callable(checker) else any(token in str(dtype).lower() for token in ("int", "float", "decimal"))


def _looks_like_pandas(value: object) -> bool:
    return hasattr(value, "select_dtypes") and hasattr(value, "to_numpy") and hasattr(value, "columns")


__all__ = ["OWKMeans", "WorkerSignals"]
