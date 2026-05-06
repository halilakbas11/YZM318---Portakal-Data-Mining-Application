from __future__ import annotations

"""
Principal Component Analysis with thread-safe Orange-style data flow.

For X in R^{n x p}, PCA centers X to X_c and studies the empirical covariance
operator C = (1/(n-1)) X_c^T X_c. The spectral equation C v_j = lambda_j v_j
finds orthonormal axes v_j that extremize variance under unit-length and
orthogonality constraints. The projected coordinates are Z = X_c V_k.

The full SVD solver computes an exact factorization X_c = U Sigma V^T and is
geometrically equivalent to rotating the covariance hyper-ellipsoid onto its
principal axes. The arpack solver is an iterative Krylov-subspace eigensolver:
it approximates only selected extremal singular directions, bending computation
toward a low-dimensional invariant subspace without constructing the entire
spectrum. The randomized solver samples a probabilistic range basis before SVD;
it preserves dominant geometry in expectation and trades exact topology of the
full covariance operator for a fast approximate subspace. The auto solver
selects among these methods according to matrix shape and requested dimension.

Topologically, PCA is a continuous linear map from R^p to R^k. It preserves
linear neighborhoods within span(V_k) and collapses the orthogonal complement,
identifying points that differ only in discarded directions. Standardization
changes the metric tensor before decomposition by rescaling coordinate axes,
so the covariance ellipsoid is interpreted in correlation-like coordinates
rather than raw measurement units.
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

from portakal_app.data.models import DatasetHandle, build_data_domain
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


class PCAWorker(QRunnable):
    """Background PCA worker; all UI updates are handled by receiver slots."""

    def __init__(
        self,
        *,
        job_id: int,
        numeric: _NumericInput,
        n_components: int,
        selection_mode: str,
        variance_threshold: float,
        solver: str,
        standardize: bool,
    ) -> None:
        super().__init__()
        self.signals = WorkerSignals()
        self._job_id = job_id
        self._numeric = numeric
        self._n_components = n_components
        self._selection_mode = selection_mode
        self._variance_threshold = variance_threshold
        self._solver = solver
        self._standardize = standardize

    @Slot()
    def run(self) -> None:
        try:
            from sklearn.decomposition import PCA

            values = self._numeric.values.astype(float, copy=True)
            if self._standardize:
                values = _standardize(values)
            limit = min(values.shape)
            requested_components = max(1, min(self._n_components, limit))
            fit_components = limit if self._selection_mode == "variance" else requested_components
            solver = self._solver
            if solver == "arpack" and limit <= 1:
                solver = "full"
            elif solver == "arpack" and fit_components >= limit:
                fit_components = max(1, limit - 1)
            pca = PCA(n_components=fit_components, svd_solver=solver, random_state=42 if solver == "randomized" else None)
            full_projected = pca.fit_transform(values)
            eigenvalues = np.asarray(pca.explained_variance_, dtype=float)
            ratios = np.asarray(pca.explained_variance_ratio_, dtype=float)
            if self._selection_mode == "variance":
                cumulative = np.cumsum(ratios)
                selected_components = int(np.searchsorted(cumulative, self._variance_threshold, side="left") + 1)
                selected_components = max(1, min(selected_components, full_projected.shape[1]))
            else:
                selected_components = full_projected.shape[1]
            projected = full_projected[:, :selected_components]
            self.signals.result_ready.emit(
                {
                    "job_id": self._job_id,
                    "projected": projected,
                    "components": np.asarray(pca.components_, dtype=float)[:selected_components],
                    "eigenvalues": eigenvalues,
                    "explained_variance_ratio": ratios,
                    "feature_names": self._numeric.feature_names,
                    "solver": solver,
                    "selection_mode": self._selection_mode,
                    "variance_threshold": self._variance_threshold,
                    "selected_components": selected_components,
                    "standardized": self._standardize,
                    "imputed_count": self._numeric.imputed_count,
                }
            )
        except Exception as exc:
            self.signals.error_ready.emit({"job_id": self._job_id, "error": str(exc)})


class OWPCA(QWidget, WorkflowNodeScreenSupport):
    """
    Production-grade PCA workflow node.

    The widget robustly extracts numeric data from DatasetHandle, Polars,
    pandas-like, custom dataframe-holding objects, or array-like objects. It
    imputes missing numeric values by column means, converts to numpy, and
    dispatches PCA to a QRunnable. The worker emits WorkerSignals.result_ready
    with eigenvalues, explained variance ratios, components, and projected
    coordinates; this main-thread widget then populates the table and emits
    the reduced dataset through output_signal.

    Solver geometry is explicit. full SVD preserves the complete covariance
    spectrum; arpack approximates extremal eigendirections through a Krylov
    subspace; randomized SVD samples an approximate dominant range; auto lets
    scikit-learn choose based on dimensional regime. Each solver returns a
    linear projection whose topology collapses discarded eigenspaces while
    preserving continuous coordinates in the retained principal subspace.
    """

    output_signal = Signal(object)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._init_workflow_node_support()
        self._thread_pool = QThreadPool.globalInstance()
        self._job_id = 0
        self._active_worker: PCAWorker | None = None
        self._numeric: _NumericInput | None = None
        self._output_dataset: DatasetHandle | None = None
        self._last_payload: dict[str, Any] | None = None
        self._n_components = 2
        self._selection_mode = "fixed"
        self._variance_threshold = 0.95
        self._solver = "auto"
        self._standardize_enabled = True
        self._auto_apply_enabled = True

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        title = QLabel(i18n.t("PCA"))
        title.setProperty("sectionTitle", True)
        layout.addWidget(title)

        self._summary = QLabel(i18n.t("Load a dataset with numeric columns."))
        self._summary.setWordWrap(True)
        self._summary.setProperty("muted", True)
        layout.addWidget(self._summary)

        settings = QGroupBox(i18n.t("Settings"))
        form = QFormLayout(settings)
        self._selection_combo = QComboBox()
        self._selection_combo.addItem(i18n.t("Fixed number of components"), "fixed")
        self._selection_combo.addItem(i18n.t("Retain variance percentage"), "variance")
        form.addRow(i18n.t("Selection:"), self._selection_combo)
        self._components_spin = QSpinBox()
        self._components_spin.setRange(1, 100)
        self._components_spin.setValue(2)
        form.addRow(i18n.t("Components:"), self._components_spin)
        self._variance_spin = QSpinBox()
        self._variance_spin.setRange(1, 100)
        self._variance_spin.setValue(95)
        self._variance_spin.setSuffix("%")
        form.addRow(i18n.t("Variance covered:"), self._variance_spin)
        self._solver_combo = QComboBox()
        for solver in ("auto", "full", "arpack", "randomized"):
            self._solver_combo.addItem(i18n.t(solver), solver)
        form.addRow(i18n.t("SVD Solver:"), self._solver_combo)
        self._standardize = QCheckBox(i18n.t("Standardize Data"))
        self._standardize.setChecked(True)
        form.addRow("", self._standardize)
        layout.addWidget(settings)

        self._table = QTableWidget(0, 3)
        self._table.setHorizontalHeaderLabels(
            [i18n.t("Principal Component"), i18n.t("Eigenvalue"), i18n.t("Explained Variance Ratio")]
        )
        layout.addWidget(self._table, 1)

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
        self._selection_combo.currentIndexChanged.connect(self._handle_settings_changed)
        self._components_spin.valueChanged.connect(self._handle_settings_changed)
        self._variance_spin.valueChanged.connect(self._handle_settings_changed)
        self._solver_combo.currentIndexChanged.connect(self._handle_settings_changed)
        self._standardize.stateChanged.connect(self._handle_settings_changed)
        self.cb_apply_auto.stateChanged.connect(self._handle_auto_apply_changed)

    @Slot(object)
    def receive_data(self, data: object) -> None:
        self.set_input_payload(data if isinstance(data, WorkflowPayload) else WorkflowPayload("Data", data))

    @Slot(object)
    def set_input_payload(self, payload: WorkflowPayload | None) -> None:
        self._output_dataset = None
        self._last_payload = None
        self._table.setRowCount(0)
        try:
            self._numeric = _extract_numeric_input(payload, prefer_feature_columns=True)
            max_components = max(1, min(self._numeric.values.shape))
            self._components_spin.setMaximum(max_components)
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
            "components": self._n_components,
            "selection_mode": self._selection_mode,
            "variance_threshold": self._variance_threshold,
            "solver": self._solver,
            "standardize": self._standardize_enabled,
            "auto_apply": self._auto_apply_enabled,
        }

    def restore_node_state(self, payload: dict[str, object]) -> None:
        selection_index = self._selection_combo.findData(str(payload.get("selection_mode", "fixed")))
        self._selection_combo.setCurrentIndex(max(0, selection_index))
        self._components_spin.setValue(int(payload.get("components", 2)))
        self._variance_spin.setValue(int(round(float(payload.get("variance_threshold", 0.95)) * 100)))
        index = self._solver_combo.findData(str(payload.get("solver", "auto")))
        self._solver_combo.setCurrentIndex(max(0, index))
        self._standardize.setChecked(bool(payload.get("standardize", True)))
        self.cb_apply_auto.setChecked(bool(payload.get("auto_apply", True)))
        self._sync_state_from_controls()
        if self._auto_apply_enabled:
            self._apply()

    def _sync_state_from_controls(self) -> None:
        self._n_components = int(self._components_spin.value())
        self._selection_mode = str(self._selection_combo.currentData())
        self._variance_threshold = float(self._variance_spin.value()) / 100.0
        self._solver = str(self._solver_combo.currentData())
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
        worker = PCAWorker(
            job_id=self._job_id,
            numeric=self._numeric,
            n_components=self._n_components,
            selection_mode=self._selection_mode,
            variance_threshold=self._variance_threshold,
            solver=self._solver,
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
        self._populate_table(payload)
        self._output_dataset = self._build_output_dataset(payload)
        self._summary.setText(
            i18n.tf(
                "Output: {rows} rows, {cols} columns",
                rows=payload["projected"].shape[0],
                cols=payload["projected"].shape[1],
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

    def _populate_table(self, payload: dict) -> None:
        eigenvalues = payload["eigenvalues"]
        ratios = payload["explained_variance_ratio"]
        self._table.setRowCount(len(eigenvalues))
        for index, (eigenvalue, ratio) in enumerate(zip(eigenvalues, ratios)):
            self._table.setItem(index, 0, QTableWidgetItem(f"PC{index + 1}"))
            self._table.setItem(index, 1, QTableWidgetItem(f"{float(eigenvalue):.8g}"))
            self._table.setItem(index, 2, QTableWidgetItem(f"{float(ratio):.6%}"))
        self._table.resizeColumnsToContents()

    def _build_output_dataset(self, payload: dict) -> DatasetHandle | None:
        numeric = self._numeric
        if numeric is None or numeric.source_dataset is None:
            return None
        projected = payload["projected"]
        pc_columns = {
            _unique_column_name(numeric.source_dataset, f"PC{i + 1}"): projected[:, i]
            for i in range(projected.shape[1])
        }
        frame = numeric.source_dataset.dataframe.with_columns(
            [pl.Series(name, values).cast(pl.Float64) for name, values in pc_columns.items()]
        )
        annotations = {
            **numeric.source_dataset.annotations,
            "source_dataset_id": numeric.source_dataset.dataset_id,
            "feature_names": list(payload["feature_names"]),
            "component_columns": list(pc_columns),
            "components": payload["components"].tolist(),
            "eigenvalues": payload["eigenvalues"].tolist(),
            "explained_variance_ratio": payload["explained_variance_ratio"].tolist(),
            "solver": payload["solver"],
            "selection_mode": payload["selection_mode"],
            "variance_threshold": payload["variance_threshold"],
            "selected_components": payload["selected_components"],
            "standardized": payload["standardized"],
        }
        return replace(
            numeric.source_dataset,
            dataset_id=f"pca-{numeric.source_dataset.dataset_id}",
            display_name=f"{numeric.source_dataset.display_name} (PCA)",
            dataframe=frame,
            row_count=frame.height,
            column_count=frame.width,
            domain=build_data_domain(frame, source_domain=numeric.source_dataset.domain),
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


def _polars_is_numeric(dtype: object) -> bool:
    checker = getattr(dtype, "is_numeric", None)
    return bool(checker()) if callable(checker) else any(token in str(dtype).lower() for token in ("int", "float", "decimal"))


def _unique_column_name(dataset: DatasetHandle, base: str) -> str:
    existing = set(dataset.dataframe.columns)
    if base not in existing:
        return base
    index = 1
    while f"{base} ({index})" in existing:
        index += 1
    return f"{base} ({index})"


def _looks_like_pandas(value: object) -> bool:
    return hasattr(value, "select_dtypes") and hasattr(value, "to_numpy") and hasattr(value, "columns")


__all__ = ["OWPCA", "WorkerSignals"]
