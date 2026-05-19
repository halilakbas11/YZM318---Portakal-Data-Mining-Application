from __future__ import annotations

import math
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn import metrics as skl_metrics

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QFont, QStandardItem, QStandardItemModel
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QTableView,
    QVBoxLayout,
    QWidget,
)

from portakal_app.data.models import DatasetHandle
from portakal_app.models import WorkflowPayload
from portakal_app.ui.screens.node_screen import WorkflowNodeScreenSupport


_CLF_METRICS = [
    ("AUC", "auc"),
    ("CA", "accuracy"),
    ("F1", "f1"),
    ("Precision", "precision"),
    ("Recall", "recall"),
]

_REG_METRICS = [
    ("MSE", "mse"),
    ("RMSE", "rmse"),
    ("MAE", "mae"),
    ("R²", "r2"),
]


class FeatureAsPredictorScreen(QWidget, WorkflowNodeScreenSupport):
    """Feature as Predictor — use a single feature column as a predictor.

    Discrete features predict directly by matching values.
    Continuous features can be transformed through a logistic (classification)
    or linear (regression) function.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._init_workflow_node_support()

        self._dataset: DatasetHandle | None = None
        self._valid_features: list[str] = []

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        self._status_label = QLabel("Connect Data with a target column.")
        self._status_label.setWordWrap(True)
        self._status_label.setStyleSheet("color: #777; background: transparent;")
        layout.addWidget(self._status_label)

        # Feature selection
        feature_box = QGroupBox("Feature Selection")
        feature_layout = QVBoxLayout(feature_box)

        feature_layout.addWidget(QLabel("Predict from:"))
        self._feature_combo = QComboBox()
        self._feature_combo.currentIndexChanged.connect(self._on_feature_changed)
        feature_layout.addWidget(self._feature_combo)

        self._cb_transform = QCheckBox("Transform through logistic/linear function")
        self._cb_transform.setChecked(False)
        self._cb_transform.stateChanged.connect(self._compute)
        feature_layout.addWidget(self._cb_transform)

        self._transform_info = QLabel("")
        self._transform_info.setStyleSheet("color: #666; font-size: 11px; background: transparent;")
        self._transform_info.setWordWrap(True)
        feature_layout.addWidget(self._transform_info)

        layout.addWidget(feature_box)

        # Results table — per-feature scores
        results_box = QGroupBox("All Features Ranking")
        results_layout = QVBoxLayout(results_box)
        self._results_model = QStandardItemModel(self)
        self._results_table = QTableView()
        self._results_table.setModel(self._results_model)
        self._results_table.setEditTriggers(QTableView.EditTrigger.NoEditTriggers)
        self._results_table.setSelectionMode(QTableView.SelectionMode.SingleSelection)
        self._results_table.setAlternatingRowColors(True)
        self._results_table.verticalHeader().hide()
        self._results_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        results_layout.addWidget(self._results_table)
        layout.addWidget(results_box, 1)

        # Selected feature metrics
        self._metric_label = QLabel("")
        self._metric_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._metric_label.setStyleSheet("font-weight: bold; background: transparent;")
        layout.addWidget(self._metric_label)

    # ── WorkflowNodeScreenSupport ─────────────────────────────────────

    def set_input_payload(self, payload: WorkflowPayload | None) -> None:
        if payload is None:
            self._dataset = None
        elif payload.port_label == "Data" and isinstance(payload.value, DatasetHandle):
            self._dataset = payload.value
        self._update_feature_list()
        self._compute()

    def current_output_payload(self) -> WorkflowPayload | None:
        if self._dataset is None or self._feature_combo.currentIndex() < 0 or not self._valid_features:
            return None
        feature_name = self._valid_features[self._feature_combo.currentIndex()]
        target_cols = self._dataset.domain.target_columns
        if not target_cols:
            return None
        target_col = target_cols[0]
        is_clf = target_col.logical_type in {"categorical", "boolean"}

        from portakal_app.data.services.sklearn_learner_service import SklearnLearnerService
        from sklearn.linear_model import LogisticRegression, LinearRegression
        from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
        from portakal_app.data.models import build_data_domain, DatasetHandle
        import uuid

        svc = SklearnLearnerService()
        feature_col = next((c for c in self._dataset.domain.columns if c.name == feature_name), None)
        is_discrete = feature_col is not None and feature_col.logical_type in {"categorical", "boolean"}

        if is_discrete:
            estimator = DecisionTreeClassifier(max_depth=10) if is_clf else DecisionTreeRegressor(max_depth=10)
        else:
            if is_clf and self._cb_transform.isChecked():
                estimator = LogisticRegression(max_iter=1000, solver="lbfgs")
            elif not is_clf:
                estimator = LinearRegression()
            else:
                estimator = LogisticRegression(max_iter=1000, solver="lbfgs")

        try:
            df = self._dataset.dataframe.select([feature_name, target_col.name])
            domain = build_data_domain(df, source_domain=self._dataset.domain)
            sub_dataset = DatasetHandle(
                dataset_id=str(uuid.uuid4()),
                display_name=f"{self._dataset.display_name} ({feature_name})",
                source=self._dataset.source,
                domain=domain,
                dataframe=df,
                row_count=df.shape[0],
                column_count=df.shape[1],
                cache_path=self._dataset.cache_path,
            )
            artifact = svc.fit(estimator, sub_dataset, f"Predictor: {feature_name}", type(estimator).__name__)
            return WorkflowPayload("Model", artifact)
        except Exception:
            return None

    def serialize_node_state(self) -> dict[str, object]:
        return {
            "feature_idx": self._feature_combo.currentIndex(),
            "transform": self._cb_transform.isChecked(),
        }

    def restore_node_state(self, payload: dict[str, object]) -> None:
        idx = int(payload.get("feature_idx", 0))
        if 0 <= idx < self._feature_combo.count():
            self._feature_combo.setCurrentIndex(idx)
        self._cb_transform.setChecked(bool(payload.get("transform", False)))

    # ── Internal logic ────────────────────────────────────────────────

    def _update_feature_list(self) -> None:
        self._feature_combo.blockSignals(True)
        self._feature_combo.clear()
        self._valid_features = []

        if self._dataset is None:
            self._status_label.setText("Connect Data with a target column.")
            self._status_label.setStyleSheet("color: #777; background: transparent;")
            self._feature_combo.blockSignals(False)
            return

        target_cols = self._dataset.domain.target_columns
        if not target_cols:
            self._status_label.setText("Dataset has no target column.")
            self._status_label.setStyleSheet("color: #c62828; background: transparent;")
            self._feature_combo.blockSignals(False)
            return

        target_col = target_cols[0]
        is_clf = target_col.logical_type in {"categorical", "boolean"}

        for col in self._dataset.domain.feature_columns:
            # For classification: accept categorical features (direct mapping)
            # and continuous features (logistic transform)
            # For regression: accept only continuous features (linear transform)
            if col.logical_type in {"numeric", "categorical", "boolean"}:
                if is_clf:
                    # For classification we accept all types
                    self._valid_features.append(col.name)
                else:
                    # For regression only numeric
                    if col.logical_type == "numeric":
                        self._valid_features.append(col.name)

        for name in self._valid_features:
            self._feature_combo.addItem(name)

        self._feature_combo.blockSignals(False)

        if not self._valid_features:
            self._status_label.setText("No suitable feature columns found.")
            self._status_label.setStyleSheet("color: #c62828; background: transparent;")

    def _on_feature_changed(self, index: int) -> None:
        self._update_transform_controls()
        self._compute()

    def _update_transform_controls(self) -> None:
        if self._dataset is None:
            return

        idx = self._feature_combo.currentIndex()
        if idx < 0 or idx >= len(self._valid_features):
            return

        feature_name = self._valid_features[idx]
        target_cols = self._dataset.domain.target_columns
        if not target_cols:
            return

        target_col = target_cols[0]
        is_clf = target_col.logical_type in {"categorical", "boolean"}

        # Find feature column schema
        feature_col = None
        for col in self._dataset.domain.columns:
            if col.name == feature_name:
                feature_col = col
                break

        if feature_col is None:
            return

        if feature_col.logical_type in {"categorical", "boolean"}:
            # Discrete feature — no transformation possible
            self._cb_transform.setChecked(False)
            self._cb_transform.setEnabled(False)
            self._transform_info.setText("Discrete feature: values used directly as predictions.")
        else:
            self._cb_transform.setEnabled(True)
            shape = "logistic" if is_clf else "linear"
            self._cb_transform.setText(f"Transform through {shape} function")
            self._transform_info.setText(
                f"Uses {shape} regression to fit prediction from this feature."
            )

    def _compute(self) -> None:
        self._results_model.clear()
        self._metric_label.setText("")
        self._transform_info.setText("")

        if self._dataset is None or not self._valid_features:
            return

        target_cols = self._dataset.domain.target_columns
        if not target_cols:
            return

        target_col = target_cols[0]
        is_clf = target_col.logical_type in {"categorical", "boolean"}
        metric_defs = _CLF_METRICS if is_clf else _REG_METRICS

        # Evaluate all features
        all_results: list[tuple[str, dict[str, float | str]]] = []

        target_series = self._dataset.dataframe.get_column(target_col.name)

        if is_clf:
            # Encode target
            target_values = target_series.drop_nulls().unique(maintain_order=True).to_list()
            target_map = {str(v): i for i, v in enumerate(target_values)}
            y_true = np.asarray(
                [target_map.get(str(v), 0) for v in target_series.to_list()],
                dtype=int,
            )
            n_classes = len(target_values)
        else:
            raw = []
            for v in target_series.to_list():
                try:
                    raw.append(float(v))
                except (TypeError, ValueError):
                    raw.append(float("nan"))
            y_arr = np.asarray(raw, dtype=float)
            finite_mask = np.isfinite(y_arr)
            mean_y = float(np.mean(y_arr[finite_mask])) if finite_mask.any() else 0.0
            y_true = np.where(finite_mask, y_arr, mean_y)
            n_classes = 0

        for feature_name in self._valid_features:
            scores = self._evaluate_feature(
                feature_name, y_true, is_clf, metric_defs,
                target_col, n_classes,
            )
            all_results.append((feature_name, scores))

        # Sort by first metric descending
        first_metric = metric_defs[0][0]
        all_results.sort(
            key=lambda x: (
                -(x[1].get(first_metric, 0.0) if isinstance(x[1].get(first_metric, 0.0), float) else 0.0)
            )
        )

        self._fill_results_table(all_results, metric_defs)

        # Update selected feature info
        selected_idx = self._feature_combo.currentIndex()
        if 0 <= selected_idx < len(self._valid_features):
            selected_name = self._valid_features[selected_idx]
            for name, scores in all_results:
                if name == selected_name:
                    parts = []
                    for mname, _ in metric_defs:
                        val = scores.get(mname, "—")
                        if isinstance(val, float):
                            parts.append(f"{mname}: {val:.4f}")
                        else:
                            parts.append(f"{mname}: {val}")
                    self._metric_label.setText(
                        f"Selected: {selected_name}  |  " + "  |  ".join(parts)
                    )
                    break

        self._update_transform_controls()

        n_features = len(all_results)
        self._status_label.setText(
            f"Dataset: {self._dataset.display_name}  |  "
            f"Target: {target_col.name}  |  "
            f"{n_features} feature(s) evaluated"
        )
        self._status_label.setStyleSheet("color: #2e7d32; background: transparent;")

    def _evaluate_feature(
        self,
        feature_name: str,
        y_true: np.ndarray,
        is_clf: bool,
        metric_defs: list[tuple[str, str]],
        target_col: Any,
        n_classes: int,
    ) -> dict[str, float | str]:
        """Evaluate a single feature as a predictor."""
        scores: dict[str, float | str] = {}

        try:
            series = self._dataset.dataframe.get_column(feature_name)
            feature_col = None
            for col in self._dataset.domain.columns:
                if col.name == feature_name:
                    feature_col = col
                    break

            is_discrete = feature_col is not None and feature_col.logical_type in {"categorical", "boolean"}

            if is_discrete and is_clf:
                # Direct mapping: feature values → target values
                y_pred = self._predict_discrete(series, y_true, n_classes)
            elif is_clf:
                # Continuous feature for classification
                X = self._encode_numeric_feature(series)
                use_transform = (
                    feature_name == self._valid_features[self._feature_combo.currentIndex()]
                    and self._cb_transform.isChecked()
                ) if self._feature_combo.currentIndex() >= 0 else False

                if use_transform:
                    y_pred = self._predict_logistic(X, y_true, n_classes)
                else:
                    y_pred = self._predict_logistic(X, y_true, n_classes)
            else:
                # Continuous feature for regression
                X = self._encode_numeric_feature(series)
                y_pred = self._predict_linear(X, y_true)

            # Compute metrics
            for mname, mkey in metric_defs:
                scores[mname] = self._compute_metric(mkey, y_true, y_pred, is_clf, n_classes)

        except Exception as exc:
            for mname, _ in metric_defs:
                scores[mname] = "Err"

        return scores

    def _encode_numeric_feature(self, series) -> np.ndarray:
        raw = []
        for v in series.to_list():
            try:
                raw.append(float(v))
            except (TypeError, ValueError):
                raw.append(float("nan"))
        arr = np.asarray(raw, dtype=float)
        finite = arr[np.isfinite(arr)]
        mean = float(np.mean(finite)) if finite.size else 0.0
        return np.where(np.isfinite(arr), arr, mean).reshape(-1, 1)

    def _predict_discrete(self, series, y_true: np.ndarray, n_classes: int) -> np.ndarray:
        """For discrete features, predict the most common target class for each feature value."""
        values = [str(v) for v in series.to_list()]
        unique_vals = sorted(set(values))

        # Build mapping: feature_value -> most common class
        val_to_class: dict[str, int] = {}
        for val in unique_vals:
            mask = np.asarray([v == val for v in values])
            if mask.any():
                subset = y_true[mask]
                counts = np.bincount(subset, minlength=n_classes)
                val_to_class[val] = int(np.argmax(counts))
            else:
                val_to_class[val] = 0

        return np.asarray([val_to_class.get(v, 0) for v in values], dtype=int)

    def _predict_logistic(self, X: np.ndarray, y: np.ndarray, n_classes: int) -> np.ndarray:
        """Fit logistic regression on single feature and predict."""
        try:
            model = LogisticRegression(max_iter=1000, solver="lbfgs")
            model.fit(X, y)
            return model.predict(X)
        except Exception:
            # Fallback: predict most common class
            return np.full(len(y), int(np.argmax(np.bincount(y))), dtype=int)

    def _predict_linear(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Fit linear regression on single feature and predict."""
        try:
            model = LinearRegression()
            model.fit(X, y)
            return model.predict(X)
        except Exception:
            return np.full(len(y), float(np.mean(y)))

    def _compute_metric(
        self,
        key: str,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        is_clf: bool,
        n_classes: int,
    ) -> float:
        try:
            if key == "accuracy":
                return float(skl_metrics.accuracy_score(y_true, y_pred))
            elif key == "f1":
                avg = "weighted" if n_classes > 2 else "binary"
                return float(skl_metrics.f1_score(y_true, y_pred, average=avg, zero_division=0))
            elif key == "precision":
                avg = "weighted" if n_classes > 2 else "binary"
                return float(skl_metrics.precision_score(y_true, y_pred, average=avg, zero_division=0))
            elif key == "recall":
                avg = "weighted" if n_classes > 2 else "binary"
                return float(skl_metrics.recall_score(y_true, y_pred, average=avg, zero_division=0))
            elif key == "auc":
                if n_classes == 2:
                    return float(skl_metrics.roc_auc_score(y_true, y_pred))
                else:
                    # Multi-class: use accuracy as proxy
                    return float(skl_metrics.accuracy_score(y_true, y_pred))
            elif key == "mse":
                return float(skl_metrics.mean_squared_error(y_true, y_pred))
            elif key == "rmse":
                return float(math.sqrt(skl_metrics.mean_squared_error(y_true, y_pred)))
            elif key == "mae":
                return float(skl_metrics.mean_absolute_error(y_true, y_pred))
            elif key == "r2":
                return float(skl_metrics.r2_score(y_true, y_pred))
        except Exception:
            pass
        return float("nan")

    def _fill_results_table(
        self,
        results: list[tuple[str, dict[str, float | str]]],
        metric_defs: list[tuple[str, str]],
    ) -> None:
        metric_names = [m for m, _ in metric_defs]
        headers = ["Feature"] + metric_names
        self._results_model.setColumnCount(len(headers))
        self._results_model.setRowCount(len(results))
        self._results_model.setHorizontalHeaderLabels(headers)

        bold = QFont()
        bold.setBold(True)

        for row_idx, (feature_name, scores) in enumerate(results):
            name_item = QStandardItem(feature_name)
            name_item.setFont(bold)
            name_item.setFlags(Qt.ItemFlag.ItemIsEnabled)
            self._results_model.setItem(row_idx, 0, name_item)

            for col_idx, mname in enumerate(metric_names, start=1):
                raw = scores.get(mname, float("nan"))
                if isinstance(raw, str):
                    item = QStandardItem(raw)
                    item.setForeground(QColor("#c62828"))
                elif isinstance(raw, float) and not math.isfinite(raw):
                    item = QStandardItem("—")
                    item.setForeground(QColor("#aaa"))
                else:
                    fmt_val = float(raw) if isinstance(raw, (int, float)) else 0.0
                    text = f"{fmt_val:.4f}"
                    item = QStandardItem(text)
                    # Color: green for high, red for low (for accuracy-like metrics)
                    if mname in {"CA", "F1", "Precision", "Recall", "AUC", "R²"}:
                        t = max(0.0, min(1.0, fmt_val))
                        r = int(200 - t * 100)
                        g = int(100 + t * 100)
                        item.setForeground(QColor(r, g, 60))
                    elif mname in {"MSE", "RMSE", "MAE"}:
                        t = max(0.0, 1.0 - min(1.0, fmt_val / 10.0))
                        r = int(200 - t * 100)
                        g = int(100 + t * 100)
                        item.setForeground(QColor(r, g, 60))

                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                item.setFlags(Qt.ItemFlag.ItemIsEnabled)
                self._results_model.setItem(row_idx, col_idx, item)

        self._results_table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.ResizeMode.ResizeToContents
        )
        for col in range(1, len(headers)):
            self._results_table.horizontalHeader().setSectionResizeMode(
                col, QHeaderView.ResizeMode.Stretch
            )
