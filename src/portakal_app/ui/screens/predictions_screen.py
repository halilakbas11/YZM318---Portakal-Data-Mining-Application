from __future__ import annotations

import math
from typing import Any

import numpy as np
import polars as pl

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QFont, QStandardItem, QStandardItemModel
from PySide6.QtWidgets import (
    QCheckBox,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QTableView,
    QVBoxLayout,
    QWidget,
)

from portakal_app.data.models import DatasetHandle, DataDomain, ColumnSchema, SourceInfo, build_data_domain
from portakal_app.data.services.sklearn_learner_service import SklearnLearnerService
from portakal_app.logistic_regression_artifacts import LogisticRegressionClassifierArtifact
from portakal_app.models import WorkflowPayload
from portakal_app.rule_artifacts import CN2RuleClassifierArtifact
from portakal_app.scoring_sheet_artifacts import ScoringSheetClassifierArtifact
from portakal_app.sklearn_model_artifacts import SklearnModelArtifact
from portakal_app.tree_artifacts import DecisionTreeArtifact, RandomForestArtifact
from portakal_app.ui.screens.node_screen import WorkflowNodeScreenSupport

_SKLEARN_TYPES = (
    SklearnModelArtifact,
    DecisionTreeArtifact,
    RandomForestArtifact,
    LogisticRegressionClassifierArtifact,
)
_RULE_TYPES = (ScoringSheetClassifierArtifact, CN2RuleClassifierArtifact)
_ALL_ARTIFACT_TYPES = _SKLEARN_TYPES + _RULE_TYPES


def _is_supported_artifact(value: object) -> bool:
    return isinstance(value, _ALL_ARTIFACT_TYPES)


class PredictionsScreen(QWidget, WorkflowNodeScreenSupport):
    """Predictions — show model predictions on data in a table."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._init_workflow_node_support()

        self._dataset: DatasetHandle | None = None
        self._models: dict[int, object] = {}
        self._svc = SklearnLearnerService()
        self._output_dataset: DatasetHandle | None = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        self._status_label = QLabel("Connect Data and at least one Model.")
        self._status_label.setWordWrap(True)
        self._status_label.setStyleSheet("color: #777; background: transparent;")
        layout.addWidget(self._status_label)

        # Options
        options_box = QGroupBox("Output")
        options_layout = QHBoxLayout(options_box)
        self._cb_predictions = QCheckBox("Show Predictions")
        self._cb_predictions.setChecked(True)
        self._cb_predictions.stateChanged.connect(self._compute)
        options_layout.addWidget(self._cb_predictions)
        self._cb_probabilities = QCheckBox("Show Probabilities")
        self._cb_probabilities.setChecked(False)
        self._cb_probabilities.stateChanged.connect(self._compute)
        options_layout.addWidget(self._cb_probabilities)
        layout.addWidget(options_box)

        # Table
        self._table_model = QStandardItemModel(self)
        self._table = QTableView()
        self._table.setModel(self._table_model)
        self._table.setEditTriggers(QTableView.EditTrigger.NoEditTriggers)
        self._table.setSelectionMode(QTableView.SelectionMode.SingleSelection)
        self._table.setAlternatingRowColors(True)
        self._table.verticalHeader().hide()
        layout.addWidget(self._table, 1)

    # ── WorkflowNodeScreenSupport ─────────────────────────────────────

    def set_input_payload(self, payload: WorkflowPayload | None) -> None:
        if payload is None:
            self._dataset = None
            self._models.clear()
        elif payload.port_label == "Data" and isinstance(payload.value, DatasetHandle):
            self._dataset = payload.value
        elif _is_supported_artifact(payload.value):
            self._models[id(payload.value)] = payload.value
        self._compute()

    def current_output_payload(self) -> WorkflowPayload | None:
        if self._output_dataset is None:
            return None
        return WorkflowPayload("Predictions", self._output_dataset)

    def serialize_node_state(self) -> dict[str, object]:
        return {
            "show_predictions": self._cb_predictions.isChecked(),
            "show_probabilities": self._cb_probabilities.isChecked(),
        }

    def restore_node_state(self, payload: dict[str, object]) -> None:
        self._cb_predictions.setChecked(bool(payload.get("show_predictions", True)))
        self._cb_probabilities.setChecked(bool(payload.get("show_probabilities", False)))

    # ── Computation ────────────────────────────────────────────────────

    def _compute(self) -> None:
        self._table_model.clear()
        self._output_dataset = None

        if self._dataset is None or not self._models:
            self._status_label.setText("Connect Data and at least one Model.")
            self._status_label.setStyleSheet("color: #777; background: transparent;")
            self._notify_output_changed()
            return

        target_cols = self._dataset.domain.target_columns
        target_name: str | None = None
        target_col = None
        if target_cols:
            target_col = target_cols[0]
            target_name = target_col.name

        show_predictions = self._cb_predictions.isChecked()
        show_probabilities = self._cb_probabilities.isChecked()

        # Gather data columns to show
        df = self._dataset.dataframe
        base_columns = list(df.columns)

        # Prepare predictions for each model
        model_results: list[tuple[str, np.ndarray | None, np.ndarray | None, tuple[str, ...] | None]] = []
        # (display_name, y_pred, y_prob, class_values)

        for artifact in self._models.values():
            display_name = getattr(artifact, "display_name", type(artifact).__name__)
            class_values = getattr(artifact, "class_values", ()) or ()
            is_clf = getattr(artifact, "is_classifier", None)
            if is_clf is None:
                is_clf = getattr(artifact, "kind", "") == "classification"

            try:
                if isinstance(artifact, _RULE_TYPES):
                    if not artifact.can_apply_to(self._dataset):
                        model_results.append((display_name, None, None, class_values))
                        continue
                    y_pred_raw = artifact.predict_from_dataset(self._dataset)
                    # Decode prediction indices to class labels
                    y_pred = y_pred_raw
                    y_prob = None
                else:
                    feature_names: tuple[str, ...] = getattr(artifact, "feature_names", ()) or ()
                    cat_encoders: dict = getattr(artifact, "categorical_encoders", {}) or {}
                    num_cols: tuple[str, ...] = getattr(artifact, "numeric_cols", ()) or ()
                    numeric_means: dict[str, float] = getattr(artifact, "numeric_means", {}) or {}

                    # Recalculate numeric_means from dataset when artifact doesn't store them
                    # (e.g. DecisionTreeArtifact, RandomForestArtifact)
                    if not numeric_means and num_cols:
                        for col_name in num_cols:
                            if col_name in self._dataset.dataframe.columns:
                                series = self._dataset.dataframe.get_column(col_name)
                                raw = []
                                for v in series.to_list():
                                    try:
                                        raw.append(float(v))
                                    except (TypeError, ValueError):
                                        raw.append(float("nan"))
                                arr = np.asarray(raw, dtype=float)
                                finite = arr[np.isfinite(arr)]
                                numeric_means[col_name] = float(np.mean(finite)) if finite.size else 0.0

                    if not feature_names:
                        model_results.append((display_name, None, None, class_values))
                        continue

                    X = self._svc.encode_X(
                        self._dataset, feature_names, cat_encoders, num_cols, numeric_means
                    )
                    y_pred = artifact.predict(X)
                    y_prob = None
                    if hasattr(artifact, "predict_proba"):
                        y_prob = artifact.predict_proba(X)

                model_results.append((display_name, y_pred, y_prob, class_values))

            except Exception as exc:
                model_results.append((display_name, None, None, None))

        # Build headers
        headers: list[str] = list(base_columns)
        for name, y_pred, y_prob, class_vals in model_results:
            if show_predictions:
                headers.append(f"{name}")
            if show_probabilities and y_prob is not None and class_vals:
                for cv in class_vals:
                    headers.append(f"p({cv})")

        # Build table
        n_rows = min(df.shape[0], 5000)  # Limit display rows
        self._table_model.setColumnCount(len(headers))
        self._table_model.setRowCount(n_rows)
        self._table_model.setHorizontalHeaderLabels(headers)

        bold = QFont()
        bold.setBold(True)

        # Fill data columns
        for col_idx, col_name in enumerate(base_columns):
            series = df.get_column(col_name)
            values = series.to_list()
            for row_idx in range(n_rows):
                val = values[row_idx] if row_idx < len(values) else ""
                item = QStandardItem(str(val) if val is not None else "")
                item.setFlags(Qt.ItemFlag.ItemIsEnabled)
                # Highlight target column
                if col_name == target_name:
                    item.setFont(bold)
                self._table_model.setItem(row_idx, col_idx, item)

        # Fill prediction and probability columns
        extra_col_offset = len(base_columns)
        for model_name, y_pred, y_prob, class_vals in model_results:
            if show_predictions:
                if y_pred is not None:
                    is_clf_model = class_vals and len(class_vals) > 0
                    for row_idx in range(n_rows):
                        if row_idx < len(y_pred):
                            pred_val = y_pred[row_idx]
                            if is_clf_model and isinstance(pred_val, (int, np.integer)):
                                pred_idx = int(pred_val)
                                if 0 <= pred_idx < len(class_vals):
                                    text = class_vals[pred_idx]
                                else:
                                    text = str(pred_val)
                            else:
                                text = f"{pred_val:.4f}" if isinstance(pred_val, float) else str(pred_val)
                        else:
                            text = "—"
                        item = QStandardItem(text)
                        item.setFlags(Qt.ItemFlag.ItemIsEnabled)
                        item.setFont(bold)

                        # Color correct/incorrect predictions
                        if target_name and target_col and row_idx < df.shape[0]:
                            actual_val = df.get_column(target_name).to_list()[row_idx]
                            if is_clf_model:
                                target_encoder: dict = getattr(
                                    list(self._models.values())[0], "target_encoder", {}
                                ) or {}
                                actual_encoded = target_encoder.get(str(actual_val), -1)
                                pred_idx_val = int(y_pred[row_idx]) if row_idx < len(y_pred) else -2
                                if actual_encoded == pred_idx_val:
                                    item.setForeground(QColor("#2e7d32"))
                                else:
                                    item.setForeground(QColor("#c62828"))

                        self._table_model.setItem(row_idx, extra_col_offset, item)
                else:
                    for row_idx in range(n_rows):
                        item = QStandardItem("N/A")
                        item.setForeground(QColor("#999"))
                        item.setFlags(Qt.ItemFlag.ItemIsEnabled)
                        self._table_model.setItem(row_idx, extra_col_offset, item)
                extra_col_offset += 1

            if show_probabilities and y_prob is not None and class_vals:
                for class_idx, cv in enumerate(class_vals):
                    for row_idx in range(n_rows):
                        if row_idx < y_prob.shape[0] and class_idx < y_prob.shape[1]:
                            prob = y_prob[row_idx, class_idx]
                            text = f"{prob:.4f}"
                            item = QStandardItem(text)
                            # Color by probability strength
                            intensity = max(0.0, min(1.0, float(prob)))
                            r = int(255 - intensity * 80)
                            g = int(255 - intensity * 40)
                            b = 255
                            item.setBackground(QColor(r, g, b))
                        else:
                            item = QStandardItem("—")
                        item.setFlags(Qt.ItemFlag.ItemIsEnabled)
                        item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                        self._table_model.setItem(row_idx, extra_col_offset, item)
                    extra_col_offset += 1

        # Add Metrics at the bottom
        if target_col is not None and show_predictions and len(model_results) > 0:
            target_series = df.get_column(target_name)
            is_clf_task = target_col.logical_type in {"categorical", "boolean"}

            if is_clf_task:
                target_values = target_series.drop_nulls().unique(maintain_order=True).to_list()
                target_map = {str(v): i for i, v in enumerate(target_values)}
                y_true = np.asarray([target_map.get(str(v), 0) for v in target_series.to_list()], dtype=int)
                n_classes = len(target_values)
                metric_defs = [("AUC", "auc"), ("CA", "accuracy"), ("F1", "f1"), ("Precision", "precision"), ("Recall", "recall")]
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
                metric_defs = [("MSE", "mse"), ("RMSE", "rmse"), ("MAE", "mae"), ("R²", "r2")]

            from sklearn import metrics as skl_metrics
            import math

            start_row = self._table_model.rowCount()
            self._table_model.setRowCount(start_row + len(metric_defs))

            for m_idx, (mname, mkey) in enumerate(metric_defs):
                row_idx = start_row + m_idx
                # Label
                item_label = QStandardItem(mname)
                item_label.setFont(bold)
                item_label.setBackground(QColor("#f0f0f0"))
                item_label.setFlags(Qt.ItemFlag.ItemIsEnabled)
                self._table_model.setItem(row_idx, 0, item_label)

                # For each model, compute metric
                extra_col_offset = len(base_columns)
                for model_name, y_pred, y_prob, class_vals in model_results:
                    if y_pred is not None:
                        val = float("nan")
                        try:
                            # Map predictions to integer indices if classification
                            if is_clf_task:
                                y_p = np.zeros(len(y_pred), dtype=int)
                                for i, p in enumerate(y_pred):
                                    idx = int(p) if isinstance(p, (int, np.integer)) else 0
                                    y_p[i] = idx
                            else:
                                y_p = y_pred

                            if mkey == "accuracy":
                                val = float(skl_metrics.accuracy_score(y_true, y_p))
                            elif mkey == "f1":
                                avg = "weighted" if n_classes > 2 else "binary"
                                val = float(skl_metrics.f1_score(y_true, y_p, average=avg, zero_division=0))
                            elif mkey == "precision":
                                avg = "weighted" if n_classes > 2 else "binary"
                                val = float(skl_metrics.precision_score(y_true, y_p, average=avg, zero_division=0))
                            elif mkey == "recall":
                                avg = "weighted" if n_classes > 2 else "binary"
                                val = float(skl_metrics.recall_score(y_true, y_p, average=avg, zero_division=0))
                            elif mkey == "auc":
                                if n_classes == 2 and y_prob is not None and y_prob.shape[1] == 2:
                                    val = float(skl_metrics.roc_auc_score(y_true, y_prob[:, 1]))
                                elif y_prob is not None and y_prob.shape[1] == n_classes:
                                    val = float(skl_metrics.roc_auc_score(y_true, y_prob, multi_class="ovr"))
                                else:
                                    val = float(skl_metrics.accuracy_score(y_true, y_p))
                            elif mkey == "mse":
                                val = float(skl_metrics.mean_squared_error(y_true, y_p))
                            elif mkey == "rmse":
                                val = float(math.sqrt(skl_metrics.mean_squared_error(y_true, y_p)))
                            elif mkey == "mae":
                                val = float(skl_metrics.mean_absolute_error(y_true, y_p))
                            elif mkey == "r2":
                                val = float(skl_metrics.r2_score(y_true, y_p))
                        except Exception:
                            pass

                        if math.isfinite(val):
                            text = f"{val:.4f}"
                            item = QStandardItem(text)
                            item.setFont(bold)
                            item.setBackground(QColor("#f8f8f8"))
                        else:
                            item = QStandardItem("—")
                            item.setBackground(QColor("#f8f8f8"))
                    else:
                        item = QStandardItem("—")
                        item.setBackground(QColor("#f8f8f8"))

                    item.setFlags(Qt.ItemFlag.ItemIsEnabled)
                    item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                    self._table_model.setItem(row_idx, extra_col_offset, item)
                    extra_col_offset += 1

                    if show_probabilities and y_prob is not None and class_vals:
                        extra_col_offset += len(class_vals)

        # Resize columns
        header = self._table.horizontalHeader()
        for c in range(len(base_columns)):
            header.setSectionResizeMode(c, QHeaderView.ResizeMode.Interactive)
        for c in range(len(base_columns), len(headers)):
            header.setSectionResizeMode(c, QHeaderView.ResizeMode.ResizeToContents)

        # Build output dataset with prediction columns appended
        self._build_output_dataset(model_results, show_predictions, show_probabilities)

        total_models = sum(1 for _, yp, _, _ in model_results if yp is not None)
        failed_models = sum(1 for _, yp, _, _ in model_results if yp is None)
        self._status_label.setText(
            f"Dataset: {self._dataset.display_name} ({df.shape[0]} rows)  |  "
            f"{total_models} model(s) predicted"
            + (f"  |  {failed_models} failed" if failed_models else "")
        )
        self._status_label.setStyleSheet("color: #2e7d32; background: transparent;")
        self._notify_output_changed()

    def _build_output_dataset(
        self,
        model_results: list[tuple[str, np.ndarray | None, np.ndarray | None, tuple[str, ...] | None]],
        show_predictions: bool,
        show_probabilities: bool,
    ) -> None:
        """Build an output DatasetHandle with prediction and probability columns appended."""
        if self._dataset is None:
            self._output_dataset = None
            return

        df = self._dataset.dataframe
        new_series: list[pl.Series] = []

        for model_name, y_pred, y_prob, class_vals in model_results:
            if show_predictions and y_pred is not None:
                is_clf_model = class_vals and len(class_vals) > 0
                if is_clf_model:
                    pred_labels = []
                    for p in y_pred:
                        idx = int(p) if isinstance(p, (int, np.integer)) else 0
                        if 0 <= idx < len(class_vals):
                            pred_labels.append(class_vals[idx])
                        else:
                            pred_labels.append(str(p))
                    new_series.append(pl.Series(f"{model_name}", pred_labels))
                else:
                    new_series.append(pl.Series(f"{model_name}", y_pred.tolist()))

            if show_probabilities and y_prob is not None and class_vals:
                for ci, cv in enumerate(class_vals):
                    if ci < y_prob.shape[1]:
                        new_series.append(
                            pl.Series(f"p({cv})_{model_name}", y_prob[:, ci].tolist())
                        )

        if new_series:
            new_df = df.hstack(new_series)
        else:
            new_df = df

        new_domain = build_data_domain(new_df, source_domain=self._dataset.domain)
        import uuid
        self._output_dataset = DatasetHandle(
            dataset_id=str(uuid.uuid4()),
            display_name=f"{self._dataset.display_name} (Predictions)",
            source=self._dataset.source,
            domain=new_domain,
            dataframe=new_df,
            row_count=new_df.shape[0],
            column_count=new_df.shape[1],
            cache_path=self._dataset.cache_path,
        )
