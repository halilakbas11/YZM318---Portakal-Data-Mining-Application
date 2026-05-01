from __future__ import annotations

import math
from typing import Any

import numpy as np
from sklearn.model_selection import (
    KFold,
    LeaveOneOut,
    ShuffleSplit,
    StratifiedKFold,
    StratifiedShuffleSplit,
    cross_validate,
)
from sklearn import metrics as skl_metrics

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QFont, QStandardItem, QStandardItemModel
from PySide6.QtWidgets import (
    QComboBox,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QSpinBox,
    QTableView,
    QVBoxLayout,
    QWidget,
)

from portakal_app.data.models import DatasetHandle
from portakal_app.data.services.sklearn_learner_service import SklearnLearnerService
from portakal_app.models import WorkflowPayload
from portakal_app.sklearn_model_artifacts import SklearnModelArtifact
from portakal_app.ui.screens.node_screen import WorkflowNodeScreenSupport

_MAX_MODELS = 5

# Resampling method indices
_CV_KFOLD = 0
_CV_LOO = 1
_CV_SHUFFLE = 2
_CV_TRAIN = 3

_CLF_METRICS = [
    ("CA",        "accuracy"),
    ("F1",        "f1_weighted"),
    ("Precision", "precision_weighted"),
    ("Recall",    "recall_weighted"),
]

_REG_METRICS = [
    ("MSE",  "neg_mean_squared_error"),
    ("RMSE", "_rmse"),
    ("MAE",  "neg_mean_absolute_error"),
    ("R²",   "r2"),
]

_OTHER_ARTIFACT_TYPES: tuple[type, ...] = ()
try:
    from portakal_app.tree_artifacts import DecisionTreeArtifact, RandomForestArtifact
    _OTHER_ARTIFACT_TYPES += (DecisionTreeArtifact, RandomForestArtifact)
except ImportError:
    pass
try:
    from portakal_app.logistic_regression_artifacts import LogisticRegressionClassifierArtifact
    _OTHER_ARTIFACT_TYPES += (LogisticRegressionClassifierArtifact,)
except ImportError:
    pass


def _is_model_artifact(value: object) -> bool:
    if isinstance(value, SklearnModelArtifact):
        return True
    return isinstance(value, _OTHER_ARTIFACT_TYPES)


def _get_sklearn_estimator(artifact: object) -> object | None:
    """Return an unfitted sklearn estimator clone-able for cross-validation."""
    from sklearn.base import clone

    est = getattr(artifact, "sklearn_estimator", None)
    if est is not None:
        return est
    # Tree / RandomForest / etc. expose a fitted .trained_model — clone resets fit state.
    trained = getattr(artifact, "trained_model", None)
    if trained is not None:
        try:
            return clone(trained)
        except Exception:
            return None
    return None


class TestAndScoreScreen(QWidget, WorkflowNodeScreenSupport):
    """Test & Score — evaluate models with cross-validation."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._init_workflow_node_support()

        self._dataset: DatasetHandle | None = None
        self._models: dict[int, object] = {}
        self._svc = SklearnLearnerService()

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        self._status_label = QLabel("Connect Data and at least one Model.")
        self._status_label.setWordWrap(True)
        self._status_label.setStyleSheet("color: #777; background: transparent;")
        layout.addWidget(self._status_label)

        # ── Sampling method ───────────────────────────────────────────
        sampling_box = QGroupBox("Sampling")
        sampling_layout = QVBoxLayout(sampling_box)
        sampling_layout.setSpacing(6)

        self._method_combo = QComboBox()
        self._method_combo.addItems([
            "Cross Validation",
            "Leave One Out",
            "Random Sampling (ShuffleSplit)",
            "Test on Train Data",
        ])
        self._method_combo.currentIndexChanged.connect(self._on_method_changed)
        sampling_layout.addWidget(self._method_combo)

        # K-folds row (only visible for K-Fold CV)
        self._kfold_row = QWidget()
        kfold_layout = QHBoxLayout(self._kfold_row)
        kfold_layout.setContentsMargins(0, 0, 0, 0)
        kfold_layout.addWidget(QLabel("Folds:"))
        self._kfold_spin = QSpinBox()
        self._kfold_spin.setRange(2, 20)
        self._kfold_spin.setValue(5)
        self._kfold_spin.valueChanged.connect(self._run)
        kfold_layout.addWidget(self._kfold_spin)
        kfold_layout.addStretch(1)
        sampling_layout.addWidget(self._kfold_row)

        # Shuffle split row (only visible for ShuffleSplit)
        self._shuffle_row = QWidget()
        shuffle_layout = QHBoxLayout(self._shuffle_row)
        shuffle_layout.setContentsMargins(0, 0, 0, 0)
        shuffle_layout.addWidget(QLabel("Repeats:"))
        self._shuffle_repeats = QSpinBox()
        self._shuffle_repeats.setRange(1, 20)
        self._shuffle_repeats.setValue(5)
        self._shuffle_repeats.valueChanged.connect(self._run)
        shuffle_layout.addWidget(self._shuffle_repeats)
        shuffle_layout.addWidget(QLabel("Train %:"))
        self._shuffle_pct = QSpinBox()
        self._shuffle_pct.setRange(10, 95)
        self._shuffle_pct.setValue(70)
        self._shuffle_pct.setSuffix(" %")
        self._shuffle_pct.valueChanged.connect(self._run)
        shuffle_layout.addWidget(self._shuffle_pct)
        shuffle_layout.addStretch(1)
        sampling_layout.addWidget(self._shuffle_row)
        self._shuffle_row.hide()

        layout.addWidget(sampling_box)

        # ── Results table ─────────────────────────────────────────────
        self._table_model = QStandardItemModel(self)
        self._table = QTableView()
        self._table.setModel(self._table_model)
        self._table.setEditTriggers(QTableView.EditTrigger.NoEditTriggers)
        self._table.setSelectionMode(QTableView.SelectionMode.SingleSelection)
        self._table.setAlternatingRowColors(True)
        self._table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self._table.verticalHeader().hide()
        layout.addWidget(self._table, 1)

        self._on_method_changed(0)

    # ── WorkflowNodeScreenSupport ─────────────────────────────────────

    def set_input_payload(self, payload: WorkflowPayload | None) -> None:
        if payload is None:
            self._dataset = None
            self._models.clear()
        elif payload.port_label == "Data" and isinstance(payload.value, DatasetHandle):
            self._dataset = payload.value
        elif _is_model_artifact(payload.value):
            # Key by id(artifact) so multiple instances of the same model type stay distinct.
            self._models[id(payload.value)] = payload.value
        self._run()

    def current_output_payload(self) -> WorkflowPayload | None:
        return None

    def serialize_node_state(self) -> dict[str, object]:
        return {
            "method": self._method_combo.currentIndex(),
            "kfolds": self._kfold_spin.value(),
            "shuffle_repeats": self._shuffle_repeats.value(),
            "shuffle_pct": self._shuffle_pct.value(),
        }

    def restore_node_state(self, payload: dict[str, object]) -> None:
        self._method_combo.setCurrentIndex(int(payload.get("method", 0)))
        self._kfold_spin.setValue(int(payload.get("kfolds", 5)))
        self._shuffle_repeats.setValue(int(payload.get("shuffle_repeats", 5)))
        self._shuffle_pct.setValue(int(payload.get("shuffle_pct", 70)))

    # ── UI slots ──────────────────────────────────────────────────────

    def _on_method_changed(self, index: int) -> None:
        self._kfold_row.setVisible(index == _CV_KFOLD)
        self._shuffle_row.setVisible(index == _CV_SHUFFLE)
        self._run()

    # ── Evaluation ────────────────────────────────────────────────────

    def _run(self) -> None:
        self._table_model.clear()

        if self._dataset is None or not self._models:
            self._status_label.setText("Connect Data and at least one Model.")
            self._status_label.setStyleSheet("color: #777; background: transparent;")
            return

        target_cols = self._dataset.domain.target_columns
        if not target_cols:
            self._status_label.setText("Dataset has no target column.")
            self._status_label.setStyleSheet("color: #c62828; background: transparent;")
            return

        target_col = target_cols[0]
        is_clf = target_col.logical_type in {"categorical", "boolean"}
        metric_defs = _CLF_METRICS if is_clf else _REG_METRICS

        results: list[tuple[str, dict[str, float | str]]] = []
        first_artifact = next(iter(self._models.values()))
        feature_names: tuple[str, ...] = getattr(first_artifact, "feature_names", ()) or ()
        cat_encoders: dict = getattr(first_artifact, "categorical_encoders", {}) or {}
        num_cols: tuple[str, ...] = getattr(first_artifact, "numeric_cols", ()) or ()

        if not feature_names:
            self._status_label.setText("Models have no stored feature encoding — retrain to enable evaluation.")
            self._status_label.setStyleSheet("color: #c62828; background: transparent;")
            return

        try:
            numeric_means = self._compute_numeric_means(num_cols)
            X = self._svc.encode_X(self._dataset, feature_names, cat_encoders, num_cols, numeric_means)
            y = self._encode_y(is_clf)
        except Exception as exc:
            self._status_label.setText(f"Encoding error: {exc}")
            self._status_label.setStyleSheet("color: #c62828; background: transparent;")
            return

        cv = self._build_cv(is_clf, y)

        for artifact in self._models.values():
            display = getattr(artifact, "display_name", type(artifact).__name__)
            estimator = _get_sklearn_estimator(artifact)
            if estimator is None:
                results.append((display, {m: "N/A" for m, _ in metric_defs}))
                continue
            try:
                row = self._evaluate_one(estimator, X, y, is_clf, metric_defs, cv)
                results.append((display, row))
            except Exception as exc:
                error_row = {m: "Err" for m, _ in metric_defs}
                error_row["_error"] = str(exc)
                results.append((display, error_row))

        self._fill_table(results, metric_defs, is_clf)
        method_label = self._method_combo.currentText()
        self._status_label.setText(
            f"Dataset: {self._dataset.display_name}  |  "
            f"Method: {method_label}  |  "
            f"{len(results)} model(s) evaluated"
        )
        self._status_label.setStyleSheet("color: #2e7d32; background: transparent;")

    def _compute_numeric_means(self, num_cols: tuple[str, ...]) -> dict[str, float]:
        means: dict[str, float] = {}
        df = self._dataset.dataframe  # type: ignore[union-attr]
        for col_name in num_cols:
            if col_name in df.columns:
                series = df.get_column(col_name)
                raw = []
                for v in series.to_list():
                    try:
                        raw.append(float(v))
                    except (TypeError, ValueError):
                        raw.append(float("nan"))
                arr = np.asarray(raw, dtype=float)
                finite = arr[np.isfinite(arr)]
                means[col_name] = float(np.mean(finite)) if finite.size else 0.0
        return means

    def _encode_y(self, is_clf: bool) -> np.ndarray:
        target_col = self._dataset.domain.target_columns[0]  # type: ignore[union-attr]
        target_series = self._dataset.dataframe.get_column(target_col.name)  # type: ignore[union-attr]

        if is_clf:
            first_artifact = next(iter(self._models.values()))
            target_encoder: dict = getattr(first_artifact, "target_encoder", None) or {}
            return np.asarray(
                [target_encoder.get(str(v), 0) for v in target_series.to_list()],
                dtype=int,
            )
        else:
            raw = []
            for v in target_series.to_list():
                try:
                    raw.append(float(v))
                except (TypeError, ValueError):
                    raw.append(float("nan"))
            arr = np.asarray(raw, dtype=float)
            finite = arr[np.isfinite(arr)]
            mean_y = float(np.mean(finite)) if finite.size else 0.0
            return np.where(np.isfinite(arr), arr, mean_y)

    def _build_cv(self, is_clf: bool, y: np.ndarray) -> Any:
        method = self._method_combo.currentIndex()
        n_samples = len(y)

        if method == _CV_KFOLD:
            k = self._kfold_spin.value()
            if is_clf and len(np.unique(y)) > 1:
                min_class = int(np.bincount(y).min())
                if min_class >= k:
                    return StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
            return KFold(n_splits=min(k, n_samples), shuffle=True, random_state=42)

        if method == _CV_LOO:
            return LeaveOneOut()

        if method == _CV_SHUFFLE:
            n_repeats = self._shuffle_repeats.value()
            train_frac = self._shuffle_pct.value() / 100.0
            if is_clf and len(np.unique(y)) > 1:
                return StratifiedShuffleSplit(n_splits=n_repeats, train_size=train_frac, random_state=42)
            return ShuffleSplit(n_splits=n_repeats, train_size=train_frac, random_state=42)

        # Test on train data: single "fold" that uses all data for train+test
        # Implemented as a 2-fold CV so sklearn doesn't complain, but we override below
        return None  # handled specially in _evaluate_one

    def _evaluate_one(
        self,
        estimator: object,
        X: np.ndarray,
        y: np.ndarray,
        is_clf: bool,
        metric_defs: list[tuple[str, str]],
        cv: Any,
    ) -> dict[str, float | str]:
        from sklearn.base import clone

        est = clone(estimator)

        if cv is None:
            # Test on train data
            est.fit(X, y)
            y_pred = est.predict(X)
            return self._compute_scores_from_preds(y, y_pred, is_clf, metric_defs)

        sklearn_scoring: list[str] = []
        custom_metrics: list[tuple[str, str]] = []
        for name, key in metric_defs:
            if key == "_rmse":
                custom_metrics.append((name, key))
                sklearn_scoring.append("neg_mean_squared_error")
            elif key.startswith("neg_") or key in ("accuracy", "r2", "f1_weighted",
                                                     "precision_weighted", "recall_weighted"):
                sklearn_scoring.append(key)
                custom_metrics.append((name, key))
            else:
                custom_metrics.append((name, key))

        unique_scoring = list(dict.fromkeys(sklearn_scoring))
        cv_results = cross_validate(
            est, X, y,
            cv=cv,
            scoring=unique_scoring,
            return_train_score=False,
        )

        row: dict[str, float | str] = {}
        for name, key in metric_defs:
            if key == "_rmse":
                mse_scores = cv_results.get("test_neg_mean_squared_error", np.array([]))
                val = float(np.mean(np.sqrt(-mse_scores))) if mse_scores.size else float("nan")
            elif f"test_{key}" in cv_results:
                scores = cv_results[f"test_{key}"]
                val = float(np.mean(np.abs(scores))) if "neg_" in key else float(np.mean(scores))
            else:
                val = float("nan")
            row[name] = val
        return row

    def _compute_scores_from_preds(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        is_clf: bool,
        metric_defs: list[tuple[str, str]],
    ) -> dict[str, float | str]:
        row: dict[str, float | str] = {}
        for name, key in metric_defs:
            try:
                if key == "accuracy":
                    val = float(skl_metrics.accuracy_score(y_true, y_pred))
                elif key == "f1_weighted":
                    val = float(skl_metrics.f1_score(y_true, y_pred, average="weighted", zero_division=0))
                elif key == "precision_weighted":
                    val = float(skl_metrics.precision_score(y_true, y_pred, average="weighted", zero_division=0))
                elif key == "recall_weighted":
                    val = float(skl_metrics.recall_score(y_true, y_pred, average="weighted", zero_division=0))
                elif key == "neg_mean_squared_error":
                    val = float(skl_metrics.mean_squared_error(y_true, y_pred))
                elif key == "_rmse":
                    val = float(math.sqrt(skl_metrics.mean_squared_error(y_true, y_pred)))
                elif key == "neg_mean_absolute_error":
                    val = float(skl_metrics.mean_absolute_error(y_true, y_pred))
                elif key == "r2":
                    val = float(skl_metrics.r2_score(y_true, y_pred))
                else:
                    val = float("nan")
            except Exception:
                val = float("nan")
            row[name] = val
        return row

    def _fill_table(
        self,
        results: list[tuple[str, dict[str, float | str]]],
        metric_defs: list[tuple[str, str]],
        is_clf: bool,
    ) -> None:
        metric_names = [m for m, _ in metric_defs]
        headers = ["Model"] + metric_names
        self._table_model.setColumnCount(len(headers))
        self._table_model.setRowCount(len(results))
        self._table_model.setHorizontalHeaderLabels(headers)

        bold = QFont()
        bold.setBold(True)

        for row_idx, (model_name, scores) in enumerate(results):
            name_item = QStandardItem(model_name)
            name_item.setFont(bold)
            name_item.setFlags(Qt.ItemFlag.ItemIsEnabled)
            self._table_model.setItem(row_idx, 0, name_item)

            error_msg = scores.get("_error")
            for col_idx, mname in enumerate(metric_names, start=1):
                raw = scores.get(mname, float("nan"))
                if error_msg and isinstance(raw, str):
                    text = "Error"
                    item = QStandardItem(text)
                    item.setToolTip(str(error_msg))
                    item.setForeground(QColor("#c62828"))
                elif isinstance(raw, float) and not math.isfinite(raw):
                    item = QStandardItem("—")
                    item.setForeground(QColor("#aaa"))
                else:
                    fmt_val = float(raw) if isinstance(raw, (int, float)) else 0.0
                    if is_clf or mname == "R²":
                        text = f"{fmt_val:.4f}"
                        color = self._score_color(fmt_val, higher_better=True)
                    else:
                        text = f"{fmt_val:.4f}"
                        color = self._score_color(fmt_val, higher_better=False)
                    item = QStandardItem(text)
                    item.setForeground(color)
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                item.setFlags(Qt.ItemFlag.ItemIsEnabled)
                self._table_model.setItem(row_idx, col_idx, item)

        self._table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        for col in range(1, len(headers)):
            self._table.horizontalHeader().setSectionResizeMode(col, QHeaderView.ResizeMode.Stretch)

    @staticmethod
    def _score_color(value: float, higher_better: bool) -> QColor:
        if higher_better:
            t = max(0.0, min(1.0, value))
        else:
            t = max(0.0, 1.0 - min(1.0, value / 10.0))
        r = int(200 - t * 100)
        g = int(100 + t * 100)
        b = 60
        return QColor(r, g, b)
