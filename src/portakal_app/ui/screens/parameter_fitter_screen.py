from __future__ import annotations

import math
from typing import Any

import numpy as np
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold, KFold, cross_val_score

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from portakal_app.data.models import DatasetHandle
from portakal_app.data.services.sklearn_learner_service import SklearnLearnerService
from portakal_app.sklearn_model_artifacts import SklearnModelArtifact
from portakal_app.ui.screens.node_screen import WorkflowNodeScreenSupport

try:
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.figure import Figure
except ImportError:
    FigureCanvas = None
    Figure = None

# Artifact types that expose a sklearn estimator we can clone
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
    """Return an unfitted sklearn estimator that can be cloned."""
    est = getattr(artifact, "sklearn_estimator", None)
    if est is not None:
        return est
    trained = getattr(artifact, "trained_model", None)
    if trained is not None:
        try:
            return clone(trained)
        except Exception:
            return None
    return None


# Known tunable integer parameters by estimator class name
_KNOWN_PARAMS: dict[str, list[tuple[str, str, int, int]]] = {
    # (param_name, display_label, default_min, default_max)
    "RandomForestClassifier": [
        ("n_estimators", "Number of Trees (n_estimators)", 1, 200),
        ("max_depth", "Max Depth (max_depth)", 1, 30),
    ],
    "RandomForestRegressor": [
        ("n_estimators", "Number of Trees (n_estimators)", 1, 200),
        ("max_depth", "Max Depth (max_depth)", 1, 30),
    ],
    "KNeighborsClassifier": [
        ("n_neighbors", "Number of Neighbors (n_neighbors)", 1, 50),
    ],
    "KNeighborsRegressor": [
        ("n_neighbors", "Number of Neighbors (n_neighbors)", 1, 50),
    ],
    "GradientBoostingClassifier": [
        ("n_estimators", "Number of Estimators (n_estimators)", 10, 300),
        ("max_depth", "Max Depth (max_depth)", 1, 20),
    ],
    "GradientBoostingRegressor": [
        ("n_estimators", "Number of Estimators (n_estimators)", 10, 300),
        ("max_depth", "Max Depth (max_depth)", 1, 20),
    ],
    "AdaBoostClassifier": [
        ("n_estimators", "Number of Estimators (n_estimators)", 10, 300),
    ],
    "AdaBoostRegressor": [
        ("n_estimators", "Number of Estimators (n_estimators)", 10, 300),
    ],
    "DecisionTreeClassifier": [
        ("max_depth", "Max Depth (max_depth)", 1, 30),
    ],
    "DecisionTreeRegressor": [
        ("max_depth", "Max Depth (max_depth)", 1, 30),
    ],
    "PLSRegression": [
        ("n_components", "Number of Components (n_components)", 1, 20),
    ],
    "MLPClassifier": [
        ("max_iter", "Max Iterations (max_iter)", 50, 1000),
    ],
    "MLPRegressor": [
        ("max_iter", "Max Iterations (max_iter)", 50, 1000),
    ],
    "SGDClassifier": [
        ("max_iter", "Max Iterations (max_iter)", 100, 2000),
    ],
    "SGDRegressor": [
        ("max_iter", "Max Iterations (max_iter)", 100, 2000),
    ],
}


class ParameterFitterScreen(QWidget, WorkflowNodeScreenSupport):
    """Parameter Fitter — sweep a hyperparameter and plot cross-validated performance."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._init_workflow_node_support()

        self._dataset: DatasetHandle | None = None
        self._model: object | None = None
        self._svc = SklearnLearnerService()
        self._available_params: list[tuple[str, str, int, int]] = []

        self._figure = Figure(figsize=(6, 4), facecolor="#f8f8f8") if Figure is not None else None
        self._canvas = FigureCanvas(self._figure) if FigureCanvas is not None and self._figure is not None else None

        self._build_ui()

    def _build_ui(self) -> None:
        root = QHBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)

        # ── Left panel: controls ──────────────────────────────────────
        left_panel = QWidget()
        left_panel.setFixedWidth(260)
        left = QVBoxLayout(left_panel)
        left.setContentsMargins(12, 12, 8, 12)
        left.setSpacing(8)

        # Parameter selection
        param_box = QGroupBox("Parameter")
        param_layout = QVBoxLayout(param_box)
        param_layout.addWidget(QLabel("Tune parameter:"))
        self._param_combo = QComboBox()
        self._param_combo.currentIndexChanged.connect(self._on_param_changed)
        param_layout.addWidget(self._param_combo)
        left.addWidget(param_box)

        # Range settings
        range_box = QGroupBox("Range")
        range_layout = QVBoxLayout(range_box)

        min_row = QHBoxLayout()
        min_row.addWidget(QLabel("Min:"))
        self._min_spin = QSpinBox()
        self._min_spin.setRange(1, 10000)
        self._min_spin.setValue(1)
        min_row.addWidget(self._min_spin)
        range_layout.addLayout(min_row)

        max_row = QHBoxLayout()
        max_row.addWidget(QLabel("Max:"))
        self._max_spin = QSpinBox()
        self._max_spin.setRange(1, 10000)
        self._max_spin.setValue(100)
        max_row.addWidget(self._max_spin)
        range_layout.addLayout(max_row)

        step_row = QHBoxLayout()
        step_row.addWidget(QLabel("Step:"))
        self._step_spin = QSpinBox()
        self._step_spin.setRange(1, 1000)
        self._step_spin.setValue(10)
        step_row.addWidget(self._step_spin)
        range_layout.addLayout(step_row)

        manual_row = QHBoxLayout()
        manual_row.addWidget(QLabel("Manual:"))
        self._manual_edit = QLineEdit()
        self._manual_edit.setPlaceholderText("e.g. 1,5,10,20,50")
        manual_row.addWidget(self._manual_edit)
        range_layout.addLayout(manual_row)

        left.addWidget(range_box)

        # CV settings
        cv_box = QGroupBox("Cross Validation")
        cv_layout = QHBoxLayout(cv_box)
        cv_layout.addWidget(QLabel("Folds:"))
        self._cv_spin = QSpinBox()
        self._cv_spin.setRange(2, 20)
        self._cv_spin.setValue(5)
        cv_layout.addWidget(self._cv_spin)
        left.addWidget(cv_box)

        # Run & Save buttons
        self._run_btn = QPushButton("Run")
        self._run_btn.setProperty("primary", True)
        self._run_btn.clicked.connect(self._run_fitting)
        left.addWidget(self._run_btn)

        save_btn = QPushButton("Save Image")
        save_btn.clicked.connect(self._save_image)
        left.addWidget(save_btn)

        left.addStretch(1)
        root.addWidget(left_panel)

        # ── Right panel: plot ─────────────────────────────────────────
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        if self._canvas is not None:
            right_layout.addWidget(self._canvas, 1)
        self._info_label = QLabel("Connect Data and a Model to begin.")
        self._info_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._info_label.setWordWrap(True)
        right_layout.addWidget(self._info_label)
        root.addWidget(right_panel, 1)

    # ── WorkflowNodeScreenSupport ─────────────────────────────────────

    def set_input_payload(self, payload) -> None:
        from portakal_app.models import WorkflowPayload

        if payload is None:
            self._dataset = None
            self._model = None
        elif payload.port_label == "Data" and isinstance(payload.value, DatasetHandle):
            self._dataset = payload.value
        elif _is_model_artifact(payload.value):
            self._model = payload.value
        self._update_param_list()

    def current_output_payload(self):
        return None

    def serialize_node_state(self) -> dict[str, object]:
        return {
            "min": self._min_spin.value(),
            "max": self._max_spin.value(),
            "step": self._step_spin.value(),
            "cv": self._cv_spin.value(),
            "param_idx": self._param_combo.currentIndex(),
            "manual": self._manual_edit.text(),
        }

    def restore_node_state(self, payload: dict[str, object]) -> None:
        self._min_spin.setValue(int(payload.get("min", 1)))
        self._max_spin.setValue(int(payload.get("max", 100)))
        self._step_spin.setValue(int(payload.get("step", 10)))
        self._cv_spin.setValue(int(payload.get("cv", 5)))
        idx = int(payload.get("param_idx", 0))
        if 0 <= idx < self._param_combo.count():
            self._param_combo.setCurrentIndex(idx)
        manual = str(payload.get("manual", ""))
        if manual:
            self._manual_edit.setText(manual)

    # ── Internal logic ────────────────────────────────────────────────

    def _update_param_list(self) -> None:
        self._param_combo.clear()
        self._available_params = []

        if self._model is None:
            self._info_label.setText("Connect Data and a Model to begin.")
            return

        estimator = _get_sklearn_estimator(self._model)
        if estimator is None:
            self._info_label.setText("Connected model does not expose a sklearn estimator.")
            return

        class_name = type(estimator).__name__
        params = _KNOWN_PARAMS.get(class_name, [])

        # Fallback: try to discover integer params from get_params()
        if not params:
            try:
                est_params = estimator.get_params(deep=False)
                for k, v in est_params.items():
                    if isinstance(v, int) and v > 0:
                        params.append((k, k, 1, max(v * 3, 10)))
            except Exception:
                pass

        if not params:
            self._info_label.setText(
                f"No tunable integer parameters found for {class_name}."
            )
            return

        self._available_params = params
        for _, label, _, _ in params:
            self._param_combo.addItem(label)

        if params:
            _, _, default_min, default_max = params[0]
            self._min_spin.setValue(default_min)
            self._max_spin.setValue(default_max)
            auto_step = max(1, (default_max - default_min) // 10)
            self._step_spin.setValue(auto_step)

        if self._dataset is not None:
            self._info_label.setText("Ready. Press Run to start parameter fitting.")
        else:
            self._info_label.setText("Connect Data to the widget.")

    def _on_param_changed(self, index: int) -> None:
        if 0 <= index < len(self._available_params):
            _, _, default_min, default_max = self._available_params[index]
            self._min_spin.setValue(default_min)
            self._max_spin.setValue(default_max)
            auto_step = max(1, (default_max - default_min) // 10)
            self._step_spin.setValue(auto_step)

    def _get_param_values(self) -> list[int]:
        """Return the list of parameter values to sweep."""
        manual_text = self._manual_edit.text().strip()
        if manual_text:
            try:
                values = [int(v.strip()) for v in manual_text.split(",") if v.strip()]
                return sorted(set(v for v in values if v > 0))
            except ValueError:
                pass

        lo = self._min_spin.value()
        hi = self._max_spin.value()
        step = self._step_spin.value()
        if lo > hi:
            lo, hi = hi, lo
        values = list(range(lo, hi + 1, step))
        if values and values[-1] != hi:
            values.append(hi)
        return values

    def _run_fitting(self) -> None:
        if self._dataset is None or self._model is None:
            self._info_label.setText("Connect Data and a Model first.")
            return

        idx = self._param_combo.currentIndex()
        if idx < 0 or idx >= len(self._available_params):
            self._info_label.setText("Select a parameter to tune.")
            return

        param_name, param_label, _, _ = self._available_params[idx]
        estimator = _get_sklearn_estimator(self._model)
        if estimator is None:
            self._info_label.setText("Cannot obtain sklearn estimator from model.")
            return

        # Determine classification vs regression
        target_cols = self._dataset.domain.target_columns
        if not target_cols:
            self._info_label.setText("Dataset has no target column.")
            return

        target_col = target_cols[0]
        is_clf = target_col.logical_type in {"categorical", "boolean"}

        try:
            artifact = self._model
            feature_names: tuple[str, ...] = getattr(artifact, "feature_names", ()) or ()
            cat_encoders: dict = getattr(artifact, "categorical_encoders", {}) or {}
            num_cols: tuple[str, ...] = getattr(artifact, "numeric_cols", ()) or ()
            numeric_means: dict[str, float] = getattr(artifact, "numeric_means", {}) or {}

            # Recalculate numeric_means from dataset when artifact doesn't store them
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
                X, feature_names, cat_encoders, num_cols, numeric_means = self._svc.prepare_features(self._dataset)
            else:
                X = self._svc.encode_X(self._dataset, feature_names, cat_encoders, num_cols, numeric_means)

            # Encode y
            target_series = self._dataset.dataframe.get_column(target_col.name)
            if is_clf:
                target_encoder: dict = getattr(artifact, "target_encoder", None) or {}
                if not target_encoder:
                    from portakal_app.data.services.sklearn_learner_service import _safe_unique
                    raw_classes = _safe_unique(target_series)
                    target_encoder = {str(v): i for i, v in enumerate(raw_classes)}
                y = np.asarray(
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
                y = np.where(np.isfinite(arr), arr, mean_y)

        except Exception as exc:
            self._info_label.setText(f"Encoding error: {exc}")
            return

        param_values = self._get_param_values()
        if not param_values:
            self._info_label.setText("No valid parameter values to sweep.")
            return

        scoring = "roc_auc_ovr_weighted" if is_clf else "r2"
        score_label = "AUC" if is_clf else "R²"
        n_folds = self._cv_spin.value()

        scores_mean: list[float] = []
        scores_std: list[float] = []
        valid_values: list[int] = []

        self._info_label.setText(f"Running {len(param_values)} evaluations…")

        for val in param_values:
            try:
                est = clone(estimator)
                est.set_params(**{param_name: val})

                if is_clf and len(np.unique(y)) > 1:
                    min_class = int(np.bincount(y).min())
                    k = min(n_folds, min_class) if min_class >= 2 else 2
                    cv = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
                else:
                    cv = KFold(n_splits=min(n_folds, len(y)), shuffle=True, random_state=42)

                cv_scores = cross_val_score(est, X, y, cv=cv, scoring=scoring)
                scores_mean.append(float(np.mean(cv_scores)))
                scores_std.append(float(np.std(cv_scores)))
                valid_values.append(val)
            except Exception:
                continue

        if not valid_values:
            self._info_label.setText("All parameter values failed during evaluation.")
            return

        # Plot results
        self._plot(valid_values, scores_mean, scores_std, param_label, score_label)

        best_idx = int(np.argmax(scores_mean))
        best_val = valid_values[best_idx]
        best_score = scores_mean[best_idx]
        self._info_label.setText(
            f"Best {score_label}: {best_score:.4f} at {param_name}={best_val}  "
            f"({len(valid_values)} values evaluated, {n_folds}-fold CV)"
        )

    def _plot(
        self,
        values: list[int],
        means: list[float],
        stds: list[float],
        param_label: str,
        score_label: str,
    ) -> None:
        if self._figure is None or self._canvas is None:
            return
        self._figure.clf()
        ax = self._figure.add_subplot(111)

        means_arr = np.asarray(means)
        stds_arr = np.asarray(stds)

        ax.plot(values, means, "o-", color="#2196F3", lw=2, markersize=5, label=f"Mean {score_label}")
        ax.fill_between(
            values,
            means_arr - stds_arr,
            means_arr + stds_arr,
            alpha=0.15,
            color="#2196F3",
            label="± 1 std",
        )

        best_idx = int(np.argmax(means_arr))
        ax.axvline(x=values[best_idx], color="#4CAF50", linestyle="--", lw=1, alpha=0.7)
        ax.scatter([values[best_idx]], [means[best_idx]], color="#4CAF50", s=100, zorder=5, label=f"Best ({values[best_idx]})")

        ax.set_xlabel(param_label)
        ax.set_ylabel(score_label)
        ax.set_title(f"Parameter Fitter: {score_label} vs {param_label}")
        ax.legend(loc="best", fontsize=8)
        ax.grid(True, alpha=0.3)

        self._figure.tight_layout()
        self._canvas.draw_idle()

    def _save_image(self) -> None:
        if self._figure is None:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Parameter Fitter Plot", "parameter_fitter.png",
            "PNG Image (*.png);;All Files (*)",
        )
        if path:
            self._figure.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
            self._info_label.setText(f"Saved: {path.split('/')[-1]}")

    def _clear_figure(self) -> None:
        if self._figure is not None and self._canvas is not None:
            self._figure.clf()
            self._canvas.draw_idle()
