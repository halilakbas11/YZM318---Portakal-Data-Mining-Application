from __future__ import annotations

import numpy as np
from sklearn.inspection import permutation_importance
from sklearn.metrics import make_scorer, accuracy_score, mean_squared_error

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
    QSpinBox,
)

from portakal_app.data.models import DatasetHandle
from portakal_app.data.services.sklearn_learner_service import SklearnLearnerService
from portakal_app.ui.screens.node_screen import WorkflowNodeScreenSupport

try:
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.figure import Figure
except ImportError:
    FigureCanvas = None
    Figure = None

class PermutationPlotScreen(QWidget, WorkflowNodeScreenSupport):
    """Permutation Plot — Feature importance via permutations."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._init_workflow_node_support()

        self._dataset: DatasetHandle | None = None
        self._model: object | None = None
        self._svc = SklearnLearnerService()

        self._importances: dict | None = None
        self._feature_names: list[str] = []

        self._figure = Figure(figsize=(6, 5), facecolor="#f8f8f8") if Figure is not None else None
        self._canvas = FigureCanvas(self._figure) if FigureCanvas is not None and self._figure is not None else None

        self._build_ui()

    def _build_ui(self) -> None:
        root = QHBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)

        left_panel = QWidget()
        left_panel.setFixedWidth(220)
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(12, 12, 8, 12)

        left_layout.addWidget(QLabel("Repeats:"))
        self._repeats_spin = QSpinBox()
        self._repeats_spin.setRange(1, 50)
        self._repeats_spin.setValue(5)
        self._repeats_spin.valueChanged.connect(self._compute)
        left_layout.addWidget(self._repeats_spin)

        left_layout.addStretch(1)

        save_button = QPushButton("Save Image")
        save_button.setProperty("primary", True)
        save_button.clicked.connect(self._save_image)
        left_layout.addWidget(save_button)
        root.addWidget(left_panel)

        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        if self._canvas is not None:
            right_layout.addWidget(self._canvas, 1)
        self._info_label = QLabel("Waiting for Data and Model inputs.")
        self._info_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        right_layout.addWidget(self._info_label)
        root.addWidget(right_panel, 1)

    def set_input_payload(self, payload) -> None:
        if payload is None:
            self._dataset = None
            self._model = None
        elif payload.port_label == "Data":
            self._dataset = payload.value if isinstance(payload.value, DatasetHandle) else None
        elif payload.port_label == "Model":
            self._model = payload.value
        self._compute()

    def _compute(self) -> None:
        self._importances = None
        self._feature_names = []

        if self._model is None or self._dataset is None:
            self._info_label.setText("Waiting for Data and Model inputs.")
            self._clear_figure()
            return

        artifact = self._model
        target_name = getattr(artifact, "target_name", None)
        
        if not target_name or target_name not in self._dataset.dataframe.columns:
            self._info_label.setText("Target column not found in dataset.")
            self._clear_figure()
            return
            
        estimator = getattr(artifact, "trained_model", None)
        if estimator is None:
            self._info_label.setText("Model must be trained before computing permutation importance.")
            self._clear_figure()
            return

        try:
            feature_names = getattr(artifact, "feature_names", ()) or ()
            cat_encoders = getattr(artifact, "categorical_encoders", {}) or {}
            num_cols = getattr(artifact, "numeric_cols", ()) or ()
            numeric_means = getattr(artifact, "numeric_means", {}) or {}

            X = self._svc.encode_X(self._dataset, feature_names, cat_encoders, num_cols, numeric_means)
            
            is_clf = getattr(artifact, "is_classifier", None)
            if is_clf is None:
                is_clf = getattr(artifact, "kind", "") == "classification"
            
            target_series = self._dataset.dataframe.get_column(target_name)
            
            if is_clf:
                target_encoder = getattr(artifact, "target_encoder", {}) or {}
                y = np.asarray([target_encoder.get(str(v), 0) for v in target_series.to_list()], dtype=int)
                scoring = make_scorer(accuracy_score)
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
                scoring = make_scorer(mean_squared_error, greater_is_better=False)
            
            repeats = self._repeats_spin.value()
            result = permutation_importance(
                estimator, X, y, scoring=scoring, n_repeats=repeats, random_state=42, n_jobs=1
            )
            
            self._importances = result
            self._feature_names = list(feature_names)
                
            self._info_label.setText(f"Computed importances for {len(feature_names)} features.")
            self._redraw()
        except Exception as exc:
            self._info_label.setText(f"Error: {exc}")
            self._clear_figure()

    def _clear_figure(self) -> None:
        if self._figure is not None and self._canvas is not None:
            self._figure.clf()
            self._canvas.draw_idle()

    def _save_image(self) -> None:
        if self._figure is None or self._importances is None:
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save Permutation Plot", "permutation_plot.png", "PNG Image (*.png);;All Files (*)")
        if path:
            self._figure.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
            self._info_label.setText(f"Saved: {path.split('/')[-1]}")

    def _redraw(self) -> None:
        if self._importances is None or self._figure is None or self._canvas is None:
            return
        
        try:
            self._figure.clf()
            ax = self._figure.add_subplot(111)
            
            result = self._importances
            sorted_idx = result.importances_mean.argsort()

            ax.boxplot(
                result.importances[sorted_idx].T,
                vert=False
            )
            ax.set_yticks(range(1, len(self._feature_names) + 1))
            ax.set_yticklabels(np.array(self._feature_names)[sorted_idx])
            
            ax.set_title("Permutation Importances")
            ax.grid(True, alpha=0.3, axis='x')
            
            self._figure.tight_layout()
            self._canvas.draw_idle()
        except Exception as exc:
            self._info_label.setText(f"Plot error: {exc}")
