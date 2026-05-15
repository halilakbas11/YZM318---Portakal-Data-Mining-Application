from __future__ import annotations

import numpy as np
from sklearn.calibration import calibration_curve

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QComboBox,
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

class CalibrationPlotScreen(QWidget, WorkflowNodeScreenSupport):
    """Calibration Plot — Show probability calibration plot."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._init_workflow_node_support()

        self._dataset: DatasetHandle | None = None
        self._model: object | None = None
        self._svc = SklearnLearnerService()

        self._y_true: np.ndarray | None = None
        self._y_prob: np.ndarray | None = None
        self._class_values: tuple[str, ...] = ()

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

        left_layout.addWidget(QLabel("Target Class:"))
        self._class_combo = QComboBox()
        self._class_combo.currentIndexChanged.connect(self._redraw)
        left_layout.addWidget(self._class_combo)

        left_layout.addWidget(QLabel("Bins:"))
        self._bins_spin = QSpinBox()
        self._bins_spin.setRange(3, 50)
        self._bins_spin.setValue(10)
        self._bins_spin.valueChanged.connect(self._redraw)
        left_layout.addWidget(self._bins_spin)

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
        self._y_true = None
        self._y_prob = None
        self._class_values = ()
        self._class_combo.clear()

        if self._model is None or self._dataset is None:
            self._info_label.setText("Waiting for Data and Model inputs.")
            self._clear_figure()
            return

        artifact = self._model
        is_clf = getattr(artifact, "is_classifier", None)
        if is_clf is None:
            is_clf = getattr(artifact, "kind", "") == "classification"
        if not is_clf:
            self._info_label.setText("Calibration Plot only supports classifiers.")
            self._clear_figure()
            return

        target_name = getattr(artifact, "target_name", None)
        class_values = getattr(artifact, "class_values", ()) or ()
        if not class_values or target_name not in self._dataset.dataframe.columns:
            self._info_label.setText("Model is missing class values or target column not in dataset.")
            self._clear_figure()
            return

        try:
            feature_names = getattr(artifact, "feature_names", ()) or ()
            cat_encoders = getattr(artifact, "categorical_encoders", {}) or {}
            num_cols = getattr(artifact, "numeric_cols", ()) or ()
            numeric_means = getattr(artifact, "numeric_means", {}) or {}

            X = self._svc.encode_X(self._dataset, feature_names, cat_encoders, num_cols, numeric_means)
            target_encoder = getattr(artifact, "target_encoder", {}) or {}
            target_series = self._dataset.dataframe.get_column(target_name)
            
            y_actual = np.asarray([target_encoder.get(str(v), 0) for v in target_series.to_list()], dtype=int)
            y_prob = artifact.predict_proba(X)
            
            if y_prob is None:
                self._info_label.setText("Model does not support probabilities.")
                self._clear_figure()
                return
                
            self._y_true = y_actual
            self._y_prob = y_prob
            self._class_values = class_values
            
            self._class_combo.addItems(list(class_values))
            if len(class_values) > 1:
                self._class_combo.setCurrentIndex(1)
                
            self._info_label.setText(f"Computed Calibration Plot for {len(y_actual)} instances.")
            self._redraw()
        except Exception as exc:
            self._info_label.setText(f"Error: {exc}")
            self._clear_figure()

    def _clear_figure(self) -> None:
        if self._figure is not None and self._canvas is not None:
            self._figure.clf()
            self._canvas.draw_idle()

    def _save_image(self) -> None:
        if self._figure is None or self._y_true is None:
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save Calibration Plot", "calibration_curve.png", "PNG Image (*.png);;All Files (*)")
        if path:
            self._figure.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
            self._info_label.setText(f"Saved: {path.split('/')[-1]}")

    def _redraw(self) -> None:
        if self._y_true is None or self._y_prob is None or self._figure is None or self._canvas is None:
            return
        
        target_idx = self._class_combo.currentIndex()
        if target_idx < 0 or target_idx >= len(self._class_values):
            return

        try:
            self._figure.clf()
            ax = self._figure.add_subplot(111)
            
            y_true_binary = (self._y_true == target_idx).astype(int)
            y_score = self._y_prob[:, target_idx]
            
            bins = self._bins_spin.value()
            fraction_of_positives, mean_predicted_value = calibration_curve(
                y_true_binary, y_score, n_bins=bins, strategy='uniform'
            )

            ax.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
            ax.plot(mean_predicted_value, fraction_of_positives, "s-", color="#E91E63",
                    label=f"Class: {self._class_values[target_idx]}")
            
            ax.set_xlim([0.0, 1.0])
            ax.set_ylim([0.0, 1.05])
            ax.set_xlabel("Mean predicted probability")
            ax.set_ylabel("Fraction of positives")
            ax.set_title(f"Calibration Curve (Class: {self._class_values[target_idx]})")
            ax.legend(loc="lower right")
            ax.grid(True, alpha=0.3)
            
            self._figure.tight_layout()
            self._canvas.draw_idle()
        except Exception as exc:
            self._info_label.setText(f"Plot error: {exc}")
