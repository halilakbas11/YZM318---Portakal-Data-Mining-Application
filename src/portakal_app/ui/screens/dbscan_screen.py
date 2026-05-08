from __future__ import annotations

import uuid

import numpy as np
import polars as pl
from PySide6.QtCore import QPointF, Qt
from PySide6.QtGui import QColor, QLinearGradient, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QToolTip,
    QVBoxLayout,
    QWidget,
)

from portakal_app.data.models import DatasetHandle
from portakal_app.data.services.generated_dataset_service import GeneratedDatasetService
from portakal_app.data.services.sklearn_learner_service import SklearnLearnerService
from portakal_app.models import WorkflowPayload
from portakal_app.ui import i18n
from portakal_app.ui.screens.node_screen import WorkflowNodeScreenSupport
from portakal_app.ui.screens.visualize_common import nice_ticks


class _KDistanceCanvas(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._distances: np.ndarray | None = None
        self._eps: float = 0.5
        self._hovered_index: int | None = None
        self._elbow_index: int | None = None
        self.setMinimumHeight(260)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMouseTracking(True)

    def set_data(self, distances: np.ndarray, eps: float) -> None:
        self._distances = np.asarray(distances, dtype=float)
        self._eps = float(eps)
        self._hovered_index = None
        self._elbow_index = self._find_elbow(self._distances) if len(self._distances) > 2 else None
        self.update()

    def clear(self) -> None:
        self._distances = None
        self._hovered_index = None
        self._elbow_index = None
        self.update()

    @staticmethod
    def _find_elbow(distances: np.ndarray) -> int | None:
        count = len(distances)
        if count < 3:
            return None
        start = np.array([0.0, float(distances[0])])
        end = np.array([float(count - 1), float(distances[-1])])
        baseline = end - start
        length = float(np.linalg.norm(baseline))
        if length < 1e-9:
            return None
        baseline /= length
        best_index = None
        best_distance = -1.0
        for index in range(1, count - 1):
            point = np.array([float(index), float(distances[index])])
            projection = start + np.dot(point - start, baseline) * baseline
            distance = float(np.linalg.norm(point - projection))
            if distance > best_distance:
                best_distance = distance
                best_index = index
        return best_index

    def _plot_rect(self) -> tuple[int, int, int, int]:
        left, right, top, bottom = 72, 24, 28, 56
        return left, top, max(80, self.width() - left - right), max(60, self.height() - top - bottom)

    def _to_pixel(self, index: int, value: float, count: int, y_min: float, y_max: float) -> tuple[float, float]:
        left, top, width, height = self._plot_rect()
        x = left + (index / max(count - 1, 1)) * width
        span = y_max - y_min if abs(y_max - y_min) > 1e-9 else 1.0
        y = top + height - ((value - y_min) / span) * height
        return x, y

    def paintEvent(self, _event) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), QColor("#fffdf9"))
        if self._distances is None or len(self._distances) == 0:
            painter.setPen(QColor("#8d877d"))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "No k-distance data available.\nConnect data and press Apply.")
            painter.end()
            return

        distances = self._distances
        count = len(distances)
        left, top, width, height = self._plot_rect()
        y_min = 0.0
        y_max = max(float(np.max(distances)) * 1.08, 1.0)
        fm = painter.fontMetrics()

        painter.setPen(QPen(QColor("#bbb3a8"), 1))
        painter.drawLine(int(left), int(top), int(left), int(top + height))
        painter.drawLine(int(left), int(top + height), int(left + width), int(top + height))

        painter.setPen(QPen(QColor("#ece6dd"), 1, Qt.PenStyle.DotLine))
        for tick in nice_ticks(y_min, y_max, 6):
            _, tick_y = self._to_pixel(0, tick, count, y_min, y_max)
            painter.drawLine(int(left + 1), int(tick_y), int(left + width), int(tick_y))
        painter.setPen(QColor("#5f5649"))
        for tick in nice_ticks(y_min, y_max, 6):
            _, tick_y = self._to_pixel(0, tick, count, y_min, y_max)
            label = f"{tick:.2g}"
            painter.drawText(int(left - fm.horizontalAdvance(label) - 6), int(tick_y + 4), label)

        x_step = max(1, count // min(6, count))
        for index in range(0, count, x_step):
            pixel_x, _ = self._to_pixel(index, 0.0, count, y_min, y_max)
            label = str(index)
            painter.drawText(int(pixel_x - fm.horizontalAdvance(label) / 2), int(top + height + 16), label)
        if count > 1:
            pixel_x, _ = self._to_pixel(count - 1, 0.0, count, y_min, y_max)
            label = str(count - 1)
            painter.drawText(int(pixel_x - fm.horizontalAdvance(label) / 2), int(top + height + 16), label)

        gradient_path = QPainterPath()
        start_x, start_y = self._to_pixel(0, float(distances[0]), count, y_min, y_max)
        gradient_path.moveTo(start_x, top + height)
        gradient_path.lineTo(start_x, start_y)
        for index in range(1, count):
            pixel_x, pixel_y = self._to_pixel(index, float(distances[index]), count, y_min, y_max)
            gradient_path.lineTo(pixel_x, pixel_y)
        end_x, _ = self._to_pixel(count - 1, 0.0, count, y_min, y_max)
        gradient_path.lineTo(end_x, top + height)
        gradient_path.closeSubpath()

        gradient = QLinearGradient(0, top, 0, top + height)
        gradient.setColorAt(0.0, QColor(224, 112, 32, 60))
        gradient.setColorAt(1.0, QColor(224, 112, 32, 8))
        painter.setBrush(gradient)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawPath(gradient_path)

        curve = QPainterPath()
        curve.moveTo(start_x, start_y)
        for index in range(1, count):
            pixel_x, pixel_y = self._to_pixel(index, float(distances[index]), count, y_min, y_max)
            curve.lineTo(pixel_x, pixel_y)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.setPen(QPen(QColor("#e07020"), 2.5))
        painter.drawPath(curve)

        if y_min <= self._eps <= y_max:
            _, eps_y = self._to_pixel(0, self._eps, count, y_min, y_max)
            painter.setPen(QPen(QColor("#111827"), 1.5, Qt.PenStyle.DashLine))
            painter.drawLine(int(left), int(eps_y), int(left + width), int(eps_y))
            label = f"{self._eps:.3f}"
            painter.setPen(QColor("#111827"))
            painter.drawText(int(left + width - fm.horizontalAdvance(label) - 4), int(eps_y - 5), label)

        if self._elbow_index is not None:
            elbow_x, elbow_y = self._to_pixel(self._elbow_index, float(distances[self._elbow_index]), count, y_min, y_max)
            painter.setPen(QPen(QColor("#1f1f1f"), 1.2, Qt.PenStyle.DashDotLine))
            painter.drawLine(int(elbow_x), int(top), int(elbow_x), int(top + height))
            painter.setBrush(QColor("#e07020"))
            painter.setPen(QPen(QColor("#ffffff"), 2))
            painter.drawEllipse(QPointF(elbow_x, elbow_y), 5.0, 5.0)

        if self._hovered_index is not None:
            hover_x, hover_y = self._to_pixel(self._hovered_index, float(distances[self._hovered_index]), count, y_min, y_max)
            painter.setBrush(QColor("#3b82f6"))
            painter.setPen(QPen(QColor("#ffffff"), 2))
            painter.drawEllipse(QPointF(hover_x, hover_y), 4.0, 4.0)

        painter.setPen(QColor("#463c2f"))
        painter.drawText(int(left), int(top + height + 32), int(width), 20, Qt.AlignmentFlag.AlignCenter, "Data items sorted by score")
        painter.save()
        painter.translate(14, int(top + height / 2))
        painter.rotate(-90)
        painter.drawText(int(-height / 2), 0, int(height), 20, Qt.AlignmentFlag.AlignCenter, "Distance to k-th nearest neighbour")
        painter.restore()
        painter.end()

    def mouseMoveEvent(self, event) -> None:  # noqa: N802
        if self._distances is None or len(self._distances) == 0:
            return
        left, _top, width, _height = self._plot_rect()
        ratio = (event.position().x() - left) / max(width, 1)
        index = int(round(ratio * (len(self._distances) - 1)))
        index = max(0, min(index, len(self._distances) - 1))
        if index != self._hovered_index:
            self._hovered_index = index
            self.update()
            QToolTip.showText(
                event.globalPosition().toPoint(),
                f"Index: {index}\nDistance: {float(self._distances[index]):.4f}",
                self,
            )

    def leaveEvent(self, _event) -> None:  # noqa: N802
        self._hovered_index = None
        self.update()
        QToolTip.hideText()


class DBSCANScreen(QWidget, WorkflowNodeScreenSupport):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._init_workflow_node_support()
        self._svc = SklearnLearnerService()
        self._builder = GeneratedDatasetService()
        self._dataset_handle: DatasetHandle | None = None
        self._output_payload: WorkflowPayload | None = None
        self._build_ui()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(10)

        params_box = QGroupBox(i18n.t("Parameters"))
        params_form = QFormLayout(params_box)

        self._neighbors_spin = QSpinBox()
        self._neighbors_spin.setRange(1, 100)
        self._neighbors_spin.setValue(5)
        self._neighbors_spin.setAlignment(Qt.AlignmentFlag.AlignRight)
        params_form.addRow(i18n.t("Core point neighbors:"), self._neighbors_spin)

        self._eps_spin = QDoubleSpinBox()
        self._eps_spin.setRange(0.01, 9999.0)
        self._eps_spin.setDecimals(3)
        self._eps_spin.setSingleStep(0.01)
        self._eps_spin.setValue(0.5)
        self._eps_spin.setAlignment(Qt.AlignmentFlag.AlignRight)
        params_form.addRow(i18n.t("Neighborhood distance:"), self._eps_spin)
        root.addWidget(params_box)

        metric_box = QGroupBox(i18n.t("Distance Metric"))
        metric_layout = QVBoxLayout(metric_box)
        self._metric_combo = QComboBox()
        self._metric_combo.addItem(i18n.t("Euclidean"), "euclidean")
        self._metric_combo.addItem(i18n.t("Manhattan"), "manhattan")
        self._metric_combo.addItem(i18n.t("Cosine"), "cosine")
        metric_layout.addWidget(self._metric_combo)

        self._normalize_check = QCheckBox(i18n.t("Normalize features"))
        self._normalize_check.setChecked(True)
        metric_layout.addWidget(self._normalize_check)
        root.addWidget(metric_box)

        plot_box = QGroupBox(i18n.t("k-Distance Plot"))
        plot_layout = QVBoxLayout(plot_box)
        plot_layout.setContentsMargins(6, 6, 6, 6)
        self._canvas = _KDistanceCanvas()
        plot_layout.addWidget(self._canvas)
        root.addWidget(plot_box, 1)

        self._status_label = QLabel(i18n.t("No dataset loaded."))
        self._status_label.setWordWrap(True)
        root.addWidget(self._status_label)

        footer = QHBoxLayout()
        self.cb_apply_auto = QCheckBox(i18n.t("Apply Automatically"))
        self.cb_apply_auto.setChecked(True)
        footer.addWidget(self.cb_apply_auto)
        footer.addStretch(1)
        self._apply_button = QPushButton(i18n.t("Apply"))
        self._apply_button.setProperty("primary", True)
        self._apply_button.clicked.connect(self._apply)
        footer.addWidget(self._apply_button)
        root.addLayout(footer)

        self._neighbors_spin.valueChanged.connect(lambda _value: self._schedule_auto_apply())
        self._eps_spin.valueChanged.connect(lambda _value: self._schedule_auto_apply())
        self._metric_combo.currentIndexChanged.connect(self._schedule_auto_apply)
        self._normalize_check.toggled.connect(lambda _checked: self._schedule_auto_apply())
        self.cb_apply_auto.toggled.connect(lambda _checked: self._schedule_auto_apply())

    def set_input_payload(self, payload: WorkflowPayload | None) -> None:
        self._dataset_handle = payload.dataset if payload is not None else None
        self._output_payload = None
        self._schedule_auto_apply()

    def current_output_payload(self) -> WorkflowPayload | None:
        return self._output_payload

    def help_text(self) -> str:
        return "Group items with DBSCAN and inspect the sorted distance to the k-th nearest neighbour."

    def documentation_url(self) -> str:
        return "https://orangedatamining.com/widget-catalog/unsupervised/DBSCAN/"

    def serialize_node_state(self) -> dict[str, object]:
        return {
            "neighbors": self._neighbors_spin.value(),
            "eps": self._eps_spin.value(),
            "metric": self._metric_combo.currentData(),
            "normalize": self._normalize_check.isChecked(),
            "auto_apply": self.cb_apply_auto.isChecked(),
        }

    def restore_node_state(self, payload: dict[str, object]) -> None:
        self._neighbors_spin.setValue(int(payload.get("neighbors", 5)))
        self._eps_spin.setValue(float(payload.get("eps", 0.5)))
        metric_index = self._metric_combo.findData(str(payload.get("metric", "euclidean")))
        self._metric_combo.setCurrentIndex(max(0, metric_index))
        self._normalize_check.setChecked(bool(payload.get("normalize", True)))
        self.cb_apply_auto.setChecked(bool(payload.get("auto_apply", True)))

    def _apply(self) -> None:
        self._output_payload = None
        if self._dataset_handle is None:
            self._canvas.clear()
            self._status_label.setText(i18n.t("No dataset loaded."))
            self._notify_output_changed()
            return

        try:
            from sklearn.cluster import DBSCAN
            from sklearn.neighbors import NearestNeighbors
            from sklearn.preprocessing import StandardScaler

            dataset = self._dataset_handle
            X, _feature_names, _encoders, _numeric_cols, _means = self._svc.prepare_features(dataset)
            if X.shape[0] < 2 or X.shape[1] < 1:
                raise ValueError("DBSCAN needs at least two rows and one feature.")

            working = StandardScaler().fit_transform(X) if self._normalize_check.isChecked() else X
            metric = str(self._metric_combo.currentData())
            neighbors = int(self._neighbors_spin.value())
            eps = float(self._eps_spin.value())

            estimator = DBSCAN(eps=eps, min_samples=neighbors, metric=metric)
            labels = estimator.fit_predict(working)
            cluster_count = len({int(value) for value in labels if int(value) >= 0})
            noise_count = int(np.sum(labels < 0))

            nn_count = max(1, min(neighbors, working.shape[0]))
            nn = NearestNeighbors(n_neighbors=nn_count, metric=metric)
            nn.fit(working)
            distances, _ = nn.kneighbors(working)
            k_distances = np.sort(distances[:, -1])[::-1] if distances.size else np.zeros(working.shape[0], dtype=float)
            self._canvas.set_data(k_distances, eps)

            cluster_names = [("Noise" if int(value) < 0 else f"C{int(value) + 1}") for value in labels]
            dataframe = dataset.dataframe.with_columns(
                pl.Series("Cluster", cluster_names),
                pl.Series("Cluster id", [int(value) for value in labels.tolist()]),
                pl.Series("k-distance", [float(value) for value in distances[:, -1].tolist()]),
            )
            role_overrides = {column.name: column.role for column in dataset.domain.columns}
            role_overrides.update({"Cluster": "meta", "Cluster id": "meta", "k-distance": "meta"})
            output = self._builder.build_dataset(
                dataframe,
                dataset_id=f"{dataset.dataset_id}-dbscan-{uuid.uuid4().hex[:8]}",
                display_name=f"{dataset.display_name} (DBSCAN)",
                file_name=f"{dataset.dataset_id}-dbscan.csv",
                role_overrides=role_overrides,
                annotations={
                    **dataset.annotations,
                    "dbscan": {
                        "eps": eps,
                        "min_samples": neighbors,
                        "metric": metric,
                        "normalize": bool(self._normalize_check.isChecked()),
                        "cluster_count": cluster_count,
                        "noise_count": noise_count,
                    },
                },
            )
            self._output_payload = WorkflowPayload("Data", output)
            self._status_label.setText(
                i18n.tf(
                    "{rows} instances, {clusters} clusters, {noise} noise",
                    rows=output.row_count,
                    clusters=cluster_count,
                    noise=noise_count,
                )
            )
        except Exception as exc:
            self._canvas.clear()
            self._status_label.setText(i18n.tf("Error: {err}", err=exc))

        self._notify_output_changed()
