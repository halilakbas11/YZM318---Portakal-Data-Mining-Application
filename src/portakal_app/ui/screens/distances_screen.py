from __future__ import annotations

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QRadioButton,
    QVBoxLayout,
    QWidget,
)

from portakal_app.data.models import DatasetHandle
from portakal_app.data.services.distance_matrix_service import build_distance_matrix
from portakal_app.models import WorkflowPayload
from portakal_app.ui import i18n
from portakal_app.ui.screens.node_screen import WorkflowNodeScreenSupport

try:
    from scipy.spatial.distance import cdist
    from scipy.stats import rankdata
except ImportError:  # pragma: no cover - dependency check at runtime
    cdist = None
    rankdata = None


class DistancesScreen(QWidget, WorkflowNodeScreenSupport):
    METRICS = [
        ("Euclidean (normalized)", "euclidean", True),
        ("Euclidean", "euclidean", False),
        ("Manhattan (normalized)", "cityblock", True),
        ("Manhattan", "cityblock", False),
        ("Mahalanobis", "mahalanobis", False),
        ("Hamming", "hamming", False),
        ("Cosine", "cosine", False),
        ("Pearson", "pearson", False),
        ("Pearson (absolute)", "pearson_abs", False),
        ("Spearman", "spearman", False),
        ("Spearman (absolute)", "spearman_abs", False),
        ("Jaccard", "jaccard", False),
    ]

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._init_workflow_node_support()
        self._dataset_handle: DatasetHandle | None = None
        self._output_payload: WorkflowPayload | None = None
        self._build_ui()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(10)

        compare_box = QGroupBox(i18n.t("Compare"))
        compare_layout = QHBoxLayout(compare_box)
        self._axis_group = QButtonGroup(self)
        self._rb_rows = QRadioButton(i18n.t("Rows"))
        self._rb_cols = QRadioButton(i18n.t("Columns"))
        self._rb_rows.setChecked(True)
        self._axis_group.addButton(self._rb_rows, 0)
        self._axis_group.addButton(self._rb_cols, 1)
        compare_layout.addWidget(self._rb_rows)
        compare_layout.addWidget(self._rb_cols)
        compare_layout.addStretch(1)
        root.addWidget(compare_box)

        metric_box = QGroupBox(i18n.t("Distance Metric"))
        metric_layout = QHBoxLayout(metric_box)
        left_col = QVBoxLayout()
        right_col = QVBoxLayout()
        self._metric_group = QButtonGroup(self)

        for index, (label, _metric, _norm) in enumerate(self.METRICS[:6]):
            radio = QRadioButton(i18n.t(label))
            if index == 0:
                radio.setChecked(True)
            self._metric_group.addButton(radio, index)
            left_col.addWidget(radio)
            radio.toggled.connect(self._schedule_apply)

        for index, (label, _metric, _norm) in enumerate(self.METRICS[6:], start=6):
            radio = QRadioButton(i18n.t(label))
            self._metric_group.addButton(radio, index)
            right_col.addWidget(radio)
            radio.toggled.connect(self._schedule_apply)

        metric_layout.addLayout(left_col)
        metric_layout.addLayout(right_col)
        root.addWidget(metric_box)

        self._status_label = QLabel(i18n.t("No dataset loaded."))
        self._status_label.setWordWrap(True)
        self._status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        root.addWidget(self._status_label)
        root.addStretch(1)

        footer = QHBoxLayout()
        self.cb_apply_auto = QCheckBox(i18n.t("Apply Automatically"))
        self.cb_apply_auto.setChecked(True)
        self.cb_apply_auto.toggled.connect(lambda _checked: self._schedule_apply())
        footer.addWidget(self.cb_apply_auto)
        footer.addStretch(1)

        self._apply_button = QPushButton(i18n.t("Apply"))
        self._apply_button.setProperty("primary", True)
        self._apply_button.clicked.connect(self._apply)
        footer.addWidget(self._apply_button)
        root.addLayout(footer)

        self._rb_rows.toggled.connect(self._schedule_apply)
        self._rb_cols.toggled.connect(self._schedule_apply)

    def set_input_payload(self, payload) -> None:
        self._dataset_handle = payload.dataset if payload is not None else None
        self._output_payload = None
        self._apply()

    def current_output_payload(self) -> WorkflowPayload | None:
        return self._output_payload

    def help_text(self) -> str:
        return "Compute a distance matrix from numeric features using Orange-style distance metrics."

    def documentation_url(self) -> str:
        return "https://orangedatamining.com/widget-catalog/unsupervised/distances/"

    def _schedule_apply(self) -> None:
        if self.cb_apply_auto.isChecked():
            self._apply()

    def _apply(self) -> None:
        self._output_payload = None
        if self._dataset_handle is None:
            self._status_label.setText(i18n.t("No dataset loaded."))
            self._notify_output_changed()
            return
        if cdist is None:
            self._status_label.setText("SciPy is required for Distances.")
            self._notify_output_changed()
            return

        try:
            feature_cols = [
                column.name
                for column in self._dataset_handle.domain.feature_columns
                if column.logical_type == "numeric"
            ]
            if not feature_cols:
                self._status_label.setText(i18n.t("No numeric features found."))
                self._notify_output_changed()
                return

            matrix = self._dataset_handle.dataframe.select(feature_cols).to_numpy(allow_copy=True).astype(float)
            if matrix.size == 0:
                self._status_label.setText(i18n.t("No numeric features found."))
                self._notify_output_changed()
                return

            nan_mask = np.isnan(matrix)
            if nan_mask.any():
                col_means = np.nanmean(matrix, axis=0)
                col_means = np.where(np.isnan(col_means), 0.0, col_means)
                matrix[nan_mask] = np.take(col_means, np.where(nan_mask)[1])

            metric_index = self._metric_group.checkedId()
            _label, metric_name, normalize = self.METRICS[metric_index]
            if normalize:
                col_min = matrix.min(axis=0)
                col_max = matrix.max(axis=0)
                denom = col_max - col_min
                denom[denom == 0] = 1.0
                matrix = (matrix - col_min) / denom

            if self._rb_cols.isChecked():
                matrix = matrix.T

            if metric_name == "pearson":
                distances = self._pearson_dist(matrix, absolute=False)
            elif metric_name == "pearson_abs":
                distances = self._pearson_dist(matrix, absolute=True)
            elif metric_name == "spearman":
                distances = self._spearman_dist(matrix, absolute=False)
            elif metric_name == "spearman_abs":
                distances = self._spearman_dist(matrix, absolute=True)
            else:
                kwargs = {}
                if metric_name == "mahalanobis":
                    cov = np.cov(matrix, rowvar=False)
                    inv_cov = np.linalg.pinv(cov)
                    kwargs["VI"] = inv_cov
                distances = cdist(matrix, matrix, metric=metric_name, **kwargs)

            result = build_distance_matrix(
                distances,
                metric=metric_name,
                metric_label=self.METRICS[metric_index][0],
                axis="columns" if self._rb_cols.isChecked() else "rows",
                axis_label=i18n.t("Distances between columns") if self._rb_cols.isChecked() else i18n.t("Distances between rows"),
                row_labels=tuple(
                    feature_cols if self._rb_cols.isChecked() else [str(index + 1) for index in range(distances.shape[0])]
                ),
                feature_names=tuple(feature_cols),
                source_dataset=None if self._rb_cols.isChecked() else self._dataset_handle,
                metadata={"normalized": normalize},
            )
            self._output_payload = WorkflowPayload("Distances", result)
            self._status_label.setText(
                f"{distances.shape[0]}x{distances.shape[1]} | {self.METRICS[metric_index][0]}"
            )
        except Exception as exc:
            self._status_label.setText(i18n.tf("Error: {error}", error=exc))
            self._output_payload = None

        self._notify_output_changed()

    @staticmethod
    def _pearson_dist(matrix: np.ndarray, *, absolute: bool) -> np.ndarray:
        corr = np.nan_to_num(np.corrcoef(matrix), nan=0.0)
        corr = np.clip(corr, -1.0, 1.0)
        dist = 1.0 - (np.abs(corr) if absolute else corr)
        np.fill_diagonal(dist, 0.0)
        return dist

    @staticmethod
    def _spearman_dist(matrix: np.ndarray, *, absolute: bool) -> np.ndarray:
        if rankdata is None:
            raise RuntimeError("SciPy is required for Spearman distance.")
        ranked = np.apply_along_axis(rankdata, 1, matrix)
        corr = np.nan_to_num(np.corrcoef(ranked), nan=0.0)
        corr = np.clip(corr, -1.0, 1.0)
        dist = 1.0 - (np.abs(corr) if absolute else corr)
        np.fill_diagonal(dist, 0.0)
        return dist
