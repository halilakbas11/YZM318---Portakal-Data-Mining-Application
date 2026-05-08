from __future__ import annotations

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
    QHeaderView,
)

from portakal_app.data.services.distance_matrix_service import coerce_distance_matrix
from portakal_app.models import WorkflowPayload
from portakal_app.ui import i18n
from portakal_app.ui.screens.node_screen import WorkflowNodeScreenSupport


class DistanceMatrixScreen(QWidget, WorkflowNodeScreenSupport):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._init_workflow_node_support()
        self._distances: np.ndarray | None = None
        self._distance_value = None
        self._row_labels: list[str] = []
        self._output_payload: WorkflowPayload | None = None
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(10)

        top = QHBoxLayout()
        top.addWidget(QLabel(i18n.t("Distance Matrix")))
        top.addStretch(1)
        top.addWidget(QLabel(i18n.t("Labels")))
        self._label_combo = QComboBox()
        self._label_combo.addItem(i18n.t("Row number"))
        self._label_combo.currentIndexChanged.connect(self._refresh_labels)
        top.addWidget(self._label_combo)
        layout.addLayout(top)

        self._info_label = QLabel("Distances input is waiting.")
        self._info_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._info_label.setWordWrap(True)
        layout.addWidget(self._info_label)

        self._table = QTableWidget()
        self._table.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        layout.addWidget(self._table, 1)

        footer = QHBoxLayout()
        self._auto_check = QCheckBox(i18n.t("Send Automatically"))
        self._auto_check.setChecked(True)
        self._auto_check.toggled.connect(self._on_auto_toggled)
        footer.addWidget(self._auto_check)

        self._send_button = QPushButton(i18n.t("Send"))
        self._send_button.clicked.connect(self._send)
        footer.addWidget(self._send_button)
        footer.addStretch(1)
        layout.addLayout(footer)

        self._on_auto_toggled(True)

    def set_input_payload(self, payload) -> None:
        self._output_payload = None
        if payload is None:
            self._distances = None
            self._distance_value = None
            self._row_labels = []
            self._clear_table()
            self._info_label.setText("Distances input is waiting.")
            self._notify_output_changed()
            return

        try:
            handle = coerce_distance_matrix(payload.value)
            self._distance_value = handle
            self._distances = np.array(handle.matrix, dtype=float)
            self._row_labels = list(handle.row_labels)
            self._show_matrix()
        except Exception as exc:
            self._distances = None
            self._distance_value = None
            self._row_labels = []
            self._clear_table()
            self._info_label.setText(i18n.tf("Error: {error}", error=exc))
            self._notify_output_changed()

    def current_output_payload(self) -> WorkflowPayload | None:
        return self._output_payload

    def help_text(self) -> str:
        return "Inspect an incoming distance matrix and pass it to downstream distance widgets."

    def documentation_url(self) -> str:
        return "https://orangedatamining.com/widget-catalog/unsupervised/distancematrix/"

    def _on_auto_toggled(self, checked: bool) -> None:
        self._send_button.setEnabled(not checked)
        if checked and self._distances is not None:
            self._send()

    def _send(self) -> None:
        if self._distances is None:
            self._output_payload = None
        else:
            self._output_payload = WorkflowPayload("Distances", self._distance_value)
        self._notify_output_changed()

    def _clear_table(self) -> None:
        self._table.clear()
        self._table.setRowCount(0)
        self._table.setColumnCount(0)

    def _refresh_labels(self) -> None:
        if self._distances is None:
            return
        labels = self._row_labels or [str(i + 1) for i in range(self._distances.shape[0])]
        self._table.setHorizontalHeaderLabels(labels)
        self._table.setVerticalHeaderLabels(labels)

    def _show_matrix(self) -> None:
        if self._distances is None:
            return
        matrix = self._distances
        rows = min(matrix.shape[0], 80)
        cols = min(matrix.shape[1], 80)
        labels = (self._row_labels or [str(i + 1) for i in range(matrix.shape[0])])[:rows]
        max_val = float(matrix.max()) if matrix.size and matrix.max() > 0 else 1.0

        self._table.clear()
        self._table.setRowCount(rows)
        self._table.setColumnCount(cols)
        self._table.setHorizontalHeaderLabels(labels[:cols])
        self._table.setVerticalHeaderLabels(labels[:rows])

        for row in range(rows):
            for col in range(cols):
                val = float(matrix[row, col])
                item = QTableWidgetItem(f"{val:.3f}")
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                ratio = val / max_val if max_val else 0.0
                red = 200 if ratio >= 0.5 else int(255 * ratio * 2)
                green = int(200 * (1 - max(0.0, ratio - 0.5) * 2)) if ratio >= 0.5 else 200
                item.setBackground(QColor(red, green, 80))
                self._table.setItem(row, col, item)

        if rows <= 30:
            self._table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
            self._table.verticalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        else:
            self._table.horizontalHeader().setDefaultSectionSize(50)
            self._table.verticalHeader().setDefaultSectionSize(25)

        suffix = f" (first {rows})" if matrix.shape[0] > rows else ""
        self._info_label.setText(
            f"{matrix.shape[0]}x{matrix.shape[1]} matrix{suffix} | min={matrix.min():.4f} max={matrix.max():.4f}"
        )
        self._send()
