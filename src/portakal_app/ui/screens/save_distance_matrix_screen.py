from __future__ import annotations

from pathlib import Path

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from portakal_app.data.services.distance_matrix_service import coerce_distance_matrix
from portakal_app.ui import i18n
from portakal_app.ui.screens.node_screen import WorkflowNodeScreenSupport


class SaveDistanceMatrixScreen(QWidget, WorkflowNodeScreenSupport):
    FORMAT_LABELS = ["CSV (.csv)", "Tab-delimited (.txt)", "NumPy Binary (.npy)"]
    FORMAT_EXTS = [".csv", ".txt", ".npy"]
    FORMAT_DELIMS = [",", "\t", None]

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._init_workflow_node_support()
        self._distances: np.ndarray | None = None
        self._row_labels: list[str] = []
        self._last_dir = str(Path.home())
        self._build_ui()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(10)

        format_row = QHBoxLayout()
        format_row.addWidget(QLabel(i18n.t("Format")))
        self._format_combo = QComboBox()
        self._format_combo.addItems(self.FORMAT_LABELS)
        format_row.addWidget(self._format_combo)
        format_row.addStretch(1)
        root.addLayout(format_row)

        self._header_check = QCheckBox("Include row/column indices")
        self._header_check.setChecked(True)
        root.addWidget(self._header_check)

        self._status_label = QLabel("Distance matrix input is waiting.")
        self._status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._status_label.setWordWrap(True)
        root.addWidget(self._status_label)
        root.addStretch(1)

        path_row = QHBoxLayout()
        self._path_edit = QLineEdit()
        self._path_edit.setReadOnly(True)
        self._path_edit.setPlaceholderText("Select an output file...")
        path_row.addWidget(self._path_edit, 1)

        browse_button = QPushButton("Browse...")
        browse_button.clicked.connect(self._browse)
        path_row.addWidget(browse_button)
        root.addLayout(path_row)

        self._save_button = QPushButton(i18n.t("Save"))
        self._save_button.setProperty("primary", True)
        self._save_button.clicked.connect(self._save)
        self._save_button.setEnabled(False)
        root.addWidget(self._save_button)

    def set_input_payload(self, payload) -> None:
        if payload is None:
            self._distances = None
            self._status_label.setText("Distance matrix input is waiting.")
            self._save_button.setEnabled(False)
            return

        try:
            handle = coerce_distance_matrix(payload.value)
            self._distances = np.array(handle.matrix, dtype=float)
            self._row_labels = list(handle.row_labels)
            self._status_label.setText(
                f"{self._distances.shape[0]}x{self._distances.shape[1]} distance matrix received."
            )
            self._save_button.setEnabled(True)
        except Exception as exc:
            self._distances = None
            self._status_label.setText(i18n.tf("Error: {error}", error=exc))
            self._save_button.setEnabled(False)

    def help_text(self) -> str:
        return "Save an incoming distance matrix as CSV, tab-delimited text, or NPY."

    def documentation_url(self) -> str:
        return "https://orangedatamining.com/widget-catalog/unsupervised/savedistancematrix/"

    def _browse(self) -> None:
        index = self._format_combo.currentIndex()
        ext = self.FORMAT_EXTS[index]
        path, _selected = QFileDialog.getSaveFileName(
            self,
            i18n.t("Save As ..."),
            self._last_dir,
            f"{self.FORMAT_LABELS[index]} (*{ext})",
        )
        if not path:
            return
        if not path.endswith(ext):
            path += ext
        self._path_edit.setText(path)
        self._last_dir = str(Path(path).parent)

    def _save(self) -> None:
        if self._distances is None:
            return
        path = self._path_edit.text().strip()
        if not path:
            self._browse()
            path = self._path_edit.text().strip()
            if not path:
                return

        index = self._format_combo.currentIndex()
        try:
            if index == 2:
                np.save(path, self._distances)
            else:
                delimiter = self.FORMAT_DELIMS[index]
                if self._header_check.isChecked():
                    header_labels = self._row_labels or [str(i + 1) for i in range(self._distances.shape[1])]
                    header = delimiter.join([""] + list(header_labels))
                    with Path(path).open("w", encoding="utf-8", newline="") as handle:
                        handle.write(header + "\n")
                        for label, row in zip(header_labels, self._distances.tolist()):
                            handle.write(delimiter.join([str(label)] + [f"{float(cell):.6f}" for cell in row]) + "\n")
                else:
                    np.savetxt(path, self._distances, delimiter=delimiter, fmt="%.6f")
            self._status_label.setText(f"Saved: {Path(path).name}")
        except Exception as exc:
            self._status_label.setText(i18n.tf("Error: {error}", error=exc))
