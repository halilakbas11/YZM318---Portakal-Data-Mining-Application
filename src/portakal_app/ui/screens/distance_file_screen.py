from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from portakal_app.data.services.distance_file_service import DistanceFileService
from portakal_app.models import WorkflowPayload
from portakal_app.ui import i18n
from portakal_app.ui.screens.node_screen import WorkflowNodeScreenSupport


class DistanceFileScreen(QWidget, WorkflowNodeScreenSupport):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._init_workflow_node_support()
        self._service = DistanceFileService()
        self._path: str = ""
        self._output_payload: WorkflowPayload | None = None
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(10)

        self._title = QLabel(i18n.t("Distance File"))
        self._title.setProperty("sectionTitle", True)
        layout.addWidget(self._title)

        self._path_edit = QLineEdit(self)
        self._path_edit.setReadOnly(True)
        self._path_edit.setPlaceholderText(i18n.t("Select a distance matrix file..."))
        layout.addWidget(self._path_edit)

        row = QHBoxLayout()
        self._browse_button = QPushButton(i18n.t("Browse..."))
        self._browse_button.clicked.connect(self._browse)
        row.addWidget(self._browse_button)

        self._reload_button = QPushButton(i18n.t("Reload"))
        self._reload_button.clicked.connect(self._load_current_path)
        row.addWidget(self._reload_button)
        row.addStretch(1)
        layout.addLayout(row)

        self._symmetric_check = QCheckBox(i18n.t("Treat triangular matrices as symmetric"))
        self._symmetric_check.setChecked(True)
        self._symmetric_check.toggled.connect(self._load_current_path)
        layout.addWidget(self._symmetric_check)

        self._status_label = QLabel(i18n.t("No distance file selected."))
        self._status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._status_label.setWordWrap(True)
        layout.addWidget(self._status_label)
        layout.addStretch(1)

    def current_output_payload(self) -> WorkflowPayload | None:
        return self._output_payload

    def serialize_node_state(self) -> dict[str, object]:
        return {
            "path": self._path,
            "symmetric": self._symmetric_check.isChecked(),
        }

    def restore_node_state(self, payload: dict[str, object]) -> None:
        self._path = str(payload.get("path", "") or "")
        self._path_edit.setText(self._path)
        self._symmetric_check.setChecked(bool(payload.get("symmetric", True)))
        self._load_current_path()

    def help_text(self) -> str:
        return "Load a saved distance matrix from CSV, TXT, TSV, or XLSX."

    def documentation_url(self) -> str:
        return "https://orangedatamining.com/widget-catalog/unsupervised/distancefile/"

    def _browse(self) -> None:
        start_dir = self._path or str(Path.home())
        path, _selected = QFileDialog.getOpenFileName(
            self,
            i18n.t("Open Distance Matrix"),
            start_dir,
            "Distance Files (*.csv *.tsv *.txt *.xlsx);;All Files (*.*)",
        )
        if not path:
            return
        self._path = path
        self._path_edit.setText(path)
        self._load_current_path()

    def _load_current_path(self) -> None:
        self._output_payload = None
        if not self._path:
            self._status_label.setText(i18n.t("No distance file selected."))
            self._notify_output_changed()
            return
        try:
            handle = self._service.load(
                self._path,
                treat_triangular_as_symmetric=self._symmetric_check.isChecked(),
            )
            self._output_payload = WorkflowPayload("Distances", handle)
            self._status_label.setText(
                i18n.tf(
                    "{name}: {rows} x {cols}",
                    name=Path(self._path).name,
                    rows=handle.matrix.shape[0],
                    cols=handle.matrix.shape[1],
                )
            )
        except Exception as exc:
            self._status_label.setText(i18n.tf("Error: {err}", err=exc))
        self._notify_output_changed()

