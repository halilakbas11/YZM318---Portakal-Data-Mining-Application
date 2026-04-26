from __future__ import annotations

import pickle
from pathlib import Path

from PySide6.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from portakal_app.models import WorkflowPayload
from portakal_app.ui.screens.node_screen import WorkflowNodeScreenSupport


class LoadModelScreen(QWidget, WorkflowNodeScreenSupport):
    """Load Model — load a pickled model artifact from a .portakal file."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._init_workflow_node_support()
        self._model_artifact = None
        self._load_path: Path | None = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(10)

        title = QLabel("Load Model")
        title.setProperty("sectionTitle", True)
        title.setStyleSheet("background: transparent;")
        layout.addWidget(title)

        self._path_label = QLabel("No file selected.")
        self._path_label.setWordWrap(True)
        self._path_label.setStyleSheet("color: #777; background: transparent;")
        layout.addWidget(self._path_label)

        btn_row = QHBoxLayout()
        self._browse_btn = QPushButton("Browse…")
        self._browse_btn.clicked.connect(self._browse_and_load)
        btn_row.addWidget(self._browse_btn)
        self._reload_btn = QPushButton("Reload")
        self._reload_btn.clicked.connect(self._do_load)
        self._reload_btn.setEnabled(False)
        btn_row.addWidget(self._reload_btn)
        btn_row.addStretch(1)
        layout.addLayout(btn_row)

        self._status_label = QLabel("")
        self._status_label.setWordWrap(True)
        self._status_label.setStyleSheet("background: transparent;")
        layout.addWidget(self._status_label)

        layout.addStretch(1)

    # ── Workflow integration ──────────────────────────────────────────

    def set_input_payload(self, payload: WorkflowPayload | None) -> None:
        pass  # no data input

    def current_output_payload(self) -> WorkflowPayload | None:
        if self._model_artifact is None:
            return None
        return WorkflowPayload("Model", self._model_artifact)

    def serialize_node_state(self) -> dict[str, object]:
        return {"load_path": str(self._load_path) if self._load_path else ""}

    def restore_node_state(self, payload: dict[str, object]) -> None:
        p = payload.get("load_path", "")
        if p:
            self._load_path = Path(p)
            self._path_label.setText(str(self._load_path))
            self._reload_btn.setEnabled(True)
            self._do_load()

    # ── File operations ───────────────────────────────────────────────

    def _browse_and_load(self) -> None:
        start = str(self._load_path.parent) if self._load_path else ""
        path, _ = QFileDialog.getOpenFileName(
            self, "Load Model", start, "Portakal Model (*.portakal);;All Files (*)"
        )
        if not path:
            return
        self._load_path = Path(path)
        self._path_label.setText(str(self._load_path))
        self._reload_btn.setEnabled(True)
        self._do_load()

    def _do_load(self) -> None:
        if self._load_path is None:
            return
        try:
            with open(self._load_path, "rb") as f:
                self._model_artifact = pickle.load(f)  # noqa: S301
            name = getattr(self._model_artifact, "display_name", type(self._model_artifact).__name__)
            self._status_label.setText(f"Loaded: {name}")
            self._status_label.setStyleSheet("color: #2e7d32; background: transparent;")
        except Exception as exc:
            self._model_artifact = None
            self._status_label.setText(f"Load failed: {exc}")
            self._status_label.setStyleSheet("color: #c62828; background: transparent;")
        self._notify_output_changed()
