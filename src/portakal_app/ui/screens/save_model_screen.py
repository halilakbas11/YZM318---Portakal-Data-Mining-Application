from __future__ import annotations

import pickle
from pathlib import Path

from PySide6.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from portakal_app.models import WorkflowPayload
from portakal_app.ui.screens.node_screen import WorkflowNodeScreenSupport


class SaveModelScreen(QWidget, WorkflowNodeScreenSupport):
    """Save Model — pickle a trained model artifact to a .portakal file."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._init_workflow_node_support()
        self._model_artifact = None
        self._save_path: Path | None = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(10)

        self._model_label = QLabel("No model on input.")
        self._model_label.setWordWrap(True)
        self._model_label.setStyleSheet("font-weight: bold; background: transparent;")
        layout.addWidget(self._model_label)

        self._path_label = QLabel("No file selected.")
        self._path_label.setWordWrap(True)
        self._path_label.setStyleSheet("color: #777; background: transparent;")
        layout.addWidget(self._path_label)

        btn_row = QHBoxLayout()
        self._browse_btn = QPushButton("Save As…")
        self._browse_btn.clicked.connect(self._browse_and_save)
        btn_row.addWidget(self._browse_btn)
        self._save_btn = QPushButton("Save")
        self._save_btn.setProperty("primary", True)
        self._save_btn.clicked.connect(self._do_save)
        self._save_btn.setEnabled(False)
        btn_row.addWidget(self._save_btn)
        btn_row.addStretch(1)
        layout.addLayout(btn_row)

        self._status_label = QLabel("")
        self._status_label.setWordWrap(True)
        self._status_label.setStyleSheet("background: transparent;")
        layout.addWidget(self._status_label)

        layout.addStretch(1)

    # ── Workflow integration ──────────────────────────────────────────

    def set_input_payload(self, payload: WorkflowPayload | None) -> None:
        if payload is None:
            self._model_artifact = None
            self._model_label.setText("No model on input.")
        else:
            self._model_artifact = payload.value
            name = getattr(payload.value, "display_name", type(payload.value).__name__)
            self._model_label.setText(f"Model: {name}")
        self._save_btn.setEnabled(self._model_artifact is not None and self._save_path is not None)
        self._status_label.setText("")

    def current_output_payload(self) -> WorkflowPayload | None:
        return None

    def serialize_node_state(self) -> dict[str, object]:
        return {"save_path": str(self._save_path) if self._save_path else ""}

    def restore_node_state(self, payload: dict[str, object]) -> None:
        p = payload.get("save_path", "")
        if p:
            self._save_path = Path(p)
            self._path_label.setText(str(self._save_path))

    # ── File operations ───────────────────────────────────────────────

    def _browse_and_save(self) -> None:
        start = str(self._save_path.parent) if self._save_path else ""
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Model", start, "Portakal Model (*.portakal);;All Files (*)"
        )
        if not path:
            return
        if not path.endswith(".portakal"):
            path += ".portakal"
        self._save_path = Path(path)
        self._path_label.setText(str(self._save_path))
        self._save_btn.setEnabled(self._model_artifact is not None)
        self._do_save()

    def _do_save(self) -> None:
        if self._model_artifact is None:
            self._status_label.setText("No model to save.")
            self._status_label.setStyleSheet("color: #c62828; background: transparent;")
            return
        if self._save_path is None:
            self._browse_and_save()
            return
        try:
            with open(self._save_path, "wb") as f:
                pickle.dump(self._model_artifact, f)
            self._status_label.setText(f"Saved to {self._save_path.name}")
            self._status_label.setStyleSheet("color: #2e7d32; background: transparent;")
        except Exception as exc:
            self._status_label.setText(f"Save failed: {exc}")
            self._status_label.setStyleSheet("color: #c62828; background: transparent;")
