from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QSize
from PySide6.QtWidgets import (
    QCheckBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from portakal_app.data.errors import DatasetSaveError, PortakalDataError, UnsupportedFormatError
from portakal_app.data.models import DatasetHandle
from portakal_app.data.services.file_import_service import FileImportService
from portakal_app.data.services.save_data_service import SaveDataService
from portakal_app.ui.screens.node_screen import WorkflowNodeScreenSupport


def _dataset_label(dataset_handle: DatasetHandle | None) -> str:
    if dataset_handle is None:
        return "none"
    return dataset_handle.source.path.name


class SaveDataScreen(QWidget, WorkflowNodeScreenSupport):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._init_workflow_node_support()
        self._import_service = FileImportService()
        self._save_data_service = SaveDataService()
        self._dataset_handle: DatasetHandle | None = None
        self.setObjectName("saveDataScreen")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(14)

        self._title = QLabel("Save Data")
        self._title.setProperty("sectionTitle", True)
        self._title.setStyleSheet("background: transparent;")
        layout.addWidget(self._title)

        self._dataset = QLabel("Dataset: none")
        self._dataset.setProperty("muted", True)
        self._dataset.setStyleSheet("background: transparent;")
        layout.addWidget(self._dataset)

        self._annotations_checkbox = QCheckBox("Add type annotations to header")
        self._annotations_checkbox.setStyleSheet("background: transparent;")
        layout.addWidget(self._annotations_checkbox)

        self._autosave_checkbox = QCheckBox("Autosave when receiving new data")
        self._autosave_checkbox.setStyleSheet("background: transparent;")
        layout.addWidget(self._autosave_checkbox)

        buttons_row = QHBoxLayout()
        buttons_row.setSpacing(12)

        self._save_button = QPushButton("Save")
        self._save_button.setProperty("primary", True)
        self._save_button.setMinimumWidth(150)
        self._save_button.setEnabled(False)
        self._save_button.clicked.connect(self._save_default)
        buttons_row.addWidget(self._save_button)

        self._save_as_button = QPushButton("Save as ...")
        self._save_as_button.setProperty("secondary", True)
        self._save_as_button.setMinimumWidth(150)
        self._save_as_button.setEnabled(False)
        self._save_as_button.clicked.connect(self._save_as)
        buttons_row.addWidget(self._save_as_button)

        layout.addLayout(buttons_row)
        layout.addStretch(1)

    def sizeHint(self) -> QSize:
        return QSize(560, 220)

    def minimumSizeHint(self) -> QSize:
        return QSize(500, 200)

    def set_save_data_service(self, service: SaveDataService) -> None:
        self._save_data_service = service

    def set_dataset(self, dataset_handle: DatasetHandle | str | None) -> None:
        if isinstance(dataset_handle, str):
            try:
                dataset_handle = self._import_service.load(dataset_handle)
            except PortakalDataError:
                dataset_handle = None
        self._dataset_handle = dataset_handle
        self._dataset.setText(f"Dataset: {_dataset_label(dataset_handle)}")
        enabled = self._dataset_handle is not None
        self._save_button.setEnabled(enabled)
        self._save_as_button.setEnabled(enabled)

    def help_text(self) -> str:
        return (
            "Save the current dataset to disk. The screen mirrors Orange's Save Data widget "
            "with annotation and autosave options reserved for the backend integration."
        )

    def documentation_url(self) -> str:
        return "https://orangedatamining.com/widget-catalog/data/save-data/"

    def footer_status_text(self) -> str:
        if self._dataset_handle is None:
            return "0"
        return self._dataset_handle.source.path.suffix.lower() or "data"

    def set_input_payload(self, payload) -> None:
        dataset = payload.dataset if payload is not None else None
        self.set_dataset(dataset)

    def _save_default(self) -> None:
        if self._dataset_handle is None:
            return
        default_path = self._default_target_path()
        self._write_dataset(default_path)

    def _save_as(self) -> None:
        if self._dataset_handle is None:
            return
        target_path, _selected_filter = QFileDialog.getSaveFileName(
            self,
            "Save Data As",
            str(self._default_target_path()),
            "Data Files (*.csv *.xlsx *.parquet);;All Files (*.*)",
        )
        if not target_path:
            return
        self._write_dataset(Path(target_path))

    def _default_target_path(self) -> Path:
        assert self._dataset_handle is not None
        source_path = self._dataset_handle.source.path
        supported_export_formats = {"csv": ".csv", "xlsx": ".xlsx", "parquet": ".parquet"}
        suffix = supported_export_formats.get(self._dataset_handle.source.format, ".csv")
        return source_path.with_name(f"{source_path.stem}_copy{suffix}")

    def _write_dataset(self, target_path: Path) -> None:
        if self._dataset_handle is None:
            return
        try:
            if target_path.resolve() == self._dataset_handle.source.path.resolve():
                QMessageBox.information(self, "Save Data", "Choose a different output path.")
                return
        except OSError:
            pass

        try:
            self._save_data_service.save(self._dataset_handle, str(target_path))
        except UnsupportedFormatError as exc:
            QMessageBox.warning(self, "Save Data", str(exc))
            return
        except DatasetSaveError as exc:
            QMessageBox.warning(self, "Save Data", f"Could not save dataset.\n\n{exc}")
            return

        QMessageBox.information(self, "Save Data", f"Dataset saved to:\n{target_path}")
