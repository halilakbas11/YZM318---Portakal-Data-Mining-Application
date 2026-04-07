from __future__ import annotations

import os
from dataclasses import replace

import polars as pl
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication, QFileDialog, QMessageBox

from portakal_app.data.services.generated_dataset_service import GeneratedDatasetService
from portakal_app.models import WorkflowPayload
from portakal_app.ui.catalog import build_widgets
from portakal_app.ui.main_window import MainWindow
from portakal_app.ui.screens.venn_diagram_screen import VennDiagramScreen


@pytest.fixture(scope="session")
def app():
    return QApplication.instance() or QApplication([])


def _build_dataset(service: GeneratedDatasetService, dataframe: pl.DataFrame, dataset_id: str):
    return service.build_dataset(
        dataframe,
        dataset_id=dataset_id,
        display_name=dataset_id.upper(),
        file_name=f"{dataset_id}.csv",
    )


class _SaveStub:
    def __init__(self) -> None:
        self.calls: list[tuple[object, str]] = []

    def save(self, dataset, path: str, format: str | None = None) -> None:
        _ = format
        self.calls.append((dataset, path))


def test_catalog_registers_venn_diagram_widget():
    widgets = {widget.id: widget for widget in build_widgets()}
    assert "venn-diagram" in widgets
    assert widgets["venn-diagram"].enabled is True
    assert widgets["venn-diagram"].input_channels == ("Data 1", "Data 2", "Data 3", "Data 4", "Data 5")
    assert widgets["venn-diagram"].output_channels == ("Selected Data", "Annotated Data")


def test_main_window_can_open_venn_diagram_widget(app):
    window = MainWindow()
    window._workspace.canvas.add_workflow_node("venn-diagram")
    window._show_widget("venn-diagram")

    assert isinstance(window._workspace.current_widget(), VennDiagramScreen)
    assert window._widget_index["venn-diagram"].enabled is True


def test_venn_diagram_rowwise_identity_outputs_selected_and_annotated_data(app):
    service = GeneratedDatasetService()
    base = _build_dataset(service, pl.DataFrame({"name": ["A", "B", "C"], "value": [1, 2, 3]}), "base")
    subset = _build_dataset(service, pl.DataFrame({"name": ["B", "C"], "value": [2, 3]}), "subset")
    subset = replace(subset, source=base.source)

    screen = VennDiagramScreen()
    screen.set_input_payload(None)
    screen.set_input_payload(WorkflowPayload("Data 1", base))
    screen.set_input_payload(WorkflowPayload("Data 2", subset))
    app.processEvents()

    screen._diagram.set_selected_areas({3})
    app.processEvents()

    outputs = screen.current_output_datasets()
    selected = outputs["Selected Data"]
    annotated = outputs["Annotated Data"]

    assert selected is not None
    assert annotated is not None
    assert selected.row_count == 2
    assert annotated.row_count == 3
    assert annotated.dataframe["Selected"].to_list() == [False, True, True]


def test_venn_diagram_feature_matching_can_output_duplicates(app):
    service = GeneratedDatasetService()
    left = _build_dataset(service, pl.DataFrame({"key": ["A", "A", "B"], "value": [1, 2, 3]}), "left")
    right = _build_dataset(service, pl.DataFrame({"key": ["A", "B"], "score": [10, 20]}), "right")

    screen = VennDiagramScreen()
    screen.set_input_payload(None)
    screen.set_input_payload(WorkflowPayload("Data 1", left))
    screen.set_input_payload(WorkflowPayload("Data 2", right))
    app.processEvents()

    for index in range(screen._match_combo.count()):
        if screen._match_combo.itemText(index) == "key":
            screen._match_combo.setCurrentIndex(index)
            break
    screen._duplicates_checkbox.setChecked(True)
    app.processEvents()

    screen._diagram.set_selected_areas({3})
    app.processEvents()

    outputs = screen.current_output_datasets()
    selected = outputs["Selected Data"]
    annotated = outputs["Annotated Data"]

    assert selected is not None
    assert annotated is not None
    assert selected.row_count == 5
    assert set(selected.dataframe.columns) == {"value (1)", "key (1)", "score (2)", "key (2)"}
    assert annotated.row_count == 2


def test_venn_diagram_columns_mode_outputs_selected_features_and_targets(app):
    service = GeneratedDatasetService()
    base = _build_dataset(service, pl.DataFrame({"a": [1, 2], "b": [3, 4], "target": ["x", "y"]}), "first")
    other = _build_dataset(service, pl.DataFrame({"b": [3, 4], "c": [5, 6], "target": ["x", "y"]}), "second")

    row_keys = [("shared", index) for index in range(base.row_count)]
    base = replace(base, annotations={"row_identity_keys": row_keys})
    other = replace(other, annotations={"row_identity_keys": row_keys})

    screen = VennDiagramScreen()
    screen.set_input_payload(None)
    screen.set_input_payload(WorkflowPayload("Data 1", base))
    screen.set_input_payload(WorkflowPayload("Data 2", other))
    app.processEvents()

    screen._columns_radio.setChecked(True)
    app.processEvents()

    screen._diagram.set_selected_areas({3})
    app.processEvents()

    outputs = screen.current_output_datasets()
    selected = outputs["Selected Data"]
    annotated = outputs["Annotated Data"]

    assert selected is not None
    assert annotated is not None
    assert selected.dataframe.columns == ["b", "target"]
    assert annotated.dataframe.columns == ["a", "b", "target", "c"]
    assert annotated.annotations["selected_features"] == ["b"]


def test_venn_diagram_save_export_prefers_selected_output(app, monkeypatch, tmp_path):
    service = GeneratedDatasetService()
    base = _build_dataset(service, pl.DataFrame({"name": ["A", "B", "C"], "value": [1, 2, 3]}), "base")
    subset = _build_dataset(service, pl.DataFrame({"name": ["B", "C"], "value": [2, 3]}), "subset")
    subset = replace(subset, source=base.source)

    screen = VennDiagramScreen()
    save_stub = _SaveStub()
    screen.set_save_data_service(save_stub)
    screen.set_input_payload(None)
    screen.set_input_payload(WorkflowPayload("Data 1", base))
    screen.set_input_payload(WorkflowPayload("Data 2", subset))
    app.processEvents()

    screen._diagram.set_selected_areas({3})
    app.processEvents()

    target_path = tmp_path / "venn-selected.csv"
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *args, **kwargs: (str(target_path), "Data Files"))
    monkeypatch.setattr(QMessageBox, "information", lambda *args, **kwargs: None)

    selected = screen.current_output_datasets()["Selected Data"]
    screen.save_export_dataset()

    assert selected is not None
    assert save_stub.calls == [(selected, str(target_path))]


def test_venn_diagram_dialog_footer_shows_save_button(app):
    service = GeneratedDatasetService()
    base = _build_dataset(service, pl.DataFrame({"name": ["A", "B", "C"], "value": [1, 2, 3]}), "base")
    subset = _build_dataset(service, pl.DataFrame({"name": ["B", "C"], "value": [2, 3]}), "subset")
    subset = replace(subset, source=base.source)

    window = MainWindow()
    window._workspace.canvas.add_workflow_node("venn-diagram")
    app.processEvents()
    window._show_widget("venn-diagram")

    screen = window._workspace.current_widget()
    assert isinstance(screen, VennDiagramScreen)

    node_id = window._workspace._last_opened_node_id
    assert node_id is not None
    dialog = window._workspace._dialogs[node_id]
    assert dialog._save_export_button is not None
    assert dialog._save_export_button.text() == "Save"
    assert dialog._save_export_button.isEnabled() is False

    screen.set_input_payload(None)
    screen.set_input_payload(WorkflowPayload("Data 1", base))
    screen.set_input_payload(WorkflowPayload("Data 2", subset))
    app.processEvents()
    window._workspace.refresh_dialog_footers(node_id)

    assert dialog._save_export_button.isEnabled() is True
