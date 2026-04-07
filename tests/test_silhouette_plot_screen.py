from __future__ import annotations

import os

import polars as pl
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication, QFileDialog, QMessageBox

from portakal_app.data.services.generated_dataset_service import GeneratedDatasetService
from portakal_app.models import WorkflowPayload
from portakal_app.ui.catalog import build_widgets
from portakal_app.ui.main_window import MainWindow
from portakal_app.ui.screens.silhouette_plot_screen import SilhouettePlotScreen


@pytest.fixture(scope="session")
def app():
    return QApplication.instance() or QApplication([])


def _build_dataset(
    service: GeneratedDatasetService,
    dataframe: pl.DataFrame,
    dataset_id: str,
    *,
    role_overrides: dict[str, str] | None = None,
):
    return service.build_dataset(
        dataframe,
        dataset_id=dataset_id,
        display_name=dataset_id.upper(),
        file_name=f"{dataset_id}.csv",
        role_overrides=role_overrides,
    )


def _score_column_name(dataset) -> str:
    return next(column for column in dataset.dataframe.columns if column.startswith("Silhouette ("))


class _SaveStub:
    def __init__(self) -> None:
        self.calls: list[tuple[object, str]] = []

    def save(self, dataset, path: str, format: str | None = None) -> None:
        _ = format
        self.calls.append((dataset, path))


def test_catalog_registers_silhouette_plot_widget():
    widgets = {widget.id: widget for widget in build_widgets()}
    assert "silhouette-plot" in widgets
    assert widgets["silhouette-plot"].enabled is True
    assert widgets["silhouette-plot"].output_channels == ("Selected Data", "Annotated Data")


def test_main_window_can_open_silhouette_plot_widget(app):
    window = MainWindow()
    window._workspace.canvas.add_workflow_node("silhouette-plot")
    window._show_widget("silhouette-plot")

    assert isinstance(window._workspace.current_widget(), SilhouettePlotScreen)
    assert window._widget_index["silhouette-plot"].enabled is True


def test_silhouette_plot_outputs_selected_and_annotated_data(app):
    service = GeneratedDatasetService()
    dataset = _build_dataset(
        service,
        pl.DataFrame(
            {
                "x": [0.0, 0.2, 3.0, 3.2],
                "y": [0.0, 0.1, 3.1, 3.3],
                "cluster": ["A", "A", "B", "B"],
                "label": ["r1", "r2", "r3", "r4"],
            }
        ),
        "silhouette_base",
        role_overrides={"x": "feature", "y": "feature", "cluster": "target", "label": "meta"},
    )

    screen = SilhouettePlotScreen()
    screen.set_input_payload(WorkflowPayload("Data", dataset))
    app.processEvents()

    for index in range(screen._annotation_combo.count()):
        if screen._annotation_combo.itemData(index) == "label":
            screen._annotation_combo.setCurrentIndex(index)
            break
    screen._bar_slider.setValue(6)
    screen._canvas.set_selection([0, 1])
    app.processEvents()

    outputs = screen.current_output_datasets()
    selected = outputs["Selected Data"]
    annotated = outputs["Annotated Data"]

    assert selected is not None
    assert annotated is not None
    assert selected.row_count == 2
    assert annotated.row_count == 4

    score_name = _score_column_name(annotated)
    assert score_name in selected.dataframe.columns
    assert annotated.dataframe["Selected"].to_list() == [True, True, False, False]
    assert all(value is not None for value in selected.dataframe[score_name].to_list())


def test_silhouette_plot_save_export_prefers_selected_output(app, monkeypatch, tmp_path):
    service = GeneratedDatasetService()
    dataset = _build_dataset(
        service,
        pl.DataFrame(
            {
                "x": [0.0, 0.2, 3.0, 3.2],
                "y": [0.0, 0.1, 3.1, 3.3],
                "cluster": ["A", "A", "B", "B"],
            }
        ),
        "silhouette_export_base",
        role_overrides={"x": "feature", "y": "feature", "cluster": "target"},
    )

    screen = SilhouettePlotScreen()
    save_stub = _SaveStub()
    screen.set_save_data_service(save_stub)
    screen.set_input_payload(WorkflowPayload("Data", dataset))
    app.processEvents()

    screen._canvas.set_selection([0, 1])
    app.processEvents()

    target_path = tmp_path / "silhouette-selected.csv"
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *args, **kwargs: (str(target_path), "Data Files"))
    monkeypatch.setattr(QMessageBox, "information", lambda *args, **kwargs: None)

    selected = screen.current_output_datasets()["Selected Data"]
    screen.save_export_dataset()

    assert selected is not None
    assert save_stub.calls == [(selected, str(target_path))]


def test_silhouette_plot_dialog_footer_shows_save_button(app):
    service = GeneratedDatasetService()
    dataset = _build_dataset(
        service,
        pl.DataFrame(
            {
                "x": [0.0, 0.2, 3.0, 3.2],
                "y": [0.0, 0.1, 3.1, 3.3],
                "cluster": ["A", "A", "B", "B"],
            }
        ),
        "silhouette_footer_base",
        role_overrides={"x": "feature", "y": "feature", "cluster": "target"},
    )

    window = MainWindow()
    window._workspace.canvas.add_workflow_node("silhouette-plot")
    app.processEvents()
    window._show_widget("silhouette-plot")

    screen = window._workspace.current_widget()
    assert isinstance(screen, SilhouettePlotScreen)

    node_id = window._workspace._last_opened_node_id
    assert node_id is not None
    dialog = window._workspace._dialogs[node_id]
    assert dialog._save_export_button is not None
    assert dialog._save_export_button.text() == "Save"
    assert dialog._save_export_button.isEnabled() is False

    screen.set_input_payload(WorkflowPayload("Data", dataset))
    app.processEvents()
    window._workspace.refresh_dialog_footers(node_id)

    assert dialog._save_export_button.isEnabled() is True


def test_silhouette_plot_reports_singleton_cluster_error(app):
    service = GeneratedDatasetService()
    dataset = _build_dataset(
        service,
        pl.DataFrame(
            {
                "x": [0.0, 1.0, 2.0],
                "y": [0.0, 1.0, 2.0],
                "cluster": ["A", "B", "C"],
            }
        ),
        "silhouette_singletons",
        role_overrides={"x": "feature", "y": "feature", "cluster": "target"},
    )

    screen = SilhouettePlotScreen()
    screen.set_input_payload(WorkflowPayload("Data", dataset))
    app.processEvents()

    outputs = screen.current_output_datasets()
    assert outputs["Selected Data"] is None
    assert outputs["Annotated Data"] is None
    assert "singletons" in screen._status_label.text().lower()


def test_silhouette_plot_cosine_marks_zero_vectors_as_undefined(app):
    service = GeneratedDatasetService()
    dataset = _build_dataset(
        service,
        pl.DataFrame(
            {
                "x": [0.0, 1.0, 0.5, 1.5],
                "y": [0.0, 0.0, 1.2, 1.1],
                "cluster": ["A", "A", "B", "B"],
            }
        ),
        "silhouette_cosine",
        role_overrides={"x": "feature", "y": "feature", "cluster": "target"},
    )

    screen = SilhouettePlotScreen()
    screen.set_input_payload(WorkflowPayload("Data", dataset))
    app.processEvents()

    for index in range(screen._metric_combo.count()):
        if screen._metric_combo.itemData(index) == "cosine":
            screen._metric_combo.setCurrentIndex(index)
            break
    app.processEvents()

    outputs = screen.current_output_datasets()
    annotated = outputs["Annotated Data"]

    assert annotated is not None
    score_name = _score_column_name(annotated)
    assert annotated.dataframe[score_name].to_list()[0] is None
    assert "undefined distances" in screen._status_label.text().lower()


def test_silhouette_plot_uses_unique_score_column_name(app):
    service = GeneratedDatasetService()
    dataset = _build_dataset(
        service,
        pl.DataFrame(
            {
                "x": [0.0, 0.2, 3.0, 3.2],
                "y": [0.0, 0.1, 3.1, 3.3],
                "cluster": ["A", "A", "B", "B"],
                "Silhouette (cluster)": [1.0, 1.0, 1.0, 1.0],
            }
        ),
        "silhouette_name_conflict",
        role_overrides={"x": "feature", "y": "feature", "cluster": "target", "Silhouette (cluster)": "meta"},
    )

    screen = SilhouettePlotScreen()
    screen.set_input_payload(WorkflowPayload("Data", dataset))
    app.processEvents()

    annotated = screen.current_output_datasets()["Annotated Data"]
    assert annotated is not None
    assert "Silhouette (cluster) (1)" in annotated.dataframe.columns
