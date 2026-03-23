from __future__ import annotations

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PySide6")

from PySide6.QtTest import QTest

from portakal_app.app import create_application
from portakal_app.data.models import PaintDataPoint
from portakal_app.data.services.file_import_service import FileImportService
from portakal_app.ui.main_window import MainWindow
from portakal_app.ui.screens.color_screen import ColorScreen
from portakal_app.ui.screens.datasets_screen import DatasetsScreen
from portakal_app.ui.screens.paint_data_screen import PaintDataScreen


@pytest.fixture(scope="session")
def app():
    return create_application()


@pytest.fixture()
def sample_dataset(tmp_path):
    path = tmp_path / "sample.csv"
    path.write_text("x,y,label\n0.1,0.2,A\n0.9,0.8,B\n0.2,0.4,A\n", encoding="utf-8")
    return FileImportService().load(str(path))


def test_datasets_screen_lists_curated_entries(app):
    screen = DatasetsScreen()
    assert screen._table.rowCount() == 15
    assert screen._domain_combo.count() >= 5
    assert "datasets shown" in screen._summary_label.text().lower()


def test_datasets_screen_filters_by_query(app):
    screen = DatasetsScreen()
    screen._search_input.setText("titanic")
    assert screen._table.rowCount() == 1
    assert "Titanic" in screen._table.item(0, 0).text()


def test_paint_data_screen_loads_points_from_dataset_and_emits_dataset(app, sample_dataset):
    screen = PaintDataScreen()
    captured = {}
    screen.on_apply_requested(lambda dataset: captured.setdefault("dataset", dataset))

    screen.set_dataset(sample_dataset)
    assert screen._canvas.plot_point_count() == sample_dataset.row_count
    screen._emit_dataset()

    emitted = captured["dataset"]
    assert emitted.row_count == sample_dataset.row_count
    assert emitted.column_count == 3
    assert emitted.display_name == "Painted Data"


def test_paint_data_screen_keeps_selected_label_after_own_output_round_trip(app, sample_dataset):
    screen = PaintDataScreen()
    screen.set_dataset(sample_dataset)
    screen._labels_list.setCurrentRow(1)
    selected_before = screen._labels_list.currentItem().text()

    emitted = screen._service.build_dataset(screen._current_snapshot())
    screen.set_dataset(emitted)

    assert screen._labels_list.currentItem().text() == selected_before


def test_paint_data_screen_reset_restores_last_external_input(app, sample_dataset):
    screen = PaintDataScreen()
    screen.set_dataset(sample_dataset)
    original_points = screen._canvas.plot_point_count()

    screen._canvas.set_current_label(screen._labels_list.item(1).text())
    screen._canvas._points.append(PaintDataPoint(0.55, 0.66, screen._labels_list.item(1).text()))
    emitted = screen._service.build_dataset(screen._current_snapshot())
    screen.set_dataset(emitted)
    assert screen._canvas.plot_point_count() == original_points + 1

    screen._reset_to_input()

    assert screen._canvas.plot_point_count() == original_points


def test_color_screen_builds_assignments_and_emits_annotations(app, sample_dataset):
    screen = ColorScreen()
    captured = {}
    screen.on_apply_requested(lambda dataset: captured.setdefault("dataset", dataset))

    screen.set_dataset(sample_dataset)
    assert len(screen._state["discrete"]) == 1
    assert len(screen._state["numeric"]) == 2
    screen._emit_dataset()

    emitted = captured["dataset"]
    assert "color_settings" in emitted.annotations
    assert emitted.annotations["color_settings"]["discrete"]["label"]["A"]


def test_color_screen_reset_restores_service_defaults(app, sample_dataset):
    screen = ColorScreen()
    screen.set_dataset(sample_dataset)
    original = screen._state["numeric"]["x"]
    screen._set_numeric_palette("x", "Berry")
    assert screen._state["numeric"]["x"] == "Berry"
    screen._reset_state()
    assert screen._state["numeric"]["x"] == original


def test_main_window_opens_new_active_widgets(app):
    window = MainWindow()
    window.show()
    QTest.qWait(50)

    window._workspace.canvas.add_workflow_node("datasets")
    window._show_widget("datasets")
    assert isinstance(window._workspace.current_widget(), DatasetsScreen)
    assert window._widget_index["datasets"].enabled is True

    window._workspace.canvas.add_workflow_node("paint-data")
    window._show_widget("paint-data")
    assert isinstance(window._workspace.current_widget(), PaintDataScreen)
    assert window._widget_index["paint-data"].enabled is True

    window._workspace.canvas.add_workflow_node("color")
    window._show_widget("color")
    assert isinstance(window._workspace.current_widget(), ColorScreen)
    assert window._widget_index["color"].enabled is True


def test_disconnected_input_widget_clears_dataset(app, tmp_path):
    path = tmp_path / "connected.csv"
    path.write_text("x,y,label\n0.1,0.2,A\n0.9,0.8,B\n", encoding="utf-8")

    window = MainWindow()
    canvas = window._workspace.canvas
    scene = canvas.workflow_scene
    source = canvas.add_workflow_node("file")
    target = canvas.add_workflow_node("paint-data")
    assert scene.create_connection(source.node_id, target.node_id)

    window._handle_file_selected(str(path))
    window._show_widget(target.node_id)
    paint_screen = window._workspace.current_widget()
    assert isinstance(paint_screen, PaintDataScreen)
    assert paint_screen._canvas.plot_point_count() == 2

    scene.clearSelection()
    edge = scene._edges[0]
    edge.setSelected(True)
    assert scene.delete_selected_items()

    assert paint_screen._canvas.plot_point_count() == 0
