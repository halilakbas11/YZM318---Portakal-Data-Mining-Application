from __future__ import annotations

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PySide6")

from PySide6.QtCore import Qt
from PySide6.QtTest import QTest

from portakal_app.app import create_application
from portakal_app.data.models import PaintDataPoint
from portakal_app.data.services.file_import_service import FileImportService
from portakal_app.ui.main_window import MainWindow
from portakal_app.ui.screens.ca_screen import CAScreen
from portakal_app.ui.screens.color_screen import ColorScreen
from portakal_app.ui.screens.correlations_screen import CorrelationsScreen
from portakal_app.ui.screens.dbscan_screen import DBSCANScreen
from portakal_app.ui.screens.distance_file_screen import DistanceFileScreen
from portakal_app.ui.screens.datasets_screen import DatasetsScreen
from portakal_app.ui.screens.distance_map_screen import DistanceMapScreen
from portakal_app.ui.screens.distance_matrix_screen import DistanceMatrixScreen
from portakal_app.ui.screens.distance_transformation_screen import DistanceTransformationScreen
from portakal_app.ui.screens.distances_screen import DistancesScreen
from portakal_app.ui.screens.mds_screen import MDSScreen
from portakal_app.ui.screens.outliers_screen import OutliersScreen
from portakal_app.ui.screens.paint_data_screen import PaintDataScreen
from portakal_app.ui.screens.save_distance_matrix_screen import SaveDistanceMatrixScreen


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


def test_paint_data_screen_variable_names_change_emitted_column_names(app, sample_dataset):
    screen = PaintDataScreen()
    screen.set_dataset(sample_dataset)

    screen._x_name_input.setText("feature_x")
    screen._y_name_input.setText("feature_y")
    screen._emit_dataset()

    emitted = screen.current_output_dataset()
    assert emitted is not None
    assert emitted.dataframe.columns == ["feature_x", "feature_y", "label"]
    assert screen._canvas._x_axis_label == "feature_x"
    assert screen._canvas._y_axis_label == "feature_y"


def test_paint_data_screen_source_column_change_rebuilds_graph_and_preserves_custom_output_names(app, tmp_path):
    path = tmp_path / "paint-columns.csv"
    path.write_text(
        "axis_a,axis_b,axis_c,label\n"
        "1,10,100,A\n"
        "2,20,100,A\n"
        "3,30,300,B\n",
        encoding="utf-8",
    )
    dataset = FileImportService().load(str(path))

    screen = PaintDataScreen()
    screen.set_dataset(dataset)
    screen._x_name_input.setText("feature_x")
    screen._y_name_input.setText("feature_y")

    screen._x_source_combo.setCurrentText("axis_c")

    snapshot = screen._current_snapshot()
    assert snapshot.x_source_name == "axis_c"
    assert snapshot.y_source_name == "axis_b"
    assert snapshot.x_name == "feature_x"
    assert snapshot.y_name == "feature_y"
    assert [point.x for point in snapshot.points] == [0.0, 0.0, 1.0]
    assert [point.y for point in snapshot.points] == [0.0, 0.5, 1.0]
    assert snapshot.labels == ("A", "B")
    assert screen._canvas._x_axis_label == "feature_x"
    assert screen._canvas._y_axis_label == "feature_y"

    screen._emit_dataset()
    emitted = screen.current_output_dataset()
    assert emitted is not None
    assert emitted.dataframe.columns == ["feature_x", "feature_y", "label"]
    assert emitted.dataframe["feature_x"].to_list() == [0.0, 0.0, 1.0]
    assert emitted.dataframe["feature_y"].to_list() == [0.0, 0.5, 1.0]


def test_paint_data_screen_typing_existing_column_name_rebuilds_graph(app, tmp_path):
    path = tmp_path / "paint-typed-columns.csv"
    path.write_text(
        "axis_a,axis_b,axis_c,label\n"
        "1,10,100,A\n"
        "2,20,100,A\n"
        "3,30,300,B\n",
        encoding="utf-8",
    )
    dataset = FileImportService().load(str(path))

    screen = PaintDataScreen()
    screen.set_dataset(dataset)

    screen._x_name_input.setText("axis_c")
    screen._y_name_input.setText("axis_b")

    snapshot = screen._current_snapshot()
    assert snapshot.x_source_name == "axis_c"
    assert snapshot.y_source_name == "axis_b"
    assert snapshot.x_name == "axis_c"
    assert snapshot.y_name == "axis_b"
    assert [point.x for point in snapshot.points] == [0.0, 0.0, 1.0]
    assert [point.y for point in snapshot.points] == [0.0, 0.5, 1.0]


def test_paint_data_screen_wraps_sidebar_in_scroll_area(app):
    screen = PaintDataScreen()

    assert screen._sidebar_scroll.widget() is screen._sidebar
    assert screen._sidebar_scroll.horizontalScrollBarPolicy() == Qt.ScrollBarPolicy.ScrollBarAlwaysOff


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

    window._workspace.canvas.add_workflow_node("distances")
    window._show_widget("distances")
    assert isinstance(window._workspace.current_widget(), DistancesScreen)
    assert window._widget_index["distances"].enabled is True

    window._workspace.canvas.add_workflow_node("distance-matrix")
    window._show_widget("distance-matrix")
    assert isinstance(window._workspace.current_widget(), DistanceMatrixScreen)
    assert window._widget_index["distance-matrix"].enabled is True

    window._workspace.canvas.add_workflow_node("distance-file")
    window._show_widget("distance-file")
    assert isinstance(window._workspace.current_widget(), DistanceFileScreen)
    assert window._widget_index["distance-file"].enabled is True

    window._workspace.canvas.add_workflow_node("distance-transformation")
    window._show_widget("distance-transformation")
    assert isinstance(window._workspace.current_widget(), DistanceTransformationScreen)
    assert window._widget_index["distance-transformation"].enabled is True

    window._workspace.canvas.add_workflow_node("save-distance-matrix")
    window._show_widget("save-distance-matrix")
    assert isinstance(window._workspace.current_widget(), SaveDistanceMatrixScreen)
    assert window._widget_index["save-distance-matrix"].enabled is True

    window._workspace.canvas.add_workflow_node("correlations")
    window._show_widget("correlations")
    assert isinstance(window._workspace.current_widget(), CorrelationsScreen)
    assert window._widget_index["correlations"].enabled is True

    window._workspace.canvas.add_workflow_node("distance-map")
    window._show_widget("distance-map")
    assert isinstance(window._workspace.current_widget(), DistanceMapScreen)
    assert window._widget_index["distance-map"].enabled is True

    window._workspace.canvas.add_workflow_node("outliers")
    window._show_widget("outliers")
    assert isinstance(window._workspace.current_widget(), OutliersScreen)
    assert window._widget_index["outliers"].enabled is True

    window._workspace.canvas.add_workflow_node("dbscan")
    window._show_widget("dbscan")
    assert isinstance(window._workspace.current_widget(), DBSCANScreen)
    assert window._widget_index["dbscan"].enabled is True

    window._workspace.canvas.add_workflow_node("correspondence-analysis")
    window._show_widget("correspondence-analysis")
    assert isinstance(window._workspace.current_widget(), CAScreen)
    assert window._widget_index["correspondence-analysis"].enabled is True

    window._workspace.canvas.add_workflow_node("mds")
    window._show_widget("mds")
    assert isinstance(window._workspace.current_widget(), MDSScreen)
    assert window._widget_index["mds"].enabled is True


def test_file_widget_column_edits_flow_to_downstream_data_table(app, tmp_path):
    path = tmp_path / "edited.csv"
    path.write_text("a,b,c\n1,2,x\n3,4,y\n", encoding="utf-8")

    window = MainWindow()
    canvas = window._workspace.canvas
    scene = canvas.workflow_scene
    source = canvas.add_workflow_node("file")
    target = canvas.add_workflow_node("data-table")
    assert scene.create_connection(source.node_id, target.node_id)

    window._handle_file_selected(str(path))
    file_screen = window._node_runtimes[source.node_id].screen
    data_table = window._node_runtimes[target.node_id].screen

    file_screen._columns_table.item(0, 0).setText("amount")
    file_screen._columns_table.cellWidget(2, 2).setCurrentText("meta")
    file_screen._handle_apply_clicked()

    assert data_table._headers == ["amount", "b", "c"]
    assert [(column.name, column.role) for column in data_table._dataset_handle.domain.columns] == [
        ("amount", "feature"),
        ("b", "feature"),
        ("c", "meta"),
    ]


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
