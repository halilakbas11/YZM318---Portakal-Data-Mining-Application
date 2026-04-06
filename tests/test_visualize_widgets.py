from __future__ import annotations

import os

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PySide6")

from PySide6.QtTest import QTest

from portakal_app.app import create_application
from portakal_app.data.services.file_import_service import FileImportService
from portakal_app.data.services.generated_dataset_service import GeneratedDatasetService
from portakal_app.models import WorkflowPayload
from portakal_app.ui.catalog import build_widgets
from portakal_app.ui.main_window import MainWindow
from portakal_app.ui.screens.box_plot_screen import BoxPlotScreen
from portakal_app.ui.screens.distributions_screen import DistributionsScreen
from portakal_app.ui.screens.line_plot_screen import LinePlotScreen
from portakal_app.ui.screens.scatter_plot_screen import ScatterPlotScreen
from portakal_app.ui.screens.tree_viewer_screen import TreeViewerScreen
from portakal_app.ui.screens.violin_plot_screen import ViolinPlotScreen


@pytest.fixture(scope="session")
def app():
    return create_application()


@pytest.fixture()
def sample_dataset(tmp_path):
    path = tmp_path / "visual.csv"
    path.write_text(
        "sepal_length,sepal_width,petal_length,petal_width,species\n"
        "5.1,3.5,1.4,0.2,setosa\n"
        "4.9,3.0,1.4,0.2,setosa\n"
        "6.2,3.4,5.4,2.3,virginica\n"
        "5.9,3.0,5.1,1.8,virginica\n"
        "6.0,2.2,4.0,1.0,versicolor\n",
        encoding="utf-8",
    )
    return FileImportService().load(str(path))


@pytest.fixture()
def mixed_numeric_dataset(tmp_path):
    path = tmp_path / "mixed.csv"
    path.write_text(
        "score,group,segment\n"
        "1,A,North\n"
        "2,A,North\n"
        "3,B,North\n"
        "4,B,South\n"
        "5,C,South\n"
        "6,C,South\n"
        "7,A,North\n"
        "8,A,South\n"
        "9,B,South\n"
        "10,B,South\n"
        "11,C,North\n"
        "12,C,North\n",
        encoding="utf-8",
    )
    return FileImportService().load(str(path))


def _select_list_item(widget, text: str) -> None:
    for row in range(widget.count()):
        item = widget.item(row)
        if item.text() == text:
            widget.setCurrentRow(row)
            return
    raise AssertionError(f"item not found: {text}")


def test_visualize_catalog_contains_orange_first_six_widgets():
    widgets = {widget.id: widget for widget in build_widgets()}

    for widget_id in (
        "tree-viewer",
        "box-plot",
        "violin-plot",
        "distributions",
        "line-plot",
    ):
        assert widget_id in widgets
        assert widgets[widget_id].enabled is True
        assert widgets[widget_id].output_channels == ("Selected Data", "Annotated Data")

    assert widgets["tree-viewer"].input_ports[0].label == "Tree"
    assert widgets["scatter-plot"].enabled is True
    assert widgets["scatter-plot"].output_channels == ("Selected Data", "Annotated Data", "Features")
    assert widgets["scatter-plot"].input_channels == ("Data", "Data Subset", "Features")
    assert widgets["line-plot"].input_channels == ("Data", "Data Subset")


@pytest.mark.parametrize(
    ("screen_cls", "selected_rows"),
    [
        (ScatterPlotScreen, [0, 2]),
        (BoxPlotScreen, [1, 3]),
        (ViolinPlotScreen, [0, 1]),
        (DistributionsScreen, [2, 4]),
        (LinePlotScreen, [1, 2]),
    ],
)
def test_visualize_screens_emit_selected_and_annotated_outputs(app, sample_dataset, screen_cls, selected_rows):
    screen = screen_cls()
    screen.set_input_payload(WorkflowPayload("Data", sample_dataset))

    if isinstance(screen, (ScatterPlotScreen, LinePlotScreen)):
        subset = GeneratedDatasetService().build_dataset(
            sample_dataset.dataframe.head(2),
            dataset_id=f"{sample_dataset.dataset_id}-subset",
            display_name=f"{sample_dataset.display_name} subset",
            file_name="subset.csv",
            role_overrides={column.name: column.role for column in sample_dataset.domain.columns},
        )
        screen.set_input_payload(WorkflowPayload("Data Subset", subset))

    screen._handle_selection_changed(selected_rows)

    outputs = screen.current_output_datasets()
    assert outputs is not None
    assert outputs["Selected Data"] is not None
    assert outputs["Selected Data"].row_count == len(selected_rows)
    assert outputs["Annotated Data"] is not None
    assert outputs["Annotated Data"].annotations["selected_rows"] == sorted(selected_rows)


def test_scatter_plot_emits_feature_payload(app, sample_dataset):
    screen = ScatterPlotScreen()
    screen.set_input_payload(WorkflowPayload("Data", sample_dataset))
    screen._x_combo.setCurrentText("sepal_length")
    screen._y_combo.setCurrentText("petal_length")

    payloads = screen.current_output_payloads()
    assert payloads is not None
    assert payloads["Features"] is not None
    assert payloads["Features"].port_label == "Features"


def test_scatter_plot_matches_orange_sidebar_controls(app, sample_dataset):
    screen = ScatterPlotScreen()
    screen.set_input_payload(WorkflowPayload("Data", sample_dataset))

    assert screen._label_only_selected_cb.text() == "Label only selection and subset"
    assert screen._aggregate_cb.text() == "Aggregate points in dense regions"
    assert screen._class_density_cb.text() == "Show color regions"
    assert screen._legend_cb.text() == "Show legend"
    assert screen._show_grid_cb.text() == "Show gridlines"
    assert screen._tooltip_all_cb.text() == "Show all data on mouse hover"
    assert screen._zoom_in_button.text() == "Zoom In"
    assert screen._zoom_out_button.text() == "Zoom Out"
    assert screen._reset_zoom_button.text() == "Reset Zoom"


def test_box_plot_matches_orange_sidebar_structure(app, sample_dataset):
    screen = BoxPlotScreen()
    screen.set_input_payload(WorkflowPayload("Data", sample_dataset))

    assert screen._attribute_list.count() >= 5
    assert screen._group_list.item(0).text() == "None"
    assert screen._order_attr_cb.text() == "Order by relevance to subgroups"
    assert screen._order_group_cb.text() == "Order by relevance to variable"
    assert screen._show_annotations_cb.text() == "Annotate"
    assert screen._compare_none_rb.text() == "No comparison"
    assert screen._compare_medians_rb.text() == "Compare medians"
    assert screen._compare_means_rb.text() == "Compare means"


def test_violin_plot_matches_orange_sidebar_structure(app, sample_dataset):
    screen = ViolinPlotScreen()
    screen.set_input_payload(WorkflowPayload("Data", sample_dataset))

    assert screen._order_value_cb.text() == "Order by relevance to subgroups"
    assert screen._order_group_cb.text() == "Order by relevance to variable"
    assert screen._show_box_cb.text() == "Box plot"
    assert screen._show_strip_cb.text() == "Density dots"
    assert screen._show_rug_cb.text() == "Density lines"
    assert screen._order_cb.text() == "Order subgroups"
    assert screen._show_grid_cb.text() == "Show grid"
    assert screen._horizontal_rb.text() == "Horizontal"
    assert screen._vertical_rb.text() == "Vertical"


def test_distributions_matches_orange_sidebar_structure(app, sample_dataset):
    screen = DistributionsScreen()
    screen.set_input_payload(WorkflowPayload("Data", sample_dataset))

    assert screen._variable_filter_edit.placeholderText() == "Filter..."
    assert screen._sort_freq_cb.text() == "Sort categories by frequency"
    assert screen._stacked_cb.text() == "Stack columns"
    assert screen._show_probs_cb.text() == "Show probabilities"
    assert screen._cumulative_cb.text() == "Show cumulative distribution"
    assert screen._legend_cb.text() == "Show legend"
    assert screen._auto_apply_cb.text() == "Apply Automatically"
    assert [screen._fit_combo.itemText(index) for index in range(screen._fit_combo.count())] == [
        "None",
        "Normal",
        "Beta",
        "Gamma",
        "Rayleigh",
        "Pareto",
        "Exponential",
        "Kernel density",
    ]


def test_box_plot_discrete_mode_uses_stacked_group_rows(app, mixed_numeric_dataset):
    screen = BoxPlotScreen()
    screen.set_input_payload(WorkflowPayload("Data", mixed_numeric_dataset))
    _select_list_item(screen._attribute_list, "group")
    _select_list_item(screen._group_list, "segment")
    screen._refresh_plot()

    assert screen._display_box.isHidden() is True
    assert screen._stretching_box.isHidden() is False
    assert len(screen._canvas._discrete_rows) == 2


def test_line_plot_matches_orange_sidebar_structure(app, sample_dataset):
    screen = LinePlotScreen()
    screen.set_input_payload(WorkflowPayload("Data", sample_dataset))

    assert screen._profiles_cb.text() == "Lines"
    assert screen._range_cb.text() == "Range"
    assert screen._mean_cb.text() == "Mean"
    assert screen._error_cb.text() == "Error bars"
    assert screen._group_filter_edit.placeholderText() == "Filter..."
    assert screen._auto_apply_cb.text() == "Apply Automatically"
    assert screen._auto_apply_cb.isChecked() is True
    assert [screen._group_list.item(i).text() for i in range(screen._group_list.count())] == ["None", "species"]
    assert screen._current_group_name() == "species"
    assert screen._canvas._feature_names == ["sepal_length", "sepal_width", "petal_length", "petal_width"]
    assert screen._canvas._y_domain[0] < 0.2
    assert screen._canvas._y_domain[1] > 6.2


def test_tree_viewer_accepts_embedded_tree_payload(app, sample_dataset):
    screen = TreeViewerScreen()
    screen.set_input_payload(
        WorkflowPayload(
            "Tree",
            {
                "dataset": sample_dataset,
                "tree": {
                    "name": "Root",
                    "children": [
                        {"name": "Left", "rows": [0, 1]},
                        {"name": "Right", "rows": [2, 3, 4]},
                    ],
                },
            },
        )
    )

    assert screen._root is not None
    assert screen._dataset is sample_dataset
    assert screen._count_nodes(screen._root) == 3


def test_tree_viewer_matches_orange_sidebar_structure(app, sample_dataset):
    screen = TreeViewerScreen()
    screen.set_tree_data(
        {
            "label": "petal length",
            "children": [
                {"label": "Left", "prediction": "setosa", "distribution": [2, 0, 0], "row_indices": [0, 1]},
                {"label": "Right", "prediction": "virginica", "distribution": [0, 1, 2], "row_indices": [2, 3, 4]},
            ],
        },
        sample_dataset,
    )

    assert screen._infolabel.text() == "3 nodes, 2 leaves"
    assert screen._color_label.text() == "Target class:"
    assert screen._depth_combo.itemText(0) == "Unlimited"
    assert [screen._edge_width_combo.itemText(i) for i in range(screen._edge_width_combo.count())] == [
        "Fixed",
        "Relative to root",
        "Relative to parent",
    ]
    assert screen._details_cb.text() == "Show details in non-leaves"
    color_items = [screen._color_combo.itemText(i) for i in range(screen._color_combo.count())]
    assert color_items[0] == "None"
    assert "Prediction" not in color_items


def test_tree_viewer_accepts_mock_tree_and_emits_selected_output(app, sample_dataset):
    screen = TreeViewerScreen()
    screen.set_tree_data(
        {
            "label": "Root",
            "summary": "5 rows",
            "children": [
                {"label": "Left", "row_indices": [0, 1]},
                {"label": "Right", "row_indices": [2, 3, 4]},
            ],
        },
        sample_dataset,
    )

    screen._handle_selection_changed([2, 3, 4])

    selected = screen.current_output_dataset()
    annotated = screen.current_output_datasets()["Annotated Data"]
    assert selected is not None
    assert selected.row_count == 3
    assert annotated is not None
    assert annotated.annotations["selected_rows"] == [2, 3, 4]


def test_main_window_can_open_orange_first_six_visualize_widgets(app):
    window = MainWindow()
    window.show()
    QTest.qWait(50)

    for widget_id in (
        "tree-viewer",
        "box-plot",
        "violin-plot",
        "distributions",
        "scatter-plot",
        "line-plot",
    ):
        window._workspace.canvas.add_workflow_node(widget_id)
        node_id = next(
            record.node_id
            for record in window._workspace.canvas.workflow_scene.node_records()
            if record.widget_id == widget_id
        )
        window._show_widget(node_id)

    assert len(window._workspace.visible_dialogs()) >= 6
