from __future__ import annotations

import math
import os

import polars as pl
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication

from portakal_app.data.services.generated_dataset_service import GeneratedDatasetService
from portakal_app.models import WorkflowPayload
from portakal_app.tree_artifacts import DecisionTreeArtifact, TreeNodeArtifact
from portakal_app.ui.catalog import build_widgets
from portakal_app.ui.main_window import MainWindow
from portakal_app.ui.screens.pythagorean_tree_screen import Point, PythagoreanTreeScreen, Square, _PythagorasLayout


@pytest.fixture(scope="session")
def app():
    return QApplication.instance() or QApplication([])


def _build_dataset(
    service: GeneratedDatasetService,
    dataframe: pl.DataFrame,
    dataset_id: str,
    *,
    role_overrides: dict[str, str],
):
    return service.build_dataset(
        dataframe,
        dataset_id=dataset_id,
        display_name=dataset_id.upper(),
        file_name=f"{dataset_id}.csv",
        role_overrides=role_overrides,
    )


def _build_classification_tree(service: GeneratedDatasetService) -> DecisionTreeArtifact:
    dataset = _build_dataset(
        service,
        pl.DataFrame(
            {
                "x": [0.1, 0.4, 1.0, 2.4, 2.8, 3.1],
                "class": ["A", "A", "A", "B", "B", "B"],
                "label": ["r1", "r2", "r3", "r4", "r5", "r6"],
            }
        ),
        "pytree_classification",
        role_overrides={"x": "feature", "class": "target", "label": "meta"},
    )
    nodes = {
        0: TreeNodeArtifact(0, [0, 1, 2, 3, 4, 5], [1, 2], "x", (), "", (3, 3), "A"),
        1: TreeNodeArtifact(1, [0, 1, 2], [3, 4], "x", ("x <= 1.5",), "<= 1.5", (3, 0), "A"),
        2: TreeNodeArtifact(2, [3, 4, 5], [5, 6], "x", ("x > 1.5",), "> 1.5", (0, 3), "B"),
        3: TreeNodeArtifact(3, [0, 1], [], None, ("x <= 0.5",), "<= 0.5", (2, 0), "A"),
        4: TreeNodeArtifact(4, [2], [], None, ("0.5 < x <= 1.5",), "> 0.5", (1, 0), "A"),
        5: TreeNodeArtifact(5, [3], [], None, ("1.5 < x <= 2.6",), "<= 2.6", (0, 1), "B"),
        6: TreeNodeArtifact(6, [4, 5], [], None, ("x > 2.6",), "> 2.6", (0, 2), "B"),
    }
    return DecisionTreeArtifact(
        tree_id="classification-tree",
        display_name="Classification Tree",
        instances=dataset,
        root_id=0,
        nodes=nodes,
        target_name="class",
        kind="classification",
        class_values=("A", "B"),
    )


def _build_regression_tree(service: GeneratedDatasetService) -> DecisionTreeArtifact:
    dataset = _build_dataset(
        service,
        pl.DataFrame(
            {
                "x": [0.1, 0.4, 1.0, 2.4, 2.8, 3.1],
                "y": [1.0, 1.5, 2.0, 5.2, 5.6, 6.0],
                "label": ["r1", "r2", "r3", "r4", "r5", "r6"],
            }
        ),
        "pytree_regression",
        role_overrides={"x": "feature", "y": "target", "label": "meta"},
    )
    nodes = {
        0: TreeNodeArtifact(0, [0, 1, 2, 3, 4, 5], [1, 2], "x", (), "", (), 3.55),
        1: TreeNodeArtifact(1, [0, 1, 2], [3, 4], "x", ("x <= 1.5",), "<= 1.5", (), 1.5),
        2: TreeNodeArtifact(2, [3, 4, 5], [5, 6], "x", ("x > 1.5",), "> 1.5", (), 5.6),
        3: TreeNodeArtifact(3, [0, 1], [], None, ("x <= 0.5",), "<= 0.5", (), 1.25),
        4: TreeNodeArtifact(4, [2], [], None, ("0.5 < x <= 1.5",), "> 0.5", (), 2.0),
        5: TreeNodeArtifact(5, [3], [], None, ("1.5 < x <= 2.6",), "<= 2.6", (), 5.2),
        6: TreeNodeArtifact(6, [4, 5], [], None, ("x > 2.6",), "> 2.6", (), 5.8),
    }
    return DecisionTreeArtifact(
        tree_id="regression-tree",
        display_name="Regression Tree",
        instances=dataset,
        root_id=0,
        nodes=nodes,
        target_name="y",
        kind="regression",
    )


def test_catalog_registers_pythagorean_tree_widget():
    widgets = {widget.id: widget for widget in build_widgets()}
    assert "pythagorean-tree" in widgets
    assert widgets["pythagorean-tree"].enabled is True
    assert widgets["pythagorean-tree"].input_ports[0].label == "Tree"
    assert widgets["pythagorean-tree"].output_channels == ("Selected Data", "Annotated Data")


def test_main_window_can_open_pythagorean_tree_widget(app):
    window = MainWindow()
    window._workspace.canvas.add_workflow_node("pythagorean-tree")
    window._show_widget("pythagorean-tree")

    assert isinstance(window._workspace.current_widget(), PythagoreanTreeScreen)
    assert window._widget_index["pythagorean-tree"].enabled is True


def test_pythagoras_layout_matches_orange_reference_geometry():
    builder = _PythagorasLayout()

    point = builder._get_point_on_square_edge(center=Point(2.7, 2.77), length=1.65, angle=math.radians(20.97))
    assert point.x == pytest.approx(3.48, abs=0.1)
    assert point.y == pytest.approx(3.07, abs=0.1)

    initial_square = Square(Point(1.5, 1.5), length=2.24, angle=math.radians(63.43))
    center = builder._compute_center(initial_square, length=1.65, alpha=math.radians(95.06))
    assert center.x == pytest.approx(3.48, abs=0.1)
    assert center.y == pytest.approx(3.07, abs=0.1)


def test_pythagorean_tree_outputs_selected_and_annotated_data(app):
    service = GeneratedDatasetService()
    tree = _build_classification_tree(service)
    screen = PythagoreanTreeScreen()
    screen.set_input_payload(WorkflowPayload("Tree", tree))
    app.processEvents()

    screen._items[1].setSelected(True)
    app.processEvents()

    outputs = screen.current_output_datasets()
    selected = outputs["Selected Data"]
    annotated = outputs["Annotated Data"]

    assert selected is not None
    assert annotated is not None
    assert selected.row_count == 3
    assert annotated.dataframe["Selected"].to_list() == [True, True, True, False, False, False]


def test_pythagorean_tree_depth_limit_hides_nodes(app):
    service = GeneratedDatasetService()
    tree = _build_classification_tree(service)
    screen = PythagoreanTreeScreen()
    screen.set_input_payload(WorkflowPayload("Tree", tree))
    app.processEvents()

    assert len(screen._items) == 7

    screen.depth_slider.setValue(1)
    app.processEvents()
    assert len(screen._items) == 3

    screen.depth_slider.setValue(2)
    app.processEvents()
    assert len(screen._items) == 7


def test_pythagorean_tree_target_class_changes_colors(app):
    service = GeneratedDatasetService()
    tree = _build_classification_tree(service)
    screen = PythagoreanTreeScreen()
    screen.set_input_payload(WorkflowPayload("Tree", tree))
    app.processEvents()

    before = [screen._items[node_id].brush().color().name() for node_id in sorted(screen._items)]
    screen.target_class_combo.setCurrentIndex(1)
    app.processEvents()
    after = [screen._items[node_id].brush().color().name() for node_id in sorted(screen._items)]

    assert any(left != right for left, right in zip(before, after))


def test_pythagorean_tree_regression_legend_visibility(app):
    service = GeneratedDatasetService()
    tree = _build_regression_tree(service)
    screen = PythagoreanTreeScreen()
    screen.show()
    screen.set_input_payload(WorkflowPayload("Tree", tree))
    app.processEvents()

    screen.cb_show_legend.setChecked(True)
    app.processEvents()
    assert screen._legend.isVisible() is False

    screen.target_class_combo.setCurrentIndex(1)
    app.processEvents()
    assert screen._legend.isVisible() is True


def test_pythagorean_tree_reverses_subtree_order(app):
    service = GeneratedDatasetService()
    tree = _build_classification_tree(service)
    screen = PythagoreanTreeScreen()
    screen.set_input_payload(WorkflowPayload("Tree", tree))
    app.processEvents()

    before_x = {
        1: screen._items[1].sceneBoundingRect().center().x(),
        2: screen._items[2].sceneBoundingRect().center().x(),
    }
    assert before_x[1] < before_x[2]

    screen._reverse_subtree(0)
    app.processEvents()

    after_x = {
        1: screen._items[1].sceneBoundingRect().center().x(),
        2: screen._items[2].sceneBoundingRect().center().x(),
    }
    assert tree.children(0) == [2, 1]
    assert after_x[1] > after_x[2]
