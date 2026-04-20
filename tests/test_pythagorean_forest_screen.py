from __future__ import annotations

import os

import polars as pl
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication

from portakal_app.data.services.generated_dataset_service import GeneratedDatasetService
from portakal_app.models import WorkflowPayload
from portakal_app.tree_artifacts import DecisionTreeArtifact, RandomForestArtifact, TreeNodeArtifact
from portakal_app.ui.catalog import build_widgets
from portakal_app.ui.main_window import MainWindow
from portakal_app.ui.screens.pythagorean_forest_screen import PythagoreanForestScreen
from portakal_app.ui.screens.pythagorean_tree_screen import PythagoreanTreeScreen


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


def _classification_tree(dataset, tree_id: str, left_bias: bool) -> DecisionTreeArtifact:
    if left_bias:
        root_dist = (4, 2)
        left_dist = (3, 0)
        right_dist = (1, 2)
    else:
        root_dist = (2, 4)
        left_dist = (0, 3)
        right_dist = (2, 1)

    nodes = {
        0: TreeNodeArtifact(0, [0, 1, 2, 3, 4, 5], [1, 2], "x", (), "", root_dist, "A"),
        1: TreeNodeArtifact(1, [0, 1, 2], [3, 4], "x", ("x <= 1.5",), "<= 1.5", left_dist, "A"),
        2: TreeNodeArtifact(2, [3, 4, 5], [5, 6], "x", ("x > 1.5",), "> 1.5", right_dist, "B"),
        3: TreeNodeArtifact(3, [0, 1], [], None, ("x <= 0.5",), "<= 0.5", (2, 0), "A"),
        4: TreeNodeArtifact(4, [2], [], None, ("0.5 < x <= 1.5",), "> 0.5", (1, 0), "A"),
        5: TreeNodeArtifact(5, [3], [], None, ("1.5 < x <= 2.6",), "<= 2.6", (0, 1), "B"),
        6: TreeNodeArtifact(6, [4, 5], [], None, ("x > 2.6",), "> 2.6", (0, 2), "B"),
    }
    return DecisionTreeArtifact(
        tree_id=tree_id,
        display_name=tree_id,
        instances=dataset,
        root_id=0,
        nodes=nodes,
        target_name="class",
        kind="classification",
        class_values=("A", "B"),
    )


def _regression_tree(dataset, tree_id: str, offset: float) -> DecisionTreeArtifact:
    nodes = {
        0: TreeNodeArtifact(0, [0, 1, 2, 3, 4, 5], [1, 2], "x", (), "", (), 3.55 + offset),
        1: TreeNodeArtifact(1, [0, 1, 2], [3, 4], "x", ("x <= 1.5",), "<= 1.5", (), 1.5 + offset),
        2: TreeNodeArtifact(2, [3, 4, 5], [5, 6], "x", ("x > 1.5",), "> 1.5", (), 5.6 + offset),
        3: TreeNodeArtifact(3, [0, 1], [], None, ("x <= 0.5",), "<= 0.5", (), 1.25 + offset),
        4: TreeNodeArtifact(4, [2], [], None, ("0.5 < x <= 1.5",), "> 0.5", (), 2.0 + offset),
        5: TreeNodeArtifact(5, [3], [], None, ("1.5 < x <= 2.6",), "<= 2.6", (), 5.2 + offset),
        6: TreeNodeArtifact(6, [4, 5], [], None, ("x > 2.6",), "> 2.6", (), 5.8 + offset),
    }
    return DecisionTreeArtifact(
        tree_id=tree_id,
        display_name=tree_id,
        instances=dataset,
        root_id=0,
        nodes=nodes,
        target_name="y",
        kind="regression",
    )


def _classification_forest(service: GeneratedDatasetService) -> RandomForestArtifact:
    dataset = _build_dataset(
        service,
        pl.DataFrame(
            {
                "x": [0.1, 0.4, 1.0, 2.4, 2.8, 3.1],
                "class": ["A", "A", "A", "B", "B", "B"],
                "label": ["r1", "r2", "r3", "r4", "r5", "r6"],
            }
        ),
        "forest_classification",
        role_overrides={"x": "feature", "class": "target", "label": "meta"},
    )
    trees = [
        _classification_tree(dataset, "tree-1", True),
        _classification_tree(dataset, "tree-2", False),
        _classification_tree(dataset, "tree-3", True),
    ]
    return RandomForestArtifact(
        forest_id="classification-forest",
        display_name="Classification Forest",
        instances=dataset,
        trees=trees,
        kind="classification",
        class_values=("A", "B"),
    )


def _regression_forest(service: GeneratedDatasetService) -> RandomForestArtifact:
    dataset = _build_dataset(
        service,
        pl.DataFrame(
            {
                "x": [0.1, 0.4, 1.0, 2.4, 2.8, 3.1],
                "y": [1.0, 1.5, 2.0, 5.2, 5.6, 6.0],
                "label": ["r1", "r2", "r3", "r4", "r5", "r6"],
            }
        ),
        "forest_regression",
        role_overrides={"x": "feature", "y": "target", "label": "meta"},
    )
    trees = [
        _regression_tree(dataset, "rtree-1", 0.0),
        _regression_tree(dataset, "rtree-2", 0.4),
        _regression_tree(dataset, "rtree-3", -0.3),
    ]
    return RandomForestArtifact(
        forest_id="regression-forest",
        display_name="Regression Forest",
        instances=dataset,
        trees=trees,
        kind="regression",
    )


def test_catalog_registers_pythagorean_forest_widget():
    widgets = {widget.id: widget for widget in build_widgets()}
    assert "pythagorean-forest" in widgets
    assert widgets["pythagorean-forest"].enabled is True
    assert widgets["pythagorean-forest"].input_ports[0].label == "Random Forest"
    assert widgets["pythagorean-forest"].output_ports[0].label == "Tree"


def test_main_window_can_open_pythagorean_forest_widget(app):
    window = MainWindow()
    window._workspace.canvas.add_workflow_node("pythagorean-forest")
    window._show_widget("pythagorean-forest")

    assert isinstance(window._workspace.current_widget(), PythagoreanForestScreen)
    assert window._widget_index["pythagorean-forest"].enabled is True


def test_pythagorean_forest_draws_tree_previews_and_clears_on_none(app):
    service = GeneratedDatasetService()
    forest = _classification_forest(service)
    screen = PythagoreanForestScreen()
    screen.set_input_payload(WorkflowPayload("Random Forest", forest))
    app.processEvents()

    assert len(screen._cards) == 3
    assert "Trees: 3" in screen.ui_info.text()

    screen.set_input_payload(None)
    app.processEvents()
    assert len(screen._cards) == 0


def test_pythagorean_forest_target_class_changes_preview_colors(app):
    service = GeneratedDatasetService()
    forest = _classification_forest(service)
    screen = PythagoreanForestScreen()
    screen.set_input_payload(WorkflowPayload("Random Forest", forest))
    app.processEvents()

    before = screen._cards[0].square_colors()
    screen.ui_target_class_combo.setCurrentIndex(1)
    app.processEvents()
    after = screen._cards[0].square_colors()

    assert any(left != right for left, right in zip(before, after))


def test_pythagorean_forest_size_adjustment_changes_preview_geometry(app):
    service = GeneratedDatasetService()
    forest = _classification_forest(service)
    screen = PythagoreanForestScreen()
    screen.set_input_payload(WorkflowPayload("Random Forest", forest))
    app.processEvents()

    before = screen._cards[0].square_rects()
    screen.ui_size_calc_combo.setCurrentIndex(1)
    app.processEvents()
    after = screen._cards[0].square_rects()

    assert any(left != right for left, right in zip(before, after))


def test_pythagorean_forest_zoom_updates_item_size(app):
    service = GeneratedDatasetService()
    forest = _classification_forest(service)
    screen = PythagoreanForestScreen()
    screen.set_input_payload(WorkflowPayload("Random Forest", forest))
    app.processEvents()

    before = screen._list_widget.item(0).sizeHint()
    screen.ui_zoom_slider.setValue(320)
    app.processEvents()
    after = screen._list_widget.item(0).sizeHint()

    assert after.width() > before.width()
    assert after.height() > before.height()


def test_pythagorean_forest_outputs_tree_with_visual_settings(app):
    service = GeneratedDatasetService()
    forest = _classification_forest(service)
    screen = PythagoreanForestScreen()
    screen.set_input_payload(WorkflowPayload("Random Forest", forest))
    app.processEvents()

    screen.ui_target_class_combo.setCurrentIndex(1)
    screen.ui_size_calc_combo.setCurrentIndex(2)
    screen.ui_depth_slider.setValue(1)
    screen._list_widget.setCurrentRow(1)
    app.processEvents()

    selected_tree = screen.current_output_dataset()
    assert isinstance(selected_tree, DecisionTreeArtifact)
    assert selected_tree.tree_id == "tree-2"
    assert selected_tree.meta_target_class_index == 1
    assert selected_tree.meta_size_calc_idx == 2
    assert selected_tree.meta_depth_limit == 1

    tree_screen = PythagoreanTreeScreen()
    tree_screen.set_input_payload(WorkflowPayload("Tree", selected_tree))
    app.processEvents()

    assert tree_screen.target_class_combo.currentIndex() == 1
    assert tree_screen.size_calc_combo.currentIndex() == 2
    assert tree_screen.depth_slider.value() == 1


def test_pythagorean_forest_regression_colors_change(app):
    service = GeneratedDatasetService()
    forest = _regression_forest(service)
    screen = PythagoreanForestScreen()
    screen.set_input_payload(WorkflowPayload("Random Forest", forest))
    app.processEvents()

    before = screen._cards[0].square_colors()
    screen.ui_target_class_combo.setCurrentIndex(1)
    app.processEvents()
    after = screen._cards[0].square_colors()

    assert any(left != right for left, right in zip(before, after))


def test_main_window_handles_non_tabular_tree_output_preview(app):
    service = GeneratedDatasetService()
    forest = _classification_forest(service)
    window = MainWindow()
    record = window._workspace.canvas.add_workflow_node("pythagorean-forest")
    window._sync_node_runtimes()
    runtime = window._node_runtimes[record.node_id]
    runtime.screen.set_input_payload(WorkflowPayload("Random Forest", forest))
    runtime.screen._list_widget.setCurrentRow(0)
    app.processEvents()

    tree = runtime.screen.current_output_dataset()
    assert tree is not None
    runtime.output_payload = WorkflowPayload("Tree", tree)
    snapshot = window._node_data_preview_snapshot(record.node_id)

    assert snapshot["headers"] == []
    assert "No tabular preview available" in snapshot["summary"]
