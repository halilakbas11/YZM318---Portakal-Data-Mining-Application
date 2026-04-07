from __future__ import annotations

import os

import polars as pl
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication

from portakal_app.data.services.generated_dataset_service import GeneratedDatasetService
from portakal_app.data.services.logistic_regression_service import LogisticRegressionService
from portakal_app.logistic_regression_artifacts import LogisticRegressionClassifierArtifact
from portakal_app.models import WorkflowPayload, workflow_ports_are_compatible
from portakal_app.ui.catalog import build_widgets
from portakal_app.ui.main_window import MainWindow
from portakal_app.ui.screens.logistic_regression_screen import LogisticRegressionScreen
from portakal_app.ui.screens.nomogram_screen import NomogramScreen


@pytest.fixture(scope="session")
def app():
    return QApplication.instance() or QApplication([])


def _build_dataset(service: GeneratedDatasetService):
    return service.build_dataset(
        pl.DataFrame(
            {
                "age": [22, 24, 27, 29, 33, 36, 38, 41],
                "chol": [180, 190, 175, 200, 205, 210, 195, 230],
                "sex": ["F", "F", "F", "F", "M", "M", "M", "M"],
                "survived": ["yes", "yes", "yes", "yes", "no", "no", "no", "no"],
            }
        ),
        dataset_id="nomogram_sample",
        display_name="Nomogram Sample",
        file_name="nomogram_sample.csv",
        role_overrides={"age": "feature", "chol": "feature", "sex": "feature", "survived": "target"},
    )


def _build_classifier(dataset) -> LogisticRegressionClassifierArtifact:
    return LogisticRegressionService().fit(dataset)


def test_catalog_registers_nomogram_and_logistic_regression():
    widgets = {widget.id: widget for widget in build_widgets()}
    assert "nomogram" in widgets
    assert "logistic-regression" in widgets
    assert widgets["nomogram"].enabled is True
    assert widgets["nomogram"].input_ports[0].label == "Classifier"
    assert widgets["nomogram"].input_ports[1].label == "Data"
    assert widgets["nomogram"].output_ports[0].label == "Features"
    assert widgets["logistic-regression"].output_ports[0].label == "Classifier"
    assert workflow_ports_are_compatible("nomogram", "Features", "data-table", "Data") is True


def test_main_window_can_open_nomogram_and_logistic_regression(app):
    window = MainWindow()
    window._workspace.canvas.add_workflow_node("nomogram")
    window._show_widget("nomogram")
    assert isinstance(window._workspace.current_widget(), NomogramScreen)

    window._workspace.canvas.add_workflow_node("logistic-regression")
    window._show_widget("logistic-regression")
    assert isinstance(window._workspace.current_widget(), LogisticRegressionScreen)


def test_logistic_regression_screen_builds_classifier(app):
    dataset = _build_dataset(GeneratedDatasetService())
    screen = LogisticRegressionScreen()
    screen.set_input_payload(WorkflowPayload("Data", dataset))
    app.processEvents()

    classifier = screen.current_output_dataset()
    assert isinstance(classifier, LogisticRegressionClassifierArtifact)
    assert classifier.target_name == "survived"
    assert classifier.class_values == ("yes", "no")
    assert [feature.name for feature in classifier.features] == ["age", "chol", "sex"]
    assert classifier.can_apply_to(dataset) is True


def test_nomogram_outputs_features_dataset(app):
    dataset = _build_dataset(GeneratedDatasetService())
    classifier = _build_classifier(dataset)
    screen = NomogramScreen()
    screen.set_input_payload(None)
    screen.set_input_payload(WorkflowPayload("Classifier", classifier))
    app.processEvents()

    output = screen.current_output_dataset()
    assert output is not None
    assert output.row_count == len(classifier.features)
    assert output.dataframe["Feature"].to_list() == ["age", "chol", "sex"]
    assert screen._target_combo.count() == 2


def test_nomogram_best_ranked_limits_output_rows(app):
    dataset = _build_dataset(GeneratedDatasetService())
    classifier = _build_classifier(dataset)
    screen = NomogramScreen()
    screen.set_input_payload(WorkflowPayload("Classifier", classifier))
    app.processEvents()

    screen._display_combo.setCurrentIndex(1)
    screen._best_n_spin.setValue(2)
    app.processEvents()

    output = screen.current_output_dataset()
    assert output is not None
    assert output.row_count == 2


def test_nomogram_seeds_markers_from_first_input_row_and_updates_footer(app):
    dataset = _build_dataset(GeneratedDatasetService())
    classifier = _build_classifier(dataset)
    screen = NomogramScreen()
    screen.set_input_payload(None)
    screen.set_input_payload(WorkflowPayload("Classifier", classifier))
    screen.set_input_payload(WorkflowPayload("Data", dataset))
    app.processEvents()

    first_row = dataset.dataframe.row(0, named=True)
    assert screen._marker_values["age"] == first_row["age"]
    assert screen._marker_values["chol"] == first_row["chol"]
    assert screen._marker_values["sex"] == first_row["sex"]

    before = screen._footer_widget._state.current_probability
    screen._on_row_value_changed("age", 41.0)
    app.processEvents()
    after = screen._footer_widget._state.current_probability

    assert before != after
