from __future__ import annotations

import os

import polars as pl
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PySide6")

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication

from portakal_app.data.services.generated_dataset_service import GeneratedDatasetService
from portakal_app.data.services.scoring_sheet_service import ScoringSheetService
from portakal_app.models import WorkflowPayload
from portakal_app.scoring_sheet_artifacts import ScoringSheetClassifierArtifact
from portakal_app.ui.catalog import build_widgets
from portakal_app.ui.main_window import MainWindow
from portakal_app.ui.screens.scoring_sheet_viewer_screen import ScoringSheetViewerScreen


@pytest.fixture(scope="session")
def app():
    return QApplication.instance() or QApplication([])


def _build_dataset(service: GeneratedDatasetService):
    return service.build_dataset(
        pl.DataFrame(
            {
                "age": [22, 24, 27, 29, 33, 36, 38, 41],
                "sex": ["F", "F", "F", "F", "M", "M", "M", "M"],
                "smoker": ["no", "no", "no", "yes", "yes", "yes", "yes", "yes"],
                "risk": ["low", "low", "low", "low", "high", "high", "high", "high"],
            }
        ),
        dataset_id="scoring_sheet_sample",
        display_name="Scoring Sheet Sample",
        file_name="scoring_sheet_sample.csv",
        role_overrides={"age": "feature", "sex": "feature", "smoker": "feature", "risk": "target"},
    )


def _build_classifier(dataset) -> ScoringSheetClassifierArtifact:
    return ScoringSheetService().fit(dataset)


def test_catalog_registers_scoring_sheet_widgets():
    widgets = {widget.id: widget for widget in build_widgets()}
    assert "scoring-sheet-viewer" in widgets
    assert widgets["scoring-sheet-viewer"].input_ports[0].label == "Classifier"
    assert widgets["scoring-sheet-viewer"].input_ports[1].label == "Data"
    assert widgets["scoring-sheet-viewer"].output_ports[0].label == "Features"


def test_main_window_can_open_scoring_sheet_viewer(app):
    window = MainWindow()
    window._workspace.canvas.add_workflow_node("scoring-sheet-viewer")
    window._show_widget("scoring-sheet-viewer")
    assert isinstance(window._workspace.current_widget(), ScoringSheetViewerScreen)


def test_scoring_sheet_viewer_populates_table_slider_and_output(app):
    dataset = _build_dataset(GeneratedDatasetService())
    classifier = _build_classifier(dataset)
    screen = ScoringSheetViewerScreen()
    screen.set_input_payload(None)
    screen.set_input_payload(WorkflowPayload("Classifier", classifier))
    app.processEvents()

    assert screen._coefficient_table.rowCount() == len(classifier.rules)
    assert len(screen._risk_slider.points) == len(screen._risk_slider.probabilities)
    assert screen._class_combo.count() == 2
    output = screen.current_output_dataset()
    assert output is not None
    assert output.row_count == len(classifier.rules)


def test_scoring_sheet_viewer_slider_updates_on_checkbox_toggle(app):
    dataset = _build_dataset(GeneratedDatasetService())
    classifier = _build_classifier(dataset)
    screen = ScoringSheetViewerScreen()
    screen.set_classifier(classifier)
    app.processEvents()

    zero_index = screen._risk_slider.points.index(0.0)
    assert screen._risk_slider.slider.value() == zero_index

    checkbox_item = screen._coefficient_table.item(0, 2)
    coefficient = float(screen._coefficient_table.item(0, 1).text())
    checkbox_item.setCheckState(Qt.CheckState.Checked)
    app.processEvents()

    assert screen._risk_slider.points[screen._risk_slider.slider.value()] == coefficient


def test_scoring_sheet_viewer_target_class_change_flips_coefficients(app):
    dataset = _build_dataset(GeneratedDatasetService())
    classifier = _build_classifier(dataset)
    screen = ScoringSheetViewerScreen()
    screen.set_classifier(classifier)
    app.processEvents()

    original = list(screen._coefficients)
    screen._class_combo.setCurrentIndex(1)
    app.processEvents()
    flipped = list(screen._coefficients)

    assert original != flipped


def test_scoring_sheet_viewer_marks_first_instance_rules(app):
    dataset = _build_dataset(GeneratedDatasetService())
    classifier = _build_classifier(dataset)
    screen = ScoringSheetViewerScreen()
    screen.set_classifier(classifier)
    screen.set_data(dataset)
    app.processEvents()

    expected = classifier.first_row_matches(dataset)
    actual = [
        1 if screen._coefficient_table.item(row, 2).checkState() == Qt.CheckState.Checked else 0
        for row in range(screen._coefficient_table.rowCount())
    ]
    assert actual == expected


def test_scoring_sheet_viewer_rejects_invalid_classifier(app):
    screen = ScoringSheetViewerScreen()
    screen.set_classifier(object())
    app.processEvents()

    assert screen.current_output_dataset() is None
    assert "only accepts" in screen._status_label.text().lower()
