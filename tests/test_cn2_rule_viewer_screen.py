from __future__ import annotations

import os

import polars as pl
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PySide6")

from PySide6.QtCore import QItemSelectionModel
from PySide6.QtWidgets import QApplication

from portakal_app.data.services.cn2_rule_induction_service import CN2InductionSettings, CN2RuleInductionService
from portakal_app.data.services.generated_dataset_service import GeneratedDatasetService
from portakal_app.models import WorkflowPayload
from portakal_app.rule_artifacts import CN2RuleArtifact, CN2RuleClassifierArtifact, RuleConditionArtifact
from portakal_app.ui.catalog import build_widgets
from portakal_app.ui.main_window import MainWindow
from portakal_app.ui.screens.cn2_rule_induction_screen import CN2RuleInductionScreen
from portakal_app.ui.screens.cn2_rule_viewer_screen import CN2RuleViewerScreen


@pytest.fixture(scope="session")
def app():
    return QApplication.instance() or QApplication([])


def _build_dataset(service: GeneratedDatasetService):
    return service.build_dataset(
        pl.DataFrame(
            {
                "gender": ["F", "F", "F", "M", "M", "M"],
                "segment": ["A", "A", "B", "B", "C", "C"],
                "survived": ["yes", "yes", "yes", "no", "no", "no"],
            }
        ),
        dataset_id="cn2_rules",
        display_name="CN2 Rules",
        file_name="cn2_rules.csv",
        role_overrides={"gender": "feature", "segment": "feature", "survived": "target"},
    )


def _build_classifier(dataset) -> CN2RuleClassifierArtifact:
    return CN2RuleClassifierArtifact(
        classifier_id="classifier-1",
        display_name="CN2 Rules",
        instances=dataset,
        original_feature_names=("gender", "segment"),
        target_name="survived",
        class_values=("yes", "no"),
        rule_list=[
            CN2RuleArtifact(
                target_name="survived",
                prediction="yes",
                selectors=(RuleConditionArtifact("gender", "==", "F"),),
                curr_class_dist=(3.0, 0.0),
                probabilities=(1.0, 0.0),
                quality=0.5,
                covered_count=3,
                learning_covered_indices=(0, 1, 2),
            ),
            CN2RuleArtifact(
                target_name="survived",
                prediction="no",
                selectors=(RuleConditionArtifact("gender", "==", "M"),),
                curr_class_dist=(0.0, 3.0),
                probabilities=(0.0, 1.0),
                quality=0.5,
                covered_count=3,
                learning_covered_indices=(3, 4, 5),
            ),
            CN2RuleArtifact(
                target_name="survived",
                prediction="yes",
                selectors=(),
                curr_class_dist=(3.0, 3.0),
                probabilities=(0.5, 0.5),
                quality=0.0,
                covered_count=6,
                learning_covered_indices=(0, 1, 2, 3, 4, 5),
                is_default=True,
            ),
        ],
    )


def test_catalog_registers_cn2_widgets():
    widgets = {widget.id: widget for widget in build_widgets()}
    assert "cn2-rule-viewer" in widgets
    assert "cn2-rule-induction" in widgets
    assert widgets["cn2-rule-viewer"].enabled is True
    assert widgets["cn2-rule-viewer"].input_ports[0].label == "Data"
    assert widgets["cn2-rule-viewer"].input_ports[1].label == "Classifier"
    assert widgets["cn2-rule-viewer"].output_channels == ("Selected Data", "Annotated Data")
    assert widgets["cn2-rule-induction"].output_ports[0].label == "Classifier"


def test_main_window_can_open_cn2_rule_viewer(app):
    window = MainWindow()
    window._workspace.canvas.add_workflow_node("cn2-rule-viewer")
    window._show_widget("cn2-rule-viewer")

    assert isinstance(window._workspace.current_widget(), CN2RuleViewerScreen)


def test_cn2_rule_viewer_outputs_selected_and_annotated_rows(app):
    dataset = _build_dataset(GeneratedDatasetService())
    classifier = _build_classifier(dataset)
    screen = CN2RuleViewerScreen()
    screen.set_input_payload(None)
    screen.set_input_payload(WorkflowPayload("Data", dataset))
    screen.set_input_payload(WorkflowPayload("Classifier", classifier))
    app.processEvents()

    selection_model = screen._view.selectionModel()
    selection_model.select(
        screen._proxy_model.index(0, 0),
        QItemSelectionModel.SelectionFlag.Select | QItemSelectionModel.SelectionFlag.Rows,
    )
    app.processEvents()

    outputs = screen.current_output_datasets()
    selected = outputs["Selected Data"]
    annotated = outputs["Annotated Data"]

    assert selected is not None
    assert annotated is not None
    assert selected.row_count == 3
    assert selected.dataframe["gender"].to_list() == ["F", "F", "F"]
    assert annotated.dataframe["Selected"].to_list() == [True, True, True, False, False, False]


def test_cn2_rule_viewer_without_classifier_still_annotates_input(app):
    dataset = _build_dataset(GeneratedDatasetService())
    screen = CN2RuleViewerScreen()
    screen.set_input_payload(None)
    screen.set_input_payload(WorkflowPayload("Data", dataset))
    app.processEvents()

    outputs = screen.current_output_datasets()
    assert outputs["Selected Data"] is None
    assert outputs["Annotated Data"] is not None
    assert outputs["Annotated Data"].dataframe["Selected"].to_list() == [False, False, False, False, False, False]


def test_cn2_rule_viewer_preserves_selection_in_compact_mode(app):
    dataset = _build_dataset(GeneratedDatasetService())
    classifier = _build_classifier(dataset)
    screen = CN2RuleViewerScreen()
    screen.set_input_payload(None)
    screen.set_input_payload(WorkflowPayload("Classifier", classifier))
    app.processEvents()

    selection_model = screen._view.selectionModel()
    selection_model.select(
        screen._proxy_model.index(1, 0),
        QItemSelectionModel.SelectionFlag.Select | QItemSelectionModel.SelectionFlag.Rows,
    )
    app.processEvents()
    screen._compact_checkbox.setChecked(True)
    app.processEvents()

    assert screen._selected_rule_rows == [1]


def test_cn2_rule_induction_screen_builds_classifier(app):
    dataset = _build_dataset(GeneratedDatasetService())
    screen = CN2RuleInductionScreen()
    screen.set_input_payload(WorkflowPayload("Data", dataset))
    app.processEvents()

    classifier = screen.current_output_dataset()
    assert isinstance(classifier, CN2RuleClassifierArtifact)
    assert len(classifier.rule_list) >= 2
    assert classifier.rule_list[-1].is_default is True


def test_cn2_induction_service_generates_classifier_for_categorical_target():
    dataset = _build_dataset(GeneratedDatasetService())
    classifier = CN2RuleInductionService().induce(dataset, CN2InductionSettings(max_rules=4, min_covered_examples=2))

    assert classifier.target_name == "survived"
    assert classifier.class_values == ("yes", "no")
    assert classifier.rule_list[-1].is_default is True
