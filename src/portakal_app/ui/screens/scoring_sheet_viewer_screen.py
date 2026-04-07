from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl

from PySide6.QtCore import QRect, QSize, Qt, Signal
from PySide6.QtGui import QFontMetrics, QHelpEvent, QPainter, QPalette
from PySide6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QProxyStyle,
    QSlider,
    QStyle,
    QStyleOptionSlider,
    QTableWidget,
    QTableWidgetItem,
    QToolTip,
    QVBoxLayout,
    QWidget,
)

from portakal_app.data.models import DatasetHandle
from portakal_app.data.services.generated_dataset_service import GeneratedDatasetService
from portakal_app.models import WorkflowPayload
from portakal_app.scoring_sheet_artifacts import ScoringSheetClassifierArtifact
from portakal_app.ui import i18n
from portakal_app.ui.screens.node_screen import WorkflowNodeScreenSupport


class _ScoringSheetTable(QTableWidget):
    state_changed = Signal(int)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setColumnCount(3)
        self.setHorizontalHeaderLabels((i18n.t("Attribute Name"), i18n.t("Points"), i18n.t("Selected")))
        self.itemChanged.connect(self._handle_item_changed)

    def populate_table(self, attributes: list[str], coefficients: list[int]) -> None:
        self.blockSignals(True)
        self.setRowCount(len(attributes))
        for row, (attribute, coefficient) in enumerate(zip(attributes, coefficients)):
            self.setItem(row, 0, QTableWidgetItem(attribute))
            coefficient_item = QTableWidgetItem(str(coefficient))
            coefficient_item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            self.setItem(row, 1, coefficient_item)
            checkbox = QTableWidgetItem()
            checkbox.setCheckState(Qt.CheckState.Unchecked)
            self.setItem(row, 2, checkbox)
            for column in range(self.columnCount()):
                item = self.item(row, column)
                item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable & ~Qt.ItemFlag.ItemIsSelectable)
        self.blockSignals(False)
        self.resize_columns_to_contents()

    def set_checkbox_states(self, states: list[int]) -> None:
        self.blockSignals(True)
        for row in range(self.rowCount()):
            item = self.item(row, 2)
            if item is None:
                continue
            item.setCheckState(Qt.CheckState.Checked if row < len(states) and states[row] else Qt.CheckState.Unchecked)
        self.blockSignals(False)
        self.state_changed.emit(-1)

    def resize_columns_to_contents(self) -> None:
        for column in range(self.columnCount()):
            self.resizeColumnToContents(column)

    def _handle_item_changed(self, item: QTableWidgetItem) -> None:
        if item.column() == 2:
            self.state_changed.emit(item.row())


class _CustomSliderStyle(QProxyStyle):
    def drawComplexControl(self, cc, opt, painter, widget=None):
        if cc != QStyle.ComplexControl.CC_Slider:
            super().drawComplexControl(cc, opt, painter, widget)
            return

        slider_opt = QStyleOptionSlider(opt)
        slider_opt.subControls &= ~QStyle.SubControl.SC_SliderHandle
        super().drawComplexControl(cc, slider_opt, painter, widget)

        handle_rect = self.subControlRect(cc, opt, QStyle.SubControl.SC_SliderHandle, widget)
        painter.save()
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QPalette().color(QPalette.ColorRole.WindowText))
        height = handle_rect.height()
        painter.drawRoundedRect(
            QRect(handle_rect.center().x() - 2, handle_rect.y() + int(0.2 * height), 4, int(0.6 * height)),
            3,
            3,
        )
        painter.restore()


class _RiskSlider(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.points: list[float] = []
        self.probabilities: list[float] = []
        self.target_class = ""
        self.label_frequency = 1
        self.text_margin = 1

        layout = QHBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)

        label_layout = QVBoxLayout()
        self.points_label = QLabel(i18n.t("<b>Total:</b>"))
        label_layout.addWidget(self.points_label)
        label_layout.addSpacing(23)
        self.probability_label = QLabel(i18n.t("<b>Probabilities (%):</b>"))
        label_layout.addWidget(self.probability_label)
        layout.addLayout(label_layout)
        layout.addSpacing(28)

        self.slider = QSlider(Qt.Orientation.Horizontal, self)
        self.slider.setStyle(_CustomSliderStyle())
        self.slider.setEnabled(False)
        self.slider.installEventFilter(self)
        layout.addWidget(self.slider)
        self._setup_slider()

    def _setup_slider(self) -> None:
        self.slider.setMinimum(0)
        self.slider.setMaximum(len(self.points) - 1 if self.points else 0)
        self.slider.setTickPosition(QSlider.TickPosition.TicksBothSides)
        self.slider.setTickInterval(1)

    def set_values(self, points: list[float], probabilities: list[float], target_class: str) -> None:
        self.points = list(points)
        self.probabilities = list(probabilities)
        self.target_class = target_class
        self._setup_slider()
        self.update_label_frequency()
        self.update()

    def move_to_value(self, value: float) -> None:
        if not self.points:
            self.slider.setValue(0)
            return
        closest_index = min(range(len(self.points)), key=lambda index: abs(self.points[index] - value))
        self.slider.setValue(closest_index)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self.update_label_frequency()

    def update_label_frequency(self) -> None:
        total_width = self.slider.width()
        label_width = QFontMetrics(self.font()).boundingRect("100.0%").width()
        max_labels = max(1, total_width // max(label_width, 1))
        for frequency in (1, 2, 5, 10, 20, 50, 100):
            if max_labels >= max(1, len(self.points) / frequency):
                self.label_frequency = frequency
                break

    def paintEvent(self, event) -> None:
        super().paintEvent(event)
        if not self.points:
            return

        painter = QPainter(self)
        fm = QFontMetrics(painter.font())
        for index, point in enumerate(self.points):
            if index % self.label_frequency != 0:
                continue
            x_pos = QStyle.sliderPositionFromValue(self.slider.minimum(), self.slider.maximum(), index, self.slider.width()) + self.slider.x()

            point_text = str(int(point) if float(point).is_integer() else round(point, 2))
            point_rect = fm.boundingRect(point_text)
            point_x = int(x_pos - point_rect.width() / 2)
            point_y = int(self.slider.y() - self.text_margin - point_rect.height())
            painter.drawText(QRect(point_x, point_y, point_rect.width(), point_rect.height()), Qt.AlignmentFlag.AlignCenter, point_text)

            probability_text = f"{round(self.probabilities[index], 1)}%"
            prob_rect = fm.boundingRect(probability_text)
            prob_x = int(x_pos - prob_rect.width() / 2)
            prob_y = int(self.slider.y() + self.slider.height() + self.text_margin)
            painter.drawText(QRect(prob_x, prob_y, prob_rect.width(), prob_rect.height()), Qt.AlignmentFlag.AlignCenter, probability_text)
        painter.end()

    def eventFilter(self, watched, event) -> bool:
        if watched == self.slider and isinstance(event, QHelpEvent):
            self._handle_hover_event(event.pos())
            return True
        return super().eventFilter(watched, event)

    def _handle_hover_event(self, pos) -> None:
        thumb_rect = self._thumb_rect()
        if thumb_rect.contains(pos) and self.points:
            index = self.slider.value()
            tooltip = (
                f"<b>{self.target_class}</b>"
                "<hr style='margin: 0px; padding: 0px; border: 0px; height: 1px; background-color: #000000'>"
                f"<b>{i18n.t('Points')}:</b> {int(round(self.points[index]))}<br>"
                f"<b>{i18n.t('Probability')}:</b> {self.probabilities[index]:.1f}%"
            )
            QToolTip.showText(self.slider.mapToGlobal(pos), tooltip)
        else:
            QToolTip.hideText()

    def _thumb_rect(self) -> QRect:
        opt = QStyleOptionSlider()
        self.slider.initStyleOption(opt)
        style = self.slider.style()
        return style.subControlRect(QStyle.ComplexControl.CC_Slider, opt, QStyle.SubControl.SC_SliderHandle, self.slider)


class ScoringSheetViewerScreen(QWidget, WorkflowNodeScreenSupport):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._init_workflow_node_support()
        self._generated_datasets = GeneratedDatasetService()
        self._classifier: ScoringSheetClassifierArtifact | None = None
        self._data: DatasetHandle | None = None
        self._attributes: list[str] = []
        self._coefficients: list[int] = []
        self._all_scores: list[float] = []
        self._all_risks: list[float] = []
        self._instance_matches: list[int] = []
        self._features_dataset: DatasetHandle | None = None

        self._build_ui()
        self._reset_ui_to_original_state()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(10)

        title = QLabel(i18n.t("Scoring Sheet Viewer"))
        title.setProperty("sectionTitle", True)
        layout.addWidget(title)

        combo_layout = QHBoxLayout()
        combo_layout.addWidget(QLabel(i18n.t("Target class:")))
        self._class_combo = QComboBox(self)
        self._class_combo.currentIndexChanged.connect(self._class_combo_changed)
        self._class_combo.setFixedWidth(120)
        combo_layout.addWidget(self._class_combo)
        combo_layout.addStretch(1)
        layout.addLayout(combo_layout)

        self._coefficient_table = _ScoringSheetTable(self)
        self._coefficient_table.state_changed.connect(self._update_slider_value)
        layout.addWidget(self._coefficient_table, 1)

        self._risk_slider = _RiskSlider(self)
        layout.addWidget(self._risk_slider)

        self._status_label = QLabel(self)
        self._status_label.setWordWrap(True)
        self._status_label.setProperty("muted", True)
        layout.addWidget(self._status_label)

    def sizeHint(self) -> QSize:
        return QSize(720, 460)

    def set_input_payload(self, payload: WorkflowPayload | None) -> None:
        if payload is None:
            self._classifier = None
            self._data = None
            self._features_dataset = None
            self._reset_ui_to_original_state()
            self._status_label.setText(i18n.t("Connect a Scoring Sheet classifier to visualize it."))
            self._notify_output_changed()
            return
        if payload.port_label == "Classifier":
            self.set_classifier(payload.value)
        elif payload.port_label == "Data":
            self.set_data(payload.dataset if isinstance(payload.dataset, DatasetHandle) else None)

    def current_output_dataset(self):
        return self._features_dataset

    def documentation_url(self) -> str:
        return "https://orangedatamining.com/widget-catalog/visualize/scoringsheetviewer/"

    def help_text(self) -> str:
        return i18n.t("Visualize a scoring-sheet model, inspect rule points, and project the first input row onto the score slider.")

    def footer_status_text(self) -> str:
        return str(self._coefficient_table.rowCount())

    def serialize_node_state(self) -> dict[str, object]:
        return {"target_class_index": self._class_combo.currentIndex()}

    def restore_node_state(self, payload: dict[str, object]) -> None:
        self._class_combo.setCurrentIndex(int(payload.get("target_class_index", 0) or 0))

    def set_classifier(self, classifier: object) -> None:
        if classifier is None:
            self._classifier = None
            self._features_dataset = None
            self._reset_ui_to_original_state()
            self._status_label.setText(i18n.t("Connect a Scoring Sheet classifier to visualize it."))
            self._notify_output_changed()
            return
        if not isinstance(classifier, ScoringSheetClassifierArtifact):
            self._classifier = None
            self._features_dataset = None
            self._reset_ui_to_original_state()
            self._status_label.setText(i18n.t("Scoring Sheet Viewer only accepts a Scoring Sheet model."))
            self._notify_output_changed()
            return

        self._classifier = classifier
        self._extract_data_from_model()
        self._update_controls()
        self._update_features_output()
        self._notify_output_changed()

    def set_data(self, data: DatasetHandle | None) -> None:
        self._data = data
        if data is not None and data.row_count > 1:
            self._status_label.setText(i18n.t("The input data contains multiple instances. Only the first instance will be used."))
        elif self._classifier is not None:
            self._status_label.setText(i18n.t("Viewer ready."))
        self._set_instance_matches()

    def _reset_ui_to_original_state(self) -> None:
        self._coefficient_table.clearContents()
        self._coefficient_table.setRowCount(0)
        self._risk_slider.set_values([], [], "")
        self._risk_slider.slider.setValue(0)
        self._class_combo.blockSignals(True)
        self._class_combo.clear()
        self._class_combo.blockSignals(False)
        self._attributes = []
        self._coefficients = []
        self._all_scores = []
        self._all_risks = []
        self._instance_matches = []

    def _extract_data_from_model(self) -> None:
        assert self._classifier is not None
        self._attributes = list(self._classifier.rule_names)
        self._adjust_for_target_class(self._class_combo.currentIndex() if self._class_combo.count() else 0)

    def _update_controls(self) -> None:
        self._populate_interface()
        self._setup_class_combo()
        self._set_instance_matches()

    def _populate_interface(self) -> None:
        self._coefficient_table.populate_table(self._attributes, self._coefficients)
        assert self._classifier is not None
        class_value = self._classifier.class_values[self._class_combo.currentIndex() if self._class_combo.count() else 0]
        self._risk_slider.set_values(
            self._all_scores,
            self._all_risks,
            f"{self._classifier.target_name} = {class_value}",
        )
        self._update_slider_value()

    def _setup_class_combo(self) -> None:
        assert self._classifier is not None
        current = self._class_combo.currentIndex()
        self._class_combo.blockSignals(True)
        self._class_combo.clear()
        self._class_combo.addItems(list(self._classifier.class_values))
        self._class_combo.setCurrentIndex(max(0, min(current, self._class_combo.count() - 1)))
        self._class_combo.blockSignals(False)

    def _class_combo_changed(self) -> None:
        if self._classifier is None:
            return
        self._adjust_for_target_class(self._class_combo.currentIndex())
        self._populate_interface()
        self._set_instance_matches()
        self._update_features_output()
        self._notify_output_changed()

    def _adjust_for_target_class(self, class_index: int) -> None:
        assert self._classifier is not None
        self._coefficients, self._all_scores, self._all_risks = self._classifier.class_view(class_index)

    def _set_instance_matches(self) -> None:
        if self._classifier is None or self._data is None or not self._classifier.can_apply_to(self._data):
            self._instance_matches = []
            self._coefficient_table.set_checkbox_states([])
            return
        self._instance_matches = self._classifier.first_row_matches(self._data)
        self._coefficient_table.set_checkbox_states(self._instance_matches)

    def _update_slider_value(self) -> None:
        total = 0.0
        for row in range(self._coefficient_table.rowCount()):
            checkbox = self._coefficient_table.item(row, 2)
            coefficient_item = self._coefficient_table.item(row, 1)
            if checkbox is None or coefficient_item is None:
                continue
            if checkbox.checkState() == Qt.CheckState.Checked:
                total += float(coefficient_item.text())
        self._risk_slider.move_to_value(total)

    def _update_features_output(self) -> None:
        if self._classifier is None:
            self._features_dataset = None
            return
        dataframe = pl.DataFrame(
            {
                "Feature": self._attributes,
                "Source": [rule.source_feature for rule in self._classifier.rules],
                "Points": self._coefficients,
            }
        )
        self._features_dataset = self._generated_datasets.build_dataset(
            dataframe,
            dataset_id=f"{self._classifier.classifier_id}-viewer-features",
            display_name="Scoring Sheet Features",
            file_name="scoring-sheet-features.csv",
            role_overrides={column: "feature" for column in dataframe.columns},
            annotations={"target_class": self._class_combo.currentText() or ""},
        )
