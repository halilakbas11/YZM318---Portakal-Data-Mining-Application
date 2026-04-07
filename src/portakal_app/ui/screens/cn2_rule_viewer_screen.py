from __future__ import annotations

from html import escape
from typing import Any

import numpy as np
import polars as pl

from PySide6.QtCore import QAbstractTableModel, QModelIndex, QItemSelectionModel, QSortFilterProxyModel, Qt, QSize
from PySide6.QtGui import QBrush, QColor, QPainter, QPen
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QCheckBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QStyledItemDelegate,
    QTableView,
    QVBoxLayout,
    QWidget,
)

from portakal_app.data.models import DatasetHandle
from portakal_app.data.services.generated_dataset_service import GeneratedDatasetService
from portakal_app.models import WorkflowPayload
from portakal_app.rule_artifacts import CN2RuleArtifact, CN2RuleClassifierArtifact
from portakal_app.tree_artifacts import DEFAULT_TREE_CLASS_COLORS
from portakal_app.ui import i18n
from portakal_app.ui.screens.node_screen import WorkflowNodeScreenSupport


OUTPUT_CHANNELS = ("Selected Data", "Annotated Data")
SELECTED_COLUMN_NAME = "Selected"


def _display_value(value: Any) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    return str(value)


def _dataset_summary(dataset: DatasetHandle | None) -> str:
    if dataset is None:
        return i18n.t("No data")
    return i18n.tf("{name}: {rows} rows x {cols} cols", name=dataset.display_name, rows=dataset.row_count, cols=dataset.column_count)


def _preview_rows(dataset: DatasetHandle | None, limit: int = 200) -> list[list[str]]:
    if dataset is None:
        return []
    return [[_display_value(value) for value in row] for row in dataset.dataframe.head(limit).iter_rows(named=False)]


def _role_overrides(dataset: DatasetHandle) -> dict[str, str]:
    return {column.name: column.role for column in dataset.domain.columns}


def _subset_frame(dataset: DatasetHandle, indices: list[int]) -> pl.DataFrame:
    if not indices:
        return dataset.dataframe.head(0)
    mask = [False] * dataset.row_count
    for index in indices:
        if 0 <= index < dataset.row_count:
            mask[index] = True
    return dataset.dataframe.filter(pl.Series("__mask__", mask))


class _DistributionDelegate(QStyledItemDelegate):
    def __init__(self, colors: list[QColor], parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._colors = colors

    def paint(self, painter: QPainter, option, index) -> None:
        values = index.data(Qt.ItemDataRole.DisplayRole)
        if not isinstance(values, (list, tuple)) or not values:
            super().paint(painter, option, index)
            return

        distribution = np.asarray(values, dtype=float)
        total = float(np.sum(distribution))
        painter.save()
        self.drawBackground(painter, option, index)
        rect = option.rect.adjusted(6, 6, -6, -6)
        if total > 0:
            start_x = float(rect.left())
            width = float(rect.width())
            baseline = rect.center().y()
            bar_height = max(3.0, rect.height() / 4)
            for offset, value in enumerate(distribution):
                if value <= 0:
                    continue
                proportion = value / total
                end_x = start_x + proportion * width
                painter.setPen(QPen(self._colors[offset % len(self._colors)], bar_height, Qt.PenStyle.SolidLine, Qt.PenCapStyle.FlatCap))
                painter.drawLine(int(start_x), int(baseline), int(end_x), int(baseline))
                start_x = end_x
        painter.setPen(option.palette.color(option.palette.ColorRole.Text))
        painter.drawText(option.rect, Qt.AlignmentFlag.AlignCenter, str(int(total)) if total.is_integer() else f"{total:.1f}")
        painter.restore()


class _RuleTableModel(QAbstractTableModel):
    SortRole = Qt.ItemDataRole.UserRole + 1

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._classifier: CN2RuleClassifierArtifact | None = None
        self._rules: list[CN2RuleArtifact] = []
        self._compact_view = False
        self._horizontal_headers = [
            i18n.t("IF conditions"),
            "",
            i18n.t("THEN class"),
            i18n.t("Distribution"),
            i18n.t("Probabilities [%]"),
            i18n.t("Quality"),
            i18n.t("Length"),
        ]

    def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:  # noqa: N802
        return 0 if parent.isValid() else len(self._rules)

    def columnCount(self, parent: QModelIndex = QModelIndex()) -> int:  # noqa: N802
        return 0 if parent.isValid() else len(self._horizontal_headers)

    def headerData(self, section: int, orientation: Qt.Orientation, role: int = Qt.ItemDataRole.DisplayRole):
        if role != Qt.ItemDataRole.DisplayRole:
            return None
        if orientation == Qt.Orientation.Horizontal:
            return self._horizontal_headers[section] if 0 <= section < len(self._horizontal_headers) else None
        return str(section)

    def wrap(self, classifier: CN2RuleClassifierArtifact | None) -> None:
        self.beginResetModel()
        self._classifier = classifier
        self._rules = list(classifier.rule_list) if classifier is not None else []
        self.endResetModel()

    def set_compact_view(self, compact_view: bool) -> None:
        self.beginResetModel()
        self._compact_view = bool(compact_view)
        self.endResetModel()

    def rule_at(self, row: int) -> CN2RuleArtifact:
        return self._rules[row]

    def data(self, index: QModelIndex, role: int = Qt.ItemDataRole.DisplayRole):
        if not index.isValid() or self._classifier is None:
            return None

        rule = self._rules[index.row()]
        column = index.column()

        if role == Qt.ItemDataRole.DisplayRole:
            if column == 0:
                return rule.conditions_text(compact_view=self._compact_view)
            if column == 1:
                return "->"
            if column == 2:
                return rule.prediction_text()
            if column == 3:
                return list(rule.curr_class_dist)
            if column == 4:
                return " : ".join(str(int(round(probability * 100))) for probability in rule.probabilities)
            if column == 5:
                return self._format_quality(rule.quality)
            if column == 6:
                return rule.length
            return None

        if role == Qt.ItemDataRole.ToolTipRole:
            if column == 0:
                return escape(rule.conditions_text(compact_view=False))
            if column == 3:
                return "\n".join(
                    f"{class_value}: {rule.curr_class_dist[offset]:.1f}"
                    for offset, class_value in enumerate(self._classifier.class_values)
                )
            if column == 4:
                return "\n".join(
                    f"{class_value}: {rule.probabilities[offset] * 100:.1f}%"
                    for offset, class_value in enumerate(self._classifier.class_values)
                )
            return str(self.data(index, Qt.ItemDataRole.DisplayRole))

        if role == self.SortRole:
            if column == 0:
                return rule.length
            if column == 3:
                return int(sum(rule.curr_class_dist))
            if column == 5:
                return float(rule.quality)
            if column == 6:
                return int(rule.length)
            return self.data(index, Qt.ItemDataRole.DisplayRole)

        if role == Qt.ItemDataRole.TextAlignmentRole:
            aligns = (
                Qt.AlignmentFlag.AlignRight if self._compact_view else Qt.AlignmentFlag.AlignLeft,
                Qt.AlignmentFlag.AlignCenter,
                Qt.AlignmentFlag.AlignLeft,
                Qt.AlignmentFlag.AlignCenter,
                Qt.AlignmentFlag.AlignCenter,
                Qt.AlignmentFlag.AlignRight,
                Qt.AlignmentFlag.AlignRight,
            )
            return Qt.AlignmentFlag.AlignVCenter | aligns[column]

        return None

    def _format_quality(self, value: float) -> str:
        abs_value = abs(value)
        integer_length = len(str(int(abs_value))) if abs_value >= 1 else 1
        precision = 2 if abs_value < 0.001 else 3 if integer_length < 2 else 1 if integer_length < 5 else 0
        return f"{value:.{precision}f}" if abs_value == 0 or (abs_value >= 0.001 and integer_length < 6) else f"{value:.3e}"


class CN2RuleViewerScreen(QWidget, WorkflowNodeScreenSupport):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._init_workflow_node_support()
        self._generated_datasets = GeneratedDatasetService()
        self._input_data: DatasetHandle | None = None
        self._classifier: CN2RuleClassifierArtifact | None = None
        self._selected_rule_rows: list[int] | None = None
        self._selected_dataset: DatasetHandle | None = None
        self._annotated_dataset: DatasetHandle | None = None

        self._model = _RuleTableModel(self)
        self._proxy_model = QSortFilterProxyModel(self)
        self._proxy_model.setSourceModel(self._model)
        self._proxy_model.setSortRole(_RuleTableModel.SortRole)

        self._build_ui()
        self._set_classifier(None)
        self._commit()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(10)

        header = QLabel(i18n.t("CN2 Rule Viewer"))
        header.setProperty("sectionTitle", True)
        layout.addWidget(header)

        description = QLabel(i18n.t("Review rules induced from data, sort them, and send selected rows downstream."))
        description.setWordWrap(True)
        description.setProperty("muted", True)
        layout.addWidget(description)

        self._view = QTableView(self)
        self._view.setModel(self._proxy_model)
        self._view.setWordWrap(False)
        self._view.setSortingEnabled(True)
        self._view.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self._view.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self._view.verticalHeader().setVisible(True)
        self._view.horizontalHeader().setStretchLastSection(False)
        self._view.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self._view.selectionModel().selectionChanged.connect(lambda *_args: self._commit())
        layout.addWidget(self._view, 1)

        self._status_label = QLabel(self)
        self._status_label.setWordWrap(True)
        self._status_label.setProperty("muted", True)
        layout.addWidget(self._status_label)

        footer = QHBoxLayout()
        footer.setSpacing(10)
        self._compact_checkbox = QCheckBox(i18n.t("Compact view"))
        self._compact_checkbox.toggled.connect(self._on_compact_view_changed)
        footer.addWidget(self._compact_checkbox)

        self._copy_button = QPushButton(i18n.t("Copy Selected Rules"))
        self._copy_button.clicked.connect(self.copy_to_clipboard)
        footer.addWidget(self._copy_button)

        self._restore_order_button = QPushButton(i18n.t("Restore original order"))
        self._restore_order_button.clicked.connect(self.restore_original_order)
        footer.addWidget(self._restore_order_button)
        footer.addStretch(1)
        layout.addLayout(footer)

    def sizeHint(self) -> QSize:
        return QSize(820, 480)

    def set_input_payload(self, payload: WorkflowPayload | None) -> None:
        if payload is None:
            self._input_data = None
            self._set_classifier(None)
            self._commit()
            return

        if payload.port_label == "Data" and isinstance(payload.dataset, DatasetHandle):
            self._input_data = payload.dataset
        elif payload.port_label == "Classifier" and isinstance(payload.value, CN2RuleClassifierArtifact):
            self._set_classifier(payload.value)
        self._commit()

    def _set_classifier(self, classifier: CN2RuleClassifierArtifact | None) -> None:
        self._classifier = classifier
        self._selected_rule_rows = None
        self._model.wrap(classifier)
        colors = [QColor(color) for color in DEFAULT_TREE_CLASS_COLORS[: max(1, len(classifier.class_values) if classifier else 1)]]
        self._view.setItemDelegateForColumn(3, _DistributionDelegate(colors, self._view))
        self._resize_table()
        self._update_status()

    def current_output_datasets(self) -> dict[str, DatasetHandle | None] | None:
        return {
            OUTPUT_CHANNELS[0]: self._selected_dataset,
            OUTPUT_CHANNELS[1]: self._annotated_dataset,
        }

    def documentation_url(self) -> str:
        return "https://orangedatamining.com/widget-catalog/visualize/cn2ruleviewer/"

    def help_text(self) -> str:
        return i18n.t("Display induced CN2 rules in a sortable table and send rows covered by the selected rules.")

    def footer_status_text(self) -> str:
        annotated_rows = self._annotated_dataset.row_count if self._annotated_dataset is not None else 0
        selected_rows = self._selected_dataset.row_count if self._selected_dataset is not None else 0
        if annotated_rows == 0:
            return "0"
        if selected_rows:
            return f"{selected_rows} | {annotated_rows}"
        return str(annotated_rows)

    def data_preview_snapshot(self) -> dict[str, object]:
        dataset = self._selected_dataset or self._annotated_dataset
        if dataset is None:
            return {"summary": i18n.t("No preview available."), "headers": [], "rows": []}
        return {
            "summary": _dataset_summary(dataset),
            "headers": list(dataset.dataframe.columns),
            "rows": _preview_rows(dataset),
        }

    def detailed_data_snapshot(self) -> dict[str, object]:
        return {
            "selected_summary": _dataset_summary(self._selected_dataset) if self._selected_dataset is not None else i18n.t("Selected Data: -"),
            "selected_headers": list(self._selected_dataset.dataframe.columns) if self._selected_dataset is not None else [],
            "selected_rows": _preview_rows(self._selected_dataset) if self._selected_dataset is not None else [],
            "data_summary": _dataset_summary(self._annotated_dataset) if self._annotated_dataset is not None else i18n.t("Data: -"),
            "data_headers": list(self._annotated_dataset.dataframe.columns) if self._annotated_dataset is not None else [],
            "data_rows": _preview_rows(self._annotated_dataset) if self._annotated_dataset is not None else [],
        }

    def serialize_node_state(self) -> dict[str, object]:
        return {
            "compact_view": self._compact_checkbox.isChecked(),
            "selected_rules": list(self._selected_rule_rows or []),
        }

    def restore_node_state(self, payload: dict[str, object]) -> None:
        self._compact_checkbox.setChecked(bool(payload.get("compact_view", False)))
        selected_rows = payload.get("selected_rules", [])
        if isinstance(selected_rows, list):
            self._selected_rule_rows = [int(row) for row in selected_rows if isinstance(row, int)]
        self._restore_selected()
        self._commit()

    def restore_original_order(self) -> None:
        self._proxy_model.sort(-1)

    def copy_to_clipboard(self) -> None:
        self._save_selected(actual=True)
        if not self._selected_rule_rows or self._classifier is None:
            return
        output = "\n".join(self._classifier.rule_list[row].to_text() for row in self._selected_rule_rows)
        QApplication.clipboard().setText(output)

    def _on_compact_view_changed(self) -> None:
        self._save_selected(actual=True)
        self._model.set_compact_view(self._compact_checkbox.isChecked())
        if self._compact_checkbox.isChecked():
            self._view.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Interactive)
        else:
            self._view.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self._resize_table()
        self._restore_selected()

    def _resize_table(self) -> None:
        self._view.resizeColumnsToContents()
        self._view.resizeRowsToContents()

    def _save_selected(self, actual: bool = False) -> None:
        self._selected_rule_rows = None
        selection_model = self._view.selectionModel()
        if selection_model is None or not selection_model.hasSelection():
            return
        selection = selection_model.selection()
        if actual:
            selection = self._proxy_model.mapSelectionToSource(selection)
        self._selected_rule_rows = sorted({index.row() for index in selection.indexes()})

    def _restore_selected(self) -> None:
        if not self._selected_rule_rows:
            return
        selection_model = self._view.selectionModel()
        if selection_model is None:
            return
        for row in self._selected_rule_rows:
            source_index = self._model.index(row, 0)
            proxy_index = self._proxy_model.mapFromSource(source_index)
            if proxy_index.isValid():
                selection_model.select(
                    proxy_index,
                    QItemSelectionModel.SelectionFlag.Select | QItemSelectionModel.SelectionFlag.Rows,
                )

    def _active_data(self) -> DatasetHandle | None:
        return self._input_data or (self._classifier.instances if self._classifier is not None else None)

    def _commit(self) -> None:
        self._save_selected(actual=True)
        data = self._active_data()
        selected_indices: list[int] = []
        selected_mask = np.zeros(data.row_count, dtype=bool) if data is not None else np.asarray([], dtype=bool)

        if (
            data is not None
            and self._classifier is not None
            and self._selected_rule_rows is not None
            and self._classifier.can_apply_to(data)
        ):
            rule_mask = np.ones(data.row_count, dtype=bool)
            for row in self._selected_rule_rows:
                if 0 <= row < len(self._classifier.rule_list):
                    rule_mask &= self._classifier.rule_list[row].evaluate_data(data)
            selected_indices = np.flatnonzero(rule_mask).tolist()
            if selected_indices:
                selected_mask[selected_indices] = True

        if data is None:
            self._selected_dataset = None
            self._annotated_dataset = None
            self._update_status()
            self._notify_output_changed()
            return

        self._selected_dataset = self._build_selected_output(data, selected_indices)
        self._annotated_dataset = self._build_annotated_output(data, selected_mask)
        self._update_status()
        self._notify_output_changed()

    def _build_selected_output(self, data: DatasetHandle, selected_indices: list[int]) -> DatasetHandle | None:
        if not selected_indices:
            return None
        frame = _subset_frame(data, selected_indices)
        return self._generated_datasets.build_dataset(
            frame,
            dataset_id=f"cn2-viewer-{data.dataset_id}-selected",
            display_name=i18n.t("Selected Data"),
            file_name=f"cn2-viewer-{data.dataset_id}-selected.csv",
            role_overrides=_role_overrides(data),
            annotations={"source_row_indices": selected_indices},
        )

    def _build_annotated_output(self, data: DatasetHandle, selected_mask: np.ndarray) -> DatasetHandle:
        role_overrides = _role_overrides(data)
        role_overrides[SELECTED_COLUMN_NAME] = "meta"
        frame = data.dataframe.with_columns(pl.Series(SELECTED_COLUMN_NAME, selected_mask.tolist()))
        return self._generated_datasets.build_dataset(
            frame,
            dataset_id=f"cn2-viewer-{data.dataset_id}-annotated",
            display_name=i18n.t("Annotated Data"),
            file_name=f"cn2-viewer-{data.dataset_id}-annotated.csv",
            role_overrides=role_overrides,
            annotations={"selected_row_indices": np.flatnonzero(selected_mask).tolist()},
        )

    def _update_status(self) -> None:
        if self._classifier is None and self._input_data is None:
            self._status_label.setText(i18n.t("Connect data and a classifier to inspect induced rules."))
            return
        if self._classifier is None:
            self._status_label.setText(i18n.t("Classifier not connected. Annotated output mirrors the incoming data."))
            return
        if self._active_data() is not None and not self._classifier.can_apply_to(self._active_data()):
            self._status_label.setText(i18n.t("Connected data does not match the classifier domain."))
            return
        selected_rules = len(self._selected_rule_rows or [])
        self._status_label.setText(
            i18n.tf(
                "{rules} rules | {selected} selected rules | target: {target}",
                rules=len(self._classifier.rule_list),
                selected=selected_rules,
                target=self._classifier.target_name,
            )
        )
