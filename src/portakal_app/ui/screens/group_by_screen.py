from __future__ import annotations

from PySide6.QtWidgets import (
    QCheckBox,
    QGroupBox,
    QGridLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)
from PySide6.QtCore import Qt

from portakal_app.data.models import DatasetHandle
from portakal_app.data.services.group_by_service import AGGREGATIONS, GroupByService
from portakal_app.ui.screens.node_screen import WorkflowNodeScreenSupport
from portakal_app.ui import i18n
from portakal_app.ui.shared.type_icons import type_badge_icon


AGG_SUPPORTED_TYPES = {
    "Mean": {"numeric", "datetime"},
    "Median": {"numeric", "datetime"},
    "Q1": {"numeric", "datetime"},
    "Q3": {"numeric", "datetime"},
    "Min. value": {"numeric", "datetime"},
    "Max. value": {"numeric", "datetime"},
    "Mode": {"numeric", "datetime", "categorical", "boolean", "string"},
    "Standard deviation": {"numeric", "datetime"},
    "Variance": {"numeric", "datetime"},
    "Sum": {"numeric"},
    "Concatenate": {"numeric", "datetime", "categorical", "boolean", "string"},
    "Span": {"numeric", "datetime"},
    "First value": {"numeric", "datetime", "categorical", "boolean", "string"},
    "Last value": {"numeric", "datetime", "categorical", "boolean", "string"},
    "Random value": {"numeric", "datetime", "categorical", "boolean", "string"},
    "Count defined": {"numeric", "datetime", "categorical", "boolean", "string"},
    "Count": {"numeric", "datetime", "categorical", "boolean", "string"},
    "Proportion defined": {"numeric", "datetime", "categorical", "boolean", "string"},
}

DEFAULT_AGGREGATIONS = {
    "numeric": {"Mean"},
    "datetime": {"Mean"},
    "categorical": {"Mode"},
    "string": {"Concatenate"},
    "boolean": {"Mode"},
}


class TristateCheckBox(QCheckBox):
    def __init__(self, text: str, screen: 'GroupByScreen', parent: QWidget | None = None):
        super().__init__(text, parent)
        self.screen = screen

    def nextCheckState(self) -> None:
        if self.checkState() == Qt.CheckState.Checked:
            self.setCheckState(Qt.CheckState.Unchecked)
        else:
            agg_name = self.text()
            selected_attrs = self.screen._get_selected_attr_names()
            logical_types = {self.screen._get_logical_type(attr) for attr in selected_attrs}
            
            supported_types = AGG_SUPPORTED_TYPES.get(agg_name, set())
            can_be_applied_all = logical_types.issubset(supported_types) if logical_types else False

            # true if aggregation applied to all attributes that can be aggregated
            applied_all = True
            for attr in selected_attrs:
                t = self.screen._get_logical_type(attr)
                if t in supported_types and agg_name not in self.screen._attr_aggregations.get(attr, set()):
                    applied_all = False
                    break

            if self.checkState() == Qt.CheckState.PartiallyChecked:
                if can_be_applied_all:
                    self.setCheckState(Qt.CheckState.Checked)
                elif applied_all:
                    self.setCheckState(Qt.CheckState.Unchecked)
                else:
                    self.setCheckState(Qt.CheckState.PartiallyChecked)
                    self.stateChanged.emit(self.checkState().value)
            else:  # Qt.CheckState.Unchecked
                self.setCheckState(Qt.CheckState.Checked if can_be_applied_all else Qt.CheckState.PartiallyChecked)


class GroupByScreen(QWidget, WorkflowNodeScreenSupport):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._init_workflow_node_support()
        self._service = GroupByService()
        self._dataset_handle: DatasetHandle | None = None
        self._output_dataset: DatasetHandle | None = None
        # Per-attribute aggregation: {col_name: set of agg names}
        self._attr_aggregations: dict[str, set[str]] = {}
        self._value_columns: list[str] = []

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(10)

        self._dataset_label = QLabel(i18n.t("Dataset: none"))
        self._dataset_label.setProperty("sectionTitle", True)
        self._dataset_label.setStyleSheet("font-size: 12pt; background: transparent;")
        layout.addWidget(self._dataset_label)

        # --- Splitter: Group By list (left) | Attributes table (right) ---
        splitter = QSplitter(Qt.Orientation.Horizontal)

        # Left: Group By column selection
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(0, 0, 0, 0)
        group_group = QGroupBox(i18n.t("Group By"))
        group_layout = QVBoxLayout(group_group)
        group_layout.setContentsMargins(10, 10, 10, 10)
        self._group_list = QListWidget()
        self._group_list.setSelectionMode(QListWidget.SelectionMode.MultiSelection)
        self._group_list.itemSelectionChanged.connect(self._on_group_selection_changed)
        group_layout.addWidget(self._group_list)
        left_layout.addWidget(group_group)
        splitter.addWidget(left_widget)

        # Right: Attributes + Aggregations table
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(0, 0, 0, 0)

        self._attr_table = QTableWidget()
        self._attr_table.setColumnCount(2)
        self._attr_table.setHorizontalHeaderLabels([i18n.t("Attributes"), i18n.t("Aggregations")])
        self._attr_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self._attr_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self._attr_table.setSelectionMode(QTableWidget.SelectionMode.ExtendedSelection)
        self._attr_table.itemSelectionChanged.connect(self._on_attr_selection_changed)
        self._rebuilding_table = False
        right_layout.addWidget(self._attr_table)
        splitter.addWidget(right_widget)

        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 2)
        layout.addWidget(splitter, 1)

        # --- Aggregations checkboxes (grid like Orange) ---
        agg_group = QGroupBox(i18n.t("Aggregations"))
        agg_grid = QGridLayout(agg_group)
        agg_grid.setContentsMargins(10, 10, 10, 10)
        agg_grid.setSpacing(4)
        self._agg_checks: dict[str, TristateCheckBox] = {}
        agg_list = list(AGGREGATIONS)
        
        break_rows = (6, 6, 99)
        col = 0
        row = 0
        for agg in agg_list:
            cb = TristateCheckBox(agg, self)
            cb.setEnabled(False)
            cb.stateChanged.connect(lambda state, a=agg: self._on_agg_checkbox_changed(a))
            self._agg_checks[agg] = cb
            agg_grid.addWidget(cb, row, col)
            row += 1
            if row == break_rows[col]:
                row = 0
                col += 1
                
        layout.addWidget(agg_group)

        self._result_label = QLabel("")
        self._result_label.setWordWrap(True)
        layout.addWidget(self._result_label)

        # Auto-apply hooks
        self._group_list.itemSelectionChanged.connect(self._check_auto_apply)

        footer = QHBoxLayout()
        self.cb_apply_auto = QCheckBox(i18n.t("Apply Automatically"))
        self.cb_apply_auto.setChecked(True)
        footer.addWidget(self.cb_apply_auto)
        footer.addStretch(1)
        self._apply_button = QPushButton(i18n.t("Apply"))
        self._apply_button.setProperty("primary", True)
        self._apply_button.clicked.connect(self._apply)
        footer.addWidget(self._apply_button)
        layout.addLayout(footer)

    def set_input_payload(self, payload) -> None:
        dataset = payload.dataset if payload is not None else None
        if payload is not None and dataset is self._dataset_handle:
            return
        self._dataset_handle = dataset
        self._output_dataset = None
        self._group_list.clear()
        self._attr_table.setRowCount(0)
        self._attr_aggregations.clear()
        self._value_columns.clear()

        if dataset:
            self._dataset_label.setText(i18n.tf("Dataset: {name}", name=dataset.display_name))
            for col in dataset.domain.columns:
                item = QListWidgetItem(type_badge_icon(col.logical_type), col.name)
                self._group_list.addItem(item)
            
            # Initialize per-attribute aggregations with defaults
            for col in dataset.domain.columns:
                default_agg = DEFAULT_AGGREGATIONS.get(col.logical_type, {"First value"})
                self._attr_aggregations[col.name] = set(default_agg)
                self._value_columns.append(col.name)
            self._rebuild_attr_table()
            # Select all rows by default so checkboxes are instantly usable
            self._attr_table.selectAll()
        else:
            self._dataset_label.setText(i18n.t("Dataset: none"))
            self._result_label.setText("")
        self._apply()

    def _get_logical_type(self, attr_name: str) -> str:
        if not self._dataset_handle:
            return "numeric"
        for col in self._dataset_handle.domain.columns:
            if col.name == attr_name:
                return col.logical_type
        return "numeric"

    def _on_group_selection_changed(self) -> None:
        """When group-by selection changes, update the available attributes to aggregate."""
        self._rebuild_attr_table()

    def _rebuild_attr_table(self) -> None:
        """Rebuild the attributes table showing columns and their aggregations."""
        prev_selected = self._get_selected_attr_names()

        self._rebuilding_table = True
        self._attr_table.setRowCount(0)
        if not self._dataset_handle:
            self._rebuilding_table = False
            return
        group_cols = {
            self._group_list.item(i).text()
            for i in range(self._group_list.count())
            if self._group_list.item(i).isSelected()
        }
        visible = [c for c in self._dataset_handle.domain.columns if c.name not in group_cols]
        self._attr_table.setRowCount(len(visible))
        
        agg_order = list(AGGREGATIONS)
        
        for row, col in enumerate(visible):
            name_item = QTableWidgetItem(type_badge_icon(col.logical_type), col.name)
            name_item.setFlags(Qt.ItemFlag.ItemIsSelectable | Qt.ItemFlag.ItemIsEnabled)
            self._attr_table.setItem(row, 0, name_item)
            
            aggs = self._attr_aggregations.get(col.name, set())
            aggs_sorted = sorted(aggs, key=lambda x: agg_order.index(x) if x in agg_order else 999)
            
            # Format aggregations string, e.g. "Mean, Median and 2 more"
            if len(aggs_sorted) <= 3:
                agg_text = ", ".join(aggs_sorted) if aggs_sorted else "-"
            else:
                agg_text = ", ".join(aggs_sorted[:3]) + " " + i18n.tf("and {count} more", count=len(aggs_sorted) - 3)
            
            agg_item = QTableWidgetItem(agg_text)
            agg_item.setFlags(Qt.ItemFlag.ItemIsSelectable | Qt.ItemFlag.ItemIsEnabled)
            self._attr_table.setItem(row, 1, agg_item)
            
        self._rebuilding_table = False

        # Restore selection
        for row in range(self._attr_table.rowCount()):
            item = self._attr_table.item(row, 0)
            if item and item.text() in prev_selected:
                self._attr_table.selectRow(row)

    def _on_attr_selection_changed(self) -> None:
        """When table selection changes, update aggregation checkboxes."""
        if self._rebuilding_table:
            return
        selected_attrs = self._get_selected_attr_names()
        if not selected_attrs:
            for cb in self._agg_checks.values():
                cb.setEnabled(False)
                cb.blockSignals(True)
                cb.setCheckState(Qt.CheckState.Unchecked)
                cb.blockSignals(False)
            return
            
        types = {self._get_logical_type(attr) for attr in selected_attrs}
        active_aggregations = [self._attr_aggregations.get(attr, set()) for attr in selected_attrs]

        # Enable checkboxes based on type compatibility
        for agg_name, cb in self._agg_checks.items():
            supported_types = AGG_SUPPORTED_TYPES.get(agg_name, set())
            cb.setEnabled(bool(types & supported_types))

            # Check state: checked if ALL selected have it, partial if SOME, unchecked if NONE
            active = {agg_name in aggs for aggs in active_aggregations}
            cb.blockSignals(True)
            if active == {True}:
                cb.setCheckState(Qt.CheckState.Checked)
            elif active == {False}:
                cb.setCheckState(Qt.CheckState.Unchecked)
            else:
                cb.setCheckState(Qt.CheckState.PartiallyChecked)
            cb.blockSignals(False)

    def _check_auto_apply(self):
        if self.cb_apply_auto.isChecked():
            self._apply()

    def _on_agg_checkbox_changed(self, agg_name: str) -> None:
        """When an aggregation checkbox is toggled, update selected attributes."""
        selected_attrs = self._get_selected_attr_names()
        cb = self._agg_checks[agg_name]
        is_checked = cb.checkState() != Qt.CheckState.Unchecked
        supported_types = AGG_SUPPORTED_TYPES.get(agg_name, set())
        
        for attr in selected_attrs:
            if attr not in self._attr_aggregations:
                self._attr_aggregations[attr] = set()
            t = self._get_logical_type(attr)
            
            if is_checked and t in supported_types:
                self._attr_aggregations[attr].add(agg_name)
            else:
                self._attr_aggregations[attr].discard(agg_name)
                
        self._rebuild_attr_table()
        self._check_auto_apply()

    def _get_selected_attr_names(self) -> list[str]:
        selected_rows = set()
        for idx in self._attr_table.selectionModel().selectedRows():
            selected_rows.add(idx.row())
        names = []
        for row in sorted(selected_rows):
            item = self._attr_table.item(row, 0)
            if item:
                names.append(item.text())
        return names

    def current_output_dataset(self) -> DatasetHandle | None:
        return self._output_dataset

    def serialize_node_state(self) -> dict[str, object]:
        group_cols = [
            self._group_list.item(i).text()
            for i in range(self._group_list.count())
            if self._group_list.item(i).isSelected()
        ]
        aggs = {k: list(v) for k, v in self._attr_aggregations.items()}
        return {"group_columns": group_cols, "aggregations": aggs, "auto_apply": self.cb_apply_auto.isChecked()}

    def restore_node_state(self, payload: dict[str, object]) -> None:
        group_cols = payload.get("group_columns", [])
        if isinstance(group_cols, list):
            for i in range(self._group_list.count()):
                item = self._group_list.item(i)
                item.setSelected(item.text() in group_cols)
        aggs = payload.get("aggregations", {})
        if isinstance(aggs, dict):
            for k, v in aggs.items():
                if isinstance(v, list):
                    self._attr_aggregations[k] = set(v)
        self._rebuild_attr_table()
        self.cb_apply_auto.setChecked(bool(payload.get("auto_apply", True)))

    def help_text(self) -> str:
        return i18n.t("Group the dataset by selected columns and compute per-attribute aggregations.")

    def documentation_url(self) -> str:
        return "https://orangedatamining.com/widget-catalog/transform/groupby/"

    def _apply(self) -> None:
        if self._dataset_handle is None:
            self._output_dataset = None
            self._result_label.setText("")
            self._notify_output_changed()
            return

        group_cols = [
            self._group_list.item(i).text()
            for i in range(self._group_list.count())
            if self._group_list.item(i).isSelected()
        ]

        # Build per-column aggregations (exclude group columns)
        aggregations: dict[str, list[str]] = {}
        for col_name, agg_set in self._attr_aggregations.items():
            if col_name not in group_cols and agg_set:
                aggregations[col_name] = list(agg_set)

        self._output_dataset = self._service.group_by(
            self._dataset_handle,
            group_columns=group_cols,
            aggregations=aggregations,
        )

        self._result_label.setText(
            i18n.tf(
                "Groups: {groups} | Columns: {columns}",
                groups=self._output_dataset.row_count,
                columns=self._output_dataset.column_count,
            )
        )
        self._notify_output_changed()

    def refresh_translations(self) -> None:
        if self._dataset_handle is None:
            self._dataset_label.setText(i18n.t("Dataset: none"))
        else:
            self._dataset_label.setText(i18n.tf("Dataset: {name}", name=self._dataset_handle.display_name))
