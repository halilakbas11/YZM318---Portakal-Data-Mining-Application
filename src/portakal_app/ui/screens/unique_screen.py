from __future__ import annotations

from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QVBoxLayout,
    QWidget,
)
from PySide6.QtCore import Qt

from portakal_app.data.models import DatasetHandle
from portakal_app.data.services.unique_service import TIEBREAKERS, UniqueService
from portakal_app.ui import i18n
from portakal_app.ui.screens.node_screen import WorkflowNodeScreenSupport
from portakal_app.ui.shared.type_icons import type_badge_icon


class UniqueScreen(QWidget, WorkflowNodeScreenSupport):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._init_workflow_node_support()
        self._service = UniqueService()
        self._dataset_handle: DatasetHandle | None = None
        self._output_dataset: DatasetHandle | None = None
        self._output_removed: DatasetHandle | None = None
        self._output_annotated: DatasetHandle | None = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)

        # ── Dataset label ─────────────────────────────────────────────
        self._dataset_label = QLabel(i18n.t("Dataset: none"))
        self._dataset_label.setProperty("sectionTitle", True)
        self._dataset_label.setStyleSheet(
            "font-size: 12pt; background: transparent;"
        )
        layout.addWidget(self._dataset_label)

        # ── Group By ──────────────────────────────────────────────────
        group_by_group = QGroupBox(i18n.t("Group By"))
        group_by_layout = QVBoxLayout(group_by_group)
        group_by_layout.setContentsMargins(6, 6, 6, 6)
        group_by_layout.setSpacing(4)

        # Filter search box
        self._filter_edit = QLineEdit()
        self._filter_edit.setPlaceholderText(i18n.t("Filter"))
        self._filter_edit.setClearButtonEnabled(True)
        self._filter_edit.textChanged.connect(self._apply_filter)
        group_by_layout.addWidget(self._filter_edit)

        # Column list (Target columns shown first with visual badge)
        self._column_list = QListWidget()
        self._column_list.setSelectionMode(
            QListWidget.SelectionMode.MultiSelection
        )
        group_by_layout.addWidget(self._column_list)

        # Select all / deselect all buttons
        sel_row = QHBoxLayout()
        sel_row.setContentsMargins(0, 0, 0, 0)
        sel_row.setSpacing(4)
        self._select_all_btn = QPushButton(i18n.t("Select All"))
        self._select_all_btn.clicked.connect(self._select_all)
        sel_row.addWidget(self._select_all_btn)
        self._deselect_all_btn = QPushButton(i18n.t("Deselect All"))
        self._deselect_all_btn.clicked.connect(self._deselect_all)
        sel_row.addWidget(self._deselect_all_btn)
        sel_row.addStretch(1)
        group_by_layout.addLayout(sel_row)

        layout.addWidget(group_by_group, 1)

        # ── Tiebreaker ────────────────────────────────────────────────
        tiebreaker_group = QGroupBox(i18n.t("Instance to select in each group:"))
        tiebreaker_layout = QVBoxLayout(tiebreaker_group)
        tiebreaker_layout.setContentsMargins(6, 6, 6, 6)
        tiebreaker_layout.setSpacing(4)

        self._tiebreaker_combo = QComboBox()
        self._tiebreaker_combo.addItems(list(TIEBREAKERS))
        self._tiebreaker_combo.setCurrentText("First instance")
        tiebreaker_layout.addWidget(self._tiebreaker_combo)

        layout.addWidget(tiebreaker_group)

        # ── Status bar: Input → Output (duplicates removed) ───────────
        status_layout = QHBoxLayout()
        status_layout.setContentsMargins(0, 0, 0, 0)
        status_layout.setSpacing(8)

        self._status_label = QLabel("")
        self._status_label.setStyleSheet(
            "font-size: 9pt; color: #6b5d50; background: transparent;"
        )
        self._status_label.setWordWrap(True)
        status_layout.addWidget(self._status_label, 1)
        layout.addLayout(status_layout)

        # ── Apply button ──────────────────────────────────────────────
        footer = QHBoxLayout()
        footer.setContentsMargins(0, 0, 0, 0)

        self.cb_apply_auto = QCheckBox(i18n.t("Apply Automatically"))
        self.cb_apply_auto.setChecked(True)
        footer.addWidget(self.cb_apply_auto)

        footer.addStretch(1)
        self._apply_button = QPushButton(i18n.t("Apply"))
        self._apply_button.setProperty("primary", True)
        self._apply_button.clicked.connect(self._apply)
        footer.addWidget(self._apply_button)
        layout.addLayout(footer)

        # ── Signal connections for auto-apply ─────────────────────
        self._column_list.itemSelectionChanged.connect(lambda: self._check_auto_apply())
        self._tiebreaker_combo.currentIndexChanged.connect(lambda: self._check_auto_apply())

    # ── Data pipeline ─────────────────────────────────────────────────

    def set_input_payload(self, payload) -> None:
        dataset = payload.dataset if payload is not None else None
        self._dataset_handle = dataset
        self._output_dataset = None
        self._output_removed = None
        self._output_annotated = None
        self._column_list.clear()
        self._filter_edit.clear()

        if dataset:
            self._dataset_label.setText(
                i18n.tf("Dataset: {name}", name=dataset.display_name)
            )
            # Populate column list: Target → Meta → Features hierarchy
            ordered_cols = (
                list(dataset.domain.target_columns)
                + list(dataset.domain.meta_columns)
                + list(dataset.domain.feature_columns)
            )
            for col in ordered_cols:
                role_tag = ""
                if col.role == "target":
                    role_tag = " [Target]"
                elif col.role == "meta":
                    role_tag = " [Meta]"
                display = f"{col.name}{role_tag}"
                item = QListWidgetItem(type_badge_icon(col.logical_type), display)
                item.setData(Qt.ItemDataRole.UserRole, col.name)
                item.setSelected(True)
                self._column_list.addItem(item)
            self._status_label.setText(
                i18n.tf("Input: {n} satır", n=dataset.row_count)
            )
        else:
            self._dataset_label.setText(i18n.t("Dataset: none"))
            self._status_label.setText("")

        if self._dataset_handle is not None:
            self._apply()

    def current_output_dataset(self) -> DatasetHandle | None:
        return self._output_dataset

    def current_output_datasets(self) -> dict[str, DatasetHandle | None] | None:
        return {
            "Unique Data": self._output_dataset,
            "Removed Duplicates": self._output_removed,
            "Annotated Data": self._output_annotated,
        }

    def serialize_node_state(self) -> dict[str, object]:
        selected = self._get_selected_col_names()
        return {
            "group_by": selected,
            "tiebreaker": self._tiebreaker_combo.currentText(),
            "auto_apply": self.cb_apply_auto.isChecked(),
        }

    def restore_node_state(self, payload: dict[str, object]) -> None:
        group_by = payload.get("group_by", [])
        if isinstance(group_by, list):
            group_set = set(group_by)
            for i in range(self._column_list.count()):
                item = self._column_list.item(i)
                col_name = item.data(Qt.ItemDataRole.UserRole)
                item.setSelected(col_name in group_set)
        tiebreaker = payload.get("tiebreaker", "First instance")
        if isinstance(tiebreaker, str):
            self._tiebreaker_combo.setCurrentText(tiebreaker)
        self.cb_apply_auto.setChecked(bool(payload.get("auto_apply", True)))

    def help_text(self) -> str:
        return "Filter the dataset to keep only unique rows based on selected columns."

    def documentation_url(self) -> str:
        return "https://orangedatamining.com/widget-catalog/transform/unique/"

    # ── Internal helpers ──────────────────────────────────────────

    def _check_auto_apply(self) -> None:
        if self._is_auto_apply() and self._dataset_handle is not None:
            self._apply()

    def _get_selected_col_names(self) -> list[str]:
        result = []
        for i in range(self._column_list.count()):
            item = self._column_list.item(i)
            if item.isSelected():
                result.append(item.data(Qt.ItemDataRole.UserRole))
        return result

    def _apply_filter(self, text: str) -> None:
        needle = text.lower().strip()
        for i in range(self._column_list.count()):
            item = self._column_list.item(i)
            col_name = item.data(Qt.ItemDataRole.UserRole) or ""
            item.setHidden(needle != "" and needle not in col_name.lower())

    def _select_all(self) -> None:
        for i in range(self._column_list.count()):
            self._column_list.item(i).setSelected(True)

    def _deselect_all(self) -> None:
        for i in range(self._column_list.count()):
            self._column_list.item(i).setSelected(False)

    def _apply(self) -> None:
        if self._dataset_handle is None:
            self._output_dataset = None
            self._output_removed = None
            self._output_annotated = None
            self._status_label.setText("")
            self._notify_output_changed()
            return

        selected_cols = self._get_selected_col_names()

        unique_data, removed_data, annotated_data = self._service.filter_unique(
            self._dataset_handle,
            group_by_columns=selected_cols,
            tiebreaker=self._tiebreaker_combo.currentText(),
        )
        self._output_dataset = unique_data
        self._output_removed = removed_data
        self._output_annotated = annotated_data

        before = self._dataset_handle.row_count
        after = self._output_dataset.row_count
        removed = before - after

        self._status_label.setText(
            i18n.tf(
                "Input: {before} satır → Output: {after} satır ({removed} tekrarlı satır kaldırıldı)",
                before=before,
                after=after,
                removed=removed,
            )
        )
        self._notify_output_changed()

    # ── i18n ──────────────────────────────────────────────────────────

    def refresh_translations(self) -> None:
        if self._dataset_handle:
            self._dataset_label.setText(
                i18n.tf(
                    "Dataset: {name}",
                    name=self._dataset_handle.display_name,
                )
            )
        else:
            self._dataset_label.setText(i18n.t("Dataset: none"))
        if self._output_dataset and self._dataset_handle:
            before = self._dataset_handle.row_count
            after = self._output_dataset.row_count
            removed = before - after
            self._status_label.setText(
                i18n.tf(
                    "Input: {before} satır → Output: {after} satır ({removed} tekrarlı satır kaldırıldı)",
                    before=before,
                    after=after,
                    removed=removed,
                )
            )
