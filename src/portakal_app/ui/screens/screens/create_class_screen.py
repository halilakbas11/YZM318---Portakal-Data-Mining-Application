from __future__ import annotations

from typing import Any
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
    QScrollArea,
)
from PySide6.QtCore import Qt

from portakal_app.data.models import DatasetHandle
from portakal_app.data.services.create_class_service import CreateClassService
from portakal_app.ui.screens.node_screen import WorkflowNodeScreenSupport
from portakal_app.ui import i18n

class _RuleRow(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.layout_hbox = QHBoxLayout(self)
        self.layout_hbox.setContentsMargins(0, 0, 0, 0)
        self.layout_hbox.setSpacing(6)

        self.remove_btn = QPushButton("×")
        self.remove_btn.setFixedWidth(24)
        self.layout_hbox.addWidget(self.remove_btn)

        self.label_edit = QLineEdit()
        self.layout_hbox.addWidget(self.label_edit, 1)

        self.pattern_edit = QLineEdit()
        self.layout_hbox.addWidget(self.pattern_edit, 1)

        self.count_label = QLabel("")
        self.count_label.setFixedWidth(40)
        self.count_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self.layout_hbox.addWidget(self.count_label)

    def get_rule(self) -> tuple[str, str]:
        return (self.label_edit.text(), self.pattern_edit.text())


class CreateClassScreen(QWidget, WorkflowNodeScreenSupport):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._init_workflow_node_support()
        self._service = CreateClassService()
        self._dataset_handle: DatasetHandle | None = None
        self._output_dataset: DatasetHandle | None = None
        self._rule_rows: list[_RuleRow] = []

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(10)

        # "Dataset: " display for Workflow node compat
        self._dataset_label = QLabel(i18n.t("Dataset: none"))
        self._dataset_label.setProperty("sectionTitle", True)
        self._dataset_label.setStyleSheet("font-size: 12pt; background: transparent;")
        layout.addWidget(self._dataset_label)

        # 1. New Class Name group
        name_group = QGroupBox(i18n.t("New Class Name"))
        name_layout = QVBoxLayout(name_group)
        name_layout.setContentsMargins(10, 10, 10, 10)
        self._class_name = QLineEdit("class")
        name_layout.addWidget(self._class_name)
        layout.addWidget(name_group)

        # 2. Match by Substring group
        match_group = QGroupBox(i18n.t("Match by Substring"))
        match_layout = QVBoxLayout(match_group)
        match_layout.setContentsMargins(10, 10, 10, 10)

        src_row = QHBoxLayout()
        src_row.addWidget(QLabel(i18n.t("From column:")))
        self._source_combo = QComboBox()
        self._source_combo.currentIndexChanged.connect(self._check_auto_apply)
        src_row.addWidget(self._source_combo, 1)
        match_layout.addLayout(src_row)
        
        # Grid Headers
        header_row = QHBoxLayout()
        header_row.setContentsMargins(0, 5, 0, 5)
        # spacer for x button
        spacer = QLabel("")
        spacer.setFixedWidth(24)
        header_row.addWidget(spacer)
        
        name_lbl = QLabel(i18n.t("Name"))
        header_row.addWidget(name_lbl, 1)
        sub_lbl = QLabel(i18n.t("Substring"))
        header_row.addWidget(sub_lbl, 1)
        count_lbl = QLabel(i18n.t("Count"))
        count_lbl.setFixedWidth(40)
        count_lbl.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        header_row.addWidget(count_lbl)
        
        match_layout.addLayout(header_row)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        self._rules_container = QWidget()
        self._rules_layout = QVBoxLayout(self._rules_container)
        self._rules_layout.setContentsMargins(0, 0, 0, 0)
        self._rules_layout.setSpacing(6)
        scroll.setWidget(self._rules_container)
        match_layout.addWidget(scroll, 1)
        
        add_btn_layout = QHBoxLayout()
        add_btn_layout.addStretch(1)
        self._add_btn = QPushButton("+")
        self._add_btn.setFixedWidth(30)
        self._add_btn.clicked.connect(self._add_rule)
        add_btn_layout.addWidget(self._add_btn)
        match_layout.addLayout(add_btn_layout)
        
        layout.addWidget(match_group, 1)

        # 3. Options Group
        opts_group = QGroupBox(i18n.t("Options"))
        opts_layout = QVBoxLayout(opts_group)
        opts_layout.setContentsMargins(10, 10, 10, 10)

        self._use_regex = QCheckBox(i18n.t("Use regular expressions"))
        self._use_regex.stateChanged.connect(self._check_auto_apply)
        opts_layout.addWidget(self._use_regex)

        self._match_beginning = QCheckBox(i18n.t("Match only at the beginning"))
        self._match_beginning.stateChanged.connect(self._check_auto_apply)
        opts_layout.addWidget(self._match_beginning)

        self._case_sensitive = QCheckBox(i18n.t("Case sensitive"))
        self._case_sensitive.stateChanged.connect(self._check_auto_apply)
        opts_layout.addWidget(self._case_sensitive)
        
        layout.addWidget(opts_group)

        self._result_label = QLabel("")
        self._result_label.setWordWrap(True)
        layout.addWidget(self._result_label)

        # Auto-apply hooks
        self._class_name.textChanged.connect(self._check_auto_apply)
        self._source_combo.currentIndexChanged.connect(self._check_auto_apply)
        self._use_regex.stateChanged.connect(self._check_auto_apply)
        self._match_beginning.stateChanged.connect(self._check_auto_apply)
        self._case_sensitive.stateChanged.connect(self._check_auto_apply)

        # Footer
        footer = QHBoxLayout()
        self.cb_apply_auto = QCheckBox(i18n.t("Apply Automatically"))
        self.cb_apply_auto.setChecked(True)
        footer.addWidget(self.cb_apply_auto)
        footer.addStretch(1)

        # Apply Button (Full width in Orange)
        self._apply_button = QPushButton(i18n.t("Apply"))
        self._apply_button.setProperty("primary", True)
        self._apply_button.clicked.connect(self._apply)
        footer.addWidget(self._apply_button)
        layout.addLayout(footer)
        
        # Add 2 initial empty rules to mimic Oranges initialization screenshot
        self._add_rule()
        self._add_rule()

    def set_input_payload(self, payload) -> None:
        dataset = payload.dataset if payload is not None else None
        self._dataset_handle = dataset
        self._output_dataset = None
        self._source_combo.blockSignals(True)
        self._source_combo.clear()

        if dataset:
            self._dataset_label.setText(i18n.tf("Dataset: {name}", name=dataset.display_name))
            for col in dataset.domain.columns:
                if col.logical_type not in ("categorical", "text"):
                    continue
                type_label = "Cat" if col.logical_type == "categorical" else "Txt"
                self._source_combo.addItem(f"{col.name} ({type_label})", userData=col.name)
        else:
            self._dataset_label.setText(i18n.t("Dataset: none"))
            self._result_label.setText("")

        self._source_combo.blockSignals(False)
        self._apply()

    def _check_auto_apply(self) -> None:
        if self.cb_apply_auto.isChecked():
            self._apply()

    def current_output_dataset(self) -> DatasetHandle | None:
        return self._output_dataset

    def serialize_node_state(self) -> dict[str, object]:
        rules = [row.get_rule() for row in self._rule_rows]
        return {
            "source": self._source_combo.currentData(),
            "class_name": self._class_name.text(),
            "case_sensitive": self._case_sensitive.isChecked(),
            "use_regex": self._use_regex.isChecked(),
            "match_beginning": self._match_beginning.isChecked(),
            "rules": rules,
            "auto_apply": self.cb_apply_auto.isChecked(),
        }

    def restore_node_state(self, payload: dict[str, object]) -> None:
        src_name = payload.get("source")
        if src_name:
             for i in range(self._source_combo.count()):
                 if self._source_combo.itemData(i) == src_name:
                     self._source_combo.setCurrentIndex(i)
                     break
        
        self._class_name.setText(str(payload.get("class_name", "class")))
        self._case_sensitive.setChecked(bool(payload.get("case_sensitive", False)))
        self._use_regex.setChecked(bool(payload.get("use_regex", False)))
        self._match_beginning.setChecked(bool(payload.get("match_beginning", False)))
        self.cb_apply_auto.setChecked(bool(payload.get("auto_apply", True)))

        rules = payload.get("rules", [])
        if rules:
            for row in list(self._rule_rows):
                self._remove_rule(row)
            for lbl, pat in rules:
                self._add_rule()
                r = self._rule_rows[-1]
                r.label_edit.setText(str(lbl))
                r.pattern_edit.setText(str(pat))

    def help_text(self) -> str:
        return "Create a new categorical class/target feature by slicing values based on substrings or regular expressions."

    def documentation_url(self) -> str:
        return "https://orangedatamining.com/widget-catalog/transform/createclass/"

    def refresh_translations(self) -> None:
        if self._dataset_handle is None:
            self._dataset_label.setText(i18n.t("Dataset: none"))
        else:
            self._dataset_label.setText(i18n.tf("Dataset: {name}", name=self._dataset_handle.display_name))

    def _add_rule(self) -> None:
        row = _RuleRow(self)
        row.remove_btn.clicked.connect(lambda: self._remove_rule(row))
        # Trigger re-eval when text edited only if auto-update is desired
        # row.label_edit.textEdited.connect(self._check_apply)
        # row.pattern_edit.textEdited.connect(self._check_apply)
        self._rule_rows.append(row)
        self._rules_layout.addWidget(row)

    def _remove_rule(self, row: _RuleRow) -> None:
        if row in self._rule_rows:
            self._rule_rows.remove(row)
            self._rules_layout.removeWidget(row)
            row.deleteLater()
            self._check_auto_apply()

    def _apply(self) -> None:
        if self._dataset_handle is None or self._source_combo.count() == 0:
            self._output_dataset = None
            self._result_label.setText("")
            self._notify_output_changed()
            return

        source_col = self._source_combo.currentData()
        rules = [row.get_rule() for row in self._rule_rows if row.label_edit.text() or row.pattern_edit.text()]

        try:
            self._output_dataset, counts = self._service.create_class(
                self._dataset_handle,
                source_column=str(source_col),
                rules=rules,
                class_name=self._class_name.text() or "class",
                case_sensitive=self._case_sensitive.isChecked(),
                use_regex=self._use_regex.isChecked(),
                match_beginning=self._match_beginning.isChecked(),
            )
            
            # Update counts on rows
            valid_idx = 0
            for row in self._rule_rows:
                if row.label_edit.text() or row.pattern_edit.text():
                    row.count_label.setText(str(counts.get(valid_idx, 0)))
                    valid_idx += 1
                else:
                    row.count_label.setText("")

            unique_classes = self._output_dataset.dataframe.get_column(
                self._class_name.text() or "class"
            ).n_unique()
            self._result_label.setText(i18n.tf("Created '{name}' with {count} matching categories", name=self._class_name.text(), count=unique_classes))
        except Exception as e:
            self._output_dataset = None
            self._result_label.setText(i18n.tf("Error mapping class: {error}", error=e))
            for row in self._rule_rows:
                row.count_label.setText("")

        self._notify_output_changed()
