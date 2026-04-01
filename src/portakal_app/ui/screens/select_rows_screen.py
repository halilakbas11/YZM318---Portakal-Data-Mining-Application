from __future__ import annotations

from PySide6.QtWidgets import (
    QAbstractItemView,
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)
from PySide6.QtGui import QIcon, QPixmap, QPainter, QColor, QFont
from PySide6.QtCore import Qt, QRectF

from portakal_app.data.models import DatasetHandle
from portakal_app.data.services.select_rows_service import (
    DUAL_VALUE_OPS,
    NO_VALUE_OPS,
    OPERATORS_CATEGORICAL,
    OPERATORS_NUMERIC,
    OPERATORS_STRING,
    SelectRowsService,
)
from portakal_app.ui import i18n
from portakal_app.ui.screens.node_screen import WorkflowNodeScreenSupport


def _create_type_icon(logical_type: str) -> QIcon:
    pixmap = QPixmap(16, 16)
    pixmap.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)

    if logical_type == "numeric":
        color = QColor("#ef4444")
        text = "N"
    elif logical_type in ("categorical", "boolean"):
        color = QColor("#22c55e")
        text = "C"
    elif logical_type in ("text", "string"):
        color = QColor("#8b5cf6")
        text = "S"
    elif logical_type in ("datetime", "date", "time"):
        color = QColor("#3b82f6")
        text = "D"
    else:
        color = QColor("#6b7280")
        text = "?"

    painter.setBrush(color)
    painter.setPen(Qt.PenStyle.NoPen)
    painter.drawRoundedRect(0, 0, 16, 16, 3, 3)

    painter.setPen(QColor("white"))
    font = QFont("Arial", 9, QFont.Weight.Bold)
    painter.setFont(font)
    painter.drawText(QRectF(0, 0, 16, 16), Qt.AlignmentFlag.AlignCenter, text)
    painter.end()

    return QIcon(pixmap)


# ─────────────────────────────────────────────────────────────────────────────
#  Single condition row  (compact, type-aware)
# ─────────────────────────────────────────────────────────────────────────────


class _ConditionRow(QWidget):
    """One condition row: [Column ▼] [Operator ▼] [Value …] [×]"""

    def __init__(
        self,
        columns: list[tuple[str, str, tuple[str, ...]]],
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        self._columns = {
            name: (ltype, samples) for name, ltype, samples in columns
        }

        # Column selector
        self._col_combo = QComboBox()
        self._col_combo.setMinimumWidth(100)
        for name, ltype, _ in columns:
            self._col_combo.addItem(_create_type_icon(ltype), name)
        self._col_combo.currentTextChanged.connect(self._on_col_changed)
        layout.addWidget(self._col_combo, 2)

        # Operator selector
        self._op_combo = QComboBox()
        self._op_combo.setMinimumWidth(90)
        self._op_combo.currentTextChanged.connect(self._on_op_changed)
        layout.addWidget(self._op_combo, 2)

        # Primary value input (editable combo for numeric/text, dropdown for categorical)
        self._value_combo = QComboBox()
        self._value_combo.setEditable(True)
        self._value_combo.setMinimumWidth(80)
        layout.addWidget(self._value_combo, 2)

        # Secondary value input (for "is between" / "is outside")
        self._value2_combo = QComboBox()
        self._value2_combo.setEditable(True)
        self._value2_combo.setMinimumWidth(80)
        self._value2_combo.setVisible(False)
        layout.addWidget(self._value2_combo, 2)

        # "is one of" multi-select list
        self._multi_list = QListWidget()
        self._multi_list.setSelectionMode(
            QAbstractItemView.SelectionMode.MultiSelection
        )
        self._multi_list.setMaximumHeight(80)
        self._multi_list.setVisible(False)
        layout.addWidget(self._multi_list, 3)

        # Remove button
        self._remove_btn = QPushButton("×")
        self._remove_btn.setFixedSize(24, 24)
        self._remove_btn.setStyleSheet(
            "font-size: 14pt; border: none; color: #999;"
        )
        layout.addWidget(self._remove_btn)

        self._on_col_changed(self._col_combo.currentText())

    # ── Column type changed → rebuild operators ──────────────────────
    def _on_col_changed(self, col_name: str) -> None:
        self._op_combo.blockSignals(True)
        self._op_combo.clear()
        ltype, samples = self._columns.get(col_name, ("text", ()))

        if ltype == "numeric":
            self._op_combo.addItems(list(OPERATORS_NUMERIC))
        elif ltype in ("categorical", "boolean"):
            self._op_combo.addItems(list(OPERATORS_CATEGORICAL))
        else:
            self._op_combo.addItems(list(OPERATORS_STRING))

        self._op_combo.blockSignals(False)

        # Store samples for "is one of" and categorical value dropdown
        self._current_samples = samples
        self._current_ltype = ltype

        self._on_op_changed(self._op_combo.currentText())

    # ── Operator changed → show/hide value fields ────────────────────
    def _on_op_changed(self, op_text: str) -> None:
        ltype = getattr(self, "_current_ltype", "text")
        samples = getattr(self, "_current_samples", ())

        is_no_value = op_text in NO_VALUE_OPS
        is_dual = op_text in DUAL_VALUE_OPS
        is_one_of = op_text == "is one of"

        # Primary value
        self._value_combo.setVisible(not is_no_value and not is_one_of)
        self._value2_combo.setVisible(is_dual)
        self._multi_list.setVisible(is_one_of)

        if is_one_of:
            self._multi_list.clear()
            for val in samples:
                item = QListWidgetItem(val)
                self._multi_list.addItem(item)
        elif not is_no_value:
            self._value_combo.clear()
            if ltype in ("categorical", "boolean"):
                self._value_combo.setEditable(False)
                self._value_combo.addItems(list(samples))
            else:
                self._value_combo.setEditable(True)

    # ── Read condition from this row ─────────────────────────────────
    def get_condition(self) -> tuple[str, str, str]:
        col = self._col_combo.currentText()
        op = self._op_combo.currentText()

        if op in NO_VALUE_OPS:
            return (col, op, "")
        if op == "is one of":
            selected = [
                self._multi_list.item(i).text()
                for i in range(self._multi_list.count())
                if self._multi_list.item(i).isSelected()
            ]
            return (col, op, ",".join(selected))
        if op in DUAL_VALUE_OPS:
            v1 = self._value_combo.currentText()
            v2 = self._value2_combo.currentText()
            return (col, op, f"{v1};{v2}")

        return (col, op, self._value_combo.currentText())

    # ── Restore a previously saved condition ─────────────────────────
    def set_condition(self, col: str, op: str, value: str) -> None:
        idx = self._col_combo.findText(col)
        if idx >= 0:
            self._col_combo.setCurrentIndex(idx)

        idx = self._op_combo.findText(op)
        if idx >= 0:
            self._op_combo.setCurrentIndex(idx)

        if op == "is one of":
            vals = {v.strip() for v in value.split(",") if v.strip()}
            for i in range(self._multi_list.count()):
                item = self._multi_list.item(i)
                item.setSelected(item.text() in vals)
        elif op in DUAL_VALUE_OPS:
            parts = value.split(";")
            if len(parts) == 2:
                self._value_combo.setEditText(parts[0].strip())
                self._value2_combo.setEditText(parts[1].strip())
        elif op not in NO_VALUE_OPS:
            if self._value_combo.isEditable():
                self._value_combo.setEditText(value)
            else:
                vi = self._value_combo.findText(value)
                if vi >= 0:
                    self._value_combo.setCurrentIndex(vi)


# ─────────────────────────────────────────────────────────────────────────────
#  Main screen
# ─────────────────────────────────────────────────────────────────────────────


class SelectRowsScreen(QWidget, WorkflowNodeScreenSupport):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._init_workflow_node_support()
        self._service = SelectRowsService()
        self._dataset_handle: DatasetHandle | None = None
        self._output_dataset: DatasetHandle | None = None
        self._unmatched_dataset: DatasetHandle | None = None
        self._condition_rows: list[_ConditionRow] = []

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

        # ── AND / OR logic selector ───────────────────────────────────
        logic_layout = QHBoxLayout()
        logic_layout.setContentsMargins(0, 0, 0, 0)
        logic_layout.setSpacing(8)

        self._logic_group = QButtonGroup(self)
        self._radio_and = QRadioButton(i18n.t("All conditions must match (AND)"))
        self._radio_and.setChecked(True)
        self._logic_group.addButton(self._radio_and, 0)
        logic_layout.addWidget(self._radio_and)

        self._radio_or = QRadioButton(i18n.t("At least one condition (OR)"))
        self._logic_group.addButton(self._radio_or, 1)
        logic_layout.addWidget(self._radio_or)
        logic_layout.addStretch(1)
        layout.addLayout(logic_layout)

        # ── Conditions group (scrollable) ─────────────────────────────
        cond_group = QGroupBox(i18n.t("Conditions"))
        cond_group_layout = QVBoxLayout(cond_group)
        cond_group_layout.setContentsMargins(6, 6, 6, 6)
        cond_group_layout.setSpacing(0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)

        self._cond_container = QWidget()
        self._cond_layout = QVBoxLayout(self._cond_container)
        self._cond_layout.setContentsMargins(0, 0, 0, 0)
        self._cond_layout.setSpacing(3)
        self._cond_layout.addStretch(1)
        scroll.setWidget(self._cond_container)

        cond_group_layout.addWidget(scroll)
        layout.addWidget(cond_group, 1)

        # ── Button row ────────────────────────────────────────────────
        btn_row = QHBoxLayout()
        btn_row.setContentsMargins(0, 0, 0, 0)
        btn_row.setSpacing(6)
        self._add_btn = QPushButton(i18n.t("+ Add Condition"))
        self._add_btn.clicked.connect(self._add_condition)
        btn_row.addWidget(self._add_btn)
        self._clear_btn = QPushButton(i18n.t("Remove All"))
        self._clear_btn.clicked.connect(self._clear_conditions)
        btn_row.addWidget(self._clear_btn)
        btn_row.addStretch(1)
        layout.addLayout(btn_row)

        # ── Purge checkboxes (Orange compatibility) ───────────────────
        self._purge_attrs_cb = QCheckBox(
            i18n.t("Remove unused values and constant features")
        )
        layout.addWidget(self._purge_attrs_cb)

        self._purge_classes_cb = QCheckBox(
            i18n.t("Remove unused classes")
        )
        layout.addWidget(self._purge_classes_cb)

        # ── Status bar: Selected | Remaining | Total ──────────────────
        status_layout = QHBoxLayout()
        status_layout.setContentsMargins(0, 0, 0, 0)
        status_layout.setSpacing(16)

        self._selected_label = QLabel("")
        self._selected_label.setStyleSheet(
            "font-size: 9pt; font-weight: bold; color: #2e7d32; background: transparent;"
        )
        status_layout.addWidget(self._selected_label)

        self._remaining_label = QLabel("")
        self._remaining_label.setStyleSheet(
            "font-size: 9pt; color: #c75000; background: transparent;"
        )
        status_layout.addWidget(self._remaining_label)

        self._total_label = QLabel("")
        self._total_label.setStyleSheet(
            "font-size: 9pt; color: #6b5d50; background: transparent;"
        )
        status_layout.addWidget(self._total_label)
        status_layout.addStretch(1)
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
        self._logic_group.idClicked.connect(lambda: self._check_auto_apply())
        self._purge_attrs_cb.stateChanged.connect(lambda: self._check_auto_apply())
        self._purge_classes_cb.stateChanged.connect(lambda: self._check_auto_apply())

    # ── Data pipeline ─────────────────────────────────────────────────

    def set_input_payload(self, payload) -> None:
        dataset = payload.dataset if payload is not None else None
        self._dataset_handle = dataset
        self._output_dataset = None
        self._unmatched_dataset = None
        self._clear_conditions()

        if dataset:
            self._dataset_label.setText(
                i18n.tf("Dataset: {name}", name=dataset.display_name)
            )
            self._total_label.setText(
                i18n.tf("Total: {n}", n=dataset.row_count)
            )
        else:
            self._dataset_label.setText(i18n.t("Dataset: none"))
            self._clear_status()

        if self._dataset_handle is not None:
            self._apply()

    def current_output_dataset(self) -> DatasetHandle | None:
        return self._output_dataset

    def current_output_datasets(self) -> dict[str, DatasetHandle | None] | None:
        return {
            "Matching Data": self._output_dataset,
            "Unmatched Data": self._unmatched_dataset,
        }

    def serialize_node_state(self) -> dict[str, object]:
        conditions = [row.get_condition() for row in self._condition_rows]
        return {
            "conditions": conditions,
            "conjunction": "any" if self._radio_or.isChecked() else "all",
            "purge_attributes": self._purge_attrs_cb.isChecked(),
            "purge_classes": self._purge_classes_cb.isChecked(),
            "auto_apply": self.cb_apply_auto.isChecked(),
        }

    def restore_node_state(self, payload: dict[str, object]) -> None:
        # Restore conjunction
        conj = payload.get("conjunction", "all")
        if conj == "any":
            self._radio_or.setChecked(True)
        else:
            self._radio_and.setChecked(True)

        # Restore purge checkboxes
        self._purge_attrs_cb.setChecked(bool(payload.get("purge_attributes", False)))
        self._purge_classes_cb.setChecked(bool(payload.get("purge_classes", False)))
        self.cb_apply_auto.setChecked(bool(payload.get("auto_apply", True)))

        # Restore conditions
        saved_conditions = payload.get("conditions", [])
        if saved_conditions and self._dataset_handle:
            columns = self._get_columns()
            for col, op, val in saved_conditions:
                row = _ConditionRow(columns, self)
                row._remove_btn.clicked.connect(
                    lambda checked=False, r=row: self._remove_condition(r)
                )
                row.set_condition(col, op, val)
                self._condition_rows.append(row)
                self._cond_layout.insertWidget(
                    self._cond_layout.count() - 1, row
                )

    def help_text(self) -> str:
        return (
            "Filter rows using conditions on column values. "
            "Outputs matching and unmatched subsets."
        )

    def documentation_url(self) -> str:
        return "https://orangedatamining.com/widget-catalog/transform/selectrows/"

    # ── Internals ─────────────────────────────────────────────────────

    def _check_auto_apply(self) -> None:
        if self._is_auto_apply() and self._dataset_handle is not None:
            self._apply()

    def _get_columns(self) -> list[tuple[str, str, tuple[str, ...]]]:
        if self._dataset_handle is None:
            return []

        columns_data = []
        for col in self._dataset_handle.domain.columns:
            samples: tuple[str, ...] = ()
            if col.logical_type in ("categorical", "boolean"):
                series = (
                    self._dataset_handle.dataframe.get_column(col.name).drop_nulls()
                )
                unique_vals = [str(x) for x in series.unique().to_list()]
                samples = tuple(sorted(unique_vals))
            columns_data.append((col.name, col.logical_type, samples))
        return columns_data

    def _add_condition(self) -> None:
        columns = self._get_columns()
        if not columns:
            return
        row = _ConditionRow(columns, self)
        row._remove_btn.clicked.connect(
            lambda checked=False, r=row: self._remove_condition(r)
        )
        self._condition_rows.append(row)
        # Insert before the stretch
        self._cond_layout.insertWidget(self._cond_layout.count() - 1, row)

    def _remove_condition(self, row: _ConditionRow) -> None:
        if row in self._condition_rows:
            self._condition_rows.remove(row)
            self._cond_layout.removeWidget(row)
            row.deleteLater()

    def _clear_conditions(self) -> None:
        for row in self._condition_rows:
            self._cond_layout.removeWidget(row)
            row.deleteLater()
        self._condition_rows.clear()

    def _clear_status(self) -> None:
        self._selected_label.setText("")
        self._remaining_label.setText("")
        self._total_label.setText("")

    def _apply(self) -> None:
        if self._dataset_handle is None:
            self._output_dataset = None
            self._unmatched_dataset = None
            self._clear_status()
            self._notify_output_changed()
            return

        conditions = [row.get_condition() for row in self._condition_rows]
        conjunction = "any" if self._radio_or.isChecked() else "all"

        matching, non_matching = self._service.filter_rows(
            self._dataset_handle,
            conditions=conditions,
            conjunction=conjunction,
            purge_attributes=self._purge_attrs_cb.isChecked(),
            purge_classes=self._purge_classes_cb.isChecked(),
        )

        self._output_dataset = matching
        self._unmatched_dataset = non_matching

        m = matching.row_count if matching else 0
        nm = non_matching.row_count if non_matching else 0
        total = self._dataset_handle.row_count

        self._selected_label.setText(i18n.tf("Selected: {n}", n=m))
        self._remaining_label.setText(i18n.tf("Remaining: {n}", n=nm))
        self._total_label.setText(i18n.tf("Total: {n}", n=total))

        self._notify_output_changed()

    # ── i18n ──────────────────────────────────────────────────────────

    def refresh_translations(self) -> None:
        if self._dataset_handle is None:
            self._dataset_label.setText(i18n.t("Dataset: none"))
        else:
            self._dataset_label.setText(
                i18n.tf(
                    "Dataset: {name}", name=self._dataset_handle.display_name
                )
            )
        if self._output_dataset is not None:
            m = self._output_dataset.row_count
            nm = (
                self._unmatched_dataset.row_count
                if self._unmatched_dataset
                else 0
            )
            total = self._dataset_handle.row_count if self._dataset_handle else 0
            self._selected_label.setText(i18n.tf("Selected: {n}", n=m))
            self._remaining_label.setText(i18n.tf("Remaining: {n}", n=nm))
            self._total_label.setText(i18n.tf("Total: {n}", n=total))
