from __future__ import annotations

import json
from PySide6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QRadioButton,
    QVBoxLayout,
    QWidget,
)
from PySide6.QtGui import QIcon, QPixmap, QPainter, QColor, QFont
from PySide6.QtCore import Qt, QRectF, QSize

from portakal_app.data.models import DatasetHandle
from portakal_app.data.services.merge_data_service import JOIN_TYPES, MergeDataService
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


class MatchRow:
    def __init__(self, parent_screen: MergeDataScreen, left_text: str = "", right_text: str = ""):
        self.parent_screen = parent_screen
        self.widget = QWidget()
        layout = QHBoxLayout(self.widget)
        layout.setContentsMargins(0, 0, 0, 0)
        
        self.left_combo = QComboBox()
        self.left_combo.setIconSize(QSize(16, 16))
        layout.addWidget(self.left_combo, 1)
        
        layout.addWidget(QLabel(i18n.t("matches")))
        
        self.right_combo = QComboBox()
        self.right_combo.setIconSize(QSize(16, 16))
        layout.addWidget(self.right_combo, 1)
        
        self.del_btn = QPushButton("x")
        self.del_btn.setFixedSize(24, 24)
        layout.addWidget(self.del_btn)
        
        parent_screen._populate_combo(self.left_combo, parent_screen._dataset_handle)
        parent_screen._populate_combo(self.right_combo, parent_screen._extra_handle)
        
        if left_text:
            idx = self.left_combo.findData(left_text)
            if idx < 0: idx = self.left_combo.findText(left_text)
            if idx >= 0: self.left_combo.setCurrentIndex(idx)
            
        if right_text:
            idx = self.right_combo.findData(right_text)
            if idx < 0: idx = self.right_combo.findText(right_text)
            if idx >= 0: self.right_combo.setCurrentIndex(idx)
            
        self.del_btn.clicked.connect(lambda: parent_screen._remove_match_row(self))


class MergeDataScreen(QWidget, WorkflowNodeScreenSupport):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._init_workflow_node_support()
        self._service = MergeDataService()
        self._dataset_handle: DatasetHandle | None = None
        self._extra_handle: DatasetHandle | None = None
        self._output_dataset: DatasetHandle | None = None
        
        self._match_rows: list[MatchRow] = []

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(10)

        self._dataset_label = QLabel(i18n.t("Data: none"))
        self._dataset_label.setProperty("sectionTitle", True)
        self._dataset_label.setStyleSheet("font-size: 12pt; background: transparent;")
        layout.addWidget(self._dataset_label)

        options_layout = QVBoxLayout()
        options_layout.setSpacing(6)
        
        self._join_group = QButtonGroup(self)
        
        self._radio_left = QRadioButton(i18n.t("Append columns from Extra Data"))
        self._radio_inner = QRadioButton(i18n.t("Find matching pairs of rows"))
        self._radio_outer = QRadioButton(i18n.t("Concatenate tables"))
        
        self._radio_left.setChecked(True)
        
        self._join_group.addButton(self._radio_left, 0)
        self._join_group.addButton(self._radio_inner, 1)
        self._join_group.addButton(self._radio_outer, 2)
        
        options_layout.addWidget(self._radio_left)
        options_layout.addWidget(self._radio_inner)
        options_layout.addWidget(self._radio_outer)
        
        layout.addLayout(options_layout)

        match_group = QGroupBox(i18n.t("Row matching"))
        match_group_layout = QVBoxLayout(match_group)
        match_group_layout.setContentsMargins(10, 10, 10, 10)
        match_group_layout.setSpacing(5)

        self._matches_layout = QVBoxLayout()
        self._matches_layout.setSpacing(5)
        match_group_layout.addLayout(self._matches_layout)

        plus_layout = QHBoxLayout()
        plus_layout.addStretch(1)
        self._add_match_btn = QPushButton("+")
        self._add_match_btn.setFixedSize(24, 24)
        self._add_match_btn.clicked.connect(lambda: self._add_match_row())
        plus_layout.addWidget(self._add_match_btn)
        match_group_layout.addLayout(plus_layout)

        layout.addWidget(match_group)

        self._data_info = QLabel(i18n.t("Data: -  |  Extra: -"))
        layout.addWidget(self._data_info)

        self._result_label = QLabel("")
        self._result_label.setWordWrap(True)
        layout.addWidget(self._result_label)

        layout.addStretch(1)

        # Auto-apply hooks
        self._join_group.idClicked.connect(self._check_auto_apply)

        footer = QHBoxLayout()
        self.cb_apply_auto = QCheckBox(i18n.t("Apply Automatically"))
        self.cb_apply_auto.setChecked(True)
        footer.addWidget(self.cb_apply_auto)
        footer.addStretch(1)
        self._apply_button = QPushButton(i18n.t("MERGE"))
        self._apply_button.setProperty("primary", True)
        self._apply_button.clicked.connect(self._apply)
        footer.addWidget(self._apply_button)
        layout.addLayout(footer)
        
        # Initialize with one empty row
        self._add_match_row()

    def set_input_payload(self, payload) -> None:
        if payload is None:
            self._dataset_handle = None
            self._extra_handle = None
        elif payload.port_label == "Data":
            self._dataset_handle = payload.dataset
        elif payload.port_label == "Extra Data":
            self._extra_handle = payload.dataset
        self._update_all_combos()
        # Auto-apply when both inputs are available
        if self._dataset_handle is not None and self._extra_handle is not None:
            self._apply()

    def current_output_dataset(self) -> DatasetHandle | None:
        return self._output_dataset

    def _check_auto_apply(self):
        if self.cb_apply_auto.isChecked():
            self._apply()

    def serialize_node_state(self) -> dict[str, object]:
        left_cols = [row.left_combo.currentData() for row in self._match_rows if row.left_combo.currentData()]
        right_cols = [row.right_combo.currentData() for row in self._match_rows if row.right_combo.currentData()]
        
        return {
            "left_on": json.dumps(left_cols),
            "right_on": json.dumps(right_cols),
            "join_type": self._join_group.checkedId(),
            "auto_apply": self.cb_apply_auto.isChecked(),
        }

    def restore_node_state(self, payload: dict[str, object]) -> None:
        try:
            left_cols = json.loads(str(payload.get("left_on", "[]")))
            right_cols = json.loads(str(payload.get("right_on", "[]")))
            if not isinstance(left_cols, list): left_cols = []
            if not isinstance(right_cols, list): right_cols = []
        except Exception:
            left_cols = []
            right_cols = []
            
        # Rebuild rows to match state length
        while len(self._match_rows) > 0:
            self._remove_match_row(self._match_rows[0], force=True)
            
        # Create new rows
        num_rows = max(1, len(left_cols))
        for i in range(num_rows):
            lt = left_cols[i] if i < len(left_cols) else ""
            rt = right_cols[i] if i < len(right_cols) else ""
            self._add_match_row(lt, rt)

        jt = int(payload.get("join_type", 0))
        btn = self._join_group.button(jt)
        if btn:
            btn.setChecked(True)
        self.cb_apply_auto.setChecked(bool(payload.get("auto_apply", True)))

    def help_text(self) -> str:
        return "Merge two datasets by matching one or more column pairs (left, inner, or outer join)."

    def documentation_url(self) -> str:
        return "https://orangedatamining.com/widget-catalog/transform/mergedata/"

    def _populate_combo(self, combo: QComboBox, dataset: DatasetHandle | None):
        combo.clear()
        if not dataset:
            return
        combo.addItem(QIcon(), i18n.t("Row index"), "Row index")
        combo.addItem(QIcon(), i18n.t("Instance id"), "Instance id")
        for col in dataset.domain.columns:
            combo.addItem(_create_type_icon(col.logical_type), col.name, col.name)
            
    def _add_match_row(self, left_text="", right_text=""):
        row = MatchRow(self, left_text, right_text)
        self._match_rows.append(row)
        self._matches_layout.addWidget(row.widget)
        
    def _remove_match_row(self, row, force=False):
        if len(self._match_rows) <= 1 and not force:
            return
        self._match_rows.remove(row)
        row.widget.deleteLater()

    def _update_all_combos(self) -> None:
        if self._dataset_handle:
            self._dataset_label.setText(i18n.tf("Data: {name}", name=self._dataset_handle.display_name))
        else:
            self._dataset_label.setText(i18n.t("Data: none"))

        for row in self._match_rows:
            lt = row.left_combo.currentData() or row.left_combo.currentText()
            rt = row.right_combo.currentData() or row.right_combo.currentText()
            self._populate_combo(row.left_combo, self._dataset_handle)
            self._populate_combo(row.right_combo, self._extra_handle)
            
            if lt:
                idx = row.left_combo.findData(lt)
                if idx < 0: idx = row.left_combo.findText(lt)
                if idx >= 0: row.left_combo.setCurrentIndex(idx)
            if rt:
                idx = row.right_combo.findData(rt)
                if idx < 0: idx = row.right_combo.findText(rt)
                if idx >= 0: row.right_combo.setCurrentIndex(idx)

        d_info = f"{self._dataset_handle.row_count}r" if self._dataset_handle else "-"
        e_info = f"{self._extra_handle.row_count}r" if self._extra_handle else "-"
        self._data_info.setText(i18n.tf("Data: {d}  |  Extra: {e}", d=d_info, e=e_info))

    def _apply(self) -> None:
        if self._dataset_handle is None or self._extra_handle is None:
            self._output_dataset = None
            self._result_label.setText("")
            self._notify_output_changed()
            return
            
        left_on = []
        right_on = []
        for row in self._match_rows:
            lt = row.left_combo.currentData()
            rt = row.right_combo.currentData()
            if lt and rt:
                left_on.append(lt)
                right_on.append(rt)
                
        if not left_on:
            self._result_label.setText(i18n.t("Please select valid columns to match."))
            return

        join_types = {0: "Left Join", 1: "Inner Join", 2: "Outer Join"}
        jt = join_types.get(self._join_group.checkedId(), "Left Join")

        try:
            self._output_dataset = self._service.merge(
                self._dataset_handle,
                self._extra_handle,
                left_on=left_on,
                right_on=right_on,
                join_type=jt,
            )
            self._result_label.setText(
                i18n.tf("Result: {rows} rows x {cols} columns", rows=self._output_dataset.row_count, cols=self._output_dataset.column_count)
            )
        except Exception as e:
            self._result_label.setText(i18n.tf("Merge Error: {err}", err=str(e)))
            self._output_dataset = None

        self._notify_output_changed()

    def refresh_translations(self) -> None:
        if self._dataset_handle:
            self._dataset_label.setText(
                i18n.tf("Data: {name}", name=self._dataset_handle.display_name)
            )
        else:
            self._dataset_label.setText(i18n.t("Data: none"))
        d_info = f"{self._dataset_handle.row_count}r" if self._dataset_handle else "-"
        e_info = f"{self._extra_handle.row_count}r" if self._extra_handle else "-"
        self._data_info.setText(i18n.tf("Data: {d}  |  Extra: {e}", d=d_info, e=e_info))
        if self._output_dataset is not None:
            self._result_label.setText(
                i18n.tf("Result: {rows} rows x {cols} columns", rows=self._output_dataset.row_count, cols=self._output_dataset.column_count)
            )
