from __future__ import annotations

from PySide6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QRadioButton,
    QVBoxLayout,
    QWidget,
)
from PySide6.QtGui import QIcon, QPixmap, QPainter, QColor, QFont
from PySide6.QtCore import Qt, QRectF

from portakal_app.data.models import DatasetHandle
from portakal_app.data.services.split_service import OUTPUT_TYPES, SplitService
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


class SplitScreen(QWidget, WorkflowNodeScreenSupport):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._init_workflow_node_support()
        self._service = SplitService()
        self._dataset_handle: DatasetHandle | None = None
        self._output_dataset: DatasetHandle | None = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(10)

        self._dataset_label = QLabel(i18n.t("Dataset: none"))
        self._dataset_label.setProperty("sectionTitle", True)
        self._dataset_label.setStyleSheet("font-size: 12pt; background: transparent;")
        layout.addWidget(self._dataset_label)

        # Variable Group
        var_group = QGroupBox(i18n.t("Variable"))
        var_layout = QVBoxLayout(var_group)
        var_layout.setContentsMargins(10, 10, 10, 10)
        var_layout.setSpacing(8)

        self._column_combo = QComboBox()
        var_layout.addWidget(self._column_combo)

        delim_row = QHBoxLayout()
        delim_row.addWidget(QLabel(i18n.t("Delimiter:")))
        self._delimiter_edit = QLineEdit(";")
        self._delimiter_edit.setMaximumWidth(40)
        delim_row.addWidget(self._delimiter_edit)
        delim_row.addStretch(1)
        var_layout.addLayout(delim_row)

        layout.addWidget(var_group)

        # Output Values Group
        out_group = QGroupBox(i18n.t("Output Values"))
        out_layout = QVBoxLayout(out_group)
        out_layout.setContentsMargins(10, 10, 10, 10)
        out_layout.setSpacing(6)

        self._type_group = QButtonGroup(self)
        
        self._radio_cat = QRadioButton(i18n.t("Categorical (No, Yes)"))
        self._radio_num = QRadioButton(i18n.t("Numerical (0, 1)"))
        self._radio_counts = QRadioButton(i18n.t("Counts"))
        
        self._type_group.addButton(self._radio_cat, 0)
        self._type_group.addButton(self._radio_num, 1)
        self._type_group.addButton(self._radio_counts, 2)
        
        out_layout.addWidget(self._radio_cat)
        out_layout.addWidget(self._radio_num)
        out_layout.addWidget(self._radio_counts)
        self._radio_num.setChecked(True)

        layout.addWidget(out_group)

        self._result_label = QLabel("")
        self._result_label.setWordWrap(True)
        layout.addWidget(self._result_label)

        layout.addStretch(1)

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

        # ── Signal connections for auto-apply ─────────────────────
        self._column_combo.currentIndexChanged.connect(lambda: self._check_auto_apply())
        self._delimiter_edit.textChanged.connect(lambda: self._check_auto_apply())
        self._type_group.idClicked.connect(lambda: self._check_auto_apply())

    def set_input_payload(self, payload) -> None:
        dataset = payload.dataset if payload is not None else None
        self._dataset_handle = dataset
        self._output_dataset = None
        self._column_combo.clear()

        if dataset:
            self._dataset_label.setText(i18n.tf("Dataset: {name}", name=dataset.display_name))
            for col in dataset.domain.columns:
                if col.logical_type in ("text", "categorical"):
                    self._column_combo.addItem(_create_type_icon(col.logical_type), col.name)
        else:
            self._dataset_label.setText(i18n.t("Dataset: none"))
            self._result_label.setText("")

        if self._dataset_handle is not None:
            self._apply()

    def current_output_dataset(self) -> DatasetHandle | None:
        return self._output_dataset

    def serialize_node_state(self) -> dict[str, object]:
        return {
            "column": self._column_combo.currentText(),
            "delimiter": self._delimiter_edit.text(),
            "output_type": self._get_selected_type(),
            "auto_apply": self.cb_apply_auto.isChecked(),
        }
        
    def _get_selected_type(self) -> str:
        if self._radio_cat.isChecked():
            return "Categorical (No, Yes)"
        if self._radio_num.isChecked():
            return "Numerical (0, 1)"
        if self._radio_counts.isChecked():
            return "Counts"
        return "Numerical (0, 1)"

    def restore_node_state(self, payload: dict[str, object]) -> None:
        col = str(payload.get("column", ""))
        if col and self._column_combo.findText(col) >= 0:
            self._column_combo.setCurrentText(col)
        self._delimiter_edit.setText(str(payload.get("delimiter", ";")))
        ot = str(payload.get("output_type", "Numerical (0, 1)"))
        
        if ot == "Categorical (No, Yes)":
            self._radio_cat.setChecked(True)
        elif ot == "Counts":
            self._radio_counts.setChecked(True)
        else:
            self._radio_num.setChecked(True)
        self.cb_apply_auto.setChecked(bool(payload.get("auto_apply", True)))

    def help_text(self) -> str:
        return "Split a string column by a delimiter and create indicator columns for each unique value."

    def documentation_url(self) -> str:
        return "https://orangedatamining.com/widget-catalog/transform/split/"

    def _check_auto_apply(self) -> None:
        if self._is_auto_apply() and self._dataset_handle is not None:
            self._apply()

    def _apply(self) -> None:
        if self._dataset_handle is None or not self._column_combo.currentText():
            self._output_dataset = None
            self._result_label.setText("")
            self._notify_output_changed()
            return

        self._output_dataset = self._service.split_column(
            self._dataset_handle,
            column_name=self._column_combo.currentText(),
            delimiter=self._delimiter_edit.text() or ";",
            output_type=self._get_selected_type(),
        )

        new_cols = self._output_dataset.column_count - self._dataset_handle.column_count
        self._result_label.setText(i18n.tf("Added {count} indicator column(s)", count=new_cols))
        self._notify_output_changed()

    def refresh_translations(self) -> None:
        if self._dataset_handle is None:
            self._dataset_label.setText(i18n.t("Dataset: none"))
        else:
            self._dataset_label.setText(
                i18n.tf("Dataset: {name}", name=self._dataset_handle.display_name)
            )
        if self._output_dataset is not None and self._dataset_handle is not None:
            new_cols = self._output_dataset.column_count - self._dataset_handle.column_count
            self._result_label.setText(
                i18n.tf("Added {count} indicator column(s)", count=new_cols)
            )
