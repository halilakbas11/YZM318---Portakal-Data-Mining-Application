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
from portakal_app.data.services.transpose_service import TransposeService
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


class TransposeScreen(QWidget, WorkflowNodeScreenSupport):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._init_workflow_node_support()
        self._service = TransposeService()
        self._dataset_handle: DatasetHandle | None = None
        self._output_dataset: DatasetHandle | None = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(10)

        self._dataset_label = QLabel(i18n.t("Dataset: none"))
        self._dataset_label.setProperty("sectionTitle", True)
        self._dataset_label.setStyleSheet("font-size: 12pt; background: transparent;")
        layout.addWidget(self._dataset_label)

        # ── Feature names group ─────────────────────────────────────────
        names_group = QGroupBox(i18n.t("Feature names"))
        names_layout = QVBoxLayout(names_group)
        names_layout.setContentsMargins(10, 10, 10, 10)
        names_layout.setSpacing(8)

        self._name_mode_group = QButtonGroup(self)
        
        self._radio_generic = QRadioButton(i18n.t("Generic"))
        self._radio_generic.setChecked(True)
        self._name_mode_group.addButton(self._radio_generic, 0)
        names_layout.addWidget(self._radio_generic)

        prefix_row = QHBoxLayout()
        prefix_row.setContentsMargins(20, 0, 0, 4)
        self._prefix_edit = QLineEdit("Feature")
        self._prefix_edit.setPlaceholderText(i18n.t("Type a prefix ..."))
        prefix_row.addWidget(self._prefix_edit)
        names_layout.addLayout(prefix_row)

        self._radio_from_col = QRadioButton(i18n.t("From variable:"))
        self._name_mode_group.addButton(self._radio_from_col, 1)
        names_layout.addWidget(self._radio_from_col)

        col_row = QHBoxLayout()
        col_row.setContentsMargins(20, 0, 0, 4)
        self._column_combo = QComboBox()
        col_row.addWidget(self._column_combo, 1)
        names_layout.addLayout(col_row)

        # ── Remove redundant instance checkbox ──────────────────────────
        self._remove_redundant_check = QCheckBox(
            i18n.t("Remove redundant instance")
        )
        self._remove_redundant_check.setChecked(False)
        names_layout.addWidget(self._remove_redundant_check)

        layout.addWidget(names_group)

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

        self._radio_generic.toggled.connect(self._update_ui_state)
        self._radio_from_col.toggled.connect(self._update_ui_state)
        self._update_ui_state()

        # ── Signal connections for auto-apply ─────────────────────
        self._name_mode_group.idClicked.connect(lambda: self._check_auto_apply())
        self._prefix_edit.textChanged.connect(lambda: self._check_auto_apply())
        self._column_combo.currentIndexChanged.connect(lambda: self._check_auto_apply())
        self._remove_redundant_check.stateChanged.connect(lambda: self._check_auto_apply())

    def _update_ui_state(self) -> None:
        is_generic = self._radio_generic.isChecked()
        self._prefix_edit.setEnabled(is_generic)
        self._column_combo.setEnabled(not is_generic)
        # "Remove redundant instance" only makes sense in "From variable" mode
        self._remove_redundant_check.setEnabled(not is_generic)

    def set_input_payload(self, payload) -> None:
        dataset = payload.dataset if payload is not None else None
        self._dataset_handle = dataset
        self._output_dataset = None
        self._column_combo.clear()

        if dataset:
            self._dataset_label.setText(i18n.tf("Dataset: {name}", name=dataset.display_name))
            for col in dataset.domain.columns:
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
            "name_mode": self._name_mode_group.checkedId(),
            "prefix": self._prefix_edit.text(),
            "from_column": self._column_combo.currentText(),
            "remove_redundant": self._remove_redundant_check.isChecked(),
            "auto_apply": self.cb_apply_auto.isChecked(),
        }

    def restore_node_state(self, payload: dict[str, object]) -> None:
        mode_id = int(payload.get("name_mode", 0))
        btn = self._name_mode_group.button(mode_id)
        if btn:
            btn.setChecked(True)
        self._prefix_edit.setText(str(payload.get("prefix", "Feature")))
        col = str(payload.get("from_column", ""))
        if col and self._column_combo.findText(col) >= 0:
            self._column_combo.setCurrentText(col)
        self._remove_redundant_check.setChecked(
            bool(payload.get("remove_redundant", False))
        )
        self.cb_apply_auto.setChecked(bool(payload.get("auto_apply", True)))
        self._update_ui_state()

    def help_text(self) -> str:
        return "Transpose the dataset: flip rows and columns."

    def documentation_url(self) -> str:
        return "https://orangedatamining.com/widget-catalog/transform/transpose/"

    def _check_auto_apply(self) -> None:
        if self._is_auto_apply() and self._dataset_handle is not None:
            self._apply()

    def _apply(self) -> None:
        if self._dataset_handle is None:
            self._output_dataset = None
            self._result_label.setText("")
            self._notify_output_changed()
            return

        from_col = None
        if self._name_mode_group.checkedId() == 1 and self._column_combo.currentText():
            from_col = self._column_combo.currentText()

        self._output_dataset = self._service.transpose(
            self._dataset_handle,
            feature_names_from=from_col,
            feature_name_prefix=self._prefix_edit.text() or "Feature",
            auto_column_name="column",
            remove_redundant_instance=self._remove_redundant_check.isChecked(),
        )

        self._result_label.setText(
            i18n.tf("Result: {rows} rows x {cols} columns", rows=self._output_dataset.row_count, cols=self._output_dataset.column_count)
        )
        self._notify_output_changed()

    def refresh_translations(self) -> None:
        if self._dataset_handle is None:
            self._dataset_label.setText(i18n.t("Dataset: none"))
        else:
            self._dataset_label.setText(
                i18n.tf("Dataset: {name}", name=self._dataset_handle.display_name)
            )
        if self._output_dataset is not None:
            self._result_label.setText(
                i18n.tf("Result: {rows} rows x {cols} columns", rows=self._output_dataset.row_count, cols=self._output_dataset.column_count)
            )
