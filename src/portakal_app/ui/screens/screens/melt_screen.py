from __future__ import annotations

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
)

from portakal_app.data.models import DatasetHandle
from portakal_app.data.services.melt_service import MeltService
from portakal_app.ui.screens.node_screen import WorkflowNodeScreenSupport
from portakal_app.ui import i18n


class MeltScreen(QWidget, WorkflowNodeScreenSupport):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._init_workflow_node_support()
        self._service = MeltService()
        self._dataset_handle: DatasetHandle | None = None
        self._output_dataset: DatasetHandle | None = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(10)

        # Dataset label
        self._dataset_label = QLabel(i18n.t("Dataset: none"))
        self._dataset_label.setProperty("sectionTitle", True)
        self._dataset_label.setStyleSheet("font-size: 12pt; background: transparent;")
        layout.addWidget(self._dataset_label)

        # Unique Row Identifier
        id_group = QGroupBox(i18n.t("Unique Row Identifier"))
        id_layout = QVBoxLayout(id_group)
        id_layout.setContentsMargins(10, 10, 10, 10)
        self._id_combo = QComboBox()
        self._id_combo.addItem(i18n.t("Row number"))
        id_layout.addWidget(self._id_combo)
        layout.addWidget(id_group)

        # Filter
        filter_group = QGroupBox(i18n.t("Filter"))
        filter_layout = QVBoxLayout(filter_group)
        filter_layout.setContentsMargins(10, 10, 10, 10)
        self._ignore_non_numeric = QCheckBox(i18n.t("Ignore non-numeric features"))
        self._ignore_non_numeric.setChecked(True)
        filter_layout.addWidget(self._ignore_non_numeric)

        self._exclude_zeros = QCheckBox(i18n.t("Exclude zero values"))
        self._exclude_zeros.setChecked(False)
        filter_layout.addWidget(self._exclude_zeros)
        layout.addWidget(filter_group)

        # Names for generated features
        names_group = QGroupBox(i18n.t("Names for generated features"))
        names_layout = QVBoxLayout(names_group)
        names_layout.setContentsMargins(10, 10, 10, 10)

        item_row = QHBoxLayout()
        item_row.addWidget(QLabel(i18n.t("Item:")))
        self._item_name = QLineEdit("item")
        item_row.addWidget(self._item_name)
        names_layout.addLayout(item_row)

        value_row = QHBoxLayout()
        value_row.addWidget(QLabel(i18n.t("Value:")))
        self._value_name = QLineEdit("value")
        value_row.addWidget(self._value_name)
        names_layout.addLayout(value_row)
        layout.addWidget(names_group)

        # Output label
        self._result_label = QLabel("")
        self._result_label.setWordWrap(True)
        layout.addWidget(self._result_label)

        layout.addStretch(1)

        # Auto-apply hooks
        self._ignore_non_numeric.stateChanged.connect(self._check_auto_apply)
        self._exclude_zeros.stateChanged.connect(self._check_auto_apply)
        self._item_name.textChanged.connect(self._check_auto_apply)
        self._value_name.textChanged.connect(self._check_auto_apply)
        self._id_combo.currentIndexChanged.connect(self._check_auto_apply)

        # Footer
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

    def _check_auto_apply(self):
        if self.cb_apply_auto.isChecked():
            self._apply()

    def set_input_payload(self, payload) -> None:
        import polars as pl
        dataset = payload.dataset if payload is not None else None
        self._dataset_handle = dataset
        self._output_dataset = None
        self._id_combo.blockSignals(True)
        self._id_combo.clear()
        self._id_combo.addItem(i18n.t("Row number"))

        if dataset:
            self._dataset_label.setText(i18n.tf("Dataset: {name}", name=dataset.display_name))
            df = dataset.dataframe
            for col in dataset.domain.columns:
                if col.logical_type in ("text", "categorical"):
                    try:
                        s = df.get_column(col.name).drop_nulls()
                        if s.dtype in (pl.Utf8, pl.Categorical, pl.Enum):
                            s = s.filter(s.cast(pl.Utf8) != "")
                        if len(s) > 0 and s.n_unique() == len(s):
                            self._id_combo.addItem(col.name)
                    except Exception:
                        pass
        else:
            self._dataset_label.setText(i18n.t("Dataset: none"))
            self._result_label.setText("")

        self._id_combo.blockSignals(False)
        self._apply()

    def current_output_dataset(self) -> DatasetHandle | None:
        return self._output_dataset

    def serialize_node_state(self) -> dict[str, object]:
        return {
            "id_column": self._id_combo.currentText(),
            "ignore_non_numeric": self._ignore_non_numeric.isChecked(),
            "exclude_zeros": self._exclude_zeros.isChecked(),
            "item_name": self._item_name.text(),
            "value_name": self._value_name.text(),
            "auto_apply": self.cb_apply_auto.isChecked(),
        }

    def restore_node_state(self, payload: dict[str, object]) -> None:
        id_col = str(payload.get("id_column", "Row number"))
        if self._id_combo.findText(id_col) >= 0:
            self._id_combo.setCurrentText(id_col)
            
        # Fallback for older saves
        ignore = payload.get("ignore_non_numeric")
        if ignore is None:
            # reverse boolean of exclude_numeric if it was saved that way previously
            ignore = not payload.get("exclude_numeric", False)
            
        self._ignore_non_numeric.setChecked(bool(ignore))
        self._exclude_zeros.setChecked(bool(payload.get("exclude_zeros", False)))
        self._item_name.setText(str(payload.get("item_name", "item")))
        self._value_name.setText(str(payload.get("value_name", "value")))
        self.cb_apply_auto.setChecked(bool(payload.get("auto_apply", True)))

    def help_text(self) -> str:
        return "Convert wide-format data to long-format (unpivot). Each value variable becomes a row."

    def documentation_url(self) -> str:
        return "https://orangedatamining.com/widget-catalog/transform/melt/"

    def refresh_translations(self) -> None:
        if self._dataset_handle is None:
            self._dataset_label.setText(i18n.t("Dataset: none"))
        else:
            self._dataset_label.setText(i18n.tf("Dataset: {name}", name=self._dataset_handle.display_name))
        if self._output_dataset is not None:
            self._result_label.setText(
                i18n.tf("Result: {rows} rows x {cols} columns", rows=self._output_dataset.row_count, cols=self._output_dataset.column_count)
            )

    def _apply(self) -> None:
        if self._dataset_handle is None:
            self._output_dataset = None
            self._result_label.setText("")
            self._notify_output_changed()
            return

        id_col = self._id_combo.currentText()
        if id_col == "Row number":
            id_col = None

        try:
            result = self._service.melt(
                self._dataset_handle,
                id_column=id_col,
                ignore_non_numeric=self._ignore_non_numeric.isChecked(),
                exclude_zeros=self._exclude_zeros.isChecked(),
                item_name=self._item_name.text() or "item",
                value_name=self._value_name.text() or "value",
            )
            if result is None:
                self._output_dataset = None
                self._result_label.setText(i18n.t("No features to melt."))
            else:
                self._output_dataset = result
                self._result_label.setText(
                    i18n.tf("Result: {rows} rows x {cols} columns", rows=self._output_dataset.row_count, cols=self._output_dataset.column_count)
                )
        except Exception as e:
            self._output_dataset = None
            self._result_label.setText(i18n.tf("Error applying Melt: {error}", error=e))

        self._notify_output_changed()
