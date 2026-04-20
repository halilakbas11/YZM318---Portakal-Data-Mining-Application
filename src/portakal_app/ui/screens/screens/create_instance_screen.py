from __future__ import annotations

from typing import Any, Union
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
    QTableWidget,
    QTableWidgetItem,
    QHeaderView,
    QSizePolicy,
)
from PySide6.QtCore import Qt

from portakal_app.data.models import DatasetHandle
from portakal_app.data.services.create_instance_service import CreateInstanceService
from portakal_app.ui.screens.node_screen import WorkflowNodeScreenSupport
from portakal_app.ui import i18n


# ---------------------------------------------------------------------------
# Per-type editors  (inspired by Orange3 owcreateinstance.py)
# ---------------------------------------------------------------------------

class _CategoricalEditor(QWidget):
    """QComboBox listing unique values + '?' for missing."""

    def __init__(self, unique_values: list[str], parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(4, 0, 4, 0)
        self._combo = QComboBox()
        self._combo.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self._combo.addItems(list(unique_values) + ["?"])
        layout.addWidget(self._combo)

    def value(self) -> str | None:
        text = self._combo.currentText()
        return None if text == "?" else text

    def set_value(self, val: str | None) -> None:
        if val is None:
            self._combo.setCurrentIndex(self._combo.count() - 1)
            return
        idx = self._combo.findText(str(val))
        if idx >= 0:
            self._combo.setCurrentIndex(idx)
        else:
            self._combo.setCurrentIndex(self._combo.count() - 1)


class _NumericEditor(QWidget):
    """QDoubleSpinBox + QSlider between min..max, like Orange3."""

    def __init__(
        self,
        min_val: float,
        max_val: float,
        decimals: int = 2,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(4, 0, 4, 0)

        self._min = min_val
        self._max = max_val
        self._decimals = decimals
        self._slider_factor = 10 ** decimals

        self._spin = QDoubleSpinBox()
        self._spin.setDecimals(decimals)
        self._spin.setRange(-1e15, 1e15)
        self._spin.setSingleStep(10 ** (-decimals))
        self._spin.setMinimumWidth(70)
        sp_spin = QSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        sp_spin.setHorizontalStretch(1)
        self._spin.setSizePolicy(sp_spin)

        self._label_min = QLabel(f"{min_val:.{decimals}f}")
        self._label_min.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self._label_min.setMinimumWidth(40)

        self._slider = QSlider(Qt.Orientation.Horizontal)
        self._slider.setRange(
            int(min_val * self._slider_factor),
            int(max_val * self._slider_factor),
        )
        sp_slider = QSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        sp_slider.setHorizontalStretch(5)
        self._slider.setSizePolicy(sp_slider)

        self._label_max = QLabel(f"{max_val:.{decimals}f}")
        self._label_max.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        self._label_max.setMinimumWidth(40)

        layout.addWidget(self._spin)
        layout.addWidget(self._label_min)
        layout.addWidget(self._slider)
        layout.addWidget(self._label_max)

        self._updating = False
        self._slider.valueChanged.connect(self._slider_changed)
        self._spin.valueChanged.connect(self._spin_changed)

    # ---- public API ----
    def value(self) -> float | None:
        return round(self._spin.value(), self._decimals)

    def set_value(self, val: float | None) -> None:
        if val is None:
            val = self._min
        self._updating = True
        self._spin.setValue(round(val, self._decimals))
        self._slider.setValue(int(val * self._slider_factor))
        self._updating = False

    # ---- internal sync ----
    def _slider_changed(self, v: int) -> None:
        if self._updating:
            return
        self._updating = True
        self._spin.setValue(v / self._slider_factor)
        self._updating = False

    def _spin_changed(self, v: float) -> None:
        if self._updating:
            return
        self._updating = True
        self._slider.setValue(int(v * self._slider_factor))
        self._updating = False


class _TextEditor(QWidget):
    """Plain QLineEdit for string / unknown types."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(4, 0, 4, 0)
        self._edit = QLineEdit()
        self._edit.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        layout.addWidget(self._edit)

    def value(self) -> str | None:
        t = self._edit.text().strip()
        return t if t else None

    def set_value(self, val: str | None) -> None:
        self._edit.setText(str(val) if val is not None else "")


# ---------------------------------------------------------------------------
# Screen
# ---------------------------------------------------------------------------

class CreateInstanceScreen(QWidget, WorkflowNodeScreenSupport):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._init_workflow_node_support()
        self._service = CreateInstanceService()
        self._data_ds: DatasetHandle | None = None
        self._ref_ds: DatasetHandle | None = None
        self._output_dataset: DatasetHandle | None = None
        # col_name → editor widget
        self._editors: dict[str, _CategoricalEditor | _NumericEditor | _TextEditor] = {}

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        self._filter_edit = QLineEdit()
        self._filter_edit.setPlaceholderText(i18n.t("Filter..."))
        self._filter_edit.textChanged.connect(self._on_filter_changed)
        layout.addWidget(self._filter_edit)

        self._table = QTableWidget()
        self._table.setColumnCount(2)
        self._table.setHorizontalHeaderLabels([i18n.t("Variable"), i18n.t("Value")])
        header = self._table.horizontalHeader()
        if header:
            header.setSectionResizeMode(0, QHeaderView.ResizeMode.Interactive)
            header.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self._table.verticalHeader().setVisible(False)
        self._table.setAlternatingRowColors(True)
        # Explicit cross-platform styling (Linux/GTK ignores palette-based
        # alternating row colours; setting them via stylesheet fixes this).
        self._table.setStyleSheet("""
            QTableWidget {
                alternate-background-color: rgba(0, 0, 0, 0.04);
                background-color: palette(base);
                gridline-color: rgba(0, 0, 0, 0.08);
            }
            QTableWidget::item {
                padding: 4px;
            }
            QHeaderView::section {
                background-color: palette(button);
                border: 1px solid palette(mid);
                padding: 4px;
            }
        """)
        layout.addWidget(self._table, 1)

        btn_row = QHBoxLayout()
        btn_row.addStretch(1)

        self._btn_median = QPushButton(i18n.t("Median"))
        self._btn_median.clicked.connect(lambda: self._fill_defaults("median"))
        btn_row.addWidget(self._btn_median)

        self._btn_mean = QPushButton(i18n.t("Mean"))
        self._btn_mean.clicked.connect(lambda: self._fill_defaults("mean"))
        btn_row.addWidget(self._btn_mean)

        self._btn_random = QPushButton(i18n.t("Random"))
        self._btn_random.clicked.connect(lambda: self._fill_defaults("random"))
        btn_row.addWidget(self._btn_random)

        self._btn_input = QPushButton(i18n.t("Input"))
        self._btn_input.clicked.connect(self._fill_from_input)
        btn_row.addWidget(self._btn_input)

        btn_row.addStretch(1)
        layout.addLayout(btn_row)

        bottom_row = QHBoxLayout()
        self._append_cb = QCheckBox(i18n.t("Append this instance to input data"))
        self._append_cb.setChecked(True)
        self._append_cb.stateChanged.connect(self._check_auto_apply)
        bottom_row.addWidget(self._append_cb)

        bottom_row.addStretch(1)
        self.cb_apply_auto = QCheckBox(i18n.t("Apply Automatically"))
        self.cb_apply_auto.setChecked(True)
        bottom_row.addWidget(self.cb_apply_auto)

        self._apply_button = QPushButton(i18n.t("Create"))
        self._apply_button.setProperty("primary", True)
        self._apply_button.clicked.connect(self._apply)
        bottom_row.addWidget(self._apply_button)
        layout.addLayout(bottom_row)

        self._result_label = QLabel("")
        self._result_label.setWordWrap(True)
        layout.addWidget(self._result_label)

    def _check_auto_apply(self) -> None:
        if self.cb_apply_auto.isChecked():
            self._apply()

    # ------------------------------------------------------------------ I/O
    def set_input_payload(self, payload) -> None:
        if payload is None:
            self._data_ds = None
            self._ref_ds = None
        else:
            if getattr(payload, "port_label", None) == "Data":
                self._data_ds = getattr(payload, "dataset", None)
            elif getattr(payload, "port_label", None) == "Reference":
                self._ref_ds = getattr(payload, "dataset", None)
        self._rebuild_fields()
        self._apply()

    def current_output_dataset(self) -> DatasetHandle | None:
        return self._output_dataset

    def serialize_node_state(self) -> dict[str, object]:
        values = {}
        for name, editor in self._editors.items():
            v = editor.value()
            values[name] = v if v is not None else ""
        return {
            "values": values,
            "append": self._append_cb.isChecked(),
            "auto_apply": self.cb_apply_auto.isChecked(),
        }

    def restore_node_state(self, payload: dict[str, object]) -> None:
        values = payload.get("values", {})
        if isinstance(values, dict):
            for name, val in values.items():
                if name in self._editors:
                    self._editors[name].set_value(val if val != "" else None)
        self._append_cb.setChecked(bool(payload.get("append", True)))
        self.cb_apply_auto.setChecked(bool(payload.get("auto_apply", True)))

    def help_text(self) -> str:
        return "Create a single instance (row) by setting column values manually or automatically based on logic."

    def documentation_url(self) -> str:
        return "https://orangedatamining.com/widget-catalog/transform/createinstance/"

    # ------------------------------------------------------ build the table
    def _get_schema_ds(self) -> DatasetHandle | None:
        return self._ref_ds if self._ref_ds is not None else self._data_ds

    def _rebuild_fields(self) -> None:
        # cache old values
        old_values: dict[str, Any] = {}
        for name, editor in self._editors.items():
            old_values[name] = editor.value()
        self._editors.clear()
        self._table.setRowCount(0)

        schema_ds = self._get_schema_ds()
        if schema_ds is None:
            return

        cols = schema_ds.domain.columns
        df = schema_ds.dataframe
        self._table.setRowCount(len(cols))

        for idx, col in enumerate(cols):
            type_label = col.logical_type
            if type_label == "numeric":
                type_label = "Num"
            elif type_label == "categorical":
                type_label = "Cat"
            elif type_label == "text":
                type_label = "Txt"
            elif type_label == "datetime":
                type_label = "Time"

            var_item = QTableWidgetItem(f"{col.name} ({type_label})")
            var_item.setData(99, col.name)
            var_item.setFlags(var_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self._table.setItem(idx, 0, var_item)

            # --- create the right editor for this column type ---
            series = df.get_column(col.name)
            editor: _CategoricalEditor | _NumericEditor | _TextEditor

            if col.logical_type == "categorical":
                unique_vals = sorted(
                    str(v) for v in series.drop_nulls().unique().to_list()
                )
                editor = _CategoricalEditor(unique_vals)
                # default: mode (most frequent)
                if unique_vals:
                    mode = series.drop_nulls().mode()
                    default = str(mode[0]) if mode.len() > 0 else unique_vals[0]
                    editor.set_value(default)

            elif col.logical_type == "numeric":
                non_null = series.drop_nulls()
                if non_null.len() > 0:
                    mn = float(non_null.min())  # type: ignore[arg-type]
                    mx = float(non_null.max())  # type: ignore[arg-type]
                    if mn == mx:
                        mx = mn + 1.0
                    med = float(non_null.median())  # type: ignore[arg-type]
                else:
                    mn, mx, med = 0.0, 1.0, 0.0
                editor = _NumericEditor(mn, mx, decimals=2)
                editor.set_value(med)

            else:
                editor = _TextEditor()

            # restore old value if the column existed before
            if col.name in old_values and old_values[col.name] is not None:
                editor.set_value(old_values[col.name])

            self._editors[col.name] = editor
            self._table.setCellWidget(idx, 1, editor)
            self._table.setRowHeight(idx, 40)

        self._on_filter_changed()

    # ----------------------------------------------------------- filtering
    def _on_filter_changed(self) -> None:
        text = self._filter_edit.text().strip().lower()
        for idx in range(self._table.rowCount()):
            item = self._table.item(idx, 0)
            if not item:
                continue
            name = item.data(99) or ""
            if not text or text in name.lower() or text in item.text().lower():
                self._table.setRowHidden(idx, False)
            else:
                self._table.setRowHidden(idx, True)

    # -------------------------------------------------- fill-value buttons
    def _fill_defaults(self, stat_type: str) -> None:
        schema_ds = self._get_schema_ds()
        if schema_ds is None:
            return
        try:
            defaults = self._service.get_defaults(schema_ds, stat_type)
            for name, val in defaults.items():
                if name in self._editors:
                    editor = self._editors[name]
                    if isinstance(editor, _NumericEditor):
                        try:
                            editor.set_value(float(val) if val else None)
                        except (ValueError, TypeError):
                            editor.set_value(None)
                    else:
                        editor.set_value(val if val else None)
            self._check_auto_apply()
        except Exception:
            pass

    def _fill_from_input(self) -> None:
        schema_ds = self._get_schema_ds()
        if schema_ds is None or schema_ds.dataframe.height == 0:
            return
        first_row = schema_ds.dataframe.row(0, named=True)
        for name, editor in self._editors.items():
            if name in first_row:
                val = first_row[name]
                if isinstance(editor, _NumericEditor):
                    try:
                        editor.set_value(float(val) if val is not None else None)
                    except (ValueError, TypeError):
                        editor.set_value(None)
                else:
                    editor.set_value(str(val) if val is not None else None)
        self._check_auto_apply()

    # --------------------------------------------------------- translations
    def refresh_translations(self) -> None:
        if self._output_dataset is not None:
            self._result_label.setText(
                i18n.tf("Created instance. Output: {rows} rows", rows=self._output_dataset.row_count)
            )

    # -------------------------------------------------------------- apply
    def _apply(self) -> None:
        schema_ds = self._get_schema_ds()
        if schema_ds is None:
            self._output_dataset = None
            self._result_label.setText("")
            self._notify_output_changed()
            return

        values: dict[str, object] = {}
        for name, editor in self._editors.items():
            v = editor.value()
            values[name] = v

        try:
            self._output_dataset = self._service.create(
                reference=self._ref_ds,
                data=self._data_ds,
                values=values,
                append_to_data=self._append_cb.isChecked(),
            )
            self._result_label.setText(
                i18n.tf("Created instance. Output: {rows} rows", rows=self._output_dataset.row_count)
            )
        except Exception as e:
            self._output_dataset = None
            self._result_label.setText(i18n.tf("Error creating instance: {error}", error=e))

        self._notify_output_changed()
