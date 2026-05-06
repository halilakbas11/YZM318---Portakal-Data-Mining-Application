from __future__ import annotations

import numpy as np
from PySide6.QtCore import QRect, QSize, Qt
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QHeaderView,
    QLabel,
    QLineEdit,
    QStyledItemDelegate,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from portakal_app.data.models import DatasetHandle
from portakal_app.ui import i18n
from portakal_app.ui.screens.node_screen import WorkflowNodeScreenSupport

try:
    from scipy.stats import pearsonr, spearmanr
except ImportError:  # pragma: no cover - dependency check at runtime
    pearsonr = None
    spearmanr = None


class CorrelationBarDelegate(QStyledItemDelegate):
    def initStyleOption(self, option, index):
        super().initStyleOption(option, index)
        option.text = ""

    def sizeHint(self, option, index):
        size = super().sizeHint(option, index)
        return QSize(max(85, size.width()), size.height())

    def paint(self, painter, option, index):
        super().paint(painter, option, index)
        try:
            value = float(index.data(Qt.ItemDataRole.DisplayRole))
        except (TypeError, ValueError):
            value = 0.0

        color = QColor(136, 190, 20) if value >= 0 else QColor(50, 150, 255)
        rect = option.rect
        painter.save()
        painter.setRenderHint(painter.RenderHint.Antialiasing)
        painter.setPen(color)
        font = option.font
        font.setBold(True)
        painter.setFont(font)
        painter.drawText(QRect(rect.left() + 5, rect.top(), rect.width() - 10, rect.height() - 8), Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, index.data(Qt.ItemDataRole.DisplayRole))
        bar_width = int((rect.width() - 15) * abs(value))
        painter.setBrush(color)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawRoundedRect(QRect(rect.left() + 5, rect.bottom() - 8, bar_width, 4), 2, 2)
        painter.restore()


class CorrelationsScreen(QWidget, WorkflowNodeScreenSupport):
    METHODS = ["Pearson correlation", "Spearman correlation"]

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._init_workflow_node_support()
        self._dataset_handle: DatasetHandle | None = None
        self._output_dataset: DatasetHandle | None = None
        self._all_pairs: list[tuple[float, str, str, float]] = []
        self._build_ui()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(10)

        self._method_combo = QComboBox()
        self._method_combo.addItems(self.METHODS)
        self._method_combo.currentIndexChanged.connect(self._apply)
        root.addWidget(self._method_combo)

        self._feature_combo = QComboBox()
        self._feature_combo.addItem("(All combinations)")
        self._feature_combo.currentIndexChanged.connect(self._apply_filter)
        root.addWidget(self._feature_combo)

        self._impute_check = QCheckBox("Impute missing values")
        self._impute_check.setChecked(True)
        self._impute_check.toggled.connect(self._apply)
        root.addWidget(self._impute_check)

        self._filter_edit = QLineEdit()
        self._filter_edit.setPlaceholderText("Filter ...")
        self._filter_edit.textChanged.connect(self._apply_filter)
        root.addWidget(self._filter_edit)

        self._table = QTableWidget()
        self._table.setColumnCount(5)
        self._table.horizontalHeader().setVisible(False)
        self._table.verticalHeader().setVisible(False)
        self._table.setShowGrid(False)
        self._table.setAlternatingRowColors(True)
        self._table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self._table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self._table.setItemDelegateForColumn(1, CorrelationBarDelegate(self._table))
        self._table.verticalHeader().setDefaultSectionSize(35)
        self._table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        self._table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        self._table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        self._table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
        self._table.horizontalHeader().setSectionResizeMode(4, QHeaderView.ResizeMode.Stretch)
        root.addWidget(self._table, 1)

        self._info_label = QLabel(i18n.t("Finished"))
        self._info_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        root.addWidget(self._info_label)

    def set_input_payload(self, payload) -> None:
        self._dataset_handle = payload.dataset if payload is not None else None
        self._output_dataset = self._dataset_handle
        self._apply()

    def current_output_dataset(self) -> DatasetHandle | None:
        return self._output_dataset

    def help_text(self) -> str:
        return "Inspect pairwise Pearson or Spearman correlations between numeric features."

    def documentation_url(self) -> str:
        return "https://orangedatamining.com/widget-catalog/data/correlations/"

    def _apply(self) -> None:
        self._all_pairs = []
        self._table.setRowCount(0)
        self._output_dataset = self._dataset_handle

        if self._dataset_handle is None:
            self._info_label.setText(i18n.t("No dataset loaded."))
            self._notify_output_changed()
            return
        if pearsonr is None or spearmanr is None:
            self._info_label.setText("SciPy is required for Correlations.")
            self._notify_output_changed()
            return

        try:
            feature_cols = [
                column.name
                for column in self._dataset_handle.domain.feature_columns
                if column.logical_type == "numeric" and column.name.lower() != "id"
            ]
            if len(feature_cols) < 2:
                self._info_label.setText("At least 2 numeric features required.")
                self._notify_output_changed()
                return

            matrix = self._dataset_handle.dataframe.select(feature_cols).to_numpy(allow_copy=True).astype(float)
            if self._impute_check.isChecked():
                nan_mask = np.isnan(matrix)
                if nan_mask.any():
                    col_means = np.nanmean(matrix, axis=0)
                    col_means = np.where(np.isnan(col_means), 0.0, col_means)
                    matrix[nan_mask] = np.take(col_means, np.where(nan_mask)[1])
            else:
                matrix = np.nan_to_num(matrix, nan=0.0)

            corr = np.zeros((matrix.shape[1], matrix.shape[1]))
            fn = spearmanr if self._method_combo.currentIndex() == 1 else pearsonr
            for i in range(matrix.shape[1]):
                for j in range(matrix.shape[1]):
                    corr[i, j] = 1.0 if i == j else self._corr(fn, matrix[:, i], matrix[:, j])

            self._all_pairs = sorted(
                [(abs(corr[i, j]), feature_cols[i], feature_cols[j], corr[i, j]) for i in range(matrix.shape[1]) for j in range(i + 1, matrix.shape[1])],
                reverse=True,
            )

            previous = self._feature_combo.currentText()
            self._feature_combo.blockSignals(True)
            self._feature_combo.clear()
            self._feature_combo.addItem("(All combinations)")
            for name in feature_cols:
                self._feature_combo.addItem(name)
            idx = self._feature_combo.findText(previous)
            self._feature_combo.setCurrentIndex(max(0, idx))
            self._feature_combo.blockSignals(False)

            self._apply_filter()
            self._info_label.setText(i18n.t("Finished"))
        except Exception as exc:
            self._info_label.setText(i18n.tf("Error: {error}", error=exc))

        self._notify_output_changed()

    @staticmethod
    def _corr(fn, left: np.ndarray, right: np.ndarray) -> float:
        try:
            r, _p = fn(left, right)
            return 0.0 if np.isnan(r) else float(r)
        except Exception:
            return 0.0

    def _apply_filter(self) -> None:
        feature = self._feature_combo.currentText()
        query = self._filter_edit.text().strip().lower()
        pairs = self._all_pairs
        if feature != "(All combinations)":
            pairs = [pair for pair in pairs if pair[1] == feature or pair[2] == feature]
        if query:
            pairs = [pair for pair in pairs if query in pair[1].lower() or query in pair[2].lower()]
        self._fill_pairs(pairs)

    def _fill_pairs(self, pairs: list[tuple[float, str, str, float]]) -> None:
        self._table.setRowCount(len(pairs))
        for row, (_abs_val, left, right, corr) in enumerate(pairs):
            idx_item = QTableWidgetItem(str(row + 1))
            idx_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self._table.setItem(row, 0, idx_item)
            self._table.setItem(row, 1, QTableWidgetItem(f"{corr:+.3f}"))

            left_item = QTableWidgetItem(left)
            left_item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            self._table.setItem(row, 2, left_item)

            sep_item = QTableWidgetItem(":")
            sep_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self._table.setItem(row, 3, sep_item)

            right_item = QTableWidgetItem(right)
            right_item.setTextAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
            self._table.setItem(row, 4, right_item)
