from __future__ import annotations

from math import isnan, isinf

import numpy as np
from sklearn import metrics as skl_metrics

from PySide6.QtCore import Qt, QItemSelection, QItemSelectionModel
from PySide6.QtGui import QBrush, QColor, QFont, QStandardItem, QStandardItemModel
from PySide6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QSizePolicy,
    QStyledItemDelegate,
    QTableView,
    QVBoxLayout,
    QWidget,
)

from portakal_app.data.models import DatasetHandle
from portakal_app.data.services.sklearn_learner_service import SklearnLearnerService
from portakal_app.models import WorkflowPayload
from portakal_app.rule_artifacts import CN2RuleClassifierArtifact
from portakal_app.scoring_sheet_artifacts import ScoringSheetClassifierArtifact
from portakal_app.sklearn_model_artifacts import SklearnModelArtifact
from portakal_app.tree_artifacts import DecisionTreeArtifact, RandomForestArtifact
from portakal_app.ui.screens.node_screen import WorkflowNodeScreenSupport

_SKLEARN_TYPES = (SklearnModelArtifact, DecisionTreeArtifact, RandomForestArtifact)
_RULE_TYPES = (ScoringSheetClassifierArtifact, CN2RuleClassifierArtifact)
_ALL_ARTIFACT_TYPES = _SKLEARN_TYPES + _RULE_TYPES


def _is_supported_artifact(value: object) -> bool:
    return isinstance(value, _ALL_ARTIFACT_TYPES)


_BORDER_ROLE = Qt.ItemDataRole.UserRole + 100
_BORDER_COLOR_ROLE = Qt.ItemDataRole.UserRole + 101


class _BorderedDelegate(QStyledItemDelegate):
    """Draws thin border lines on specified sides of a table cell."""

    def paint(self, painter, option, index):
        super().paint(painter, option, index)
        sides = index.data(_BORDER_ROLE)
        if not sides:
            return
        color = index.data(_BORDER_COLOR_ROLE) or Qt.GlobalColor.white
        painter.save()
        painter.setPen(color)
        rect = option.rect
        for side, p1, p2 in (
            ("t", rect.topLeft(), rect.topRight()),
            ("r", rect.topRight(), rect.bottomRight()),
            ("b", rect.bottomLeft(), rect.bottomRight()),
            ("l", rect.topLeft(), rect.bottomLeft()),
        ):
            if side in sides:
                painter.drawLine(p1, p2)
        painter.restore()


class ConfusionMatrixScreen(QWidget, WorkflowNodeScreenSupport):
    """Confusion Matrix — compare classifier predictions against true labels."""

    _QUANTITIES = [
        "Number of instances",
        "Proportion of predicted",
        "Proportion of actual",
    ]

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._init_workflow_node_support()

        self._dataset: DatasetHandle | None = None
        self._model: object | None = None
        self._svc = SklearnLearnerService()
        self._cmatrix: np.ndarray | None = None
        self._class_values: tuple[str, ...] = ()

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        self._status_label = QLabel("No model or data on input.")
        self._status_label.setWordWrap(True)
        self._status_label.setStyleSheet("color: #777; background: transparent;")
        layout.addWidget(self._status_label)

        combo_row = QHBoxLayout()
        combo_row.addStretch(1)
        combo_row.addWidget(QLabel("Show:"))
        self._qty_combo = QComboBox()
        for q in self._QUANTITIES:
            self._qty_combo.addItem(q)
        # Only update values (not table structure) on combo change
        self._qty_combo.currentIndexChanged.connect(self._fill_values)
        combo_row.addWidget(self._qty_combo)
        layout.addLayout(combo_row)

        self._table_model = QStandardItemModel(self)
        self._table = QTableView()
        self._table.setModel(self._table_model)
        self._table.horizontalHeader().hide()
        self._table.verticalHeader().hide()
        self._table.setShowGrid(False)
        self._table.setEditTriggers(QTableView.EditTrigger.NoEditTriggers)
        self._table.setSelectionMode(QTableView.SelectionMode.MultiSelection)
        self._table.setItemDelegate(_BorderedDelegate())
        self._table.setSizePolicy(
            QSizePolicy.Policy.MinimumExpanding,
            QSizePolicy.Policy.MinimumExpanding,
        )
        layout.addWidget(self._table)

        btn_row = QHBoxLayout()
        self._btn_correct = QPushButton("Select Correct")
        self._btn_correct.clicked.connect(self._select_correct)
        btn_row.addWidget(self._btn_correct)
        self._btn_wrong = QPushButton("Select Misclassified")
        self._btn_wrong.clicked.connect(self._select_wrong)
        btn_row.addWidget(self._btn_wrong)
        self._btn_clear = QPushButton("Clear Selection")
        self._btn_clear.clicked.connect(self._table.clearSelection)
        btn_row.addWidget(self._btn_clear)
        layout.addLayout(btn_row)

        self._acc_label = QLabel("")
        self._acc_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._acc_label.setStyleSheet("font-weight: bold; background: transparent;")
        layout.addWidget(self._acc_label)

    # ── WorkflowNodeScreenSupport ─────────────────────────────────────

    def set_input_payload(self, payload: WorkflowPayload | None) -> None:
        if payload is None:
            self._dataset = None
            self._model = None
        elif payload.port_label == "Data" and isinstance(payload.value, DatasetHandle):
            self._dataset = payload.value
        elif _is_supported_artifact(payload.value):
            self._model = payload.value
        self._compute()

    def current_output_payload(self) -> WorkflowPayload | None:
        return None

    def serialize_node_state(self) -> dict[str, object]:
        return {"quantity": self._qty_combo.currentIndex()}

    def restore_node_state(self, payload: dict[str, object]) -> None:
        self._qty_combo.setCurrentIndex(int(payload.get("quantity", 0)))

    # ── Computation ────────────────────────────────────────────────────

    def _compute(self) -> None:
        self._table_model.clear()
        self._cmatrix = None
        self._class_values = ()
        self._acc_label.setText("")

        if self._model is None or self._dataset is None:
            self._status_label.setText("No model or data on input.")
            self._status_label.setStyleSheet("color: #777; background: transparent;")
            return

        artifact = self._model

        is_clf = getattr(artifact, "is_classifier", None)
        if is_clf is None:
            is_clf = getattr(artifact, "kind", "") == "classification"
        if not is_clf:
            self._status_label.setText(
                "Confusion Matrix only supports classifiers, not regressors."
            )
            self._status_label.setStyleSheet("color: #c62828; background: transparent;")
            return

        target_name: str | None = getattr(artifact, "target_name", None)
        class_values: tuple[str, ...] = getattr(artifact, "class_values", ()) or ()

        if not class_values:
            self._status_label.setText("Model has no class values (regressor or untrained?).")
            self._status_label.setStyleSheet("color: #c62828; background: transparent;")
            return

        if not target_name or target_name not in self._dataset.dataframe.columns:
            self._status_label.setText(
                f"Target column '{target_name}' not found in dataset."
            )
            self._status_label.setStyleSheet("color: #c62828; background: transparent;")
            return

        try:
            if isinstance(artifact, _RULE_TYPES):
                if not artifact.can_apply_to(self._dataset):
                    self._status_label.setText("Model features do not match the dataset.")
                    self._status_label.setStyleSheet("color: #c62828; background: transparent;")
                    return
                y_pred = artifact.predict_from_dataset(self._dataset)
                class_to_idx = {v: i for i, v in enumerate(class_values)}
                target_series = self._dataset.dataframe.get_column(target_name)
                y_actual = np.asarray(
                    [class_to_idx.get(str(v), 0) for v in target_series.to_list()],
                    dtype=int,
                )
            else:
                feature_names: tuple[str, ...] = getattr(artifact, "feature_names", ()) or ()
                cat_encoders: dict = getattr(artifact, "categorical_encoders", {}) or {}
                num_cols: tuple[str, ...] = getattr(artifact, "numeric_cols", ()) or ()

                if not feature_names:
                    self._status_label.setText(
                        "This model has no stored feature encoding — retrain to enable predictions."
                    )
                    self._status_label.setStyleSheet("color: #c62828; background: transparent;")
                    return

                col_names = {col.name for col in self._dataset.domain.feature_columns}
                missing = [name for name in feature_names if name not in col_names]
                if missing:
                    self._status_label.setText(
                        f"Dataset is missing feature columns: {', '.join(missing)}"
                    )
                    self._status_label.setStyleSheet("color: #c62828; background: transparent;")
                    return

                numeric_means: dict[str, float] = {}
                for col_name in num_cols:
                    if col_name in self._dataset.dataframe.columns:
                        series = self._dataset.dataframe.get_column(col_name)
                        raw = []
                        for v in series.to_list():
                            try:
                                raw.append(float(v))
                            except (TypeError, ValueError):
                                raw.append(float("nan"))
                        arr = np.asarray(raw, dtype=float)
                        finite = arr[np.isfinite(arr)]
                        numeric_means[col_name] = float(np.mean(finite)) if finite.size else 0.0

                X = self._svc.encode_X(
                    self._dataset, feature_names, cat_encoders, num_cols, numeric_means
                )
                target_encoder: dict = getattr(artifact, "target_encoder", None) or {}
                target_series = self._dataset.dataframe.get_column(target_name)
                y_actual = np.asarray(
                    [target_encoder.get(str(v), 0) for v in target_series.to_list()],
                    dtype=int,
                )
                y_pred = artifact.predict(X)  # type: ignore[union-attr]

        except Exception as exc:
            self._status_label.setText(f"Error: {exc}")
            self._status_label.setStyleSheet("color: #c62828; background: transparent;")
            return

        labels = list(range(len(class_values)))
        self._cmatrix = skl_metrics.confusion_matrix(y_actual, y_pred, labels=labels)
        self._class_values = class_values

        total = int(self._cmatrix.sum())
        correct = int(np.trace(self._cmatrix))
        accuracy = correct / total if total > 0 else 0.0
        self._acc_label.setText(
            f"Accuracy: {accuracy:.2%}   ({correct} / {total} correct)"
        )
        display_name = getattr(artifact, "display_name", "Model")
        self._status_label.setText(
            f"Model: {display_name}   |   Dataset: {self._dataset.display_name}"
        )
        self._status_label.setStyleSheet("color: #2e7d32; background: transparent;")
        self._init_table()

    # ── Table rendering ───────────────────────────────────────────────

    def _init_table(self) -> None:
        """Build the table structure (headers, spans, row/col count). Called once per new matrix."""
        if self._cmatrix is None or not self._class_values:
            return

        n = len(self._class_values)
        nrows = n + 3
        ncols = n + 3

        self._table_model.setRowCount(nrows)
        self._table_model.setColumnCount(ncols)

        bold = QFont()
        bold.setBold(True)
        gray = QColor(192, 192, 192)
        enabled = Qt.ItemFlag.ItemIsEnabled

        def _set_header(row, col, text, flags=None, align=Qt.AlignmentFlag.AlignCenter,
                        font=None, border="", border_color=None):
            it = QStandardItem()
            it.setData(str(text), Qt.ItemDataRole.DisplayRole)
            it.setTextAlignment(align)
            it.setFlags(flags if flags is not None else Qt.ItemFlag.NoItemFlags)
            if font:
                it.setFont(font)
            if border:
                it.setData(border, _BORDER_ROLE)
                it.setData(border_color or gray, _BORDER_COLOR_ROLE)
            self._table_model.setItem(row, col, it)

        # "Predicted" spanning header
        _set_header(0, 2, "Predicted", font=bold)
        self._table.setSpan(0, 2, 1, n)

        # "Actual" spanning header
        _set_header(2, 0, "Actual", font=bold)
        self._table.setSpan(2, 0, n, 1)

        hdr_align = Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter

        for p, label in enumerate(self._class_values):
            _set_header(1, p + 2, label, flags=enabled, align=hdr_align, font=bold,
                        border="b" if p < n - 1 else "", border_color=gray)
            _set_header(p + 2, 1, label, flags=enabled, align=hdr_align, font=bold,
                        border="r" if p < n - 1 else "", border_color=gray)

        _set_header(1, n + 2, "Σ", flags=enabled, font=bold)
        _set_header(n + 2, 1, "Σ", flags=enabled, font=bold)

        # Column for row-sum labels and row for col-sum labels are structural
        gray_border = QColor(192, 192, 192)
        sum_align = Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
        rowsum = self._cmatrix.sum(axis=1).astype(float)
        colsum = self._cmatrix.sum(axis=0).astype(float)

        for i in range(n):
            it = QStandardItem()
            it.setData(str(int(round(rowsum[i]))), Qt.ItemDataRole.DisplayRole)
            it.setTextAlignment(sum_align)
            it.setFlags(enabled)
            it.setFont(bold)
            it.setData("l", _BORDER_ROLE)
            it.setData(gray_border, _BORDER_COLOR_ROLE)
            self._table_model.setItem(i + 2, n + 2, it)

        for j in range(n):
            it = QStandardItem()
            it.setData(str(int(round(colsum[j]))), Qt.ItemDataRole.DisplayRole)
            it.setTextAlignment(sum_align)
            it.setFlags(enabled)
            it.setFont(bold)
            it.setData("t", _BORDER_ROLE)
            it.setData(gray_border, _BORDER_COLOR_ROLE)
            self._table_model.setItem(n + 2, j + 2, it)

        total_item = QStandardItem()
        total_item.setData(str(int(round(rowsum.sum()))), Qt.ItemDataRole.DisplayRole)
        total_item.setTextAlignment(sum_align)
        total_item.setFlags(enabled)
        total_item.setFont(bold)
        self._table_model.setItem(n + 2, n + 2, total_item)

        hdr = self._table.horizontalHeader()
        if sum(len(lbl) for lbl in self._class_values) < 120:
            hdr.setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        else:
            hdr.setDefaultSectionSize(60)

        self._fill_values()

    def _fill_values(self) -> None:
        """Update only the data-cell values based on the current Show combo selection."""
        if self._cmatrix is None or not self._class_values:
            return

        cmatrix = self._cmatrix
        n = len(self._class_values)
        qty = self._qty_combo.currentIndex()

        colsum = cmatrix.sum(axis=0).astype(float)
        rowsum = cmatrix.sum(axis=1).astype(float)

        with np.errstate(invalid="ignore", divide="ignore"):
            if qty == 0:
                normalized = cmatrix.astype(float)
                fmt = "{:.0f}"
            elif qty == 1:
                normalized = 100.0 * cmatrix / colsum
                fmt = "{:.1f} %"
            else:
                normalized = 100.0 * cmatrix / rowsum[:, np.newaxis]
                fmt = "{:.1f} %"

        raw = cmatrix.astype(float)
        max_diag = np.diag(raw).max() or 1.0
        off_diag = raw.copy()
        np.fill_diagonal(off_diag, 0.0)
        max_off = off_diag.max() or 1.0

        def _bad(v):
            try:
                return isnan(float(v)) or isinf(float(v))
            except (TypeError, ValueError):
                return True

        cell_flags = Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable
        cell_align = Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter

        for i in range(n):
            for j in range(n):
                val = normalized[i, j]
                raw_val = raw[i, j]
                text = "NA" if _bad(val) else fmt.format(val)
                if i == j:
                    intensity = raw_val / max_diag
                    bg = QColor.fromHsl(120, 160, int(255 - 60 * intensity))  # green
                else:
                    intensity = raw_val / max_off
                    bg = QColor.fromHsl(0, 160, int(255 - 60 * intensity))    # red

                it = self._table_model.item(i + 2, j + 2)
                if it is None:
                    it = QStandardItem()
                    it.setFlags(cell_flags)
                    it.setTextAlignment(cell_align)
                    it.setData("trbl", _BORDER_ROLE)
                    it.setData(Qt.GlobalColor.white, _BORDER_COLOR_ROLE)
                    self._table_model.setItem(i + 2, j + 2, it)

                it.setData(text, Qt.ItemDataRole.DisplayRole)
                it.setData(QBrush(bg), Qt.ItemDataRole.BackgroundRole)
                it.setData(QBrush(Qt.GlobalColor.black), Qt.ItemDataRole.ForegroundRole)
                it.setToolTip(
                    f"Actual: {self._class_values[i]}\n"
                    f"Predicted: {self._class_values[j]}\n"
                    f"Count: {int(raw_val)}"
                )

    # ── Selection helpers ─────────────────────────────────────────────

    def _select_correct(self) -> None:
        if not self._class_values:
            return
        n = len(self._class_values)
        sel = QItemSelection()
        for i in range(n):
            idx = self._table_model.index(i + 2, i + 2)
            sel.select(idx, idx)
        self._table.selectionModel().select(
            sel, QItemSelectionModel.SelectionFlag.ClearAndSelect
        )

    def _select_wrong(self) -> None:
        if not self._class_values:
            return
        n = len(self._class_values)
        sel = QItemSelection()
        for i in range(n):
            for j in range(n):
                if i != j:
                    idx = self._table_model.index(i + 2, j + 2)
                    sel.select(idx, idx)
        self._table.selectionModel().select(
            sel, QItemSelectionModel.SelectionFlag.ClearAndSelect
        )
