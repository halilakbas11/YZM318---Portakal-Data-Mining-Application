from __future__ import annotations

from typing import Any

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QPainter
from PySide6.QtWidgets import (
    QComboBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QCheckBox,
    QSpinBox,
    QStyledItemDelegate,
    QStyleOptionViewItem,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from portakal_app.data.models import DatasetHandle, RankedFeature
from portakal_app.data.services.feature_ranking_service import FeatureRankingService
from portakal_app.data.services.file_import_service import FileImportService
from portakal_app.ui.screens.node_screen import WorkflowNodeScreenSupport


class _ScoreBarDelegate(QStyledItemDelegate):
    def paint(self, painter: QPainter, option, index) -> None:
        styled_option = QStyleOptionViewItem(option)
        self.initStyleOption(styled_option, index)
        super().paint(painter, styled_option, index)

        score = index.data(Qt.ItemDataRole.UserRole)
        if not isinstance(score, float):
            return

        rect = option.rect.adjusted(8, option.rect.height() - 10, -8, -3)
        width = int(rect.width() * max(0.0, min(1.0, score)))
        if width <= 0:
            return
        painter.save()
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor("#9fc57e"))
        painter.drawRoundedRect(rect.x(), rect.y(), width, rect.height(), 3, 3)
        painter.restore()


class RankScreen(QWidget, WorkflowNodeScreenSupport):
    AUTO_TARGET = "Auto (Inferred)"
    NO_TARGET = "None (Heuristic)"
    FILTER_OPTIONS = {
        "All": "all",
        "Numeric only": "numeric",
        "Categorical/Text only": "categorical",
    }

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._init_workflow_node_support()
        self._import_service = FileImportService()
        self._ranking_service = FeatureRankingService()
        self._dataset_handle: DatasetHandle | None = None
        self._ranked_rows: list[RankedFeature] = []

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(10)

        header_row = QHBoxLayout()
        self._dataset_label = QLabel("Dataset: none")
        self._dataset_label.setProperty("sectionTitle", True)
        self._dataset_label.setStyleSheet("font-size: 12pt; background: transparent;")
        header_row.addWidget(self._dataset_label, 1)

        self._refresh_button = QPushButton("Refresh Ranking")
        self._refresh_button.setProperty("secondary", True)
        self._refresh_button.clicked.connect(self._refresh_ranking)
        header_row.addWidget(self._refresh_button)
        layout.addLayout(header_row)

        controls_group = QGroupBox("Controls")
        controls_layout = QFormLayout(controls_group)
        controls_layout.setContentsMargins(10, 10, 10, 10)
        controls_layout.setSpacing(10)

        self._target_combo = QComboBox(self)
        self._target_combo.currentTextChanged.connect(lambda _value: self._refresh_ranking())
        controls_layout.addRow("Target", self._target_combo)

        self._feature_filter_combo = QComboBox(self)
        self._feature_filter_combo.addItems(list(self.FILTER_OPTIONS.keys()))
        self._feature_filter_combo.currentTextChanged.connect(lambda _value: self._refresh_ranking())
        controls_layout.addRow("Feature Filter", self._feature_filter_combo)

        self._top_n_spin = QSpinBox(self)
        self._top_n_spin.setRange(1, 999)
        self._top_n_spin.setValue(10)
        self._top_n_spin.valueChanged.connect(lambda _value: self._refresh_ranking())
        controls_layout.addRow("Top N", self._top_n_spin)
        layout.addWidget(controls_group)

        self._summary_label = QLabel("Load a dataset to rank feature usefulness.")
        self._summary_label.setWordWrap(True)
        layout.addWidget(self._summary_label)

        table_group = QGroupBox("Feature Ranking")
        table_layout = QVBoxLayout(table_group)
        table_layout.setContentsMargins(10, 10, 10, 10)
        self._rank_table = QTableWidget(0, 5, self)
        self._rank_table.setHorizontalHeaderLabels(["Feature", "Type", "Score", "Method", "Details"])
        self._rank_table.horizontalHeader().setStretchLastSection(True)
        self._rank_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._rank_table.setItemDelegateForColumn(2, _ScoreBarDelegate(self._rank_table))
        table_layout.addWidget(self._rank_table)
        layout.addWidget(table_group, 1)

        footer = QHBoxLayout()
        footer.addStretch(1)

        self._auto_send_checkbox = QCheckBox("Send Automatically")
        self._auto_send_checkbox.setChecked(False)
        footer.addWidget(self._auto_send_checkbox)

        self._send_button = QPushButton("Send Data")
        self._send_button.setProperty("primary", True)
        self._send_button.clicked.connect(self._notify_output_changed)
        footer.addWidget(self._send_button)

        layout.addLayout(footer)

    def set_dataset(self, dataset_handle: DatasetHandle | str | None) -> None:
        if isinstance(dataset_handle, str):
            try:
                dataset_handle = self._import_service.load(dataset_handle)
            except Exception:
                dataset_handle = None
        self._dataset_handle = dataset_handle
        self._target_combo.clear()
        if dataset_handle is None:
            self._ranked_rows = []
            self._dataset_label.setText("Dataset: none")
            self._summary_label.setText("Load a dataset to rank feature usefulness.")
            self._rank_table.setRowCount(0)
            return
        self._dataset_label.setText(f"Dataset: {dataset_handle.source.path.name}")
        self._target_combo.addItem(self.AUTO_TARGET)
        self._target_combo.addItem(self.NO_TARGET)
        for column in dataset_handle.domain.columns:
            self._target_combo.addItem(column.name)
        self._target_combo.setCurrentText(self.AUTO_TARGET)
        self._refresh_ranking()

    def footer_status_text(self) -> str:
        return str(self._rank_table.rowCount()) if self._dataset_handle is not None else "0"

    def set_input_payload(self, payload) -> None:
        dataset = payload.dataset if payload is not None else None
        self.set_dataset(dataset)

    def current_output_dataset(self) -> DatasetHandle | None:
        if self._dataset_handle is None or not self._ranked_rows:
            return None
        return self._ranking_service.build_scores_dataset(self._dataset_handle, self._ranked_rows)

    def serialize_node_state(self) -> dict[str, object]:
        return {
            "target": self._target_combo.currentText(),
            "feature_filter": self._feature_filter_combo.currentText(),
            "top_n": self._top_n_spin.value(),
            "auto_send": getattr(self, "_auto_send_checkbox", None) is not None and self._auto_send_checkbox.isChecked(),
        }

    def restore_node_state(self, payload: dict[str, object]) -> None:
        self._feature_filter_combo.setCurrentText(str(payload.get("feature_filter") or "All"))
        self._top_n_spin.setValue(int(payload.get("top_n") or 10))
        if hasattr(self, "_auto_send_checkbox"):
            self._auto_send_checkbox.setChecked(bool(payload.get("auto_send", True)))
        target = str(payload.get("target") or self.AUTO_TARGET)
        if self._target_combo.findText(target) >= 0:
            self._target_combo.setCurrentText(target)
        self._refresh_ranking()

    def help_text(self) -> str:
        return (
            "Rank features with automatic method selection, target override, feature filters, "
            "top-N limiting, and inline score bars."
        )

    def documentation_url(self) -> str:
        return "https://orangedatamining.com/widget-catalog/data/rank/"

    def _refresh_ranking(self) -> None:
        if self._dataset_handle is None:
            self._ranked_rows = []
            return
        target_name = self._resolved_target_name()
        feature_filter = self.FILTER_OPTIONS[self._feature_filter_combo.currentText()]
        ranked_rows = self._ranking_service.rank(
            self._dataset_handle,
            target_name=target_name,
            feature_filter=feature_filter,
            top_n=self._top_n_spin.value(),
        )
        self._ranked_rows = ranked_rows
        self._populate_rankings(ranked_rows)
        effective_target = self._effective_target_name()
        if effective_target is None:
            self._summary_label.setText("No target selected. Ranking uses heuristic mode.")
        else:
            self._summary_label.setText(f"Ranking features against target '{effective_target}'.")
            
        self._send_button.setEnabled(len(self._ranked_rows) > 0)
        if hasattr(self, "_auto_send_checkbox") and self._auto_send_checkbox.isChecked():
            self._notify_output_changed()

    def _populate_rankings(self, ranked_rows: list[RankedFeature]) -> None:
        self._rank_table.setRowCount(len(ranked_rows))
        for row_index, ranked in enumerate(ranked_rows):
            self._rank_table.setItem(row_index, 0, QTableWidgetItem(ranked.feature_name))
            self._rank_table.setItem(row_index, 1, QTableWidgetItem(ranked.logical_type))

            score_item = QTableWidgetItem(f"{ranked.score:.4f}")
            score_item.setData(Qt.ItemDataRole.UserRole, float(ranked.score))
            self._rank_table.setItem(row_index, 2, score_item)

            self._rank_table.setItem(row_index, 3, QTableWidgetItem(ranked.method))
            self._rank_table.setItem(row_index, 4, QTableWidgetItem(ranked.details))
        self._rank_table.resizeColumnsToContents()

    def _resolved_target_name(self) -> str | None:
        current = self._target_combo.currentText()
        if current in {"", self.AUTO_TARGET}:
            return None
        if current == self.NO_TARGET:
            return ""
        return current

    def _effective_target_name(self) -> str | None:
        if self._dataset_handle is None:
            return None
        current = self._target_combo.currentText()
        if current == self.NO_TARGET:
            return None
        if current in {"", self.AUTO_TARGET}:
            return next((column.name for column in self._dataset_handle.domain.columns if column.role == "target"), None)
        return current
