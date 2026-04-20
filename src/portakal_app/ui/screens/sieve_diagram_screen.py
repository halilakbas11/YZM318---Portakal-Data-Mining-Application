from __future__ import annotations

import math

import numpy as np
from PySide6.QtCore import QLineF, QRect, Qt, Signal, QSize
from PySide6.QtGui import QColor, QFont, QFontMetrics, QPainter, QPen
from PySide6.QtWidgets import QComboBox, QHBoxLayout, QLabel, QPushButton, QSizePolicy, QToolTip, QVBoxLayout, QWidget

from portakal_app.data.models import DatasetHandle
from portakal_app.data.services.generated_dataset_service import GeneratedDatasetService
from portakal_app.models import WorkflowPayload
from portakal_app.ui.screens.node_screen import WorkflowNodeScreenSupport
from portakal_app.ui.screens.visualize_common import (
    build_selection_outputs,
    categorical_candidate_columns,
    categorical_view,
    feature_names_from_payload,
)


def _pearson_residual(observed: int, expected: float) -> float:
    if expected <= 0:
        return 0.0
    return (observed - expected) / math.sqrt(expected)


def _chi2_sf(value: float, degrees: int) -> float:
    if value <= 0:
        return 1.0
    z_score = ((value / degrees) ** (1 / 3) - (1 - 2 / (9 * degrees))) / math.sqrt(2 / (9 * degrees))
    if z_score > 6:
        return 0.0
    if z_score < -6:
        return 1.0
    return 0.5 * math.erfc(z_score / math.sqrt(2))


def _chi_square_and_p(
    joint: dict[str, dict[str, int]],
    row_totals: dict[str, int],
    col_totals: dict[str, int],
    n_total: int,
) -> tuple[float, float]:
    chi2 = 0.0
    for row_label, row_dict in joint.items():
        for col_label, observed in row_dict.items():
            expected = (row_totals.get(row_label, 0) * col_totals.get(col_label, 0)) / max(1, n_total)
            if expected > 0:
                chi2 += (observed - expected) ** 2 / expected
    degrees = max(1, (len(row_totals) - 1) * (len(col_totals) - 1))
    return chi2, _chi2_sf(chi2, degrees)


def _cramers_v(chi2: float, n_total: int, n_rows: int, n_cols: int) -> float:
    scale = min(n_rows - 1, n_cols - 1)
    if scale <= 0 or n_total <= 0:
        return 0.0
    return math.sqrt(chi2 / (n_total * scale))


def _check_cochran(expected: np.ndarray) -> str | None:
    cells = expected.size
    if cells == 0:
        return "no cells in contingency table"
    epsilon = 1e-12
    if int((expected < 1.0 - epsilon).sum()) > 0:
        return "some expected frequencies are below 1"
    if int((expected < 5.0 - epsilon).sum()) > 0.2 * cells:
        return "more than 20% of expected frequencies are below 5"
    return None


def _sieve_fill_color(pearson: float) -> QColor:
    if pearson > 0:
        channel = max(int(255 - 20 * pearson), 55)
        return QColor(channel, channel, 255)
    if pearson < 0:
        channel = max(int(255 + 20 * pearson), 55)
        return QColor(255, channel, channel)
    return QColor(224, 224, 224)


def _sieve_hatch_spacing(pearson: float) -> float:
    spacing = 20 - 1.6 * pearson if pearson >= 0 else 20 - 8 * pearson
    return max(3.0, spacing)


class _SieveWidget(QWidget):
    selectionChanged = Signal(list)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._data: dict[str, dict[str, tuple[int, float, float]]] = {}
        self._row_vals: list[str] = []
        self._col_vals: list[str] = []
        self._row_var = ""
        self._col_var = ""
        self._cell_rects: list[tuple[QRect, str, str, int, float, float]] = []
        self._row_indices_by_pair: dict[tuple[str, str], tuple[int, ...]] = {}
        self._selected_pairs: set[tuple[str, str]] = set()
        self.setMinimumHeight(220)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMouseTracking(True)

    def set_data(
        self,
        data: dict[str, dict[str, tuple[int, float, float]]],
        row_vals: list[str],
        col_vals: list[str],
        row_var: str = "",
        col_var: str = "",
        row_indices_by_pair: dict[tuple[str, str], tuple[int, ...]] | None = None,
    ) -> None:
        self._data = data
        self._row_vals = row_vals
        self._col_vals = col_vals
        self._row_var = row_var
        self._col_var = col_var
        self._row_indices_by_pair = dict(row_indices_by_pair or {})
        self._cell_rects = []
        self.update()

    def set_selected_pairs(self, pairs: set[tuple[str, str]]) -> None:
        self._selected_pairs = set(pairs)
        self.update()

    def mouseMoveEvent(self, event) -> None:
        pos = event.position().toPoint()
        for rect, row_label, col_label, observed, expected, pearson in self._cell_rects:
            if rect.contains(pos):
                chi_contrib = (observed - expected) ** 2 / expected if expected > 0 else 0.0
                tip = (
                    f"<b>{self._row_var}</b>: {row_label}<br>"
                    f"<b>{self._col_var}</b>: {col_label}<br>"
                    f"Observed: {observed}<br>"
                    f"Expected: {expected:.2f}<br>"
                    f"Pearson r: {pearson:+.3f}<br>"
                    f"χ² contribution: {chi_contrib:.3f}"
                )
                QToolTip.showText(event.globalPosition().toPoint(), tip, self)
                return
        QToolTip.hideText()

    def leaveEvent(self, _event) -> None:
        QToolTip.hideText()

    def mousePressEvent(self, event) -> None:
        if event.button() != Qt.MouseButton.LeftButton:
            return
        pos = event.position().toPoint()
        for rect, row_label, col_label, *_ in self._cell_rects:
            if not rect.contains(pos):
                continue
            pair = (row_label, col_label)
            if event.modifiers() & Qt.KeyboardModifier.ControlModifier:
                if pair in self._selected_pairs:
                    self._selected_pairs.remove(pair)
                else:
                    self._selected_pairs.add(pair)
            else:
                self._selected_pairs = {pair}
            rows = sorted({row for selected in self._selected_pairs for row in self._row_indices_by_pair.get(selected, ())})
            self.selectionChanged.emit(rows)
            self.update()
            return
        if not (event.modifiers() & Qt.KeyboardModifier.ControlModifier):
            self._selected_pairs = set()
            self.selectionChanged.emit([])
            self.update()

    def paintEvent(self, _event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), QColor("#fffdf9"))

        if not self._data or not self._row_vals or not self._col_vals:
            painter.setPen(QColor("#8d877d"))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "No data.\nSelect two categorical columns.")
            return

        self._cell_rects = []
        margin_left = 96
        margin_top = 48
        margin_right = 12
        margin_bottom = 60
        chart_width = self.width() - margin_left - margin_right
        chart_height = self.height() - margin_top - margin_bottom
        if chart_width < 10 or chart_height < 10:
            return

        cell_width = chart_width / max(1, len(self._col_vals))
        cell_height = chart_height / max(1, len(self._row_vals))
        max_expected = max((item[1] for row in self._data.values() for item in row.values() if item[1] > 0), default=1.0)

        painter.setFont(QFont(self.font().family(), 8))
        metrics = QFontMetrics(painter.font())
        painter.setPen(QColor("#3b2a10"))
        painter.setFont(QFont(self.font().family(), 9, QFont.Weight.Bold))
        painter.drawText(margin_left, 2, chart_width, 16, Qt.AlignmentFlag.AlignCenter, self._col_var)
        painter.save()
        painter.translate(10, margin_top + chart_height // 2)
        painter.rotate(-90)
        painter.drawText(-60, -8, 120, 16, Qt.AlignmentFlag.AlignCenter, self._row_var)
        painter.restore()

        painter.setFont(QFont(self.font().family(), 8))
        for index, label in enumerate(self._col_vals):
            text = metrics.elidedText(label, Qt.TextElideMode.ElideRight, max(8, int(cell_width) - 4))
            painter.setPen(QColor("#534b40"))
            painter.drawText(int(margin_left + index * cell_width), 18, int(cell_width), margin_top - 20, Qt.AlignmentFlag.AlignCenter, text)
        for index, label in enumerate(self._row_vals):
            text = metrics.elidedText(label, Qt.TextElideMode.ElideRight, margin_left - 26)
            painter.setPen(QColor("#534b40"))
            painter.drawText(22, int(margin_top + index * cell_height), margin_left - 26, int(cell_height), Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter, text)

        for row_index, row_label in enumerate(self._row_vals):
            for col_index, col_label in enumerate(self._col_vals):
                observed, expected, pearson = self._data.get(row_label, {}).get(col_label, (0, 0.0, 0.0))
                left = int(margin_left + col_index * cell_width)
                top = int(margin_top + row_index * cell_height)
                width = int(cell_width)
                height = int(cell_height)
                painter.setBrush(QColor("#f0ede7"))
                painter.setPen(QPen(QColor("#d0ccc3"), 1))
                painter.drawRect(left, top, width, height)

                if expected > 0:
                    scale = math.sqrt(expected / max_expected)
                    inner_width = max(4, int(width * 0.88 * scale))
                    inner_height = max(4, int(height * 0.88 * scale))
                    inner_left = left + (width - inner_width) // 2
                    inner_top = top + (height - inner_height) // 2
                    painter.setPen(Qt.PenStyle.NoPen)
                    painter.setBrush(_sieve_fill_color(pearson))
                    painter.drawRect(inner_left, inner_top, inner_width, inner_height)
                    painter.setPen(QPen(QColor(100, 100, 100, 120), 0.8))
                    painter.setClipRect(inner_left, inner_top, inner_width, inner_height)
                    spacing = _sieve_hatch_spacing(pearson)
                    if pearson >= 0:
                        offset = 0.0
                        while offset < inner_width + inner_height:
                            painter.drawLine(QLineF(float(inner_left), float(inner_top + offset), float(inner_left + offset + inner_width), float(inner_top)))
                            offset += spacing
                    else:
                        offset = 0.0
                        while offset < inner_width + inner_height:
                            painter.drawLine(QLineF(float(inner_left + offset), float(inner_top + inner_height), float(inner_left), float(inner_top + inner_height - offset)))
                            offset += spacing
                    painter.setClipping(False)
                    if width > 32 and height > 20:
                        painter.setPen(QColor("#1a1a1a"))
                        painter.drawText(left + 2, top + 2, width - 4, height - 4, Qt.AlignmentFlag.AlignCenter, f"{observed}\n({pearson:+.2f})")

                self._cell_rects.append((QRect(left, top, width, height), row_label, col_label, observed, expected, pearson))
                if (row_label, col_label) in self._selected_pairs:
                    painter.setPen(QPen(QColor("#111111"), 2.5, Qt.PenStyle.DotLine))
                    painter.setBrush(Qt.BrushStyle.NoBrush)
                    painter.drawRect(left + 1, top + 1, max(1, width - 2), max(1, height - 2))

        legend_y = self.height() - margin_bottom + 10
        items = [
            ("r > 0: more than expected (/)", _sieve_fill_color(4.0)),
            ("r < 0: less than expected (\\)", _sieve_fill_color(-4.0)),
        ]
        for legend_index, (label, color) in enumerate(items):
            legend_left = margin_left + legend_index * 220
            painter.setBrush(color)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawRect(legend_left, legend_y, 14, 12)
            painter.setPen(QColor("#534b40"))
            painter.drawText(legend_left + 18, legend_y, 190, 14, Qt.AlignmentFlag.AlignVCenter, label)

        painter.end()


class SieveDiagramScreen(QWidget, WorkflowNodeScreenSupport):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._init_workflow_node_support()
        self._dataset: DatasetHandle | None = None
        self._builder = GeneratedDatasetService()
        self._input_features: tuple[str, ...] = ()
        self._candidate_columns: list[str] = []
        self._selected_rows: list[int] = []
        self._selected_dataset: DatasetHandle | None = None
        self._annotated_dataset: DatasetHandle | None = None
        self._selected_pairs: set[tuple[str, str]] = set()

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        self._attr_box = QWidget()
        attr_layout = QHBoxLayout(self._attr_box)
        attr_layout.setContentsMargins(0, 0, 0, 0)
        attr_layout.setSpacing(8)
        self._row_combo = QComboBox()
        self._row_combo.currentTextChanged.connect(self._refresh)
        self._col_combo = QComboBox()
        self._col_combo.currentTextChanged.connect(self._refresh)
        self._best_btn = QPushButton("Score Combinations")
        self._best_btn.clicked.connect(self._find_best_pair)
        attr_layout.addWidget(self._row_combo, 1)
        cross_label = QLabel("\u2717")
        cross_label.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        attr_layout.addWidget(cross_label)
        attr_layout.addWidget(self._col_combo, 1)
        attr_layout.addWidget(self._best_btn, 0)
        layout.addWidget(self._attr_box)

        self._chart = _SieveWidget()
        self._chart.selectionChanged.connect(self._handle_selection_changed)
        layout.addWidget(self._chart, 1)

        self._warning_label = QLabel("")
        self._warning_label.setWordWrap(True)
        self._warning_label.setStyleSheet("color: #92400e;")
        layout.addWidget(self._warning_label)
        self._status_label = QLabel("Load data to visualise associations.")
        layout.addWidget(self._status_label)

    def documentation_url(self) -> str:
        return "https://orangedatamining.com/widget-catalog/visualize/sievediagram/"

    def sizeHint(self) -> QSize:
        return QSize(450, 550)

    def set_input_payload(self, payload) -> None:
        if payload is None:
            self._dataset = None
            self._input_features = ()
        elif payload.port_label == "Data":
            self._dataset = payload.dataset
        elif payload.port_label == "Features":
            self._input_features = feature_names_from_payload(payload)
        self._sync_controls()
        self._refresh()

    def current_output_dataset(self) -> DatasetHandle | None:
        return self._selected_dataset

    def current_output_datasets(self) -> dict[str, DatasetHandle | None] | None:
        return {"Selected Data": self._selected_dataset, "Annotated Data": self._annotated_dataset}

    def current_output_payloads(self) -> dict[str, WorkflowPayload | None] | None:
        return {
            "Selected Data": None if self._selected_dataset is None else WorkflowPayload("Selected Data", self._selected_dataset),
            "Annotated Data": None if self._annotated_dataset is None else WorkflowPayload("Annotated Data", self._annotated_dataset),
        }

    def serialize_node_state(self) -> dict[str, object]:
        return {"row": self._row_combo.currentText(), "col": self._col_combo.currentText(), "selected_rows": list(self._selected_rows)}

    def restore_node_state(self, payload: dict[str, object]) -> None:
        self._selected_rows = sorted({int(value) for value in payload.get("selected_rows", []) if isinstance(value, (int, float))})
        for combo, key in ((self._row_combo, "row"), (self._col_combo, "col")):
            text = str(payload.get(key, ""))
            index = combo.findText(text)
            if index >= 0:
                combo.setCurrentIndex(index)

    def _sync_controls(self) -> None:
        dataset = self._dataset
        self._candidate_columns = categorical_candidate_columns(dataset)
        self._populate_combo(self._row_combo, self._candidate_columns)
        self._populate_combo(self._col_combo, self._candidate_columns)
        if dataset is not None:
            default_row, default_col = self._default_pair()
            if self._row_combo.currentText() not in self._candidate_columns and default_row:
                self._row_combo.setCurrentText(default_row)
            if self._col_combo.currentText() not in self._candidate_columns and default_col:
                self._col_combo.setCurrentText(default_col)
            if self._col_combo.currentText() == self._row_combo.currentText() and default_col:
                self._col_combo.setCurrentText(default_col)
        self._apply_feature_override()

    def _populate_combo(self, combo: QComboBox, names: list[str]) -> None:
        current = combo.currentText()
        combo.blockSignals(True)
        combo.clear()
        combo.addItems(names)
        if current in names:
            combo.setCurrentText(current)
        elif names:
            combo.setCurrentIndex(0)
        combo.blockSignals(False)

    def _default_pair(self) -> tuple[str, str]:
        if not self._candidate_columns:
            return "", ""
        preferred_x = self._candidate_columns[0]
        if self._dataset is not None:
            for target in self._dataset.domain.target_columns:
                if target.name in self._candidate_columns:
                    preferred_x = target.name
                    break
        preferred_y = next((name for name in self._candidate_columns if name != preferred_x), preferred_x)
        return preferred_x, preferred_y

    def _apply_feature_override(self) -> None:
        if not self._input_features:
            self._attr_box.setEnabled(True)
            self._best_btn.setEnabled(bool(self._candidate_columns))
            return
        matched = [name for name in self._input_features if name in self._candidate_columns]
        if not matched:
            self._attr_box.setEnabled(True)
            self._best_btn.setEnabled(bool(self._candidate_columns))
            self._status_label.setText("Features from the input signal are not present in the data.")
            return
        selected = (matched * 2)[:2]
        self._row_combo.setCurrentText(selected[0])
        self._col_combo.setCurrentText(selected[1])
        self._attr_box.setEnabled(False)
        self._best_btn.setEnabled(False)

    def _build_contingency(
        self,
        row_col: str,
        col_col: str,
    ) -> tuple[
        dict[str, dict[str, int]],
        dict[str, int],
        dict[str, int],
        int,
        list[str],
        list[str],
        dict[tuple[str, str], tuple[int, ...]],
    ] | None:
        if self._dataset is None:
            return None
        row_view = categorical_view(self._dataset, row_col, bins=4, discretize_numeric=True)
        col_view = categorical_view(self._dataset, col_col, bins=4, discretize_numeric=True)
        if row_view is None or col_view is None:
            return None
        joint: dict[str, dict[str, int]] = {}
        row_totals: dict[str, int] = {}
        col_totals: dict[str, int] = {}
        row_indices_by_pair: dict[tuple[str, str], list[int]] = {}
        for index, (row_value, col_value) in enumerate(zip(row_view.labels, col_view.labels)):
            row_totals[row_value] = row_totals.get(row_value, 0) + 1
            col_totals[col_value] = col_totals.get(col_value, 0) + 1
            joint.setdefault(row_value, {})
            joint[row_value][col_value] = joint[row_value].get(col_value, 0) + 1
            row_indices_by_pair.setdefault((row_value, col_value), []).append(index)
        return (
            joint,
            row_totals,
            col_totals,
            self._dataset.row_count,
            list(row_view.categories),
            list(col_view.categories),
            {key: tuple(values) for key, values in row_indices_by_pair.items()},
        )

    def _find_best_pair(self) -> None:
        if len(self._candidate_columns) < 2:
            return
        best_pair = (self._candidate_columns[0], self._candidate_columns[1])
        best_score = -1.0
        for index, row_col in enumerate(self._candidate_columns):
            for col_col in self._candidate_columns[index + 1:]:
                built = self._build_contingency(row_col, col_col)
                if built is None:
                    continue
                joint, row_totals, col_totals, total, _, _, _ = built
                chi2, _ = _chi_square_and_p(joint, row_totals, col_totals, total)
                if chi2 > best_score:
                    best_score = chi2
                    best_pair = (row_col, col_col)
        self._row_combo.setCurrentText(best_pair[0])
        self._col_combo.setCurrentText(best_pair[1])

    def _refresh(self) -> None:
        if self._dataset is None:
            self._chart.set_data({}, [], [], "", "", {})
            self._warning_label.clear()
            self._status_label.setText("Load data to visualise associations.")
            self._handle_selection_changed([])
            return

        row_col = self._row_combo.currentText()
        col_col = self._col_combo.currentText()
        if not row_col or not col_col or row_col == col_col:
            self._chart.set_data({}, [], [], "", "", {})
            self._warning_label.clear()
            self._status_label.setText("Select two different columns.")
            self._handle_selection_changed([])
            return

        built = self._build_contingency(row_col, col_col)
        if built is None:
            self._chart.set_data({}, [], [], "", "", {})
            self._warning_label.clear()
            self._status_label.setText("Selected attributes are not available.")
            self._handle_selection_changed([])
            return
        joint, row_totals, col_totals, total, row_vals, col_vals, row_indices_by_pair = built

        data: dict[str, dict[str, tuple[int, float, float]]] = {}
        expected_values: list[float] = []
        for row_label in row_vals:
            data[row_label] = {}
            for col_label in col_vals:
                observed = joint.get(row_label, {}).get(col_label, 0)
                expected = (row_totals.get(row_label, 0) * col_totals.get(col_label, 0)) / max(1, total)
                expected_values.append(expected)
                data[row_label][col_label] = (observed, expected, _pearson_residual(observed, expected))

        chi2, p_value = _chi_square_and_p(joint, row_totals, col_totals, total)
        cramer_v = _cramers_v(chi2, total, len(row_vals), len(col_vals))
        self._chart.set_data(data, row_vals, col_vals, row_col, col_col, row_indices_by_pair)
        self._chart.set_selected_pairs(self._selected_pairs)

        cochran = _check_cochran(np.asarray(expected_values, dtype=float))
        self._warning_label.setText(f"Data does not meet the Cochran's rule: {cochran}" if cochran else "")
        p_text = f"{p_value:.3f}" if p_value >= 0.001 else "< 0.001"
        self._status_label.setText(f"χ²={chi2:.2f}, p={p_text}  |  N = {total}  |  V = {cramer_v:.3f}")
        self._handle_selection_changed(self._selected_rows)

    def _handle_selection_changed(self, rows: list[int]) -> None:
        normalized = sorted({int(value) for value in rows if isinstance(value, (int, float))})
        self._selected_rows = [row for row in normalized if self._dataset is not None and 0 <= row < self._dataset.row_count]
        pair_lookup: dict[tuple[str, str], tuple[int, ...]] = self._chart._row_indices_by_pair
        self._selected_pairs = {
            pair
            for pair, indices in pair_lookup.items()
            if any(row in self._selected_rows for row in indices)
        }
        self._selected_dataset, self._annotated_dataset = build_selection_outputs(
            self._dataset,
            self._selected_rows,
            generated_by="sieve-diagram",
            service=self._builder,
        )
        self._notify_output_changed()
