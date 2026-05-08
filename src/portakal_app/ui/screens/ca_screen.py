from __future__ import annotations

import uuid

import numpy as np
import polars as pl
from PySide6.QtCore import QPointF, QRect, Qt
from PySide6.QtGui import QColor, QFont, QPainter, QPen
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from portakal_app.data.models import DatasetHandle
from portakal_app.data.services.generated_dataset_service import GeneratedDatasetService
from portakal_app.models import WorkflowPayload
from portakal_app.ui import i18n
from portakal_app.ui.screens.node_screen import WorkflowNodeScreenSupport
from portakal_app.ui.screens.visualize_common import categorical_candidate_columns, nice_ticks


def _correspondence_analysis(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    total = float(np.sum(matrix))
    if total <= 0:
        raise ValueError("Contingency table is empty.")
    p = matrix / total
    row_masses = p.sum(axis=1)
    col_masses = p.sum(axis=0)
    row_safe = np.maximum(row_masses, 1e-12)
    col_safe = np.maximum(col_masses, 1e-12)
    standardized = (p - np.outer(row_masses, col_masses))
    standardized *= (1.0 / np.sqrt(row_safe))[:, None]
    standardized *= (1.0 / np.sqrt(col_safe))[None, :]
    u, sv, vt = np.linalg.svd(standardized, full_matrices=False)
    keep = min(len(row_masses) - 1, len(col_masses) - 1, len(sv))
    if keep < 1:
        raise ValueError("Need at least two row and column categories.")
    u = u[:, :keep]
    sv = sv[:keep]
    v = vt[:keep, :].T
    row_coords = u * (sv * (1.0 / np.sqrt(row_safe))[:, None])
    col_coords = v * (sv * (1.0 / np.sqrt(col_safe))[:, None])
    explained = (sv**2) / max(float(np.sum(sv**2)), 1e-12)
    return row_coords, col_coords, explained


class _CAScatterCanvas(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._points: list[tuple[float, float, str, QColor]] = []
        self._x_label = "Component 1"
        self._y_label = "Component 2"
        self.setMinimumHeight(300)

    def set_data(self, points: list[tuple[float, float, str, QColor]], x_label: str, y_label: str) -> None:
        self._points = points
        self._x_label = x_label
        self._y_label = y_label
        self.update()

    def clear(self) -> None:
        self._points = []
        self.update()

    def paintEvent(self, _event) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), QColor("#ffffff"))
        if not self._points:
            painter.setPen(QColor("#8d877d"))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "No CA data.\nConnect data and press Apply.")
            painter.end()
            return

        chart = QRect(72, 28, max(80, self.width() - 92), max(60, self.height() - 88))
        xs = [point[0] for point in self._points]
        ys = [point[1] for point in self._points]
        x0, x1 = min(xs), max(xs)
        y0, y1 = min(ys), max(ys)
        xpad = max((x1 - x0) * 0.15, 0.3)
        ypad = max((y1 - y0) * 0.15, 0.3)
        x0, x1 = x0 - xpad, x1 + xpad
        y0, y1 = y0 - ypad, y1 + ypad

        def to_px(x: float, y: float) -> tuple[int, int]:
            px = chart.left() + int((x - x0) / max(x1 - x0, 1e-9) * chart.width())
            py = chart.bottom() - int((y - y0) / max(y1 - y0, 1e-9) * chart.height())
            return px, py

        painter.setPen(QPen(QColor("#e8e4de"), 1, Qt.PenStyle.DotLine))
        for tick in nice_ticks(x0, x1, 6):
            px, _ = to_px(tick, y0)
            painter.drawLine(px, chart.top(), px, chart.bottom())
        for tick in nice_ticks(y0, y1, 6):
            _, py = to_px(x0, tick)
            painter.drawLine(chart.left(), py, chart.right(), py)

        painter.setPen(QPen(QColor("#9b9488"), 1))
        painter.drawRect(chart)

        font = QFont(self.font().family(), 9)
        painter.setFont(font)
        for x, y, label, color in self._points:
            px, py = to_px(x, y)
            painter.setBrush(color)
            painter.setPen(QPen(QColor("#ffffff"), 2))
            painter.drawEllipse(QPointF(px, py), 7.0, 7.0)
            painter.setPen(QColor("#1a1310"))
            painter.drawText(QRect(px + 10, py - 10, 220, 20), Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, label)

        bold = QFont(self.font().family(), 9)
        bold.setBold(True)
        painter.setFont(bold)
        painter.setPen(QColor("#2f2820"))
        painter.drawText(chart.adjusted(0, chart.height() + 18, 0, 36), Qt.AlignmentFlag.AlignCenter, self._x_label)
        painter.save()
        painter.translate(14, chart.center().y())
        painter.rotate(-90)
        painter.drawText(QRect(-90, -8, 180, 16), Qt.AlignmentFlag.AlignCenter, self._y_label)
        painter.restore()
        painter.end()


class CAScreen(QWidget, WorkflowNodeScreenSupport):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._init_workflow_node_support()
        self._builder = GeneratedDatasetService()
        self._dataset_handle: DatasetHandle | None = None
        self._output_payload: WorkflowPayload | None = None
        self._build_ui()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(10)

        variables_box = QGroupBox(i18n.t("Variables"))
        variables_layout = QVBoxLayout(variables_box)
        self._filter_input = QLineEdit()
        self._filter_input.setPlaceholderText(i18n.t("Filter..."))
        variables_layout.addWidget(self._filter_input)
        self._variables_list = QListWidget()
        self._variables_list.setMaximumHeight(120)
        variables_layout.addWidget(self._variables_list)
        root.addWidget(variables_box)

        axes_box = QGroupBox(i18n.t("Axes"))
        axes_form = QFormLayout(axes_box)
        self._axis_x_combo = QComboBox()
        self._axis_y_combo = QComboBox()
        for index in range(1, 11):
            self._axis_x_combo.addItem(str(index))
            self._axis_y_combo.addItem(str(index))
        self._axis_x_combo.setCurrentIndex(0)
        self._axis_y_combo.setCurrentIndex(1)
        axes_form.addRow(i18n.t("X:"), self._axis_x_combo)
        axes_form.addRow(i18n.t("Y:"), self._axis_y_combo)
        root.addWidget(axes_box)

        inertia_box = QGroupBox(i18n.t("Contribution to Inertia"))
        inertia_layout = QVBoxLayout(inertia_box)
        self._axis1_label = QLabel(i18n.t("Axis 1: -"))
        self._axis2_label = QLabel(i18n.t("Axis 2: -"))
        inertia_layout.addWidget(self._axis1_label)
        inertia_layout.addWidget(self._axis2_label)
        root.addWidget(inertia_box)

        self._canvas = _CAScatterCanvas()
        root.addWidget(self._canvas, 1)

        self._status_label = QLabel(i18n.t("No dataset loaded."))
        self._status_label.setWordWrap(True)
        root.addWidget(self._status_label)

        footer = QHBoxLayout()
        self.cb_apply_auto = QCheckBox(i18n.t("Apply Automatically"))
        self.cb_apply_auto.setChecked(True)
        footer.addWidget(self.cb_apply_auto)
        footer.addStretch(1)
        self._apply_button = QPushButton(i18n.t("Apply"))
        self._apply_button.setProperty("primary", True)
        self._apply_button.clicked.connect(self._apply)
        footer.addWidget(self._apply_button)
        root.addLayout(footer)

        self._filter_input.textChanged.connect(self._filter_vars)
        self._variables_list.currentRowChanged.connect(lambda _row: self._schedule_auto_apply())
        self._axis_x_combo.currentIndexChanged.connect(self._schedule_auto_apply)
        self._axis_y_combo.currentIndexChanged.connect(self._schedule_auto_apply)
        self.cb_apply_auto.toggled.connect(lambda _checked: self._schedule_auto_apply())

    def set_input_payload(self, payload: WorkflowPayload | None) -> None:
        self._dataset_handle = payload.dataset if payload is not None else None
        self._output_payload = None
        self._populate_var_list()
        self._schedule_auto_apply()

    def current_output_payload(self) -> WorkflowPayload | None:
        return self._output_payload

    def help_text(self) -> str:
        return "Project categorical values into correspondence-analysis coordinates."

    def documentation_url(self) -> str:
        return "https://orangedatamining.com/widget-catalog/unsupervised/correspondenceanalysis/"

    def serialize_node_state(self) -> dict[str, object]:
        return {
            "axis_x": self._axis_x_combo.currentIndex(),
            "axis_y": self._axis_y_combo.currentIndex(),
            "selected_var": self._variables_list.currentItem().data(Qt.ItemDataRole.UserRole) if self._variables_list.currentItem() else "",
            "auto_apply": self.cb_apply_auto.isChecked(),
        }

    def restore_node_state(self, payload: dict[str, object]) -> None:
        self._axis_x_combo.setCurrentIndex(int(payload.get("axis_x", 0)))
        self._axis_y_combo.setCurrentIndex(int(payload.get("axis_y", 1)))
        self.cb_apply_auto.setChecked(bool(payload.get("auto_apply", True)))

    def _populate_var_list(self) -> None:
        self._variables_list.clear()
        dataset = self._dataset_handle
        if dataset is None:
            return
        names = categorical_candidate_columns(dataset)
        for column in dataset.domain.target_columns:
            if column.name in names:
                item = QListWidgetItem(f"  {column.name}")
                item.setData(Qt.ItemDataRole.UserRole, column.name)
                item.setForeground(QColor("#0ea5e9"))
                self._variables_list.addItem(item)
        for name in names:
            if any(column.name == name for column in dataset.domain.target_columns):
                continue
            item = QListWidgetItem(f"  {name}")
            item.setData(Qt.ItemDataRole.UserRole, name)
            self._variables_list.addItem(item)
        if self._variables_list.count():
            self._variables_list.setCurrentRow(0)

    def _filter_vars(self, text: str) -> None:
        query = text.strip().lower()
        for row in range(self._variables_list.count()):
            item = self._variables_list.item(row)
            item.setHidden(query not in item.text().lower())

    def _apply(self) -> None:
        self._output_payload = None
        self._canvas.clear()
        self._axis1_label.setText(i18n.t("Axis 1: -"))
        self._axis2_label.setText(i18n.t("Axis 2: -"))
        if self._dataset_handle is None:
            self._status_label.setText(i18n.t("No dataset loaded."))
            self._notify_output_changed()
            return

        try:
            dataset = self._dataset_handle
            item = self._variables_list.currentItem()
            selected = item.data(Qt.ItemDataRole.UserRole) if item is not None else None
            if not selected or str(selected) not in dataset.dataframe.columns:
                raise ValueError("Select a variable first.")

            row_name = str(selected)
            row_values = dataset.dataframe.get_column(row_name).to_list()
            row_categories = [value for value in dict.fromkeys("(missing)" if value is None else str(value) for value in row_values)]

            dummy_columns: list[np.ndarray] = []
            dummy_labels: list[str] = []
            for column in dataset.domain.columns:
                if column.name == row_name or column.name not in dataset.dataframe.columns:
                    continue
                series = dataset.dataframe.get_column(column.name)
                values = series.to_list()
                if column.logical_type == "numeric":
                    numeric = np.asarray([float(value) if value is not None else np.nan for value in values], dtype=float)
                    finite = numeric[np.isfinite(numeric)]
                    if finite.size == 0:
                        continue
                    edges = np.unique(np.quantile(finite, [0.0, 0.25, 0.5, 0.75, 1.0]))
                    if len(edges) < 2:
                        continue
                    bins = np.digitize(numeric, edges[1:-1], right=False)
                    for bin_index in range(len(edges) - 1):
                        dummy_columns.append(np.asarray([1.0 if int(value) == bin_index else 0.0 for value in bins], dtype=float))
                        dummy_labels.append(f"{column.name}[{bin_index + 1}]")
                else:
                    categories = [value for value in dict.fromkeys("(missing)" if value is None else str(value) for value in values)]
                    for category in categories:
                        dummy_columns.append(
                            np.asarray([1.0 if ("(missing)" if value is None else str(value)) == category else 0.0 for value in values], dtype=float)
                        )
                        dummy_labels.append(f"{column.name}={category}")

            if not dummy_columns:
                raise ValueError("Need at least one additional variable for correspondence analysis.")

            dummy_matrix = np.column_stack(dummy_columns)
            row_index = {category: idx for idx, category in enumerate(row_categories)}
            contingency = np.zeros((len(row_categories), dummy_matrix.shape[1]), dtype=float)
            for sample_index, value in enumerate(row_values):
                label = "(missing)" if value is None else str(value)
                contingency[row_index[label]] += dummy_matrix[sample_index]

            nonzero_rows = contingency.sum(axis=1) > 0
            nonzero_cols = contingency.sum(axis=0) > 0
            contingency = contingency[nonzero_rows][:, nonzero_cols]
            kept_rows = [category for category, keep in zip(row_categories, nonzero_rows) if keep]
            kept_cols = [label for label, keep in zip(dummy_labels, nonzero_cols) if keep]
            row_coords, col_coords, explained = _correspondence_analysis(contingency)

            x_axis = min(self._axis_x_combo.currentIndex(), row_coords.shape[1] - 1)
            y_axis = min(self._axis_y_combo.currentIndex(), row_coords.shape[1] - 1)
            x_pct = float(explained[x_axis]) * 100.0
            y_pct = float(explained[y_axis]) * 100.0
            self._axis1_label.setText(i18n.tf("Axis {axis}: {pct:.2f}%", axis=x_axis + 1, pct=x_pct))
            self._axis2_label.setText(i18n.tf("Axis {axis}: {pct:.2f}%", axis=y_axis + 1, pct=y_pct))

            palette = [QColor("#3b82f6"), QColor("#e07020"), QColor("#22c55e"), QColor("#a855f7"), QColor("#f43f5e"), QColor("#0ea5e9")]
            points = [
                (float(row_coords[index, x_axis]), float(row_coords[index, y_axis]), kept_rows[index], palette[index % len(palette)])
                for index in range(len(kept_rows))
            ]
            self._canvas.set_data(
                points,
                f"Component {x_axis + 1} ({x_pct:.1f}%)",
                f"Component {y_axis + 1} ({y_pct:.1f}%)",
            )

            dataframe = pl.DataFrame(
                {
                    "Label": kept_rows + kept_cols,
                    "Kind": ["Row"] * len(kept_rows) + ["Column"] * len(kept_cols),
                    "Component 1": [float(value) for value in row_coords[:, x_axis].tolist()] + [float(value) for value in col_coords[:, x_axis].tolist()],
                    "Component 2": [float(value) for value in row_coords[:, y_axis].tolist()] + [float(value) for value in col_coords[:, y_axis].tolist()],
                }
            )
            output = self._builder.build_dataset(
                dataframe,
                dataset_id=f"{dataset.dataset_id}-ca-{uuid.uuid4().hex[:8]}",
                display_name=f"{dataset.display_name} (CA Coordinates)",
                file_name=f"{dataset.dataset_id}-ca.csv",
                role_overrides={"Label": "meta", "Kind": "meta", "Component 1": "feature", "Component 2": "feature"},
                annotations={**dataset.annotations, "correspondence_analysis": {"variable": row_name, "explained": explained[:2].tolist()}},
            )
            self._output_payload = WorkflowPayload("Data", output)
            self._status_label.setText(i18n.tf("{rows} row categories, {cols} column categories", rows=len(kept_rows), cols=len(kept_cols)))
        except Exception as exc:
            self._status_label.setText(i18n.tf("Error: {err}", err=exc))

        self._notify_output_changed()
