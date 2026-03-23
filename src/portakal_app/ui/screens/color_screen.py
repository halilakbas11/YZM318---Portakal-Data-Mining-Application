from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import replace

from PySide6.QtCore import QSize, Qt
from PySide6.QtGui import QColor, QLinearGradient, QPainter
from PySide6.QtWidgets import (
    QColorDialog,
    QFileDialog,
    QCheckBox,
    QComboBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from portakal_app.data.models import DatasetHandle
from portakal_app.data.services.color_settings_service import ColorSettingsService
from portakal_app.ui.screens.node_screen import WorkflowNodeScreenSupport


class ColorValueButton(QPushButton):
    def __init__(self, label: str, color_hex: str, parent: QWidget | None = None) -> None:
        super().__init__(label, parent)
        self._color_hex = color_hex
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self._sync_style()

    @property
    def color_hex(self) -> str:
        return self._color_hex

    def set_color(self, color_hex: str) -> None:
        self._color_hex = color_hex
        self._sync_style()

    def _sync_style(self) -> None:
        self.setStyleSheet(
            f"background: {self._color_hex}; color: white; border: none; border-radius: 8px; padding: 6px 10px;"
        )


class GradientPreview(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._colors: tuple[str, ...] = ()
        self.setMinimumHeight(22)

    def set_colors(self, colors: tuple[str, ...]) -> None:
        self._colors = colors
        self.update()

    def paintEvent(self, event) -> None:
        super().paintEvent(event)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        rect = self.rect().adjusted(1, 1, -1, -1)
        gradient = QLinearGradient(rect.topLeft(), rect.topRight())
        if self._colors:
            step = 1.0 / max(1, len(self._colors) - 1)
            for index, color_hex in enumerate(self._colors):
                gradient.setColorAt(index * step, QColor(color_hex))
        painter.setPen(QColor("#d0c7b8"))
        painter.setBrush(gradient)
        painter.drawRoundedRect(rect, 9, 9)


class ColorScreen(QWidget, WorkflowNodeScreenSupport):
    def __init__(self, parent: QWidget | None = None, service: ColorSettingsService | None = None) -> None:
        super().__init__(parent)
        self._init_workflow_node_support()
        self._service = service or ColorSettingsService()
        self._dataset: DatasetHandle | None = None
        self._output_dataset: DatasetHandle | None = None
        self._callbacks: list[Callable[[DatasetHandle], None]] = []
        self._state: dict[str, object] = {"discrete": {}, "numeric": {}}
        self._discrete_buttons: dict[tuple[str, str], ColorValueButton] = {}
        self._numeric_previews: dict[str, GradientPreview] = {}
        self._numeric_combos: dict[str, QComboBox] = {}

        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(10)

        self._dataset_label = QLabel("Dataset: none")
        self._dataset_label.setProperty("muted", True)
        layout.addWidget(self._dataset_label)

        layout.addWidget(self._build_discrete_panel(), 1)
        layout.addWidget(self._build_numeric_panel(), 1)
        layout.addLayout(self._build_footer())
        self._render_state()

    def sizeHint(self) -> QSize:
        return QSize(860, 720)

    def minimumSizeHint(self) -> QSize:
        return QSize(700, 620)

    def _build_discrete_panel(self) -> QFrame:
        frame = QFrame(self)
        frame.setProperty("panel", True)
        layout = QVBoxLayout(frame)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)

        title = QLabel("Discrete Variables")
        title.setProperty("sectionTitle", True)
        title.setStyleSheet("font-size: 12pt;")
        layout.addWidget(title)

        self._discrete_scroll = QScrollArea(self)
        self._discrete_scroll.setWidgetResizable(True)
        self._discrete_content = QWidget(self._discrete_scroll)
        self._discrete_layout = QVBoxLayout(self._discrete_content)
        self._discrete_layout.setContentsMargins(0, 0, 0, 0)
        self._discrete_layout.setSpacing(8)
        self._discrete_scroll.setWidget(self._discrete_content)
        layout.addWidget(self._discrete_scroll, 1)
        return frame

    def _build_numeric_panel(self) -> QFrame:
        frame = QFrame(self)
        frame.setProperty("panel", True)
        layout = QVBoxLayout(frame)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)

        title = QLabel("Numeric Variables")
        title.setProperty("sectionTitle", True)
        title.setStyleSheet("font-size: 12pt;")
        layout.addWidget(title)

        self._numeric_scroll = QScrollArea(self)
        self._numeric_scroll.setWidgetResizable(True)
        self._numeric_content = QWidget(self._numeric_scroll)
        self._numeric_layout = QVBoxLayout(self._numeric_content)
        self._numeric_layout.setContentsMargins(0, 0, 0, 0)
        self._numeric_layout.setSpacing(8)
        self._numeric_scroll.setWidget(self._numeric_content)
        layout.addWidget(self._numeric_scroll, 1)
        return frame

    def _build_footer(self) -> QHBoxLayout:
        layout = QHBoxLayout()
        layout.setSpacing(8)

        self._save_button = QPushButton("Save")
        self._save_button.setProperty("secondary", True)
        self._save_button.clicked.connect(self._save_state)
        layout.addWidget(self._save_button)

        self._load_button = QPushButton("Load")
        self._load_button.setProperty("secondary", True)
        self._load_button.clicked.connect(self._load_state)
        layout.addWidget(self._load_button)

        self._reset_button = QPushButton("Reset")
        self._reset_button.setProperty("secondary", True)
        self._reset_button.clicked.connect(self._reset_state)
        layout.addWidget(self._reset_button)

        layout.addStretch(1)

        self._auto_apply_checkbox = QCheckBox("Apply Automatically")
        self._auto_apply_checkbox.setChecked(True)
        layout.addWidget(self._auto_apply_checkbox)

        self._apply_button = QPushButton("Apply")
        self._apply_button.setProperty("primary", True)
        self._apply_button.clicked.connect(self._emit_dataset)
        layout.addWidget(self._apply_button)
        return layout

    def set_color_settings_service(self, service: ColorSettingsService) -> None:
        self._service = service

    def on_apply_requested(self, callback: Callable[[DatasetHandle], None]) -> None:
        self._callbacks.append(callback)

    def set_dataset(self, dataset: DatasetHandle | None) -> None:
        self._dataset = dataset
        self._output_dataset = None
        self._dataset_label.setText(f"Dataset: {dataset.display_name if dataset is not None else 'none'}")
        self._state = self._service.build_state(dataset)
        self._render_state()

    def help_text(self) -> str:
        return (
            "Assign colors to discrete values and choose gradient palettes for numeric variables, "
            "then apply the color metadata back to the current dataset."
        )

    def documentation_url(self) -> str:
        return "https://orangedatamining.com/widget-catalog/data/color/"

    def footer_status_text(self) -> str:
        discrete_count = len(self._state.get("discrete", {})) if isinstance(self._state.get("discrete"), dict) else 0
        numeric_count = len(self._state.get("numeric", {})) if isinstance(self._state.get("numeric"), dict) else 0
        return str(discrete_count + numeric_count)

    def set_input_payload(self, payload) -> None:
        dataset = payload.dataset if payload is not None else None
        self.set_dataset(dataset)
        if dataset is not None and self._auto_apply_checkbox.isChecked():
            self._emit_dataset()

    def current_output_dataset(self) -> DatasetHandle | None:
        return self._output_dataset

    def serialize_node_state(self) -> dict[str, object]:
        return {
            "state": json.loads(json.dumps(self._state)),
            "auto_apply": self._auto_apply_checkbox.isChecked(),
            "committed": self._output_dataset is not None,
        }

    def restore_node_state(self, payload: dict[str, object]) -> None:
        self._auto_apply_checkbox.setChecked(bool(payload.get("auto_apply", True)))
        state = payload.get("state")
        if isinstance(state, dict):
            self._state = state
            self._render_state()
        if bool(payload.get("committed")) and self._dataset is not None:
            self._emit_dataset()

    def report_snapshot(self) -> dict[str, object]:
        discrete = self._state.get("discrete", {})
        numeric = self._state.get("numeric", {})
        discrete_count = len(discrete) if isinstance(discrete, dict) else 0
        numeric_count = len(numeric) if isinstance(numeric, dict) else 0
        return {
            "title": "Color",
            "items": [
                {
                    "title": "Assignments",
                    "timestamp": "Current session",
                    "details": [
                        f"Discrete variables: {discrete_count}",
                        f"Numeric gradients: {numeric_count}",
                    ],
                }
            ],
        }

    def _render_state(self) -> None:
        self._clear_layout(self._discrete_layout)
        self._clear_layout(self._numeric_layout)
        self._discrete_buttons.clear()
        self._numeric_previews.clear()
        self._numeric_combos.clear()

        discrete = self._state.get("discrete", {})
        numeric = self._state.get("numeric", {})

        if not isinstance(discrete, dict) or not discrete:
            placeholder = QLabel("No discrete variables available for manual color assignment.")
            placeholder.setProperty("muted", True)
            self._discrete_layout.addWidget(placeholder)
        else:
            for column_name, value_map in discrete.items():
                if not isinstance(value_map, dict):
                    continue
                row = QFrame(self)
                row_layout = QHBoxLayout(row)
                row_layout.setContentsMargins(0, 0, 0, 0)
                row_layout.setSpacing(8)
                label = QLabel(column_name)
                label.setMinimumWidth(150)
                row_layout.addWidget(label)
                values_wrap = QWidget(self)
                values_layout = QHBoxLayout(values_wrap)
                values_layout.setContentsMargins(0, 0, 0, 0)
                values_layout.setSpacing(6)
                for value, color_hex in value_map.items():
                    button = ColorValueButton(value, str(color_hex), self)
                    button.clicked.connect(lambda _checked=False, c=column_name, v=value: self._pick_discrete_color(c, v))
                    values_layout.addWidget(button)
                    self._discrete_buttons[(column_name, value)] = button
                values_layout.addStretch(1)
                row_layout.addWidget(values_wrap, 1)
                self._discrete_layout.addWidget(row)
            self._discrete_layout.addStretch(1)

        if not isinstance(numeric, dict) or not numeric:
            placeholder = QLabel("No numeric variables available for gradient palettes.")
            placeholder.setProperty("muted", True)
            self._numeric_layout.addWidget(placeholder)
        else:
            for column_name, palette_name in numeric.items():
                row = QFrame(self)
                row_layout = QHBoxLayout(row)
                row_layout.setContentsMargins(0, 0, 0, 0)
                row_layout.setSpacing(8)
                label = QLabel(column_name)
                label.setMinimumWidth(150)
                row_layout.addWidget(label)

                preview = GradientPreview(self)
                preview.setMinimumWidth(180)
                preview.set_colors(self._service.GRADIENTS.get(str(palette_name), self._service.GRADIENTS["Citrus"]))
                row_layout.addWidget(preview, 1)
                self._numeric_previews[column_name] = preview

                combo = QComboBox(self)
                combo.addItems(list(self._service.GRADIENTS.keys()))
                combo.setCurrentText(str(palette_name))
                combo.currentTextChanged.connect(lambda value, column=column_name: self._set_numeric_palette(column, value))
                row_layout.addWidget(combo)
                self._numeric_combos[column_name] = combo
                self._numeric_layout.addWidget(row)
            self._numeric_layout.addStretch(1)

        enabled = self._dataset is not None
        self._save_button.setEnabled(enabled)
        self._load_button.setEnabled(enabled)
        self._reset_button.setEnabled(enabled)
        self._apply_button.setEnabled(enabled)

    def _pick_discrete_color(self, column_name: str, value: str) -> None:
        discrete = self._state.get("discrete", {})
        if not isinstance(discrete, dict):
            return
        current = discrete.get(column_name, {})
        if not isinstance(current, dict):
            return
        selected = QColorDialog.getColor(QColor(str(current.get(value, "#4db7eb"))), self, "Select Color")
        if not selected.isValid():
            return
        current[value] = selected.name()
        button = self._discrete_buttons.get((column_name, value))
        if button is not None:
            button.set_color(selected.name())
        self._auto_emit_if_needed()

    def _set_numeric_palette(self, column_name: str, palette_name: str) -> None:
        numeric = self._state.get("numeric", {})
        if not isinstance(numeric, dict):
            return
        numeric[column_name] = palette_name
        preview = self._numeric_previews.get(column_name)
        if preview is not None:
            preview.set_colors(self._service.GRADIENTS[palette_name])
        self._auto_emit_if_needed()

    def _auto_emit_if_needed(self) -> None:
        if self._auto_apply_checkbox.isChecked():
            self._emit_dataset()

    def _emit_dataset(self) -> None:
        if self._dataset is None:
            return
        colored = self._service.apply(self._dataset, self._state)
        self._dataset = colored
        self._output_dataset = colored
        for callback in self._callbacks:
            callback(colored)
        self._notify_output_changed()

    def _reset_state(self) -> None:
        base_dataset = replace(self._dataset, annotations={}) if self._dataset is not None else None
        self._state = self._service.build_state(base_dataset)
        self._render_state()

    def _save_state(self) -> None:
        if self._dataset is None:
            return
        path, _selected_filter = QFileDialog.getSaveFileName(
            self,
            "Save Color Settings",
            "color-settings.json",
            "JSON Files (*.json);;All Files (*.*)",
        )
        if not path:
            return
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(self._state, handle, indent=2)

    def _load_state(self) -> None:
        if self._dataset is None:
            return
        path, _selected_filter = QFileDialog.getOpenFileName(
            self,
            "Load Color Settings",
            "",
            "JSON Files (*.json);;All Files (*.*)",
        )
        if not path:
            return
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if isinstance(payload, dict):
            self._state = payload
            self._render_state()
            self._auto_emit_if_needed()

    def _clear_layout(self, layout: QVBoxLayout) -> None:
        while layout.count():
            item = layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
