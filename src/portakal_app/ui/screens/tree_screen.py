from __future__ import annotations

from PySide6.QtWidgets import (
    QCheckBox,
    QFormLayout,
    QGroupBox,
    QSpinBox,
    QVBoxLayout,
)

from portakal_app.data.services.tree_service import TreeService, TreeSettings
from portakal_app.ui.screens.model_base import ModelScreenBase


class TreeScreen(ModelScreenBase):
    """Decision Tree with forward pruning — Orange Tree equivalent."""

    _OUTPUT_PORT_LABEL = "Tree"

    def __init__(self, parent=None) -> None:
        self._svc = TreeService()
        super().__init__(parent)

    def _add_main_layout(self, layout: QVBoxLayout) -> None:
        box = QGroupBox("Parameters")
        form = QFormLayout(box)
        form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)

        self._binary_cb = QCheckBox("Induce binary tree")
        self._binary_cb.setChecked(True)
        self._binary_cb.stateChanged.connect(self._settings_changed)
        form.addRow(self._binary_cb)

        self._min_leaf_cb = QCheckBox("Min. instances in leaves:")
        self._min_leaf_cb.setChecked(True)
        self._min_leaf_spin = QSpinBox()
        self._min_leaf_spin.setRange(1, 1000)
        self._min_leaf_spin.setValue(2)
        self._min_leaf_cb.stateChanged.connect(self._settings_changed)
        self._min_leaf_spin.valueChanged.connect(self._settings_changed)
        form.addRow(self._min_leaf_cb, self._min_leaf_spin)

        self._min_int_cb = QCheckBox("Do not split subsets smaller than:")
        self._min_int_cb.setChecked(True)
        self._min_int_spin = QSpinBox()
        self._min_int_spin.setRange(1, 1000)
        self._min_int_spin.setValue(5)
        self._min_int_cb.stateChanged.connect(self._settings_changed)
        self._min_int_spin.valueChanged.connect(self._settings_changed)
        form.addRow(self._min_int_cb, self._min_int_spin)

        self._depth_cb = QCheckBox("Limit depth to:")
        self._depth_cb.setChecked(True)
        self._depth_spin = QSpinBox()
        self._depth_spin.setRange(1, 1000)
        self._depth_spin.setValue(100)
        self._depth_cb.stateChanged.connect(self._settings_changed)
        self._depth_spin.valueChanged.connect(self._settings_changed)
        form.addRow(self._depth_cb, self._depth_spin)

        self._majority_cb = QCheckBox("Stop when majority reaches [%]:")
        self._majority_cb.setChecked(True)
        self._majority_spin = QSpinBox()
        self._majority_spin.setRange(51, 100)
        self._majority_spin.setValue(95)
        self._majority_cb.stateChanged.connect(self._settings_changed)
        self._majority_spin.valueChanged.connect(self._settings_changed)
        form.addRow(self._majority_cb, self._majority_spin)

        layout.addWidget(box)

    def _train(self):
        settings = TreeSettings(
            binary_trees=self._binary_cb.isChecked(),
            limit_min_leaf=self._min_leaf_cb.isChecked(),
            min_leaf=self._min_leaf_spin.value(),
            limit_min_internal=self._min_int_cb.isChecked(),
            min_internal=self._min_int_spin.value(),
            limit_depth=self._depth_cb.isChecked(),
            max_depth=self._depth_spin.value(),
            limit_majority=self._majority_cb.isChecked(),
            sufficient_majority=self._majority_spin.value() / 100.0,
        )
        return self._svc.fit(self._dataset, settings)

    def serialize_node_state(self) -> dict:
        return {
            **super().serialize_node_state(),
            "binary": self._binary_cb.isChecked(),
            "min_leaf_en": self._min_leaf_cb.isChecked(),
            "min_leaf": self._min_leaf_spin.value(),
            "min_int_en": self._min_int_cb.isChecked(),
            "min_int": self._min_int_spin.value(),
            "depth_en": self._depth_cb.isChecked(),
            "max_depth": self._depth_spin.value(),
            "majority_en": self._majority_cb.isChecked(),
            "majority": self._majority_spin.value(),
        }

    def restore_node_state(self, payload: dict) -> None:
        super().restore_node_state(payload)
        self._binary_cb.setChecked(bool(payload.get("binary", True)))
        self._min_leaf_cb.setChecked(bool(payload.get("min_leaf_en", True)))
        self._min_leaf_spin.setValue(int(payload.get("min_leaf", 2)))
        self._min_int_cb.setChecked(bool(payload.get("min_int_en", True)))
        self._min_int_spin.setValue(int(payload.get("min_int", 5)))
        self._depth_cb.setChecked(bool(payload.get("depth_en", True)))
        self._depth_spin.setValue(int(payload.get("max_depth", 100)))
        self._majority_cb.setChecked(bool(payload.get("majority_en", True)))
        self._majority_spin.setValue(int(payload.get("majority", 95)))
