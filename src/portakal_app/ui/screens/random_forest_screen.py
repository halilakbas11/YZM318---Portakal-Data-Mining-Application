from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QFormLayout,
    QGroupBox,
    QSpinBox,
    QVBoxLayout,
)

from portakal_app.data.services.random_forest_service import RandomForestService, RandomForestSettings
from portakal_app.ui.screens.model_base import ModelScreenBase


class RandomForestScreen(ModelScreenBase):
    """Random Forest — ensemble of decision trees."""

    _OUTPUT_PORT_LABEL = "Random Forest"

    def __init__(self, parent=None) -> None:
        self._svc = RandomForestService()
        super().__init__(parent)

    def _add_main_layout(self, layout: QVBoxLayout) -> None:
        basic = QGroupBox("Basic Properties")
        form1 = QFormLayout(basic)
        form1.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)

        self._n_spin = QSpinBox()
        self._n_spin.setRange(1, 10000)
        self._n_spin.setValue(10)
        self._n_spin.setAlignment(Qt.AlignmentFlag.AlignRight)
        self._n_spin.valueChanged.connect(self._settings_changed)
        form1.addRow("Number of trees:", self._n_spin)

        self._max_feat_cb = QCheckBox("Attributes at each split:")
        self._max_feat_spin = QSpinBox()
        self._max_feat_spin.setRange(1, 500)
        self._max_feat_spin.setValue(5)
        self._max_feat_spin.setAlignment(Qt.AlignmentFlag.AlignRight)
        self._max_feat_cb.stateChanged.connect(self._settings_changed)
        self._max_feat_spin.valueChanged.connect(self._settings_changed)
        form1.addRow(self._max_feat_cb, self._max_feat_spin)

        self._seed_cb = QCheckBox("Replicable training")
        self._seed_cb.stateChanged.connect(self._settings_changed)
        form1.addRow(self._seed_cb)

        self._balance_cb = QCheckBox("Balance class distribution")
        self._balance_cb.stateChanged.connect(self._settings_changed)
        form1.addRow(self._balance_cb)

        layout.addWidget(basic)

        growth = QGroupBox("Growth Control")
        form2 = QFormLayout(growth)
        form2.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)

        self._depth_cb = QCheckBox("Limit depth of trees:")
        self._depth_spin = QSpinBox()
        self._depth_spin.setRange(1, 50)
        self._depth_spin.setValue(3)
        self._depth_spin.setAlignment(Qt.AlignmentFlag.AlignRight)
        self._depth_cb.stateChanged.connect(self._settings_changed)
        self._depth_spin.valueChanged.connect(self._settings_changed)
        form2.addRow(self._depth_cb, self._depth_spin)

        self._split_cb = QCheckBox("Do not split subsets smaller than:")
        self._split_cb.setChecked(True)
        self._split_spin = QSpinBox()
        self._split_spin.setRange(2, 1000)
        self._split_spin.setValue(5)
        self._split_spin.setAlignment(Qt.AlignmentFlag.AlignRight)
        self._split_cb.stateChanged.connect(self._settings_changed)
        self._split_spin.valueChanged.connect(self._settings_changed)
        form2.addRow(self._split_cb, self._split_spin)

        layout.addWidget(growth)

    def _train(self):
        settings = RandomForestSettings(
            n_estimators=self._n_spin.value(),
            use_max_features=self._max_feat_cb.isChecked(),
            max_features=self._max_feat_spin.value(),
            use_random_state=self._seed_cb.isChecked(),
            use_max_depth=self._depth_cb.isChecked(),
            max_depth=self._depth_spin.value(),
            use_min_samples_split=self._split_cb.isChecked(),
            min_samples_split=self._split_spin.value(),
            class_weight=self._balance_cb.isChecked(),
        )
        return self._svc.fit(self._dataset, settings)

    def serialize_node_state(self) -> dict:
        return {
            **super().serialize_node_state(),
            "n_estimators": self._n_spin.value(),
            "max_feat_en": self._max_feat_cb.isChecked(),
            "max_features": self._max_feat_spin.value(),
            "seed": self._seed_cb.isChecked(),
            "balance": self._balance_cb.isChecked(),
            "depth_en": self._depth_cb.isChecked(),
            "max_depth": self._depth_spin.value(),
            "split_en": self._split_cb.isChecked(),
            "min_split": self._split_spin.value(),
        }

    def restore_node_state(self, payload: dict) -> None:
        super().restore_node_state(payload)
        self._n_spin.setValue(int(payload.get("n_estimators", 10)))
        self._max_feat_cb.setChecked(bool(payload.get("max_feat_en", False)))
        self._max_feat_spin.setValue(int(payload.get("max_features", 5)))
        self._seed_cb.setChecked(bool(payload.get("seed", False)))
        self._balance_cb.setChecked(bool(payload.get("balance", False)))
        self._depth_cb.setChecked(bool(payload.get("depth_en", False)))
        self._depth_spin.setValue(int(payload.get("max_depth", 3)))
        self._split_cb.setChecked(bool(payload.get("split_en", True)))
        self._split_spin.setValue(int(payload.get("min_split", 5)))
