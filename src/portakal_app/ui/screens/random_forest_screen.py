from __future__ import annotations

from PySide6.QtCore import Qt, QRectF
from PySide6.QtGui import QPainter, QColor, QLinearGradient, QFont, QFontMetrics
from PySide6.QtWidgets import (
    QCheckBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
    QHeaderView,
)

from portakal_app.data.services.random_forest_service import RandomForestService, RandomForestSettings
from portakal_app.ui import i18n
from portakal_app.ui.screens.model_base import ModelScreenBase


class FeatureImportancePlot(QWidget):
    """Modern bar chart for Gini Feature Importance."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setMinimumHeight(250)
        self._importances: list[tuple[str, float]] = []

    def set_data(self, importances: list[tuple[str, float]]) -> None:
        # Sort by importance and take top 15
        self._importances = sorted(importances, key=lambda x: x[1], reverse=True)[:15]
        self.update()

    def paintEvent(self, event) -> None:
        if not self._importances:
            return

        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        w, h = self.width(), self.height()
        margin_left = 120
        margin_right = 40
        margin_top = 20
        margin_bottom = 20
        
        plot_w = w - margin_left - margin_right
        plot_h = h - margin_top - margin_bottom
        
        bar_spacing = plot_h / len(self._importances)
        bar_h = min(25, bar_spacing - 5)
        
        max_val = max(x[1] for x in self._importances) if self._importances else 1.0
        
        painter.setFont(QFont("Inter", 9))

        for i, (name, val) in enumerate(self._importances):
            y = margin_top + i * bar_spacing
            
            # Draw label
            label_rect = QRectF(10, y, margin_left - 20, bar_h)
            painter.setPen(QColor("#475569"))
            painter.drawText(label_rect, Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter, name)
            
            # Draw bar
            bar_w = (val / max_val) * plot_w
            bar_rect = QRectF(margin_left, y, bar_w, bar_h)
            
            grad = QLinearGradient(bar_rect.topLeft(), bar_rect.topRight())
            grad.setColorAt(0, QColor("#818cf8")) 
            grad.setColorAt(1, QColor("#4f46e5"))
            
            painter.setBrush(grad)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawRoundedRect(bar_rect, 4, 4)
            
            val_text = f"{val:.3f}"
            painter.setPen(QColor("#1e293b"))
            painter.drawText(margin_left + bar_w + 5, y + bar_h - 5, val_text)


class RandomForestScreen(ModelScreenBase):
    """Random Forest — ensemble of decision trees."""

    _OUTPUT_PORT_LABEL = "Random Forest"

    def __init__(self, parent=None) -> None:
        self._svc = RandomForestService()
        super().__init__(parent)

    def _add_main_layout(self, layout: QVBoxLayout) -> None:
        # Configuration columns
        config_layout = QHBoxLayout()
        
        # Left Side: Parameters
        params_layout = QVBoxLayout()
        
        basic = QGroupBox(i18n.t("Basic Properties"))
        form1 = QFormLayout(basic)
        form1.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)

        self._n_spin = QSpinBox()
        self._n_spin.setRange(1, 10000)
        self._n_spin.setValue(10)
        self._n_spin.setAlignment(Qt.AlignmentFlag.AlignRight)
        self._n_spin.valueChanged.connect(self._settings_changed)
        form1.addRow(i18n.t("Number of trees:"), self._n_spin)

        self._max_feat_cb = QCheckBox(i18n.t("Attributes at each split:"))
        self._max_feat_spin = QSpinBox()
        self._max_feat_spin.setRange(1, 500)
        self._max_feat_spin.setValue(5)
        self._max_feat_spin.setAlignment(Qt.AlignmentFlag.AlignRight)
        self._max_feat_cb.stateChanged.connect(self._settings_changed)
        self._max_feat_spin.valueChanged.connect(self._settings_changed)
        form1.addRow(self._max_feat_cb, self._max_feat_spin)

        self._seed_cb = QCheckBox(i18n.t("Replicable training"))
        self._seed_cb.stateChanged.connect(self._settings_changed)
        form1.addRow(self._seed_cb)

        self._balance_cb = QCheckBox(i18n.t("Balance class distribution"))
        self._balance_cb.stateChanged.connect(self._settings_changed)
        form1.addRow(self._balance_cb)

        params_layout.addWidget(basic)

        growth = QGroupBox(i18n.t("Growth Control"))
        form2 = QFormLayout(growth)
        form2.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)

        self._depth_cb = QCheckBox(i18n.t("Limit depth of trees:"))
        self._depth_spin = QSpinBox()
        self._depth_spin.setRange(1, 50)
        self._depth_spin.setValue(3)
        self._depth_spin.setAlignment(Qt.AlignmentFlag.AlignRight)
        self._depth_cb.stateChanged.connect(self._settings_changed)
        self._depth_spin.valueChanged.connect(self._settings_changed)
        form2.addRow(self._depth_cb, self._depth_spin)

        self._split_cb = QCheckBox(i18n.t("Do not split subsets smaller than:"))
        self._split_cb.setChecked(True)
        self._split_spin = QSpinBox()
        self._split_spin.setRange(2, 1000)
        self._split_spin.setValue(5)
        self._split_spin.setAlignment(Qt.AlignmentFlag.AlignRight)
        self._split_cb.stateChanged.connect(self._settings_changed)
        self._split_spin.valueChanged.connect(self._settings_changed)
        form2.addRow(self._split_cb, self._split_spin)

        params_layout.addWidget(growth)
        config_layout.addLayout(params_layout, 1)
        
        # Right Side: Importance Plot
        importance_box = QGroupBox(i18n.t("Feature Importance (Gini)"))
        importance_layout = QVBoxLayout(importance_box)
        self._importance_plot = FeatureImportancePlot()
        importance_layout.addWidget(self._importance_plot)
        config_layout.addWidget(importance_box, 2)
        
        layout.addLayout(config_layout)

        # Technical Details Table
        details_box = QGroupBox(i18n.t("Technical Details"))
        details_layout = QVBoxLayout(details_box)
        self._details_table = QTableWidget(0, 2)
        self._details_table.setHorizontalHeaderLabels([i18n.t("Property"), i18n.t("Value")])
        self._details_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self._details_table.setMaximumHeight(120)
        details_layout.addWidget(self._details_table)
        layout.addWidget(details_box)

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
        result = self._svc.fit(self._dataset, settings)
        
        # Update importance plot
        if hasattr(result.trained_model, "feature_importances_"):
            importances = list(zip(result.feature_names, result.trained_model.feature_importances_))
            self._importance_plot.set_data(importances)
            
        # Update details table
        self._details_table.setRowCount(0)
        details = [
            (i18n.t("Number of trees"), str(len(result.trees))),
            (i18n.t("Max Depth"), str(self._depth_spin.value() if self._depth_cb.isChecked() else "None")),
            (i18n.t("Forest ID"), result.forest_id[:8] + "..."),
        ]
        for prop, val in details:
            row = self._details_table.rowCount()
            self._details_table.insertRow(row)
            self._details_table.setItem(row, 0, QTableWidgetItem(prop))
            self._details_table.setItem(row, 1, QTableWidgetItem(val))
            
        return result

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
