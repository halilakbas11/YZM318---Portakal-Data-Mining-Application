from __future__ import annotations

from itertools import chain

from PySide6.QtCore import Qt, QRectF, QLineF
from PySide6.QtGui import QPainter, QColor, QFont, QFontMetrics, QPen
from PySide6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QRadioButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from portakal_app.data.services.logistic_regression_service import (
    LogisticRegressionService,
    LogisticRegressionSettings,
)
from portakal_app.models import WorkflowPayload
from portakal_app.ui import i18n
from portakal_app.ui.screens.model_base import ModelScreenBase


_C_VALUES = list(
    chain(
        range(1000, 200, -50),
        range(200, 100, -10),
        range(100, 20, -5),
        range(20, 0, -1),
        [x / 10 for x in range(9, 2, -1)],
        [x / 100 for x in range(20, 2, -1)],
        [x / 1000 for x in range(20, 0, -1)],
    )
)
_PENALTIES = ("Lasso (L1)", "Ridge (L2)", "None")
_PENALTY_SHORT = ["l1", "l2", None]


class CoefficientPlot(QWidget):
    """Bar chart for Logistic Regression coefficients."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setMinimumHeight(300)
        self._coefs: list[tuple[str, float]] = []

    def set_data(self, coefs: list[tuple[str, float]]) -> None:
        self._coefs = sorted(coefs, key=lambda x: abs(x[1]), reverse=True)[:20]
        self.update()

    def paintEvent(self, event) -> None:
        if not self._coefs:
            return

        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        w, h = self.width(), self.height()
        margin_x = 100
        margin_y = 30
        
        center_x = w / 2
        plot_w = w - margin_x * 2
        plot_h = h - margin_y * 2
        
        bar_spacing = plot_h / len(self._coefs)
        bar_h = min(20, bar_spacing - 4)
        
        max_val = max(abs(x[1]) for x in self._coefs) if self._coefs else 1.0
        
        painter.setFont(QFont("Inter", 9))
        
        # Draw center line
        painter.setPen(QPen(QColor("#94a3b8"), 1, Qt.PenStyle.DashLine))
        painter.drawLine(QLineF(center_x, margin_y, center_x, h - margin_y))

        for i, (name, val) in enumerate(self._coefs):
            y = margin_y + i * bar_spacing
            
            bar_w = (abs(val) / max_val) * (plot_w / 2)
            
            color = QColor("#3b82f6") if val > 0 else QColor("#ef4444")
            
            if val > 0:
                rect = QRectF(center_x, y, bar_w, bar_h)
                painter.setBrush(color)
                painter.setPen(Qt.PenStyle.NoPen)
                painter.drawRoundedRect(rect, 3, 3)
                
                # Label on left
                painter.setPen(QColor("#475569"))
                painter.drawText(QRectF(0, y, center_x - 10, bar_h), Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter, name)
            else:
                rect = QRectF(center_x - bar_w, y, bar_w, bar_h)
                painter.setBrush(color)
                painter.setPen(Qt.PenStyle.NoPen)
                painter.drawRoundedRect(rect, 3, 3)
                
                # Label on right
                painter.setPen(QColor("#475569"))
                painter.drawText(QRectF(center_x + 10, y, center_x - 20, bar_h), Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, name)


class LogisticRegressionScreen(ModelScreenBase):
    """Logistic Regression with L1/L2/no regularisation — Orange equivalent."""

    _OUTPUT_PORT_LABEL = "Classifier"

    def __init__(self, parent=None) -> None:
        self._svc = LogisticRegressionService()
        self._c_index = 61
        super().__init__(parent)

    def _add_main_layout(self, layout: QVBoxLayout) -> None:
        upper_layout = QHBoxLayout()
        
        # Settings Column
        settings_layout = QVBoxLayout()
        
        reg_box = QGroupBox(i18n.t("Regularization"))
        reg_layout = QVBoxLayout(reg_box)

        self._penalty_group = QButtonGroup(self)
        for i, label in enumerate(_PENALTIES):
            rb = QRadioButton(i18n.t(label))
            if i == 1:
                rb.setChecked(True)
            self._penalty_group.addButton(rb, i)
            reg_layout.addWidget(rb)
        self._penalty_group.idToggled.connect(self._on_penalty_changed)

        strength_layout = QVBoxLayout()
        strength_layout.setContentsMargins(20, 4, 0, 0)
        lbl = QLabel(i18n.t("Strength:"))
        strength_layout.addWidget(lbl)

        slider_row = QHBoxLayout()
        slider_row.addWidget(QLabel(i18n.t("Weak")))
        self._c_slider = QSlider(Qt.Orientation.Horizontal)
        self._c_slider.setRange(0, len(_C_VALUES) - 1)
        self._c_slider.setValue(self._c_index)
        self._c_slider.valueChanged.connect(self._on_c_changed)
        self._c_slider.sliderReleased.connect(self._settings_changed)
        slider_row.addWidget(self._c_slider, 1)
        slider_row.addWidget(QLabel(i18n.t("Strong")))
        strength_layout.addLayout(slider_row)

        self._c_value_label = QLabel()
        self._c_value_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        strength_layout.addWidget(self._c_value_label)
        self._update_c_label()
        reg_layout.addLayout(strength_layout)
        settings_layout.addWidget(reg_box)

        weight_box = QGroupBox()
        weight_layout = QVBoxLayout(weight_box)
        self._balance_cb = QCheckBox(i18n.t("Balance class distribution"))
        self._balance_cb.stateChanged.connect(self._settings_changed)
        weight_layout.addWidget(self._balance_cb)
        settings_layout.addWidget(weight_box)
        
        # Summary Box
        summary_box = QGroupBox(i18n.t("Statistical Summary"))
        summary_layout = QVBoxLayout(summary_box)
        self._summary_label = QLabel(i18n.t("No model trained yet."))
        self._summary_label.setStyleSheet("color: #64748b; font-size: 11px;")
        self._summary_label.setWordWrap(True)
        summary_layout.addWidget(self._summary_label)
        settings_layout.addWidget(summary_box)
        
        upper_layout.addLayout(settings_layout, 1)
        
        # Plot Column
        plot_box = QGroupBox(i18n.t("Coefficient Plot"))
        plot_layout = QVBoxLayout(plot_box)
        self._coef_plot = CoefficientPlot()
        plot_layout.addWidget(self._coef_plot)
        upper_layout.addWidget(plot_box, 2)
        
        layout.addLayout(upper_layout)

    def _on_c_changed(self, value: int) -> None:
        self._c_index = value
        self._update_c_label()

    def _update_c_label(self) -> None:
        penalty_idx = self._penalty_group.checkedId()
        if penalty_idx == 2:
            self._c_value_label.setText("N/A")
            self._c_slider.setEnabled(False)
        else:
            self._c_slider.setEnabled(True)
            c = _C_VALUES[self._c_index]
            fmt = "C={}" if c >= 1 else "C={:.3f}"
            self._c_value_label.setText(fmt.format(c))

    def _on_penalty_changed(self, _id: int, checked: bool) -> None:
        if checked:
            self._update_c_label()
            self._settings_changed()

    def _train(self):
        c = _C_VALUES[self._c_index]
        penalty_idx = self._penalty_group.checkedId()
        # Map penalty to ridge parameter: L2=1/C, no-penalty=very small ridge
        if penalty_idx == 2:
            ridge = 1e-8
        else:
            ridge = 1.0 / max(float(c), 1e-10)
        settings = LogisticRegressionSettings(
            max_iter=200,
            ridge=ridge,
            tolerance=1e-6,
        )
        result = self._svc.fit(self._dataset, settings)
        
        # Update Plot
        coef_list = []
        for feat in result.features:
            if hasattr(feat, "coefficients"):
                coef_list.append((feat.name, feat.coefficients[1]))
            elif hasattr(feat, "discrete_points"):
                # For discrete, use the max diff or just show first non-zero point
                val = feat.discrete_points[1][1] if len(feat.discrete_points[1]) > 1 else 0
                coef_list.append((feat.name, val))
        self._coef_plot.set_data(coef_list)
        
        # Update Summary
        penalty_text = _PENALTIES[penalty_idx]
        summary = (
            f"<b>{i18n.t('Target Class:')}</b> {result.class_values[1]}<br>"
            f"<b>{i18n.t('Penalty:')}</b> {penalty_text}<br>"
            f"<b>{i18n.t('Intercept:')}</b> {result.intercepts[1]:.4f}<br>"
        )
        self._summary_label.setText(summary)
        
        return result

    def current_output_payload(self) -> WorkflowPayload | None:
        if self._model_artifact is None:
            return None
        return WorkflowPayload("Classifier", self._model_artifact)

    def serialize_node_state(self) -> dict:
        return {
            **super().serialize_node_state(),
            "penalty": self._penalty_group.checkedId(),
            "c_index": self._c_slider.value(),
            "balance": self._balance_cb.isChecked(),
        }

    def restore_node_state(self, payload: dict) -> None:
        super().restore_node_state(payload)
        p_id = int(payload.get("penalty", 1))
        btn = self._penalty_group.button(p_id)
        if btn:
            btn.setChecked(True)
        self._c_slider.setValue(int(payload.get("c_index", 61)))
        self._balance_cb.setChecked(bool(payload.get("balance", False)))
        self._update_c_label()
