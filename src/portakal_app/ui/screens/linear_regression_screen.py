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

from portakal_app.data.services.sklearn_learner_service import SklearnLearnerService
from portakal_app.ui import i18n
from portakal_app.ui.screens.model_base import ModelScreenBase


_ALPHAS = list(
    chain(
        [x / 10000 for x in range(1, 10)],
        [x / 1000 for x in range(1, 20)],
        [x / 100 for x in range(2, 20)],
        [x / 10 for x in range(2, 9)],
        range(1, 20),
        range(20, 100, 5),
        range(100, 1001, 100),
    )
)

_OLS, _RIDGE, _LASSO, _ELASTIC = 0, 1, 2, 3
_REG_TYPES = ["No regularization", "Ridge regression (L2)", "Lasso regression (L1)", "Elastic net regression"]


class RegressionWeightsPlot(QWidget):
    """Bar chart for Linear Regression weights."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setMinimumHeight(350)
        self._weights: list[tuple[str, float]] = []

    def set_data(self, weights: list[tuple[str, float]]) -> None:
        self._weights = sorted(weights, key=lambda x: abs(x[1]), reverse=True)[:25]
        self.update()

    def paintEvent(self, event) -> None:
        if not self._weights:
            return

        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        w, h = self.width(), self.height()
        margin_x = 100
        margin_y = 30
        
        center_x = w / 2
        plot_w = w - margin_x * 2
        plot_h = h - margin_y * 2
        
        bar_spacing = plot_h / len(self._weights)
        bar_h = min(18, bar_spacing - 4)
        
        max_val = max(abs(x[1]) for x in self._weights) if self._weights else 1.0
        
        painter.setFont(QFont("Inter", 8))
        
        # Center line
        painter.setPen(QPen(QColor("#94a3b8"), 1, Qt.PenStyle.DashLine))
        painter.drawLine(QLineF(center_x, margin_y, center_x, h - margin_y))

        for i, (name, val) in enumerate(self._weights):
            y = margin_y + i * bar_spacing
            
            bar_w = (abs(val) / max_val) * (plot_w / 2)
            
            # Use Teal for positive, Rose for negative in Regression
            color = QColor("#0d9488") if val > 0 else QColor("#e11d48")
            
            if val > 0:
                rect = QRectF(center_x, y, bar_w, bar_h)
                painter.setBrush(color)
                painter.setPen(Qt.PenStyle.NoPen)
                painter.drawRoundedRect(rect, 3, 3)
                
                painter.setPen(QColor("#475569"))
                painter.drawText(QRectF(0, y, center_x - 10, bar_h), Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter, name)
            else:
                rect = QRectF(center_x - bar_w, y, bar_w, bar_h)
                painter.setBrush(color)
                painter.setPen(Qt.PenStyle.NoPen)
                painter.drawRoundedRect(rect, 3, 3)
                
                painter.setPen(QColor("#475569"))
                painter.drawText(QRectF(center_x + 10, y, center_x - 20, bar_h), Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, name)


class LinearRegressionScreen(ModelScreenBase):
    """Linear Regression with optional L1/L2/Elastic regularisation."""

    _OUTPUT_PORT_LABEL = "Model"

    def __init__(self, parent=None) -> None:
        self._svc = SklearnLearnerService()
        super().__init__(parent)
        # Call these AFTER super().__init__ so that _apply_button is already created
        self._update_alpha_label()
        self._update_l2_label()
        self._on_reg_changed(0, True)

    def _add_main_layout(self, layout: QVBoxLayout) -> None:
        upper_layout = QHBoxLayout()
        
        # Left: Settings
        settings_layout = QVBoxLayout()
        
        params_box = QGroupBox(i18n.t("Parameters"))
        p_layout = QVBoxLayout(params_box)
        self._intercept_cb = QCheckBox(i18n.t("Fit intercept"))
        self._intercept_cb.setChecked(True)
        self._intercept_cb.stateChanged.connect(self._settings_changed)
        p_layout.addWidget(self._intercept_cb)
        settings_layout.addWidget(params_box)

        reg_box = QGroupBox(i18n.t("Regularization"))
        reg_layout = QVBoxLayout(reg_box)
        self._reg_group = QButtonGroup(self)
        for i, label in enumerate(_REG_TYPES):
            rb = QRadioButton(i18n.t(label))
            if i == 0:
                rb.setChecked(True)
            self._reg_group.addButton(rb, i)
            reg_layout.addWidget(rb)
        self._reg_group.idToggled.connect(self._on_reg_changed)

        alpha_box = QGroupBox(i18n.t("Regularization strength:"))
        alpha_layout = QVBoxLayout(alpha_box)
        self._alpha_slider = QSlider(Qt.Orientation.Horizontal)
        self._alpha_slider.setRange(0, len(_ALPHAS) - 1)
        self._alpha_slider.setValue(0)
        self._alpha_slider.sliderReleased.connect(self._settings_changed)
        self._alpha_slider.valueChanged.connect(self._update_alpha_label)
        alpha_layout.addWidget(self._alpha_slider)
        self._alpha_label = QLabel()
        self._alpha_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        alpha_layout.addWidget(self._alpha_label)
        reg_layout.addWidget(alpha_box)

        elastic_box = QGroupBox(i18n.t("Elastic net mixing (L1 : L2):"))
        el_layout = QHBoxLayout(elastic_box)
        el_layout.addWidget(QLabel("L1"))
        self._l2_slider = QSlider(Qt.Orientation.Horizontal)
        self._l2_slider.setRange(1, 99)
        self._l2_slider.setValue(50)
        self._l2_slider.sliderReleased.connect(self._settings_changed)
        self._l2_slider.valueChanged.connect(self._update_l2_label)
        el_layout.addWidget(self._l2_slider, 1)
        el_layout.addWidget(QLabel("L2"))
        reg_layout.addWidget(elastic_box)
        settings_layout.addWidget(reg_box)
        
        upper_layout.addLayout(settings_layout, 1)
        
        # Right: Plot
        plot_box = QGroupBox(i18n.t("Regression Weights"))
        plot_layout = QVBoxLayout(plot_box)
        self._weights_plot = RegressionWeightsPlot()
        plot_layout.addWidget(self._weights_plot)
        upper_layout.addWidget(plot_box, 2)
        
        layout.addLayout(upper_layout)

        self._elastic_box = elastic_box
        self._alpha_box = alpha_box

    def _on_reg_changed(self, _id: int, checked: bool) -> None:
        if not checked:
            return
        reg = self._reg_group.checkedId()
        self._alpha_box.setEnabled(reg != _OLS)
        self._elastic_box.setEnabled(reg == _ELASTIC)
        self._settings_changed()

    def _update_alpha_label(self) -> None:
        self._alpha_label.setText(f"Alpha: {_ALPHAS[self._alpha_slider.value()]}")

    def _update_l2_label(self) -> None:
        v = self._l2_slider.value() / 100.0
        pass

    def _train(self):
        from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet

        ds = self._dataset
        target_cols = ds.domain.target_columns
        if not target_cols:
            raise ValueError("No target column.")
        if target_cols[0].logical_type not in {"numeric"}:
            raise ValueError("Linear Regression requires a numeric target.")

        reg = self._reg_group.checkedId()
        alpha = _ALPHAS[self._alpha_slider.value()]
        fit_int = self._intercept_cb.isChecked()
        l1_ratio = 1.0 - self._l2_slider.value() / 100.0

        if reg == _OLS:
            est = LinearRegression(fit_intercept=fit_int)
        elif reg == _RIDGE:
            est = Ridge(alpha=alpha, fit_intercept=fit_int)
        elif reg == _LASSO:
            est = Lasso(alpha=alpha, fit_intercept=fit_int, max_iter=10000)
        else:
            est = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, fit_intercept=fit_int, max_iter=10000)

        params = {"reg_type": _REG_TYPES[reg], "alpha": alpha, "fit_intercept": fit_int}
        result = self._svc.fit(est, ds, "Linear Regression", "linear_regression", params)
        
        # Update Weights Plot
        if hasattr(result.trained_model, "coef_"):
            coefs = result.trained_model.coef_
            names = result.feature_names
            # Sklearn might return multi-dim coef_ if multiple targets, but we enforce one.
            if len(coefs.shape) > 1:
                coefs = coefs[0]
            
            weight_list = list(zip(names, coefs))
            self._weights_plot.set_data(weight_list)
            
        return result

    def serialize_node_state(self) -> dict:
        return {
            **super().serialize_node_state(),
            "fit_intercept": self._intercept_cb.isChecked(),
            "reg_type": self._reg_group.checkedId(),
            "alpha_index": self._alpha_slider.value(),
            "l2_ratio": self._l2_slider.value(),
        }

    def restore_node_state(self, payload: dict) -> None:
        super().restore_node_state(payload)
        self._intercept_cb.setChecked(bool(payload.get("fit_intercept", True)))
        r_id = int(payload.get("reg_type", 0))
        btn = self._reg_group.button(r_id)
        if btn:
            btn.setChecked(True)
        self._alpha_slider.setValue(int(payload.get("alpha_index", 0)))
        self._l2_slider.setValue(int(payload.get("l2_ratio", 50)))
        self._update_alpha_label()
