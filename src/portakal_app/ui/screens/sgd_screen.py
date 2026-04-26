from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QSpinBox,
    QVBoxLayout,
)

from portakal_app.data.services.sklearn_learner_service import SklearnLearnerService
from portakal_app.ui.screens.model_base import ModelScreenBase

_CLS_LOSSES = [
    ("Hinge", "hinge"),
    ("Logistic regression", "log_loss"),
    ("Modified Huber", "modified_huber"),
    ("Squared Hinge", "squared_hinge"),
    ("Perceptron", "perceptron"),
    ("Squared Loss", "squared_error"),
    ("Huber", "huber"),
    ("ε insensitive", "epsilon_insensitive"),
    ("Squared ε insensitive", "squared_epsilon_insensitive"),
]
_REG_LOSSES = [
    ("Squared Loss", "squared_error"),
    ("Huber", "huber"),
    ("ε insensitive", "epsilon_insensitive"),
    ("Squared ε insensitive", "squared_epsilon_insensitive"),
]
_PENALTIES = [("None", None), ("Lasso (L1)", "l1"), ("Ridge (L2)", "l2"), ("Elastic Net", "elasticnet")]
_LEARNING_RATES = [("Constant", "constant"), ("Optimal", "optimal"), ("Inverse scaling", "invscaling")]
_EPSILON_LOSSES = {"huber", "epsilon_insensitive", "squared_epsilon_insensitive"}
MAXINT = 2 ** 31 - 1


class SGDScreen(ModelScreenBase):
    """Stochastic Gradient Descent — minimize objective with SGD approximation."""

    _OUTPUT_PORT_LABEL = "Model"

    def __init__(self, parent=None) -> None:
        self._svc = SklearnLearnerService()
        super().__init__(parent)

    def _add_main_layout(self, layout: QVBoxLayout) -> None:
        loss_box = QGroupBox("Loss Functions")
        loss_form = QFormLayout(loss_box)
        loss_form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)

        self._cls_loss_combo = QComboBox()
        self._cls_loss_combo.addItems([l[0] for l in _CLS_LOSSES])
        self._cls_loss_combo.currentIndexChanged.connect(self._on_cls_loss_change)
        loss_form.addRow("Classification:", self._cls_loss_combo)

        self._cls_eps_spin = self._make_double_spin(0.0, 1.0, 0.1, 0.01, 2)
        cls_eps_row = QHBoxLayout()
        cls_eps_row.addSpacing(16)
        cls_eps_row.addWidget(QLabel("ε:"))
        cls_eps_row.addWidget(self._cls_eps_spin)
        cls_eps_row.addStretch(1)
        w = QLabel()
        w.setLayout(cls_eps_row)
        loss_form.addRow(w)

        self._reg_loss_combo = QComboBox()
        self._reg_loss_combo.addItems([l[0] for l in _REG_LOSSES])
        self._reg_loss_combo.currentIndexChanged.connect(self._on_reg_loss_change)
        loss_form.addRow("Regression:", self._reg_loss_combo)

        self._reg_eps_spin = self._make_double_spin(0.0, 1.0, 0.1, 0.01, 2)
        reg_eps_row = QHBoxLayout()
        reg_eps_row.addSpacing(16)
        reg_eps_row.addWidget(QLabel("ε:"))
        reg_eps_row.addWidget(self._reg_eps_spin)
        reg_eps_row.addStretch(1)
        w2 = QLabel()
        w2.setLayout(reg_eps_row)
        loss_form.addRow(w2)

        layout.addWidget(loss_box)

        reg_box = QGroupBox("Regularization")
        reg_form = QFormLayout(reg_box)
        reg_form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)

        penalty_row = QHBoxLayout()
        self._penalty_combo = QComboBox()
        self._penalty_combo.addItems([p[0] for p in _PENALTIES])
        self._penalty_combo.setCurrentIndex(2)
        self._penalty_combo.currentIndexChanged.connect(self._on_regularization_change)
        penalty_row.addWidget(self._penalty_combo)
        self._l1_ratio_spin = self._make_double_spin(0.0, 1.0, 0.15, 0.01, 2)
        self._l1_ratio_label = QLabel("Mixing:")
        penalty_row.addWidget(self._l1_ratio_label)
        penalty_row.addWidget(self._l1_ratio_spin)
        w3 = QLabel()
        w3.setLayout(penalty_row)
        reg_form.addRow(w3)

        self._alpha_spin = self._make_double_spin(0.0, 10.0, 1e-5, 1e-4, 5)
        reg_form.addRow("Strength (α):", self._alpha_spin)

        layout.addWidget(reg_box)

        opt_box = QGroupBox("Optimization")
        opt_form = QFormLayout(opt_box)
        opt_form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)

        self._lr_combo = QComboBox()
        self._lr_combo.addItems([lr[0] for lr in _LEARNING_RATES])
        self._lr_combo.currentIndexChanged.connect(self._on_learning_rate_change)
        opt_form.addRow("Learning rate:", self._lr_combo)

        self._eta0_spin = self._make_double_spin(1e-4, 1.0, 0.01, 1e-4, 4)
        opt_form.addRow("Initial learning rate (η₀):", self._eta0_spin)

        self._power_t_spin = self._make_double_spin(0.0, 1.0, 0.25, 1e-4, 4)
        opt_form.addRow("Inverse scaling exponent (t):", self._power_t_spin)

        self._max_iter_spin = QSpinBox()
        self._max_iter_spin.setRange(1, MAXINT - 1)
        self._max_iter_spin.setValue(1000)
        self._max_iter_spin.setAlignment(Qt.AlignmentFlag.AlignRight)
        self._max_iter_spin.valueChanged.connect(self._settings_changed)
        opt_form.addRow("Number of iterations:", self._max_iter_spin)

        self._tol_cb = QCheckBox("Tolerance (stopping criterion):")
        self._tol_cb.setChecked(True)
        self._tol_cb.stateChanged.connect(self._settings_changed)
        self._tol_spin = self._make_double_spin(0.0, 10.0, 1e-3, 1e-4, 4)
        tol_row = QHBoxLayout()
        tol_row.addWidget(self._tol_cb)
        tol_row.addWidget(self._tol_spin)
        w4 = QLabel()
        w4.setLayout(tol_row)
        opt_form.addRow(w4)

        self._shuffle_cb = QCheckBox("Shuffle data after each iteration")
        self._shuffle_cb.setChecked(True)
        self._shuffle_cb.stateChanged.connect(self._on_shuffle_change)
        opt_form.addRow(self._shuffle_cb)

        self._use_seed_cb = QCheckBox("Fixed seed for random shuffling:")
        self._use_seed_cb.stateChanged.connect(self._settings_changed)
        self._seed_spin = QSpinBox()
        self._seed_spin.setRange(0, MAXINT)
        self._seed_spin.setValue(0)
        self._seed_spin.setAlignment(Qt.AlignmentFlag.AlignRight)
        self._seed_spin.valueChanged.connect(self._settings_changed)
        seed_row = QHBoxLayout()
        seed_row.addWidget(self._use_seed_cb)
        seed_row.addWidget(self._seed_spin)
        w5 = QLabel()
        w5.setLayout(seed_row)
        opt_form.addRow(w5)

        layout.addWidget(opt_box)

        self._on_cls_loss_change()
        self._on_reg_loss_change()
        self._on_regularization_change()
        self._on_learning_rate_change()

    def _make_double_spin(self, lo, hi, val, step, dec) -> QDoubleSpinBox:
        sp = QDoubleSpinBox()
        sp.setRange(lo, hi)
        sp.setValue(val)
        sp.setSingleStep(step)
        sp.setDecimals(dec)
        sp.setAlignment(Qt.AlignmentFlag.AlignRight)
        sp.valueChanged.connect(self._settings_changed)
        return sp

    def _on_cls_loss_change(self) -> None:
        loss = _CLS_LOSSES[self._cls_loss_combo.currentIndex()][1]
        self._cls_eps_spin.setEnabled(loss in _EPSILON_LOSSES)
        self._settings_changed()

    def _on_reg_loss_change(self) -> None:
        loss = _REG_LOSSES[self._reg_loss_combo.currentIndex()][1]
        self._reg_eps_spin.setEnabled(loss in _EPSILON_LOSSES)
        self._settings_changed()

    def _on_regularization_change(self) -> None:
        penalty = _PENALTIES[self._penalty_combo.currentIndex()][1]
        has_alpha = penalty in ("l1", "l2", "elasticnet")
        self._alpha_spin.setEnabled(has_alpha)
        is_elastic = penalty == "elasticnet"
        self._l1_ratio_spin.setVisible(is_elastic)
        self._l1_ratio_label.setVisible(is_elastic)
        self._settings_changed()

    def _on_learning_rate_change(self) -> None:
        lr = _LEARNING_RATES[self._lr_combo.currentIndex()][1]
        self._eta0_spin.setEnabled(lr in ("constant", "invscaling"))
        self._power_t_spin.setEnabled(lr == "invscaling")
        self._settings_changed()

    def _on_shuffle_change(self) -> None:
        if not self._shuffle_cb.isChecked():
            self._use_seed_cb.setChecked(False)
        self._use_seed_cb.setEnabled(self._shuffle_cb.isChecked())
        self._settings_changed()

    def _train(self):
        from sklearn.linear_model import SGDClassifier, SGDRegressor

        ds = self._dataset
        target_cols = ds.domain.target_columns
        if not target_cols:
            raise ValueError("No target column.")
        is_clf = target_cols[0].logical_type in {"categorical", "boolean"}

        cls_loss = _CLS_LOSSES[self._cls_loss_combo.currentIndex()][1]
        reg_loss = _REG_LOSSES[self._reg_loss_combo.currentIndex()][1]
        penalty = _PENALTIES[self._penalty_combo.currentIndex()][1]
        lr_name = _LEARNING_RATES[self._lr_combo.currentIndex()][1]
        alpha = self._alpha_spin.value()
        l1_ratio = self._l1_ratio_spin.value()
        eta0 = self._eta0_spin.value()
        power_t = self._power_t_spin.value()
        max_iter = self._max_iter_spin.value()
        tol = self._tol_spin.value() if self._tol_cb.isChecked() else None
        shuffle = self._shuffle_cb.isChecked()
        random_state = self._seed_spin.value() if self._use_seed_cb.isChecked() else None

        common = dict(penalty=penalty, alpha=alpha, l1_ratio=l1_ratio, shuffle=shuffle,
                      learning_rate=lr_name, eta0=eta0, power_t=power_t,
                      max_iter=max_iter, tol=tol, random_state=random_state)

        if is_clf:
            est = SGDClassifier(loss=cls_loss, epsilon=self._cls_eps_spin.value(), **common)
        else:
            est = SGDRegressor(loss=reg_loss, epsilon=self._reg_eps_spin.value(), **common)

        params = {"loss": cls_loss if is_clf else reg_loss, "penalty": penalty}
        return self._svc.fit(est, ds, "SGD", "sgd", params)

    def serialize_node_state(self) -> dict:
        return {
            **super().serialize_node_state(),
            "cls_loss": self._cls_loss_combo.currentIndex(),
            "cls_eps": self._cls_eps_spin.value(),
            "reg_loss": self._reg_loss_combo.currentIndex(),
            "reg_eps": self._reg_eps_spin.value(),
            "penalty": self._penalty_combo.currentIndex(),
            "alpha": self._alpha_spin.value(),
            "l1_ratio": self._l1_ratio_spin.value(),
            "lr": self._lr_combo.currentIndex(),
            "eta0": self._eta0_spin.value(),
            "power_t": self._power_t_spin.value(),
            "max_iter": self._max_iter_spin.value(),
            "tol_en": self._tol_cb.isChecked(),
            "tol": self._tol_spin.value(),
            "shuffle": self._shuffle_cb.isChecked(),
            "use_seed": self._use_seed_cb.isChecked(),
            "seed": self._seed_spin.value(),
        }

    def restore_node_state(self, payload: dict) -> None:
        super().restore_node_state(payload)
        self._cls_loss_combo.setCurrentIndex(int(payload.get("cls_loss", 0)))
        self._cls_eps_spin.setValue(float(payload.get("cls_eps", 0.1)))
        self._reg_loss_combo.setCurrentIndex(int(payload.get("reg_loss", 0)))
        self._reg_eps_spin.setValue(float(payload.get("reg_eps", 0.1)))
        self._penalty_combo.setCurrentIndex(int(payload.get("penalty", 2)))
        self._alpha_spin.setValue(float(payload.get("alpha", 1e-5)))
        self._l1_ratio_spin.setValue(float(payload.get("l1_ratio", 0.15)))
        self._lr_combo.setCurrentIndex(int(payload.get("lr", 0)))
        self._eta0_spin.setValue(float(payload.get("eta0", 0.01)))
        self._power_t_spin.setValue(float(payload.get("power_t", 0.25)))
        self._max_iter_spin.setValue(int(payload.get("max_iter", 1000)))
        self._tol_cb.setChecked(bool(payload.get("tol_en", True)))
        self._tol_spin.setValue(float(payload.get("tol", 1e-3)))
        self._shuffle_cb.setChecked(bool(payload.get("shuffle", True)))
        self._use_seed_cb.setChecked(bool(payload.get("use_seed", False)))
        self._seed_spin.setValue(int(payload.get("seed", 0)))
