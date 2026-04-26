from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QRadioButton,
    QSpinBox,
    QVBoxLayout,
)

from portakal_app.data.services.sklearn_learner_service import SklearnLearnerService
from portakal_app.ui.screens.model_base import ModelScreenBase

_SVM, _NU_SVM = 0, 1
_KERNELS = ["linear", "poly", "rbf", "sigmoid"]
_KERNEL_LABELS = ["Linear", "Polynomial", "RBF", "Sigmoid"]


class SVMScreen(ModelScreenBase):
    """Support Vector Machine — C-SVM and ν-SVM with kernel selection."""

    _OUTPUT_PORT_LABEL = "Model"

    def __init__(self, parent=None) -> None:
        self._svc = SklearnLearnerService()
        super().__init__(parent)

    def _add_main_layout(self, layout: QVBoxLayout) -> None:
        type_box = QGroupBox("SVM Type")
        type_form = QFormLayout(type_box)
        self._type_group = QButtonGroup(self)

        svm_rb = QRadioButton("SVM")
        svm_rb.setChecked(True)
        self._type_group.addButton(svm_rb, _SVM)
        nu_rb = QRadioButton("ν-SVM")
        self._type_group.addButton(nu_rb, _NU_SVM)

        type_form.addRow(svm_rb)
        self._c_spin = self._make_double_spin(0.1, 512.0, 1.0, 0.1, 2)
        self._epsilon_spin = self._make_double_spin(0.01, 512.0, 0.1, 0.1, 2)
        type_form.addRow(self._make_spin_row("Cost (C):", self._c_spin))
        type_form.addRow(self._make_spin_row("Regression loss epsilon (ε):", self._epsilon_spin))

        type_form.addRow(nu_rb)
        self._nu_c_spin = self._make_double_spin(0.1, 512.0, 1.0, 0.1, 2)
        self._nu_spin = self._make_double_spin(0.05, 1.0, 0.5, 0.05, 2)
        type_form.addRow(self._make_spin_row("Regression cost (C):", self._nu_c_spin))
        type_form.addRow(self._make_spin_row("Complexity bound (ν):", self._nu_spin))

        self._type_group.idToggled.connect(self._on_type_changed)
        layout.addWidget(type_box)

        kernel_box = QGroupBox("Kernel")
        kernel_layout = QVBoxLayout(kernel_box)
        self._kernel_group = QButtonGroup(self)
        for i, label in enumerate(_KERNEL_LABELS):
            rb = QRadioButton(label)
            if i == 2:
                rb.setChecked(True)
            self._kernel_group.addButton(rb, i)
            kernel_layout.addWidget(rb)
        self._kernel_group.idToggled.connect(self._on_kernel_changed)

        kp_box = QGroupBox()
        kp_form = QFormLayout(kp_box)
        self._gamma_spin = self._make_double_spin(0.0, 10.0, 0.0, 0.01, 2)
        self._gamma_spin.setSpecialValueText("auto")
        self._coef0_spin = self._make_double_spin(0.0, 10.0, 1.0, 0.01, 2)
        self._degree_spin = QSpinBox()
        self._degree_spin.setRange(0, 10)
        self._degree_spin.setValue(3)
        self._degree_spin.setAlignment(Qt.AlignmentFlag.AlignRight)
        self._degree_spin.valueChanged.connect(self._settings_changed)
        kp_form.addRow("g:", self._gamma_spin)
        kp_form.addRow("c:", self._coef0_spin)
        kp_form.addRow("d:", self._degree_spin)
        kernel_layout.addWidget(kp_box)
        self._kp_box = kp_box
        layout.addWidget(kernel_box)

        opt_box = QGroupBox("Optimization Parameters")
        opt_form = QFormLayout(opt_box)
        self._tol_spin = self._make_double_spin(1e-4, 1.0, 0.001, 1e-4, 4)
        opt_form.addRow("Numerical tolerance:", self._tol_spin)
        iter_row = QHBoxLayout()
        self._limit_iter_cb = QCheckBox("Iteration limit:")
        self._limit_iter_cb.setChecked(True)
        self._limit_iter_cb.stateChanged.connect(self._settings_changed)
        self._max_iter_spin = QSpinBox()
        self._max_iter_spin.setRange(5, 1000000)
        self._max_iter_spin.setValue(100)
        self._max_iter_spin.setAlignment(Qt.AlignmentFlag.AlignRight)
        self._max_iter_spin.valueChanged.connect(self._settings_changed)
        iter_row.addWidget(self._limit_iter_cb)
        iter_row.addWidget(self._max_iter_spin)
        w_iter = QLabel()
        w_iter.setLayout(iter_row)
        opt_form.addRow(w_iter)
        layout.addWidget(opt_box)

        self._on_type_changed(0, True)
        self._on_kernel_changed(2, True)

    # ── Helpers ───────────────────────────────────────────────────────

    def _make_double_spin(self, lo, hi, val, step, decimals) -> QDoubleSpinBox:
        sp = QDoubleSpinBox()
        sp.setRange(lo, hi)
        sp.setValue(val)
        sp.setSingleStep(step)
        sp.setDecimals(decimals)
        sp.setAlignment(Qt.AlignmentFlag.AlignRight)
        sp.valueChanged.connect(self._settings_changed)
        return sp

    @staticmethod
    def _make_spin_row(label_text: str, spin) -> QLabel:
        row = QHBoxLayout()
        row.addWidget(QLabel(label_text))
        row.addStretch(1)
        row.addWidget(spin)
        wrapper = QLabel()
        wrapper.setLayout(row)
        return wrapper

    # ── Slot handlers ─────────────────────────────────────────────────

    def _on_type_changed(self, _id: int, checked: bool) -> None:
        if not checked:
            return
        t = self._type_group.checkedId()
        self._c_spin.setEnabled(t == _SVM)
        self._epsilon_spin.setEnabled(t == _SVM)
        self._nu_c_spin.setEnabled(t == _NU_SVM)
        self._nu_spin.setEnabled(t == _NU_SVM)
        self._settings_changed()

    def _on_kernel_changed(self, _id: int, checked: bool) -> None:
        if not checked:
            return
        k = self._kernel_group.checkedId()
        mask = [[False, False, False], [True, True, True],
                [True, False, False], [True, True, False]][k]
        for spin, en in zip([self._gamma_spin, self._coef0_spin, self._degree_spin], mask):
            spin.setEnabled(en)
        self._settings_changed()

    # ── Training ──────────────────────────────────────────────────────

    def _train(self):
        from sklearn.svm import SVC, SVR, NuSVC, NuSVR

        ds = self._dataset
        target_cols = ds.domain.target_columns
        if not target_cols:
            raise ValueError("No target column.")
        is_clf = target_cols[0].logical_type in {"categorical", "boolean"}

        t = self._type_group.checkedId()
        k = _KERNELS[self._kernel_group.checkedId()]
        gamma = self._gamma_spin.value() or "auto"
        coef0 = self._coef0_spin.value()
        degree = self._degree_spin.value()
        tol = self._tol_spin.value()
        max_iter = self._max_iter_spin.value() if self._limit_iter_cb.isChecked() else -1
        common = dict(kernel=k, degree=degree, gamma=gamma, coef0=coef0, tol=tol, max_iter=max_iter)

        if t == _SVM:
            C, eps = self._c_spin.value(), self._epsilon_spin.value()
            est = SVC(C=C, probability=True, **common) if is_clf else SVR(C=C, epsilon=eps, **common)
        else:
            nu = self._nu_spin.value()
            nu_C = self._nu_c_spin.value()
            est = NuSVC(nu=min(nu, 0.999), probability=True, **common) if is_clf else NuSVR(nu=nu, C=nu_C, **common)

        params = {"svm_type": "SVM" if t == _SVM else "ν-SVM", "kernel": k}
        return self._svc.fit(est, ds, "SVM", "svm", params)

    # ── Persistence ───────────────────────────────────────────────────

    def serialize_node_state(self) -> dict:
        return {
            **super().serialize_node_state(),
            "svm_type": self._type_group.checkedId(),
            "C": self._c_spin.value(),
            "epsilon": self._epsilon_spin.value(),
            "nu_C": self._nu_c_spin.value(),
            "nu": self._nu_spin.value(),
            "kernel": self._kernel_group.checkedId(),
            "gamma": self._gamma_spin.value(),
            "coef0": self._coef0_spin.value(),
            "degree": self._degree_spin.value(),
            "tol": self._tol_spin.value(),
            "limit_iter": self._limit_iter_cb.isChecked(),
            "max_iter": self._max_iter_spin.value(),
        }

    def restore_node_state(self, payload: dict) -> None:
        super().restore_node_state(payload)
        for btn_id, val_key, default in [
            (self._type_group, "svm_type", 0),
            (self._kernel_group, "kernel", 2),
        ]:
            btn = btn_id.button(int(payload.get(val_key, default)))
            if btn:
                btn.setChecked(True)
        self._c_spin.setValue(float(payload.get("C", 1.0)))
        self._epsilon_spin.setValue(float(payload.get("epsilon", 0.1)))
        self._nu_c_spin.setValue(float(payload.get("nu_C", 1.0)))
        self._nu_spin.setValue(float(payload.get("nu", 0.5)))
        self._gamma_spin.setValue(float(payload.get("gamma", 0.0)))
        self._coef0_spin.setValue(float(payload.get("coef0", 1.0)))
        self._degree_spin.setValue(int(payload.get("degree", 3)))
        self._tol_spin.setValue(float(payload.get("tol", 0.001)))
        self._limit_iter_cb.setChecked(bool(payload.get("limit_iter", True)))
        self._max_iter_spin.setValue(int(payload.get("max_iter", 100)))
