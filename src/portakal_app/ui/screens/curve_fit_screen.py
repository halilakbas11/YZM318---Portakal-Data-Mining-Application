from __future__ import annotations

import ast
from typing import Any

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QDoubleSpinBox,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from portakal_app.data.services.sklearn_learner_service import SklearnLearnerService
from portakal_app.models import WorkflowPayload
from portakal_app.ui.screens.model_base import ModelScreenBase

_SAFE_FUNCS = {k: v for k, v in np.__dict__.items()
               if k in ("arccos", "arccosh", "arcsin", "arcsinh", "arctan", "arctan2",
                        "arctanh", "ceil", "cos", "cosh", "degrees", "e", "exp", "expm1",
                        "floor", "log", "log10", "log2", "pi", "sin", "sinh", "sqrt",
                        "tan", "tanh", "inf", "nan", "abs", "round")}


class _ParamRow:
    def __init__(self, layout: QGridLayout, row: int, name: str = "", init: float = 1.0,
                 use_lower: bool = False, lower: float = 0.0,
                 use_upper: bool = False, upper: float = 100.0,
                 on_remove=None, on_change=None):
        self._on_change = on_change

        self.remove_btn = QPushButton("×")
        self.remove_btn.setFixedWidth(26)
        self.remove_btn.clicked.connect(on_remove)

        self.name_edit = QLineEdit(name)
        self.name_edit.setFixedWidth(70)
        self.name_edit.textChanged.connect(self._changed)

        self.init_spin = QDoubleSpinBox()
        self.init_spin.setRange(-1e9, 1e9)
        self.init_spin.setDecimals(4)
        self.init_spin.setValue(init)
        self.init_spin.valueChanged.connect(self._changed)

        self.lower_cb = QCheckBox()
        self.lower_cb.setChecked(use_lower)
        self.lower_spin = QDoubleSpinBox()
        self.lower_spin.setRange(-1e9, 1e9)
        self.lower_spin.setDecimals(4)
        self.lower_spin.setValue(lower)
        self.lower_spin.setEnabled(use_lower)
        self.lower_cb.stateChanged.connect(lambda: (self.lower_spin.setEnabled(self.lower_cb.isChecked()), self._changed()))
        self.lower_spin.valueChanged.connect(self._changed)

        self.upper_cb = QCheckBox()
        self.upper_cb.setChecked(use_upper)
        self.upper_spin = QDoubleSpinBox()
        self.upper_spin.setRange(-1e9, 1e9)
        self.upper_spin.setDecimals(4)
        self.upper_spin.setValue(upper)
        self.upper_spin.setEnabled(use_upper)
        self.upper_cb.stateChanged.connect(lambda: (self.upper_spin.setEnabled(self.upper_cb.isChecked()), self._changed()))
        self.upper_spin.valueChanged.connect(self._changed)

        layout.addWidget(self.remove_btn, row, 0)
        layout.addWidget(self.name_edit, row, 1)
        layout.addWidget(self.init_spin, row, 2)
        layout.addWidget(self.lower_cb, row, 3)
        layout.addWidget(self.lower_spin, row, 4)
        layout.addWidget(self.upper_cb, row, 5)
        layout.addWidget(self.upper_spin, row, 6)
        self._widgets = [self.remove_btn, self.name_edit, self.init_spin,
                         self.lower_cb, self.lower_spin, self.upper_cb, self.upper_spin]

    def remove_from(self, layout: QGridLayout) -> None:
        for w in self._widgets:
            layout.removeWidget(w)
            w.deleteLater()

    def _changed(self) -> None:
        if self._on_change:
            self._on_change()

    def to_dict(self) -> dict:
        return {
            "name": self.name_edit.text(),
            "init": self.init_spin.value(),
            "use_lower": self.lower_cb.isChecked(),
            "lower": self.lower_spin.value(),
            "use_upper": self.upper_cb.isChecked(),
            "upper": self.upper_spin.value(),
        }


class CurveFitScreen(ModelScreenBase):
    """Curve Fit — fit a user-defined function expression to numeric data."""

    _OUTPUT_PORT_LABEL = "Model"

    def __init__(self, parent=None) -> None:
        self._param_rows: list[_ParamRow] = []
        self._param_grid: QGridLayout | None = None
        super().__init__(parent)

    def _add_main_layout(self, layout: QVBoxLayout) -> None:
        param_box = QGroupBox("Parameters")
        param_outer = QVBoxLayout(param_box)

        # header
        hdr = QGridLayout()
        hdr.addWidget(QLabel(""), 0, 0)
        hdr.addWidget(QLabel("Name"), 0, 1)
        hdr.addWidget(QLabel("Initial"), 0, 2)
        hdr.addWidget(QLabel("Lower"), 0, 3, 1, 2)
        hdr.addWidget(QLabel("Upper"), 0, 5, 1, 2)
        param_outer.addLayout(hdr)

        self._param_grid = QGridLayout()
        self._param_grid.setColumnStretch(1, 1)
        param_outer.addLayout(self._param_grid)

        add_btn = QPushButton("+")
        add_btn.setFixedWidth(32)
        add_btn.clicked.connect(self._add_param_row)
        btn_row = QHBoxLayout()
        btn_row.addStretch(1)
        btn_row.addWidget(add_btn)
        param_outer.addLayout(btn_row)

        layout.addWidget(param_box)

        expr_box = QGroupBox("Expression")
        expr_layout = QVBoxLayout(expr_box)
        self._expr_edit = QLineEdit()
        self._expr_edit.setPlaceholderText("e.g. a * exp(-b * x) + c")
        self._expr_edit.textChanged.connect(self._settings_changed)
        expr_layout.addWidget(self._expr_edit)
        layout.addWidget(expr_box)

    def _add_param_row(self, name: str = "", init: float = 1.0,
                       use_lower: bool = False, lower: float = 0.0,
                       use_upper: bool = False, upper: float = 100.0) -> None:
        if not name:
            name = f"p{len(self._param_rows) + 1}"
        row_idx = len(self._param_rows) + 1  # +1 for the grid offset (0 = header)
        row = _ParamRow(self._param_grid, row_idx, name, init, use_lower, lower,
                        use_upper, upper,
                        on_remove=lambda r=None: self._remove_param_row(row),
                        on_change=self._settings_changed)
        self._param_rows.append(row)
        self._settings_changed()

    def _remove_param_row(self, row: _ParamRow) -> None:
        row.remove_from(self._param_grid)
        self._param_rows.remove(row)
        self._settings_changed()

    def _validate_expression(self, expr: str) -> bool:
        try:
            ast.parse(expr, mode="eval")
            return True
        except SyntaxError:
            return False

    def _train(self):
        from scipy.optimize import curve_fit as scipy_curve_fit
        from portakal_app.sklearn_model_artifacts import SklearnModelArtifact
        from portakal_app.data.services.sklearn_learner_service import SklearnLearnerService
        import uuid

        ds = self._dataset
        target_cols = ds.domain.target_columns
        if not target_cols:
            raise ValueError("No target column.")
        if target_cols[0].logical_type not in {"numeric"}:
            raise ValueError("Curve Fit requires a numeric target.")

        expr = self._expr_edit.text().strip()
        if not expr:
            raise ValueError("Please enter an expression.")
        if not self._validate_expression(expr):
            raise ValueError("Invalid expression syntax.")

        params = {r.to_dict()["name"]: r.to_dict() for r in self._param_rows if r.to_dict()["name"]}
        if not params:
            raise ValueError("Please define at least one fitting parameter.")

        feature_cols = [c for c in ds.domain.feature_columns if c.logical_type == "numeric"]
        if not feature_cols:
            raise ValueError("No numeric feature columns found.")

        target_name = target_cols[0].name
        feat_names = [c.name for c in feature_cols]

        X_df = ds.dataframe.select(feat_names).drop_nulls()
        y_series = ds.dataframe.get_column(target_name).drop_nulls()
        min_len = min(len(X_df), len(y_series))
        X_np = X_df.to_numpy()[:min_len]
        y_np = y_series.to_numpy()[:min_len].astype(float)

        param_names = list(params.keys())
        p0 = [params[p]["init"] for p in param_names]
        lower_bounds = [params[p]["lower"] if params[p]["use_lower"] else -np.inf for p in param_names]
        upper_bounds = [params[p]["upper"] if params[p]["use_upper"] else np.inf for p in param_names]

        local_vars = {n: X_np[:, i] for i, n in enumerate(feat_names)}

        def make_func(expression: str, feat_names_: list[str], param_names_: list[str]):
            def f(X, *args):
                env = dict(_SAFE_FUNCS)
                env.update({n: X[:, i] for i, n in enumerate(feat_names_)})
                env.update(dict(zip(param_names_, args)))
                return eval(expression, {"__builtins__": {}}, env)  # noqa: S307
            return f

        func = make_func(expr, feat_names, param_names)

        try:
            popt, _ = scipy_curve_fit(func, X_np, y_np, p0=p0,
                                      bounds=(lower_bounds, upper_bounds),
                                      maxfev=10000)
        except Exception as e:
            raise ValueError(f"Curve fitting failed: {e}") from e

        fitted_params = dict(zip(param_names, popt))

        class _CurveFitEstimator:
            """Thin wrapper that satisfies SklearnModelArtifact.trained_model interface."""
            def __init__(self, expression, feat_names, param_names, popt):
                self.expression_ = expression
                self.feat_names_ = feat_names
                self.param_names_ = param_names
                self.popt_ = popt

            def predict(self, X):
                env = dict(_SAFE_FUNCS)
                env.update({n: X[:, i] for i, n in enumerate(self.feat_names_)})
                env.update(dict(zip(self.param_names_, self.popt_)))
                return eval(self.expression_, {"__builtins__": {}}, env)  # noqa: S307

        trained = _CurveFitEstimator(expr, feat_names, param_names, popt)
        return SklearnModelArtifact(
            model_id=str(uuid.uuid4()),
            display_name="Curve Fit",
            model_type="curve_fit",
            sklearn_estimator=None,
            trained_model=trained,
            feature_names=tuple(feat_names),
            categorical_encoders={},
            numeric_cols=tuple(feat_names),
            target_name=target_name,
            is_classifier=False,
            class_values=None,
            target_encoder=None,
            params={"expression": expr, **fitted_params},
            training_dataset=ds,
        )

    def serialize_node_state(self) -> dict:
        return {
            **super().serialize_node_state(),
            "expression": self._expr_edit.text(),
            "params": [r.to_dict() for r in self._param_rows],
        }

    def restore_node_state(self, payload: dict) -> None:
        super().restore_node_state(payload)
        self._expr_edit.setText(str(payload.get("expression", "")))
        for row in list(self._param_rows):
            row.remove_from(self._param_grid)
        self._param_rows.clear()
        for p in payload.get("params", []):
            self._add_param_row(
                name=p.get("name", ""),
                init=float(p.get("init", 1.0)),
                use_lower=bool(p.get("use_lower", False)),
                lower=float(p.get("lower", 0.0)),
                use_upper=bool(p.get("use_upper", False)),
                upper=float(p.get("upper", 100.0)),
            )
