from __future__ import annotations

import ast
import uuid

import numpy as np
from PySide6.QtCore import Qt, QPointF, QRectF
from PySide6.QtGui import QColor, QFont, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from portakal_app.models import WorkflowPayload
from portakal_app.ui.screens.model_base import ModelScreenBase

_SAFE_FUNCS = {k: v for k, v in np.__dict__.items()
               if k in ("arccos", "arccosh", "arcsin", "arcsinh", "arctan", "arctan2",
                        "arctanh", "ceil", "cos", "cosh", "degrees", "e", "exp", "expm1",
                        "floor", "log", "log10", "log2", "pi", "sin", "sinh", "sqrt",
                        "tan", "tanh", "inf", "nan", "abs", "round")}

_FUNCTION_LIST = [
    "sin", "cos", "tan", "arcsin", "arccos", "arctan", "arctan2",
    "sinh", "cosh", "tanh", "arcsinh", "arccosh", "arctanh",
    "exp", "expm1", "log", "log10", "log2",
    "sqrt", "abs", "ceil", "floor", "round", "degrees",
]

_W_REMOVE = 26
_W_NAME   = 72
_W_INIT   = 90
_W_CB     = 20
_W_BOUND  = 82


# ── sklearn-compatible estimator (module-level so clone() works) ──────────────

class CurveFitEstimator:
    """Wraps scipy curve_fit as a proper sklearn-compatible estimator.

    Inherits BaseEstimator + RegressorMixin so sklearn.base.clone(),
    cross_validate(), and internal tag/param checks all work correctly.
    BaseEstimator auto-generates get_params/set_params from __init__ signature.
    RegressorMixin adds a score() method (R² via predict).
    """

    # Pull in sklearn base classes at class-body time so the inheritance is
    # established before any sklearn internal machinery inspects the class.
    try:
        from sklearn.base import BaseEstimator as _BE, RegressorMixin as _RM
        _bases_loaded = True
    except ImportError:
        _BE = object  # type: ignore[assignment,misc]
        _RM = object  # type: ignore[assignment,misc]
        _bases_loaded = False

    # Dynamically build the real class after the try/except
    pass


# Rebuild with proper bases now that we have the references
try:
    from sklearn.base import BaseEstimator as _SKBase, RegressorMixin as _SKReg

    class CurveFitEstimator(_SKBase, _SKReg):  # type: ignore[no-redef]
        """sklearn-compatible curve-fit estimator (module-level for clone())."""

        def __init__(
            self,
            expression: str = "",
            feat_names: tuple = (),
            param_names: tuple = (),
            p0: tuple = (),
            lower_bounds: tuple = (),
            upper_bounds: tuple = (),
        ) -> None:
            # Store every constructor arg as a same-named attribute —
            # BaseEstimator.get_params() reads them by __init__ signature.
            self.expression = expression
            self.feat_names = feat_names
            self.param_names = param_names
            self.p0 = p0
            self.lower_bounds = lower_bounds
            self.upper_bounds = upper_bounds

        def fit(self, X: np.ndarray, y: np.ndarray) -> "CurveFitEstimator":
            import warnings
            from scipy.optimize import curve_fit as scipy_curve_fit
            from scipy.optimize import OptimizeWarning

            def _func(X_: np.ndarray, *args: float) -> np.ndarray:
                env = dict(_SAFE_FUNCS)
                env.update({n: X_[:, i] for i, n in enumerate(self.feat_names)})
                env.update(dict(zip(self.param_names, args)))
                return np.asarray(
                    eval(self.expression, {"__builtins__": {}}, env),  # noqa: S307
                    dtype=float,
                )

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", OptimizeWarning)
                popt, _ = scipy_curve_fit(
                    _func, X, y,
                    p0=list(self.p0),
                    bounds=(list(self.lower_bounds), list(self.upper_bounds)),
                    maxfev=10000,
                )
            self.popt_ = popt  # trailing _ = sklearn convention for fitted attrs
            return self

        def predict(self, X: np.ndarray) -> np.ndarray:
            from sklearn.utils.validation import check_is_fitted
            check_is_fitted(self, "popt_")
            env = dict(_SAFE_FUNCS)
            env.update({n: X[:, i] for i, n in enumerate(self.feat_names)})
            env.update(dict(zip(self.param_names, self.popt_)))
            return np.asarray(
                eval(self.expression, {"__builtins__": {}}, env),  # noqa: S307
                dtype=float,
            )

except ImportError:
    # Fallback if sklearn is somehow not installed — keeps the file importable.
    pass


# ── Curve preview plot ────────────────────────────────────────────────────────

class CurvePlotWidget(QWidget):
    """Scatter plot of data + fitted curve, painted with QPainter."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setMinimumHeight(190)
        self._x_data: np.ndarray | None = None
        self._y_data: np.ndarray | None = None
        self._curve_x: np.ndarray | None = None
        self._curve_y: np.ndarray | None = None
        self._x_label = "x"
        self._y_label = "y"
        self._message = "Train the model to see the curve."
        self.setStyleSheet(
            "background: #1a1c22; border-radius: 8px; border: 1px solid #2d2f3b;"
        )

    def set_message(self, msg: str) -> None:
        self._x_data = None
        self._curve_x = None
        self._message = msg
        self.update()

    def set_plot(
        self,
        x_data: np.ndarray,
        y_data: np.ndarray,
        curve_x: np.ndarray,
        curve_y: np.ndarray,
        x_label: str = "x",
        y_label: str = "y",
    ) -> None:
        self._x_data = np.asarray(x_data, dtype=float)
        self._y_data = np.asarray(y_data, dtype=float)
        self._curve_x = np.asarray(curve_x, dtype=float)
        self._curve_y = np.asarray(curve_y, dtype=float)
        self._x_label = x_label
        self._y_label = y_label
        self._message = ""
        self.update()

    def paintEvent(self, event) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        W, H = self.width(), self.height()
        painter.fillRect(0, 0, W, H, QColor("#1a1c22"))

        if self._x_data is None:
            painter.setPen(QColor("#475569"))
            painter.setFont(QFont("Arial", 9))
            painter.drawText(QRectF(0, 0, W, H), Qt.AlignmentFlag.AlignCenter, self._message)
            return

        ml, mr, mt, mb = 50, 12, 10, 30
        pw = W - ml - mr
        ph = H - mt - mb
        if pw <= 10 or ph <= 10:
            return

        # Determine axis extents from both data and curve
        all_x = self._x_data
        all_y_parts = [self._y_data]
        if self._curve_y is not None:
            finite_cy = self._curve_y[np.isfinite(self._curve_y)]
            if finite_cy.size:
                all_y_parts.append(finite_cy)
        all_y = np.concatenate(all_y_parts)

        x_min, x_max = float(np.nanmin(all_x)), float(np.nanmax(all_x))
        y_min, y_max = float(np.nanmin(all_y)), float(np.nanmax(all_y))

        xpad = (x_max - x_min) * 0.05 or 0.5
        ypad = (y_max - y_min) * 0.08 or 0.5
        x_min -= xpad;  x_max += xpad
        y_min -= ypad;  y_max += ypad
        xr = x_max - x_min
        yr = y_max - y_min

        def px(x: float) -> float:
            return ml + (x - x_min) / xr * pw

        def py(y: float) -> float:
            return mt + ph - (y - y_min) / yr * ph

        # Subtle grid
        painter.setPen(QPen(QColor(255, 255, 255, 12), 1))
        for i in range(1, 4):
            gx = ml + i * pw / 4
            gy = mt + i * ph / 4
            painter.drawLine(QPointF(gx, mt), QPointF(gx, mt + ph))
            painter.drawLine(QPointF(ml, gy), QPointF(ml + pw, gy))

        # Axes border
        painter.setPen(QPen(QColor("#334155"), 1))
        painter.drawRect(QRectF(ml, mt, pw, ph))

        # Fitted curve (orange line)
        if self._curve_x is not None and self._curve_y is not None:
            path = QPainterPath()
            started = False
            for xi, yi in zip(self._curve_x, self._curve_y):
                if not (np.isfinite(xi) and np.isfinite(yi)):
                    started = False
                    continue
                # clip y to visible range (±50% of plot height beyond bounds)
                yi_clipped = max(y_min - yr * 0.5, min(y_max + yr * 0.5, yi))
                pt = QPointF(px(xi), py(yi_clipped))
                if not started:
                    path.moveTo(pt)
                    started = True
                else:
                    path.lineTo(pt)

            if not path.isEmpty():
                pen = QPen(QColor("#f97316"), 2)
                pen.setCapStyle(Qt.PenCapStyle.RoundCap)
                pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
                painter.setPen(pen)
                painter.setBrush(Qt.BrushStyle.NoBrush)
                painter.drawPath(path)

        # Scatter: data points (indigo dots)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(99, 102, 241, 200))
        for xi, yi in zip(self._x_data, self._y_data):
            if not (np.isfinite(xi) and np.isfinite(yi)):
                continue
            if xi < x_min or xi > x_max or yi < y_min or yi > y_max:
                continue
            painter.drawEllipse(QPointF(px(xi), py(yi)), 3.0, 3.0)

        # Y-axis tick values
        painter.setPen(QColor("#475569"))
        painter.setFont(QFont("Arial", 6))
        for frac in (0.0, 0.5, 1.0):
            v = y_min + frac * yr
            y_px = py(v)
            painter.drawText(
                QRectF(2, y_px - 7, ml - 5, 14),
                Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
                f"{v:.3g}",
            )

        # X-axis tick values
        for frac in (0.0, 0.5, 1.0):
            v = x_min + frac * xr
            x_px = px(v)
            painter.drawText(
                QRectF(x_px - 20, mt + ph + 2, 40, 12),
                Qt.AlignmentFlag.AlignCenter,
                f"{v:.3g}",
            )

        # Axis labels
        painter.setPen(QColor("#64748b"))
        painter.setFont(QFont("Arial", 7))

        # X label centred below axis
        painter.drawText(
            QRectF(ml, H - 14, pw, 12),
            Qt.AlignmentFlag.AlignCenter,
            self._x_label,
        )

        # Y label rotated on the left
        painter.save()
        painter.translate(8, mt + ph / 2)
        painter.rotate(-90)
        painter.drawText(QRectF(-30, -6, 60, 12), Qt.AlignmentFlag.AlignCenter, self._y_label)
        painter.restore()


# ── Parameter row widget ──────────────────────────────────────────────────────

class _ParamRow(QWidget):
    """One row: [×] [name] [initial] [☑ lower] [☑ upper]"""

    def __init__(self, name: str = "", init: float = 1.0,
                 use_lower: bool = False, lower: float = 0.0,
                 use_upper: bool = False, upper: float = 100.0,
                 on_remove=None, on_change=None,
                 parent=None) -> None:
        super().__init__(parent)
        self._on_change = on_change

        row = QHBoxLayout(self)
        row.setContentsMargins(0, 1, 0, 1)
        row.setSpacing(4)

        self.remove_btn = QPushButton("×")
        self.remove_btn.setFixedWidth(_W_REMOVE)
        self.remove_btn.setToolTip("Remove parameter")
        if on_remove:
            self.remove_btn.clicked.connect(on_remove)

        self.name_edit = QLineEdit(name)
        self.name_edit.setFixedWidth(_W_NAME)
        self.name_edit.setPlaceholderText("name")

        self.init_spin = QDoubleSpinBox()
        self.init_spin.setRange(-1e9, 1e9)
        self.init_spin.setDecimals(4)
        self.init_spin.setValue(init)
        self.init_spin.setFixedWidth(_W_INIT)

        self.lower_cb = QCheckBox()
        self.lower_cb.setFixedWidth(_W_CB)
        self.lower_cb.setChecked(use_lower)
        self.lower_spin = QDoubleSpinBox()
        self.lower_spin.setRange(-1e9, 1e9)
        self.lower_spin.setDecimals(4)
        self.lower_spin.setValue(lower)
        self.lower_spin.setEnabled(use_lower)
        self.lower_spin.setFixedWidth(_W_BOUND)

        self.upper_cb = QCheckBox()
        self.upper_cb.setFixedWidth(_W_CB)
        self.upper_cb.setChecked(use_upper)
        self.upper_spin = QDoubleSpinBox()
        self.upper_spin.setRange(-1e9, 1e9)
        self.upper_spin.setDecimals(4)
        self.upper_spin.setValue(upper)
        self.upper_spin.setEnabled(use_upper)
        self.upper_spin.setFixedWidth(_W_BOUND)

        row.addWidget(self.remove_btn)
        row.addWidget(self.name_edit)
        row.addWidget(self.init_spin)
        row.addWidget(self.lower_cb, 0, Qt.AlignmentFlag.AlignHCenter)
        row.addWidget(self.lower_spin)
        row.addWidget(self.upper_cb, 0, Qt.AlignmentFlag.AlignHCenter)
        row.addWidget(self.upper_spin)
        row.addStretch(1)

        self.name_edit.textChanged.connect(self._emit_change)
        self.init_spin.valueChanged.connect(self._emit_change)
        self.lower_cb.stateChanged.connect(self._on_lower_cb)
        self.lower_spin.valueChanged.connect(self._emit_change)
        self.upper_cb.stateChanged.connect(self._on_upper_cb)
        self.upper_spin.valueChanged.connect(self._emit_change)

    def _on_lower_cb(self) -> None:
        self.lower_spin.setEnabled(self.lower_cb.isChecked())
        self._emit_change()

    def _on_upper_cb(self) -> None:
        self.upper_spin.setEnabled(self.upper_cb.isChecked())
        self._emit_change()

    def _emit_change(self) -> None:
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


# ── Screen ────────────────────────────────────────────────────────────────────

class CurveFitScreen(ModelScreenBase):
    """Curve Fit — fit a user-defined function expression to numeric data."""

    _OUTPUT_PORT_LABEL = "Model"

    def __init__(self, parent=None) -> None:
        self._param_rows: list[_ParamRow] = []
        self._rows_layout: QVBoxLayout | None = None
        self._feature_combo: QComboBox | None = None
        self._param_combo: QComboBox | None = None
        self._expr_edit: QLineEdit | None = None
        self._plot: CurvePlotWidget | None = None
        super().__init__(parent)

    # ── Layout ────────────────────────────────────────────────────────────────

    def _add_main_layout(self, layout: QVBoxLayout) -> None:
        # ── Parameters group ─────────────────────────────────────────
        param_box = QGroupBox("Parameters")
        param_outer = QVBoxLayout(param_box)
        param_outer.setSpacing(2)

        top_row = QHBoxLayout()
        top_row.addStretch(1)
        add_btn = QPushButton("+")
        add_btn.setFixedWidth(28)
        add_btn.setToolTip("Add parameter")
        add_btn.clicked.connect(lambda: self._add_param_row())
        top_row.addWidget(add_btn)
        param_outer.addLayout(top_row)

        # Column header (widths mirror _ParamRow)
        hdr = QHBoxLayout()
        hdr.setSpacing(4)
        hdr.setContentsMargins(0, 0, 0, 2)

        def _h(text: str, w: int, align=Qt.AlignmentFlag.AlignLeft):
            lbl = QLabel(text)
            lbl.setFixedWidth(w)
            lbl.setStyleSheet("color: #64748b; background: transparent; font-size: 11px;")
            lbl.setAlignment(align)
            return lbl

        hdr.addWidget(_h("", _W_REMOVE))
        hdr.addWidget(_h("Name", _W_NAME))
        hdr.addWidget(_h("Initial", _W_INIT))
        hdr.addWidget(_h("☑", _W_CB, Qt.AlignmentFlag.AlignHCenter))
        hdr.addWidget(_h("Lower", _W_BOUND))
        hdr.addWidget(_h("☑", _W_CB, Qt.AlignmentFlag.AlignHCenter))
        hdr.addWidget(_h("Upper", _W_BOUND))
        hdr.addStretch(1)
        param_outer.addLayout(hdr)

        self._rows_layout = QVBoxLayout()
        self._rows_layout.setSpacing(0)
        self._rows_layout.setContentsMargins(0, 0, 0, 0)
        param_outer.addLayout(self._rows_layout)

        layout.addWidget(param_box)

        # ── Expression group ──────────────────────────────────────────
        expr_box = QGroupBox("Expression")
        expr_layout = QVBoxLayout(expr_box)
        expr_layout.setSpacing(6)

        self._expr_edit = QLineEdit()
        self._expr_edit.setPlaceholderText("e.g.  a * exp(-b * x) + c")
        self._expr_edit.textChanged.connect(self._settings_changed)
        expr_layout.addWidget(self._expr_edit)

        sel_row = QHBoxLayout()
        sel_row.setSpacing(6)

        self._feature_combo = QComboBox()
        self._feature_combo.addItem("Select Feature")
        self._feature_combo.setToolTip("Insert a feature column name at the cursor")
        self._feature_combo.currentIndexChanged.connect(self._on_feature_selected)
        sel_row.addWidget(self._feature_combo, 1)

        self._param_combo = QComboBox()
        self._param_combo.addItem("Select Parameter")
        self._param_combo.setToolTip("Insert a parameter name at the cursor")
        self._param_combo.currentIndexChanged.connect(self._on_param_selected)
        sel_row.addWidget(self._param_combo, 1)

        self._func_combo = QComboBox()
        self._func_combo.addItem("Select Function")
        for fn in _FUNCTION_LIST:
            self._func_combo.addItem(fn)
        self._func_combo.setToolTip("Insert a math function at the cursor")
        self._func_combo.currentIndexChanged.connect(self._on_function_selected)
        sel_row.addWidget(self._func_combo, 1)

        expr_layout.addLayout(sel_row)
        layout.addWidget(expr_box)

        # ── Curve preview ─────────────────────────────────────────────
        curve_box = QGroupBox("Curve")
        curve_layout = QVBoxLayout(curve_box)
        curve_layout.setContentsMargins(6, 6, 6, 6)
        self._plot = CurvePlotWidget()
        curve_layout.addWidget(self._plot)
        layout.addWidget(curve_box)

    # ── Override _apply to refresh the plot after training ────────────────────

    def _apply(self) -> None:
        super()._apply()
        self._update_curve_plot()

    # ── Workflow integration ──────────────────────────────────────────────────

    def set_input_payload(self, payload: WorkflowPayload | None) -> None:
        super().set_input_payload(payload)
        if payload is not None:
            from portakal_app.data.models import DatasetHandle
            if isinstance(payload.value, DatasetHandle) and self._feature_combo is not None:
                self._refresh_feature_combo(payload.value)

    # ── Settings changed ──────────────────────────────────────────────────────

    def _settings_changed(self) -> None:
        self._refresh_param_combo()
        super()._settings_changed()

    # ── Combo helpers ─────────────────────────────────────────────────────────

    def _refresh_feature_combo(self, dataset) -> None:
        cb = self._feature_combo
        if cb is None:
            return
        cb.blockSignals(True)
        cb.clear()
        cb.addItem("Select Feature")
        for col in dataset.domain.feature_columns:
            if col.logical_type == "numeric":
                cb.addItem(col.name)
        cb.blockSignals(False)

    def _refresh_param_combo(self) -> None:
        cb = self._param_combo
        if cb is None:
            return
        cb.blockSignals(True)
        cb.clear()
        cb.addItem("Select Parameter")
        for row in self._param_rows:
            name = row.name_edit.text().strip()
            if name:
                cb.addItem(name)
        cb.blockSignals(False)

    def _insert_at_cursor(self, text: str) -> None:
        pos = self._expr_edit.cursorPosition()
        current = self._expr_edit.text()
        self._expr_edit.setText(current[:pos] + text + current[pos:])
        self._expr_edit.setCursorPosition(pos + len(text))
        self._expr_edit.setFocus()

    def _on_feature_selected(self, index: int) -> None:
        if index <= 0 or self._feature_combo is None:
            return
        self._insert_at_cursor(self._feature_combo.itemText(index))
        self._feature_combo.blockSignals(True)
        self._feature_combo.setCurrentIndex(0)
        self._feature_combo.blockSignals(False)

    def _on_param_selected(self, index: int) -> None:
        if index <= 0 or self._param_combo is None:
            return
        self._insert_at_cursor(self._param_combo.itemText(index))
        self._param_combo.blockSignals(True)
        self._param_combo.setCurrentIndex(0)
        self._param_combo.blockSignals(False)

    def _on_function_selected(self, index: int) -> None:
        if index <= 0:
            return
        name = self._func_combo.itemText(index)
        pos = self._expr_edit.cursorPosition()
        current = self._expr_edit.text()
        insertion = name + "()"
        self._expr_edit.setText(current[:pos] + insertion + current[pos:])
        self._expr_edit.setCursorPosition(pos + len(name) + 1)
        self._expr_edit.setFocus()
        self._func_combo.blockSignals(True)
        self._func_combo.setCurrentIndex(0)
        self._func_combo.blockSignals(False)

    # ── Param row management ──────────────────────────────────────────────────

    def _add_param_row(self, name: str = "", init: float = 1.0,
                       use_lower: bool = False, lower: float = 0.0,
                       use_upper: bool = False, upper: float = 100.0) -> None:
        if not name:
            name = f"p{len(self._param_rows) + 1}"

        row = _ParamRow(
            name=name, init=init,
            use_lower=use_lower, lower=lower,
            use_upper=use_upper, upper=upper,
            on_remove=None,
            on_change=self._settings_changed,
        )
        row.remove_btn.clicked.connect(
            lambda _checked=False, r=row: self._remove_param_row(r)
        )

        self._rows_layout.addWidget(row)
        self._param_rows.append(row)
        self._settings_changed()

    def _remove_param_row(self, row: _ParamRow) -> None:
        if row in self._param_rows:
            self._param_rows.remove(row)
        self._rows_layout.removeWidget(row)
        row.deleteLater()
        self._settings_changed()

    # ── Curve plot update ─────────────────────────────────────────────────────

    def _update_curve_plot(self) -> None:
        if self._plot is None:
            return

        if self._dataset is None:
            self._plot.set_message("Connect data to see the curve.")
            return

        feat_cols = [c for c in self._dataset.domain.feature_columns if c.logical_type == "numeric"]
        target_cols = self._dataset.domain.target_columns

        if len(feat_cols) != 1:
            msg = ("Curve plot requires exactly 1 numeric feature."
                   if feat_cols else "No numeric feature columns.")
            self._plot.set_message(msg)
            return

        if not target_cols or target_cols[0].logical_type != "numeric":
            self._plot.set_message("Curve plot requires a numeric target column.")
            return

        trained: CurveFitEstimator | None = getattr(self._model_artifact, "trained_model", None)
        if trained is None or not isinstance(trained, CurveFitEstimator) or trained.popt_ is None:
            self._plot.set_message("Train the model to see the fitted curve.")
            return

        x_col = feat_cols[0].name
        y_col = target_cols[0].name

        try:
            df = self._dataset.dataframe.select([x_col, y_col]).drop_nulls()
            x_data = df.get_column(x_col).to_numpy().astype(float)
            y_data = df.get_column(y_col).to_numpy().astype(float)
        except Exception:
            self._plot.set_message("Could not load data for plot.")
            return

        x_curve = np.linspace(x_data.min(), x_data.max(), 300)
        X_curve = x_curve.reshape(-1, 1)

        try:
            y_curve = trained.predict(X_curve)
        except Exception:
            self._plot.set_message("Could not evaluate expression for plot.")
            return

        self._plot.set_plot(x_data, y_data, x_curve, y_curve, x_col, y_col)

    # ── Training ──────────────────────────────────────────────────────────────

    def _validate_expression(self, expr: str) -> bool:
        try:
            ast.parse(expr, mode="eval")
            return True
        except SyntaxError:
            return False

    def _train(self):
        from sklearn.base import clone
        from portakal_app.sklearn_model_artifacts import SklearnModelArtifact

        ds = self._dataset
        target_cols = ds.domain.target_columns
        if not target_cols:
            raise ValueError("No target column.")
        if target_cols[0].logical_type != "numeric":
            raise ValueError("Curve Fit requires a numeric target.")

        expr = self._expr_edit.text().strip()
        if not expr:
            raise ValueError("Please enter an expression.")
        if not self._validate_expression(expr):
            raise ValueError("Invalid expression syntax.")

        param_dicts = {
            r.to_dict()["name"]: r.to_dict()
            for r in self._param_rows if r.to_dict()["name"].strip()
        }
        if not param_dicts:
            raise ValueError("Please define at least one fitting parameter.")

        feature_cols = [c for c in ds.domain.feature_columns if c.logical_type == "numeric"]
        if not feature_cols:
            raise ValueError("No numeric feature columns found.")

        target_name = target_cols[0].name
        feat_names = [c.name for c in feature_cols]

        df = ds.dataframe.select(feat_names + [target_name]).drop_nulls()
        X_np = df.select(feat_names).to_numpy().astype(float)
        y_np = df.get_column(target_name).to_numpy().astype(float)

        param_names = list(param_dicts.keys())
        p0 = tuple(param_dicts[p]["init"] for p in param_names)
        lower_bounds = tuple(
            param_dicts[p]["lower"] if param_dicts[p]["use_lower"] else -np.inf
            for p in param_names
        )
        upper_bounds = tuple(
            param_dicts[p]["upper"] if param_dicts[p]["use_upper"] else np.inf
            for p in param_names
        )

        # Unfitted estimator — used by Test & Score for cross-validation
        unfitted = CurveFitEstimator(
            expression=expr,
            feat_names=tuple(feat_names),
            param_names=tuple(param_names),
            p0=p0,
            lower_bounds=lower_bounds,
            upper_bounds=upper_bounds,
        )

        # Fitted estimator — used for predictions and the curve plot
        try:
            fitted = clone(unfitted)
            fitted.fit(X_np, y_np)
        except Exception as e:
            raise ValueError(f"Curve fitting failed: {e}") from e

        return SklearnModelArtifact(
            model_id=str(uuid.uuid4()),
            display_name="Curve Fit",
            model_type="curve_fit",
            sklearn_estimator=unfitted,   # unfitted: sklearn clone+fit per cv fold
            trained_model=fitted,         # fitted: for predict & plot
            feature_names=tuple(feat_names),
            categorical_encoders={},
            numeric_cols=tuple(feat_names),
            target_name=target_name,
            is_classifier=False,
            class_values=None,
            target_encoder=None,
            params={
                "expression": expr,
                **dict(zip(param_names, fitted.popt_)),
            },
            training_dataset=ds,
        )

    # ── Serialization ─────────────────────────────────────────────────────────

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
            self._rows_layout.removeWidget(row)
            row.deleteLater()
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
