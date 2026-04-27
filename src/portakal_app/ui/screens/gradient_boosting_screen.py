from __future__ import annotations

from itertools import chain

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QLabel,
    QSlider,
    QSpinBox,
    QVBoxLayout,
)

from portakal_app.data.services.sklearn_learner_service import SklearnLearnerService
from portakal_app.ui.screens.model_base import ModelScreenBase


_LAMBDAS = list(chain(
    [x / 10000 for x in range(1, 10)],
    [x / 1000 for x in range(1, 20)],
    [x / 100 for x in range(2, 20)],
    [x / 10 for x in range(2, 9)],
    range(1, 20),
    range(20, 100, 5),
    range(100, 1001, 100),
))


class GradientBoostingScreen(ModelScreenBase):
    """Gradient Boosting on decision trees — sklearn GradientBoosting equivalent.

    Supports both classification (GradientBoostingClassifier) and regression
    (GradientBoostingRegressor) depending on the target column type.
    """

    _OUTPUT_PORT_LABEL = "Model"

    def __init__(self, parent=None) -> None:
        self._svc = SklearnLearnerService()
        super().__init__(parent)

    def _add_main_layout(self, layout: QVBoxLayout) -> None:
        basic = QGroupBox("Basic Properties")
        form1 = QFormLayout(basic)
        form1.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)

        self._n_spin = QSpinBox()
        self._n_spin.setRange(1, 10000)
        self._n_spin.setValue(100)
        self._n_spin.setAlignment(Qt.AlignmentFlag.AlignRight)
        self._n_spin.setToolTip("Total number of boosting stages to perform.")
        self._n_spin.valueChanged.connect(self._settings_changed)
        form1.addRow("Number of trees:", self._n_spin)

        self._lr_spin = QDoubleSpinBox()
        self._lr_spin.setRange(0.0, 1.0)
        self._lr_spin.setValue(0.1)
        self._lr_spin.setSingleStep(0.001)
        self._lr_spin.setDecimals(3)
        self._lr_spin.setAlignment(Qt.AlignmentFlag.AlignRight)
        self._lr_spin.setToolTip("Shrinks the contribution of each tree. Lower values need more trees.")
        self._lr_spin.valueChanged.connect(self._settings_changed)
        form1.addRow("Learning rate:", self._lr_spin)

        self._replicable_cb = QCheckBox("Replicable training")
        self._replicable_cb.setChecked(True)
        self._replicable_cb.stateChanged.connect(self._settings_changed)
        form1.addRow(self._replicable_cb)

        self._lambda_label = QLabel()
        self._lambda_slider = QSlider(Qt.Orientation.Horizontal)
        self._lambda_slider.setRange(0, len(_LAMBDAS) - 1)
        self._lambda_slider.setValue(0)
        self._lambda_slider.sliderReleased.connect(self._settings_changed)
        self._lambda_slider.valueChanged.connect(self._update_lambda_label)
        self._update_lambda_label()
        form1.addRow("Regularization:", self._lambda_slider)
        form1.addRow(self._lambda_label)

        layout.addWidget(basic)

        growth = QGroupBox("Growth Control")
        form2 = QFormLayout(growth)
        form2.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)

        self._depth_spin = QSpinBox()
        self._depth_spin.setRange(1, 50)
        self._depth_spin.setValue(3)
        self._depth_spin.setAlignment(Qt.AlignmentFlag.AlignRight)
        self._depth_spin.valueChanged.connect(self._settings_changed)
        form2.addRow("Limit depth of individual trees:", self._depth_spin)

        self._min_split_spin = QSpinBox()
        self._min_split_spin.setRange(2, 1000)
        self._min_split_spin.setValue(2)
        self._min_split_spin.setAlignment(Qt.AlignmentFlag.AlignRight)
        self._min_split_spin.valueChanged.connect(self._settings_changed)
        form2.addRow("Do not split subsets smaller than:", self._min_split_spin)

        layout.addWidget(growth)

        sub = QGroupBox("Subsampling")
        form3 = QFormLayout(sub)
        form3.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)

        self._subsample_spin = QDoubleSpinBox()
        self._subsample_spin.setRange(0.05, 1.0)
        self._subsample_spin.setValue(1.0)
        self._subsample_spin.setSingleStep(0.05)
        self._subsample_spin.setDecimals(2)
        self._subsample_spin.setAlignment(Qt.AlignmentFlag.AlignRight)
        self._subsample_spin.valueChanged.connect(self._settings_changed)
        form3.addRow("Fraction of training instances:", self._subsample_spin)

        layout.addWidget(sub)

    def _update_lambda_label(self) -> None:
        lam = _LAMBDAS[self._lambda_slider.value()]
        self._lambda_label.setText(f"Lambda: {lam}")
        self._lambda_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

    def _train(self):
        from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor

        ds = self._dataset
        target_cols = ds.domain.target_columns
        if not target_cols:
            raise ValueError("No target column.")
        is_clf = target_cols[0].logical_type in {"categorical", "boolean"}

        n = self._n_spin.value()
        lr = self._lr_spin.value()
        random_state = 0 if self._replicable_cb.isChecked() else None
        max_depth = self._depth_spin.value()
        min_split = self._min_split_spin.value()
        subsample = self._subsample_spin.value()

        kw = dict(n_estimators=n, learning_rate=lr, random_state=random_state,
                  max_depth=max_depth, min_samples_split=min_split, subsample=subsample)
        est = GradientBoostingClassifier(**kw) if is_clf else GradientBoostingRegressor(**kw)
        params = {
            "n_estimators": n,
            "learning_rate": lr,
            "max_depth": max_depth,
            "min_samples_split": min_split,
            "subsample": subsample,
        }
        return self._svc.fit(est, ds, "Gradient Boosting", "gradient_boosting", params)

    def serialize_node_state(self) -> dict:
        return {
            **super().serialize_node_state(),
            "n_estimators": self._n_spin.value(),
            "learning_rate": self._lr_spin.value(),
            "replicable": self._replicable_cb.isChecked(),
            "lambda_index": self._lambda_slider.value(),
            "max_depth": self._depth_spin.value(),
            "min_split": self._min_split_spin.value(),
            "subsample": self._subsample_spin.value(),
        }

    def restore_node_state(self, payload: dict) -> None:
        super().restore_node_state(payload)
        self._n_spin.setValue(int(payload.get("n_estimators", 100)))
        self._lr_spin.setValue(float(payload.get("learning_rate", 0.1)))
        self._replicable_cb.setChecked(bool(payload.get("replicable", True)))
        self._lambda_slider.setValue(int(payload.get("lambda_index", 0)))
        self._depth_spin.setValue(int(payload.get("max_depth", 3)))
        self._min_split_spin.setValue(int(payload.get("min_split", 2)))
        self._subsample_spin.setValue(float(payload.get("subsample", 1.0)))
        self._update_lambda_label()
