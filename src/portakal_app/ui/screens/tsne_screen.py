from __future__ import annotations

import math

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QPainter
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QSplitter,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from portakal_app.data.models import DatasetHandle
from portakal_app.data.services.tsne_service import TSNEService
from portakal_app.models import WorkflowPayload
from portakal_app.ui.screens.node_screen import WorkflowNodeScreenSupport
from portakal_app.ui import i18n
from portakal_app.ui.shared.cards import SectionHeader

_PALETTE = [
    QColor(76, 114, 176), QColor(221, 132, 82), QColor(85, 168, 104),
    QColor(196, 78, 82), QColor(129, 114, 178), QColor(147, 120, 96),
    QColor(218, 139, 195), QColor(140, 140, 140), QColor(204, 185, 116),
    QColor(100, 181, 205),
]


class _ScatterCanvas(QWidget):
    """Simple scatter plot canvas for t-SNE visualization."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setMinimumSize(250, 250)
        self._xs: list[float] = []
        self._ys: list[float] = []
        self._dot_colors: list[QColor] = []
        self._legend: list[tuple[str, QColor]] = []

    def set_data(
        self,
        xs: list[float],
        ys: list[float],
        labels: list | None,
        symbol_size: int = 6,
        opacity: int = 200,
    ) -> None:
        self._xs = xs
        self._ys = ys
        self._symbol_size = symbol_size

        if labels is not None:
            unique = sorted(set(str(l) for l in labels))
            cat_map = {c: i for i, c in enumerate(unique)}
            self._dot_colors = [
                QColor(*_PALETTE[cat_map[str(l)] % len(_PALETTE)].getRgb()[:3], opacity)
                for l in labels
            ]
            self._legend = [(c, _PALETTE[i % len(_PALETTE)]) for i, c in enumerate(unique)]
        else:
            c = QColor(*_PALETTE[0].getRgb()[:3], opacity)
            self._dot_colors = [c] * len(xs)
            self._legend = []

        self.update()

    def clear(self) -> None:
        self._xs = []
        self._ys = []
        self._dot_colors = []
        self._legend = []
        self.update()

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        w, h = self.width(), self.height()
        painter.fillRect(0, 0, w, h, QColor(248, 248, 248))

        if not self._xs:
            painter.setPen(QColor(160, 160, 160))
            painter.drawText(0, 0, w, h, Qt.AlignmentFlag.AlignCenter, i18n.t("No data"))
            return

        legend_w = 110 if self._legend else 0
        margin = 20
        plot_w = w - legend_w - margin * 2
        plot_h = h - margin * 2

        xs, ys = self._xs, self._ys
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)
        xr = x_max - x_min or 1.0
        yr = y_max - y_min or 1.0

        def sx(x):
            return margin + (x - x_min) / xr * plot_w

        def sy(y):
            return margin + plot_h - (y - y_min) / yr * plot_h

        r = getattr(self, "_symbol_size", 6) // 2
        painter.setPen(Qt.PenStyle.NoPen)
        for x, y, color in zip(xs, ys, self._dot_colors):
            painter.setBrush(color)
            painter.drawEllipse(int(sx(x) - r), int(sy(y) - r), 2 * r, 2 * r)

        # Legend
        if self._legend:
            lx = w - legend_w + 4
            painter.setFont(painter.font())
            for i, (label, color) in enumerate(self._legend[:12]):
                ly = margin + i * 20
                painter.setBrush(color)
                painter.setPen(Qt.PenStyle.NoPen)
                painter.drawEllipse(lx, ly, 10, 10)
                painter.setPen(QColor(40, 40, 40))
                painter.drawText(lx + 14, ly + 10, label[:12])


class TSNEScreen(QWidget, WorkflowNodeScreenSupport):
    """t-SNE dimensionality reduction widget matching Orange layout."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._init_workflow_node_support()
        self._service = TSNEService()
        self._dataset_handle: DatasetHandle | None = None
        self._output_dataset: DatasetHandle | None = None

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        root.addWidget(splitter)

        # ── Left panel ────────────────────────────────────────────────
        left = QWidget()
        left.setMinimumWidth(210)
        left.setMaximumWidth(300)
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(8, 8, 8, 8)
        left_layout.setSpacing(6)

        left_layout.addWidget(SectionHeader(
            i18n.t("t-SNE"),
            i18n.t("t-Distributed Stochastic Neighbor Embedding"),
        ))

        self._info_label = QLabel(i18n.t("No data connected."))
        self._info_label.setStyleSheet("font-size: 10pt; color: gray; background: transparent;")
        left_layout.addWidget(self._info_label)

        # Preprocessing
        pre_group = QGroupBox(i18n.t("Preprocessing"))
        pre_layout = QVBoxLayout(pre_group)
        pre_layout.setContentsMargins(8, 8, 8, 8)
        pre_layout.setSpacing(4)

        self._cb_normalize = QCheckBox(i18n.t("Normalize data"))
        self._cb_normalize.setChecked(True)
        pre_layout.addWidget(self._cb_normalize)

        self._cb_pca = QCheckBox(i18n.t("Apply PCA preprocessing"))
        self._cb_pca.setChecked(False)
        pre_layout.addWidget(self._cb_pca)

        pca_row = QWidget()
        pca_hl = QHBoxLayout(pca_row)
        pca_hl.setContentsMargins(16, 0, 0, 0)
        pca_hl.setSpacing(4)
        pca_hl.addWidget(QLabel(i18n.t("PCA Components:")))
        self._slider_pca = QSlider(Qt.Orientation.Horizontal)
        self._slider_pca.setRange(2, 50)
        self._slider_pca.setValue(20)
        self._lbl_pca = QLabel("20")
        self._lbl_pca.setFixedWidth(22)
        self._slider_pca.valueChanged.connect(lambda v: self._lbl_pca.setText(str(v)))
        pca_hl.addWidget(self._slider_pca)
        pca_hl.addWidget(self._lbl_pca)
        pre_layout.addWidget(pca_row)
        self._cb_pca.stateChanged.connect(lambda s: pca_row.setEnabled(bool(s)))
        pca_row.setEnabled(False)

        left_layout.addWidget(pre_group)

        # Parameters
        param_group = QGroupBox(i18n.t("Parameters"))
        param_form = QFormLayout(param_group)
        param_form.setContentsMargins(8, 8, 8, 8)
        param_form.setSpacing(5)

        self._combo_init = QComboBox()
        self._combo_init.addItems([i18n.t("PCA"), i18n.t("Random")])
        param_form.addRow(i18n.t("Initialization:"), self._combo_init)

        self._combo_metric = QComboBox()
        self._combo_metric.addItems(["Euclidean", "Manhattan", "Cosine"])
        param_form.addRow(i18n.t("Distance metric:"), self._combo_metric)

        self._spin_perplexity = QDoubleSpinBox()
        self._spin_perplexity.setRange(1.0, 100.0)
        self._spin_perplexity.setValue(30.0)
        self._spin_perplexity.setSingleStep(1.0)
        param_form.addRow(i18n.t("Perplexity:"), self._spin_perplexity)

        self._spin_max_iter = QSpinBox()
        self._spin_max_iter.setRange(100, 5000)
        self._spin_max_iter.setValue(1000)
        self._spin_max_iter.setSingleStep(100)
        param_form.addRow(i18n.t("Max iterations:"), self._spin_max_iter)

        self._spin_seed = QSpinBox()
        self._spin_seed.setRange(0, 999999)
        self._spin_seed.setValue(42)
        param_form.addRow(i18n.t("Random state:"), self._spin_seed)

        self._btn_start = QPushButton(i18n.t("Start"))
        self._btn_start.setEnabled(False)
        self._btn_start.clicked.connect(self._apply)
        param_form.addRow(self._btn_start)

        left_layout.addWidget(param_group)

        # Attributes
        attr_group = QGroupBox(i18n.t("Attributes"))
        attr_form = QFormLayout(attr_group)
        attr_form.setContentsMargins(8, 8, 8, 8)
        attr_form.setSpacing(5)

        self._combo_color = QComboBox()
        attr_form.addRow(i18n.t("Color:"), self._combo_color)

        self._combo_shape = QComboBox()
        self._combo_shape.addItem("(Same shape)")
        attr_form.addRow(i18n.t("Shape:"), self._combo_shape)

        self._combo_size = QComboBox()
        self._combo_size.addItem("(Same size)")
        attr_form.addRow(i18n.t("Size:"), self._combo_size)

        self._combo_label = QComboBox()
        self._combo_label.addItem("(No labels)")
        attr_form.addRow(i18n.t("Label:"), self._combo_label)

        sym_row = QWidget()
        sym_hl = QHBoxLayout(sym_row)
        sym_hl.setContentsMargins(0, 0, 0, 0)
        self._slider_sym_size = QSlider(Qt.Orientation.Horizontal)
        self._slider_sym_size.setRange(1, 20)
        self._slider_sym_size.setValue(8)
        self._lbl_sym = QLabel("8")
        self._lbl_sym.setFixedWidth(20)
        self._slider_sym_size.valueChanged.connect(lambda v: (
            self._lbl_sym.setText(str(v)),
            self._update_canvas(),
        ))
        sym_hl.addWidget(self._slider_sym_size)
        sym_hl.addWidget(self._lbl_sym)
        attr_form.addRow(i18n.t("Symbol size:"), sym_row)

        opac_row = QWidget()
        opac_hl = QHBoxLayout(opac_row)
        opac_hl.setContentsMargins(0, 0, 0, 0)
        self._slider_opacity = QSlider(Qt.Orientation.Horizontal)
        self._slider_opacity.setRange(10, 255)
        self._slider_opacity.setValue(200)
        self._lbl_opacity = QLabel("200")
        self._lbl_opacity.setFixedWidth(24)
        self._slider_opacity.valueChanged.connect(lambda v: (
            self._lbl_opacity.setText(str(v)),
            self._update_canvas(),
        ))
        opac_hl.addWidget(self._slider_opacity)
        opac_hl.addWidget(self._lbl_opacity)
        attr_form.addRow(i18n.t("Opacity:"), opac_row)

        self._combo_color.currentTextChanged.connect(self._update_canvas)

        left_layout.addWidget(attr_group)

        self.cb_apply_auto = QCheckBox(i18n.t("Send Automatically"))
        self.cb_apply_auto.setChecked(False)
        self.cb_apply_auto.stateChanged.connect(self._schedule_auto_apply)
        left_layout.addWidget(self.cb_apply_auto)

        self._status_label = QLabel("")
        self._status_label.setStyleSheet("color: gray; background: transparent; font-size: 9pt;")
        self._status_label.setWordWrap(True)
        left_layout.addWidget(self._status_label)

        left_layout.addStretch(1)
        splitter.addWidget(left)

        # ── Right panel: scatter canvas ───────────────────────────────
        self._canvas = _ScatterCanvas()
        splitter.addWidget(self._canvas)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([250, 450])

        # Wire auto-apply signals
        for w in (self._spin_perplexity, self._spin_max_iter, self._spin_seed):
            w.valueChanged.connect(self._schedule_auto_apply)
        for w in (self._combo_init, self._combo_metric):
            w.currentIndexChanged.connect(self._schedule_auto_apply)
        for w in (self._cb_normalize, self._cb_pca):
            w.stateChanged.connect(self._schedule_auto_apply)
        self._slider_pca.valueChanged.connect(self._schedule_auto_apply)

    # ── WorkflowNodeScreenSupport ──────────────────────────────────────

    def set_input_payload(self, payload: WorkflowPayload | None) -> None:
        if payload is None:
            self._dataset_handle = None
            self._output_dataset = None
            self._info_label.setText(i18n.t("No data connected."))
            self._btn_start.setEnabled(False)
            self._canvas.clear()
            self._combo_color.clear()
            self._notify_output_changed()
            return

        self._dataset_handle = payload.dataset
        if self._dataset_handle:
            name = self._dataset_handle.display_name
            rows = self._dataset_handle.row_count
            cols = self._dataset_handle.column_count
            self._info_label.setText(f"{name}  ({rows} rows × {cols} cols)")

            # Populate color dropdown from input columns
            prev = self._combo_color.currentText()
            self._combo_color.blockSignals(True)
            self._combo_color.clear()
            self._combo_color.addItem(i18n.t("(None)"))
            for col in self._dataset_handle.dataframe.columns:
                self._combo_color.addItem(col)
            idx = self._combo_color.findText(prev)
            if idx >= 0:
                self._combo_color.setCurrentIndex(idx)
            elif self._combo_color.count() > 1:
                # Default to last column (often the class/target)
                self._combo_color.setCurrentIndex(self._combo_color.count() - 1)
            self._combo_color.blockSignals(False)

        self._btn_start.setEnabled(True)
        self._auto_apply_on_input()

    def current_output_dataset(self) -> DatasetHandle | None:
        return self._output_dataset

    def serialize_node_state(self) -> dict[str, object]:
        return {
            "normalize": self._cb_normalize.isChecked(),
            "apply_pca": self._cb_pca.isChecked(),
            "pca_components": self._slider_pca.value(),
            "init": self._combo_init.currentText(),
            "metric": self._combo_metric.currentText().lower(),
            "perplexity": self._spin_perplexity.value(),
            "max_iter": self._spin_max_iter.value(),
            "random_state": self._spin_seed.value(),
            "color_by": self._combo_color.currentText(),
            "symbol_size": self._slider_sym_size.value(),
            "opacity": self._slider_opacity.value(),
            "auto_apply": self.cb_apply_auto.isChecked(),
        }

    def restore_node_state(self, payload: dict[str, object]) -> None:
        self._cb_normalize.setChecked(bool(payload.get("normalize", True)))
        self._cb_pca.setChecked(bool(payload.get("apply_pca", False)))
        self._slider_pca.setValue(int(payload.get("pca_components", 20)))
        init = str(payload.get("init", "PCA"))
        idx = self._combo_init.findText(init)
        if idx >= 0:
            self._combo_init.setCurrentIndex(idx)
        metric = str(payload.get("metric", "euclidean")).capitalize()
        idx = self._combo_metric.findText(metric)
        if idx >= 0:
            self._combo_metric.setCurrentIndex(idx)
        self._spin_perplexity.setValue(float(payload.get("perplexity", 30.0)))
        self._spin_max_iter.setValue(int(payload.get("max_iter", 1000)))
        self._spin_seed.setValue(int(payload.get("random_state", 42)))
        self._slider_sym_size.setValue(int(payload.get("symbol_size", 8)))
        self._slider_opacity.setValue(int(payload.get("opacity", 200)))
        self.cb_apply_auto.setChecked(bool(payload.get("auto_apply", False)))
        color_by = str(payload.get("color_by", ""))
        if color_by:
            idx = self._combo_color.findText(color_by)
            if idx >= 0:
                self._combo_color.setCurrentIndex(idx)

    # ── Internal ───────────────────────────────────────────────────────

    def _apply(self) -> None:
        if self._dataset_handle is None:
            return
        try:
            self._status_label.setText(i18n.t("Running t-SNE…"))
            init_text = self._combo_init.currentText().lower()
            metric_text = self._combo_metric.currentText().lower()
            self._output_dataset = self._service.run(
                self._dataset_handle,
                n_components=2,
                perplexity=self._spin_perplexity.value(),
                max_iter=self._spin_max_iter.value(),
                random_state=self._spin_seed.value(),
                normalize=self._cb_normalize.isChecked(),
                init=init_text,
                metric=metric_text,
                apply_pca=self._cb_pca.isChecked(),
                pca_components=self._slider_pca.value(),
            )
            self._update_canvas()
            self._status_label.setText(
                f"✓  {self._output_dataset.row_count} rows"
            )
        except Exception as exc:
            self._output_dataset = None
            self._canvas.clear()
            self._status_label.setText(f"✗  {exc}")
        finally:
            self._notify_output_changed()

    def _update_canvas(self) -> None:
        if self._output_dataset is None:
            return
        df = self._output_dataset.dataframe
        if "tsne-1" not in df.columns or "tsne-2" not in df.columns:
            return
        xs = df["tsne-1"].to_list()
        ys = df["tsne-2"].to_list()
        color_col = self._combo_color.currentText()
        labels = None
        if color_col and color_col != i18n.t("(None)") and color_col in df.columns:
            labels = df[color_col].to_list()
        self._canvas.set_data(
            xs, ys, labels,
            symbol_size=self._slider_sym_size.value() * 2,
            opacity=self._slider_opacity.value(),
        )
