from __future__ import annotations

import math

from PySide6.QtCore import Qt, QPointF
from PySide6.QtGui import QColor, QPainter, QPen, QPolygonF
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
from portakal_app.data.services.som_service import SOMService
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


class _SOMCanvas(QWidget):
    """SOM grid visualization canvas."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setMinimumSize(250, 250)
        self._grid_rows = 5
        self._grid_cols = 5
        self._counts: dict[tuple[int, int], int] = {}
        self._cell_colors: dict[tuple[int, int], QColor] = {}
        self._grid_type = "hexagonal"
        self._size_by_count = True

    def set_data(
        self,
        grid_rows: int,
        grid_cols: int,
        bmu_rows: list[int],
        bmu_cols: list[int],
        labels: list | None,
        grid_type: str = "hexagonal",
        size_by_count: bool = True,
    ) -> None:
        self._grid_rows = grid_rows
        self._grid_cols = grid_cols
        self._grid_type = grid_type
        self._size_by_count = size_by_count

        counts: dict[tuple[int, int], int] = {}
        label_groups: dict[tuple[int, int], dict[str, int]] = {}

        for r, c, lbl in zip(bmu_rows, bmu_cols, labels if labels is not None else [None] * len(bmu_rows)):
            key = (r, c)
            counts[key] = counts.get(key, 0) + 1
            if lbl is not None:
                lg = label_groups.setdefault(key, {})
                s = str(lbl)
                lg[s] = lg.get(s, 0) + 1

        self._counts = counts

        all_labels: list[str] = sorted({s for lg in label_groups.values() for s in lg})
        label_to_color = {l: _PALETTE[i % len(_PALETTE)] for i, l in enumerate(all_labels)}

        self._cell_colors = {}
        for key, lg in label_groups.items():
            most_common = max(lg, key=lg.get)
            self._cell_colors[key] = label_to_color.get(most_common, _PALETTE[0])

        self.update()

    def clear(self) -> None:
        self._counts = {}
        self._cell_colors = {}
        self.update()

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        w, h = self.width(), self.height()
        painter.fillRect(0, 0, w, h, QColor(248, 248, 248))

        rows, cols = self._grid_rows, self._grid_cols
        margin = 20
        max_count = max(self._counts.values()) if self._counts else 1

        if self._grid_type == "hexagonal":
            # pointy-top hexagon layout
            hex_w = (w - 2 * margin) / (cols + 0.5)
            hex_h = (h - 2 * margin) / (rows * 0.75 + 0.25)
            hex_size = min(hex_w, hex_h) * 0.55

            for r in range(rows):
                for c in range(cols):
                    offset_x = hex_size if r % 2 == 1 else 0
                    cx = margin + c * hex_w + hex_w / 2 + offset_x
                    cy = margin + r * (hex_h * 0.75) + hex_h / 2

                    key = (r, c)
                    count = self._counts.get(key, 0)
                    if self._size_by_count and max_count > 0:
                        factor = 0.25 + 0.75 * (count / max_count)
                    else:
                        factor = 1.0
                    draw_size = hex_size * factor

                    color = self._cell_colors.get(key, QColor(210, 210, 210))

                    pts = []
                    for i in range(6):
                        angle = math.pi / 3 * i + math.pi / 6
                        pts.append(QPointF(
                            cx + draw_size * math.cos(angle),
                            cy + draw_size * math.sin(angle),
                        ))
                    polygon = QPolygonF(pts)
                    painter.setBrush(color)
                    painter.setPen(QPen(QColor(255, 255, 255), 1))
                    painter.drawPolygon(polygon)
        else:
            # Square grid: draw circles
            cell_size = min((w - 2 * margin) / cols, (h - 2 * margin) / rows)
            for r in range(rows):
                for c in range(cols):
                    cx = margin + c * cell_size + cell_size / 2
                    cy = margin + r * cell_size + cell_size / 2
                    key = (r, c)
                    count = self._counts.get(key, 0)
                    if self._size_by_count and max_count > 0:
                        factor = 0.25 + 0.75 * (count / max_count)
                    else:
                        factor = 1.0
                    radius = cell_size / 2 * factor * 0.85
                    color = self._cell_colors.get(key, QColor(210, 210, 210))
                    painter.setBrush(color)
                    painter.setPen(QPen(QColor(255, 255, 255), 1))
                    painter.drawEllipse(
                        int(cx - radius), int(cy - radius),
                        int(radius * 2), int(radius * 2),
                    )


class SOMScreen(QWidget, WorkflowNodeScreenSupport):
    """Self-Organizing Map (Kohonen Map) widget matching Orange layout."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._init_workflow_node_support()
        self._service = SOMService()
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
            i18n.t("SOM"),
            i18n.t("Self-Organizing Map (Kohonen Map)"),
        ))

        self._info_label = QLabel(i18n.t("No data connected."))
        self._info_label.setStyleSheet("font-size: 10pt; color: gray; background: transparent;")
        left_layout.addWidget(self._info_label)

        # SOM Group
        som_group = QGroupBox(i18n.t("SOM"))
        som_form = QFormLayout(som_group)
        som_form.setContentsMargins(8, 8, 8, 8)
        som_form.setSpacing(5)

        self._combo_grid_type = QComboBox()
        self._combo_grid_type.addItems([i18n.t("Hexagonal grid"), i18n.t("Square grid")])
        som_form.addRow(i18n.t("Grid type:"), self._combo_grid_type)

        self._cb_auto_dim = QCheckBox(i18n.t("Set dimensions automatically"))
        self._cb_auto_dim.setChecked(True)
        som_form.addRow(self._cb_auto_dim)

        dim_row = QWidget()
        dim_hl = QHBoxLayout(dim_row)
        dim_hl.setContentsMargins(0, 0, 0, 0)
        dim_hl.setSpacing(4)
        self._spin_rows = QSpinBox()
        self._spin_rows.setRange(2, 50)
        self._spin_rows.setValue(8)
        self._spin_cols = QSpinBox()
        self._spin_cols.setRange(2, 50)
        self._spin_cols.setValue(8)
        dim_hl.addWidget(self._spin_rows)
        dim_hl.addWidget(QLabel("×"))
        dim_hl.addWidget(self._spin_cols)
        som_form.addRow(i18n.t("Grid size:"), dim_row)
        self._cb_auto_dim.stateChanged.connect(lambda s: dim_row.setEnabled(not bool(s)))
        dim_row.setEnabled(False)

        self._combo_init = QComboBox()
        self._combo_init.addItems([i18n.t("Initialize with PCA"), i18n.t("Random initialization")])
        som_form.addRow(i18n.t("Initialization:"), self._combo_init)

        # Training params (collapsed into SOM group)
        self._spin_iter = QSpinBox()
        self._spin_iter.setRange(100, 50000)
        self._spin_iter.setValue(1000)
        self._spin_iter.setSingleStep(500)
        som_form.addRow(i18n.t("Iterations:"), self._spin_iter)

        self._spin_lr = QDoubleSpinBox()
        self._spin_lr.setRange(0.01, 1.0)
        self._spin_lr.setValue(0.5)
        self._spin_lr.setSingleStep(0.05)
        self._spin_lr.setDecimals(2)
        som_form.addRow(i18n.t("Learning rate:"), self._spin_lr)

        self._spin_sigma = QDoubleSpinBox()
        self._spin_sigma.setRange(0.1, 20.0)
        self._spin_sigma.setValue(1.0)
        self._spin_sigma.setSingleStep(0.1)
        self._spin_sigma.setDecimals(2)
        som_form.addRow(i18n.t("Sigma:"), self._spin_sigma)

        self._spin_seed = QSpinBox()
        self._spin_seed.setRange(0, 999999)
        self._spin_seed.setValue(42)
        som_form.addRow(i18n.t("Random state:"), self._spin_seed)

        self._btn_start = QPushButton(i18n.t("Start"))
        self._btn_start.setEnabled(False)
        self._btn_start.clicked.connect(self._apply)
        som_form.addRow(self._btn_start)

        left_layout.addWidget(som_group)

        # Color Group
        color_group = QGroupBox(i18n.t("Color"))
        color_form = QFormLayout(color_group)
        color_form.setContentsMargins(8, 8, 8, 8)
        color_form.setSpacing(5)

        self._combo_color = QComboBox()
        color_form.addRow(i18n.t("Color by:"), self._combo_color)

        self._cb_size_by_count = QCheckBox(i18n.t("Size by number of instances"))
        self._cb_size_by_count.setChecked(True)
        color_form.addRow(self._cb_size_by_count)

        self._combo_color.currentTextChanged.connect(self._update_canvas)
        self._cb_size_by_count.stateChanged.connect(lambda _: self._update_canvas())

        left_layout.addWidget(color_group)

        self.cb_apply_auto = QCheckBox(i18n.t("Apply Automatically"))
        self.cb_apply_auto.setChecked(False)
        self.cb_apply_auto.stateChanged.connect(self._schedule_auto_apply)
        left_layout.addWidget(self.cb_apply_auto)

        self._status_label = QLabel("")
        self._status_label.setStyleSheet("color: gray; background: transparent; font-size: 9pt;")
        self._status_label.setWordWrap(True)
        left_layout.addWidget(self._status_label)

        left_layout.addStretch(1)
        splitter.addWidget(left)

        # ── Right panel: SOM canvas ───────────────────────────────────
        self._canvas = _SOMCanvas()
        splitter.addWidget(self._canvas)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([250, 450])

        # Wire auto-apply
        for w in (self._spin_rows, self._spin_cols, self._spin_iter, self._spin_seed):
            w.valueChanged.connect(self._schedule_auto_apply)
        for w in (self._spin_lr, self._spin_sigma):
            w.valueChanged.connect(self._schedule_auto_apply)
        for w in (self._combo_grid_type, self._combo_init):
            w.currentIndexChanged.connect(self._schedule_auto_apply)
        for w in (self._cb_auto_dim,):
            w.stateChanged.connect(self._schedule_auto_apply)

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
                self._combo_color.setCurrentIndex(self._combo_color.count() - 1)
            self._combo_color.blockSignals(False)

        self._btn_start.setEnabled(True)
        self._auto_apply_on_input()

    def current_output_dataset(self) -> DatasetHandle | None:
        return self._output_dataset

    def serialize_node_state(self) -> dict[str, object]:
        return {
            "grid_type": self._combo_grid_type.currentText(),
            "auto_dim": self._cb_auto_dim.isChecked(),
            "grid_rows": self._spin_rows.value(),
            "grid_cols": self._spin_cols.value(),
            "init": self._combo_init.currentText(),
            "n_iter": self._spin_iter.value(),
            "learning_rate": self._spin_lr.value(),
            "sigma": self._spin_sigma.value(),
            "random_state": self._spin_seed.value(),
            "color_by": self._combo_color.currentText(),
            "size_by_count": self._cb_size_by_count.isChecked(),
            "auto_apply": self.cb_apply_auto.isChecked(),
        }

    def restore_node_state(self, payload: dict[str, object]) -> None:
        gt = str(payload.get("grid_type", i18n.t("Hexagonal grid")))
        idx = self._combo_grid_type.findText(gt)
        if idx >= 0:
            self._combo_grid_type.setCurrentIndex(idx)
        self._cb_auto_dim.setChecked(bool(payload.get("auto_dim", True)))
        self._spin_rows.setValue(int(payload.get("grid_rows", 8)))
        self._spin_cols.setValue(int(payload.get("grid_cols", 8)))
        init = str(payload.get("init", i18n.t("Initialize with PCA")))
        idx = self._combo_init.findText(init)
        if idx >= 0:
            self._combo_init.setCurrentIndex(idx)
        self._spin_iter.setValue(int(payload.get("n_iter", 1000)))
        self._spin_lr.setValue(float(payload.get("learning_rate", 0.5)))
        self._spin_sigma.setValue(float(payload.get("sigma", 1.0)))
        self._spin_seed.setValue(int(payload.get("random_state", 42)))
        self._cb_size_by_count.setChecked(bool(payload.get("size_by_count", True)))
        self.cb_apply_auto.setChecked(bool(payload.get("auto_apply", False)))
        color_by = str(payload.get("color_by", ""))
        if color_by:
            idx = self._combo_color.findText(color_by)
            if idx >= 0:
                self._combo_color.setCurrentIndex(idx)

    # ── Internal ───────────────────────────────────────────────────────

    def _get_grid_size(self) -> tuple[int, int]:
        if self._cb_auto_dim.isChecked() and self._dataset_handle:
            n = self._dataset_handle.row_count
            side = max(2, round(5 * (n ** 0.25)))
            return side, side
        return self._spin_rows.value(), self._spin_cols.value()

    def _apply(self) -> None:
        if self._dataset_handle is None:
            return
        grid_rows, grid_cols = self._get_grid_size()
        use_pca = self._combo_init.currentIndex() == 0
        try:
            self._status_label.setText(
                i18n.t(f"Training SOM {grid_rows}×{grid_cols}…")
            )
            self._output_dataset = self._service.run(
                self._dataset_handle,
                grid_rows=grid_rows,
                grid_cols=grid_cols,
                sigma=self._spin_sigma.value(),
                learning_rate=self._spin_lr.value(),
                n_iter=self._spin_iter.value(),
                random_state=self._spin_seed.value(),
                normalize=True,
                init="pca" if use_pca else "random",
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
        if "som-row" not in df.columns or "som-col" not in df.columns:
            return

        bmu_rows = df["som-row"].to_list()
        bmu_cols = df["som-col"].to_list()

        color_col = self._combo_color.currentText()
        labels = None
        if color_col and color_col != i18n.t("(None)") and color_col in df.columns:
            labels = df[color_col].to_list()

        grid_rows, grid_cols = self._get_grid_size()
        grid_type = (
            "hexagonal"
            if self._combo_grid_type.currentIndex() == 0
            else "square"
        )
        self._canvas.set_data(
            grid_rows, grid_cols, bmu_rows, bmu_cols, labels,
            grid_type=grid_type,
            size_by_count=self._cb_size_by_count.isChecked(),
        )
