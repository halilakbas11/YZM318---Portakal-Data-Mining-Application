from __future__ import annotations

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from portakal_app.data.services.distance_matrix_service import coerce_distance_matrix
from portakal_app.ui.screens.node_screen import WorkflowNodeScreenSupport

try:
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.figure import Figure
except ImportError:  # pragma: no cover - dependency check at runtime
    FigureCanvas = None
    Figure = None

try:
    from scipy.cluster.hierarchy import dendrogram, linkage, optimal_leaf_ordering
    from scipy.spatial.distance import squareform
except ImportError:  # pragma: no cover - dependency check at runtime
    dendrogram = None
    linkage = None
    optimal_leaf_ordering = None
    squareform = None


class DistanceMapScreen(QWidget, WorkflowNodeScreenSupport):
    COLORMAPS = ["Greys", "Blues", "viridis", "plasma", "inferno", "magma", "coolwarm", "RdYlGn_r", "Oranges", "YlOrRd"]
    CMAP_LABELS = ["Dim gray (Greys)", "Blues", "Viridis", "Plasma", "Inferno", "Magma", "Coolwarm", "RdYlGn_r", "Oranges", "YlOrRd"]
    SORT_OPTIONS = ["None", "Clustering", "Clustering with ordered leaves"]
    ANNOTATION_OPTIONS = ["None", "Enumeration"]

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._init_workflow_node_support()
        self._distances: np.ndarray | None = None
        self._row_labels: list[str] = []
        self._figure = Figure(figsize=(8, 6), facecolor="#f8f8f8") if Figure is not None else None
        self._canvas = FigureCanvas(self._figure) if FigureCanvas is not None and self._figure is not None else None
        self._build_ui()

    def _build_ui(self) -> None:
        root = QHBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)

        left_panel = QWidget()
        left_panel.setFixedWidth(220)
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(12, 12, 8, 12)

        self._sort_combo = QComboBox()
        self._sort_combo.addItems(self.SORT_OPTIONS)
        self._sort_combo.currentIndexChanged.connect(self._redraw)
        left_layout.addWidget(QLabel("Element Sorting"))
        left_layout.addWidget(self._sort_combo)

        self._cmap_combo = QComboBox()
        self._cmap_combo.addItems(self.CMAP_LABELS)
        self._cmap_combo.currentIndexChanged.connect(self._redraw)
        left_layout.addWidget(QLabel("Colors"))
        left_layout.addWidget(self._cmap_combo)

        self._range_slider = QSlider(Qt.Orientation.Horizontal)
        self._range_slider.setRange(0, 100)
        self._range_slider.setValue(100)
        self._range_slider.valueChanged.connect(self._redraw)
        left_layout.addWidget(QLabel("Range"))
        left_layout.addWidget(self._range_slider)

        self._annotation_combo = QComboBox()
        self._annotation_combo.addItems(self.ANNOTATION_OPTIONS)
        self._annotation_combo.currentIndexChanged.connect(self._redraw)
        left_layout.addWidget(QLabel("Annotations"))
        left_layout.addWidget(self._annotation_combo)
        left_layout.addStretch(1)

        save_button = QPushButton("Save Image")
        save_button.setProperty("primary", True)
        save_button.clicked.connect(self._save_image)
        left_layout.addWidget(save_button)
        root.addWidget(left_panel)

        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        if self._canvas is not None:
            right_layout.addWidget(self._canvas, 1)
        self._info_label = QLabel("Distance matrix input is waiting.")
        self._info_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        right_layout.addWidget(self._info_label)
        root.addWidget(right_panel, 1)

    def set_input_payload(self, payload) -> None:
        if payload is None:
            self._distances = None
            self._info_label.setText("Distance matrix input is waiting.")
            self._clear_figure()
            return
        try:
            handle = coerce_distance_matrix(payload.value)
            self._distances = np.array(handle.matrix, dtype=float)
            self._row_labels = list(handle.row_labels)
            self._redraw()
        except Exception as exc:
            self._distances = None
            self._row_labels = []
            self._info_label.setText(f"Error: {exc}")
            self._clear_figure()

    def help_text(self) -> str:
        return "Visualize a distance matrix as a heat map with optional clustering."

    def documentation_url(self) -> str:
        return "https://orangedatamining.com/widget-catalog/unsupervised/distancemap/"

    def _clear_figure(self) -> None:
        if self._figure is not None and self._canvas is not None:
            self._figure.clf()
            self._canvas.draw_idle()

    def _save_image(self) -> None:
        if self._figure is None or self._distances is None:
            return
        path, _selected = QFileDialog.getSaveFileName(
            self,
            "Save Distance Map",
            "distance_map.png",
            "PNG Image (*.png);;JPEG Image (*.jpg);;All Files (*)",
        )
        if not path:
            return
        try:
            self._figure.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
            self._info_label.setText(f"Saved: {path.split('/')[-1]}")
        except Exception as exc:
            self._info_label.setText(f"Save error: {exc}")

    def _redraw(self) -> None:
        if self._distances is None or self._figure is None or self._canvas is None:
            return
        if linkage is None or dendrogram is None or squareform is None:
            self._info_label.setText("SciPy and Matplotlib are required for Distance Map.")
            return

        try:
            distances = self._distances.copy()
            size = distances.shape[0]
            sort_index = self._sort_combo.currentIndex()
            order = list(range(size))
            self._figure.clf()
            vmax = distances.max() * (self._range_slider.value() / 100.0) if distances.max() > 0 else 1.0
            cmap = self.COLORMAPS[self._cmap_combo.currentIndex()]

            if sort_index > 0 and size > 1:
                sym = (distances + distances.T) / 2.0
                np.fill_diagonal(sym, 0.0)
                np.clip(sym, 0, None, out=sym)
                condensed = squareform(sym, checks=False)
                linkage_matrix = linkage(condensed, method="average")
                if sort_index == 2 and optimal_leaf_ordering is not None:
                    try:
                        linkage_matrix = optimal_leaf_ordering(linkage_matrix, condensed)
                    except Exception:
                        pass

                ax_left = self._figure.add_axes([0.05, 0.15, 0.15, 0.60])
                ax_top = self._figure.add_axes([0.21, 0.76, 0.60, 0.15])
                ax_heat = self._figure.add_axes([0.21, 0.15, 0.60, 0.60])
                cax = self._figure.add_axes([0.83, 0.15, 0.03, 0.60])
                dendro_top = dendrogram(linkage_matrix, orientation="top", ax=ax_top, color_threshold=0, above_threshold_color="black")
                dendrogram(linkage_matrix, orientation="left", ax=ax_left, color_threshold=0, above_threshold_color="black")
                ax_top.axis("off")
                ax_left.axis("off")
                ax_left.invert_yaxis()
                order = dendro_top["leaves"]
            else:
                ax_heat = self._figure.add_axes([0.15, 0.15, 0.65, 0.75])
                cax = self._figure.add_axes([0.83, 0.15, 0.03, 0.75])

            distances = distances[np.ix_(order, order)]
            base_labels = self._row_labels or [str(index + 1) for index in range(size)]
            labels = [base_labels[order[i]] for i in range(size)]
            image = ax_heat.imshow(distances, cmap=cmap, aspect="auto", interpolation="nearest", vmin=0.0, vmax=vmax)
            self._figure.colorbar(image, cax=cax)
            cax.set_ylabel("Distance", fontsize=9)

            step = max(1, size // 20)
            ticks = list(range(0, size, step))
            tick_labels = [labels[i] for i in ticks]
            ax_heat.set_xticks(ticks)
            ax_heat.set_yticks(ticks)
            if self._annotation_combo.currentIndex() == 0:
                ax_heat.set_xticklabels([])
                ax_heat.set_yticklabels([])
            else:
                ax_heat.set_xticklabels(tick_labels, rotation=45, ha="right", fontsize=7)
                ax_heat.set_yticklabels(tick_labels, fontsize=7)

            ax_heat.set_title(f"Distance Map ({size}x{size}) - {self.SORT_OPTIONS[sort_index]}", fontsize=10, fontweight="bold")
            self._canvas.draw_idle()
            self._info_label.setText(f"{size}x{size} | min={distances.min():.4f} max={self._distances.max():.4f}")
        except Exception as exc:
            self._info_label.setText(f"Draw error: {exc}")
