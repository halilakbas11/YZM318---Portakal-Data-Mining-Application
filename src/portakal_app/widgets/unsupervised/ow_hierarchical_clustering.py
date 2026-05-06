from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import polars as pl

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from portakal_app.data.models import DataDomain, DatasetHandle, build_data_domain
from portakal_app.data.services.distance_matrix_service import DistanceMatrixHandle, coerce_distance_matrix
from portakal_app.models import WorkflowPayload
from portakal_app.ui import i18n
from portakal_app.ui.screens.node_screen import WorkflowNodeScreenSupport

try:
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.figure import Figure
except ImportError:  # pragma: no cover
    FigureCanvas = None
    Figure = None


@dataclass(frozen=True)
class HierarchicalClusteringResult:
    linkage_matrix: np.ndarray
    labels: tuple[int, ...]
    cluster_count: int
    method: str
    row_labels: tuple[str, ...]
    source_dataset: DatasetHandle | None


class OWHierarchicalClustering(QWidget, WorkflowNodeScreenSupport):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._init_workflow_node_support()
        self._distance_result: DistanceMatrixHandle | None = None
        self._data_subset: DatasetHandle | None = None
        self._last_result: HierarchicalClusteringResult | None = None
        self._selected_clusters: set[int] = set()
        self._output_payloads: dict[str, WorkflowPayload | None] = {}
        self._figure = Figure(figsize=(8, 5), facecolor="#f8f8f8") if Figure is not None else None
        self._canvas = FigureCanvas(self._figure) if FigureCanvas is not None and self._figure is not None else None
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(10)

        controls = QWidget(self)
        controls.setFixedWidth(280)
        controls_layout = QVBoxLayout(controls)
        controls_layout.setContentsMargins(0, 0, 0, 0)
        controls_layout.setSpacing(10)

        title = QLabel(i18n.t("Hierarchical Clustering"))
        title.setProperty("sectionTitle", True)
        controls_layout.addWidget(title)

        self._summary = QLabel(i18n.t("Connect a distance matrix to view a dendrogram."))
        self._summary.setWordWrap(True)
        self._summary.setProperty("muted", True)
        controls_layout.addWidget(self._summary)

        linkage_group = QGroupBox(i18n.t("Linkage"))
        linkage_form = QFormLayout(linkage_group)
        self._linkage_combo = QComboBox()
        for label, method in (
            ("Single", "single"),
            ("Average", "average"),
            ("Weighted", "weighted"),
            ("Complete", "complete"),
            ("Ward", "ward"),
        ):
            self._linkage_combo.addItem(i18n.t(label), method)
        linkage_form.addRow(i18n.t("Method:"), self._linkage_combo)
        controls_layout.addWidget(linkage_group)

        display_group = QGroupBox(i18n.t("Display"))
        display_form = QFormLayout(display_group)
        self._annotation_combo = QComboBox()
        self._annotation_combo.addItem(i18n.t("Labels"), "labels")
        self._annotation_combo.addItem(i18n.t("Enumeration"), "enumeration")
        self._annotation_combo.addItem(i18n.t("Show labels only for subset"), "subset")
        display_form.addRow(i18n.t("Annotation:"), self._annotation_combo)

        self._prune_spin = QSpinBox()
        self._prune_spin.setRange(0, 50)
        self._prune_spin.setValue(0)
        self._prune_spin.setSpecialValueText(i18n.t("Off"))
        display_form.addRow(i18n.t("Pruning:"), self._prune_spin)
        controls_layout.addWidget(display_group)

        selection_group = QGroupBox(i18n.t("Selection"))
        selection_form = QFormLayout(selection_group)
        self._selection_combo = QComboBox()
        self._selection_combo.addItem(i18n.t("Manual"), "manual")
        self._selection_combo.addItem(i18n.t("Height ratio"), "height")
        self._selection_combo.addItem(i18n.t("Top N"), "top_n")
        selection_form.addRow(i18n.t("Mode:"), self._selection_combo)

        self._top_n_spin = QSpinBox()
        self._top_n_spin.setRange(2, 100)
        self._top_n_spin.setValue(3)
        selection_form.addRow(i18n.t("Top N:"), self._top_n_spin)

        self._height_slider = QSlider(Qt.Orientation.Horizontal)
        self._height_slider.setRange(1, 100)
        self._height_slider.setValue(60)
        selection_form.addRow(i18n.t("Height ratio:"), self._height_slider)
        controls_layout.addWidget(selection_group)

        clusters_group = QGroupBox(i18n.t("Clusters"))
        clusters_layout = QVBoxLayout(clusters_group)
        self._cluster_table = QTableWidget(0, 2)
        self._cluster_table.setHorizontalHeaderLabels([i18n.t("Cluster"), i18n.t("Size")])
        self._cluster_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self._cluster_table.setSelectionMode(QTableWidget.SelectionMode.MultiSelection)
        clusters_layout.addWidget(self._cluster_table)
        controls_layout.addWidget(clusters_group, 1)

        footer = QHBoxLayout()
        self.cb_apply_auto = QCheckBox(i18n.t("Auto send"))
        self.cb_apply_auto.setChecked(True)
        footer.addWidget(self.cb_apply_auto)
        footer.addStretch(1)
        self._apply_button = QPushButton(i18n.t("Send Data"))
        self._apply_button.setProperty("primary", True)
        self._apply_button.clicked.connect(self._emit_outputs)
        footer.addWidget(self._apply_button)
        controls_layout.addLayout(footer)

        layout.addWidget(controls)

        visual = QWidget(self)
        visual_layout = QVBoxLayout(visual)
        visual_layout.setContentsMargins(0, 0, 0, 0)
        visual_layout.setSpacing(8)
        if self._canvas is not None:
            visual_layout.addWidget(self._canvas, 1)
        else:
            visual_layout.addWidget(QLabel(i18n.t("Matplotlib is required for the dendrogram.")))
        layout.addWidget(visual, 1)

        self._linkage_combo.currentIndexChanged.connect(self._recompute)
        self._annotation_combo.currentIndexChanged.connect(self._redraw_and_emit)
        self._prune_spin.valueChanged.connect(lambda _value: self._redraw_and_emit())
        self._selection_combo.currentIndexChanged.connect(self._handle_selection_mode_changed)
        self._top_n_spin.valueChanged.connect(lambda _value: self._recompute())
        self._height_slider.valueChanged.connect(lambda _value: self._recompute())
        self._cluster_table.itemSelectionChanged.connect(self._handle_cluster_selection_changed)
        self._handle_selection_mode_changed()

    def set_input_payload(self, payload: WorkflowPayload | None) -> None:
        if payload is None:
            self._distance_result = None
            self._data_subset = None
            self._last_result = None
            self._output_payloads = {}
            self._selected_clusters = set()
            self._cluster_table.setRowCount(0)
            self._summary.setText(i18n.t("Connect a distance matrix to view a dendrogram."))
            self._clear_figure()
            self._notify_output_changed()
            return

        if payload.port_label == "Data Subset":
            self._data_subset = payload.dataset
        else:
            self._distance_result = coerce_distance_matrix(payload.value)
            if self._distance_result.matrix.shape[0] > 1:
                self._top_n_spin.setMaximum(int(self._distance_result.matrix.shape[0]))
        self._recompute()

    def current_output_payloads(self) -> dict[str, WorkflowPayload | None] | None:
        return self._output_payloads or None

    def serialize_node_state(self) -> dict[str, object]:
        return {
            "linkage": self._linkage_combo.currentData(),
            "annotation": self._annotation_combo.currentData(),
            "pruning": self._prune_spin.value(),
            "selection_mode": self._selection_combo.currentData(),
            "top_n": self._top_n_spin.value(),
            "height_ratio": self._height_slider.value(),
            "auto_apply": self.cb_apply_auto.isChecked(),
        }

    def restore_node_state(self, payload: dict[str, object]) -> None:
        self._set_combo_data(self._linkage_combo, str(payload.get("linkage", "single")))
        self._set_combo_data(self._annotation_combo, str(payload.get("annotation", "labels")))
        self._set_combo_data(self._selection_combo, str(payload.get("selection_mode", "manual")))
        self._prune_spin.setValue(int(payload.get("pruning", 0)))
        self._top_n_spin.setValue(int(payload.get("top_n", 3)))
        self._height_slider.setValue(int(payload.get("height_ratio", 60)))
        self.cb_apply_auto.setChecked(bool(payload.get("auto_apply", True)))
        self._handle_selection_mode_changed()
        self._recompute()

    def help_text(self) -> str:
        return "Orange-style hierarchical clustering with dendrogram display and cluster selection."

    def documentation_url(self) -> str:
        return "https://orangedatamining.com/widget-catalog/unsupervised/hierarchicalclustering/"

    def _set_combo_data(self, combo: QComboBox, value: str) -> None:
        index = combo.findData(value)
        combo.setCurrentIndex(max(0, index))

    def _handle_selection_mode_changed(self) -> None:
        mode = str(self._selection_combo.currentData())
        self._top_n_spin.setEnabled(mode in {"manual", "top_n"})
        self._height_slider.setEnabled(mode == "height")
        self._recompute()

    def _recompute(self) -> None:
        self._output_payloads = {}
        self._last_result = None
        self._selected_clusters = set()
        self._cluster_table.clearSelection()
        self._cluster_table.setRowCount(0)

        if self._distance_result is None:
            self._summary.setText(i18n.t("Connect a distance matrix to view a dendrogram."))
            self._clear_figure()
            self._notify_output_changed()
            return

        try:
            from scipy.cluster.hierarchy import dendrogram, fcluster, linkage
            from scipy.spatial.distance import squareform

            matrix = _validated_square_distance_matrix(self._distance_result.matrix)
            linkage_method = str(self._linkage_combo.currentData())
            condensed = squareform(matrix, checks=False)
            linkage_matrix = linkage(condensed, method=linkage_method)

            mode = str(self._selection_combo.currentData())
            if mode == "height":
                heights = linkage_matrix[:, 2]
                max_height = float(np.max(heights)) if heights.size else 0.0
                cutoff = max_height * (self._height_slider.value() / 100.0)
                labels = fcluster(linkage_matrix, t=cutoff, criterion="distance")
            else:
                cluster_count = max(2, min(self._top_n_spin.value(), matrix.shape[0]))
                labels = fcluster(linkage_matrix, t=cluster_count, criterion="maxclust")

            self._last_result = HierarchicalClusteringResult(
                linkage_matrix=np.asarray(linkage_matrix, dtype=float),
                labels=tuple(int(value) for value in labels.tolist()),
                cluster_count=len(set(int(value) for value in labels.tolist())),
                method=linkage_method,
                row_labels=self._distance_result.row_labels,
                source_dataset=self._distance_result.source_dataset,
            )
            self._populate_cluster_table(self._last_result)
            self._redraw_dendrogram(dendrogram)
            self._summary.setText(
                i18n.tf(
                    "{rows} items, {clusters} clusters",
                    rows=len(self._last_result.labels),
                    clusters=self._last_result.cluster_count,
                )
            )
            self._select_all_clusters()
            if self.cb_apply_auto.isChecked():
                self._emit_outputs()
            else:
                self._notify_output_changed()
        except Exception as exc:
            self._summary.setText(i18n.tf("Error: {err}", err=exc))
            self._clear_figure()
            self._notify_output_changed()

    def _populate_cluster_table(self, result: HierarchicalClusteringResult) -> None:
        counts: dict[int, int] = {}
        for label in result.labels:
            counts[label] = counts.get(label, 0) + 1
        ordered = sorted(counts.items())
        self._cluster_table.setRowCount(len(ordered))
        for row_index, (cluster_id, size) in enumerate(ordered):
            item_cluster = QTableWidgetItem(f"C{cluster_id}")
            item_cluster.setData(Qt.ItemDataRole.UserRole, cluster_id)
            self._cluster_table.setItem(row_index, 0, item_cluster)
            self._cluster_table.setItem(row_index, 1, QTableWidgetItem(str(size)))
        self._cluster_table.resizeColumnsToContents()

    def _select_all_clusters(self) -> None:
        cluster_ids = set()
        for row_index in range(self._cluster_table.rowCount()):
            item = self._cluster_table.item(row_index, 0)
            if item is None:
                continue
            cluster_ids.add(int(item.data(Qt.ItemDataRole.UserRole)))
            self._cluster_table.selectRow(row_index)
        self._selected_clusters = cluster_ids

    def _handle_cluster_selection_changed(self) -> None:
        selected: set[int] = set()
        for item in self._cluster_table.selectedItems():
            if item.column() != 0:
                continue
            value = item.data(Qt.ItemDataRole.UserRole)
            if value is not None:
                selected.add(int(value))
        if selected:
            self._selected_clusters = selected
        if self.cb_apply_auto.isChecked():
            self._emit_outputs()

    def _emit_outputs(self) -> None:
        self._output_payloads = {}
        if self._last_result is None:
            self._notify_output_changed()
            return
        annotated = self._build_annotated_dataset(self._last_result, self._selected_clusters)
        selected = self._build_selected_dataset(annotated, self._selected_clusters)
        self._output_payloads = {
            "Selected Data": WorkflowPayload("Selected Data", selected) if selected is not None else None,
            "Other Data": WorkflowPayload("Other Data", self._build_other_dataset(annotated, self._selected_clusters))
            if annotated is not None else None,
        }
        self._notify_output_changed()

    def _build_annotated_dataset(
        self,
        result: HierarchicalClusteringResult,
        selected_clusters: set[int],
    ) -> DatasetHandle | None:
        dataset = result.source_dataset
        if dataset is None:
            return None
        cluster_name = _unique_column_name(dataset, "HC Cluster")
        selected_name = _unique_column_name(dataset, "Selected")
        labels = [f"C{label}" for label in result.labels]
        selected_flags = [label in selected_clusters for label in result.labels]
        frame = dataset.dataframe.with_columns(
            pl.Series(cluster_name, labels),
            pl.Series(selected_name, selected_flags),
        )
        domain = _domain_with_role_overrides(frame, dataset, {cluster_name: "meta", selected_name: "meta"})
        return replace(
            dataset,
            dataset_id=f"hierarchical-{dataset.dataset_id}",
            display_name=f"{dataset.display_name} (Hierarchical Clustering)",
            dataframe=frame,
            row_count=frame.height,
            column_count=frame.width,
            domain=domain,
            annotations={
                **dataset.annotations,
                "hierarchical_clustering": {
                    "linkage": result.method,
                    "cluster_count": result.cluster_count,
                },
            },
        )

    def _build_selected_dataset(
        self,
        annotated: DatasetHandle | None,
        selected_clusters: set[int],
    ) -> DatasetHandle | None:
        if annotated is None or self._last_result is None:
            return None
        flags = [label in selected_clusters for label in self._last_result.labels]
        frame = annotated.dataframe.filter(pl.Series(flags))
        if frame.height == 0:
            return None
        return replace(
            annotated,
            dataset_id=f"{annotated.dataset_id}-selected",
            display_name=f"{annotated.display_name} (Selected)",
            dataframe=frame,
            row_count=frame.height,
            column_count=frame.width,
            domain=build_data_domain(frame, source_domain=annotated.domain),
        )

    def _build_other_dataset(
        self,
        annotated: DatasetHandle | None,
        selected_clusters: set[int],
    ) -> DatasetHandle | None:
        if annotated is None or self._last_result is None:
            return None
        flags = [label not in selected_clusters for label in self._last_result.labels]
        frame = annotated.dataframe.filter(pl.Series(flags))
        if frame.height == 0:
            return None
        return replace(
            annotated,
            dataset_id=f"{annotated.dataset_id}-other",
            display_name=f"{annotated.display_name} (Other)",
            dataframe=frame,
            row_count=frame.height,
            column_count=frame.width,
            domain=build_data_domain(frame, source_domain=annotated.domain),
        )

    def _redraw_and_emit(self) -> None:
        if self._last_result is None:
            return
        try:
            from scipy.cluster.hierarchy import dendrogram

            self._redraw_dendrogram(dendrogram)
            if self.cb_apply_auto.isChecked():
                self._emit_outputs()
            else:
                self._notify_output_changed()
        except Exception as exc:
            self._summary.setText(i18n.tf("Error: {err}", err=exc))

    def _redraw_dendrogram(self, dendrogram) -> None:
        if self._figure is None or self._canvas is None or self._last_result is None:
            return
        self._figure.clf()
        ax = self._figure.add_subplot(111)
        labels = self._build_labels()
        prune_depth = self._prune_spin.value()
        color_threshold = self._color_threshold()
        kwargs = {
            "orientation": "right",
            "labels": labels,
            "leaf_font_size": 8,
            "ax": ax,
            "color_threshold": color_threshold,
            "above_threshold_color": "#666666",
        }
        if prune_depth > 0:
            kwargs["truncate_mode"] = "level"
            kwargs["p"] = prune_depth
        dendrogram(self._last_result.linkage_matrix, **kwargs)
        cutoff = self._cutoff_height()
        if cutoff is not None:
            ax.axvline(cutoff, color="#d9534f", linewidth=1.5, linestyle="--")
        ax.set_title(i18n.t("Dendrogram"))
        ax.set_xlabel(i18n.t("Distance"))
        ax.grid(axis="x", alpha=0.15)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        self._figure.tight_layout()
        self._canvas.draw_idle()

    def _cutoff_height(self) -> float | None:
        if self._last_result is None:
            return None
        heights = np.asarray(self._last_result.linkage_matrix[:, 2], dtype=float)
        if heights.size == 0:
            return None
        mode = str(self._selection_combo.currentData())
        if mode == "height":
            return float(np.max(heights)) * (self._height_slider.value() / 100.0)
        if mode == "top_n":
            cluster_count = max(2, min(self._top_n_spin.value(), len(self._last_result.labels)))
            return _threshold_for_top_n(heights, cluster_count)
        return None

    def _color_threshold(self) -> float:
        cutoff = self._cutoff_height()
        if cutoff is None:
            return 0.0
        return cutoff

    def _build_labels(self) -> list[str]:
        assert self._last_result is not None
        mode = str(self._annotation_combo.currentData())
        if mode == "enumeration":
            return [str(index + 1) for index in range(len(self._last_result.row_labels))]
        if mode == "subset":
            subset_indices = self._subset_indices()
            return [
                label if index in subset_indices else ""
                for index, label in enumerate(self._last_result.row_labels)
            ]
        return list(self._last_result.row_labels)

    def _subset_indices(self) -> set[int]:
        if self._distance_result is None or self._distance_result.source_dataset is None or self._data_subset is None:
            return set()
        source_rows = [tuple("" if value is None else str(value) for value in row) for row in self._distance_result.source_dataset.dataframe.rows()]
        subset_rows = {
            tuple("" if value is None else str(value) for value in row)
            for row in self._data_subset.dataframe.rows()
        }
        return {index for index, row in enumerate(source_rows) if row in subset_rows}

    def _clear_figure(self) -> None:
        if self._figure is not None and self._canvas is not None:
            self._figure.clf()
            self._canvas.draw_idle()


def _validated_square_distance_matrix(matrix: np.ndarray) -> np.ndarray:
    values = np.asarray(matrix, dtype=float)
    if values.ndim != 2 or values.shape[0] != values.shape[1]:
        raise ValueError(i18n.t("Distance Matrix must be square."))
    if values.shape[0] < 2:
        raise ValueError(i18n.t("Need at least two rows for hierarchical clustering."))
    if not np.isfinite(values).all():
        raise ValueError(i18n.t("Distance Matrix must contain only finite values."))
    values = values.copy()
    values = (values + values.T) / 2.0
    values[values < 0.0] = 0.0
    np.fill_diagonal(values, 0.0)
    return values


def _threshold_for_top_n(heights: np.ndarray, cluster_count: int) -> float:
    if heights.size == 0:
        return 0.0
    n_items = heights.size + 1
    cluster_count = max(1, min(cluster_count, n_items))
    if cluster_count <= 1:
        return float(np.max(heights))
    if cluster_count >= n_items:
        return max(0.0, float(np.min(heights)) - 1e-9)
    index = n_items - cluster_count - 1
    lower = float(heights[index])
    upper = float(heights[index + 1])
    return (lower + upper) / 2.0


def _unique_column_name(dataset: DatasetHandle, base: str) -> str:
    existing = set(dataset.dataframe.columns)
    if base not in existing:
        return base
    index = 1
    while f"{base} ({index})" in existing:
        index += 1
    return f"{base} ({index})"


def _domain_with_role_overrides(frame: pl.DataFrame, dataset: DatasetHandle, roles: dict[str, str]) -> DataDomain:
    domain = build_data_domain(frame, source_domain=dataset.domain)
    return DataDomain(columns=tuple(replace(column, role=roles.get(column.name, column.role)) for column in domain.columns))


__all__ = ["HierarchicalClusteringResult", "OWHierarchicalClustering"]
