from __future__ import annotations

from typing import Any

from PySide6.QtCore import Qt
from PySide6.QtGui import QStandardItem, QStandardItemModel
from PySide6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QPushButton,
    QRadioButton,
    QSlider,
    QSpinBox,
    QSplitter,
    QTreeView,
    QVBoxLayout,
    QWidget,
    QSizePolicy
)

from portakal_app.data.models import DatasetHandle
from portakal_app.data.services.associate_service import AssociateService
from portakal_app.models import WorkflowPayload
from portakal_app.ui import i18n
from portakal_app.ui.screens.node_screen import WorkflowNodeScreenSupport


class FrequentItemsetsScreen(QWidget, WorkflowNodeScreenSupport):
    """Find frequent itemsets in the data."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._init_workflow_node_support()
        
        self._dataset: DatasetHandle | None = None
        self._itemsets_df: Any = None
        self._filtered_df: Any = None
        self._svc = AssociateService()

        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(8, 8, 8, 8)

        # Right panel: TreeView (Create early so buttons can connect)
        right_panel_widget = QWidget()
        right_panel_widget.setMinimumWidth(500)
        right_panel = QVBoxLayout(right_panel_widget)
        right_panel.setContentsMargins(0, 0, 0, 0)
        self._tree = QTreeView()
        
        # Enable scrollbars
        self._tree.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self._tree.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self._tree.setHorizontalScrollMode(QTreeView.ScrollMode.ScrollPerPixel)
        
        self._tree_model = QStandardItemModel()
        self._tree_model.setHorizontalHeaderLabels(["Itemsets", "Support", "%"])
        self._tree.setModel(self._tree_model)
        self._tree.setAlternatingRowColors(True)
        self._tree.setEditTriggers(QTreeView.EditTrigger.NoEditTriggers)
        
        # Allow interactive resizing of columns so horizontal scrolling can happen
        self._tree.header().setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        self._tree.header().setMinimumSectionSize(80)
        self._tree.header().setStretchLastSection(False)
        
        right_panel.addWidget(self._tree)

        # Left panel: Settings
        left_panel = QVBoxLayout()
        left_panel.setSpacing(12)
        left_panel_widget = QWidget()
        left_panel_widget.setLayout(left_panel)
        left_panel_widget.setMinimumWidth(280)
        left_panel_widget.setMaximumWidth(350)

        # Info Box
        info_box = QGroupBox(i18n.t("Info"))
        info_layout = QVBoxLayout(info_box)
        
        self._lbl_num_itemsets = QLabel(i18n.t("Number of itemsets:"))
        self._lbl_sel_itemsets = QLabel(i18n.t("Selected itemsets:"))
        self._lbl_sel_examples = QLabel(i18n.t("Selected examples:"))
        info_layout.addWidget(self._lbl_num_itemsets)
        info_layout.addWidget(self._lbl_sel_itemsets)
        info_layout.addWidget(self._lbl_sel_examples)
        
        expand_layout = QHBoxLayout()
        self._btn_expand_all = QPushButton(i18n.t("Expand all"))
        self._btn_collapse_all = QPushButton(i18n.t("Collapse all"))
        self._btn_expand_all.clicked.connect(self._tree.expandAll)
        self._btn_collapse_all.clicked.connect(self._tree.collapseAll)
        expand_layout.addWidget(self._btn_expand_all)
        expand_layout.addWidget(self._btn_collapse_all)
        info_layout.addLayout(expand_layout)
        left_panel.addWidget(info_box)

        # Find itemsets Box
        find_box = QGroupBox(i18n.t("Find itemsets"))
        find_layout = QVBoxLayout(find_box)
        
        support_layout = QHBoxLayout()
        support_layout.addWidget(QLabel(i18n.t("Minimal support:")))
        self._slider_min_supp = QSlider(Qt.Orientation.Horizontal)
        self._slider_min_supp.setRange(1, 100)
        self._slider_min_supp.setValue(5)
        self._spin_min_supp = QSpinBox()
        self._spin_min_supp.setRange(1, 100)
        self._spin_min_supp.setValue(5)
        self._spin_min_supp.setSuffix("%")
        self._slider_min_supp.valueChanged.connect(self._spin_min_supp.setValue)
        self._spin_min_supp.valueChanged.connect(self._slider_min_supp.setValue)
        support_layout.addWidget(self._slider_min_supp)
        support_layout.addWidget(self._spin_min_supp)
        find_layout.addLayout(support_layout)
        
        max_itemsets_layout = QHBoxLayout()
        max_itemsets_layout.addWidget(QLabel(i18n.t("Max. number of itemsets:")))
        self._slider_max_items = QSlider(Qt.Orientation.Horizontal)
        self._slider_max_items.setRange(1, 10000)
        self._slider_max_items.setValue(10000)
        self._spin_max_items = QSpinBox()
        self._spin_max_items.setRange(1, 10000)
        self._spin_max_items.setValue(10000)
        self._slider_max_items.valueChanged.connect(self._spin_max_items.setValue)
        self._spin_max_items.valueChanged.connect(self._slider_max_items.setValue)
        max_itemsets_layout.addWidget(self._slider_max_items)
        max_itemsets_layout.addWidget(self._spin_max_items)
        find_layout.addLayout(max_itemsets_layout)
        
        find_btn_layout = QHBoxLayout()
        self._chk_auto_find = QCheckBox()
        self._btn_find = QPushButton(i18n.t("Find Itemsets"))
        self._btn_find.clicked.connect(self._run)
        find_btn_layout.addWidget(self._chk_auto_find)
        find_btn_layout.addWidget(self._btn_find)
        find_layout.addLayout(find_btn_layout)
        
        left_panel.addWidget(find_box)

        # Filter itemsets Box
        filter_box = QGroupBox(i18n.t("Filter itemsets"))
        filter_layout = QVBoxLayout(filter_box)
        
        contains_layout = QHBoxLayout()
        contains_layout.addWidget(QLabel(i18n.t("Contains:")))
        self._txt_contains = QLineEdit()
        contains_layout.addWidget(self._txt_contains)
        filter_layout.addLayout(contains_layout)
        
        items_layout = QHBoxLayout()
        items_layout.addWidget(QLabel(i18n.t("Min. items:")))
        self._spin_min_filter = QSpinBox()
        self._spin_min_filter.setRange(1, 999)
        self._spin_min_filter.setValue(2)
        items_layout.addWidget(self._spin_min_filter)
        items_layout.addWidget(QLabel(i18n.t("Max. items:")))
        self._spin_max_filter = QSpinBox()
        self._spin_max_filter.setRange(1, 999)
        self._spin_max_filter.setValue(999)
        items_layout.addWidget(self._spin_max_filter)
        filter_layout.addLayout(items_layout)
        
        self._chk_apply_filters = QCheckBox(i18n.t("Apply these filters in search"))
        self._chk_apply_filters.setChecked(True)
        filter_layout.addWidget(self._chk_apply_filters)
        
        left_panel.addWidget(filter_box)
        
        left_panel.addStretch(1)
        
        # Send Selection Automatically
        send_layout = QHBoxLayout()
        self._chk_auto_send = QCheckBox()
        self._chk_auto_send.setChecked(True)
        send_layout.addWidget(self._chk_auto_send)
        lbl_auto_send = QLabel(i18n.t("Send Selection Automatically"))
        lbl_auto_send.setStyleSheet("color: #777;")
        send_layout.addWidget(lbl_auto_send)
        send_layout.addStretch(1)
        left_panel.addLayout(send_layout)
        
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(left_panel_widget)
        splitter.addWidget(right_panel_widget)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([350, 800])
        main_layout.addWidget(splitter)
        
        # Connect auto-run triggers
        self._slider_min_supp.sliderReleased.connect(self._auto_run_check)
        self._spin_min_supp.editingFinished.connect(self._auto_run_check)
        self._slider_max_items.sliderReleased.connect(self._auto_run_check)
        self._spin_max_items.editingFinished.connect(self._auto_run_check)
        self._txt_contains.textChanged.connect(self._auto_run_check)
        self._spin_min_filter.valueChanged.connect(self._auto_run_check)
        self._spin_max_filter.valueChanged.connect(self._auto_run_check)
        self._chk_apply_filters.stateChanged.connect(self._auto_run_check)

    def _auto_run_check(self) -> None:
        if self._chk_auto_find.isChecked():
            self._run()

    def set_input_payload(self, payload: WorkflowPayload | None) -> None:
        if payload is None:
            self._dataset = None
            self._itemsets_df = None
            self._filtered_df = None
            self._lbl_num_itemsets.setText(i18n.t("Number of itemsets:"))
            self._tree_model.removeRows(0, self._tree_model.rowCount())
            self._notify_output_changed()
            return

        if payload.port_label == "Data" and isinstance(payload.value, DatasetHandle):
            self._dataset = payload.value
            self._run()

    def current_output_payload(self) -> WorkflowPayload | None:
        if self._filtered_df is None or self._filtered_df.empty:
            return None
        return WorkflowPayload("Itemsets", self._filtered_df)

    def serialize_node_state(self) -> dict[str, object]:
        return {
            "min_support": self._spin_min_supp.value(),
            "max_itemsets": self._spin_max_items.value(),
            "auto_find": self._chk_auto_find.isChecked(),
            "contains": self._txt_contains.text(),
            "min_filter": self._spin_min_filter.value(),
            "max_filter": self._spin_max_filter.value(),
            "apply_filters": self._chk_apply_filters.isChecked(),
            "auto_send": self._chk_auto_send.isChecked()
        }

    def restore_node_state(self, payload: dict[str, object]) -> None:
        self._spin_min_supp.setValue(int(payload.get("min_support", 5)))
        self._spin_max_items.setValue(int(payload.get("max_itemsets", 10000)))
        self._chk_auto_find.setChecked(bool(payload.get("auto_find", False)))
        self._txt_contains.setText(str(payload.get("contains", "")))
        self._spin_min_filter.setValue(int(payload.get("min_filter", 2)))
        self._spin_max_filter.setValue(int(payload.get("max_filter", 999)))
        self._chk_apply_filters.setChecked(bool(payload.get("apply_filters", True)))
        self._chk_auto_send.setChecked(bool(payload.get("auto_send", True)))

    def _run(self) -> None:
        if self._dataset is None:
            return

        try:
            min_supp = self._spin_min_supp.value() / 100.0
            
            self._itemsets_df = self._svc.find_frequent_itemsets(
                self._dataset, 
                min_support=min_supp, 
                use_fpgrowth=True
            )
            
            self._filtered_df = self._itemsets_df.copy()
            if not self._filtered_df.empty:
                self._filtered_df['length'] = self._filtered_df['itemsets'].apply(lambda x: len(x))
                
                if self._chk_apply_filters.isChecked():
                    min_len = self._spin_min_filter.value()
                    max_len = self._spin_max_filter.value()
                    self._filtered_df = self._filtered_df[(self._filtered_df['length'] >= min_len) & (self._filtered_df['length'] <= max_len)]
                    
                    contains_txt = self._txt_contains.text().strip().lower()
                    if contains_txt:
                        def has_item(items):
                            return any(contains_txt in str(item).lower() for item in items)
                        self._filtered_df = self._filtered_df[self._filtered_df['itemsets'].apply(has_item)]
                        
                    max_count = self._spin_max_items.value()
                    if len(self._filtered_df) > max_count:
                        # Sort by support descending to keep the best itemsets
                        self._filtered_df = self._filtered_df.sort_values(by="support", ascending=False).head(max_count)
            
            self._update_tree()
            
            self._lbl_num_itemsets.setText(f"Number of itemsets: {len(self._filtered_df)}")
            if self._chk_auto_send.isChecked():
                self._notify_output_changed()
            
        except Exception as exc:
            self._lbl_num_itemsets.setText(f"Number of itemsets: Error")

    def _update_tree(self) -> None:
        self._tree_model.removeRows(0, self._tree_model.rowCount())
        if self._filtered_df is None or self._filtered_df.empty:
            return

        df = self._filtered_df.copy()
        
        for length in sorted(df['length'].unique()):
            len_df = df[df['length'] == length].sort_values(by="support", ascending=False)
            
            parent_item = QStandardItem(f"{len(len_df)} itemsets of length {length}")
            parent_item.setFlags(Qt.ItemFlag.ItemIsEnabled)
            parent_supp = QStandardItem("")
            parent_pct = QStandardItem("")
            
            self._tree_model.appendRow([parent_item, parent_supp, parent_pct])
            
            for _, row in len_df.iterrows():
                supp_val = row['support']
                items_val = ", ".join(list(row['itemsets']))
                
                child_item = QStandardItem(items_val)
                child_supp = QStandardItem(f"{supp_val:.3f}")
                child_pct = QStandardItem(f"{supp_val*100:.1f}%")
                parent_item.appendRow([child_item, child_supp, child_pct])
                
        self._tree.expandAll()
        self._tree.resizeColumnToContents(0)
        self._tree.resizeColumnToContents(1)
        self._tree.resizeColumnToContents(2)
