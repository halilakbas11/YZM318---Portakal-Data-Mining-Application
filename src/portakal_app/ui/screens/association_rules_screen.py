from __future__ import annotations
from typing import Any
from PySide6.QtCore import Qt
from PySide6.QtGui import QStandardItem, QStandardItemModel
from PySide6.QtWidgets import (
    QCheckBox, QDoubleSpinBox, QGroupBox, QHBoxLayout, QHeaderView,
    QLabel, QLineEdit, QPushButton, QSlider, QSpinBox, QSplitter,
    QTableView, QVBoxLayout, QWidget,
)
from portakal_app.data.models import DatasetHandle
from portakal_app.data.services.associate_service import AssociateService
from portakal_app.models import WorkflowPayload
from portakal_app.ui import i18n
from portakal_app.ui.screens.node_screen import WorkflowNodeScreenSupport


class AssociationRulesScreen(QWidget, WorkflowNodeScreenSupport):
    """Generate and filter association rules from data."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._init_workflow_node_support()
        self._dataset: DatasetHandle | None = None
        self._rules_df: Any = None
        self._filtered_df: Any = None
        self._svc = AssociateService()

        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(8, 8, 8, 8)

        # Right panel: TableView
        right_panel_widget = QWidget()
        right_panel_widget.setMinimumWidth(500)
        right_panel = QVBoxLayout(right_panel_widget)
        right_panel.setContentsMargins(0, 0, 0, 0)
        self._table = QTableView()
        self._table.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self._table.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self._table.setHorizontalScrollMode(QTableView.ScrollMode.ScrollPerPixel)
        self._table_model = QStandardItemModel()
        self._table.setModel(self._table_model)
        self._table.setAlternatingRowColors(True)
        self._table.setEditTriggers(QTableView.EditTrigger.NoEditTriggers)
        self._table.setSortingEnabled(True)
        self._table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        self._table.horizontalHeader().setMinimumSectionSize(80)
        self._table.horizontalHeader().setStretchLastSection(False)
        right_panel.addWidget(self._table)

        # Left panel
        left_panel = QVBoxLayout()
        left_panel.setSpacing(12)
        left_panel_widget = QWidget()
        left_panel_widget.setLayout(left_panel)
        left_panel_widget.setMinimumWidth(280)
        left_panel_widget.setMaximumWidth(350)

        # Info
        info_box = QGroupBox(i18n.t("Info"))
        info_layout = QVBoxLayout(info_box)
        self._lbl_info_rules = QLabel(i18n.t("Rules: 0 (shown 0)"))
        info_layout.addWidget(self._lbl_info_rules)
        left_panel.addWidget(info_box)

        # Find association rules
        find_box = QGroupBox(i18n.t("Find association rules"))
        find_layout = QVBoxLayout(find_box)

        supp_layout = QHBoxLayout()
        supp_layout.addWidget(QLabel(i18n.t("Min. supp.:")))
        self._slider_supp = QSlider(Qt.Orientation.Horizontal)
        self._slider_supp.setRange(1, 10000)
        self._slider_supp.setValue(5)
        self._spin_supp = QDoubleSpinBox()
        self._spin_supp.setRange(0.01, 100.00)
        self._spin_supp.setValue(0.05)
        self._spin_supp.setDecimals(2)
        self._spin_supp.setSuffix(" %")
        self._slider_supp.valueChanged.connect(lambda v: self._spin_supp.setValue(v / 100.0))
        self._spin_supp.valueChanged.connect(lambda v: self._slider_supp.setValue(int(v * 100)))
        supp_layout.addWidget(self._slider_supp)
        supp_layout.addWidget(self._spin_supp)
        find_layout.addLayout(supp_layout)

        conf_layout = QHBoxLayout()
        conf_layout.addWidget(QLabel(i18n.t("Min. conf.:")))
        self._slider_conf = QSlider(Qt.Orientation.Horizontal)
        self._slider_conf.setRange(1, 10000)
        self._slider_conf.setValue(5000)
        self._spin_conf = QDoubleSpinBox()
        self._spin_conf.setRange(0.01, 100.00)
        self._spin_conf.setValue(50.00)
        self._spin_conf.setDecimals(2)
        self._spin_conf.setSuffix(" %")
        self._slider_conf.valueChanged.connect(lambda v: self._spin_conf.setValue(v / 100.0))
        self._spin_conf.valueChanged.connect(lambda v: self._slider_conf.setValue(int(v * 100)))
        conf_layout.addWidget(self._slider_conf)
        conf_layout.addWidget(self._spin_conf)
        find_layout.addLayout(conf_layout)

        rules_layout = QHBoxLayout()
        rules_layout.addWidget(QLabel(i18n.t("Max. rules:")))
        self._slider_rules = QSlider(Qt.Orientation.Horizontal)
        self._slider_rules.setRange(1, 100000)
        self._slider_rules.setValue(10000)
        self._spin_rules = QSpinBox()
        self._spin_rules.setRange(1, 100000)
        self._spin_rules.setValue(10000)
        self._slider_rules.valueChanged.connect(self._spin_rules.setValue)
        self._spin_rules.valueChanged.connect(self._slider_rules.setValue)
        rules_layout.addWidget(self._slider_rules)
        rules_layout.addWidget(self._spin_rules)
        find_layout.addLayout(rules_layout)

        self._chk_class_rules = QCheckBox(i18n.t("Induce only classification rules"))
        find_layout.addWidget(self._chk_class_rules)
        self._chk_restrict = QCheckBox(i18n.t("Restrict search by below filters"))
        find_layout.addWidget(self._chk_restrict)
        self._btn_find = QPushButton(i18n.t("Find Rules"))
        self._btn_find.clicked.connect(self._run)
        find_layout.addWidget(self._btn_find)
        left_panel.addWidget(find_box)

        # Filter by Antecedent
        ant_box = QGroupBox(i18n.t("Filter by Antecedent"))
        ant_layout = QVBoxLayout(ant_box)
        ant_c = QHBoxLayout()
        ant_c.addWidget(QLabel(i18n.t("Contains:")))
        self._txt_ant_contains = QLineEdit()
        ant_c.addWidget(self._txt_ant_contains)
        ant_layout.addLayout(ant_c)
        ant_i = QHBoxLayout()
        ant_i.addWidget(QLabel(i18n.t("Items, min:")))
        self._spin_ant_min = QSpinBox()
        self._spin_ant_min.setRange(1, 999)
        self._spin_ant_min.setValue(1)
        ant_i.addWidget(self._spin_ant_min)
        ant_i.addWidget(QLabel(i18n.t("max:")))
        self._spin_ant_max = QSpinBox()
        self._spin_ant_max.setRange(1, 999)
        self._spin_ant_max.setValue(999)
        ant_i.addWidget(self._spin_ant_max)
        ant_layout.addLayout(ant_i)
        left_panel.addWidget(ant_box)

        # Filter by Consequent
        con_box = QGroupBox(i18n.t("Filter by Consequent"))
        con_layout = QVBoxLayout(con_box)
        con_c = QHBoxLayout()
        con_c.addWidget(QLabel(i18n.t("Contains:")))
        self._txt_con_contains = QLineEdit()
        con_c.addWidget(self._txt_con_contains)
        con_layout.addLayout(con_c)
        con_i = QHBoxLayout()
        con_i.addWidget(QLabel(i18n.t("Items, min:")))
        self._spin_con_min = QSpinBox()
        self._spin_con_min.setRange(1, 999)
        self._spin_con_min.setValue(1)
        con_i.addWidget(self._spin_con_min)
        con_i.addWidget(QLabel(i18n.t("max:")))
        self._spin_con_max = QSpinBox()
        self._spin_con_max.setRange(1, 999)
        self._spin_con_max.setValue(999)
        con_i.addWidget(self._spin_con_max)
        con_layout.addLayout(con_i)
        left_panel.addWidget(con_box)

        left_panel.addStretch(1)

        send_layout = QHBoxLayout()
        self._chk_auto_send = QCheckBox()
        self._chk_auto_send.setChecked(True)
        send_layout.addWidget(self._chk_auto_send)
        lbl_send = QLabel(i18n.t("Send selection"))
        lbl_send.setStyleSheet("color: #777;")
        send_layout.addWidget(lbl_send)
        send_layout.addStretch(1)
        left_panel.addLayout(send_layout)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(left_panel_widget)
        splitter.addWidget(right_panel_widget)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([350, 800])
        main_layout.addWidget(splitter)

        # Live filtering connections
        self._txt_ant_contains.textChanged.connect(self._apply_filters)
        self._spin_ant_min.valueChanged.connect(self._apply_filters)
        self._spin_ant_max.valueChanged.connect(self._apply_filters)
        self._txt_con_contains.textChanged.connect(self._apply_filters)
        self._spin_con_min.valueChanged.connect(self._apply_filters)
        self._spin_con_max.valueChanged.connect(self._apply_filters)
        self._chk_class_rules.stateChanged.connect(self._apply_filters)
        self._chk_restrict.stateChanged.connect(self._apply_filters)

    def set_input_payload(self, payload: WorkflowPayload | None) -> None:
        if payload is None:
            self._dataset = None
            self._rules_df = None
            self._filtered_df = None
            self._lbl_info_rules.setText(i18n.t("Rules: 0 (shown 0)"))
            self._table_model.clear()
            self._notify_output_changed()
            return
        if payload.port_label == "Data" and isinstance(payload.value, DatasetHandle):
            self._dataset = payload.value
            self._run()
        elif payload.port_label == "Itemsets":
            pass

    def current_output_payload(self) -> WorkflowPayload | None:
        if self._filtered_df is None or self._filtered_df.empty:
            return None
        return WorkflowPayload("Rules", self._filtered_df)

    def serialize_node_state(self) -> dict[str, object]:
        return {
            "min_support": self._spin_supp.value(),
            "min_conf": self._spin_conf.value(),
            "max_rules": self._spin_rules.value(),
            "class_rules": self._chk_class_rules.isChecked(),
            "restrict": self._chk_restrict.isChecked(),
            "ant_contains": self._txt_ant_contains.text(),
            "ant_min": self._spin_ant_min.value(),
            "ant_max": self._spin_ant_max.value(),
            "con_contains": self._txt_con_contains.text(),
            "con_min": self._spin_con_min.value(),
            "con_max": self._spin_con_max.value(),
            "auto_send": self._chk_auto_send.isChecked()
        }

    def restore_node_state(self, payload: dict[str, object]) -> None:
        self._spin_supp.setValue(float(payload.get("min_support", 0.05)))
        self._spin_conf.setValue(float(payload.get("min_conf", 50.00)))
        self._spin_rules.setValue(int(payload.get("max_rules", 10000)))
        self._chk_class_rules.setChecked(bool(payload.get("class_rules", False)))
        self._chk_restrict.setChecked(bool(payload.get("restrict", False)))
        self._txt_ant_contains.setText(str(payload.get("ant_contains", "")))
        self._spin_ant_min.setValue(int(payload.get("ant_min", 1)))
        self._spin_ant_max.setValue(int(payload.get("ant_max", 999)))
        self._txt_con_contains.setText(str(payload.get("con_contains", "")))
        self._spin_con_min.setValue(int(payload.get("con_min", 1)))
        self._spin_con_max.setValue(int(payload.get("con_max", 999)))
        self._chk_auto_send.setChecked(bool(payload.get("auto_send", True)))

    def _run(self) -> None:
        if self._dataset is None:
            return
        self._lbl_info_rules.setText(i18n.t("Rules: Running..."))
        self.repaint()
        try:
            min_supp = self._spin_supp.value() / 100.0
            min_conf = self._spin_conf.value() / 100.0
            itemsets = self._svc.find_frequent_itemsets(self._dataset, min_support=min_supp, use_fpgrowth=True)
            self._rules_df = self._svc.generate_rules(itemsets, metric="confidence", min_threshold=min_conf)
            self._apply_filters()
        except Exception:
            self._lbl_info_rules.setText("Rules: Error")

    def _apply_filters(self) -> None:
        if self._rules_df is None or self._rules_df.empty:
            self._lbl_info_rules.setText(i18n.t("Rules: 0 (shown 0)"))
            self._filtered_df = self._rules_df
            self._update_table()
            return
        df = self._rules_df.copy()
        total_rules = len(self._rules_df)
        ant_min = self._spin_ant_min.value()
        ant_max = self._spin_ant_max.value()
        con_min = self._spin_con_min.value()
        con_max = self._spin_con_max.value()
        ant_contains = self._txt_ant_contains.text().strip().lower()
        con_contains = self._txt_con_contains.text().strip().lower()
        df['ant_len'] = df['antecedents'].apply(lambda x: len(x))
        df['con_len'] = df['consequents'].apply(lambda x: len(x))
        df = df[(df['ant_len'] >= ant_min) & (df['ant_len'] <= ant_max)]
        df = df[(df['con_len'] >= con_min) & (df['con_len'] <= con_max)]
        if ant_contains:
            df = df[df['antecedents'].apply(lambda items: any(ant_contains in str(i).lower() for i in items))]
        if con_contains:
            df = df[df['consequents'].apply(lambda items: any(con_contains in str(i).lower() for i in items))]
        if self._chk_class_rules.isChecked():
            df = df[df['con_len'] == 1]
        max_rules = self._spin_rules.value()
        if len(df) > max_rules:
            df = df.sort_values(by="confidence", ascending=False).head(max_rules)
        self._filtered_df = df
        self._update_table()
        shown_rules = len(self._filtered_df)
        self._lbl_info_rules.setText(f"Rules: {total_rules} (shown {shown_rules})")
        if self._chk_auto_send.isChecked():
            self._notify_output_changed()

    def _update_table(self) -> None:
        self._table_model.clear()
        headers = ["Antecedents", "Consequents", "Support", "Confidence", "Coverage", "Lift", "Leverage"]
        self._table_model.setHorizontalHeaderLabels(headers)
        if self._filtered_df is None or self._filtered_df.empty:
            return
        df = self._filtered_df
        for _, row in df.iterrows():
            ant = ", ".join(list(row['antecedents']))
            con = ", ".join(list(row['consequents']))
            supp = f"{row['support']:.3f}"
            conf = f"{row['confidence']:.3f}"
            cov = f"{row['antecedent support']:.3f}"
            lift = f"{row['lift']:.3f}"
            lev = f"{row['leverage']:.3f}"
            items = [
                QStandardItem(ant), QStandardItem(con), QStandardItem(supp),
                QStandardItem(conf), QStandardItem(cov), QStandardItem(lift),
                QStandardItem(lev),
            ]
            for item in items:
                item.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable)
            self._table_model.appendRow(items)
        self._table.resizeColumnsToContents()
