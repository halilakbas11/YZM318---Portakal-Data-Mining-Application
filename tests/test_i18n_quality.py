from __future__ import annotations

import os
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QCheckBox, QComboBox, QGroupBox, QLabel, QLineEdit, QListWidget, QMessageBox, QPushButton, QRadioButton

from portakal_app.app import create_application
from portakal_app.ui import i18n
from portakal_app.ui.main_window import MainWindow
from portakal_app.ui.screens.color_screen import ColorScreen
from portakal_app.ui.screens.column_statistics_screen import ColumnStatisticsScreen
from portakal_app.ui.screens.csv_import_screen import CSVImportScreen
from portakal_app.ui.screens.data_info_screen import DataInfoScreen
from portakal_app.ui.screens.data_table_screen import DataTableScreen
from portakal_app.ui.screens.datasets_screen import DatasetsScreen
from portakal_app.ui.screens.edit_domain_screen import EditDomainScreen
from portakal_app.ui.screens.file_screen import FileScreen
from portakal_app.ui.screens.paint_data_screen import PaintDataScreen
from portakal_app.ui.screens.rank_screen import RankScreen
from portakal_app.ui.screens.save_data_screen import SaveDataScreen


@pytest.fixture(scope="session")
def app():
    return create_application()


def _collect_visible_texts(widget) -> set[str]:
    texts: set[str] = set()
    classes = [QLabel, QPushButton, QCheckBox, QGroupBox, QRadioButton, QLineEdit, QComboBox]

    if hasattr(widget, "menuBar"):
        texts.update(action.text() for action in widget.menuBar().actions() if action.text())
        for menu in getattr(widget, "_menus", {}).values():
            texts.update(action.text() for action in menu.actions() if not action.isSeparator() and action.text())

    for child in widget.findChildren(QListWidget):
        for index in range(child.count()):
            item = child.item(index)
            if item is not None and item.text():
                texts.add(item.text())

    for cls in classes:
        for child in widget.findChildren(cls):
            if isinstance(child, QGroupBox):
                if child.title():
                    texts.add(child.title())
            elif isinstance(child, QLineEdit):
                if child.placeholderText():
                    texts.add(child.placeholderText())
            elif isinstance(child, QComboBox):
                for index in range(child.count()):
                    item_text = child.itemText(index)
                    if item_text:
                        texts.add(item_text)
                if child.currentText():
                    texts.add(child.currentText())
            else:
                if child.text():
                    texts.add(child.text())
    return texts


def test_active_ui_surfaces_do_not_leave_known_english_labels_in_turkish_mode(app, monkeypatch):
    previous_language = i18n.current_language()
    monkeypatch.setattr(QMessageBox, "information", lambda *args, **kwargs: 0)

    widgets = [
        MainWindow(),
        FileScreen(),
        CSVImportScreen(),
        DatasetsScreen(),
        DataTableScreen(),
        PaintDataScreen(),
        DataInfoScreen(),
        RankScreen(),
        EditDomainScreen(),
        ColorScreen(),
        ColumnStatisticsScreen(),
        SaveDataScreen(),
    ]

    forbidden = {
        "New",
        "Open",
        "Open and Freeze",
        "Open Recent",
        "Close Window",
        "Save",
        "Save As ...",
        "Quit",
        "Edit",
        "View",
        "Widget",
        "Window",
        "Options",
        "Documentation",
        "Workflow",
        "Transform",
        "Visualize",
        "Evaluate",
        "Unsupervised",
        "Reload",
        "Apply",
        "Send Automatically",
        "Apply Automatically",
        "Send Data",
        "Load",
        "Source",
        "Info",
        "Summary",
        "Column Profiles",
        "AI Analysis",
        "Preview",
        "Variables",
        "Selection",
        "Clear Selection",
        "Restore Original Order",
        "Show variable labels (if present)",
        "Color by instance classes",
        "Select full rows",
        "Names",
        "Labels",
        "Tools",
        "Brush",
        "Put",
        "Select",
        "Jitter",
        "Magnet",
        "Clear",
        "Radius:",
        "Intensity:",
        "Symbol:",
        "Reset to Input Data",
        "Discrete Variables",
        "Numeric Variables",
        "Save Color Settings",
        "Load Color Settings",
        "Select Color",
        "Dataset: none",
        "Load a dataset to edit column names, types and roles.",
        "Load a dataset to rank feature usefulness.",
        "Load a dataset to inspect per-column statistics.",
        "No source selected yet.",
        "Use Browse, Reload and Apply after the data backend is connected.",
        "Select a local dataset...",
        "Loading data...",
        "Polars DataFrame is being prepared, please wait.",
        "You can start by selecting a file from the File widget.",
        "Provider",
        "Risks",
        "Suggestions",
        "Search",
        "Data: none",
        "Data Subset: -",
    }

    try:
        i18n.set_language("tr")
        failures: dict[str, list[str]] = {}
        for widget in widgets:
            i18n.apply_to_widget(widget)
            refresh = getattr(widget, "refresh_translations", None)
            if callable(refresh):
                refresh()
            hits = sorted(text for text in _collect_visible_texts(widget) if text in forbidden)
            if hits:
                failures[widget.__class__.__name__] = hits
        assert failures == {}
    finally:
        i18n.set_language(previous_language)
        for widget in widgets:
            widget.close()


def test_text_sources_do_not_contain_common_mojibake_sequences():
    forbidden_sequences = ("\u00C3", "\u00C4", "\u00C5", "\uFFFD")
    allowed_suffixes = {".py", ".md", ".toml"}
    ignored_parts = {".git", ".venv", "venv", "__pycache__", "portakal_app.egg-info", ".vscode"}
    offenders: list[str] = []

    for path in Path(".").rglob("*"):
        if path.is_dir():
            continue
        if any(part in ignored_parts for part in path.parts):
            continue
        if path.suffix.lower() not in allowed_suffixes:
            continue
        text = path.read_text(encoding="utf-8")
        hits = [sequence for sequence in forbidden_sequences if sequence in text]
        if hits:
            offenders.append(f"{path}: {hits}")

    assert offenders == []
