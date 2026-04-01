from __future__ import annotations

from pathlib import Path

from PySide6.QtGui import QColor, QPalette
from PySide6.QtWidgets import QApplication


_CHECK_ICON_PATH = (Path(__file__).resolve().parent / "assets" / "checkbox-check.svg").as_posix()
_RADIO_ICON_PATH = (Path(__file__).resolve().parent / "assets" / "radio-check.svg").as_posix()

APP_STYLESHEET = """
QWidget {
    background-color: #f4f3f0;
    color: #2b2b2b;
    font-family: "Segoe UI";
    font-size: 10pt;
}
QMainWindow {
    background-color: #ebe8e1;
}
QDialog#widgetPopup {
    background: transparent;
    border: none;
}
QFrame#widgetPopupSurface {
    background-color: #fbfaf7;
    border: 2px solid #bca98c;
    border-radius: 18px;
}
QFrame#widgetPopupDragHandle {
    background: transparent;
    border: none;
}
QMenuBar {
    background-color: #f7f6f2;
    border-bottom: 1px solid #d0ccc3;
}
QMenuBar::item {
    background: transparent;
    padding: 6px 10px;
}
QMenuBar::item:selected {
    background: #e8e1d1;
}
QStatusBar {
    background-color: #f7f6f2;
    border-top: 1px solid #d0ccc3;
}
QFrame#categoryPanel {
    background-color: #ebe8e1;
    border-right: none;
}
QPushButton#workflowInfoButton {
    background-color: #fff8eb;
    border: 1px solid #d5c7b1;
    border-radius: 12px;
    padding: 10px 12px;
    font-weight: 600;
    color: #3c3022;
}
QPushButton#workflowInfoButton:hover {
    background-color: #f7ead2;
    border-color: #d8af6e;
}
QFrame#catalogPanel {
    background-color: #ece9e2;
    border-right: 1px solid #c9c3b8;
}
QFrame#contentPanel {
    background-color: #fbfaf7;
}
QFrame#workflowQuickTools {
    background-color: rgba(251, 250, 247, 0.96);
    border: 1px solid #d8cdbd;
    border-radius: 18px;
}
QFrame#workflowQuickTools QToolButton {
    background-color: #fffaf0;
    border: 1px solid #ddcfbb;
    border-radius: 12px;
    padding: 0;
    min-width: 34px;
    min-height: 34px;
    color: #3d3123;
    font-weight: 600;
}
QFrame#workflowQuickTools QToolButton:hover {
    background-color: #f6e7cf;
    border-color: #d6ae6f;
}
QFrame#workflowQuickTools QToolButton:checked {
    background-color: #f0c98a;
    border-color: #cf9440;
}
QFrame#workflowMiniMap {
    background-color: rgba(251, 250, 247, 0.92);
    border: 1px solid #d8cdbd;
    border-radius: 14px;
}
QListWidget#categoryList {
    background: transparent;
    border: none;
    padding: 8px 0;
    outline: 0;
    font-family: "Trebuchet MS", "Segoe UI";
    font-size: 10.5pt;
    font-weight: 600;
}
QListWidget#categoryList::item {
    margin: 1px 8px;
    padding: 10px 14px;
    border-radius: 10px;
    border: 1px solid transparent;
    color: #473a2a;
}
QListWidget#categoryList::item:hover {
    background-color: #e9e1d5;
    border: 1px solid #d5cab8;
}
QListWidget#categoryList::item:selected {
    background-color: #f1cb88;
    border: 1px solid #cf9e49;
    color: #2f2417;
}
QLineEdit#catalogSearch {
    border: 1px solid #c9c3b8;
    border-radius: 8px;
    background: #fffdf7;
    padding: 6px 10px;
}
QPushButton[card="true"] {
    background-color: #fffdf9;
    border: 1px solid #d9d1c4;
    border-radius: 12px;
    padding: 10px 10px;
    text-align: left;
}
QPushButton[card="true"]:hover {
    background-color: #fff4df;
    border-color: #e3b56a;
}
QPushButton[comingSoon="true"] {
    color: #77706a;
    background-color: #f1ede6;
}
QPushButton[primary="true"] {
    background-color: #e2a952;
    color: white;
    border: none;
    border-radius: 10px;
    padding: 10px 16px;
    font-weight: 600;
}
QPushButton[primary="true"]:hover {
    background-color: #cf9440;
}
QPushButton[primary="true"]:disabled {
    background-color: #ead7b2;
    color: #8a7350;
}
QPushButton[secondary="true"] {
    background-color: #e7ded0;
    color: #3a2f22;
    border: none;
    border-radius: 10px;
    padding: 10px 16px;
    font-weight: 600;
}
QPushButton[secondary="true"]:hover {
    background-color: #d9cfbe;
}
QPushButton[secondary="true"]:disabled {
    background-color: #eee7db;
    color: #8b8377;
}
QPushButton#widgetPopupCloseButton {
    background-color: #d8d2c7;
    color: #2e2a23;
    border: none;
    border-radius: 10px;
    padding: 8px 18px;
    font-weight: 600;
}
QPushButton#widgetPopupCloseButton:hover {
    background-color: #c8c0b2;
}
QLineEdit#workflowInfoTitleInput,
QTextEdit#workflowInfoDescriptionInput {
    background: #fffdf9;
    border: 1px solid #d8cdbd;
    border-radius: 12px;
    padding: 10px 12px;
}
QFrame[panel="true"] {
    background: #ffffff;
    border: 1px solid #e0d8cd;
    border-radius: 14px;
}
QFrame[infoCard="true"] {
    background: #fff8eb;
    border: 1px solid #ecd8b1;
    border-radius: 12px;
}
QLabel[sectionTitle="true"] {
    font-size: 16pt;
    font-weight: 700;
    color: #2e2a23;
}
QLabel[muted="true"] {
    color: #6f6a62;
}
QListWidget {
    background: white;
    border: 1px solid #ddd7cb;
    border-radius: 10px;
}
QTableWidget {
    background: white;
    border: 1px solid #ddd7cb;
    border-radius: 10px;
    gridline-color: #ece8df;
    selection-background-color: #cfe6ff;
    selection-color: #1f2f3d;
}
QTableWidget::item:selected {
    background-color: #cfe6ff;
    color: #1f2f3d;
}
QTableWidget::item:selected:active {
    background-color: #b9daff;
    color: #17222c;
}
QHeaderView::section {
    background: #efe7d8;
    border: none;
    border-right: 1px solid #ddd7cb;
    border-bottom: 1px solid #ddd7cb;
    padding: 8px;
    font-weight: 600;
}
QComboBox {
    background: white;
    border: 1px solid #d1cabf;
    border-radius: 8px;
    padding: 6px 10px;
    color: #2b2b2b;
}
QComboBox QAbstractItemView {
    background: white;
    color: #2b2b2b;
    border: 1px solid #d1cabf;
    selection-background-color: #e8e1d1;
    selection-color: #2b2b2b;
}
QComboBox::drop-down {
    border: none;
}
QCheckBox {
    spacing: 8px;
}
QCheckBox::indicator {
    width: 16px;
    height: 16px;
    border: 1px solid #2f2417;
    border-radius: 3px;
    background: #fffdf9;
}
QCheckBox::indicator:hover {
    background: #f8ecd7;
    border-color: #cf9440;
}
QCheckBox::indicator:checked {
    background: #fffdf9;
    border-color: #2f2417;
    image: url("__CHECK_ICON__");
}
QCheckBox::indicator:unchecked {
    background: #fffdf9;
}
QCheckBox:disabled {
    color: #b0a99e;
}
QCheckBox::indicator:disabled {
    border-color: #c8c0b4;
    background: #eae7e1;
}
QCheckBox::indicator:disabled:checked {
    border-color: #c8c0b4;
    background: #eae7e1;
}
QRadioButton {
    spacing: 8px;
}
QRadioButton::indicator {
    width: 16px;
    height: 16px;
    border: 1px solid #2f2417;
    border-radius: 8px;
    background: #fffdf9;
}
QRadioButton::indicator:hover {
    background: #f8ecd7;
    border-color: #cf9440;
}
QRadioButton::indicator:checked {
    background: #fffdf9;
    border-color: #2f2417;
    image: url("__RADIO_ICON__");
}
QRadioButton::indicator:unchecked {
    background: #fffdf9;
}
QRadioButton:disabled {
    color: #b0a99e;
}
QRadioButton::indicator:disabled {
    border-color: #c8c0b4;
    background: #eae7e1;
}
QPushButton#fileSourceActionButton {
    background-color: #f5ead5;
    color: #3a2f22;
    border: 1px solid #d8b57a;
    border-radius: 10px;
    padding: 8px 12px;
    font-weight: 600;
}
QPushButton#fileSourceActionButton:hover {
    background-color: #efd8b4;
    border-color: #cf9440;
}
QPushButton#fileSourceActionButton:disabled {
    background-color: #eee7db;
    border-color: #d7cfbf;
    color: #8b8377;
}
QToolButton#widgetPopupToolButton {
    background-color: #fff7ea;
    border: 1px solid #d7c4a8;
    border-radius: 10px;
    padding: 6px;
}
QToolButton#widgetPopupToolButton:hover {
    background-color: #f7e6cb;
    border-color: #cf9440;
}
QLabel#fileTypeBadge {
    min-width: 18px;
}
""".replace("__CHECK_ICON__", _CHECK_ICON_PATH).replace("__RADIO_ICON__", _RADIO_ICON_PATH)


def apply_theme(app: QApplication) -> None:
    palette = app.palette()
    palette.setColor(QPalette.ColorRole.Window, QColor("#f4f3f0"))
    palette.setColor(QPalette.ColorRole.Base, QColor("#ffffff"))
    palette.setColor(QPalette.ColorRole.Button, QColor("#ffffff"))
    app.setPalette(palette)
    app.setStyleSheet(APP_STYLESHEET)
