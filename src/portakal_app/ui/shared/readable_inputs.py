from __future__ import annotations

from PySide6.QtWidgets import QLineEdit, QSpinBox


READABLE_LINE_EDIT_STYLE = """
QLineEdit {
    background-color: #2b2b2b;
    color: #ffffff;
    border: 1px solid #6b6b6b;
    border-radius: 8px;
    padding: 6px 8px;
    selection-background-color: #efaa3a;
    selection-color: #ffffff;
}
QLineEdit:focus {
    border: 1px solid #f0a433;
}
"""

READABLE_SPIN_BOX_STYLE = """
QSpinBox {
    background-color: #2b2b2b;
    color: #ffffff;
    border: 1px solid #6b6b6b;
    border-radius: 8px;
    padding: 4px 8px;
    selection-background-color: #efaa3a;
    selection-color: #ffffff;
}
QSpinBox:focus {
    border: 1px solid #f0a433;
}
"""


def apply_readable_line_edit_style(line_edit: QLineEdit) -> None:
    line_edit.setStyleSheet(READABLE_LINE_EDIT_STYLE)


def apply_readable_spin_box_style(spin_box: QSpinBox) -> None:
    spin_box.setStyleSheet(READABLE_SPIN_BOX_STYLE)
