from __future__ import annotations

import sys

from PySide6.QtWidgets import QApplication

from portakal_app.ui.main_window import MainWindow
from portakal_app.ui.theme import apply_theme


def create_application() -> QApplication:
    app = QApplication.instance() or QApplication(sys.argv)
    app.setApplicationName("Portakal")
    app.setOrganizationName("Portakal")
    apply_theme(app)
    return app


def run() -> int:
    app = create_application()
    window = MainWindow()
    window.show()
    return app.exec()
