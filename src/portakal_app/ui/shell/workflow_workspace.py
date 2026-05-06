from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QEvent, QPoint, QPointF, QRect, QSize, Qt, QTimer, QUrl, Signal
from PySide6.QtGui import QCloseEvent, QColor, QCursor, QDesktopServices, QPainter, QPen, QResizeEvent
from PySide6.QtPrintSupport import QPrintDialog, QPrinter
from PySide6.QtWidgets import (
    QApplication,
    QDialog,
    QFrame,
    QFileDialog,
    QGraphicsDropShadowEffect,
    QHBoxLayout,
    QLabel,
    QMenu,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QToolButton,
    QToolTip,
    QVBoxLayout,
    QWidget,
    QSizePolicy,
    QScrollArea,
)

from portakal_app.models import WidgetDefinition
from portakal_app.ui import i18n
from portakal_app.ui.icons import get_toolbar_icon
from portakal_app.ui.shell.workflow_canvas import WorkflowCanvas


class WidgetReportDialog(QDialog):
    def __init__(self, report_data: dict[str, object], parent: QWidget | None = None) -> None:
        super().__init__(parent, Qt.WindowType.Dialog | Qt.WindowType.FramelessWindowHint)
        self.setObjectName("widgetPopup")
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setWindowTitle(i18n.t("Report"))
        self.resize(860, 620)
        self._report_data = report_data
        self._drag_offset: QPoint | None = None
        shadow = QGraphicsDropShadowEffect(self)
        shadow.setBlurRadius(36)
        shadow.setOffset(0, 12)
        shadow.setColor(QColor(54, 41, 24, 70))

        outer_layout = QVBoxLayout(self)
        outer_layout.setContentsMargins(12, 12, 12, 12)
        outer_layout.setSpacing(0)

        self._surface = QFrame(self)
        self._surface.setObjectName("widgetPopupSurface")
        self._surface.setGraphicsEffect(shadow)
        outer_layout.addWidget(self._surface)

        layout = QVBoxLayout(self._surface)
        layout.setContentsMargins(18, 18, 18, 18)
        layout.setSpacing(12)

        self._drag_handle = QFrame(self._surface)
        self._drag_handle.setObjectName("widgetPopupDragHandle")
        self._drag_handle.setFixedHeight(18)
        self._drag_handle.setCursor(Qt.CursorShape.OpenHandCursor)
        self._drag_handle.installEventFilter(self)
        layout.addWidget(self._drag_handle)

        body_layout = QHBoxLayout()
        body_layout.setSpacing(12)
        layout.addLayout(body_layout, 1)

        self._list_panel = QFrame(self)
        self._list_panel.setProperty("panel", True)
        list_layout = QVBoxLayout(self._list_panel)
        list_layout.setContentsMargins(12, 12, 12, 12)
        list_layout.setSpacing(8)
        title = QLabel(report_data.get("title", i18n.t("Report")))
        title.setProperty("sectionTitle", True)
        list_layout.addWidget(title)
        for item in report_data.get("items", []):
            label = QLabel(str(item.get("title", "")))
            label.setStyleSheet("padding: 8px; background: #f3efe8; border-radius: 8px;")
            list_layout.addWidget(label)
        list_layout.addStretch(1)
        body_layout.addWidget(self._list_panel, 1)

        self._content_panel = QFrame(self)
        self._content_panel.setProperty("panel", True)
        content_layout = QVBoxLayout(self._content_panel)
        content_layout.setContentsMargins(16, 16, 16, 16)
        content_layout.setSpacing(12)

        comment_box = QPlainTextEdit(self)
        comment_box.setPlaceholderText(i18n.t("Write a comment..."))
        comment_box.setMaximumHeight(84)
        content_layout.addWidget(comment_box)
        self._comment_boxes = [comment_box]

        for item in report_data.get("items", []):
            section = QFrame(self)
            section.setProperty("panel", True)
            section_layout = QVBoxLayout(section)
            section_layout.setContentsMargins(0, 0, 0, 0)
            section_layout.setSpacing(0)

            header = QFrame(section)
            header.setStyleSheet("background: #f4ead8; color: #2f2417; border-bottom: 1px solid #d8cdbd;")
            header_layout = QHBoxLayout(header)
            header_layout.setContentsMargins(10, 8, 10, 8)
            header_layout.addWidget(QLabel(str(item.get("title", ""))))
            header_layout.addStretch(1)
            header_layout.addWidget(QLabel(str(item.get("timestamp", ""))))
            section_layout.addWidget(header)

            body = QWidget(section)
            body_content_layout = QVBoxLayout(body)
            body_content_layout.setContentsMargins(12, 12, 12, 12)
            body_content_layout.setSpacing(4)
            for detail in item.get("details", []):
                detail_label = QLabel(str(detail))
                detail_label.setWordWrap(True)
                body_content_layout.addWidget(detail_label)
            section_layout.addWidget(body)
            content_layout.addWidget(section)

        tail_comment = QPlainTextEdit(self)
        tail_comment.setPlaceholderText(i18n.t("Write a comment..."))
        tail_comment.setMaximumHeight(84)
        content_layout.addWidget(tail_comment)
        self._comment_boxes.append(tail_comment)

        content_layout.addStretch(1)
        body_layout.addWidget(self._content_panel, 2)

        footer = QHBoxLayout()
        footer.setSpacing(10)
        layout.addLayout(footer)

        self._back_button = QPushButton(i18n.t("Back to Last Schema"))
        self._back_button.setProperty("secondary", True)
        self._back_button.clicked.connect(self.accept)
        footer.addWidget(self._back_button)

        footer.addStretch(1)

        self._save_button = QPushButton(i18n.t("Save"))
        self._save_button.setProperty("secondary", True)
        self._save_button.clicked.connect(self._save_report)
        footer.addWidget(self._save_button)

        self._print_button = QPushButton(i18n.t("Print"))
        self._print_button.setProperty("primary", True)
        self._print_button.clicked.connect(self._print_report)
        footer.addWidget(self._print_button)

    def eventFilter(self, watched, event) -> bool:
        if watched is self._drag_handle:
            if event.type() == QEvent.Type.MouseButtonPress and event.button() == Qt.MouseButton.LeftButton:
                self._drag_handle.setCursor(Qt.CursorShape.ClosedHandCursor)
                self._drag_offset = event.position().toPoint()
                return True
            if event.type() == QEvent.Type.MouseMove and self._drag_offset is not None and event.buttons() & Qt.MouseButton.LeftButton:
                parent_widget = self.parentWidget()
                global_top_left = event.globalPosition().toPoint() - self._drag_offset
                if parent_widget is not None:
                    self.move(parent_widget.mapFromGlobal(global_top_left))
                else:
                    self.move(global_top_left)
                return True
            if event.type() == QEvent.Type.MouseButtonRelease:
                self._drag_handle.setCursor(Qt.CursorShape.OpenHandCursor)
                self._drag_offset = None
                return True
        return super().eventFilter(watched, event)

    def _save_report(self) -> None:
        path, _selected_filter = QFileDialog.getSaveFileName(
            self,
            i18n.t("Save Report"),
            "report.txt",
            "Text Files (*.txt);;All Files (*.*)",
        )
        if not path:
            return
        content = self._report_plain_text()
        Path(path).write_text(content, encoding="utf-8")

    def _print_report(self) -> None:
        printer = QPrinter()
        dialog = QPrintDialog(printer, self)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        document = "\n\n".join(
            [
                self._report_data.get("title", i18n.t("Report")),
                *[
                    f"{item.get('title', '')}\n" + "\n".join(str(detail) for detail in item.get("details", []))
                    for item in self._report_data.get("items", [])
                ],
            ]
        )
        from PySide6.QtGui import QTextDocument

        text_document = QTextDocument(document)
        text_document.print(printer)

    def _report_plain_text(self) -> str:
        sections = [str(self._report_data.get("title", i18n.t("Report")))]
        for item in self._report_data.get("items", []):
            sections.append(str(item.get("title", "")))
            sections.extend(str(detail) for detail in item.get("details", []))
            sections.append("")
        comments = [box.toPlainText().strip() for box in self._comment_boxes if box.toPlainText().strip()]
        if comments:
            sections.append(i18n.t("Comments"))
            sections.extend(comments)
        return "\n".join(sections)

class WidgetDataPreviewDialog(QDialog):
    def __init__(self, title: str, preview_data: dict[str, object], parent: QWidget | None = None) -> None:
        super().__init__(parent, Qt.WindowType.Dialog | Qt.WindowType.FramelessWindowHint)
        self.setObjectName("widgetPopup")
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setWindowTitle(title)
        self.resize(900, 640)
        self._drag_offset: QPoint | None = None
        self._preview_data = preview_data
        shadow = QGraphicsDropShadowEffect(self)
        shadow.setBlurRadius(36)
        shadow.setOffset(0, 12)
        shadow.setColor(QColor(54, 41, 24, 70))

        outer_layout = QVBoxLayout(self)
        outer_layout.setContentsMargins(18, 18, 18, 18)
        outer_layout.setSpacing(0)

        self._surface = QFrame(self)
        self._surface.setObjectName("widgetPopupSurface")
        self._surface.setGraphicsEffect(shadow)
        outer_layout.addWidget(self._surface)

        layout = QVBoxLayout(self._surface)
        layout.setContentsMargins(18, 18, 18, 18)
        layout.setSpacing(10)

        self._drag_handle = QFrame(self._surface)
        self._drag_handle.setObjectName("widgetPopupDragHandle")
        self._drag_handle.setFixedHeight(18)
        self._drag_handle.setCursor(Qt.CursorShape.OpenHandCursor)
        self._drag_handle.installEventFilter(self)
        layout.addWidget(self._drag_handle)

        summary = QLabel(str(preview_data.get("summary", "")))
        summary.setWordWrap(True)
        layout.addWidget(summary)

        headers = list(preview_data.get("headers", []))
        rows = list(preview_data.get("rows", []))
        table = QTableWidget(len(rows), len(headers))
        table.setHorizontalHeaderLabels(headers)
        for row_index, row in enumerate(rows):
            for column_index, value in enumerate(row):
                table.setItem(row_index, column_index, QTableWidgetItem(str(value)))
        table.resizeColumnsToContents()
        layout.addWidget(table, 1)

        footer = QHBoxLayout()
        footer.addStretch(1)
        self._close_button = QPushButton(i18n.t("Close"))
        self._close_button.setObjectName("widgetPopupCloseButton")
        self._close_button.clicked.connect(self.accept)
        footer.addWidget(self._close_button)
        layout.addLayout(footer)

    def eventFilter(self, watched, event) -> bool:
        if watched is self._drag_handle:
            if event.type() == QEvent.Type.MouseButtonPress and event.button() == Qt.MouseButton.LeftButton:
                self._drag_handle.setCursor(Qt.CursorShape.ClosedHandCursor)
                self._drag_offset = event.globalPosition().toPoint() - self.frameGeometry().topLeft()
                return True
            if event.type() == QEvent.Type.MouseMove and self._drag_offset is not None and event.buttons() & Qt.MouseButton.LeftButton:
                self.move(event.globalPosition().toPoint() - self._drag_offset)
                return True
            if event.type() == QEvent.Type.MouseButtonRelease:
                self._drag_handle.setCursor(Qt.CursorShape.OpenHandCursor)
                self._drag_offset = None
                return True
        return super().eventFilter(watched, event)


class WidgetSelectedDataDialog(QDialog):
    def __init__(self, title: str, snapshot: dict[str, object], parent: QWidget | None = None) -> None:
        super().__init__(parent, Qt.WindowType.Dialog | Qt.WindowType.FramelessWindowHint)
        self.setObjectName("widgetPopup")
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setWindowTitle(title)
        self.resize(920, 680)
        self._drag_offset: QPoint | None = None
        shadow = QGraphicsDropShadowEffect(self)
        shadow.setBlurRadius(36)
        shadow.setOffset(0, 12)
        shadow.setColor(QColor(54, 41, 24, 70))

        outer_layout = QVBoxLayout(self)
        outer_layout.setContentsMargins(18, 18, 18, 18)
        outer_layout.setSpacing(0)

        self._surface = QFrame(self)
        self._surface.setObjectName("widgetPopupSurface")
        self._surface.setGraphicsEffect(shadow)
        outer_layout.addWidget(self._surface)

        layout = QVBoxLayout(self._surface)
        layout.setContentsMargins(18, 18, 18, 18)
        layout.setSpacing(10)

        self._drag_handle = QFrame(self._surface)
        self._drag_handle.setObjectName("widgetPopupDragHandle")
        self._drag_handle.setFixedHeight(18)
        self._drag_handle.setCursor(Qt.CursorShape.OpenHandCursor)
        self._drag_handle.installEventFilter(self)
        layout.addWidget(self._drag_handle)

        selected_summary = QLabel(str(snapshot.get("selected_summary", i18n.t("Selected Data: -"))))
        selected_summary.setWordWrap(True)
        selected_summary.setProperty("sectionTitle", True)
        selected_summary.setStyleSheet("font-size: 10.5pt;")
        layout.addWidget(selected_summary)

        selected_table = QTableWidget(
            len(list(snapshot.get("selected_rows", []))),
            len(list(snapshot.get("selected_headers", []))),
        )
        selected_headers = list(snapshot.get("selected_headers", []))
        selected_rows = list(snapshot.get("selected_rows", []))
        selected_table.setHorizontalHeaderLabels(selected_headers)
        for row_index, row in enumerate(selected_rows):
            for column_index, value in enumerate(row):
                selected_table.setItem(row_index, column_index, QTableWidgetItem(str(value)))
        selected_table.resizeColumnsToContents()
        selected_table.setMinimumHeight(170)
        layout.addWidget(selected_table)

        divider = QFrame(self._surface)
        divider.setFrameShape(QFrame.Shape.HLine)
        divider.setStyleSheet("color: #d7cfbf;")
        layout.addWidget(divider)

        data_summary = QLabel(str(snapshot.get("data_summary", i18n.t("Data: -"))))
        data_summary.setWordWrap(True)
        data_summary.setProperty("sectionTitle", True)
        data_summary.setStyleSheet("font-size: 10.5pt;")
        layout.addWidget(data_summary)

        data_table = QTableWidget(
            len(list(snapshot.get("data_rows", []))),
            len(list(snapshot.get("data_headers", []))),
        )
        data_headers = list(snapshot.get("data_headers", []))
        data_rows = list(snapshot.get("data_rows", []))
        data_table.setHorizontalHeaderLabels(data_headers)
        for row_index, row in enumerate(data_rows):
            for column_index, value in enumerate(row):
                data_table.setItem(row_index, column_index, QTableWidgetItem(str(value)))
        data_table.resizeColumnsToContents()
        layout.addWidget(data_table, 1)

        footer = QHBoxLayout()
        footer.addStretch(1)
        self._close_button = QPushButton(i18n.t("Close"))
        self._close_button.setObjectName("widgetPopupCloseButton")
        self._close_button.clicked.connect(self.accept)
        footer.addWidget(self._close_button)
        layout.addLayout(footer)

    def eventFilter(self, watched, event) -> bool:
        if watched is self._drag_handle:
            if event.type() == QEvent.Type.MouseButtonPress and event.button() == Qt.MouseButton.LeftButton:
                self._drag_handle.setCursor(Qt.CursorShape.ClosedHandCursor)
                self._drag_offset = event.position().toPoint()
                return True
            if event.type() == QEvent.Type.MouseMove and self._drag_offset is not None and event.buttons() & Qt.MouseButton.LeftButton:
                parent_widget = self.parentWidget()
                global_top_left = event.globalPosition().toPoint() - self._drag_offset
                if parent_widget is not None:
                    self.move(parent_widget.mapFromGlobal(global_top_left))
                else:
                    self.move(global_top_left)
                return True
            if event.type() == QEvent.Type.MouseButtonRelease:
                self._drag_handle.setCursor(Qt.CursorShape.OpenHandCursor)
                self._drag_offset = None
                return True
        return super().eventFilter(watched, event)


class WidgetScreenDialog(QDialog):
    DEFAULT_POPUP_PROFILE = {
        "base_width": 760,
        "base_height": 640,
        "width_padding": 64,
        "height_padding": 120,
        "min_width": 520,
        "min_height": 420,
        "scroll": True,
    }
    SAVE_DATA_POPUP_PROFILE = {
        "base_width": 600,
        "base_height": 300,
        "width_padding": 24,
        "height_padding": 70,
        "min_width": 560,
        "min_height": 260,
        "scroll": False,
    }

    def __init__(
        self,
        widget_definition: WidgetDefinition,
        screen: QWidget,
        parent: QWidget | None = None,
        data_preview_provider=None,
        data_preview_enabled_provider=None,
    ) -> None:
        super().__init__(parent, Qt.WindowType.SubWindow | Qt.WindowType.FramelessWindowHint)
        self._widget_definition = widget_definition
        self._screen = screen
        self._data_preview_provider = data_preview_provider
        self._data_preview_enabled_provider = data_preview_enabled_provider
        self.setWindowTitle(widget_definition.label)
        self.setModal(False)
        popup_profile = self._popup_profile_for_widget(widget_definition.id)
        size_hint = screen.sizeHint()
        preferred_width = max(int(popup_profile["base_width"]), size_hint.width() + int(popup_profile["width_padding"]))
        preferred_height = max(int(popup_profile["base_height"]), size_hint.height() + int(popup_profile["height_padding"]))
        self.resize(preferred_width, preferred_height)
        self._preferred_size = QSize(preferred_width, preferred_height)
        self._popup_profile = popup_profile
        self.setObjectName("widgetPopup")
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self._drag_offset: QPoint | None = None
        
        outer_layout = QVBoxLayout(self)
        outer_layout.setContentsMargins(18, 18, 18, 18)
        outer_layout.setSpacing(0)

        self._surface = QFrame(self)
        self._surface.setObjectName("widgetPopupSurface")
        outer_layout.addWidget(self._surface)

        layout = QVBoxLayout(self._surface)
        layout.setContentsMargins(16, 16, 16, 14)
        layout.setSpacing(10)

        self._drag_handle = QFrame(self._surface)
        self._drag_handle.setObjectName("widgetPopupDragHandle")
        self._drag_handle.setFixedHeight(18)
        self._drag_handle.setCursor(Qt.CursorShape.OpenHandCursor)
        self._drag_handle.installEventFilter(self)
        layout.addWidget(self._drag_handle)
        self._scroll_area = None
        if not bool(popup_profile["scroll"]):
            screen.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
            layout.addWidget(screen, 0)
        else:
            screen.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
            self._scroll_area = QScrollArea(self._surface)
            self._scroll_area.setWidgetResizable(True)
            self._scroll_area.setFrameShape(QFrame.Shape.NoFrame)
            self._scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
            self._scroll_area.setWidget(screen)
            layout.addWidget(self._scroll_area, 1)
        layout.addSpacing(2)

        footer = QHBoxLayout()
        footer.setSpacing(8)

        self._menu_button = self._make_footer_tool("menu", i18n.t("Open widget menu"))
        self._menu_button.clicked.connect(self._show_menu)
        footer.addWidget(self._menu_button)

        self._help_button = self._make_footer_tool("help", i18n.t("Help"))
        self._help_button.clicked.connect(self._show_help)
        footer.addWidget(self._help_button)

        self._data_button = self._make_footer_tool("data", i18n.t("Open data preview"))
        self._data_button.clicked.connect(self._show_data_preview)
        footer.addWidget(self._data_button)

        self._selected_data_button = None
        if callable(getattr(self._screen, "detailed_data_snapshot", None)):
            self._selected_data_button = self._make_footer_tool("preview", i18n.t("Open selected data preview"))
            self._selected_data_button.clicked.connect(self._show_selected_data_preview)
            footer.addWidget(self._selected_data_button)

        self._status_label = QLabel(self._footer_status_text())
        self._status_label.setProperty("muted", True)
        footer.addWidget(self._status_label)

        footer.addStretch(1)

        self._close_button = QPushButton(i18n.t("Close"))
        self._close_button.setObjectName("widgetPopupCloseButton")
        self._close_button.clicked.connect(self.hide)
        footer.addWidget(self._close_button)
        layout.addLayout(footer)

    def _popup_profile_for_widget(self, widget_id: str) -> dict[str, int | bool]:
        if widget_id == "save-data":
            return dict(self.SAVE_DATA_POPUP_PROFILE)
        return dict(self.DEFAULT_POPUP_PROFILE)

    def closeEvent(self, event: QCloseEvent) -> None:
        self.hide()
        event.ignore()

    def refresh_footer(self) -> None:
        self._status_label.setText(self._footer_status_text())
        if callable(self._data_preview_enabled_provider):
            enabled = bool(self._data_preview_enabled_provider())
            self._data_button.setEnabled(enabled)

    def update_widget_definition(self, widget_definition: WidgetDefinition) -> None:
        self._widget_definition = widget_definition

    def refresh_translations(self) -> None:
        self._menu_button.setToolTip(i18n.t("Open widget menu"))
        self._help_button.setToolTip(i18n.t("Help"))
        self._data_button.setToolTip(i18n.t("Open data preview"))
        if self._selected_data_button is not None:
            self._selected_data_button.setToolTip(i18n.t("Open selected data preview"))
        self._close_button.setText(i18n.t("Close"))
        i18n.apply_to_widget(self)
        self.refresh_footer()

    def _make_footer_tool(self, icon_name: str, tooltip: str) -> QToolButton:
        button = QToolButton(self)
        button.setIcon(get_toolbar_icon(icon_name))
        button.setIconSize(QSize(18, 18))
        button.setToolTip(tooltip)
        button.setObjectName("widgetPopupToolButton")
        button.setCursor(Qt.CursorShape.PointingHandCursor)
        button.setMouseTracking(True)
        original_enter = button.enterEvent
        original_leave = button.leaveEvent

        def enter_event(event) -> None:
            QToolTip.showText(QCursor.pos(), button.toolTip(), button)
            original_enter(event)

        def leave_event(event) -> None:
            QToolTip.hideText()
            original_leave(event)

        button.enterEvent = enter_event
        button.leaveEvent = leave_event
        return button

    def _show_menu(self) -> None:
        menu = QMenu(self)
        self._build_file_submenu(menu.addMenu(i18n.t("File")))
        self._build_view_submenu(menu.addMenu(i18n.t("View")))
        self._build_window_submenu(menu.addMenu(i18n.t("Window")))
        self._build_help_submenu(menu.addMenu(i18n.t("Help")))
        menu.exec(self._menu_button.mapToGlobal(self._menu_button.rect().bottomLeft()))

    def _build_file_submenu(self, submenu: QMenu) -> None:
        actions_added = False

        if hasattr(self._screen, "_browse_button"):
            browse_action = submenu.addAction(i18n.t("Open Dataset..."))
            browse_action.triggered.connect(lambda: getattr(self._screen, "_browse_button").click())
            actions_added = True

        if hasattr(self._screen, "_reload_button"):
            reload_action = submenu.addAction(i18n.t("Reload Source"))
            reload_action.triggered.connect(lambda: getattr(self._screen, "_reload_button").click())
            reload_action.setEnabled(getattr(self._screen, "_reload_button").isEnabled())
            actions_added = True

        if hasattr(self._screen, "_apply_button"):
            apply_action = submenu.addAction("Apply")
            apply_action.triggered.connect(lambda: getattr(self._screen, "_apply_button").click())
            apply_action.setEnabled(getattr(self._screen, "_apply_button").isEnabled())
            actions_added = True

        if hasattr(self._screen, "_reset_button"):
            reset_action = submenu.addAction("Reset")
            reset_action.triggered.connect(lambda: getattr(self._screen, "_reset_button").click())
            reset_action.setEnabled(getattr(self._screen, "_reset_button").isEnabled())
            actions_added = True

        if hasattr(self._screen, "_save_button"):
            save_action = submenu.addAction("Save")
            save_action.triggered.connect(lambda: getattr(self._screen, "_save_button").click())
            save_action.setEnabled(getattr(self._screen, "_save_button").isEnabled())
            actions_added = True

        if hasattr(self._screen, "_save_as_button"):
            save_as_action = submenu.addAction(i18n.t("Save As..."))
            save_as_action.triggered.connect(lambda: getattr(self._screen, "_save_as_button").click())
            save_as_action.setEnabled(getattr(self._screen, "_save_as_button").isEnabled())
            actions_added = True

        if actions_added:
            submenu.addSeparator()

        close_action = submenu.addAction(i18n.t("Close"))
        close_action.triggered.connect(self.hide)

    def _build_view_submenu(self, submenu: QMenu) -> None:
        data_action = submenu.addAction(i18n.t("Data Preview"))
        data_action.triggered.connect(self._show_data_preview)
        data_action.setEnabled(self._data_button.isEnabled())

        if self._selected_data_button is not None:
            selected_action = submenu.addAction(i18n.t("Selected Data Preview"))
            selected_action.triggered.connect(self._show_selected_data_preview)

        status_action = submenu.addAction(i18n.tf("Status: {status}", status=self._footer_status_text() or "-"))
        status_action.setEnabled(False)

    def _build_window_submenu(self, submenu: QMenu) -> None:
        center_action = submenu.addAction(i18n.t("Center"))
        center_action.triggered.connect(self._center_in_parent)

        raise_action = submenu.addAction(i18n.t("Bring to Front"))
        raise_action.triggered.connect(self._raise_window)

        pin_action = submenu.addAction(i18n.t("Always on Top"))
        pin_action.setCheckable(True)
        pin_action.setChecked(bool(self.windowFlags() & Qt.WindowType.WindowStaysOnTopHint))
        pin_action.triggered.connect(self._toggle_always_on_top)

    def _build_help_submenu(self, submenu: QMenu) -> None:
        help_action = submenu.addAction(i18n.t("Widget Help"))
        help_action.triggered.connect(self._show_help)

        documentation_url = getattr(self._screen, "documentation_url", None)
        if callable(documentation_url):
            docs_action = submenu.addAction(i18n.t("Documentation"))
            docs_action.triggered.connect(lambda: QDesktopServices.openUrl(QUrl(documentation_url())))

    def _center_in_parent(self) -> None:
        parent_widget = self.parentWidget()
        if parent_widget is not None:
            target_rect = parent_widget.rect()
            self.move(
                max(0, (target_rect.width() - self.width()) // 2),
                max(0, (target_rect.height() - self.height()) // 2),
            )
            return

        screen = QApplication.primaryScreen()
        if screen is None:
            return
        available = screen.availableGeometry()
        self.move(
            available.left() + max(0, (available.width() - self.width()) // 2),
            available.top() + max(0, (available.height() - self.height()) // 2),
        )

    def _raise_window(self) -> None:
        self.show()
        self.raise_()
        self.activateWindow()

    def _toggle_always_on_top(self, checked: bool) -> None:
        geometry = self.geometry()
        self.hide()
        flags = self.windowFlags()
        if checked:
            flags |= Qt.WindowType.WindowStaysOnTopHint
        else:
            flags &= ~Qt.WindowType.WindowStaysOnTopHint
        self.setWindowFlags(flags)
        self.show()
        self.setGeometry(geometry)
        self.raise_()
        self.activateWindow()

    def _show_help(self) -> None:
        help_text = getattr(self._screen, "help_text", lambda: self._widget_definition.description or i18n.t("No help available."))()
        QMessageBox.information(self, f"{self.windowTitle()} {i18n.t('Help')}", help_text)

    def _show_data_preview(self) -> None:
        if callable(self._data_preview_enabled_provider) and not self._data_preview_enabled_provider():
            return
        preview_data = None
        if callable(self._data_preview_provider):
            preview_data = self._data_preview_provider()
        if not preview_data:
            preview_data = getattr(
                self._screen,
                "data_preview_snapshot",
                lambda: {"summary": i18n.t("No preview available."), "headers": [], "rows": []},
            )()
        dialog = WidgetDataPreviewDialog(f"{self.windowTitle()} - {i18n.t('Data Preview')}", preview_data, self)
        dialog.exec()

    def _show_selected_data_preview(self) -> None:
        snapshot = getattr(
            self._screen,
            "detailed_data_snapshot",
            lambda: {
                "selected_summary": i18n.t("Selected Data: -"),
                "selected_headers": [],
                "selected_rows": [],
                "data_summary": i18n.t("Data: -"),
                "data_headers": [],
                "data_rows": [],
            },
        )()
        dialog = WidgetSelectedDataDialog(f"{self.windowTitle()} - {i18n.t('Selected Data Preview')}", snapshot, self)
        dialog.exec()

    def _footer_status_text(self) -> str:
        provider = getattr(self._screen, "footer_status_text", None)
        if callable(provider):
            return str(provider())
        return ""

    def eventFilter(self, watched, event) -> bool:
        if watched is self._drag_handle:
            if event.type() == QEvent.Type.MouseButtonPress and event.button() == Qt.MouseButton.LeftButton:
                self._drag_handle.setCursor(Qt.CursorShape.ClosedHandCursor)
                self._drag_offset = event.globalPosition().toPoint() - self.frameGeometry().topLeft()
                return True
            if event.type() == QEvent.Type.MouseMove and self._drag_offset is not None and event.buttons() & Qt.MouseButton.LeftButton:
                self.move(event.globalPosition().toPoint() - self._drag_offset)
                return True
            if event.type() == QEvent.Type.MouseButtonRelease:
                self._drag_handle.setCursor(Qt.CursorShape.OpenHandCursor)
                self._drag_offset = None
                return True
        return super().eventFilter(watched, event)

class WorkflowQuickTools(QFrame):
    textAnnotationRequested = Signal()
    arrowAnnotationRequested = Signal()
    layoutChanged = Signal()

    def __init__(self, canvas: WorkflowCanvas, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("workflowQuickTools")
        self._canvas = canvas
        self._expanded = False

        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        self._pan_button = self._make_button("pan", i18n.t("Pan canvas"), checkable=True)
        self._pan_button.toggled.connect(self._canvas.set_pan_mode)
        layout.addWidget(self._pan_button)

        self._zoom_out_button = self._make_button("zoom_out", i18n.t("Zoom out"))
        self._zoom_out_button.clicked.connect(self._canvas.zoom_out)
        layout.addWidget(self._zoom_out_button)

        self._zoom_in_button = self._make_button("zoom_in", i18n.t("Zoom in"))
        self._zoom_in_button.clicked.connect(self._canvas.zoom_in)
        layout.addWidget(self._zoom_in_button)

        self._reset_button = self._make_button("reset", i18n.t("Reset zoom"))
        self._reset_button.clicked.connect(self._canvas.reset_zoom)
        layout.addWidget(self._reset_button)

        self._text_button = self._make_button("text", i18n.t("Add text annotation"))
        self._text_button.clicked.connect(self.textAnnotationRequested.emit)
        layout.addWidget(self._text_button)

        self._arrow_button = self._make_button("arrow", i18n.t("Add arrow annotation"))
        self._arrow_button.clicked.connect(self.arrowAnnotationRequested.emit)
        layout.addWidget(self._arrow_button)

        self._expand_button = self._make_button("expand", i18n.t("Show more tools"))
        self._expand_button.clicked.connect(self.toggle_expanded)
        layout.addWidget(self._expand_button)

        self._sync_visibility()

    def _make_button(self, icon_name: str, tooltip: str, *, checkable: bool = False) -> QToolButton:
        button = QToolButton(self)
        button.setIcon(get_toolbar_icon(icon_name))
        button.setIconSize(QSize(18, 18))
        button.setCheckable(checkable)
        button.setCursor(Qt.CursorShape.PointingHandCursor)
        button.setAutoRaise(False)
        button.setToolTip(tooltip)
        button.setMouseTracking(True)
        original_enter = button.enterEvent
        original_leave = button.leaveEvent

        def enter_event(event) -> None:
            QToolTip.showText(QCursor.pos(), button.toolTip(), button)
            original_enter(event)

        def leave_event(event) -> None:
            QToolTip.hideText()
            original_leave(event)

        button.enterEvent = enter_event
        button.leaveEvent = leave_event
        return button

    def toggle_expanded(self) -> None:
        self._expanded = not self._expanded
        self._sync_visibility()

    def is_expanded(self) -> bool:
        return self._expanded

    def refresh_translations(self) -> None:
        self._pan_button.setToolTip(i18n.t("Pan canvas"))
        self._zoom_out_button.setToolTip(i18n.t("Zoom out"))
        self._zoom_in_button.setToolTip(i18n.t("Zoom in"))
        self._reset_button.setToolTip(i18n.t("Reset zoom"))
        self._text_button.setToolTip(i18n.t("Add text annotation"))
        self._arrow_button.setToolTip(i18n.t("Add arrow annotation"))
        self._sync_visibility()

    def _sync_visibility(self) -> None:
        for widget in (self._reset_button, self._text_button, self._arrow_button):
            widget.setVisible(self._expanded)
        self._expand_button.setIcon(get_toolbar_icon("collapse" if self._expanded else "expand"))
        self._expand_button.setToolTip(i18n.t("Hide extra tools") if self._expanded else i18n.t("Show more tools"))
        self.adjustSize()
        self.layoutChanged.emit()


class WorkflowMiniMap(QFrame):
    def __init__(self, canvas: WorkflowCanvas, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("workflowMiniMap")
        self._canvas = canvas
        self.setFixedSize(176, 120)
        self.hide()
        self._hide_timer = QTimer(self)
        self._hide_timer.setSingleShot(True)
        self._hide_timer.timeout.connect(self.hide)
        self._canvas.viewChanged.connect(self.show_temporarily)

    def show_temporarily(self) -> None:
        scene = self._canvas.workflow_scene
        if not scene._nodes and not scene._annotations and not scene._edges:
            return
        self.show()
        self.raise_()
        self.update()
        self._hide_timer.start(1400)

    def paintEvent(self, event) -> None:
        super().paintEvent(event)
        scene = self._canvas.workflow_scene
        scene_rect = scene.itemsBoundingRect().united(scene.sceneRect()).adjusted(-36, -36, 36, 36)
        if scene_rect.isNull():
            return

        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        inner = self.rect().adjusted(10, 10, -10, -10)
        scale = min(inner.width() / scene_rect.width(), inner.height() / scene_rect.height())
        if scale <= 0:
            painter.end()
            return

        def map_rect(rect):
            x = inner.left() + ((rect.left() - scene_rect.left()) * scale)
            y = inner.top() + ((rect.top() - scene_rect.top()) * scale)
            return rect.__class__(x, y, rect.width() * scale, rect.height() * scale)

        def map_point(point):
            x = inner.left() + ((point.x() - scene_rect.left()) * scale)
            y = inner.top() + ((point.y() - scene_rect.top()) * scale)
            return QPointF(x, y)

        painter.setBrush(QColor(255, 252, 246, 230))
        painter.setPen(QPen(QColor("#d6c7b3"), 1))
        painter.drawRoundedRect(inner, 10, 10)

        for node in scene._nodes.values():
            node_rect = node.sceneBoundingRect()
            mini_rect = map_rect(node_rect)
            painter.setBrush(QColor("#e7d0a7"))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawRoundedRect(mini_rect, 4, 4)

        for annotation in scene._annotations:
            mini_rect = map_rect(annotation.sceneBoundingRect())
            painter.setBrush(QColor(0, 0, 0, 0))
            painter.setPen(QPen(QColor("#b59d7d"), 1))
            painter.drawRoundedRect(mini_rect, 3, 3)

        for edge in scene._edges:
            path = edge.path()
            painter.setPen(QPen(QColor("#b5b5b5"), 1))
            painter.drawLine(map_point(path.pointAtPercent(0.0)), map_point(path.pointAtPercent(1.0)))

        view_rect = self._canvas.mapToScene(self._canvas.viewport().rect()).boundingRect()
        painter.setBrush(QColor(226, 169, 82, 40))
        painter.setPen(QPen(QColor("#cf9440"), 1.2))
        painter.drawRoundedRect(map_rect(view_rect), 4, 4)
        painter.end()


class WorkflowWorkspace(QFrame):
    textAnnotationRequested = Signal()
    arrowAnnotationRequested = Signal()

    def __init__(self, widget_index: dict[str, WidgetDefinition], parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("contentPanel")
        self._widget_index = widget_index
        self._screens: dict[str, QWidget] = {}
        self._screen_widget_ids: dict[str, str] = {}
        self._dialogs: dict[str, WidgetScreenDialog] = {}
        self._data_preview_providers: dict[str, object] = {}
        self._data_preview_enabled_providers: dict[str, object] = {}
        self._last_opened_node_id: str | None = None
        self._dialogs_on_top = False

        self._outer_layout = QVBoxLayout(self)
        self._outer_layout.setContentsMargins(18, 18, 18, 18)
        self._outer_layout.setSpacing(14)

        self._title_label = QLabel(i18n.t("Workflow"))
        self._title_label.setProperty("sectionTitle", True)
        self._outer_layout.addWidget(self._title_label)

        self._subtitle_label = QLabel(
            i18n.t(
                "Drag widgets from the catalog onto the canvas. Click a node to inspect it. Build connections by dragging from an output port onto a compatible input port."
            )
        )
        self._subtitle_label.setProperty("muted", True)
        self._subtitle_label.setWordWrap(True)
        self._outer_layout.addWidget(self._subtitle_label)

        self._canvas = WorkflowCanvas(widget_index)
        self._outer_layout.addWidget(self._canvas, 1)
        self._quick_tools = WorkflowQuickTools(self._canvas, self)
        self._quick_tools.textAnnotationRequested.connect(self.textAnnotationRequested.emit)
        self._quick_tools.arrowAnnotationRequested.connect(self.arrowAnnotationRequested.emit)
        self._quick_tools.layoutChanged.connect(self._position_overlays)
        self._quick_tools.raise_()

        self._mini_map = WorkflowMiniMap(self._canvas, self)

    @property
    def canvas(self) -> WorkflowCanvas:
        return self._canvas

    @property
    def quick_tools(self) -> WorkflowQuickTools:
        return self._quick_tools

    @property
    def mini_map(self) -> WorkflowMiniMap:
        return self._mini_map

    def register_node_screen(
        self,
        node_id: str,
        widget_id: str,
        screen: QWidget,
        *,
        data_preview_provider=None,
        data_preview_enabled_provider=None,
    ) -> None:
        self._screens[node_id] = screen
        self._screen_widget_ids[node_id] = widget_id
        self._data_preview_providers[node_id] = data_preview_provider
        self._data_preview_enabled_providers[node_id] = data_preview_enabled_provider

    def show_widget(self, node_id: str) -> None:
        dialog = self._dialogs.get(node_id)
        if dialog is None:
            widget_id = self._screen_widget_ids[node_id]
            dialog = WidgetScreenDialog(
                self._widget_index[widget_id],
                self._screens[node_id],
                self.window(),
                data_preview_provider=self._data_preview_providers.get(node_id),
                data_preview_enabled_provider=self._data_preview_enabled_providers.get(node_id),
            )
            self._apply_dialog_flags(dialog)
            self._dialogs[node_id] = dialog
        record = self._canvas.workflow_scene.node_record(node_id)
        if record is not None:
            dialog.setWindowTitle(record.label)
        dialog.refresh_translations()
        dialog.refresh_footer()
        self._fit_dialog_to_screen(dialog)
        dialog.show()
        QApplication.processEvents()
        self._center_dialog(dialog)
        dialog.raise_()
        dialog.activateWindow()
        self._last_opened_node_id = node_id

    def add_workflow_node(self, widget_id: str):
        return self._canvas.add_workflow_node(widget_id)

    def update_widget_index(self, widget_index: dict[str, WidgetDefinition]) -> None:
        self._widget_index = widget_index
        self._canvas.update_widget_index(widget_index)
        for node_id, dialog in self._dialogs.items():
            widget_id = self._screen_widget_ids.get(node_id)
            if widget_id is None:
                continue
            widget_definition = widget_index.get(widget_id)
            if widget_definition is not None:
                dialog.update_widget_definition(widget_definition)
        self.refresh_translations()

    def current_widget(self) -> QWidget:
        if self._last_opened_node_id is None:
            return None
        return self._screens[self._last_opened_node_id]

    def all_screens(self) -> list[QWidget]:
        return list(self._screens.values())

    def screen_items(self) -> list[tuple[str, QWidget]]:
        return list(self._screens.items())

    def screen_for_node(self, node_id: str) -> QWidget | None:
        return self._screens.get(node_id)

    def is_widget_dialog_visible(self, node_id: str) -> bool:
        dialog = self._dialogs.get(node_id)
        return bool(dialog and dialog.isVisible())

    def close_all_dialogs(self) -> None:
        for dialog in self._dialogs.values():
            dialog.hide()

    def close_active_dialog(self) -> bool:
        active_window = QApplication.activeWindow()
        if isinstance(active_window, WidgetScreenDialog):
            active_window.close()
            return True
        return False

    def visible_dialogs(self) -> dict[str, WidgetScreenDialog]:
        return {node_id: dialog for node_id, dialog in self._dialogs.items() if dialog.isVisible()}

    def refresh_dialog_footers(self, node_id: str | None = None) -> None:
        if node_id is not None:
            dialog = self._dialogs.get(node_id)
            if dialog is not None:
                dialog.refresh_footer()
            return
        for dialog in self._dialogs.values():
            dialog.refresh_footer()

    def refresh_translations(self) -> None:
        self._title_label.setText(i18n.t("Workflow"))
        self._subtitle_label.setText(
            i18n.t(
                "Drag widgets from the catalog onto the canvas. Click a node to inspect it. Build connections by dragging from an output port onto a compatible input port."
            )
        )
        self._quick_tools.refresh_translations()
        for node_id, dialog in self._dialogs.items():
            record = self._canvas.workflow_scene.node_record(node_id)
            if record is not None:
                dialog.setWindowTitle(record.label)
            dialog.refresh_translations()

    def remove_node(self, node_id: str) -> None:
        dialog = self._dialogs.pop(node_id, None)
        if dialog is not None:
            dialog.hide()
            dialog.deleteLater()
        screen = self._screens.pop(node_id, None)
        if screen is not None and screen.parent() is None:
            screen.deleteLater()
        self._screen_widget_ids.pop(node_id, None)
        self._data_preview_providers.pop(node_id, None)
        self._data_preview_enabled_providers.pop(node_id, None)
        if self._last_opened_node_id == node_id:
            self._last_opened_node_id = None

    def raise_all_dialogs(self) -> None:
        for dialog in self.visible_dialogs().values():
            dialog.show()
            dialog.raise_()
            dialog.activateWindow()

    def set_dialogs_on_top(self, enabled: bool) -> None:
        self._dialogs_on_top = enabled
        for dialog in self._dialogs.values():
            was_visible = dialog.isVisible()
            self._apply_dialog_flags(dialog)
            if was_visible:
                dialog.show()

    def dialogs_on_top(self) -> bool:
        return self._dialogs_on_top

    def set_margins_visible(self, visible: bool) -> None:
        margin = 18 if visible else 0
        self._outer_layout.setContentsMargins(margin, margin, margin, margin)

    def margins_visible(self) -> bool:
        return self._outer_layout.contentsMargins().left() > 0

    def reset_widget_settings(self) -> None:
        self.set_dialogs_on_top(False)
        for dialog in self._dialogs.values():
            dialog.resize(dialog._preferred_size)
            self._center_dialog(dialog)

    def _center_dialog(self, dialog: QDialog) -> None:
        parent_widget = dialog.parentWidget()
        dialog_size = dialog.size()
        if parent_widget is not None:
            target_rect = parent_widget.rect()
            left = max(0, (target_rect.width() - dialog_size.width()) // 2)
            top = max(0, (target_rect.height() - dialog_size.height()) // 2)
            dialog.move(left, top)
            return

        target_area = self._dialog_target_area()
        left = target_area.left() + max(0, (target_area.width() - dialog_size.width()) // 2)
        top = target_area.top() + max(0, (target_area.height() - dialog_size.height()) // 2)
        dialog.move(left, top)

    def _fit_dialog_to_screen(self, dialog: WidgetScreenDialog) -> None:
        parent_widget = dialog.parentWidget()
        if parent_widget is not None:
            target_width_limit = parent_widget.width() - 40
            target_height_limit = parent_widget.height() - 40
        else:
            target_area = self._dialog_target_area()
            target_width_limit = target_area.width() - 32
            target_height_limit = target_area.height() - 32
        preferred = dialog._preferred_size
        max_width = max(700, target_width_limit)
        max_height = max(520, target_height_limit)
        width = min(preferred.width(), max_width, target_width_limit)
        height = min(preferred.height(), max_height, target_height_limit)
        min_width = int(dialog._popup_profile["min_width"])
        min_height = int(dialog._popup_profile["min_height"])
        if dialog._widget_definition.id == "save-data":
            min_height = max(min_height, dialog._screen.sizeHint().height() + 120)
        target_width = max(min_width, width)
        target_height = max(min_height, height)
        dialog.setMinimumSize(min_width, min_height)
        dialog.setMaximumSize(target_width, target_height)
        dialog.resize(target_width, target_height)

    def _dialog_target_area(self) -> QRect:
        parent_widget = self.window()
        if isinstance(parent_widget, QWidget) and parent_widget.screen() is not None:
            return parent_widget.screen().availableGeometry()
        screen = QApplication.primaryScreen()
        return screen.availableGeometry() if screen is not None else QRect(0, 0, 1280, 800)

    def _apply_dialog_flags(self, dialog: QDialog) -> None:
        flags = dialog.windowFlags()
        if self._dialogs_on_top:
            flags |= Qt.WindowType.WindowStaysOnTopHint
        else:
            flags &= ~Qt.WindowType.WindowStaysOnTopHint
        dialog.setWindowFlags(flags)

    def resizeEvent(self, event: QResizeEvent) -> None:
        super().resizeEvent(event)
        self._position_overlays()

    def _position_overlays(self) -> None:
        self._quick_tools.adjustSize()
        self._quick_tools.move(
            self.width() - self._quick_tools.width() - 18,
            self.height() - self._quick_tools.height() - 18,
        )
        self._mini_map.move(18, self.height() - self._mini_map.height() - 18)
