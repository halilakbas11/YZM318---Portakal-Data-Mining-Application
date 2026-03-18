from __future__ import annotations

import copy
import json
import re
from collections import defaultdict
from pathlib import Path

from PySide6.QtCore import QPoint, QPointF, QRect, Qt, QUrl
from PySide6.QtGui import QAction, QActionGroup, QCloseEvent, QDesktopServices, QKeySequence
from PySide6.QtWidgets import (
    QApplication,
    QDialog,
    QFrame,
    QFileDialog,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMenu,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from portakal_app.data.errors import PortakalDataError
from portakal_app.data.models import DatasetHandle
from portakal_app.data.services.file_import_service import FileImportService
from portakal_app.data.services.preview_service import PreviewService
from portakal_app.data.services.save_data_service import SaveDataService
from portakal_app.models import AppState, DataInfoViewModel, MetricCardData, SuggestionItem, WidgetDefinition
from portakal_app.ui.catalog import build_categories, build_widgets
from portakal_app.ui.screens.data_info_screen import DataInfoScreen
from portakal_app.ui.screens.data_table_screen import DataTableScreen
from portakal_app.ui.screens.file_screen import FileScreen
from portakal_app.ui.screens.save_data_screen import SaveDataScreen
from portakal_app.ui.shell.sidebar import SidebarCategoryList
from portakal_app.ui.shell.state_store import AppStateStore
from portakal_app.ui.shell.status_bar import StatusBarController
from portakal_app.ui.shell.widget_catalog import WidgetCatalogPanel
from portakal_app.ui.shell.workflow_canvas import WorkflowScene
from portakal_app.ui.shell.workflow_workspace import WorkflowWorkspace


class WorkflowInfoDialog(QDialog):
    def __init__(self, title: str, description: str, parent: QWidget | None = None) -> None:
        super().__init__(parent, Qt.WindowType.Dialog | Qt.WindowType.FramelessWindowHint)
        self.setWindowTitle("Workflow Info")
        self.setModal(True)
        self.resize(760, 560)
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
        layout.setContentsMargins(24, 24, 24, 16)
        layout.setSpacing(12)

        title_header = QLabel("Title")
        title_header.setProperty("sectionTitle", True)
        layout.addWidget(title_header)

        self.title_input = QLineEdit(title or "Untitled")
        self.title_input.setObjectName("workflowInfoTitleInput")
        layout.addWidget(self.title_input)

        description_header = QLabel("Description")
        description_header.setProperty("sectionTitle", True)
        layout.addWidget(description_header)

        self.description_input = QTextEdit(description)
        self.description_input.setObjectName("workflowInfoDescriptionInput")
        layout.addWidget(self.description_input, 1)

        footer = QHBoxLayout()
        footer.addStretch(1)

        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.setProperty("secondary", True)
        self.cancel_button.setAutoDefault(False)
        self.cancel_button.clicked.connect(self.reject)
        footer.addWidget(self.cancel_button)

        self.save_button = QPushButton("Save")
        self.save_button.setProperty("primary", True)
        self.save_button.setDefault(True)
        self.save_button.setAutoDefault(True)
        self.save_button.clicked.connect(self.accept)
        footer.addWidget(self.save_button)
        layout.addLayout(footer)

        self.title_input.returnPressed.connect(self.accept)

    def workflow_title(self) -> str:
        return self.title_input.text().strip() or "Untitled"

    def workflow_description(self) -> str:
        return self.description_input.toPlainText().strip()

    def closeEvent(self, event: QCloseEvent) -> None:
        self.reject()
        event.accept()

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag_offset = event.globalPosition().toPoint() - self.frameGeometry().topLeft()
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:
        if self._drag_offset is not None and event.buttons() & Qt.MouseButton.LeftButton:
            self.move(event.globalPosition().toPoint() - self._drag_offset)
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:
        self._drag_offset = None
        super().mouseReleaseEvent(event)


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self._apply_initial_geometry()

        self._categories = build_categories()
        self._widgets = build_widgets()
        self._widgets_by_category = defaultdict(list)
        self._widget_index: dict[str, WidgetDefinition] = {}
        for widget in self._widgets:
            self._widgets_by_category[widget.category_id].append(widget)
            self._widget_index[widget.id] = widget

        self._state_store = AppStateStore(AppState())
        self._status_bar = StatusBarController()
        self.setStatusBar(self._status_bar)
        self._state_store.stateChanged.connect(self._on_state_changed)

        self._actions: dict[str, QAction] = {}
        self._menus: dict[str, QMenu] = {}
        self._recent_workflow_paths: list[str] = []
        self._current_workflow_path: str | None = None
        self._last_workflow_path: str | None = None
        self._workflow_modified = False
        self._workflow_frozen = False
        self._tool_dock_expanded = True
        self._status_log: list[str] = []
        self._undo_history: list[dict[str, object]] = []
        self._redo_history: list[dict[str, object]] = []
        self._saved_snapshot: dict[str, object] | None = None
        self._history_guard = False
        self._file_import_service = FileImportService()
        self._preview_service = PreviewService()
        self._save_data_service = SaveDataService()

        self._build_layout()
        self._register_screens()
        self._build_menu()
        self._connect_scene_signals()
        QApplication.clipboard().dataChanged.connect(self._update_action_states)

        self._sidebar.set_current_category(self._state_store.state.selected_category)
        self._on_category_selected(self._state_store.state.selected_category)
        self._reset_history(self._serialize_workflow())
        self._refresh_window_title()
        self._update_action_states()

    @property
    def state(self) -> AppState:
        return self._state_store.state

    def _build_menu(self) -> None:
        menu_bar = self.menuBar()
        menu_bar.clear()
        self._actions.clear()
        self._menus.clear()

        self._build_file_menu(menu_bar.addMenu("File"))
        self._build_edit_menu(menu_bar.addMenu("Edit"))
        self._build_view_menu(menu_bar.addMenu("View"))
        self._build_widget_menu(menu_bar.addMenu("Widget"))
        self._build_window_menu(menu_bar.addMenu("Window"))
        self._build_options_menu(menu_bar.addMenu("Options"))
        self._build_help_menu(menu_bar.addMenu("Help"))

    def _build_file_menu(self, menu: QMenu) -> None:
        self._menus["file"] = menu
        menu.addAction(self._create_action("file_new", "New", self._new_workflow, QKeySequence.StandardKey.New))
        menu.addAction(self._create_action("file_open", "Open", self._open_workflow_dialog, QKeySequence.StandardKey.Open))
        menu.addAction(
            self._create_action("file_open_freeze", "Open and Freeze", self._open_and_freeze_workflow, "Ctrl+Alt+O")
        )
        menu.addAction(
            self._create_action("file_reload", "Reload Last Workflow", self._reload_last_workflow, "Ctrl+R")
        )
        self._recent_menu = menu.addMenu("Open Recent")
        self._recent_menu.aboutToShow.connect(self._populate_recent_workflows_menu)
        menu.addAction(self._create_action("file_close_window", "Close Window", self._close_window, "Ctrl+F4"))
        menu.addSeparator()
        menu.addAction(self._create_action("file_save", "Save", self._save_workflow, QKeySequence.StandardKey.Save))
        menu.addAction(self._create_action("file_save_as", "Save As ...", self._save_workflow_as, QKeySequence.StandardKey.SaveAs))
        menu.addAction(
            self._create_action("file_export_svg", "Save Workflow Image as SVG ...", self._export_workflow_image)
        )
        menu.addSeparator()
        menu.addAction(self._create_action("file_info", "Workflow Info", self._show_workflow_info, "Ctrl+I"))
        menu.addAction(self._create_action("file_quit", "Quit", QApplication.instance().quit, QKeySequence.StandardKey.Quit))

    def _build_edit_menu(self, menu: QMenu) -> None:
        self._menus["edit"] = menu
        menu.addAction(self._create_action("edit_undo", "Undo", self._undo_workflow, QKeySequence.StandardKey.Undo))
        menu.addAction(self._create_action("edit_redo", "Redo", self._redo_workflow, QKeySequence.StandardKey.Redo))
        menu.addSeparator()
        menu.addAction(self._create_action("edit_remove", "Remove", self._delete_selection, QKeySequence(Qt.Key.Key_Backspace)))
        menu.addAction(self._create_action("edit_duplicate", "Duplicate", self._duplicate_selection, QKeySequence("Ctrl+D")))
        menu.addAction(self._create_action("edit_copy", "Copy", self._copy_selection, QKeySequence.StandardKey.Copy))
        menu.addAction(self._create_action("edit_paste", "Paste", self._paste_selection, QKeySequence.StandardKey.Paste))
        menu.addAction(self._create_action("edit_select_all", "Select all", self._select_all_workflow_items, QKeySequence.StandardKey.SelectAll))
        menu.addSeparator()
        menu.addAction(self._create_action("edit_text_annotation", "Text Annotation", self._add_text_annotation))
        menu.addAction(self._create_action("edit_arrow_annotation", "Arrow Annotation", self._add_arrow_annotation))

    def _build_view_menu(self, menu: QMenu) -> None:
        self._menus["view"] = menu
        self._window_groups_menu = menu.addMenu("Window Groups")
        self._window_groups_menu.aboutToShow.connect(self._populate_window_groups_menu)
        menu.addSeparator()
        menu.addAction(
            self._create_action(
                "view_expand_dock",
                "Expand Tool Dock",
                self._toggle_tool_dock,
                "Ctrl+Shift+D",
                checkable=True,
                checked=True,
            )
        )
        menu.addAction(self._create_action("view_log", "Log", self._show_log_dialog))
        menu.addSeparator()
        menu.addAction(self._create_action("view_zoom_in", "Zoom in", self._zoom_in, QKeySequence.ZoomIn))
        menu.addAction(self._create_action("view_zoom_out", "Zoom out", self._zoom_out, QKeySequence.ZoomOut))
        menu.addAction(self._create_action("view_zoom_reset", "Reset Zoom", self._reset_zoom, "Ctrl+0"))
        menu.addSeparator()
        menu.addAction(
            self._create_action(
                "view_margins",
                "Show Workflow Margins",
                self._toggle_workflow_margins,
                checkable=True,
                checked=True,
            )
        )

    def _build_widget_menu(self, menu: QMenu) -> None:
        self._menus["widget"] = menu
        menu.addAction(self._create_action("widget_open", "Open", self._open_selected_widget))
        menu.addAction(self._create_action("widget_rename", "Rename", self._rename_selected_widget, "F2"))
        menu.addAction(self._create_action("widget_remove", "Remove", self._delete_selection))
        menu.addSeparator()
        menu.addAction(self._create_action("widget_help", "Help", self._show_selected_widget_help, "F1"))

    def _build_window_menu(self, menu: QMenu) -> None:
        self._menus["window"] = menu
        menu.addAction(self._create_action("window_raise_widgets", "Bring Widgets to Front", self._raise_widget_windows, "Ctrl+Down"))
        menu.addAction(
            self._create_action(
                "window_top",
                "Display Widgets on Top",
                self._toggle_widgets_on_top,
                checkable=True,
                checked=False,
            )
        )
        menu.addSeparator()
        self._workflow_window_group = QActionGroup(self)
        self._workflow_window_group.setExclusive(True)
        self._workflow_window_action = QAction(self)
        self._workflow_window_action.setCheckable(True)
        self._workflow_window_action.setChecked(True)
        self._workflow_window_action.triggered.connect(self.activateWindow)
        self._workflow_window_group.addAction(self._workflow_window_action)
        menu.addAction(self._workflow_window_action)

    def _build_options_menu(self, menu: QMenu) -> None:
        self._menus["options"] = menu
        menu.addAction(self._create_action("options_settings", "Settings", self._show_settings_dialog))
        menu.addAction(self._create_action("options_reset_widgets", "Reset Widget Settings...", self._reset_widget_settings))
        menu.addAction(self._create_action("options_addons", "Add-ons...", self._show_addons_dialog))

    def _build_help_menu(self, menu: QMenu) -> None:
        self._menus["help"] = menu
        menu.addAction(self._create_action("help_docs", "Documentation", self._open_documentation))
        menu.addAction(self._create_action("help_about", "About Portakal", self._show_about_dialog))

    def _create_action(
        self,
        key: str,
        text: str,
        handler,
        shortcut: QKeySequence | str | None = None,
        *,
        checkable: bool = False,
        checked: bool = False,
    ) -> QAction:
        action = QAction(text, self)
        action.setCheckable(checkable)
        if shortcut is not None:
            action.setShortcut(shortcut)
        if checkable:
            action.setChecked(checked)
        action.triggered.connect(handler)
        self._actions[key] = action
        return action

    def _apply_initial_geometry(self) -> None:
        screen = QApplication.primaryScreen()
        if screen is None:
            self.resize(1400, 860)
            return

        available = screen.availableGeometry()
        target_width = min(1400, max(640, available.width() - 80))
        target_height = min(860, max(520, available.height() - 80))
        target_width = min(target_width, available.width())
        target_height = min(target_height, available.height())
        self.resize(target_width, target_height)

        frame = QRect(0, 0, target_width, target_height)
        frame.moveCenter(available.center())
        frame.moveTop(max(available.top() + 24, frame.top()))
        self.setGeometry(frame)

    def _build_layout(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        layout = QHBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self._sidebar = SidebarCategoryList(self._categories)
        self._sidebar.setFixedWidth(self._sidebar.recommended_width())
        self._sidebar.categorySelected.connect(self._on_category_selected)
        self._sidebar.workflowInfoRequested.connect(self._edit_workflow_info)
        layout.addWidget(self._sidebar)

        self._catalog = WidgetCatalogPanel()
        self._catalog.setFixedWidth(320)
        self._catalog.widgetSelected.connect(self._spawn_widget_from_catalog)
        layout.addWidget(self._catalog)

        self._workspace = WorkflowWorkspace(self._widget_index)
        self._workspace.canvas.nodeActivated.connect(self._show_widget)
        self._workspace.canvas.statusMessage.connect(lambda message: self._state_store.update(status_message=message))
        self._workspace.textAnnotationRequested.connect(self._add_text_annotation)
        self._workspace.arrowAnnotationRequested.connect(self._add_arrow_annotation)
        layout.addWidget(self._workspace, 1)

    def _register_screens(self) -> None:
        for widget in self._widgets:
            screen = widget.screen_factory()
            self._workspace.register_screen(widget.id, screen)
            if isinstance(screen, FileScreen):
                screen.on_open_file_requested(self._handle_file_selected)
                screen.on_reload_requested(self._handle_file_selected)
                screen.on_apply_requested(self._handle_file_selected)
            if isinstance(screen, DataInfoScreen):
                self._prime_data_info_screen(screen)
            if isinstance(screen, DataTableScreen):
                screen.set_preview_service(self._preview_service)
                screen.set_dataset(None)
            if isinstance(screen, SaveDataScreen):
                screen.set_save_data_service(self._save_data_service)
                screen.set_dataset(None)

    def _connect_scene_signals(self) -> None:
        scene = self._workspace.canvas.workflow_scene
        scene.workflowChanged.connect(self._handle_workflow_changed)
        scene.selectionChanged.connect(self._update_action_states)

    def _prime_data_info_screen(self, screen: DataInfoScreen) -> None:
        demo_view_model = DataInfoViewModel(
            summary_cards=[
                MetricCardData("Rows", "0", "Loaded by the data team"),
                MetricCardData("Columns", "0", "Schema pending"),
                MetricCardData("Missing", "0%", "Profile service not connected"),
            ],
            column_highlights=["No dataset connected yet."],
            suggestions=[
                SuggestionItem(
                    "LLM placeholder",
                    "Once the analyzer is connected, this list will hold short recommendations and data quality risks.",
                    "info",
                )
            ],
            llm_status="LLM optional: fallback summary mode",
        )
        screen.set_view_model(demo_view_model)
        screen.set_dataset(None)

    def _on_category_selected(self, category_id: str) -> None:
        category = next((item for item in self._categories if item.id == category_id), None)
        if category is None:
            return
        widgets = self._widgets_by_category[category_id]
        self._catalog.set_widgets(category.label, widgets)
        target_widget = widgets[0].id if widgets else self.state.selected_widget
        self._state_store.update(
            selected_category=category_id,
            selected_widget=target_widget,
            status_message=f"{category.label} category selected.",
        )

    def _show_widget(self, widget_id: str) -> None:
        widget = self._widget_index[widget_id]
        self._workspace.show_widget(widget_id)
        self._state_store.update(selected_widget=widget_id, status_message=f"{widget.label} screen opened.")

    def _spawn_widget_from_catalog(self, widget_id: str) -> None:
        widget = self._widget_index[widget_id]
        self._workspace.canvas.add_workflow_node(widget_id)
        self._state_store.update(selected_widget=widget_id, status_message=f"{widget.label} added to the workflow.")

    def _handle_file_selected(self, path: str) -> None:
        try:
            dataset = self._file_import_service.load(path)
        except PortakalDataError as exc:
            QMessageBox.warning(self, "Open Dataset", str(exc))
            self._state_store.update(status_message=f"Could not load dataset: {Path(path).name}")
            return
        self._apply_dataset(dataset, status_message=f"Selected dataset: {dataset.source.path.name}")

    def _on_state_changed(self, state: AppState) -> None:
        self._status_bar.set_message(state.status_message)
        if not self._status_log or self._status_log[-1] != state.status_message:
            self._status_log.append(state.status_message)

    def _serialize_workflow(self) -> dict[str, object]:
        return {
            "workflow": self._workspace.canvas.workflow_scene.snapshot(),
            "dataset_path": self.state.current_dataset_path,
            "workflow_info": {
                "title": self.state.workflow_title,
                "description": self.state.workflow_description,
            },
        }

    def _restore_serialized_workflow(self, payload: dict[str, object], *, reset_saved_state: bool = False) -> None:
        workflow = payload.get("workflow", {})
        dataset_path = payload.get("dataset_path")
        workflow_info = payload.get("workflow_info", {})
        self._history_guard = True
        try:
            self._workspace.close_all_dialogs()
            self._workspace.canvas.workflow_scene.restore_snapshot(workflow if isinstance(workflow, dict) else {}, emit_status=False)
            self._apply_dataset_path(str(dataset_path) if isinstance(dataset_path, str) and dataset_path else None)
            self._apply_workflow_info(workflow_info if isinstance(workflow_info, dict) else {})
        finally:
            self._history_guard = False
        if reset_saved_state:
            snapshot = self._serialize_workflow()
            self._reset_history(snapshot)
        self._refresh_modified_state()
        self._update_action_states()

    def _apply_dataset_path(self, path: str | None) -> None:
        dataset: DatasetHandle | None = None
        if path:
            try:
                dataset = self._file_import_service.load(path)
            except PortakalDataError as exc:
                QMessageBox.warning(self, "Load Workflow Dataset", str(exc))
        self._apply_dataset(dataset, path=path, status_message="Workflow loaded." if path else "Ready")

    def _apply_dataset(
        self,
        dataset: DatasetHandle | None,
        *,
        path: str | None = None,
        status_message: str,
    ) -> None:
        dataset_path = str(dataset.source.path) if dataset is not None else path
        dataset_id = dataset.dataset_id if dataset is not None else (Path(path).name if path else None)
        self._state_store.update(
            current_dataset=dataset,
            current_dataset_id=dataset_id,
            current_dataset_path=dataset_path,
            status_message=status_message,
        )
        self._workspace.set_current_dataset_path(dataset_path)
        for widget in self._workspace.all_screens():
            if isinstance(widget, FileScreen):
                widget.set_selected_file(dataset or dataset_path)
            if isinstance(widget, DataInfoScreen):
                widget.set_dataset(dataset or dataset_path)
            if isinstance(widget, DataTableScreen):
                widget.set_dataset(dataset or dataset_path)
            if isinstance(widget, SaveDataScreen):
                widget.set_dataset(dataset)
        self._workspace.refresh_dialog_footers()

    def _apply_workflow_info(self, workflow_info: dict[str, object]) -> None:
        title = str(workflow_info.get("title") or "Untitled")
        description = str(workflow_info.get("description") or "")
        self._state_store.update(workflow_title=title, workflow_description=description, status_message=self.state.status_message)

    def _reset_history(self, snapshot: dict[str, object]) -> None:
        self._undo_history = [copy.deepcopy(snapshot)]
        self._redo_history = []
        self._saved_snapshot = copy.deepcopy(snapshot)
        self._workflow_modified = False

    def _handle_workflow_changed(self) -> None:
        if self._history_guard:
            return
        snapshot = self._serialize_workflow()
        if self._undo_history and self._undo_history[-1] == snapshot:
            return
        self._undo_history.append(copy.deepcopy(snapshot))
        self._redo_history.clear()
        self._refresh_modified_state()
        self._update_action_states()

    def _refresh_modified_state(self) -> None:
        current = self._serialize_workflow()
        self._workflow_modified = self._saved_snapshot is None or current != self._saved_snapshot
        self._refresh_window_title()

    def _refresh_window_title(self) -> None:
        name = self.state.workflow_title or "Untitled"
        suffix = " *" if self._workflow_modified else ""
        self.setWindowTitle(f"{name}{suffix} - Portakal")
        self._workflow_window_action.setText(f"{name}{suffix}")

    def _selected_node(self):
        return self._workspace.canvas.workflow_scene.primary_selected_node()

    def _selected_widget_id(self) -> str | None:
        node = self._selected_node()
        return node.widget_definition.id if node is not None else None

    def _clipboard_payload(self) -> dict[str, object] | None:
        text = QApplication.clipboard().text().strip()
        if not text:
            return None
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            return None
        if payload.get("kind") != "portakal-workflow-selection":
            return None
        return payload

    def _workflow_filename_from_title(self, title: str) -> str:
        cleaned = re.sub(r'[<>:"/\\|?*\x00-\x1F]+', "-", title).strip().strip(".")
        cleaned = re.sub(r"\s+", " ", cleaned)
        return f"{cleaned or 'Untitled'}.portakal.json"

    def _preferred_workflow_path(self) -> str | None:
        if self._current_workflow_path is None:
            return None
        current_path = Path(self._current_workflow_path)
        return str(current_path.with_name(self._workflow_filename_from_title(self.state.workflow_title)))

    def _new_workflow(self) -> None:
        self._current_workflow_path = None
        self._set_frozen(False)
        self._restore_serialized_workflow({"workflow": {"nodes": [], "edges": [], "annotations": []}, "dataset_path": None}, reset_saved_state=True)
        self._state_store.update(workflow_title="Untitled", workflow_description="", status_message="New workflow created.")

    def _open_workflow_dialog(self) -> None:
        path, _selected_filter = QFileDialog.getOpenFileName(
            self,
            "Open Workflow",
            "",
            "Portakal Workflow (*.portakal.json *.json);;All Files (*.*)",
        )
        if path:
            self._open_workflow(path, freeze=False)

    def _open_and_freeze_workflow(self) -> None:
        path, _selected_filter = QFileDialog.getOpenFileName(
            self,
            "Open Workflow and Freeze",
            "",
            "Portakal Workflow (*.portakal.json *.json);;All Files (*.*)",
        )
        if path:
            self._open_workflow(path, freeze=True)

    def _open_workflow(self, path: str, *, freeze: bool) -> None:
        try:
            data = json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            self._recent_workflow_paths = [item for item in self._recent_workflow_paths if item != path]
            QMessageBox.warning(self, "Open Workflow", f"Could not open workflow.\n\n{exc}")
            self._update_action_states()
            return
        self._current_workflow_path = path
        self._last_workflow_path = path
        self._restore_serialized_workflow(data, reset_saved_state=True)
        if not self.state.workflow_title or self.state.workflow_title == "Untitled":
            self._state_store.update(workflow_title=Path(path).stem, status_message=self.state.status_message)
            self._saved_snapshot = copy.deepcopy(self._serialize_workflow())
            self._refresh_modified_state()
        self._push_recent_workflow(path)
        self._set_frozen(freeze)
        self._state_store.update(status_message=f"Opened workflow: {Path(path).name}")

    def _reload_last_workflow(self) -> None:
        if not self._last_workflow_path:
            self._state_store.update(status_message="No workflow has been opened yet.")
            return
        self._open_workflow(self._last_workflow_path, freeze=self._workflow_frozen)

    def _save_workflow(self) -> None:
        if not self._current_workflow_path:
            self._save_workflow_as()
            return
        self._write_workflow_file(self._preferred_workflow_path() or self._current_workflow_path)

    def _save_workflow_as(self) -> None:
        suggested_path = self._current_workflow_path or self._workflow_filename_from_title(self.state.workflow_title)
        path, _selected_filter = QFileDialog.getSaveFileName(
            self,
            "Save Workflow",
            suggested_path,
            "Portakal Workflow (*.portakal.json);;JSON Files (*.json)",
        )
        if path:
            if not path.endswith(".json"):
                path = f"{path}.json"
            self._write_workflow_file(path)

    def _write_workflow_file(self, path: str) -> None:
        payload = self._serialize_workflow()
        Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")
        self._current_workflow_path = path
        self._last_workflow_path = path
        self._push_recent_workflow(path)
        self._saved_snapshot = copy.deepcopy(payload)
        self._refresh_modified_state()
        self._state_store.update(status_message=f"Saved workflow: {Path(path).name}")
        self._update_action_states()

    def _export_workflow_image(self) -> None:
        path, _selected_filter = QFileDialog.getSaveFileName(
            self,
            "Export Workflow SVG",
            "workflow.svg",
            "SVG Files (*.svg)",
        )
        if not path:
            return
        if not path.lower().endswith(".svg"):
            path = f"{path}.svg"
        self._workspace.canvas.export_svg(path)
        self._state_store.update(status_message=f"Saved workflow image: {Path(path).name}")

    def _show_workflow_info(self) -> None:
        self._edit_workflow_info()

    def _edit_workflow_info(self) -> None:
        dialog = WorkflowInfoDialog(self.state.workflow_title, self.state.workflow_description, self)
        self._center_dialog(dialog)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return

        title = dialog.workflow_title()
        description = dialog.workflow_description()
        self._state_store.update(
            workflow_title=title,
            workflow_description=description,
            status_message="Workflow info updated.",
        )
        self._refresh_modified_state()

    def _center_dialog(self, dialog: QDialog) -> None:
        parent_geometry = self.frameGeometry()
        dialog_geometry = dialog.frameGeometry()
        dialog_geometry.moveCenter(parent_geometry.center())
        dialog.move(dialog_geometry.topLeft())

    def _close_window(self) -> None:
        if self._workspace.close_active_dialog():
            self._state_store.update(status_message="Closed active widget window.")
            return
        self.close()

    def _undo_workflow(self) -> None:
        if len(self._undo_history) <= 1:
            self._state_store.update(status_message="Nothing to undo.")
            return
        current = self._undo_history.pop()
        self._redo_history.append(current)
        snapshot = copy.deepcopy(self._undo_history[-1])
        self._restore_serialized_workflow(snapshot)
        self._state_store.update(status_message="Undo completed.")

    def _redo_workflow(self) -> None:
        if not self._redo_history:
            self._state_store.update(status_message="Nothing to redo.")
            return
        snapshot = copy.deepcopy(self._redo_history.pop())
        self._undo_history.append(copy.deepcopy(snapshot))
        self._restore_serialized_workflow(snapshot)
        self._state_store.update(status_message="Redo completed.")

    def _delete_selection(self) -> None:
        if self._workflow_frozen:
            self._state_store.update(status_message="Workflow is frozen.")
            return
        self._workspace.canvas.workflow_scene.delete_selected_items()

    def _copy_selection(self) -> None:
        selection = self._workspace.canvas.workflow_scene.copy_selection()
        if not selection["nodes"] and not selection["annotations"]:
            self._state_store.update(status_message="Nothing selected to copy.")
            return
        QApplication.clipboard().setText(json.dumps({"kind": "portakal-workflow-selection", "payload": selection}))
        self._state_store.update(status_message="Selection copied.")

    def _paste_selection(self) -> None:
        if self._workflow_frozen:
            self._state_store.update(status_message="Workflow is frozen.")
            return
        payload = self._clipboard_payload()
        if payload is None:
            self._state_store.update(status_message="Clipboard does not contain a workflow selection.")
            return
        self._workspace.canvas.workflow_scene.paste_snapshot(payload["payload"], offset=QPointF(28, 28))

    def _duplicate_selection(self) -> None:
        if self._workflow_frozen:
            self._state_store.update(status_message="Workflow is frozen.")
            return
        self._workspace.canvas.workflow_scene.duplicate_selected_items()

    def _select_all_workflow_items(self) -> None:
        self._workspace.canvas.workflow_scene.select_all_items()

    def _add_text_annotation(self) -> None:
        if self._workflow_frozen:
            self._state_store.update(status_message="Workflow is frozen.")
            return
        self._workspace.canvas.add_text_annotation()

    def _add_arrow_annotation(self) -> None:
        if self._workflow_frozen:
            self._state_store.update(status_message="Workflow is frozen.")
            return
        self._workspace.canvas.add_arrow_annotation()

    def _toggle_tool_dock(self, checked: bool) -> None:
        self._tool_dock_expanded = checked
        self._sidebar.setVisible(checked)
        self._catalog.setVisible(checked)
        self._state_store.update(status_message="Tool dock expanded." if checked else "Tool dock collapsed.")

    def _zoom_in(self) -> None:
        self._workspace.canvas.zoom_in()
        self._state_store.update(status_message="Zoomed in.")

    def _zoom_out(self) -> None:
        self._workspace.canvas.zoom_out()
        self._state_store.update(status_message="Zoomed out.")

    def _reset_zoom(self) -> None:
        self._workspace.canvas.reset_zoom()
        self._state_store.update(status_message="Zoom reset.")

    def _toggle_workflow_margins(self, checked: bool) -> None:
        self._workspace.set_margins_visible(checked)
        self._state_store.update(status_message="Workflow margins shown." if checked else "Workflow margins hidden.")

    def _open_selected_widget(self) -> None:
        widget_id = self._selected_widget_id()
        if widget_id is None:
            self._state_store.update(status_message="Select a widget node first.")
            return
        self._show_widget(widget_id)

    def _rename_selected_widget(self) -> None:
        node = self._selected_node()
        if node is None:
            self._state_store.update(status_message="Select a widget node first.")
            return
        new_name, accepted = QInputDialog.getText(self, "Rename Widget", "Widget name:", text=node.display_label)
        if accepted and new_name.strip():
            self._workspace.canvas.workflow_scene.rename_selected_node(new_name.strip())

    def _show_selected_widget_help(self) -> None:
        widget_id = self._selected_widget_id()
        if widget_id is None:
            self._state_store.update(status_message="Select a widget node first.")
            return
        widget = self._widget_index[widget_id]
        QMessageBox.information(self, f"{widget.label} Help", widget.description or "No help text available.")

    def _raise_widget_windows(self) -> None:
        self._workspace.raise_all_dialogs()
        self._state_store.update(status_message="Widget windows raised.")

    def _toggle_widgets_on_top(self, checked: bool) -> None:
        self._workspace.set_dialogs_on_top(checked)
        self._state_store.update(status_message="Widgets pinned on top." if checked else "Widget pinning disabled.")

    def _show_settings_dialog(self) -> None:
        summary = "\n".join(
            [
                f"Tool dock expanded: {'Yes' if self._tool_dock_expanded else 'No'}",
                f"Workflow margins: {'Yes' if self._workspace.margins_visible() else 'No'}",
                f"Widgets on top: {'Yes' if self._workspace.dialogs_on_top() else 'No'}",
                f"Frozen workflow: {'Yes' if self._workflow_frozen else 'No'}",
            ]
        )
        QMessageBox.information(self, "Settings", summary)

    def _reset_widget_settings(self) -> None:
        self._workspace.reset_widget_settings()
        self._actions["window_top"].setChecked(False)
        self._state_store.update(status_message="Widget settings reset.")
        self._update_action_states()

    def _show_addons_dialog(self) -> None:
        QMessageBox.information(self, "Add-ons", "Add-on manager will be connected in a later milestone.")

    def _open_documentation(self) -> None:
        readme_path = Path(__file__).resolve().parents[3] / "README.md"
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(readme_path)))
        self._state_store.update(status_message="Documentation opened.")

    def _show_about_dialog(self) -> None:
        QMessageBox.about(self, "About Portakal", "Portakal\n\nOrange-inspired desktop shell built with PySide6.")

    def _show_log_dialog(self) -> None:
        dialog = QDialog(self)
        dialog.setWindowTitle("Log")
        dialog.resize(640, 420)
        layout = QVBoxLayout(dialog)
        editor = QPlainTextEdit(dialog)
        editor.setReadOnly(True)
        editor.setPlainText("\n".join(self._status_log) or "No log entries yet.")
        layout.addWidget(editor)
        dialog.exec()

    def _populate_recent_workflows_menu(self) -> None:
        self._recent_menu.clear()
        if not self._recent_workflow_paths:
            empty_action = self._recent_menu.addAction("No Recent Workflows")
            empty_action.setEnabled(False)
            return
        for path in self._recent_workflow_paths:
            action = self._recent_menu.addAction(Path(path).name)
            action.triggered.connect(lambda _checked=False, workflow_path=path: self._open_workflow(workflow_path, freeze=False))

    def _populate_window_groups_menu(self) -> None:
        self._window_groups_menu.clear()
        visible_dialogs = self._workspace.visible_dialogs()
        if not visible_dialogs:
            empty_action = self._window_groups_menu.addAction("No Widget Windows")
            empty_action.setEnabled(False)
            return
        for widget_id, dialog in visible_dialogs.items():
            action = self._window_groups_menu.addAction(self._widget_index[widget_id].label)
            action.triggered.connect(lambda _checked=False, target=dialog: self._raise_dialog(target))

    def _raise_dialog(self, dialog: QDialog) -> None:
        dialog.show()
        dialog.raise_()
        dialog.activateWindow()

    def _set_frozen(self, frozen: bool) -> None:
        self._workflow_frozen = frozen
        self._workspace.canvas.set_frozen(frozen)
        self._catalog.setEnabled(not frozen)
        self._update_action_states()

    def _push_recent_workflow(self, path: str) -> None:
        normalized = str(Path(path))
        self._recent_workflow_paths = [item for item in self._recent_workflow_paths if item != normalized]
        self._recent_workflow_paths.insert(0, normalized)
        self._recent_workflow_paths = self._recent_workflow_paths[:8]

    def _update_action_states(self) -> None:
        scene: WorkflowScene = self._workspace.canvas.workflow_scene
        has_selection = bool(scene.selectedItems())
        has_node = self._selected_node() is not None
        self._actions["edit_undo"].setEnabled(len(self._undo_history) > 1)
        self._actions["edit_redo"].setEnabled(bool(self._redo_history))
        self._actions["edit_remove"].setEnabled(has_selection and not self._workflow_frozen)
        self._actions["edit_duplicate"].setEnabled(has_selection and not self._workflow_frozen)
        self._actions["edit_copy"].setEnabled(has_selection)
        self._actions["edit_paste"].setEnabled(self._clipboard_payload() is not None and not self._workflow_frozen)
        self._actions["widget_open"].setEnabled(has_node)
        self._actions["widget_rename"].setEnabled(has_node and not self._workflow_frozen)
        self._actions["widget_remove"].setEnabled(has_selection and not self._workflow_frozen)
        self._actions["widget_help"].setEnabled(has_node)
        self._actions["file_reload"].setEnabled(bool(self._last_workflow_path))
        self._actions["window_top"].setChecked(self._workspace.dialogs_on_top())
        self._actions["view_expand_dock"].setChecked(self._tool_dock_expanded)
        self._actions["view_margins"].setChecked(self._workspace.margins_visible())

    def _confirm_close_if_modified(self) -> bool:
        if not self._workflow_modified:
            return True

        response = QMessageBox.warning(
            self,
            "Unsaved Workflow",
            "Workflow has unsaved changes.\n\nDo you want to save before closing? Unsaved changes will be lost.",
            QMessageBox.StandardButton.Save
            | QMessageBox.StandardButton.Discard
            | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Save,
        )
        if response == QMessageBox.StandardButton.Cancel:
            return False
        if response == QMessageBox.StandardButton.Discard:
            return True

        before_path = self._current_workflow_path
        before_modified = self._workflow_modified
        self._save_workflow()
        if self._workflow_modified:
            return False
        if before_modified and before_path is None and self._current_workflow_path is None:
            return False
        return True

    def closeEvent(self, event) -> None:
        if not self._confirm_close_if_modified():
            event.ignore()
            return
        self._workspace.close_all_dialogs()
        super().closeEvent(event)
