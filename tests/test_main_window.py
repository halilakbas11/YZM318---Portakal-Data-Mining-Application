from __future__ import annotations

import os
import time

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PySide6")

from PySide6.QtCore import QEvent, QPointF, Qt
from PySide6.QtGui import QCloseEvent, QFocusEvent, QImage, QKeyEvent, QPainter
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QComboBox, QDialog, QMenu, QMessageBox, QPushButton, QScrollArea, QWidget

from portakal_app.app import create_application
from portakal_app.data.models import AnalysisSuggestion
from portakal_app.models import LLMSessionConfig
from portakal_app.ui import i18n
from portakal_app.ui.main_window import LLMSettingsDialog, MainWindow, WorkflowInfoDialog
from portakal_app.ui.screens.column_statistics_screen import ColumnStatisticsScreen
from portakal_app.ui.screens.csv_import_screen import CSVImportScreen
from portakal_app.ui.screens.data_info_screen import DataInfoScreen
from portakal_app.ui.screens.data_table_screen import DataTableScreen
from portakal_app.ui.screens.edit_domain_screen import EditDomainScreen
from portakal_app.ui.screens.file_screen import FileScreen
from portakal_app.ui.screens.paint_data_screen import PaintDataScreen
from portakal_app.ui.screens.rank_screen import RankScreen
from portakal_app.ui.screens.save_data_screen import SaveDataScreen
from portakal_app.ui.shell.widget_catalog import WidgetCatalogButton, WidgetCatalogPanel
from portakal_app.ui.shell.workflow_workspace import WidgetDataPreviewDialog, WidgetSelectedDataDialog


@pytest.fixture(scope="session")
def app():
    return create_application()


def _create_saved_workflow(tmp_path, name: str, node_ids: tuple[str, ...] = ("file",), *, title: str | None = None):
    window = MainWindow()
    for widget_id in node_ids:
        window._workspace.canvas.add_workflow_node(widget_id)
    if title is not None:
        window._state_store.update(workflow_title=title)
    path = tmp_path / name
    window._write_workflow_file(str(path))
    window.close()
    return path


def _open_widget_dialog(window: MainWindow, widget_id: str):
    window._show_widget(widget_id)
    node_id = window._resolve_node_id(widget_id)
    assert node_id is not None
    return node_id, window._workspace._dialogs[node_id]


def test_main_window_defaults_to_data_category(app):
    window = MainWindow()
    assert window.state.selected_category == "data"
    assert window.state.selected_widget == "file"
    assert window.windowTitle() == "Untitled - Portakal"
    layout = window.centralWidget().layout()
    assert layout.count() == 3
    assert window._workspace.current_widget() is None


def test_main_window_initial_geometry_stays_within_screen(app):
    window = MainWindow()
    available = app.primaryScreen().availableGeometry()
    geometry = window.geometry()
    assert geometry.width() <= available.width()
    assert geometry.height() <= available.height()
    assert geometry.top() >= available.top()
    assert geometry.left() >= available.left()


def test_menu_bar_contains_expected_sections(app):
    window = MainWindow()
    menu_titles = [action.text() for action in window.menuBar().actions()]
    assert menu_titles == ["File", "Edit", "View", "Widget", "Window", "Options", "Help"]

    file_menu = window._menus["file"]
    assert [action.text() for action in file_menu.actions() if not action.isSeparator()][0:5] == [
        "New",
        "Open",
        "Open and Freeze",
        "Reload Last Workflow",
        "Open Recent",
    ]


def test_remove_actions_use_backspace_shortcut(app):
    window = MainWindow()

    assert window._actions["edit_remove"].shortcut().toString() == "Backspace"
    assert window._actions["widget_remove"].shortcut().toString() == ""


def test_file_new_action_resets_workflow_and_unfreezes(app):
    window = MainWindow()
    window._workspace.canvas.add_workflow_node("file")
    window._state_store.update(workflow_title="Existing Flow", workflow_description="Saved description")
    window._set_frozen(True)

    window._actions["file_new"].trigger()

    assert window._workspace.canvas.workflow_scene.node_count() == 0
    assert window._current_workflow_path is None
    assert window.state.workflow_title == "Untitled"
    assert window.state.workflow_description == ""
    assert window.state.status_message == "New workflow created."
    assert window._workspace.canvas.is_frozen() is False
    assert window.windowTitle() == "Untitled - Portakal"


def test_file_open_action_loads_selected_workflow_from_dialog(app, tmp_path, monkeypatch):
    workflow_path = _create_saved_workflow(tmp_path, "from-open.portakal.json", ("file", "data-table"))
    window = MainWindow()

    monkeypatch.setattr(
        "portakal_app.ui.main_window.QFileDialog.getOpenFileName",
        lambda *args, **kwargs: (str(workflow_path), "Portakal Workflow (*.portakal.json *.json)"),
    )

    window._actions["file_open"].trigger()

    assert window._workspace.canvas.workflow_scene.node_count() == 2
    assert window._current_workflow_path == str(workflow_path)
    assert window._last_workflow_path == str(workflow_path)
    assert window._workspace.canvas.is_frozen() is False
    assert window.state.status_message == f"Opened workflow: {workflow_path.name}"
    assert window._recent_workflow_paths[0] == str(workflow_path)


def test_file_open_and_freeze_action_loads_selected_workflow_as_read_only(app, tmp_path, monkeypatch):
    workflow_path = _create_saved_workflow(tmp_path, "from-open-freeze.portakal.json", ("file",))
    window = MainWindow()

    monkeypatch.setattr(
        "portakal_app.ui.main_window.QFileDialog.getOpenFileName",
        lambda *args, **kwargs: (str(workflow_path), "Portakal Workflow (*.portakal.json *.json)"),
    )

    window._actions["file_open_freeze"].trigger()

    assert window._workspace.canvas.workflow_scene.node_count() == 1
    assert window._workspace.canvas.is_frozen() is True
    assert window._catalog.isEnabled() is False
    assert window.state.status_message == f"Opened workflow: {workflow_path.name}"


def test_file_reload_action_restores_last_saved_workflow(app, tmp_path):
    workflow_path = _create_saved_workflow(tmp_path, "reloadable.portakal.json", ("file",))
    window = MainWindow()

    assert window._actions["file_reload"].isEnabled() is False
    window._open_workflow(str(workflow_path), freeze=False)
    assert window._actions["file_reload"].isEnabled() is True

    window._workspace.canvas.add_workflow_node("data-table")
    assert window._workspace.canvas.workflow_scene.node_count() == 2

    window._actions["file_reload"].trigger()

    assert window._workspace.canvas.workflow_scene.node_count() == 1
    assert window._current_workflow_path == str(workflow_path)
    assert window.state.status_message == f"Opened workflow: {workflow_path.name}"


def test_open_recent_menu_shows_placeholder_and_reopens_workflow(app, tmp_path):
    window = MainWindow()
    window._populate_recent_workflows_menu()
    recent_actions = window._recent_menu.actions()
    assert len(recent_actions) == 1
    assert recent_actions[0].text() == "No Recent Workflows"
    assert recent_actions[0].isEnabled() is False

    workflow_path = _create_saved_workflow(tmp_path, "recent.portakal.json", ("file",))
    window._open_workflow(str(workflow_path), freeze=False)
    window._workspace.canvas.add_workflow_node("data-table")
    assert window._workspace.canvas.workflow_scene.node_count() == 2

    window._populate_recent_workflows_menu()
    recent_actions = window._recent_menu.actions()
    assert [action.text() for action in recent_actions] == [workflow_path.name]

    recent_actions[0].trigger()

    assert window._workspace.canvas.workflow_scene.node_count() == 1
    assert window.state.status_message == f"Opened workflow: {workflow_path.name}"


def test_file_close_window_action_closes_active_widget_dialog_first(app, monkeypatch):
    window = MainWindow()
    window.show()
    QTest.qWait(50)
    _node_id, dialog = _open_widget_dialog(window, "file")
    assert dialog.isVisible() is True

    monkeypatch.setattr("portakal_app.ui.main_window.QApplication.activeWindow", lambda: dialog)

    window._actions["file_close_window"].trigger()

    assert dialog.isVisible() is False
    assert window.state.status_message == "Closed active widget window."


def test_file_close_window_action_closes_main_window_when_no_widget_dialog_is_active(app, monkeypatch):
    window = MainWindow()
    closed = {"value": False}

    monkeypatch.setattr("portakal_app.ui.main_window.QApplication.activeWindow", lambda: window)
    monkeypatch.setattr(window, "close", lambda: closed.__setitem__("value", True))

    window._actions["file_close_window"].trigger()

    assert closed["value"] is True


def test_file_save_action_falls_back_to_save_as_for_unsaved_workflow(app, tmp_path, monkeypatch):
    window = MainWindow()
    window._workspace.canvas.add_workflow_node("file")
    window._state_store.update(workflow_title="Action Save")
    requested_path = tmp_path / "action-save"

    monkeypatch.setattr(
        "portakal_app.ui.main_window.QFileDialog.getSaveFileName",
        lambda *args, **kwargs: (str(requested_path), "Portakal Workflow (*.portakal.json)"),
    )

    window._actions["file_save"].trigger()

    saved_path = tmp_path / "action-save.json"
    assert saved_path.exists()
    assert window._current_workflow_path == str(saved_path)
    assert window.state.status_message == "Saved workflow: action-save.json"


def test_file_save_as_action_writes_to_selected_path(app, tmp_path, monkeypatch):
    window = MainWindow()
    window._workspace.canvas.add_workflow_node("file")
    requested_path = tmp_path / "save-as-target"

    monkeypatch.setattr(
        "portakal_app.ui.main_window.QFileDialog.getSaveFileName",
        lambda *args, **kwargs: (str(requested_path), "Portakal Workflow (*.portakal.json)"),
    )

    window._actions["file_save_as"].trigger()

    saved_path = tmp_path / "save-as-target.json"
    assert saved_path.exists()
    assert window._current_workflow_path == str(saved_path)
    assert window._last_workflow_path == str(saved_path)
    assert window.state.status_message == "Saved workflow: save-as-target.json"


def test_file_export_svg_action_writes_svg_file(app, tmp_path, monkeypatch):
    window = MainWindow()
    window._workspace.canvas.add_workflow_node("file")
    requested_path = tmp_path / "workflow-export"

    monkeypatch.setattr(
        "portakal_app.ui.main_window.QFileDialog.getSaveFileName",
        lambda *args, **kwargs: (str(requested_path), "SVG Files (*.svg)"),
    )

    window._actions["file_export_svg"].trigger()

    exported_path = tmp_path / "workflow-export.svg"
    assert exported_path.exists()
    assert "<svg" in exported_path.read_text(encoding="utf-8")
    assert window.state.status_message == "Saved workflow image: workflow-export.svg"


def test_file_workflow_info_action_updates_title_and_description(app, monkeypatch):
    window = MainWindow()

    monkeypatch.setattr("portakal_app.ui.main_window.WorkflowInfoDialog.exec", lambda self: QDialog.DialogCode.Accepted)
    monkeypatch.setattr("portakal_app.ui.main_window.WorkflowInfoDialog.workflow_title", lambda self: "Reviewed Flow")
    monkeypatch.setattr(
        "portakal_app.ui.main_window.WorkflowInfoDialog.workflow_description",
        lambda self: "Validated through the File menu action.",
    )

    window._actions["file_info"].trigger()

    assert window.state.workflow_title == "Reviewed Flow"
    assert window.state.workflow_description == "Validated through the File menu action."
    assert window.state.status_message == "Workflow info updated."
    assert window.windowTitle() == "Reviewed Flow * - Portakal"


def test_file_quit_action_invokes_application_quit_handler(app, monkeypatch):
    class FakeApp:
        def __init__(self, real_app) -> None:
            self._real_app = real_app
            self.quit_called = False

        def quit(self) -> None:
            self.quit_called = True

        def __getattr__(self, name: str):
            return getattr(self._real_app, name)

    fake_app = FakeApp(app)
    monkeypatch.setattr("portakal_app.ui.main_window.QApplication.instance", lambda: fake_app)

    window = MainWindow()
    window._actions["file_quit"].trigger()

    assert fake_app.quit_called is True


def test_category_switch_updates_catalog_and_content(app):
    window = MainWindow()
    window._sidebar.set_current_category("transform")
    assert window.state.selected_category == "transform"
    assert "select-columns" in window._catalog.current_widget_ids()
    assert window._workspace.current_widget() is None


def test_data_widget_navigation_opens_expected_screens(app):
    window = MainWindow()
    window.show()
    QTest.qWait(50)
    info_node_id, _info_dialog = _open_widget_dialog(window, "data-info")
    assert isinstance(window._workspace.current_widget(), DataInfoScreen)
    assert window._workspace.is_widget_dialog_visible(info_node_id)
    file_node_id, dialog = _open_widget_dialog(window, "file")
    assert isinstance(window._workspace.current_widget(), FileScreen)
    assert window._workspace.is_widget_dialog_visible(file_node_id)
    assert dialog.windowFlags() & Qt.WindowType.FramelessWindowHint
    assert dialog._close_button.text() == "Close"
    assert dialog._menu_button.isVisible()
    assert dialog._data_button.isVisible()

    screen_geometry = window.screen().availableGeometry()
    dialog_geometry = dialog.geometry()
    assert abs(dialog_geometry.center().x() - screen_geometry.center().x()) < 80
    assert abs(dialog_geometry.center().y() - screen_geometry.center().y()) < 80


def test_file_screen_exposes_orange_like_sections(app):
    screen = FileScreen()
    assert screen._file_radio.text() == "File:"
    assert screen._url_radio.text() == "URL:"
    assert screen._file_type_combo.count() == 7
    assert screen._columns_table.columnCount() == 4
    assert screen._columns_table.horizontalHeaderItem(0).text() == "Name"
    assert screen._browse_button.objectName() == "fileSourceActionButton"
    assert screen._reload_button.objectName() == "fileSourceActionButton"
    assert screen._url_combo.isEnabled()
    assert screen._columns_table.rowCount() == 0
    assert screen._columns_table.isEnabled() is False
    assert screen._apply_button.isEnabled() is False


def test_file_screen_switches_to_url_mode_on_url_focus(app):
    screen = FileScreen()
    assert screen._file_radio.isChecked()
    focus_event = QFocusEvent(QEvent.Type.FocusIn)
    QApplication.sendEvent(screen._url_combo.lineEdit(), focus_event)
    assert screen._url_radio.isChecked()


def test_file_screen_shows_kaggle_credentials_only_in_url_mode(app):
    screen = FileScreen()

    assert screen._kaggle_username_input.isHidden() is True
    assert screen._kaggle_key_input.isHidden() is True

    screen._url_radio.setChecked(True)

    assert screen._kaggle_username_input.isHidden() is False
    assert screen._kaggle_key_input.isHidden() is False


def test_file_screen_passes_manual_kaggle_credentials_to_service(app, monkeypatch):
    screen = FileScreen()
    captured = {}

    def fake_load_from_url(url, *, kaggle_username=None, kaggle_key=None):
        captured["url"] = url
        captured["kaggle_username"] = kaggle_username
        captured["kaggle_key"] = kaggle_key
        raise RuntimeError("stop after capture")

    monkeypatch.setattr(screen._import_service, "load_from_url", fake_load_from_url)

    screen._url_radio.setChecked(True)
    screen._kaggle_username_input.setText("manual-user")
    screen._kaggle_key_input.setText("manual-key")
    screen.set_remote_url("https://www.kaggle.com/datasets/owner/demo-dataset")

    assert captured["url"].startswith("https://www.kaggle.com/datasets/")
    assert captured["kaggle_username"] == "manual-user"
    assert captured["kaggle_key"] == "manual-key"


def test_file_screen_shows_actionable_message_for_kaggle_code_urls(app):
    previous_language = i18n.current_language()
    try:
        i18n.set_language("tr")
        screen = FileScreen()
        screen._url_radio.setChecked(True)
        screen.set_remote_url("https://www.kaggle.com/code/mragpavank/breast-cancer-wisconsin")

        assert screen._dataset_title.text() == "URL yüklenirken hata oluştu"
        assert "Kaggle notebook/code bağlantıları desteklenmiyor." in screen._dataset_description.text()
        assert "datasets/<owner>/<dataset-name>" in screen._dataset_description.text()
        assert screen._columns_table.rowCount() == 0
    finally:
        i18n.set_language(previous_language)


def test_file_screen_turkish_dynamic_info_and_apply_feedback(app):
    previous_language = i18n.current_language()
    try:
        i18n.set_language("tr")
        screen = FileScreen()
        i18n.apply_to_widget(screen)

        assert screen._dataset_title.text() == "Veri seti seçilmedi"
        assert "Meta verileri incelemek" in screen._dataset_description.text()
        assert screen._url_combo.lineEdit().placeholderText() == "Uzak veri seti URL'sini yapıştırın..."
        assert screen._file_type_combo.itemText(0) == "Türü dosya uzantısından belirle"
        assert screen._columns_table.rowCount() == 0
        assert screen._apply_button.isEnabled() is False

        screen._handle_apply_clicked()
        assert "Uygulamadan önce dosya yolu veya URL seçin." in screen._dataset_description.text()
    finally:
        i18n.set_language(previous_language)


def test_file_screen_shows_apply_error_instead_of_silent_fallback(app, tmp_path):
    screen = FileScreen()
    csv_path = tmp_path / "duplicate-target.csv"
    csv_path.write_text("id,diagnosis,\n1,M,\n2,B,\n", encoding="utf-8")
    screen.set_selected_file(str(csv_path))

    for row in range(screen._columns_table.rowCount()):
        name_item = screen._columns_table.item(row, 0)
        if name_item is not None and name_item.text() == "id":
            screen._columns_table.cellWidget(row, 2).setCurrentText("target")
            break

    screen._handle_apply_clicked()

    assert screen._dataset_title.text() == "Apply failed"
    assert "Exactly one target column is allowed." in screen._dataset_description.text()


def test_save_data_screen_exposes_orange_like_actions(app, tmp_path):
    screen = SaveDataScreen()
    assert screen._annotations_checkbox.text() == "Add type annotations to header"
    assert screen._autosave_checkbox.text() == "Autosave when receiving new data"
    assert screen._save_button.text() == "Save"
    assert screen._save_as_button.text() == "Save as ..."
    assert screen._save_button.isEnabled() is False

    csv_path = tmp_path / "dataset.csv"
    csv_path.write_text("a,b\n1,2\n", encoding="utf-8")
    screen.set_dataset(str(csv_path))
    assert screen._save_button.isEnabled() is True
    assert screen._save_as_button.isEnabled() is True


def test_file_screen_populates_selected_file_metadata(app, tmp_path):
    screen = FileScreen()
    csv_path = tmp_path / "iris.csv"
    csv_path.write_text("sepal_length,sepal_width,species\n5.1,3.5,setosa\n", encoding="utf-8")
    screen.set_selected_file(str(csv_path))
    assert "iris" in screen._dataset_title.text().lower()
    assert "Source path:" in screen._dataset_metrics.text()
    assert "1 rows detected" in screen._dataset_metrics.text()
    assert screen.footer_status_text() == "1"
    assert screen._columns_table.rowCount() >= 3
    assert isinstance(screen._columns_table.cellWidget(0, 1), QWidget)
    assert isinstance(screen._columns_table.cellWidget(0, 2), QComboBox)


def test_file_screen_data_preview_returns_full_csv_rows(app, tmp_path):
    screen = FileScreen()
    csv_path = tmp_path / "appointments.csv"
    csv_path.write_text(
        "id,name\n1,A\n2,B\n3,C\n4,D\n5,E\n6,F\n7,G\n8,H\n9,I\n10,J\n11,K\n12,L\n13,M\n",
        encoding="utf-8",
    )
    screen.set_selected_file(str(csv_path))
    preview = screen.data_preview_snapshot()
    assert preview["headers"] == ["id", "name"]
    assert len(preview["rows"]) == 13
    assert screen.footer_status_text() == "13"


def test_data_table_screen_loads_csv_and_updates_info(app, tmp_path):
    screen = DataTableScreen()
    csv_path = tmp_path / "iris.csv"
    csv_path.write_text(
        "iris,sepal_length,sepal_width,petal_length,petal_width\n"
        "Iris-setosa,5.1,3.5,1.4,0.2\n"
        "Iris-setosa,4.9,3.0,1.4,0.2\n",
        encoding="utf-8",
    )
    screen.set_dataset(str(csv_path))
    assert screen._table.model().rowCount() == 2
    assert screen._table.model().columnCount() == 5
    assert "2 instances" in screen._info_label.text()
    assert "Target:" in screen._summary_label.text()
    assert screen._table.editTriggers() == screen._table.EditTrigger.NoEditTriggers


def test_data_table_screen_footer_tracks_selection(app, tmp_path):
    screen = DataTableScreen()
    csv_path = tmp_path / "sample.csv"
    csv_path.write_text("label,value\nA,1\nB,2\nC,3\n", encoding="utf-8")
    screen.set_dataset(str(csv_path))
    screen._table.selectRow(0)
    assert screen.footer_status_text() == "1 | 3"


def test_data_info_screen_loads_dataset_properties(app, tmp_path):
    screen = DataInfoScreen()
    csv_path = tmp_path / "iris.csv"
    csv_path.write_text(
        "iris,sepal_length,sepal_width,petal_length,petal_width\n"
        "Iris-setosa,5.1,3.5,1.4,0.2\n"
        "Iris-setosa,4.9,3.0,1.4,0.2\n",
        encoding="utf-8",
    )
    screen.set_dataset(str(csv_path))
    assert screen._dataset_label.text() == "Dataset: iris.csv"
    assert screen._column_profiles_table.rowCount() == 5
    assert screen._llm_status.text() == "Not analyzed yet"
    assert screen._risk_list.item(0).text() == "No AI risks yet."


def test_csv_import_screen_imports_semicolon_file_without_header(app, tmp_path):
    screen = CSVImportScreen()
    csv_path = tmp_path / "raw.txt"
    csv_path.write_text("1;2;yes\n3;4;no\n", encoding="utf-8")
    captured = {}
    screen.on_import_requested(lambda dataset: captured.setdefault("dataset", dataset))

    screen._path_input.setText(str(csv_path))
    screen._delimiter_combo.setCurrentText("Semicolon (;)")
    screen._has_header_checkbox.setChecked(False)
    screen._apply_button.click()

    dataset = captured["dataset"]
    assert dataset.row_count == 2
    assert dataset.dataframe.columns == ["Column 1", "Column 2", "Column 3"]
    assert screen._preview_table.rowCount() == 2
    assert "Delimiter: Semicolon (;)" in screen._settings_label.text()


def test_csv_import_screen_auto_detects_delimiter_and_skip_rows(app, tmp_path):
    screen = CSVImportScreen()
    csv_path = tmp_path / "auto.txt"
    csv_path.write_text("# comment\ncity;value\nAnkara;1\nIzmir;2\n", encoding="utf-8")

    screen._path_input.setText(str(csv_path))
    screen._delimiter_combo.setCurrentText("Auto")
    screen._skip_rows_spin.setValue(1)
    screen._reload_button.click()

    assert "Delimiter: Semicolon (;)" in screen._settings_label.text()
    assert screen._preview_table.horizontalHeaderItem(0).text() == "city"
    assert screen._preview_table.rowCount() == 2


def test_csv_import_screen_reads_cp1254_with_auto_encoding(app, tmp_path):
    screen = CSVImportScreen()
    csv_path = tmp_path / "tr.csv"
    csv_path.write_bytes("şehir;değer\nİzmir;10\n".encode("cp1254"))

    screen._path_input.setText(str(csv_path))
    screen._delimiter_combo.setCurrentText("Auto")
    screen._encoding_combo.setCurrentText("Auto")
    screen._reload_button.click()

    assert screen._preview_table.horizontalHeaderItem(0).text() == "şehir"
    assert screen._preview_table.item(0, 0).text() == "İzmir"


def test_edit_domain_screen_applies_renames_and_roles(app, tmp_path):
    screen = EditDomainScreen()
    csv_path = tmp_path / "domain.csv"
    csv_path.write_text("value,label\n1,A\n2,B\n", encoding="utf-8")
    captured = {}
    screen.on_apply_requested(lambda dataset: captured.setdefault("dataset", dataset))
    screen.set_dataset(str(csv_path))

    screen._columns_table.item(0, 0).setText("amount")
    role_widget = screen._columns_table.cellWidget(1, 2)
    role_widget.setCurrentText("meta")
    screen._apply_button.click()

    dataset = captured["dataset"]
    assert dataset.dataframe.columns == ["amount", "label"]
    roles = {column.name: column.role for column in dataset.domain.columns}
    assert roles["amount"] == "feature"
    assert roles["label"] == "meta"


def test_edit_domain_screen_enforces_single_target_and_restore_inferred(app, tmp_path):
    screen = EditDomainScreen()
    csv_path = tmp_path / "domain-restore.csv"
    csv_path.write_text("value,label,target2\n1,A,K1\n2,B,K2\n", encoding="utf-8")
    screen.set_dataset(str(csv_path))

    first_role = screen._columns_table.cellWidget(0, 2)
    third_role = screen._columns_table.cellWidget(2, 2)
    first_role.setCurrentText("target")
    third_role.setCurrentText("target")

    assert first_role.currentText() == "feature"
    assert third_role.currentText() == "target"

    screen._columns_table.item(0, 0).setText("amount")
    screen._restore_inferred_button.click()

    assert screen._columns_table.item(0, 0).text() == "value"
    assert "Restored inferred domain" in screen._change_summary_label.text()


def test_edit_domain_screen_skip_drops_column(app, tmp_path):
    screen = EditDomainScreen()
    csv_path = tmp_path / "domain-skip.csv"
    csv_path.write_text("value,label,city\n1,A,Ankara\n2,B,Izmir\n", encoding="utf-8")
    captured = {}
    screen.on_apply_requested(lambda dataset: captured.setdefault("dataset", dataset))
    screen.set_dataset(str(csv_path))

    role_widget = screen._columns_table.cellWidget(2, 2)
    role_widget.setCurrentText("skip")
    screen._apply_button.click()

    assert captured["dataset"].dataframe.columns == ["value", "label"]
    assert "Dropped 1 skipped column" in screen._change_summary_label.text()


def test_column_statistics_screen_shows_numeric_metrics(app, tmp_path):
    screen = ColumnStatisticsScreen()
    csv_path = tmp_path / "stats.csv"
    csv_path.write_text("value,label\n1,A\n3,A\n5,B\n", encoding="utf-8")
    screen.set_dataset(str(csv_path))
    screen._column_combo.setCurrentText("value")

    metrics = {screen._stats_table.item(row, 0).text(): screen._stats_table.item(row, 1).text() for row in range(screen._stats_table.rowCount())}
    assert metrics["Type"] == "numeric"
    assert metrics["Mean"] == "3.0000"
    assert screen._top_values_table.rowCount() >= 3
    assert screen._histogram_widget is not None


def test_column_statistics_screen_filters_columns_and_shows_warning_badges(app, tmp_path):
    screen = ColumnStatisticsScreen()
    csv_path = tmp_path / "stats-filter.csv"
    csv_path.write_text("city,id_text\n" + "\n".join([f"A,id-{index}" for index in range(19)]) + "\nB,id-19\n", encoding="utf-8")
    screen.set_dataset(str(csv_path))
    screen._search_input.setText("id")

    assert screen._column_combo.count() == 1
    assert screen._column_combo.currentText() == "id_text"
    badge_texts = [screen._warning_badges_layout.itemAt(index).widget().text() for index in range(screen._warning_badges_layout.count() - 1)]
    assert "high cardinality" in badge_texts


def test_rank_screen_orders_signal_feature_first(app, tmp_path):
    screen = RankScreen()
    csv_path = tmp_path / "rank.csv"
    csv_path.write_text("signal,noise,target\n0.1,5,A\n0.2,1,A\n0.9,3,B\n0.8,4,B\n", encoding="utf-8")
    screen.set_dataset(str(csv_path))

    assert screen._rank_table.rowCount() >= 2
    assert screen._rank_table.item(0, 0).text() == "signal"
    first_score = float(screen._rank_table.item(0, 2).text())
    second_score = float(screen._rank_table.item(1, 2).text())
    assert first_score >= second_score


def test_rank_screen_supports_target_override_filter_and_top_n(app, tmp_path):
    screen = RankScreen()
    csv_path = tmp_path / "rank-controls.csv"
    csv_path.write_text("signal,noise,city,target\n0.1,5,A,A\n0.2,1,A,A\n0.9,3,B,B\n0.8,4,B,B\n", encoding="utf-8")
    screen.set_dataset(str(csv_path))
    screen._feature_filter_combo.setCurrentText("Numeric only")
    screen._top_n_spin.setValue(1)
    screen._target_combo.setCurrentText("target")

    assert screen._rank_table.rowCount() == 1
    assert screen._rank_table.item(0, 0).text() == "signal"
    assert "target 'target'" in screen._summary_label.text()

    screen._target_combo.setCurrentText("None (Heuristic)")
    assert "heuristic mode" in screen._summary_label.text().lower()


def test_edit_domain_apply_updates_main_window_dataset(app, tmp_path):
    window = MainWindow()
    source = window._workspace.canvas.add_workflow_node("file")
    target = window._workspace.canvas.add_workflow_node("edit-domain")
    assert window._workspace.canvas.workflow_scene.create_connection(source.node_id, target.node_id)
    csv_path = tmp_path / "window-domain.csv"
    csv_path.write_text("value,label\n1,A\n2,B\n", encoding="utf-8")
    window._handle_file_selected(str(csv_path))

    screen = next(widget for widget in window._workspace.all_screens() if isinstance(widget, EditDomainScreen))
    screen._columns_table.item(0, 0).setText("amount")
    screen._apply_button.click()

    assert window.state.current_dataset is not None
    assert window.state.current_dataset.dataframe.columns[0] == "amount"


def test_csv_import_apply_updates_main_window_dataset_and_data_info(app, tmp_path):
    window = MainWindow()
    source = window._workspace.canvas.add_workflow_node("csv-import")
    target = window._workspace.canvas.add_workflow_node("data-info")
    assert window._workspace.canvas.workflow_scene.create_connection(source.node_id, target.node_id)
    screen = next(widget for widget in window._workspace.all_screens() if isinstance(widget, CSVImportScreen))
    csv_path = tmp_path / "window-import.txt"
    csv_path.write_text("# comment\ncity;value\nAnkara;1\nIzmir;2\n", encoding="utf-8")

    screen._path_input.setText(str(csv_path))
    screen._delimiter_combo.setCurrentText("Auto")
    screen._skip_rows_spin.setValue(1)
    screen._apply_button.click()

    assert window.state.current_dataset is not None
    assert window.state.current_dataset.row_count == 2
    data_info = next(widget for widget in window._workspace.all_screens() if isinstance(widget, DataInfoScreen))
    assert data_info._dataset_label.text() == "Dataset: window-import.txt"


def test_edit_domain_apply_updates_global_preview_snapshot(app, tmp_path):
    window = MainWindow()
    source = window._workspace.canvas.add_workflow_node("file")
    target = window._workspace.canvas.add_workflow_node("edit-domain")
    assert window._workspace.canvas.workflow_scene.create_connection(source.node_id, target.node_id)
    csv_path = tmp_path / "preview-domain.csv"
    csv_path.write_text("value,label,city\n1,A,Ankara\n2,B,Izmir\n", encoding="utf-8")
    window._handle_file_selected(str(csv_path))

    screen = next(widget for widget in window._workspace.all_screens() if isinstance(widget, EditDomainScreen))
    role_widget = screen._columns_table.cellWidget(2, 2)
    role_widget.setCurrentText("skip")
    screen._apply_button.click()

    preview = window._node_data_preview_snapshot(target.node_id)
    assert preview["headers"] == ["value", "label"]


def test_file_apply_propagates_column_renames_to_paint_data(app, tmp_path):
    window = MainWindow()
    source = window._workspace.canvas.add_workflow_node("file")
    target = window._workspace.canvas.add_workflow_node("paint-data")
    assert window._workspace.canvas.workflow_scene.create_connection(source.node_id, target.node_id)

    csv_path = tmp_path / "file-paint.csv"
    csv_path.write_text("x,y,label\n1,2,A\n3,4,B\n", encoding="utf-8")
    window._handle_file_selected(str(csv_path))

    file_screen = window._workspace.screen_for_node(source.node_id)
    paint_screen = window._workspace.screen_for_node(target.node_id)
    assert isinstance(file_screen, FileScreen)
    assert isinstance(paint_screen, PaintDataScreen)

    file_screen._columns_table.item(0, 0).setText("feature_x")
    file_screen._columns_table.item(1, 0).setText("feature_y")
    file_screen._apply_button.click()
    assert file_screen.current_output_dataset() is not None
    assert file_screen.current_output_dataset().dataframe.columns[:2] == ["feature_x", "feature_y"]
    window._recompute_node_runtime()

    snapshot = paint_screen._current_snapshot()
    assert snapshot.x_name == "feature_x"
    assert snapshot.y_name == "feature_y"
    assert paint_screen._canvas._x_axis_label == "feature_x"
    assert paint_screen._canvas._y_axis_label == "feature_y"


def test_file_apply_role_changes_update_paint_data_snapshot_and_labels(app, tmp_path):
    window = MainWindow()
    source = window._workspace.canvas.add_workflow_node("file")
    target = window._workspace.canvas.add_workflow_node("paint-data")
    assert window._workspace.canvas.workflow_scene.create_connection(source.node_id, target.node_id)

    csv_path = tmp_path / "breast-cancer-like.csv"
    csv_path.write_text(
        "id,diagnosis,radius_mean,texture_mean,perimeter_mean,\n"
        "1,M,17.99,10.38,122.8,\n"
        "2,B,20.57,17.77,132.9,\n",
        encoding="utf-8",
    )
    window._handle_file_selected(str(csv_path))

    file_screen = window._workspace.screen_for_node(source.node_id)
    paint_screen = window._workspace.screen_for_node(target.node_id)
    assert isinstance(file_screen, FileScreen)
    assert isinstance(paint_screen, PaintDataScreen)

    role_by_name = {}
    for row in range(file_screen._columns_table.rowCount()):
        name_item = file_screen._columns_table.item(row, 0)
        role_widget = file_screen._columns_table.cellWidget(row, 2)
        if name_item is not None and isinstance(role_widget, QComboBox):
            role_by_name[name_item.text()] = role_widget

    role_by_name["id"].setCurrentText("skip")
    role_by_name["diagnosis"].setCurrentText("target")
    role_by_name["texture_mean"].setCurrentText("skip")
    role_by_name["perimeter_mean"].setCurrentText("feature")
    file_screen._handle_apply_clicked()

    snapshot = paint_screen._current_snapshot()
    assert snapshot.x_name == "radius_mean"
    assert snapshot.y_name == "perimeter_mean"
    assert snapshot.labels == ("M", "B")

def test_data_table_selection_drives_downstream_save_data_input(app, tmp_path):
    window = MainWindow()
    source = window._workspace.canvas.add_workflow_node("file")
    table_node = window._workspace.canvas.add_workflow_node("data-table")
    save_node = window._workspace.canvas.add_workflow_node("save-data")
    scene = window._workspace.canvas.workflow_scene
    assert scene.create_connection(source.node_id, table_node.node_id)
    assert scene.create_connection(table_node.node_id, save_node.node_id)

    csv_path = tmp_path / "selection.csv"
    csv_path.write_text("label,value\nA,1\nB,2\nC,3\n", encoding="utf-8")
    window._handle_file_selected(str(csv_path))

    save_screen = window._workspace.screen_for_node(save_node.node_id)
    assert isinstance(save_screen, SaveDataScreen)
    assert save_screen._save_button.isEnabled() is False

    table_screen = window._workspace.screen_for_node(table_node.node_id)
    assert isinstance(table_screen, DataTableScreen)
    table_screen._table.selectRow(1)
    QTest.qWait(20)

    assert save_screen._save_button.isEnabled() is True
    preview = window._node_data_preview_snapshot(save_node.node_id)
    assert preview["headers"] == ["label", "value"]
    assert preview["rows"] == [["B", "2"]]


def test_rank_scores_connect_only_to_compatible_consumers(app, tmp_path):
    window = MainWindow()
    source = window._workspace.canvas.add_workflow_node("file")
    rank_node = window._workspace.canvas.add_workflow_node("rank")
    table_node = window._workspace.canvas.add_workflow_node("data-table")
    save_node = window._workspace.canvas.add_workflow_node("save-data")
    info_node = window._workspace.canvas.add_workflow_node("data-info")
    scene = window._workspace.canvas.workflow_scene

    assert scene.create_connection(source.node_id, rank_node.node_id)
    assert scene.create_connection(rank_node.node_id, table_node.node_id)
    assert scene.create_connection(rank_node.node_id, save_node.node_id)
    assert not scene.create_connection(rank_node.node_id, info_node.node_id)

    csv_path = tmp_path / "rank-flow.csv"
    csv_path.write_text("signal,noise,target\n1,9,A\n2,8,A\n9,1,B\n8,2,B\n", encoding="utf-8")
    window._handle_file_selected(str(csv_path))

    table_preview = window._node_data_preview_snapshot(table_node.node_id)
    assert table_preview["headers"] == ["feature", "type", "score", "method", "details"]

    save_screen = window._workspace.screen_for_node(save_node.node_id)
    assert isinstance(save_screen, SaveDataScreen)
    assert save_screen._save_button.isEnabled() is True


def test_workflow_restore_preserves_data_table_selection_output(app, tmp_path):
    window = MainWindow()
    source = window._workspace.canvas.add_workflow_node("file")
    table_node = window._workspace.canvas.add_workflow_node("data-table")
    save_node = window._workspace.canvas.add_workflow_node("save-data")
    scene = window._workspace.canvas.workflow_scene
    assert scene.create_connection(source.node_id, table_node.node_id)
    assert scene.create_connection(table_node.node_id, save_node.node_id)

    csv_path = tmp_path / "persist-selection.csv"
    csv_path.write_text("label,value\nA,1\nB,2\nC,3\n", encoding="utf-8")
    window._handle_file_selected(str(csv_path))

    table_screen = window._workspace.screen_for_node(table_node.node_id)
    assert isinstance(table_screen, DataTableScreen)
    table_screen._table.selectRow(2)
    QTest.qWait(20)

    workflow_path = tmp_path / "selection-flow.portakal.json"
    window._write_workflow_file(str(workflow_path))

    restored = MainWindow()
    restored._open_workflow(str(workflow_path), freeze=False)

    restored_scene = restored._workspace.canvas.workflow_scene
    restored_table = next(record for record in restored_scene.node_records() if record.widget_id == "data-table")
    restored_save = next(record for record in restored_scene.node_records() if record.widget_id == "save-data")
    preview = restored._node_data_preview_snapshot(restored_save.node_id)
    assert preview["rows"] == [["C", "3"]]

    restored_table_screen = restored._workspace.screen_for_node(restored_table.node_id)
    assert isinstance(restored_table_screen, DataTableScreen)
    assert restored_table_screen.footer_status_text() == "1 | 3"


def test_two_data_table_nodes_keep_independent_selection_state(app, tmp_path):
    window = MainWindow()
    source = window._workspace.canvas.add_workflow_node("file")
    table_one = window._workspace.canvas.add_workflow_node("data-table")
    table_two = window._workspace.canvas.add_workflow_node("data-table")
    save_one = window._workspace.canvas.add_workflow_node("save-data")
    save_two = window._workspace.canvas.add_workflow_node("save-data")
    scene = window._workspace.canvas.workflow_scene
    assert scene.create_connection(source.node_id, table_one.node_id)
    assert scene.create_connection(source.node_id, table_two.node_id)
    assert scene.create_connection(table_one.node_id, save_one.node_id)
    assert scene.create_connection(table_two.node_id, save_two.node_id)

    csv_path = tmp_path / "multi-table.csv"
    csv_path.write_text("label,value\nA,1\nB,2\nC,3\n", encoding="utf-8")
    window._handle_file_selected(str(csv_path))

    first_screen = window._workspace.screen_for_node(table_one.node_id)
    second_screen = window._workspace.screen_for_node(table_two.node_id)
    assert isinstance(first_screen, DataTableScreen)
    assert isinstance(second_screen, DataTableScreen)

    first_screen._table.selectRow(0)
    second_screen._table.selectRow(2)
    QTest.qWait(20)

    first_preview = window._node_data_preview_snapshot(save_one.node_id)
    second_preview = window._node_data_preview_snapshot(save_two.node_id)
    assert first_preview["rows"] == [["A", "1"]]
    assert second_preview["rows"] == [["C", "3"]]


def test_data_table_screen_shows_full_dataset_and_keeps_total_count(app, tmp_path):
    screen = DataTableScreen()
    csv_path = tmp_path / "big.csv"
    rows = ["label,value"] + [f"A,{index}" for index in range(600)]
    csv_path.write_text("\n".join(rows), encoding="utf-8")
    screen.set_dataset(str(csv_path))
    assert screen._table.model().rowCount() == 600
    assert screen.footer_status_text() == "600"
    assert "Showing all rows in the table" in screen._summary_label.text()


def test_data_table_screen_color_toggle_updates_without_rebuilding(app, tmp_path):
    screen = DataTableScreen()
    csv_path = tmp_path / "sample.csv"
    csv_path.write_text("label,value\nA,1\nB,2\nC,3\n", encoding="utf-8")
    screen.set_dataset(str(csv_path))
    original_model = screen._table.model()
    screen._color_checkbox.setChecked(False)
    assert screen._table.model() is original_model


def test_data_table_selected_preview_uses_selected_columns_for_partial_selection(app, tmp_path):
    screen = DataTableScreen()
    csv_path = tmp_path / "sample.csv"
    csv_path.write_text("label,value,city\nA,1,Ankara\nB,2,Istanbul\nC,3,Izmir\n", encoding="utf-8")
    screen.set_dataset(str(csv_path))
    screen._select_full_rows_checkbox.setChecked(False)
    selection_model = screen._table.selectionModel()
    top_left = screen._table.model().index(0, 0)
    bottom_right = screen._table.model().index(1, 1)
    from PySide6.QtCore import QItemSelection, QItemSelectionModel

    selection = QItemSelection(top_left, bottom_right)
    selection_model.select(selection, QItemSelectionModel.SelectionFlag.ClearAndSelect)
    snapshot = screen.detailed_data_snapshot()
    assert snapshot["selected_headers"] == ["label", "value"]
    assert snapshot["selected_rows"] == [["A", "1"], ["B", "2"]]
    assert "2 instances, 2 variables" in snapshot["selected_summary"]


def test_file_dialog_footer_refreshes_after_file_selection(app, tmp_path):
    window = MainWindow()
    _node_id, dialog = _open_widget_dialog(window, "file")
    assert dialog._status_label.text() == "0"

    csv_path = tmp_path / "rows.csv"
    csv_path.write_text("id\n1\n2\n3\n4\n", encoding="utf-8")
    window._handle_file_selected(str(csv_path))

    assert dialog._status_label.text() == "4"


def test_main_window_stores_dataset_handle_after_file_selection(app, tmp_path):
    window = MainWindow()
    csv_path = tmp_path / "rows.csv"
    csv_path.write_text("id,label\n1,A\n2,B\n", encoding="utf-8")

    window._handle_file_selected(str(csv_path))

    assert window.state.current_dataset is not None
    assert window.state.current_dataset.row_count == 2
    assert window.state.current_dataset.column_count == 2
    assert window.state.current_dataset_path == str(csv_path)
    assert window.state.current_dataset.source.path == csv_path


def test_widget_popup_data_action_opens_dialog(app, monkeypatch):
    window = MainWindow()
    file_node = window._workspace.canvas.add_workflow_node("file")
    csv_path = window.state.current_dataset_path
    if not csv_path:
        import tempfile
        from pathlib import Path

        temp_dir = tempfile.mkdtemp()
        path = Path(temp_dir) / "preview.csv"
        path.write_text("a,b\n1,2\n", encoding="utf-8")
        window._handle_file_selected(str(path))
    window._show_widget(file_node.node_id)
    dialog = window._workspace._dialogs[file_node.node_id]

    called = {"data": False}

    def fake_data_exec(self):
        called["data"] = isinstance(self, WidgetDataPreviewDialog)
        return 0

    monkeypatch.setattr(WidgetDataPreviewDialog, "exec", fake_data_exec)

    dialog._show_data_preview()

    assert called["data"] is True
    assert not dialog._help_button.icon().isNull()


def test_widget_popup_data_action_uses_workflow_dataset_for_other_modules(app, tmp_path, monkeypatch):
    window = MainWindow()
    csv_path = tmp_path / "shared.csv"
    csv_path.write_text("a,b\n1,2\n3,4\n", encoding="utf-8")
    file_node = window._workspace.canvas.add_workflow_node("file")
    window._handle_file_selected(str(csv_path))
    save_node = window._workspace.canvas.add_workflow_node("save-data")
    window._workspace.canvas.workflow_scene.create_connection(file_node.node_id, save_node.node_id)
    window._show_widget(save_node.node_id)
    dialog = window._workspace._dialogs[save_node.node_id]

    captured = {}

    def fake_exec(self):
        captured["headers"] = self._preview_data["headers"]
        captured["rows"] = self._preview_data["rows"]
        return 0

    monkeypatch.setattr(WidgetDataPreviewDialog, "exec", fake_exec)
    dialog._show_data_preview()

    assert captured["headers"] == ["a", "b"]
    assert captured["rows"][:2] == [["1", "2"], ["3", "4"]]


def test_widget_popup_data_action_disabled_when_widget_not_connected(app, tmp_path):
    window = MainWindow()
    csv_path = tmp_path / "shared.csv"
    csv_path.write_text("a,b\n1,2\n3,4\n", encoding="utf-8")
    window._handle_file_selected(str(csv_path))
    window._workspace.canvas.add_workflow_node("file")
    info_node = window._workspace.canvas.add_workflow_node("data-info")
    window._show_widget(info_node.node_id)
    dialog = window._workspace._dialogs[info_node.node_id]
    dialog.refresh_footer()
    assert dialog._data_button.isEnabled() is False


def test_widget_popup_selected_data_action_opens_dialog(app, monkeypatch):
    window = MainWindow()
    _node_id, dialog = _open_widget_dialog(window, "data-table")

    called = {"selected": False}

    def fake_exec(self):
        called["selected"] = isinstance(self, WidgetSelectedDataDialog)
        return 0

    monkeypatch.setattr(WidgetSelectedDataDialog, "exec", fake_exec)

    dialog._show_selected_data_preview()

    assert called["selected"] is True


def test_file_widget_footer_menu_contains_real_actions(app):
    window = MainWindow()
    _node_id, dialog = _open_widget_dialog(window, "file")
    menu = QMenu()

    dialog._build_file_submenu(menu)

    texts = [action.text() for action in menu.actions() if not action.isSeparator()]
    assert "Open Dataset..." in texts
    assert "Reload Source" in texts
    assert "Apply" in texts
    assert "Reset" in texts
    assert "Close" in texts


def test_data_info_help_menu_contains_documentation(app):
    window = MainWindow()
    _node_id, dialog = _open_widget_dialog(window, "data-info")
    menu = QMenu()

    dialog._build_help_submenu(menu)

    texts = [action.text() for action in menu.actions()]
    assert texts == ["Widget Help", "Documentation"]


def test_llm_settings_dialog_updates_provider_fields(app, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-env")
    dialog = LLMSettingsDialog(LLMSessionConfig(provider="OpenAI", model="gpt-test"), None)

    assert "Environment key detected" in dialog.env_status_label.text()
    assert dialog.base_url_input.text() == "https://api.openai.com/v1"
    dialog.provider_combo.setCurrentText("Ollama")

    assert dialog.api_key_input.isEnabled() is False
    assert dialog.base_url_input.text() == "http://localhost:11434"
    assert "does not require an API key" in dialog.env_status_label.text()


def test_language_switch_updates_open_widget_texts_immediately(app, monkeypatch):
    previous_language = i18n.current_language()
    window = MainWindow()
    file_node = window._workspace.canvas.add_workflow_node("file")
    window._show_widget(file_node.node_id)
    file_screen = window._workspace.current_widget()
    assert isinstance(file_screen, FileScreen)

    monkeypatch.setattr(QMessageBox, "information", lambda *args, **kwargs: 0)

    try:
        i18n.set_language("en")
        i18n.set_language("tr")
        assert window.menuBar().actions()[0].text() == "Dosya"
        assert window._menus["options"].actions()[0].text() == "Dil"
        assert file_screen._file_radio.text() == "Dosya:"
        assert file_screen._reload_button.text() == "Yeniden Yükle"
        i18n.set_language("en")
        assert window.menuBar().actions()[0].text() == "File"
        assert file_screen._file_radio.text() == "File:"
    finally:
        i18n.set_language(previous_language)


def test_language_switch_updates_shell_catalog_and_node_labels(app):
    previous_language = i18n.current_language()
    window = MainWindow()
    record = window._workspace.canvas.add_workflow_node("column-statistics")
    scene = window._workspace.canvas.workflow_scene
    node = scene._nodes[record.node_id]

    try:
        i18n.set_language("en")
        i18n.set_language("tr")

        assert window._workspace._title_label.text() == "İş Akışı"
        assert window._workspace._subtitle_label.text().startswith("Bileşenleri katalogdan tuvale sürükleyin.")
        assert [window._sidebar._list.item(index).text() for index in range(window._sidebar._list.count())] == [
            "Veri",
            "Dönüştür",
            "Görselleştir",
            "Model",
            "Değerlendir",
            "Gözetimsiz",
        ]
        assert node.display_label == "Sütun İstatistikleri"

        window._on_category_selected("transform")
        button_texts = [button.text() for button in window._catalog.findChildren(WidgetCatalogButton)]
        assert window._catalog._title.text() == "Dönüştür"
        assert any(text.startswith("Sütun Seç") for text in button_texts)
        assert any(text.startswith("Preprocess") for text in button_texts)

        i18n.set_language("en")
        assert window._workspace._title_label.text() == "Workflow"
        assert window._sidebar._list.item(1).text() == "Transform"
        assert node.display_label == "Column Statistics"
    finally:
        i18n.set_language(previous_language)


def test_settings_dialog_updates_global_llm_config_without_serializing_it(app, monkeypatch):
    window = MainWindow()

    def fake_exec(self):
        self.provider_combo.setCurrentText("Gemini")
        self.model_input.setText("gemini-2.5-flash")
        self.base_url_input.setText("https://generativelanguage.googleapis.com/v1beta")
        self.api_key_input.setText("gem-key")
        return QDialog.DialogCode.Accepted

    monkeypatch.setattr(LLMSettingsDialog, "exec", fake_exec)

    window._show_settings_dialog()
    payload = window._serialize_workflow()
    payload_text = str(payload)

    assert window._llm_session_config.provider == "Gemini"
    assert window._llm_session_config.model == "gemini-2.5-flash"
    assert "gem-key" not in payload_text
    assert "Gemini" not in payload_text


def test_data_info_screen_runs_async_ai_analysis_and_renders_results(app, tmp_path):
    class SlowAnalyzer:
        def analyze(self, summary, context, config):
            time.sleep(0.1)
            return [
                AnalysisSuggestion("Leakage", "Potential target leakage detected.", kind="risk", severity="high"),
                AnalysisSuggestion("Impute", "Review null-heavy fields before training.", kind="suggestion", severity="medium"),
            ]

    screen = DataInfoScreen()
    screen.set_llm_analyzer(SlowAnalyzer())  # type: ignore[arg-type]
    screen.set_llm_session_config(LLMSessionConfig(provider="Ollama", model="llama3.1", base_url="http://localhost:11434"))
    csv_path = tmp_path / "iris.csv"
    csv_path.write_text("a,b,target\n1,2,yes\n3,4,no\n", encoding="utf-8")
    screen.set_dataset(str(csv_path))

    screen._analyze_button.click()

    assert screen._analyze_button.isEnabled() is False
    for _ in range(40):
        if screen._analysis_thread is None:
            break
        QTest.qWait(50)

    assert screen._analysis_thread is None
    assert "Analyzed with Ollama" in screen._llm_status.text()
    assert "Leakage" in screen._risk_list.item(0).text()
    assert "Impute" in screen._suggestion_list.item(0).text()


def test_data_info_screen_shows_ai_error_without_losing_summary(app, tmp_path):
    class FailingAnalyzer:
        def analyze(self, summary, context, config):
            raise RuntimeError("Bad key")

    screen = DataInfoScreen()
    screen.set_llm_analyzer(FailingAnalyzer())  # type: ignore[arg-type]
    screen.set_llm_session_config(LLMSessionConfig(provider="Ollama", model="llama3.1", base_url="http://localhost:11434"))
    csv_path = tmp_path / "iris.csv"
    csv_path.write_text("a,b,target\n1,2,yes\n3,4,no\n", encoding="utf-8")
    screen.set_dataset(str(csv_path))
    summary_card_count = screen._summary_cards_layout.count()

    screen._analyze_button.click()
    for _ in range(40):
        if screen._analysis_thread is None:
            break
        QTest.qWait(50)

    assert screen._analysis_thread is None
    assert screen._llm_status.text() == "AI analysis failed"
    assert screen._llm_error_label.text() == "Bad key"
    assert screen._summary_cards_layout.count() == summary_card_count


def test_widget_popup_close_button_is_clickable(app):
    window = MainWindow()
    window.show()
    QTest.qWait(50)
    _node_id, dialog = _open_widget_dialog(window, "file")
    assert dialog.isVisible()

    QTest.mouseClick(dialog._close_button, Qt.MouseButton.LeftButton)

    assert not dialog.isVisible()


def test_file_dialog_wraps_screen_in_scroll_area(app):
    window = MainWindow()
    _node_id, dialog = _open_widget_dialog(window, "file")
    assert isinstance(dialog._scroll_area, QScrollArea)
    assert dialog._scroll_area.widget() is dialog._screen


def test_save_data_dialog_opens_compact(app):
    window = MainWindow()
    _node_id, dialog = _open_widget_dialog(window, "save-data")
    assert dialog.width() <= 620
    assert dialog.height() <= 420
    assert dialog._scroll_area is None


def test_data_preview_dialog_uses_popup_surface_style(app):
    dialog = WidgetDataPreviewDialog("Preview", {"summary": "Rows", "headers": ["a"], "rows": [["1"]]})
    assert dialog.objectName() == "widgetPopup"
    assert dialog._close_button.text() == "Close"


def test_catalog_click_spawns_node_without_opening_detail(app):
    window = MainWindow()
    window.show()
    QTest.qWait(50)
    window._sidebar.set_current_category("data")
    button = next(card for card in window._catalog.findChildren(WidgetCatalogButton) if card.widget_id == "file")
    QTest.mouseClick(button, Qt.MouseButton.LeftButton)
    QTest.qWait(220)
    assert window._workspace.canvas.workflow_scene.node_count() == 1
    assert window._workspace.current_widget() is None


def test_catalog_double_click_spawns_node_once_without_opening_detail(app):
    window = MainWindow()
    window.show()
    QTest.qWait(50)
    window._sidebar.set_current_category("data")
    button = next(card for card in window._catalog.findChildren(WidgetCatalogButton) if card.widget_id == "file")
    QTest.mouseDClick(button, Qt.MouseButton.LeftButton)
    QTest.qWait(220)
    assert window._workspace.canvas.workflow_scene.node_count() == 1
    assert window._workspace.current_widget() is None


def test_widget_cards_have_icons(app):
    window = MainWindow()
    window._sidebar.set_current_category("data")
    cards = window._catalog.findChildren(WidgetCatalogButton)
    assert cards
    assert all(not card.icon().isNull() for card in cards)


def test_catalog_panel_scroll_area_is_borderless_and_transparent(app):
    window = MainWindow()
    catalog = window.findChild(WidgetCatalogPanel)
    assert catalog is not None
    scroll_areas = catalog.findChildren(QScrollArea)
    assert len(scroll_areas) == 1
    scroll = scroll_areas[0]
    assert scroll.widgetResizable()
    assert scroll.horizontalScrollBarPolicy() == Qt.ScrollBarPolicy.ScrollBarAlwaysOff


def test_sidebar_scrollbars_are_hidden(app):
    window = MainWindow()
    assert window._sidebar._list.horizontalScrollBarPolicy() == Qt.ScrollBarPolicy.ScrollBarAlwaysOff
    assert window._sidebar._list.verticalScrollBarPolicy() == Qt.ScrollBarPolicy.ScrollBarAlwaysOff


def test_sidebar_width_expands_for_longest_category_label(app):
    window = MainWindow()
    assert window._sidebar.width() == window._sidebar.recommended_width()
    assert window._sidebar.width() >= 188
    assert window._sidebar._list.textElideMode() == Qt.TextElideMode.ElideNone
    assert isinstance(window._sidebar._workflow_info_button, QPushButton)
    assert window._sidebar._workflow_info_button.text() == "Workflow Info"


def test_workflow_canvas_adds_nodes_and_edges(app):
    window = MainWindow()
    canvas = window._workspace.canvas
    file_node = canvas.add_workflow_node("file")
    info_node = canvas.add_workflow_node("data-info")
    assert canvas.workflow_scene.node_count() == 2
    assert canvas.workflow_scene.edge_count() == 0
    assert canvas.workflow_scene.begin_connection_drag(file_node.node_id)
    drop_point = canvas.workflow_scene._nodes[info_node.node_id].port_scene_position("input", "in-1")
    assert canvas.workflow_scene.finish_connection_drag_at(drop_point)
    assert canvas.workflow_scene.edge_count() == 1
    window._show_widget("data-info")
    assert isinstance(window._workspace.current_widget(), DataInfoScreen)


def test_workflow_canvas_accepts_connection_near_port_not_just_exact_dot(app):
    window = MainWindow()
    canvas = window._workspace.canvas
    file_node = canvas.add_workflow_node("file")
    info_node = canvas.add_workflow_node("data-info")
    assert canvas.workflow_scene.begin_connection_drag(file_node.node_id)
    exact_point = canvas.workflow_scene._nodes[info_node.node_id].port_scene_position("input", "in-1")
    near_point = QPointF(exact_point.x() + 12, exact_point.y() + 6)
    assert canvas.workflow_scene.finish_connection_drag_at(near_point)
    assert canvas.workflow_scene.edge_count() == 1


def test_workflow_canvas_can_start_connection_from_visible_port_edge(app):
    window = MainWindow()
    window.show()
    QTest.qWait(50)
    canvas = window._workspace.canvas
    file_node = canvas.add_workflow_node("file")
    node = canvas.workflow_scene._nodes[file_node.node_id]
    port_scene_pos = node.port_scene_position("output", "out-1") + QPointF(4, 0)
    viewport_pos = canvas.mapFromScene(port_scene_pos)

    QTest.mousePress(canvas.viewport(), Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, viewport_pos)

    assert canvas.workflow_scene._pending_output is not None

    QTest.mouseRelease(canvas.viewport(), Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, viewport_pos)


def test_workflow_canvas_hides_scrollbars_and_supports_keyboard_pan(app):
    window = MainWindow()
    window.show()
    QTest.qWait(50)
    canvas = window._workspace.canvas
    assert canvas.horizontalScrollBarPolicy() == Qt.ScrollBarPolicy.ScrollBarAlwaysOff
    assert canvas.verticalScrollBarPolicy() == Qt.ScrollBarPolicy.ScrollBarAlwaysOff

    center_before = canvas.mapToScene(canvas.viewport().rect().center())
    canvas.setFocus()
    QTest.keyPress(canvas, Qt.Key.Key_D)
    QTest.qWait(80)
    QTest.keyRelease(canvas, Qt.Key.Key_D)
    center_after = canvas.mapToScene(canvas.viewport().rect().center())
    assert center_after.x() > center_before.x()


def test_workflow_canvas_supports_diagonal_keyboard_pan(app):
    window = MainWindow()
    window.show()
    QTest.qWait(50)
    canvas = window._workspace.canvas
    center_before = canvas.mapToScene(canvas.viewport().rect().center())
    canvas.setFocus()
    QTest.keyPress(canvas, Qt.Key.Key_D)
    QTest.keyPress(canvas, Qt.Key.Key_W)
    QTest.qWait(80)
    QTest.keyRelease(canvas, Qt.Key.Key_D)
    QTest.keyRelease(canvas, Qt.Key.Key_W)
    center_after = canvas.mapToScene(canvas.viewport().rect().center())
    assert center_after.x() > center_before.x()
    assert center_after.y() < center_before.y()


def test_workflow_canvas_mouse_wheel_zooms(app):
    window = MainWindow()
    canvas = window._workspace.canvas
    zoom_before = canvas.zoom_percentage()
    QTest.qWait(20)
    wheel_event = None
    from PySide6.QtCore import QPoint
    from PySide6.QtGui import QWheelEvent

    position = canvas.viewport().rect().center()
    wheel_event = QWheelEvent(
        position,
        canvas.mapToGlobal(position),
        QPoint(0, 0),
        QPoint(0, 120),
        Qt.MouseButton.NoButton,
        Qt.KeyboardModifier.NoModifier,
        Qt.ScrollPhase.ScrollUpdate,
        False,
    )
    QApplication.sendEvent(canvas.viewport(), wheel_event)
    assert canvas.zoom_percentage() > zoom_before


def test_workflow_quick_tools_start_collapsed_and_can_expand(app):
    window = MainWindow()
    window.show()
    QTest.qWait(50)
    tools = window._workspace.quick_tools
    assert not tools.is_expanded()
    assert not tools._text_button.isVisible()
    tools.toggle_expanded()
    assert tools.is_expanded()
    assert tools._text_button.isVisible()
    assert tools._pan_button.text() == ""
    assert not tools._pan_button.icon().isNull()
    assert tools._pan_button.toolTip() == "Pan canvas"


def test_new_items_spawn_near_current_view_center(app):
    window = MainWindow()
    canvas = window._workspace.canvas
    canvas.pan_view(180, 120)
    center = canvas.viewport_scene_center()

    record = canvas.add_workflow_node("file")
    node = canvas.workflow_scene._nodes[record.node_id]
    assert abs(node.sceneBoundingRect().center().x() - center.x()) < 120
    assert abs(node.sceneBoundingRect().center().y() - center.y()) < 120

    annotation = canvas.add_arrow_annotation()
    assert abs(annotation.sceneBoundingRect().center().x() - center.x()) < 140


def test_minimap_appears_after_navigation(app):
    window = MainWindow()
    window.show()
    QTest.qWait(50)
    canvas = window._workspace.canvas
    canvas.add_workflow_node("file")
    canvas.zoom_in()
    QTest.qWait(20)
    assert window._workspace.mini_map.isVisible()


def test_text_annotation_supports_editing(app):
    window = MainWindow()
    annotation = window._workspace.canvas.add_text_annotation()
    assert annotation.annotation_size().width() <= 128
    annotation.begin_editing()
    annotation._text_item.setPlainText("Notes")
    assert annotation.toPlainText() == "Notes"


def test_text_annotation_grows_with_content(app):
    window = MainWindow()
    annotation = window._workspace.canvas.add_text_annotation()
    size_before = annotation.annotation_size()
    annotation._text_item.setPlainText("This is a much longer line that should make the annotation box grow horizontally.")
    size_after = annotation.annotation_size()
    assert size_after.width() == size_before.width()
    assert size_after.height() > size_before.height()


def test_text_annotation_keeps_growing_for_long_multiline_content(app):
    window = MainWindow()
    annotation = window._workspace.canvas.add_text_annotation()
    annotation._text_item.setPlainText("\n".join(["long text line"] * 12))
    assert annotation.annotation_size().height() > 180


def test_text_annotation_grows_while_typing_many_lines(app):
    window = MainWindow()
    annotation = window._workspace.canvas.add_text_annotation()
    annotation.begin_editing()
    before = annotation.annotation_size().height()
    for _ in range(14):
        cursor = annotation._text_item.textCursor()
        cursor.insertText("aaaaaaaaaaaaaa\n")
        annotation._text_item.setTextCursor(cursor)
    after = annotation.annotation_size().height()
    assert after > before + 120


def test_typing_in_text_annotation_does_not_pan_canvas(app):
    window = MainWindow()
    window.show()
    QTest.qWait(50)
    canvas = window._workspace.canvas
    annotation = canvas.add_text_annotation()
    annotation.begin_editing()
    center_before = canvas.mapToScene(canvas.viewport().rect().center())
    QTest.keyClicks(canvas.viewport(), "wasd")
    center_after = canvas.mapToScene(canvas.viewport().rect().center())
    assert annotation.toPlainText().endswith("wasd")
    assert abs(center_after.x() - center_before.x()) < 0.1
    assert abs(center_after.y() - center_before.y()) < 0.1


def test_backspace_edits_text_annotation_instead_of_deleting_it(app):
    window = MainWindow()
    window.show()
    QTest.qWait(50)
    canvas = window._workspace.canvas
    annotation = canvas.add_text_annotation()
    annotation._text_item.setPlainText("abc")
    annotation.begin_editing()
    QTest.keyClick(canvas.viewport(), Qt.Key.Key_Backspace)
    assert annotation.toPlainText() == "ab"
    assert annotation in canvas.workflow_scene._annotations


def test_invalid_connection_to_file_is_rejected(app):
    window = MainWindow()
    canvas = window._workspace.canvas
    file_node = canvas.add_workflow_node("file")
    info_node = canvas.add_workflow_node("data-info")
    assert canvas.workflow_scene.begin_connection_drag(info_node.node_id) is False
    assert not canvas.workflow_scene.create_connection(info_node.node_id, file_node.node_id)
    assert canvas.workflow_scene.edge_count() == 0


def test_duplicate_or_second_input_connection_is_rejected(app):
    window = MainWindow()
    canvas = window._workspace.canvas
    source_one = canvas.add_workflow_node("file")
    source_two = canvas.add_workflow_node("csv-import")
    target = canvas.add_workflow_node("data-table")
    assert canvas.workflow_scene.create_connection(source_one.node_id, target.node_id)
    assert not canvas.workflow_scene.create_connection(source_two.node_id, target.node_id)
    assert canvas.workflow_scene.edge_count() == 1


def test_selected_connection_can_be_deleted(app):
    window = MainWindow()
    scene = window._workspace.canvas.workflow_scene
    source = window._workspace.canvas.add_workflow_node("file")
    target = window._workspace.canvas.add_workflow_node("data-table")
    assert scene.create_connection(source.node_id, target.node_id)
    scene.clearSelection()
    edge = scene._edges[0]
    edge.setSelected(True)
    assert scene.delete_selected_items()
    assert scene.edge_count() == 0
    assert scene.node_count() == 2


def test_selected_node_deletes_attached_connections(app):
    window = MainWindow()
    scene = window._workspace.canvas.workflow_scene
    source = window._workspace.canvas.add_workflow_node("file")
    target = window._workspace.canvas.add_workflow_node("data-table")
    assert scene.create_connection(source.node_id, target.node_id)
    scene.clearSelection()
    node = scene._nodes[source.node_id]
    node.setSelected(True)
    assert scene.delete_selected_items()
    assert scene.node_count() == 1
    assert scene.edge_count() == 0


def test_backspace_and_delete_both_remove_selected_node(app):
    window = MainWindow()
    canvas = window._workspace.canvas
    scene = canvas.workflow_scene

    first = canvas.add_workflow_node("file")
    first_node = scene._nodes[first.node_id]
    scene.clearSelection()
    first_node.setSelected(True)
    canvas.keyPressEvent(QKeyEvent(QEvent.Type.KeyPress, Qt.Key.Key_Delete, Qt.KeyboardModifier.NoModifier))
    assert scene.node_count() == 0

    second = canvas.add_workflow_node("file")
    second_node = scene._nodes[second.node_id]
    scene.clearSelection()
    second_node.setSelected(True)
    canvas.keyPressEvent(QKeyEvent(QEvent.Type.KeyPress, Qt.Key.Key_Backspace, Qt.KeyboardModifier.NoModifier))
    assert scene.node_count() == 0


def test_selected_node_can_be_deleted_from_visible_delete_button(app):
    window = MainWindow()
    window.show()
    QTest.qWait(50)
    canvas = window._workspace.canvas
    record = canvas.add_workflow_node("file")
    node = canvas.workflow_scene._nodes[record.node_id]
    delete_center = node.mapToScene(node._delete_button_rect().center())
    viewport_pos = canvas.mapFromScene(delete_center)

    QTest.mouseClick(canvas.viewport(), Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, viewport_pos)

    assert canvas.workflow_scene.node_count() == 0


def test_workflow_canvas_renders_dotted_background(app):
    window = MainWindow()
    window.show()
    QTest.qWait(50)
    canvas = window._workspace.canvas
    canvas.resize(420, 260)
    image = QImage(canvas.viewport().size(), QImage.Format.Format_ARGB32)
    image.fill(Qt.GlobalColor.transparent)
    painter = QPainter(image)
    canvas.render(painter)
    painter.end()

    base_color = "#f8f8f6"
    found_non_base = False
    for x in range(0, image.width(), 4):
        for y in range(0, image.height(), 4):
            if image.pixelColor(x, y).name() != base_color:
                found_non_base = True
                break
        if found_non_base:
            break

    assert found_non_base is True


def test_undo_and_redo_restore_workflow_state(app):
    window = MainWindow()
    canvas = window._workspace.canvas
    canvas.add_workflow_node("file")
    assert canvas.workflow_scene.node_count() == 1
    window._undo_workflow()
    assert canvas.workflow_scene.node_count() == 0
    window._redo_workflow()
    assert canvas.workflow_scene.node_count() == 1


def test_widget_actions_can_rename_copy_and_paste_selection(app, monkeypatch):
    window = MainWindow()
    scene = window._workspace.canvas.workflow_scene
    record = window._workspace.canvas.add_workflow_node("file")
    node = scene._nodes[record.node_id]
    scene.clearSelection()
    node.setSelected(True)

    monkeypatch.setattr("portakal_app.ui.main_window.QInputDialog.getText", lambda *args, **kwargs: ("Input File", True))
    window._rename_selected_widget()
    assert node.display_label == "Input File"

    window._copy_selection()
    window._paste_selection()
    assert scene.node_count() == 2


def test_workflow_can_be_saved_reopened_and_frozen(app, tmp_path):
    window = MainWindow()
    scene = window._workspace.canvas.workflow_scene
    source = window._workspace.canvas.add_workflow_node("file")
    target = window._workspace.canvas.add_workflow_node("data-table")
    assert scene.create_connection(source.node_id, target.node_id)

    workflow_path = tmp_path / "demo.portakal.json"
    window._write_workflow_file(str(workflow_path))
    assert workflow_path.exists()

    window._new_workflow()
    assert scene.node_count() == 0

    window._open_workflow(str(workflow_path), freeze=True)
    assert scene.node_count() == 2
    assert scene.edge_count() == 1
    assert window._workspace.canvas.is_frozen()
    assert window.windowTitle() == "demo.portakal - Portakal"


def test_workflow_info_is_serialized(app):
    window = MainWindow()
    window._state_store.update(workflow_title="Customer Flow", workflow_description="Checks data quality")
    payload = window._serialize_workflow()
    assert payload["workflow_info"]["title"] == "Customer Flow"
    assert payload["workflow_info"]["description"] == "Checks data quality"


def test_workflow_info_dialog_enter_accepts_save(app):
    dialog = WorkflowInfoDialog("Untitled", "", None)
    dialog.show()
    QTest.qWait(30)
    accepted = {"value": False}
    dialog.accepted.connect(lambda: accepted.__setitem__("value", True))
    dialog.title_input.setFocus()
    QTest.keyClick(dialog.title_input, Qt.Key.Key_Return)
    assert accepted["value"] is True
    assert dialog.save_button.isDefault()


def test_save_uses_workflow_title_for_filename(app, tmp_path):
    window = MainWindow()
    original_path = tmp_path / "old-name.portakal.json"
    window._current_workflow_path = str(original_path)
    window._state_store.update(workflow_title="Renamed Flow")
    window._save_workflow()
    expected = tmp_path / "Renamed Flow.portakal.json"
    assert expected.exists()
    assert window._current_workflow_path == str(expected)


def test_view_and_window_actions_update_workspace_flags(app):
    window = MainWindow()
    window._toggle_tool_dock(False)
    assert not window._sidebar.isVisible()
    assert not window._catalog.isVisible()

    window._toggle_workflow_margins(False)
    assert not window._workspace.margins_visible()

    window._show_widget("file")
    window._toggle_widgets_on_top(True)
    assert window._workspace.dialogs_on_top()


def test_close_event_prompts_and_cancels_when_workflow_is_modified(app, monkeypatch):
    window = MainWindow()
    window._workspace.canvas.add_workflow_node("file")
    assert window._workflow_modified is True

    monkeypatch.setattr(
        "portakal_app.ui.main_window.QMessageBox.warning",
        lambda *args, **kwargs: QMessageBox.StandardButton.Cancel,
    )

    closed = {"value": False}
    original_ignore = QCloseEvent.ignore

    def mark_ignore(self):
        closed["value"] = True
        original_ignore(self)

    monkeypatch.setattr(QCloseEvent, "ignore", mark_ignore)

    event = QCloseEvent()
    window.closeEvent(event)

    assert closed["value"] is True


def test_close_event_discards_modified_workflow_when_user_confirms(app, monkeypatch):
    window = MainWindow()
    window._workspace.canvas.add_workflow_node("file")

    monkeypatch.setattr(
        "portakal_app.ui.main_window.QMessageBox.warning",
        lambda *args, **kwargs: QMessageBox.StandardButton.Discard,
    )

    event = QCloseEvent()
    window.closeEvent(event)
    assert event.isAccepted()
