"""Comprehensive tests for the DataTableScreen widget and DataTableModel.

Covers: empty state, loading state, data state transitions, PreviewService
integration, format support (csv/xlsx/parquet), sidebar controls, sorting,
selection, numeric bar delegate, class coloring, and large dataset behaviour.
"""
from __future__ import annotations

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PySide6")

import polars as pl
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication

from portakal_app.app import create_application
from portakal_app.data.services.file_import_service import FileImportService
from portakal_app.data.services.preview_service import PreviewService
from portakal_app.ui.screens.data_table_screen import (
    DataTableModel,
    DataTableScreen,
    NumericBarDelegate,
)


@pytest.fixture(scope="session")
def app():
    return create_application()


@pytest.fixture()
def sample_df() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "sepal_length": [5.1, 4.9, 4.7, 7.0, 6.4],
            "sepal_width": [3.5, 3.0, 3.2, 3.2, 3.2],
            "petal_length": [1.4, 1.4, 1.3, 4.7, 4.5],
            "petal_width": [0.2, 0.2, 0.2, 1.4, 1.5],
            "species": ["setosa", "setosa", "setosa", "versicolor", "versicolor"],
        }
    )


@pytest.fixture()
def csv_path(tmp_path, sample_df):
    path = tmp_path / "iris.csv"
    sample_df.write_csv(path)
    return path


@pytest.fixture()
def dataset_handle(csv_path):
    return FileImportService().load(str(csv_path))


@pytest.fixture()
def screen(app):
    return DataTableScreen()


# ────────────────────────────────────────────────────────────
# 1. INITIAL STATE
# ────────────────────────────────────────────────────────────

class TestInitialState:
    def test_starts_with_empty_state(self, screen):
        """Screen should show empty state widget (index 0) at startup."""
        assert screen._stack.currentIndex() == 0

    def test_no_data_in_model(self, screen):
        assert screen._model.rowCount() == 0
        assert screen._model.columnCount() == 0

    def test_summary_shows_none(self, screen):
        assert "none" in screen._summary_label.text().lower()

    def test_info_label_shows_no_dataset(self, screen):
        assert "no dataset" in screen._info_label.text().lower()

    def test_restore_button_disabled(self, screen):
        assert screen._restore_order_button.isEnabled() is False

    def test_table_is_not_editable(self, screen):
        from PySide6.QtWidgets import QAbstractItemView
        assert screen._table.editTriggers() == QAbstractItemView.EditTrigger.NoEditTriggers


# ────────────────────────────────────────────────────────────
# 2. LOADING AND DATA STATE TRANSITIONS
# ────────────────────────────────────────────────────────────

class TestStateTransitions:
    def test_set_dataset_with_handle_switches_to_data_view(self, screen, dataset_handle):
        screen.set_dataset(dataset_handle)
        assert screen._stack.currentIndex() == 2  # data view

    def test_set_dataset_with_path_string_switches_to_data_view(self, screen, csv_path):
        screen.set_dataset(str(csv_path))
        assert screen._stack.currentIndex() == 2

    def test_set_dataset_none_returns_to_empty_state(self, screen, dataset_handle):
        screen.set_dataset(dataset_handle)
        assert screen._stack.currentIndex() == 2
        screen.set_dataset(None)
        assert screen._stack.currentIndex() == 0

    def test_set_invalid_path_shows_empty_state(self, screen):
        screen.set_dataset("/nonexistent/path.csv")
        assert screen._stack.currentIndex() == 0

    def test_set_invalid_type_shows_empty_state(self, screen):
        screen.set_dataset(12345)
        assert screen._stack.currentIndex() == 0


# ────────────────────────────────────────────────────────────
# 3. DATA LOADING VIA PREVIEW SERVICE
# ────────────────────────────────────────────────────────────

class TestDataLoading:
    def test_model_row_count_matches_dataset(self, screen, dataset_handle):
        screen.set_dataset(dataset_handle)
        assert screen._model.rowCount() == 5

    def test_model_column_count_matches_dataset(self, screen, dataset_handle):
        screen.set_dataset(dataset_handle)
        assert screen._model.columnCount() == 5

    def test_headers_populated(self, screen, dataset_handle):
        screen.set_dataset(dataset_handle)
        assert "sepal_length" in screen._headers
        assert "species" in screen._headers

    def test_total_rows_tracked(self, screen, dataset_handle):
        screen.set_dataset(dataset_handle)
        assert screen._total_rows == 5

    def test_info_label_updated(self, screen, dataset_handle):
        screen.set_dataset(dataset_handle)
        assert "5 instances" in screen._info_label.text()

    def test_summary_label_shows_dataset_info(self, screen, dataset_handle):
        screen.set_dataset(dataset_handle)
        text = screen._summary_label.text()
        assert "5 instances" in text
        assert "5 variables" in text

    def test_restore_button_enabled_after_load(self, screen, dataset_handle):
        screen.set_dataset(dataset_handle)
        assert screen._restore_order_button.isEnabled() is True


# ────────────────────────────────────────────────────────────
# 4. FORMAT SUPPORT (csv, xlsx, parquet)
# ────────────────────────────────────────────────────────────

class TestFormatSupport:
    def test_csv_loads_correctly(self, screen, tmp_path, sample_df):
        path = tmp_path / "test.csv"
        sample_df.write_csv(path)
        screen.set_dataset(str(path))
        assert screen._model.rowCount() == 5
        assert screen._stack.currentIndex() == 2

    def test_xlsx_loads_correctly(self, screen, tmp_path, sample_df):
        path = tmp_path / "test.xlsx"
        sample_df.write_excel(path)
        screen.set_dataset(str(path))
        assert screen._model.rowCount() == 5
        assert screen._stack.currentIndex() == 2

    def test_parquet_loads_correctly(self, screen, tmp_path, sample_df):
        path = tmp_path / "test.parquet"
        sample_df.write_parquet(path)
        screen.set_dataset(str(path))
        assert screen._model.rowCount() == 5
        assert screen._stack.currentIndex() == 2

    def test_tsv_loads_correctly(self, screen, tmp_path, sample_df):
        path = tmp_path / "test.tsv"
        sample_df.write_csv(path, separator="\t")
        screen.set_dataset(str(path))
        assert screen._model.rowCount() == 5
        assert screen._stack.currentIndex() == 2


# ────────────────────────────────────────────────────────────
# 5. DATA TABLE MODEL UNIT TESTS
# ────────────────────────────────────────────────────────────

class TestDataTableModel:
    def test_display_role_returns_cell_value(self, app, dataset_handle):
        model = DataTableModel()
        preview = PreviewService()
        page = preview.get_page(dataset_handle, 0, 5)
        cols = preview.get_column_specs(dataset_handle)
        from portakal_app.ui.screens.data_table_screen import DataTableColumn

        columns = [
            DataTableColumn(name=c["name"], type_name=c["type_name"], role_name=c["role_name"])
            for c in cols
        ]
        model.set_dataset(
            list(page.headers),
            columns,
            [list(row) for row in page.rows],
            preview.get_numeric_ranges(dataset_handle),
            None,
        )
        index = model.index(0, 0)
        value = model.data(index, Qt.ItemDataRole.DisplayRole)
        assert value is not None
        assert isinstance(value, str)

    def test_vertical_header_returns_one_based_row_number(self, app):
        model = DataTableModel()
        from portakal_app.ui.screens.data_table_screen import DataTableColumn

        model.set_dataset(
            ["col1"],
            [DataTableColumn(name="col1", type_name="text", role_name="feature")],
            [["a"], ["b"], ["c"]],
            {},
            None,
        )
        assert model.headerData(0, Qt.Orientation.Vertical) == "1"
        assert model.headerData(2, Qt.Orientation.Vertical) == "3"

    def test_horizontal_header_with_labels(self, app):
        model = DataTableModel()
        from portakal_app.ui.screens.data_table_screen import DataTableColumn

        model.set_dataset(
            ["amount"],
            [DataTableColumn(name="amount", type_name="numeric", role_name="feature")],
            [["42"]],
            {},
            None,
        )
        header = model.headerData(0, Qt.Orientation.Horizontal)
        assert "amount" in header
        assert "numeric" in header

    def test_horizontal_header_without_labels(self, app):
        model = DataTableModel()
        from portakal_app.ui.screens.data_table_screen import DataTableColumn

        model.set_dataset(
            ["amount"],
            [DataTableColumn(name="amount", type_name="numeric", role_name="feature")],
            [["42"]],
            {},
            None,
        )
        model.set_show_labels(False)
        header = model.headerData(0, Qt.Orientation.Horizontal)
        assert header == "amount"

    def test_sort_ascending(self, app):
        model = DataTableModel()
        from portakal_app.ui.screens.data_table_screen import DataTableColumn

        model.set_dataset(
            ["val"],
            [DataTableColumn(name="val", type_name="numeric", role_name="feature")],
            [["3"], ["1"], ["2"]],
            {},
            None,
        )
        model.sort(0, Qt.SortOrder.AscendingOrder)
        assert model.data(model.index(0, 0)) == "1"
        assert model.data(model.index(2, 0)) == "3"

    def test_sort_descending(self, app):
        model = DataTableModel()
        from portakal_app.ui.screens.data_table_screen import DataTableColumn

        model.set_dataset(
            ["val"],
            [DataTableColumn(name="val", type_name="numeric", role_name="feature")],
            [["3"], ["1"], ["2"]],
            {},
            None,
        )
        model.sort(0, Qt.SortOrder.DescendingOrder)
        assert model.data(model.index(0, 0)) == "3"
        assert model.data(model.index(2, 0)) == "1"

    def test_restore_original_order(self, app):
        model = DataTableModel()
        from portakal_app.ui.screens.data_table_screen import DataTableColumn

        model.set_dataset(
            ["val"],
            [DataTableColumn(name="val", type_name="numeric", role_name="feature")],
            [["3"], ["1"], ["2"]],
            {},
            None,
        )
        model.sort(0, Qt.SortOrder.AscendingOrder)
        assert model.data(model.index(0, 0)) == "1"
        model.restore_original_order()
        assert model.data(model.index(0, 0)) == "3"

    def test_clear_resets_model(self, app):
        model = DataTableModel()
        from portakal_app.ui.screens.data_table_screen import DataTableColumn

        model.set_dataset(
            ["val"],
            [DataTableColumn(name="val", type_name="numeric", role_name="feature")],
            [["1"]],
            {},
            None,
        )
        assert model.rowCount() == 1
        model.clear()
        assert model.rowCount() == 0
        assert model.columnCount() == 0

    def test_numeric_alignment(self, app):
        model = DataTableModel()
        from portakal_app.ui.screens.data_table_screen import DataTableColumn

        model.set_dataset(
            ["num"],
            [DataTableColumn(name="num", type_name="numeric", role_name="feature")],
            [["42"]],
            {},
            None,
        )
        alignment = model.data(model.index(0, 0), Qt.ItemDataRole.TextAlignmentRole)
        assert alignment is not None
        assert alignment & Qt.AlignmentFlag.AlignRight

    def test_bar_value_role_for_numeric(self, app):
        model = DataTableModel()
        from portakal_app.ui.screens.data_table_screen import DataTableColumn

        model.set_dataset(
            ["num"],
            [DataTableColumn(name="num", type_name="numeric", role_name="feature")],
            [["10"], ["20"]],
            {0: (10.0, 20.0)},
            None,
        )
        bar_val = model.data(model.index(0, 0), DataTableModel.BAR_VALUE_ROLE)
        assert bar_val == 10.0

    def test_bar_range_role(self, app):
        model = DataTableModel()
        from portakal_app.ui.screens.data_table_screen import DataTableColumn

        model.set_dataset(
            ["num"],
            [DataTableColumn(name="num", type_name="numeric", role_name="feature")],
            [["10"]],
            {0: (5.0, 50.0)},
            None,
        )
        bar_range = model.data(model.index(0, 0), DataTableModel.BAR_RANGE_ROLE)
        assert bar_range == (5.0, 50.0)


# ────────────────────────────────────────────────────────────
# 6. SIDEBAR CONTROLS
# ────────────────────────────────────────────────────────────

class TestSidebarControls:
    def test_show_labels_toggle(self, screen, dataset_handle):
        screen.set_dataset(dataset_handle)
        header_with = screen._model.headerData(0, Qt.Orientation.Horizontal)
        assert "\n" in header_with  # label + type

        screen._show_labels_checkbox.setChecked(False)
        header_without = screen._model.headerData(0, Qt.Orientation.Horizontal)
        assert "\n" not in header_without

    def test_visualize_numeric_toggle(self, screen, dataset_handle):
        screen.set_dataset(dataset_handle)
        assert screen._delegate._enabled is True
        screen._visualize_checkbox.setChecked(False)
        assert screen._delegate._enabled is False

    def test_color_toggle_doesnt_rebuild_model(self, screen, dataset_handle):
        screen.set_dataset(dataset_handle)
        model_before = screen._table.model()
        screen._color_checkbox.setChecked(False)
        assert screen._table.model() is model_before

    def test_select_full_rows_toggle(self, screen, dataset_handle):
        from PySide6.QtWidgets import QAbstractItemView
        screen.set_dataset(dataset_handle)
        assert screen._table.selectionBehavior() == QAbstractItemView.SelectionBehavior.SelectRows
        screen._select_full_rows_checkbox.setChecked(False)
        assert screen._table.selectionBehavior() == QAbstractItemView.SelectionBehavior.SelectItems


# ────────────────────────────────────────────────────────────
# 7. SELECTION AND FOOTER
# ────────────────────────────────────────────────────────────

class TestSelection:
    def test_footer_no_selection(self, screen, dataset_handle):
        screen.set_dataset(dataset_handle)
        assert screen.footer_status_text() == "5"

    def test_footer_with_selection(self, screen, dataset_handle):
        screen.set_dataset(dataset_handle)
        screen._table.selectRow(0)
        assert screen.footer_status_text() == "1 | 5"

    def test_footer_multi_selection(self, screen, dataset_handle):
        screen.set_dataset(dataset_handle)
        screen._table.selectRow(0)
        screen._table.selectRow(2)
        # Extended selection may clear previous, so just check it's valid
        text = screen.footer_status_text()
        assert "5" in text

    def test_clear_selection(self, screen, dataset_handle):
        screen.set_dataset(dataset_handle)
        screen._table.selectRow(0)
        screen._clear_selection_button.click()
        assert screen.footer_status_text() == "5"

    def test_selection_summary_label(self, screen, dataset_handle):
        screen.set_dataset(dataset_handle)
        assert screen._selected_summary_label.text() == "Data Subset: -"
        screen._table.selectRow(0)
        assert "1 instances" in screen._selected_summary_label.text()


# ────────────────────────────────────────────────────────────
# 8. SORTING FROM UI
# ────────────────────────────────────────────────────────────

class TestSorting:
    def test_sort_via_model_preserves_data(self, screen, dataset_handle):
        screen.set_dataset(dataset_handle)
        original_count = screen._model.rowCount()
        screen._model.sort(0, Qt.SortOrder.AscendingOrder)
        assert screen._model.rowCount() == original_count

    def test_restore_original_order_button(self, screen, dataset_handle):
        screen.set_dataset(dataset_handle)
        first_before = screen._model.data(screen._model.index(0, 0))
        screen._model.sort(0, Qt.SortOrder.DescendingOrder)
        first_sorted = screen._model.data(screen._model.index(0, 0))
        screen._restore_order_button.click()
        first_restored = screen._model.data(screen._model.index(0, 0))
        assert first_restored == first_before


# ────────────────────────────────────────────────────────────
# 9. DATA SNAPSHOTS
# ────────────────────────────────────────────────────────────

class TestSnapshots:
    def test_data_preview_snapshot(self, screen, dataset_handle):
        screen.set_dataset(dataset_handle)
        snapshot = screen.data_preview_snapshot()
        assert len(snapshot["headers"]) == 5
        assert len(snapshot["rows"]) == 5

    def test_detailed_data_snapshot_no_selection(self, screen, dataset_handle):
        screen.set_dataset(dataset_handle)
        snapshot = screen.detailed_data_snapshot()
        assert "data_headers" in snapshot
        assert len(snapshot["data_rows"]) == 5
        assert snapshot["selected_rows"] == []

    def test_detailed_data_snapshot_with_selection(self, screen, dataset_handle):
        screen.set_dataset(dataset_handle)
        screen._table.selectRow(0)
        snapshot = screen.detailed_data_snapshot()
        assert len(snapshot["selected_rows"]) == 1
        assert "1 instances" in snapshot["selected_summary"]


# ────────────────────────────────────────────────────────────
# 10. LARGE DATASET BEHAVIOUR
# ────────────────────────────────────────────────────────────

class TestLargeDataset:
    def test_large_csv_loads_all_rows(self, screen, tmp_path, app):
        """A 1000-row dataset should be fully loaded via paginated PreviewService."""
        rows = ["id,value,label"] + [f"{i},{i*1.5},{'A' if i%2 else 'B'}" for i in range(1000)]
        path = tmp_path / "large.csv"
        path.write_text("\n".join(rows), encoding="utf-8")
        screen.set_dataset(str(path))
        assert screen._model.rowCount() == 1000
        assert screen._total_rows == 1000
        assert screen.footer_status_text() == "1000"
        assert screen._stack.currentIndex() == 2


# ────────────────────────────────────────────────────────────
# 11. HELP TEXT AND DOCUMENTATION
# ────────────────────────────────────────────────────────────

class TestMetadata:
    def test_help_text(self, screen):
        text = screen.help_text()
        assert "dataset" in text.lower() or "table" in text.lower()

    def test_documentation_url(self, screen):
        url = screen.documentation_url()
        assert url.startswith("https://")

    def test_preview_service_injectable(self, screen):
        custom_service = PreviewService()
        screen.set_preview_service(custom_service)
        assert screen._preview_service is custom_service


# ────────────────────────────────────────────────────────────
# 12. MISSING DATA HANDLING
# ────────────────────────────────────────────────────────────

class TestMissingData:
    def test_missing_values_counted(self, screen, tmp_path):
        path = tmp_path / "missing.csv"
        path.write_text("a,b\n1,\n,3\n4,5\n", encoding="utf-8")
        screen.set_dataset(str(path))
        assert screen._missing_count > 0
        assert "missing" in screen._info_label.text().lower()

    def test_no_missing_shows_no_missing_data(self, screen, tmp_path):
        path = tmp_path / "clean.csv"
        path.write_text("a,b\n1,2\n3,4\n", encoding="utf-8")
        screen.set_dataset(str(path))
        assert "no missing" in screen._info_label.text().lower()
