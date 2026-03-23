from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from portakal_app.data.services.file_import_service import FileImportService
from portakal_app.data.services.preview_service import PreviewService


@pytest.fixture()
def sample_dataframe() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "amount": [10.5, 20.0, 13.25, 7.0, 30.5, 42.1, 5.0, 18.75],
            "event_time": [
                "2026-03-18 10:00", "2026-03-19 11:30", "2026-03-20 09:15", "2026-03-21 14:00",
                "2026-03-22 08:45", "2026-03-23 16:30", "2026-03-24 12:00", "2026-03-25 10:15",
            ],
            "species": ["setosa", "versicolor", "setosa", "virginica", "versicolor", "setosa", "virginica", "setosa"],
        }
    )


@pytest.fixture()
def sample_dataset(tmp_path, sample_dataframe) -> "DatasetHandle":
    service = FileImportService()
    csv_path = tmp_path / "sample.csv"
    sample_dataframe.write_csv(csv_path)
    return service.load(str(csv_path))


def test_get_page_returns_correct_slice(sample_dataset):
    preview = PreviewService()
    page = preview.get_page(sample_dataset, offset=0, limit=3)

    assert page.total_rows == 8
    assert page.offset == 0
    assert page.limit == 3
    assert len(page.rows) == 3
    assert len(page.headers) == 3
    assert "amount" in page.headers
    assert "species" in page.headers


def test_get_page_offset_beyond_total_returns_empty(sample_dataset):
    preview = PreviewService()
    page = preview.get_page(sample_dataset, offset=100, limit=10)

    assert page.total_rows == 8
    assert len(page.rows) == 0


def test_get_page_all_cells_are_strings(sample_dataset):
    preview = PreviewService()
    page = preview.get_page(sample_dataset, offset=0, limit=8)

    for row in page.rows:
        for cell in row:
            assert isinstance(cell, str), f"Cell value {cell!r} is not a string"


def test_get_total_rows(sample_dataset):
    preview = PreviewService()
    assert preview.get_total_rows(sample_dataset) == 8


def test_get_numeric_ranges_for_numeric_columns(sample_dataset):
    preview = PreviewService()
    ranges = preview.get_numeric_ranges(sample_dataset)

    assert len(ranges) >= 1
    # amount column should be at index 0
    assert 0 in ranges
    col_min, col_max = ranges[0]
    assert col_min == pytest.approx(5.0)
    assert col_max == pytest.approx(42.1)


def test_get_column_specs_returns_all_columns(sample_dataset):
    preview = PreviewService()
    specs = preview.get_column_specs(sample_dataset)

    assert len(specs) == 3
    names = [spec["name"] for spec in specs]
    assert "amount" in names
    assert "event_time" in names
    assert "species" in names

    amount_spec = next(s for s in specs if s["name"] == "amount")
    assert amount_spec["type_name"] == "numeric"

    species_spec = next(s for s in specs if s["name"] == "species")
    assert species_spec["type_name"] == "categorical"


def test_get_page_with_xlsx_format(tmp_path, sample_dataframe):
    service = FileImportService()
    xlsx_path = tmp_path / "sample.xlsx"
    sample_dataframe.write_excel(xlsx_path)
    dataset = service.load(str(xlsx_path))

    preview = PreviewService()
    page = preview.get_page(dataset, offset=0, limit=5)

    assert page.total_rows == 8
    assert len(page.rows) == 5


def test_get_page_with_parquet_format(tmp_path, sample_dataframe):
    service = FileImportService()
    parquet_path = tmp_path / "sample.parquet"
    sample_dataframe.write_parquet(parquet_path)
    dataset = service.load(str(parquet_path))

    preview = PreviewService()
    page = preview.get_page(dataset, offset=2, limit=3)

    assert page.total_rows == 8
    assert page.offset == 2
    assert len(page.rows) == 3
