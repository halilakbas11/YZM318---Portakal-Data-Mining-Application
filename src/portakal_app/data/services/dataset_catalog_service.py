from __future__ import annotations

import tempfile
from pathlib import Path

import httpx

from portakal_app.data.models import DatasetCatalogEntry, DatasetHandle
from portakal_app.data.services.file_import_service import FileImportService


_SEABORN_BASE_URL = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master"


class DatasetCatalogService:
    _DATASETS: tuple[DatasetCatalogEntry, ...] = (
        DatasetCatalogEntry(
            dataset_id="iris",
            title="Iris",
            description="Classic flower classification dataset with four measurements and three iris species.",
            domain="Biology",
            target="categorical",
            tags=("classification", "flowers", "intro"),
            download_url=f"{_SEABORN_BASE_URL}/iris.csv",
            size_text="3.8 KB",
            row_count=150,
            column_count=5,
        ),
        DatasetCatalogEntry(
            dataset_id="titanic",
            title="Titanic",
            description="Passenger survival records used for binary classification and feature importance demos.",
            domain="Society",
            target="categorical",
            tags=("classification", "survival", "tabular"),
            download_url=f"{_SEABORN_BASE_URL}/titanic.csv",
            size_text="55.7 KB",
            row_count=891,
            column_count=15,
        ),
        DatasetCatalogEntry(
            dataset_id="penguins",
            title="Penguins",
            description="Palmer penguins dataset commonly used as a modern alternative to Iris.",
            domain="Biology",
            target="categorical",
            tags=("classification", "ecology", "measurements"),
            download_url=f"{_SEABORN_BASE_URL}/penguins.csv",
            size_text="13.2 KB",
            row_count=344,
            column_count=7,
        ),
        DatasetCatalogEntry(
            dataset_id="tips",
            title="Tips",
            description="Restaurant tipping dataset for regression, grouping, and distribution exploration.",
            domain="Business",
            target="numeric",
            tags=("regression", "hospitality", "eda"),
            download_url=f"{_SEABORN_BASE_URL}/tips.csv",
            size_text="9.5 KB",
            row_count=244,
            column_count=7,
        ),
        DatasetCatalogEntry(
            dataset_id="diamonds",
            title="Diamonds",
            description="Large retail pricing dataset used for feature engineering and regression workflows.",
            domain="Retail",
            target="numeric",
            tags=("regression", "pricing", "large"),
            download_url=f"{_SEABORN_BASE_URL}/diamonds.csv",
            size_text="2.6 MB",
            row_count=53940,
            column_count=10,
        ),
        DatasetCatalogEntry(
            dataset_id="mpg",
            title="Auto MPG",
            description="Vehicle efficiency dataset with engine and model-year attributes.",
            domain="Transport",
            target="numeric",
            tags=("regression", "automotive", "benchmark"),
            download_url=f"{_SEABORN_BASE_URL}/mpg.csv",
            size_text="20.7 KB",
            row_count=398,
            column_count=9,
        ),
        DatasetCatalogEntry(
            dataset_id="flights",
            title="Flights",
            description="Monthly airline passenger counts, useful for seasonal and time-series experiments.",
            domain="Time Series",
            target="numeric",
            tags=("forecasting", "seasonality", "transport"),
            download_url=f"{_SEABORN_BASE_URL}/flights.csv",
            size_text="2.3 KB",
            row_count=144,
            column_count=3,
        ),
        DatasetCatalogEntry(
            dataset_id="planets",
            title="Planets",
            description="Exoplanet discovery summary with detection method, mass, and orbital features.",
            domain="Astronomy",
            target="none",
            tags=("science", "exploration", "mixed-types"),
            download_url=f"{_SEABORN_BASE_URL}/planets.csv",
            size_text="35.4 KB",
            row_count=1035,
            column_count=6,
        ),
        DatasetCatalogEntry(
            dataset_id="car_crashes",
            title="Car Crashes",
            description="US state-level crash statistics often used for clustering and benchmarking dashboards.",
            domain="Public Safety",
            target="numeric",
            tags=("aggregation", "states", "risk"),
            download_url=f"{_SEABORN_BASE_URL}/car_crashes.csv",
            size_text="3.2 KB",
            row_count=51,
            column_count=8,
        ),
        DatasetCatalogEntry(
            dataset_id="taxis",
            title="Taxis",
            description="Taxi trips dataset with timestamps, fare, distance, and payment metadata.",
            domain="Transport",
            target="none",
            tags=("operations", "city", "mixed-types"),
            download_url=f"{_SEABORN_BASE_URL}/taxis.csv",
            size_text="848.9 KB",
            row_count=6433,
            column_count=14,
        ),
        DatasetCatalogEntry(
            dataset_id="anscombe",
            title="Anscombe Quartet",
            description="Four small datasets with identical summary statistics but different distributions.",
            domain="Education",
            target="none",
            tags=("visualization", "statistics", "teaching"),
            download_url=f"{_SEABORN_BASE_URL}/anscombe.csv",
            size_text="556 B",
            row_count=44,
            column_count=3,
        ),
        DatasetCatalogEntry(
            dataset_id="dots",
            title="Dots",
            description="Perceptual decision dataset suited for grouped line plots and repeated-measures analysis.",
            domain="Science",
            target="numeric",
            tags=("time-series", "neuroscience", "experiment"),
            download_url=f"{_SEABORN_BASE_URL}/dots.csv",
            size_text="8.5 KB",
            row_count=848,
            column_count=5,
        ),
        DatasetCatalogEntry(
            dataset_id="geyser",
            title="Geyser",
            description="Old Faithful eruption intervals, a common clustering and density-estimation example.",
            domain="Science",
            target="categorical",
            tags=("clustering", "density", "geology"),
            download_url=f"{_SEABORN_BASE_URL}/geyser.csv",
            size_text="7.1 KB",
            row_count=272,
            column_count=3,
        ),
        DatasetCatalogEntry(
            dataset_id="seaice",
            title="Sea Ice",
            description="Daily Arctic sea ice extent measurements for long-range trend and seasonal analysis.",
            domain="Climate",
            target="numeric",
            tags=("time-series", "climate", "trend"),
            download_url=f"{_SEABORN_BASE_URL}/seaice.csv",
            size_text="237.7 KB",
            row_count=13175,
            column_count=2,
        ),
        DatasetCatalogEntry(
            dataset_id="dowjones",
            title="Dow Jones",
            description="Historic Dow Jones Industrial Average prices, useful for financial charting demos.",
            domain="Finance",
            target="numeric",
            tags=("time-series", "markets", "prices"),
            download_url=f"{_SEABORN_BASE_URL}/dowjones.csv",
            size_text="12.9 KB",
            row_count=649,
            column_count=2,
        ),
    )

    def __init__(self, import_service: FileImportService | None = None) -> None:
        self._import_service = import_service or FileImportService()
        self._cache_root = Path(tempfile.gettempdir()) / "portakal-app" / "catalog-datasets"

    def available_datasets(self) -> tuple[DatasetCatalogEntry, ...]:
        return self._DATASETS

    def available_domains(self) -> tuple[str, ...]:
        values = sorted({entry.domain for entry in self._DATASETS})
        return ("All", *values)

    def cached_path_for(self, entry: DatasetCatalogEntry) -> Path:
        self._cache_root.mkdir(parents=True, exist_ok=True)
        return self._cache_root / f"{entry.dataset_id}.{entry.format}"

    def is_downloaded(self, entry: DatasetCatalogEntry) -> bool:
        return self.cached_path_for(entry).exists()

    def downloaded_count(self) -> int:
        return sum(1 for entry in self._DATASETS if self.is_downloaded(entry))

    def load_or_download(self, entry: DatasetCatalogEntry) -> DatasetHandle:
        path = self.cached_path_for(entry)
        if not path.exists():
            self._download(entry, path)
        return self._import_service.load(str(path))

    def _download(self, entry: DatasetCatalogEntry, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with httpx.Client(timeout=45.0, follow_redirects=True) as client:
            response = client.get(entry.download_url)
            response.raise_for_status()
        path.write_bytes(response.content)
