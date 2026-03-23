from __future__ import annotations

from portakal_app.models import CategoryDefinition, PortDefinition, WidgetDefinition
from portakal_app.ui.screens.color_screen import ColorScreen
from portakal_app.ui.screens.column_statistics_screen import ColumnStatisticsScreen
from portakal_app.ui.screens.csv_import_screen import CSVImportScreen
from portakal_app.ui.screens.data_info_screen import DataInfoScreen
from portakal_app.ui.screens.data_table_screen import DataTableScreen
from portakal_app.ui.screens.datasets_screen import DatasetsScreen
from portakal_app.ui.screens.edit_domain_screen import EditDomainScreen
from portakal_app.ui.screens.file_screen import FileScreen
from portakal_app.ui.screens.paint_data_screen import PaintDataScreen
from portakal_app.ui.screens.placeholder_screen import PlaceholderScreen
from portakal_app.ui.screens.rank_screen import RankScreen
from portakal_app.ui.screens.save_data_screen import SaveDataScreen


def _placeholder_factory(title: str, description: str):
    def factory() -> PlaceholderScreen:
        return PlaceholderScreen(title=title, message=description)

    return factory


def _inputs(*labels: str) -> tuple[PortDefinition, ...]:
    return tuple(PortDefinition(id=f"in-{index}", label=label) for index, label in enumerate(labels, start=1))


def _outputs(*labels: str) -> tuple[PortDefinition, ...]:
    return tuple(PortDefinition(id=f"out-{index}", label=label) for index, label in enumerate(labels, start=1))


def build_categories() -> list[CategoryDefinition]:
    return [
        CategoryDefinition(id="data", label="Data"),
        CategoryDefinition(id="transform", label="Transform"),
        CategoryDefinition(id="visualize", label="Visualize"),
        CategoryDefinition(id="model", label="Model"),
        CategoryDefinition(id="evaluate", label="Evaluate"),
        CategoryDefinition(id="unsupervised", label="Unsupervised"),
    ]


def build_widgets() -> list[WidgetDefinition]:
    return [
        WidgetDefinition("file", "data", "File", True, FileScreen, "Load a file dataset.", "file", (), _outputs("Data")),
        WidgetDefinition(
            "csv-import",
            "data",
            "CSV File Import",
            True,
            CSVImportScreen,
            "Preset-based import flow.",
            "csv",
            (),
            _outputs("Data"),
        ),
        WidgetDefinition(
            "datasets",
            "data",
            "Datasets",
            True,
            DatasetsScreen,
            "Explore packaged datasets.",
            "dataset",
            (),
            _outputs("Data"),
        ),
        WidgetDefinition(
            "data-table",
            "data",
            "Data Table",
            True,
            DataTableScreen,
            "Inspect loaded rows.",
            "table",
            _inputs("Data"),
            _outputs("Data"),
        ),
        WidgetDefinition(
            "paint-data",
            "data",
            "Paint Data",
            True,
            PaintDataScreen,
            "Edit data manually.",
            "paint",
            _inputs("Data"),
            _outputs("Data"),
        ),
        WidgetDefinition(
            "data-info",
            "data",
            "Data Info",
            True,
            DataInfoScreen,
            "Profile the dataset.",
            "info",
            _inputs("Data"),
            (),
        ),
        WidgetDefinition(
            "rank",
            "data",
            "Rank",
            True,
            RankScreen,
            "Rank features.",
            "rank",
            _inputs("Data"),
            _outputs("Scores"),
        ),
        WidgetDefinition(
            "edit-domain",
            "data",
            "Edit Domain",
            True,
            EditDomainScreen,
            "Manage column roles.",
            "edit",
            _inputs("Data"),
            _outputs("Data"),
        ),
        WidgetDefinition(
            "color",
            "data",
            "Color",
            True,
            ColorScreen,
            "Assign color metadata.",
            "color",
            _inputs("Data"),
            _outputs("Data"),
        ),
        WidgetDefinition(
            "column-statistics",
            "data",
            "Column Statistics",
            True,
            ColumnStatisticsScreen,
            "Deep dive into column distributions.",
            "stats",
            _inputs("Data"),
            (),
        ),
        WidgetDefinition(
            "save-data",
            "data",
            "Save Data",
            True,
            SaveDataScreen,
            "Export the current dataset.",
            "save",
            _inputs("Data"),
            (),
        ),
        WidgetDefinition(
            "select-columns",
            "transform",
            "Select Columns",
            False,
            _placeholder_factory("Select Columns", "Transform widgets are planned but not part of this shell milestone."),
            "Choose feature sets.",
            "columns",
            _inputs("Data"),
            _outputs("Data"),
        ),
        WidgetDefinition(
            "normalize",
            "transform",
            "Normalize",
            False,
            _placeholder_factory("Normalize", "Transform widgets are planned but not part of this shell milestone."),
            "Scale numeric columns.",
            "normalize",
            _inputs("Data"),
            _outputs("Data"),
        ),
        WidgetDefinition(
            "scatter-plot",
            "visualize",
            "Scatter Plot",
            False,
            _placeholder_factory("Scatter Plot", "Visualization widgets will be integrated by a later group."),
            "Explore points visually.",
            "scatter",
            _inputs("Data"),
            (),
        ),
        WidgetDefinition(
            "linear-regression",
            "model",
            "Linear Regression",
            False,
            _placeholder_factory("Linear Regression", "Model widgets will be integrated by a later group."),
            "Train a baseline model.",
            "model",
            _inputs("Data"),
            _outputs("Model"),
        ),
        WidgetDefinition(
            "test-score",
            "evaluate",
            "Test & Score",
            False,
            _placeholder_factory("Test & Score", "Evaluation widgets will be integrated by a later group."),
            "Measure model performance.",
            "score",
            _inputs("Data", "Model"),
            _outputs("Scores"),
        ),
        WidgetDefinition(
            "pca",
            "unsupervised",
            "PCA",
            False,
            _placeholder_factory("PCA", "Unsupervised widgets will be integrated by a later group."),
            "Reduce dimensionality.",
            "pca",
            _inputs("Data"),
            _outputs("Data"),
        ),
    ]
