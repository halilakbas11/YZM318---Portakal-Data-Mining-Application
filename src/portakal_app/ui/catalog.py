from __future__ import annotations

from portakal_app.models import CategoryDefinition, PortDefinition, WidgetDefinition
from portakal_app.ui import i18n
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

from portakal_app.ui.screens.aggregate_columns_screen import AggregateColumnsScreen
from portakal_app.ui.screens.apply_domain_screen import ApplyDomainScreen
from portakal_app.ui.screens.continuize_screen import ContinuizeScreen
from portakal_app.ui.screens.data_sampler_screen import DataSamplerScreen
from portakal_app.ui.screens.discretize_screen import DiscretizeScreen
from portakal_app.ui.screens.melt_screen import MeltScreen
from portakal_app.ui.screens.purge_domain_screen import PurgeDomainScreen
from portakal_app.ui.screens.randomize_screen import RandomizeScreen
from portakal_app.ui.screens.select_by_index_screen import SelectByIndexScreen
from portakal_app.ui.screens.split_screen import SplitScreen
from portakal_app.ui.screens.transpose_screen import TransposeScreen
from portakal_app.ui.screens.unique_screen import UniqueScreen

from portakal_app.ui.screens.concatenate_screen import ConcatenateScreen
from portakal_app.ui.screens.formula_screen import FormulaScreen
from portakal_app.ui.screens.create_class_screen import CreateClassScreen
from portakal_app.ui.screens.create_instance_screen import CreateInstanceScreen
from portakal_app.ui.screens.group_by_screen import GroupByScreen
from portakal_app.ui.screens.impute_screen import ImputeScreen
from portakal_app.ui.screens.merge_data_screen import MergeDataScreen
from portakal_app.ui.screens.pivot_table_screen import PivotTableScreen
from portakal_app.ui.screens.preprocess_screen import PreprocessScreen
from portakal_app.ui.screens.python_script_screen import PythonScriptScreen
from portakal_app.ui.screens.select_columns_screen import SelectColumnsScreen
from portakal_app.ui.screens.select_rows_screen import SelectRowsScreen


def _placeholder_factory(title: str, description: str):
    def factory() -> PlaceholderScreen:
        return PlaceholderScreen(title=i18n.t(title), message=i18n.t(description))

    return factory


def _inputs(*labels: str) -> tuple[PortDefinition, ...]:
    return tuple(PortDefinition(id=f"in-{index}", label=label) for index, label in enumerate(labels, start=1))


def _outputs(*labels: str) -> tuple[PortDefinition, ...]:
    return tuple(PortDefinition(id=f"out-{index}", label=label) for index, label in enumerate(labels, start=1))


def build_categories() -> list[CategoryDefinition]:
    return [
        CategoryDefinition(id="data", label=i18n.t("Data")),
        CategoryDefinition(id="transform", label=i18n.t("Transform")),
        CategoryDefinition(id="visualize", label=i18n.t("Visualize")),
        CategoryDefinition(id="model", label=i18n.t("Model")),
        CategoryDefinition(id="evaluate", label=i18n.t("Evaluate")),
        CategoryDefinition(id="unsupervised", label=i18n.t("Unsupervised")),
    ]


def build_widgets() -> list[WidgetDefinition]:
    return [
        WidgetDefinition(
            "file",
            "data",
            i18n.t("File"),
            True,
            FileScreen,
            i18n.t("Load a file dataset."),
            "file",
            (),
            _outputs("Data"),
        ),
        WidgetDefinition(
            "csv-import",
            "data",
            i18n.t("CSV File Import"),
            True,
            CSVImportScreen,
            i18n.t("Preset-based import flow."),
            "csv",
            (),
            _outputs("Data"),
        ),
        WidgetDefinition(
            "datasets",
            "data",
            i18n.t("Datasets"),
            True,
            DatasetsScreen,
            i18n.t("Explore packaged datasets."),
            "dataset",
            (),
            _outputs("Data"),
        ),
        WidgetDefinition(
            "data-table",
            "data",
            i18n.t("Data Table"),
            True,
            DataTableScreen,
            i18n.t("Inspect loaded rows."),
            "table",
            _inputs("Data"),
            _outputs("Data"),
        ),
        WidgetDefinition(
            "paint-data",
            "data",
            i18n.t("Paint Data"),
            True,
            PaintDataScreen,
            i18n.t("Edit data manually."),
            "paint",
            _inputs("Data"),
            _outputs("Data"),
        ),
        WidgetDefinition(
            "data-info",
            "data",
            i18n.t("Data Info"),
            True,
            DataInfoScreen,
            i18n.t("Profile the dataset."),
            "info",
            _inputs("Data"),
            (),
        ),
        WidgetDefinition(
            "rank",
            "data",
            i18n.t("Rank"),
            True,
            RankScreen,
            i18n.t("Rank features."),
            "rank",
            _inputs("Data"),
            _outputs("Scores"),
        ),
        WidgetDefinition(
            "edit-domain",
            "data",
            i18n.t("Edit Domain"),
            True,
            EditDomainScreen,
            i18n.t("Manage column roles."),
            "edit",
            _inputs("Data"),
            _outputs("Data"),
        ),
        WidgetDefinition(
            "color",
            "data",
            i18n.t("Color"),
            True,
            ColorScreen,
            i18n.t("Assign color metadata."),
            "color",
            _inputs("Data"),
            _outputs("Data"),
        ),
        WidgetDefinition(
            "column-statistics",
            "data",
            i18n.t("Column Statistics"),
            True,
            ColumnStatisticsScreen,
            i18n.t("Deep dive into column distributions."),
            "stats",
            _inputs("Data"),
            (),
        ),
        WidgetDefinition(
            "save-data",
            "data",
            i18n.t("Save Data"),
            True,
            SaveDataScreen,
            i18n.t("Export the current dataset."),
            "save",
            _inputs("Data"),
            (),
        ),
        # --- Transform: Active widgets ---
        WidgetDefinition(
            "select-by-index",
            "transform",
            i18n.t("Select by Data Index"),
            True,
            SelectByIndexScreen,
            i18n.t("Match rows by index subset."),
            "index",
            _inputs("Data"),
            _outputs("Data"),
            output_channels=("Matching Data", "Non-matching Data"),
            input_channels=("Data", "Data Subset"),
        ),
        WidgetDefinition(
            "randomize",
            "transform",
            i18n.t("Randomize"),
            True,
            RandomizeScreen,
            i18n.t("Shuffle rows or columns."),
            "randomize",
            _inputs("Data"),
            _outputs("Data"),
        ),
        WidgetDefinition(
            "purge-domain",
            "transform",
            i18n.t("Purge Domain"),
            True,
            PurgeDomainScreen,
            i18n.t("Remove unused values and constant features."),
            "purge",
            _inputs("Data"),
            _outputs("Data"),
        ),
        WidgetDefinition(
            "unique",
            "transform",
            i18n.t("Unique"),
            True,
            UniqueScreen,
            i18n.t("Filter duplicate rows."),
            "unique",
            _inputs("Data"),
            _outputs("Data"),
        ),
        WidgetDefinition(
            "apply-domain",
            "transform",
            i18n.t("Apply Domain"),
            True,
            ApplyDomainScreen,
            i18n.t("Apply template domain structure."),
            "domain",
            _inputs("Data"),
            _outputs("Data"),
            input_channels=("Data", "Template Data"),
        ),
        # --- Transform: Placeholder widgets (to be implemented) ---
        WidgetDefinition(
            "data-sampler",
            "transform",
            i18n.t("Data Sampler"),
            True,
            DataSamplerScreen,
            i18n.t("Sample a subset of the data."),
            "sampler",
            _inputs("Data"),
            _outputs("Data"),
            output_channels=("Data Sample", "Remaining Data"),
        ),
        WidgetDefinition(
            "select-columns",
            "transform",
            i18n.t("Select Columns"),
            True,
            SelectColumnsScreen,
            i18n.t("Choose feature sets."),
            "columns",
            _inputs("Data"),
            _outputs("Data"),
        ),
        WidgetDefinition(
            "select-rows",
            "transform",
            i18n.t("Select Rows"),
            True,
            SelectRowsScreen,
            i18n.t("Filter rows by conditions."),
            "rows",
            _inputs("Data"),
            _outputs("Data"),
            output_channels=("Matching Data", "Unmatched Data"),
        ),
        WidgetDefinition(
            "transpose",
            "transform",
            i18n.t("Transpose"),
            True,
            TransposeScreen,
            i18n.t("Flip rows and columns."),
            "transpose",
            _inputs("Data"),
            _outputs("Data"),
        ),
        WidgetDefinition(
            "split",
            "transform",
            i18n.t("Split"),
            True,
            SplitScreen,
            i18n.t("Split string column into indicators."),
            "split",
            _inputs("Data"),
            _outputs("Data"),
        ),
        WidgetDefinition(
            "merge-data",
            "transform",
            i18n.t("Merge Data"),
            True,
            MergeDataScreen,
            i18n.t("Join two datasets by column values."),
            "merge",
            _inputs("Data", "Extra Data"),
            _outputs("Data"),
        ),
        WidgetDefinition(
            "concatenate",
            "transform",
            i18n.t("Concatenate"),
            True,
            ConcatenateScreen,
            i18n.t("Append datasets vertically."),
            "concatenate",
            _inputs("Data"),
            _outputs("Data"),
            input_channels=("Primary Data", "Additional Data"),
            multi_input_channels=("Additional Data",),
        ),
        WidgetDefinition(
            "aggregate-columns",
            "transform",
            i18n.t("Aggregate Columns"),
            True,
            AggregateColumnsScreen,
            i18n.t("Compute row-wise aggregations."),
            "aggregate",
            _inputs("Data"),
            _outputs("Data"),
        ),
        WidgetDefinition(
            "group-by",
            "transform",
            i18n.t("Group By"),
            True,
            GroupByScreen,
            i18n.t("Group and aggregate data."),
            "groupby",
            _inputs("Data"),
            _outputs("Data"),
        ),
        WidgetDefinition(
            "pivot-table",
            "transform",
            i18n.t("Pivot Table"),
            True,
            PivotTableScreen,
            i18n.t("Create cross-tabulations."),
            "pivot",
            _inputs("Data"),
            _outputs("Data"),
        ),
        WidgetDefinition(
            "preprocess",
            "transform",
            i18n.t("Preprocess"),
            True,
            PreprocessScreen,
            i18n.t("Build preprocessing pipelines."),
            "preprocess",
            _inputs("Data"),
            _outputs("Data"),
        ),
        WidgetDefinition(
            "impute",
            "transform",
            i18n.t("Impute"),
            True,
            ImputeScreen,
            i18n.t("Fill missing values."),
            "impute",
            _inputs("Data"),
            _outputs("Data"),
        ),
        WidgetDefinition(
            "continuize",
            "transform",
            i18n.t("Continuize"),
            True,
            ContinuizeScreen,
            i18n.t("Convert categorical to numeric."),
            "continuize",
            _inputs("Data"),
            _outputs("Data"),
        ),
        WidgetDefinition(
            "discretize",
            "transform",
            i18n.t("Discretize"),
            True,
            DiscretizeScreen,
            i18n.t("Convert numeric to categorical."),
            "discretize",
            _inputs("Data"),
            _outputs("Data"),
        ),
        WidgetDefinition(
            "melt",
            "transform",
            i18n.t("Melt"),
            True,
            MeltScreen,
            i18n.t("Wide to long format."),
            "melt",
            _inputs("Data"),
            _outputs("Data"),
        ),
        WidgetDefinition(
            "create-class",
            "transform",
            i18n.t("Create Class"),
            True,
            CreateClassScreen,
            i18n.t("Create class from string patterns."),
            "createclass",
            _inputs("Data"),
            _outputs("Data"),
        ),
        WidgetDefinition(
            "create-instance",
            "transform",
            i18n.t("Create Instance"),
            True,
            CreateInstanceScreen,
            i18n.t("Create a single data instance."),
            "createinstance",
            _inputs("Data"),
            _outputs("Data"),
            input_channels=("Data", "Reference"),
        ),
        WidgetDefinition(
            "formula",
            "transform",
            i18n.t("Formula"),
            True,
            FormulaScreen,
            i18n.t("Construct features with expressions."),
            "formula",
            _inputs("Data"),
            _outputs("Data"),
        ),
        WidgetDefinition(
            "python-script",
            "transform",
            i18n.t("Python Script"),
            True,
            PythonScriptScreen,
            i18n.t("Run custom Python code."),
            "python",
            _inputs("Data"),
            _outputs("Data"),
        ),
        WidgetDefinition(
            "scatter-plot",
            "visualize",
            i18n.t("Scatter Plot"),
            False,
            _placeholder_factory("Scatter Plot", "Visualization widgets will be integrated by a later group."),
            i18n.t("Explore points visually."),
            "scatter",
            _inputs("Data"),
            (),
        ),
        WidgetDefinition(
            "linear-regression",
            "model",
            i18n.t("Linear Regression"),
            False,
            _placeholder_factory("Linear Regression", "Model widgets will be integrated by a later group."),
            i18n.t("Train a baseline model."),
            "model",
            _inputs("Data"),
            _outputs("Model"),
        ),
        WidgetDefinition(
            "test-score",
            "evaluate",
            i18n.t("Test & Score"),
            False,
            _placeholder_factory("Test & Score", "Evaluation widgets will be integrated by a later group."),
            i18n.t("Measure model performance."),
            "score",
            _inputs("Data", "Model"),
            _outputs("Scores"),
        ),
        WidgetDefinition(
            "pca",
            "unsupervised",
            i18n.t("PCA"),
            False,
            _placeholder_factory("PCA", "Unsupervised widgets will be integrated by a later group."),
            i18n.t("Reduce dimensionality."),
            "pca",
            _inputs("Data"),
            _outputs("Data"),
        ),
    ]
