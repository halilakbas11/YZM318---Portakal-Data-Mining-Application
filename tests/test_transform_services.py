"""Tests for all 24 Transform widget services."""
from __future__ import annotations

import polars as pl
import pytest

from portakal_app.data.services.file_import_service import FileImportService


@pytest.fixture()
def basic_dataset(tmp_path):
    path = tmp_path / "basic.csv"
    path.write_text(
        "age,score,city\n25,80,Ankara\n30,90,Istanbul\n25,70,Ankara\n35,85,Izmir\n",
        encoding="utf-8",
    )
    return FileImportService().load(str(path))


@pytest.fixture()
def numeric_dataset(tmp_path):
    path = tmp_path / "numeric.csv"
    path.write_text(
        "x,y,z,label\n1,10,100,A\n2,20,200,A\n3,30,300,B\n4,40,400,B\n5,50,500,C\n",
        encoding="utf-8",
    )
    return FileImportService().load(str(path))


@pytest.fixture()
def missing_dataset(tmp_path):
    path = tmp_path / "missing.csv"
    pl.DataFrame(
        {"a": [1.0, None, 3.0, None, 5.0], "b": [10.0, 20.0, None, 40.0, 50.0], "c": ["X", "Y", "X", "Y", "X"]}
    ).write_csv(path)
    return FileImportService().load(str(path))


# 1. Select by Data Index
class TestSelectByIndexService:
    def test_select_matching_rows(self, basic_dataset, tmp_path):
        from portakal_app.data.services.select_by_index_service import SelectByIndexService

        path = tmp_path / "subset.csv"
        path.write_text("age,score,city\n25,80,Ankara\n30,90,Istanbul\n", encoding="utf-8")
        subset = FileImportService().load(str(path))

        matching, non_matching = SelectByIndexService().select(basic_dataset, subset)
        assert matching is not None
        assert matching.row_count == 2
        assert non_matching is not None
        assert non_matching.row_count == 2


# 2. Randomize
class TestRandomizeService:
    def test_shuffle_classes_with_seed(self, numeric_dataset):
        from portakal_app.data.services.randomize_service import RandomizeService

        result = RandomizeService().randomize(numeric_dataset, shuffle_classes=True, seed=42)
        assert result.row_count == numeric_dataset.row_count
        assert result.column_count == numeric_dataset.column_count

    def test_deterministic_with_same_seed(self, numeric_dataset):
        from portakal_app.data.services.randomize_service import RandomizeService

        svc = RandomizeService()
        r1 = svc.randomize(numeric_dataset, shuffle_classes=True, seed=123)
        r2 = svc.randomize(numeric_dataset, shuffle_classes=True, seed=123)
        assert r1.dataframe["label"].to_list() == r2.dataframe["label"].to_list()


# 3. Purge Domain
class TestPurgeDomainService:
    def test_removes_constant_columns(self, tmp_path):
        from portakal_app.data.services.purge_domain_service import PurgeDomainService

        path = tmp_path / "constant.csv"
        path.write_text("a,b,c\n1,5,X\n2,5,X\n3,5,X\n", encoding="utf-8")
        dataset = FileImportService().load(str(path))

        purged, stats = PurgeDomainService().purge(dataset, remove_constant_features=True)
        assert "b" not in purged.dataframe.columns
        assert stats["features"]["removed"] >= 1


# 4. Unique
class TestUniqueService:
    def test_first_instance(self, basic_dataset):
        from portakal_app.data.services.unique_service import UniqueService

        result = UniqueService().filter_unique(basic_dataset, group_by_columns=["city"], tiebreaker="First instance")
        assert result.row_count == 3  # Ankara, Istanbul, Izmir

    def test_discard_non_unique(self, basic_dataset):
        from portakal_app.data.services.unique_service import UniqueService

        result = UniqueService().filter_unique(basic_dataset, group_by_columns=["city"], tiebreaker="Discard non-unique")
        # Ankara appears twice, so only Istanbul and Izmir remain
        assert result.row_count == 2


# 5. Apply Domain
class TestApplyDomainService:
    def test_applies_template_domain(self, tmp_path):
        from portakal_app.data.services.apply_domain_service import ApplyDomainService

        svc = FileImportService()
        p1 = tmp_path / "data.csv"
        p1.write_text("a,b,c\n1,2,X\n3,4,Y\n", encoding="utf-8")
        p2 = tmp_path / "template.csv"
        p2.write_text("a,b,d\n10,20,Z\n", encoding="utf-8")

        data = svc.load(str(p1))
        template = svc.load(str(p2))

        result = ApplyDomainService().apply(data, template)
        assert "a" in result.dataframe.columns
        assert "b" in result.dataframe.columns
        assert "d" in result.dataframe.columns
        assert "c" not in result.dataframe.columns


# 6. Data Sampler
class TestDataSamplerService:
    def test_percentage_sampling(self, numeric_dataset):
        from portakal_app.data.services.data_sampler_service import DataSamplerService

        sample, remaining = DataSamplerService().sample(numeric_dataset, mode="percentage", percentage=60, seed=42)
        assert sample.row_count == 3  # 60% of 5
        assert remaining is not None
        assert sample.row_count + remaining.row_count == numeric_dataset.row_count

    def test_fixed_size_sampling(self, numeric_dataset):
        from portakal_app.data.services.data_sampler_service import DataSamplerService

        sample, remaining = DataSamplerService().sample(numeric_dataset, mode="fixed", fixed_size=2, seed=42)
        assert sample.row_count == 2

    def test_cross_validation(self, numeric_dataset):
        from portakal_app.data.services.data_sampler_service import DataSamplerService

        sample, remaining = DataSamplerService().sample(numeric_dataset, mode="cross-validation", folds=5, selected_fold=1)
        assert sample is not None
        assert remaining is not None
        assert sample.row_count + remaining.row_count == numeric_dataset.row_count


# 7. Transpose
class TestTransposeService:
    def test_basic_transpose(self, tmp_path):
        from portakal_app.data.services.transpose_service import TransposeService

        path = tmp_path / "transpose.csv"
        path.write_text("a,b,c\n1,2,3\n4,5,6\n", encoding="utf-8")
        dataset = FileImportService().load(str(path))

        result = TransposeService().transpose(dataset)
        assert result.row_count == 3  # 3 columns become 3 rows
        assert result.column_count == 3  # header col + 2 data cols


# 8. Split
class TestSplitService:
    def test_split_delimiter(self, tmp_path):
        from portakal_app.data.services.split_service import SplitService

        path = tmp_path / "split.csv"
        path.write_text("id,tags\n1,A;B\n2,B;C\n3,A;C\n", encoding="utf-8")
        dataset = FileImportService().load(str(path))

        result = SplitService().split_column(dataset, column_name="tags", delimiter=";")
        assert result.column_count > dataset.column_count
        assert "tags - A" in result.dataframe.columns
        assert "tags - B" in result.dataframe.columns
        assert "tags - C" in result.dataframe.columns


# 9. Merge Data
class TestMergeDataService:
    def test_left_join(self, tmp_path):
        from portakal_app.data.services.merge_data_service import MergeDataService

        svc = FileImportService()
        p1 = tmp_path / "left.csv"
        p1.write_text("id,name\n1,Alice\n2,Bob\n3,Charlie\n", encoding="utf-8")
        p2 = tmp_path / "right.csv"
        p2.write_text("id,score\n1,90\n2,85\n", encoding="utf-8")

        left = svc.load(str(p1))
        right = svc.load(str(p2))

        result = MergeDataService().merge(left, right, left_on="id", right_on="id", join_type="Left Join")
        assert result.row_count == 3
        assert "score" in result.dataframe.columns

    def test_inner_join(self, tmp_path):
        from portakal_app.data.services.merge_data_service import MergeDataService

        svc = FileImportService()
        p1 = tmp_path / "left2.csv"
        p1.write_text("id,name\n1,Alice\n2,Bob\n3,Charlie\n", encoding="utf-8")
        p2 = tmp_path / "right2.csv"
        p2.write_text("id,score\n1,90\n2,85\n", encoding="utf-8")

        left = svc.load(str(p1))
        right = svc.load(str(p2))

        result = MergeDataService().merge(left, right, left_on="id", right_on="id", join_type="Inner Join")
        assert result.row_count == 2


# 10. Concatenate
class TestConcatenateService:
    def test_union_concatenation(self, tmp_path):
        from portakal_app.data.services.concatenate_service import ConcatenateService

        svc = FileImportService()
        p1 = tmp_path / "d1.csv"
        p1.write_text("a,b\n1,2\n3,4\n", encoding="utf-8")
        p2 = tmp_path / "d2.csv"
        p2.write_text("a,b,c\n5,6,7\n", encoding="utf-8")

        d1 = svc.load(str(p1))
        d2 = svc.load(str(p2))

        result = ConcatenateService().concatenate([d1, d2], merge_mode="Union")
        assert result is not None
        assert result.row_count == 3
        assert "c" in result.dataframe.columns

    def test_intersection_concatenation(self, tmp_path):
        from portakal_app.data.services.concatenate_service import ConcatenateService

        svc = FileImportService()
        p1 = tmp_path / "d3.csv"
        p1.write_text("a,b\n1,2\n", encoding="utf-8")
        p2 = tmp_path / "d4.csv"
        p2.write_text("a,c\n3,4\n", encoding="utf-8")

        d1 = svc.load(str(p1))
        d2 = svc.load(str(p2))

        result = ConcatenateService().concatenate([d1, d2], merge_mode="Intersection")
        assert result is not None
        assert result.row_count == 2
        assert result.column_count == 1  # only "a" is common

    def test_source_column(self, tmp_path):
        from portakal_app.data.services.concatenate_service import ConcatenateService

        svc = FileImportService()
        p1 = tmp_path / "d5.csv"
        p1.write_text("a\n1\n", encoding="utf-8")
        p2 = tmp_path / "d6.csv"
        p2.write_text("a\n2\n", encoding="utf-8")

        d1 = svc.load(str(p1))
        d2 = svc.load(str(p2))

        result = ConcatenateService().concatenate([d1, d2], add_source_column=True, source_column_name="Origin")
        assert "Origin" in result.dataframe.columns


# 11. Aggregate Columns
class TestAggregateColumnsService:
    def test_mean_aggregation(self, numeric_dataset):
        from portakal_app.data.services.aggregate_columns_service import AggregateColumnsService

        result = AggregateColumnsService().aggregate(
            numeric_dataset, columns=["x", "y", "z"], operation="Mean", output_name="avg"
        )
        assert "avg" in result.dataframe.columns
        assert result.row_count == numeric_dataset.row_count

    def test_sum_aggregation(self, numeric_dataset):
        from portakal_app.data.services.aggregate_columns_service import AggregateColumnsService

        result = AggregateColumnsService().aggregate(
            numeric_dataset, columns=["x", "y"], operation="Sum", output_name="total"
        )
        totals = result.dataframe["total"].to_list()
        assert totals[0] == 11.0  # 1 + 10


# 12. Group By
class TestGroupByService:
    def test_group_by_with_mean(self, basic_dataset):
        from portakal_app.data.services.group_by_service import GroupByService

        result = GroupByService().group_by(
            basic_dataset,
            group_columns=["city"],
            aggregations={"score": ["Mean"], "age": ["Min. value", "Max. value"]},
        )
        assert result.row_count == 3
        assert "score_mean" in result.dataframe.columns
        assert "age_min_value" in result.dataframe.columns


# 13. Pivot Table
class TestPivotTableService:
    def test_count_pivot(self, basic_dataset):
        from portakal_app.data.services.pivot_table_service import PivotTableService

        result = PivotTableService().pivot(
            basic_dataset,
            row_column="city",
            col_column="age",
            value_column=None,
            aggregation="Count",
        )
        assert result.row_count > 0


# 14. Preprocess
class TestPreprocessService:
    def test_normalize(self, numeric_dataset):
        from portakal_app.data.services.preprocess_service import PreprocessService

        result = PreprocessService().preprocess(numeric_dataset, steps=["Normalize (0-1)"])
        x_vals = result.dataframe["x"].to_list()
        assert min(x_vals) == pytest.approx(0.0)
        assert max(x_vals) == pytest.approx(1.0)

    def test_standardize(self, numeric_dataset):
        from portakal_app.data.services.preprocess_service import PreprocessService

        result = PreprocessService().preprocess(numeric_dataset, steps=["Standardize (mean=0, var=1)"])
        x_mean = result.dataframe["x"].mean()
        assert abs(x_mean) < 0.01


# 15. Impute
class TestImputeService:
    def test_average_imputation(self, missing_dataset):
        from portakal_app.data.services.impute_service import ImputeService

        result = ImputeService().impute(missing_dataset, default_method="Average/Most frequent")
        remaining_nulls = sum(col.null_count for col in result.domain.columns)
        assert remaining_nulls == 0

    def test_fixed_value_imputation(self, missing_dataset):
        from portakal_app.data.services.impute_service import ImputeService

        result = ImputeService().impute(missing_dataset, default_method="Fixed values", default_fixed_value="0", default_fixed_value_cat="0")
        a_vals = result.dataframe["a"].to_list()
        assert None not in a_vals

    def test_drop_rows(self, missing_dataset):
        from portakal_app.data.services.impute_service import ImputeService

        result = ImputeService().impute(missing_dataset, default_method="Remove instances with unknown values")
        assert result.row_count < missing_dataset.row_count


# 16. Continuize
class TestContinuizeService:
    def test_one_hot_encoding(self, basic_dataset):
        from portakal_app.data.services.continuize_service import ContinuizeService

        result = ContinuizeService().continuize(basic_dataset, discrete_preset="One-hot encoding")
        assert result.column_count > basic_dataset.column_count
        assert any("city=" in c for c in result.dataframe.columns)

    def test_ordinal_encoding(self, basic_dataset):
        from portakal_app.data.services.continuize_service import ContinuizeService

        result = ContinuizeService().continuize(basic_dataset, discrete_preset="Treat as ordinal")
        assert result.dataframe["city"].dtype.is_numeric()

    def test_normalize_continuous(self, numeric_dataset):
        from portakal_app.data.services.continuize_service import ContinuizeService

        result = ContinuizeService().continuize(
            numeric_dataset, continuous_preset="Normalize to interval [0, 1]"
        )
        x_vals = result.dataframe["x"].to_list()
        assert min(x_vals) == pytest.approx(0.0)
        assert max(x_vals) == pytest.approx(1.0)


# 17. Discretize
class TestDiscretizeService:
    def test_equal_width(self, numeric_dataset):
        from portakal_app.data.services.discretize_service import DiscretizeService

        result = DiscretizeService().discretize(
            numeric_dataset,
            default_method="Equal width",
            column_methods={"x": {"method": "Equal width", "n_bins": 3}}
        )
        assert result.dataframe["x"].dtype == pl.Utf8  # binned → string

        result = DiscretizeService().discretize(
            numeric_dataset,
            default_method="Keep numeric",
            column_methods={"x": {"method": "Remove"}}
        )
        assert "x" not in result.dataframe.columns
        assert "label" in result.dataframe.columns


# 18. Melt
class TestMeltService:
    def test_basic_melt(self, tmp_path):
        from portakal_app.data.services.melt_service import MeltService

        path = tmp_path / "melt.csv"
        path.write_text("id,math,science\n1,90,80\n2,70,95\n", encoding="utf-8")
        dataset = FileImportService().load(str(path))

        result = MeltService().melt(dataset, id_column="id", item_name="subject", value_name="grade")
        assert result.row_count == 4  # 2 rows x 2 value columns
        assert "subject" in result.dataframe.columns
        assert "grade" in result.dataframe.columns


# 19. Create Class
class TestCreateClassService:
    def test_pattern_matching(self, tmp_path):
        from portakal_app.data.services.create_class_service import CreateClassService

        path = tmp_path / "class.csv"
        path.write_text("email\nalice@gmail.com\nbob@yahoo.com\ncharlie@gmail.com\n", encoding="utf-8")
        dataset = FileImportService().load(str(path))

        result, error = CreateClassService().create_class(
            dataset,
            source_column="email",
            rules=[("Gmail", "gmail"), ("Yahoo", "yahoo")],
            class_name="provider",
        )
        assert "provider" in result.dataframe.columns
        providers = result.dataframe["provider"].to_list()
        assert providers[0] == "Gmail"
        assert providers[1] == "Yahoo"


# 20. Create Instance
class TestCreateInstanceService:
    def test_create_single_instance(self, tmp_path):
        from portakal_app.data.services.create_instance_service import CreateInstanceService

        path = tmp_path / "inst.csv"
        path.write_text("age,score,city\n25.0,80.0,Ankara\n30.0,90.0,Istanbul\n", encoding="utf-8")
        dataset = FileImportService().load(str(path))

        result = CreateInstanceService().create(
            data=dataset, values={"age": "28", "score": "75", "city": "Bursa"}, append_to_data=False
        )
        assert result.row_count == 1

    def test_append_to_data(self, tmp_path):
        from portakal_app.data.services.create_instance_service import CreateInstanceService

        path = tmp_path / "inst2.csv"
        path.write_text("age,score,city\n25.0,80.0,Ankara\n30.0,90.0,Istanbul\n", encoding="utf-8")
        dataset = FileImportService().load(str(path))

        result = CreateInstanceService().create(
            data=dataset, values={"age": "28", "score": "75", "city": "Bursa"}, append_to_data=True
        )
        assert result.row_count == dataset.row_count + 1

    def test_get_defaults(self, basic_dataset):
        from portakal_app.data.services.create_instance_service import CreateInstanceService

        defaults = CreateInstanceService().get_defaults(basic_dataset)
        assert "age" in defaults
        assert "city" in defaults
        assert defaults["city"] != ""


# 21. Select Columns
class TestSelectColumnsService:
    def test_select_roles(self, basic_dataset):
        from portakal_app.data.services.select_columns_service import SelectColumnsService

        result = SelectColumnsService().select(
            basic_dataset, features=["age", "score"], target=["city"], metas=[]
        )
        assert result.column_count == 3
        roles = {col.name: col.role for col in result.domain.columns}
        assert roles["age"] == "feature"
        assert roles["score"] == "feature"
        assert roles["city"] == "target"

    def test_exclude_column(self, basic_dataset):
        from portakal_app.data.services.select_columns_service import SelectColumnsService

        result = SelectColumnsService().select(
            basic_dataset, features=["age"], target=[], metas=[]
        )
        assert result.column_count == 1
        assert result.dataframe.columns == ["age"]


# 22. Select Rows
class TestSelectRowsService:
    def test_numeric_filter(self, basic_dataset):
        from portakal_app.data.services.select_rows_service import SelectRowsService

        matching, unmatched = SelectRowsService().filter_rows(
            basic_dataset, conditions=[("age", ">", "25")]
        )
        assert matching is not None
        assert matching.row_count == 2  # age 30, 35

    def test_string_filter(self, basic_dataset):
        from portakal_app.data.services.select_rows_service import SelectRowsService

        matching, unmatched = SelectRowsService().filter_rows(
            basic_dataset, conditions=[("city", "equals", "Ankara")]
        )
        assert matching is not None
        assert matching.row_count == 2

    def test_no_conditions_returns_all(self, basic_dataset):
        from portakal_app.data.services.select_rows_service import SelectRowsService

        matching, unmatched = SelectRowsService().filter_rows(basic_dataset, conditions=[])
        assert matching.row_count == basic_dataset.row_count
        assert unmatched is None


# 23. Formula
class TestFormulaService:
    def test_arithmetic_expression(self, numeric_dataset):
        from portakal_app.data.services.formula_service import FormulaService

        result = FormulaService().apply_formulas(
            numeric_dataset, formulas=[("xy_sum", "col('x') + col('y')")]
        )
        assert "xy_sum" in result.dataframe.columns
        sums = result.dataframe["xy_sum"].to_list()
        assert sums[0] == 11.0  # 1 + 10

    def test_multiple_formulas(self, numeric_dataset):
        from portakal_app.data.services.formula_service import FormulaService

        result = FormulaService().apply_formulas(
            numeric_dataset,
            formulas=[
                ("double_x", "col('x') * 2"),
                ("z_div_10", "col('z') / 10"),
            ],
        )
        assert "double_x" in result.dataframe.columns
        assert "z_div_10" in result.dataframe.columns
        assert result.dataframe["double_x"][0] == 2.0
        assert result.dataframe["z_div_10"][0] == 10.0

    def test_invalid_expression_returns_null(self, numeric_dataset):
        from portakal_app.data.services.formula_service import FormulaService

        result = FormulaService().apply_formulas(
            numeric_dataset, formulas=[("bad", "this_is_not_valid!!!")]
        )
        assert "bad" in result.dataframe.columns


# 24. Python Script
class TestPythonScriptService:
    def test_basic_script(self, numeric_dataset):
        from portakal_app.data.services.python_script_service import PythonScriptService

        result = PythonScriptService().execute(
            numeric_dataset, code="out_data = in_data.filter(pl.col('x') > 2)"
        )
        assert result.error == ""
        assert result.output_dataset is not None
        assert result.output_dataset.row_count == 3  # x=3,4,5

    def test_script_with_print(self, numeric_dataset):
        from portakal_app.data.services.python_script_service import PythonScriptService

        result = PythonScriptService().execute(
            numeric_dataset, code='print("hello")\nout_data = in_data'
        )
        assert "hello" in result.stdout
        assert result.output_dataset is not None

    def test_script_error(self, numeric_dataset):
        from portakal_app.data.services.python_script_service import PythonScriptService

        result = PythonScriptService().execute(numeric_dataset, code="1/0")
        assert "ZeroDivisionError" in result.error

    def test_empty_code(self, numeric_dataset):
        from portakal_app.data.services.python_script_service import PythonScriptService

        result = PythonScriptService().execute(numeric_dataset, code="")
        assert result.error == "No code provided."

    def test_none_input(self):
        from portakal_app.data.services.python_script_service import PythonScriptService

        result = PythonScriptService().execute(
            None, code="out_data = pl.DataFrame({'a': [1, 2, 3]})"
        )
        assert result.error == ""
        assert result.output_dataset is not None
        assert result.output_dataset.row_count == 3

    def test_wrong_output_type(self, numeric_dataset):
        from portakal_app.data.services.python_script_service import PythonScriptService

        result = PythonScriptService().execute(
            numeric_dataset, code="out_data = 'not a dataframe'"
        )
        assert "must be a polars or orange DataFrame" in result.error
