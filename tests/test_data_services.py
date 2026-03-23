from __future__ import annotations

import httpx
from pathlib import Path

import polars as pl
import pytest

from portakal_app.data.errors import (
    DatasetLoadError,
    LLMConfigurationError,
    LLMRequestError,
    LLMResponseError,
    UnsupportedFormatError,
)
from portakal_app.data.models import CSVImportOptions, DomainColumnEdit, DomainEditRequest
from portakal_app.data.services.column_statistics_service import ColumnStatisticsService
from portakal_app.data.services.color_settings_service import ColorSettingsService
from portakal_app.data.services.data_info_service import DataInfoService
from portakal_app.data.services.dataset_catalog_service import DatasetCatalogService
from portakal_app.data.services.domain_transform_service import DomainTransformService
from portakal_app.data.services.feature_ranking_service import FeatureRankingService
from portakal_app.data.services.file_import_service import FileImportService
from portakal_app.data.services.llm_analyzer import LLMAnalyzer
from portakal_app.data.services.llm_context_builder import LLMContextBuilder
from portakal_app.data.services.paint_data_service import PaintDataService
from portakal_app.data.services.profiling_service import ProfilingService
from portakal_app.data.services.save_data_service import SaveDataService
from portakal_app.models import LLMSessionConfig


@pytest.fixture()
def sample_dataframe() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "amount": [10.5, 20.0, 13.25],
            "event_time": ["2026-03-18 10:00", "2026-03-19 11:30", "2026-03-20 09:15"],
            "species": ["setosa", "versicolor", "setosa"],
        }
    )


def test_file_import_service_loads_csv_tsv_tab_xlsx_and_parquet(tmp_path, sample_dataframe):
    service = FileImportService()
    csv_path = tmp_path / "sample.csv"
    tsv_path = tmp_path / "sample.tsv"
    tab_path = tmp_path / "sample.tab"
    xlsx_path = tmp_path / "sample.xlsx"
    parquet_path = tmp_path / "sample.parquet"

    sample_dataframe.write_csv(csv_path)
    sample_dataframe.write_csv(tsv_path, separator="\t")
    sample_dataframe.write_csv(tab_path, separator="\t")
    sample_dataframe.write_excel(xlsx_path)
    sample_dataframe.write_parquet(parquet_path)

    for path in (csv_path, tsv_path, tab_path, xlsx_path, parquet_path):
        dataset = service.load(str(path))
        assert dataset.row_count == 3
        assert dataset.column_count == 3
        assert dataset.cache_path.exists()
        assert dataset.source.path == path


def test_file_import_service_raises_typed_error_for_missing_path(tmp_path):
    service = FileImportService()
    missing_path = tmp_path / "missing.csv"

    with pytest.raises(DatasetLoadError):
        service.load(str(missing_path))


def test_file_import_service_raises_typed_error_for_unsupported_format(tmp_path):
    service = FileImportService()
    unsupported_path = tmp_path / "sample.json"
    unsupported_path.write_text('{"a": 1}', encoding="utf-8")

    with pytest.raises(UnsupportedFormatError):
        service.load(str(unsupported_path))


def test_file_import_service_builds_expected_domain(sample_dataframe, tmp_path):
    service = FileImportService()
    csv_path = tmp_path / "domain.csv"
    sample_dataframe.write_csv(csv_path)

    dataset = service.load(str(csv_path))
    columns = {column.name: column for column in dataset.domain.columns}

    assert columns["amount"].logical_type == "numeric"
    assert columns["amount"].role == "feature"
    assert columns["event_time"].logical_type == "datetime"
    assert columns["species"].logical_type == "categorical"
    assert columns["species"].role == "target"


def test_file_import_service_loads_delimited_text_with_custom_options(tmp_path):
    service = FileImportService()
    csv_path = tmp_path / "custom.txt"
    csv_path.write_text("1;2;yes\n3;4;no\n", encoding="utf-8")

    dataset = service.load_delimited_text(str(csv_path), CSVImportOptions(delimiter=";", has_header=False))

    assert dataset.row_count == 2
    assert dataset.column_count == 3
    assert dataset.dataframe.columns == ["Column 1", "Column 2", "Column 3"]


def test_file_import_service_auto_detects_delimiter(tmp_path):
    service = FileImportService()
    csv_path = tmp_path / "auto.txt"
    csv_path.write_text("city;value;label\nAnkara;1;A\nIzmir;2;B\n", encoding="utf-8")

    options = service.resolve_delimited_options(str(csv_path), CSVImportOptions(auto_detect_delimiter=True))

    assert options.delimiter == ";"


def test_file_import_service_user_override_beats_auto_detect(tmp_path):
    service = FileImportService()
    csv_path = tmp_path / "override.txt"
    csv_path.write_text("city,value\nAnkara,1\nIzmir,2\n", encoding="utf-8")

    options = service.resolve_delimited_options(
        str(csv_path),
        CSVImportOptions(delimiter=";", auto_detect_delimiter=False),
    )

    assert options.delimiter == ";"


def test_file_import_service_auto_encoding_fallback_reads_cp1254(tmp_path):
    service = FileImportService()
    csv_path = tmp_path / "turkish.csv"
    csv_path.write_bytes("şehir;değer\nİzmir;10\nAnkara;20\n".encode("cp1254"))

    dataset = service.load_delimited_text(
        str(csv_path),
        CSVImportOptions(auto_detect_delimiter=True, encoding="auto"),
    )

    assert dataset.dataframe.columns == ["şehir", "değer"]
    assert dataset.dataframe.get_column("şehir").to_list()[0] == "İzmir"


def test_file_import_service_skip_rows_applies_before_header(tmp_path):
    service = FileImportService()
    csv_path = tmp_path / "skip.csv"
    csv_path.write_text("# comment\ncity,value\nAnkara,1\nIzmir,2\n", encoding="utf-8")

    dataset = service.load_delimited_text(
        str(csv_path),
        CSVImportOptions(skip_rows=1, delimiter=",", has_header=True),
    )

    assert dataset.dataframe.columns == ["city", "value"]
    assert dataset.row_count == 2


def test_save_data_service_exports_csv_xlsx_and_parquet_round_trip(tmp_path, sample_dataframe):
    import_service = FileImportService()
    save_service = SaveDataService()
    source_path = tmp_path / "source.csv"
    sample_dataframe.write_csv(source_path)
    dataset = import_service.load(str(source_path))

    for extension in ("csv", "xlsx", "parquet"):
        target_path = tmp_path / f"export.{extension}"
        save_service.save(dataset, str(target_path))
        reloaded = import_service.load(str(target_path))
        assert reloaded.row_count == dataset.row_count
        assert reloaded.column_count == dataset.column_count


def test_save_data_service_raises_typed_error_for_unsupported_export(tmp_path, sample_dataframe):
    import_service = FileImportService()
    save_service = SaveDataService()
    source_path = tmp_path / "source.csv"
    sample_dataframe.write_csv(source_path)
    dataset = import_service.load(str(source_path))

    with pytest.raises(UnsupportedFormatError):
        save_service.save(dataset, str(tmp_path / "export.tsv"))


def test_profiling_service_builds_summary_and_column_profiles(tmp_path, sample_dataframe):
    source_path = tmp_path / "source.csv"
    sample_dataframe.write_csv(source_path)
    dataset = FileImportService().load(str(source_path))

    summary = ProfilingService().summarize(dataset)

    assert summary.row_count == 3
    assert summary.column_count == 3
    assert summary.target_count == 1
    assert summary.dtype_counts["numeric"] == 1
    assert summary.dtype_counts["datetime"] == 1
    assert summary.dtype_counts["categorical"] == 1
    assert summary.column_profiles[0].role == "feature"
    assert summary.column_profiles[2].role == "target"
    assert summary.column_profiles[2].sample_values == ("setosa", "versicolor")


def test_data_info_service_builds_deterministic_view_model(tmp_path, sample_dataframe):
    source_path = tmp_path / "source.csv"
    sample_dataframe.write_csv(source_path)
    dataset = FileImportService().load(str(source_path))

    view_model = DataInfoService().build(dataset)

    assert len(view_model.summary_cards) == 4
    assert view_model.summary_cards[0].title == "Rows"
    assert len(view_model.column_profiles) == 3
    assert view_model.llm_status == "Not analyzed yet"
    assert view_model.llm_error == ""
    assert view_model.risks == []
    assert view_model.suggestions == []


def test_domain_transform_service_applies_skip_and_single_target(tmp_path):
    source_path = tmp_path / "domain.csv"
    pl.DataFrame({"a": [1, 2], "b": ["x", "y"], "c": ["m", "n"]}).write_csv(source_path)
    dataset = FileImportService().load(str(source_path))

    updated = DomainTransformService().apply(
        dataset,
        DomainEditRequest(
            columns=(
                DomainColumnEdit("a", "amount", "numeric", "feature"),
                DomainColumnEdit("b", "label", "text", "target"),
                DomainColumnEdit("c", "meta_text", "text", "skip"),
            )
        ),
    )

    assert updated.dataframe.columns == ["amount", "label"]
    roles = {column.name: column.role for column in updated.domain.columns}
    assert roles == {"amount": "feature", "label": "target"}


def test_domain_transform_service_rejects_lossy_conversion(tmp_path):
    source_path = tmp_path / "lossy.csv"
    pl.DataFrame({"city": ["Ankara", "Izmir"]}).write_csv(source_path)
    dataset = FileImportService().load(str(source_path))

    with pytest.raises(ValueError, match="could not be safely converted"):
        DomainTransformService().apply(
            dataset,
            DomainEditRequest(columns=(DomainColumnEdit("city", "city", "numeric", "feature"),)),
        )


def test_column_statistics_service_reports_numeric_outliers(tmp_path):
    source_path = tmp_path / "numeric-stats.csv"
    pl.DataFrame({"value": [1, 2, 3, 4, 100], "target": ["A", "A", "A", "B", "B"]}).write_csv(source_path)
    dataset = FileImportService().load(str(source_path))

    result = ColumnStatisticsService().describe(dataset, "value")
    metrics = dict(result.metrics)

    assert metrics["Outliers"] == "1"
    assert len(result.histogram_bins) >= 1


def test_column_statistics_service_flags_high_cardinality_and_near_constant(tmp_path):
    source_path = tmp_path / "categorical-stats.csv"
    pl.DataFrame({"city": ["A"] * 19 + ["B"], "id_text": [f"id-{index}" for index in range(20)]}).write_csv(source_path)
    dataset = FileImportService().load(str(source_path))
    service = ColumnStatisticsService()

    near_constant = service.describe(dataset, "city")
    high_card = service.describe(dataset, "id_text")

    assert "near constant" in near_constant.warning_tags
    assert "high cardinality" in high_card.warning_tags


def test_feature_ranking_service_supports_top_n_filter_and_heuristic(tmp_path):
    source_path = tmp_path / "rank.csv"
    pl.DataFrame(
        {
            "signal": [0, 0, 1, 1],
            "noise": [5, 1, 3, 4],
            "city": ["A", "A", "B", "B"],
            "target": ["A", "A", "B", "B"],
        }
    ).write_csv(source_path)
    dataset = FileImportService().load(str(source_path))
    service = FeatureRankingService()

    filtered = service.rank(dataset, target_name="target", feature_filter="numeric", top_n=1)
    heuristic = service.rank(dataset, target_name="", feature_filter="all", top_n=10)

    assert len(filtered) == 1
    assert filtered[0].feature_name == "signal"
    assert heuristic[0].method == "Heuristic"


def test_dataset_catalog_service_exposes_curated_downloadable_entries():
    service = DatasetCatalogService()

    entries = service.available_datasets()

    assert len(entries) == 15
    assert entries[0].download_url.startswith("https://")
    assert "All" in service.available_domains()


def test_paint_data_service_builds_normalized_snapshot_and_dataset(tmp_path):
    source_path = tmp_path / "paint-source.csv"
    pl.DataFrame({"x_raw": [10, 20], "y_raw": [5, 15], "label": ["A", "B"]}).write_csv(source_path)
    dataset = FileImportService().load(str(source_path))
    service = PaintDataService()

    snapshot = service.build_snapshot(dataset)
    painted = service.build_dataset(snapshot)

    assert snapshot.x_name == "x_raw"
    assert snapshot.y_name == "y_raw"
    assert len(snapshot.points) == 2
    assert painted.row_count == 2
    assert painted.column_count == 3
    assert painted.annotations["generated_by"] == "paint-data"


def test_color_settings_service_builds_and_applies_state(tmp_path):
    source_path = tmp_path / "color-source.csv"
    pl.DataFrame({"amount": [1, 2, 3], "category": ["A", "B", "A"]}).write_csv(source_path)
    dataset = FileImportService().load(str(source_path))
    service = ColorSettingsService()

    state = service.build_state(dataset)
    colored = service.apply(dataset, state)

    assert "category" in state["discrete"]
    assert "amount" in state["numeric"]
    assert colored.annotations["color_settings"]["discrete"]["category"]["A"]


class _FakeResponse:
    def __init__(self, status_code: int, payload):
        self.status_code = status_code
        self._payload = payload

    def json(self):
        if isinstance(self._payload, Exception):
            raise self._payload
        return self._payload


def test_llm_context_builder_limits_column_output(tmp_path, sample_dataframe):
    wide_dataframe = pl.DataFrame({f"col_{index}": [index, index + 1] for index in range(24)})
    source_path = tmp_path / "wide.csv"
    wide_dataframe.write_csv(source_path)
    dataset = FileImportService().load(str(source_path))
    summary = ProfilingService().summarize(dataset)

    context = LLMContextBuilder().build(dataset, summary)

    assert "Dataset: Wide" in context
    assert "truncated: 4 additional columns omitted" in context


def test_llm_analyzer_builds_openai_request_and_parses_response(monkeypatch):
    captured = {}

    def fake_post(url, *, headers, params=None, json, timeout):
        captured["url"] = url
        captured["headers"] = headers
        captured["params"] = params
        captured["json"] = json
        captured["timeout"] = timeout
        return _FakeResponse(
            200,
            {
                "choices": [
                    {
                        "message": {
                            "content": '{"risks":[{"title":"Leakage","body":"Target-like columns detected.","severity":"high"}],'
                            '"suggestions":[{"title":"Profile missing","body":"Check null-heavy fields.","severity":"medium"}]}'
                        }
                    }
                ]
            },
        )

    monkeypatch.setattr("portakal_app.data.services.llm_analyzer.httpx.post", fake_post)

    result = LLMAnalyzer().analyze(
        summary=None,  # type: ignore[arg-type]
        context="Dataset summary",
        config=LLMSessionConfig(
            provider="OpenAI",
            model="gpt-test",
            base_url="https://api.openai.com/v1",
            api_key="sk-test",
        ),
    )

    assert captured["url"] == "https://api.openai.com/v1/chat/completions"
    assert captured["headers"]["Authorization"] == "Bearer sk-test"
    assert captured["json"]["model"] == "gpt-test"
    assert captured["timeout"] == 90.0
    assert result[0].kind == "risk"
    assert result[1].kind == "suggestion"


@pytest.mark.parametrize(
    ("provider", "base_url", "api_key", "expected_url", "payload"),
    [
        (
            "Qwen",
            "https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
            "dash-key",
            "https://dashscope-intl.aliyuncs.com/compatible-mode/v1/chat/completions",
            {"choices": [{"message": {"content": '{"risks":[],"suggestions":[]}'}}]},
        ),
        (
            "Claude",
            "https://api.anthropic.com",
            "anth-key",
            "https://api.anthropic.com/v1/messages",
            {"content": [{"type": "text", "text": '{"risks":[],"suggestions":[]}' }]},
        ),
        (
            "Gemini",
            "https://generativelanguage.googleapis.com/v1beta",
            "gem-key",
            "https://generativelanguage.googleapis.com/v1beta/models/gemini-test:generateContent",
            {"candidates": [{"content": {"parts": [{"text": '{"risks":[],"suggestions":[]}' }]}}]},
        ),
        (
            "Ollama",
            "http://localhost:11434",
            "",
            "http://localhost:11434/api/chat",
            {"message": {"content": '{"risks":[],"suggestions":[]}' }},
        ),
    ],
)
def test_llm_analyzer_supports_other_providers(monkeypatch, provider, base_url, api_key, expected_url, payload):
    captured = {}

    def fake_post(url, *, headers, params=None, json, timeout):
        captured["url"] = url
        captured["headers"] = headers
        captured["params"] = params
        captured["json"] = json
        captured["timeout"] = timeout
        return _FakeResponse(200, payload)

    monkeypatch.setattr("portakal_app.data.services.llm_analyzer.httpx.post", fake_post)

    config = LLMSessionConfig(provider=provider, model="gemini-test" if provider == "Gemini" else "model-test", base_url=base_url, api_key=api_key)
    if provider == "Ollama":
        config = config.with_updates(model="llama3.1")

    result = LLMAnalyzer().analyze(summary=None, context="Dataset summary", config=config)  # type: ignore[arg-type]

    assert captured["url"] == expected_url
    assert captured["timeout"] == 90.0
    if provider == "Gemini":
        assert captured["params"] == {"key": "gem-key"}
    elif provider == "Ollama":
        assert "Authorization" not in captured["headers"]
    else:
        assert captured["headers"]
    assert result == []


def test_llm_analyzer_requires_key_for_remote_providers():
    with pytest.raises(LLMConfigurationError):
        LLMAnalyzer().analyze(
            summary=None,  # type: ignore[arg-type]
            context="Dataset summary",
            config=LLMSessionConfig(provider="OpenAI", model="gpt-test", base_url="https://api.openai.com/v1", api_key=""),
        )


def test_llm_analyzer_surfaces_http_and_parse_errors(monkeypatch):
    def timeout_post(*args, **kwargs):
        raise httpx.TimeoutException("timeout")

    monkeypatch.setattr("portakal_app.data.services.llm_analyzer.httpx.post", timeout_post)
    with pytest.raises(LLMRequestError):
        LLMAnalyzer().analyze(
            summary=None,  # type: ignore[arg-type]
            context="Dataset summary",
            config=LLMSessionConfig(provider="OpenAI", model="gpt-test", base_url="https://api.openai.com/v1", api_key="sk"),
        )

    def invalid_json_post(url, *, headers, params=None, json, timeout):
        return _FakeResponse(200, {"choices": [{"message": {"content": "not-json"}}]})

    monkeypatch.setattr("portakal_app.data.services.llm_analyzer.httpx.post", invalid_json_post)
    with pytest.raises(LLMResponseError):
        LLMAnalyzer().analyze(
            summary=None,  # type: ignore[arg-type]
            context="Dataset summary",
            config=LLMSessionConfig(provider="OpenAI", model="gpt-test", base_url="https://api.openai.com/v1", api_key="sk"),
        )

    def failing_post(url, *, headers, params=None, json, timeout):
        return _FakeResponse(401, {"error": {"message": "Bad key"}})

    monkeypatch.setattr("portakal_app.data.services.llm_analyzer.httpx.post", failing_post)
    with pytest.raises(LLMRequestError, match="Bad key"):
        LLMAnalyzer().analyze(
            summary=None,  # type: ignore[arg-type]
            context="Dataset summary",
            config=LLMSessionConfig(provider="OpenAI", model="gpt-test", base_url="https://api.openai.com/v1", api_key="sk"),
        )
