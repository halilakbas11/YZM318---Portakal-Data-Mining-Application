from __future__ import annotations

import os
from dataclasses import dataclass, field, replace
from typing import Any, Callable

from portakal_app.data.models import AnalysisSuggestion, ColumnProfile, DatasetHandle


ScreenFactory = Callable[[], object]

LLM_PROVIDER_OPTIONS = ("OpenAI", "Gemini", "Claude", "Qwen", "Ollama")
LLM_PROVIDER_DEFAULT_BASE_URLS = {
    "OpenAI": "https://api.openai.com/v1",
    "Gemini": "https://generativelanguage.googleapis.com/v1beta",
    "Claude": "https://api.anthropic.com",
    "Qwen": "https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
    "Ollama": "http://localhost:11434",
}
LLM_PROVIDER_MODEL_PLACEHOLDERS = {
    "OpenAI": "gpt-4.1-mini",
    "Gemini": "gemini-2.5-flash",
    "Claude": "claude-3-5-sonnet-latest",
    "Qwen": "qwen3.5-flash",
    "Ollama": "llama3.1",
}
LLM_PROVIDER_ENV_VARS = {
    "OpenAI": ("OPENAI_API_KEY",),
    "Gemini": ("GEMINI_API_KEY", "GOOGLE_API_KEY"),
    "Claude": ("ANTHROPIC_API_KEY",),
    "Qwen": ("DASHSCOPE_API_KEY",),
    "Ollama": (),
}
WORKFLOW_PORT_COMPATIBILITY_OVERRIDES = {
    ("Scores", "Data"): frozenset({"data-table", "save-data"}),
    ("Data", "Extra Data"): frozenset({"merge-data"}),
    ("Data", "Template Data"): frozenset({"apply-domain"}),
}


@dataclass(frozen=True)
class AppState:
    selected_category: str = "data"
    selected_widget: str = "file"
    current_dataset: DatasetHandle | None = None
    current_dataset_id: str | None = None
    current_dataset_path: str | None = None
    workflow_title: str = "Untitled"
    workflow_description: str = ""
    status_message: str = "Ready"

    def with_updates(self, **changes: Any) -> "AppState":
        return replace(self, **changes)


@dataclass(frozen=True)
class LLMSessionConfig:
    provider: str = "OpenAI"
    model: str = ""
    base_url: str = LLM_PROVIDER_DEFAULT_BASE_URLS["OpenAI"]
    api_key: str = ""

    def with_updates(self, **changes: Any) -> "LLMSessionConfig":
        return replace(self, **changes)

    def default_base_url(self) -> str:
        return LLM_PROVIDER_DEFAULT_BASE_URLS.get(self.provider, "")

    def model_placeholder(self) -> str:
        return LLM_PROVIDER_MODEL_PLACEHOLDERS.get(self.provider, "")

    def env_var_names(self) -> tuple[str, ...]:
        return LLM_PROVIDER_ENV_VARS.get(self.provider, ())

    def env_key_name(self) -> str | None:
        for name in self.env_var_names():
            if os.getenv(name):
                return name
        return None

    def env_key_available(self) -> bool:
        return self.env_key_name() is not None

    def resolved_api_key(self) -> str | None:
        if self.provider == "Ollama":
            return None
        if self.api_key.strip():
            return self.api_key.strip()
        env_name = self.env_key_name()
        if env_name is None:
            return None
        value = os.getenv(env_name)
        return value.strip() if value else None


@dataclass(frozen=True)
class CategoryDefinition:
    id: str
    label: str
    enabled: bool = True


@dataclass(frozen=True)
class WidgetDefinition:
    id: str
    category_id: str
    label: str
    enabled: bool
    screen_factory: ScreenFactory
    description: str = ""
    icon_name: str = "file"
    input_ports: tuple["PortDefinition", ...] = ()
    output_ports: tuple["PortDefinition", ...] = ()
    output_channels: tuple[str, ...] = ()
    input_channels: tuple[str, ...] = ()
    multi_input_channels: tuple[str, ...] = ()  # channels that accept multiple connections


@dataclass(frozen=True)
class PortDefinition:
    id: str
    label: str


@dataclass(frozen=True)
class WorkflowPayload:
    port_label: str
    dataset: DatasetHandle


def workflow_ports_are_compatible(
    source_widget_id: str,
    source_label: str,
    target_widget_id: str,
    target_label: str,
) -> bool:
    _ = source_widget_id
    if source_label == target_label:
        return True
    allowed_targets = WORKFLOW_PORT_COMPATIBILITY_OVERRIDES.get((source_label, target_label))
    if allowed_targets is None:
        return False
    return target_widget_id in allowed_targets


@dataclass(frozen=True)
class MetricCardData:
    title: str
    value: str
    subtitle: str = ""


@dataclass(frozen=True)
class SuggestionItem:
    title: str
    body: str
    severity: str = "info"


@dataclass(frozen=True)
class DataInfoViewModel:
    summary_cards: list[MetricCardData] = field(default_factory=list)
    column_profiles: list[ColumnProfile] = field(default_factory=list)
    risks: list[AnalysisSuggestion] = field(default_factory=list)
    suggestions: list[AnalysisSuggestion] = field(default_factory=list)
    llm_status: str = "Not analyzed yet"
    llm_error: str = ""
    is_analyzing: bool = False
