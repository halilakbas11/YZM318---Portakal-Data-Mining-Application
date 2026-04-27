from __future__ import annotations

import os
from dataclasses import dataclass, field, replace
from typing import Any, Callable

from portakal_app.data.models import AnalysisSuggestion, ColumnProfile, DatasetHandle


# Factory type for creating UI screens dynamically
ScreenFactory = Callable[[], object]

# Supported LLM providers
LLM_PROVIDER_OPTIONS = ("OpenAI", "Gemini", "Claude", "Qwen", "Ollama")

# Default API base URLs for each provider
LLM_PROVIDER_DEFAULT_BASE_URLS = {
    "OpenAI": "https://api.openai.com/v1",
    "Gemini": "https://generativelanguage.googleapis.com/v1beta",
    "Claude": "https://api.anthropic.com",
    "Qwen": "https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
    "Ollama": "http://localhost:11434",
}

# Placeholder model names shown in UI before user selection
LLM_PROVIDER_MODEL_PLACEHOLDERS = {
    "OpenAI": "gpt-4.1-mini",
    "Gemini": "gemini-2.5-flash",
    "Claude": "claude-3-5-sonnet-latest",
    "Qwen": "qwen3.5-flash",
    "Ollama": "llama3.1",
}

# Environment variables used to fetch API keys for each provider
LLM_PROVIDER_ENV_VARS = {
    "OpenAI": ("OPENAI_API_KEY",),
    "Gemini": ("GEMINI_API_KEY", "GOOGLE_API_KEY"),
    "Claude": ("ANTHROPIC_API_KEY",),
    "Qwen": ("DASHSCOPE_API_KEY",),
    "Ollama": (),
}

# Special overrides defining which workflow ports can connect even if labels differ
WORKFLOW_PORT_COMPATIBILITY_OVERRIDES = {
    ("Scores", "Data"): frozenset({"data-table", "save-data"}),
    ("Features", "Data"): frozenset({"data-table", "save-data"}),
    ("Data", "Extra Data"): frozenset({"merge-data"}),
    ("Data", "Template Data"): frozenset({"apply-domain"}),
    ("Model", "Tree"): frozenset({"tree-viewer"}),
    ("Components", "Projection"): frozenset({"linear-projection"}),
    # Any trained artifact can flow into save-model or confusion-matrix
    ("Tree", "Model"): frozenset({"save-model", "stacking", "confusion-matrix"}),
    ("Classifier", "Model"): frozenset({"save-model", "stacking", "confusion-matrix"}),
    ("Random Forest", "Model"): frozenset({"save-model", "stacking", "confusion-matrix"}),
    # Viewers accept SklearnModelArtifact as "Classifier" when output is "Model"
    ("Model", "Classifier"): frozenset({"cn2-rule-viewer", "nomogram", "scoring-sheet-viewer", "calibrated-learner"}),
    # Stacking aggregate input accepts any model
    ("Tree", "Aggregate"): frozenset({"stacking"}),
    ("Classifier", "Aggregate"): frozenset({"stacking"}),
    ("Random Forest", "Aggregate"): frozenset({"stacking"}),
}


@dataclass(frozen=True)
class AppState:
    # Currently selected category in UI
    selected_category: str = "data"
    # Currently selected widget in UI
    selected_widget: str = "file"
    # Active dataset object
    current_dataset: DatasetHandle | None = None
    # Dataset identifier
    current_dataset_id: str | None = None
    # File path of dataset
    current_dataset_path: str | None = None
    # Workflow title shown in UI
    workflow_title: str = "Untitled"
    # Workflow description
    workflow_description: str = ""
    # Status message displayed to the user
    status_message: str = "Ready"

    # Returns a new immutable AppState with updated fields
    def with_updates(self, **changes: Any) -> "AppState":
        return replace(self, **changes)


@dataclass(frozen=True)
class LLMSessionConfig:
    # Selected LLM provider
    provider: str = "OpenAI"
    # Selected model name
    model: str = ""
    # API base URL
    base_url: str = LLM_PROVIDER_DEFAULT_BASE_URLS["OpenAI"]
    # API key (can be empty if using environment variable)
    api_key: str = ""

    # Returns updated config immutably
    def with_updates(self, **changes: Any) -> "LLMSessionConfig":
        return replace(self, **changes)

    # Returns default base URL for current provider
    def default_base_url(self) -> str:
        return LLM_PROVIDER_DEFAULT_BASE_URLS.get(self.provider, "")

    # Returns placeholder model name for UI display
    def model_placeholder(self) -> str:
        return LLM_PROVIDER_MODEL_PLACEHOLDERS.get(self.provider, "")

    # Returns possible environment variable names for API key
    def env_var_names(self) -> tuple[str, ...]:
        return LLM_PROVIDER_ENV_VARS.get(self.provider, ())

    # Returns the first available environment variable name that has a value
    def env_key_name(self) -> str | None:
        for name in self.env_var_names():
            if os.getenv(name):
                return name
        return None

    # Checks if an API key exists in environment variables
    def env_key_available(self) -> bool:
        return self.env_key_name() is not None

    # Resolves the API key from direct input or environment variables
    def resolved_api_key(self) -> str | None:
        if self.provider == "Ollama":
            return None  # Ollama does not require API key
        if self.api_key.strip():
            return self.api_key.strip()
        env_name = self.env_key_name()
        if env_name is None:
            return None
        value = os.getenv(env_name)
        return value.strip() if value else None


@dataclass(frozen=True)
class CategoryDefinition:
    # Unique category identifier
    id: str
    # Display label
    label: str
    # Whether category is enabled in UI
    enabled: bool = True


@dataclass(frozen=True)
class WidgetDefinition:
    # Unique widget identifier
    id: str
    # Category this widget belongs to
    category_id: str
    # Display label
    label: str
    # Whether widget is enabled
    enabled: bool
    # Factory function to create widget screen
    screen_factory: ScreenFactory
    # Optional description
    description: str = ""
    # Icon name used in UI
    icon_name: str = "file"
    # Input port definitions
    input_ports: tuple["PortDefinition", ...] = ()
    # Output port definitions
    output_ports: tuple["PortDefinition", ...] = ()
    # Output channel types
    output_channels: tuple[str, ...] = ()
    # Input channel types
    input_channels: tuple[str, ...] = ()
    # Channels that accept multiple inputs
    multi_input_channels: tuple[str, ...] = ()


@dataclass(frozen=True)
class PortDefinition:
    # Port identifier
    id: str
    # Display label
    label: str


@dataclass(frozen=True)
class WorkflowPayload:
    # Label of the port carrying this payload
    port_label: str
    # Actual data passed between widgets
    value: object

    # Returns dataset if payload contains one
    @property
    def dataset(self) -> DatasetHandle | None:
        return self.value if isinstance(self.value, DatasetHandle) else None

    # Returns a new payload with updated port label
    def with_port_label(self, port_label: str) -> "WorkflowPayload":
        return WorkflowPayload(port_label, self.value)


# Checks if two ports are compatible for connection
def workflow_ports_are_compatible(
    source_widget_id: str,
    source_label: str,
    target_widget_id: str,
    target_label: str,
) -> bool:
    _ = source_widget_id
    # Direct label match means compatible
    if source_label == target_label:
        return True
    # Check override rules
    allowed_targets = WORKFLOW_PORT_COMPATIBILITY_OVERRIDES.get((source_label, target_label))
    if allowed_targets is None:
        return False
    return target_widget_id in allowed_targets


@dataclass(frozen=True)
class MetricCardData:
    # Title of metric
    title: str
    # Main value displayed
    value: str
    # Optional subtitle
    subtitle: str = ""


@dataclass(frozen=True)
class SuggestionItem:
    # Suggestion title
    title: str
    # Detailed message
    body: str
    # Severity level (info, warning, error)
    severity: str = "info"


@dataclass(frozen=True)
class DataInfoViewModel:
    # Summary metric cards
    summary_cards: list[MetricCardData] = field(default_factory=list)
    # Column-level statistics
    column_profiles: list[ColumnProfile] = field(default_factory=list)
    # Detected risks in data
    risks: list[AnalysisSuggestion] = field(default_factory=list)
    # Improvement suggestions
    suggestions: list[AnalysisSuggestion] = field(default_factory=list)
    # LLM analysis status message
    llm_status: str = "Not analyzed yet"
    # Error message if LLM fails
    llm_error: str = ""
    # Indicates if analysis is currently running
    is_analyzing: bool = False