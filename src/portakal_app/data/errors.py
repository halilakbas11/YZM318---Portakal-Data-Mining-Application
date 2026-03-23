class PortakalDataError(Exception):
    """Base error for dataset loading and saving failures."""


class UnsupportedFormatError(PortakalDataError):
    """Raised when the requested file format is not supported."""


class DatasetLoadError(PortakalDataError):
    """Raised when a dataset cannot be loaded into a handle."""


class DatasetSaveError(PortakalDataError):
    """Raised when a dataset cannot be exported."""


class LLMConfigurationError(PortakalDataError):
    """Raised when the current LLM configuration is incomplete."""


class LLMRequestError(PortakalDataError):
    """Raised when an LLM request fails."""


class LLMResponseError(PortakalDataError):
    """Raised when an LLM response cannot be parsed."""
