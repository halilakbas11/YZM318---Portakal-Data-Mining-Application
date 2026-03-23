from __future__ import annotations

from portakal_app.data.models import DatasetHandle, DatasetSummary
from portakal_app.data.services.profiling_service import ProfilingService
from portakal_app.models import DataInfoViewModel, MetricCardData


class DataInfoService:
    def __init__(self, profiling_service: ProfilingService | None = None) -> None:
        self._profiling_service = profiling_service or ProfilingService()

    def summarize(self, dataset: DatasetHandle) -> DatasetSummary:
        return self._profiling_service.summarize(dataset)

    def build(self, dataset: DatasetHandle) -> DataInfoViewModel:
        return self.build_from_summary(self.summarize(dataset))

    def build_from_summary(self, summary: DatasetSummary) -> DataInfoViewModel:
        return DataInfoViewModel(
            summary_cards=[
                MetricCardData("Rows", f"{summary.row_count:,}", f"Duplicates: {summary.duplicate_row_count:,}"),
                MetricCardData(
                    "Columns",
                    f"{summary.column_count:,}",
                    f"Features {summary.feature_count:,} | Targets {summary.target_count:,}",
                ),
                MetricCardData(
                    "Missing",
                    f"{summary.missing_ratio * 100:.1f}%",
                    f"{summary.missing_value_count:,} missing cells",
                ),
                MetricCardData("Types", self._format_dtype_counts(summary), "Logical type distribution"),
            ],
            column_profiles=list(summary.column_profiles),
            risks=[],
            suggestions=[],
            llm_status="Not analyzed yet",
            llm_error="",
            is_analyzing=False,
        )

    def _format_dtype_counts(self, summary: DatasetSummary) -> str:
        if not summary.dtype_counts:
            return "n/a"
        ordered = sorted(summary.dtype_counts.items(), key=lambda item: (-item[1], item[0]))
        return ", ".join(f"{name}:{count}" for name, count in ordered[:3])
