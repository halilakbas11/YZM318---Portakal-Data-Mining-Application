from __future__ import annotations

from portakal_app.data.models import DatasetHandle, DatasetSummary


class LLMContextBuilder:
    MAX_COLUMNS = 20

    def build(self, dataset: DatasetHandle, summary: DatasetSummary) -> str:
        lines = [
            f"Dataset: {dataset.display_name}",
            f"Source: {dataset.source.path.name}",
            f"Rows: {summary.row_count}",
            f"Columns: {summary.column_count}",
            f"Missing ratio: {summary.missing_ratio * 100:.2f}%",
            f"Missing cells: {summary.missing_value_count}",
            f"Duplicate rows: {summary.duplicate_row_count}",
            f"Feature columns: {summary.feature_count}",
            f"Target columns: {summary.target_count}",
            f"Type distribution: {self._format_dtype_counts(summary)}",
            "Column profiles:",
        ]

        profiles = summary.column_profiles[: self.MAX_COLUMNS]
        for profile in profiles:
            sample_values = ", ".join(profile.sample_values) if profile.sample_values else "-"
            lines.append(
                f"- {profile.column_name}: type={profile.logical_type}, role={profile.role}, "
                f"nulls={profile.null_count}, null_ratio={profile.null_ratio:.3f}, "
                f"unique_hint={profile.unique_count_hint}, samples={sample_values}"
            )

        remaining_columns = len(summary.column_profiles) - len(profiles)
        if remaining_columns > 0:
            lines.append(f"- truncated: {remaining_columns} additional columns omitted")

        lines.extend(
            [
                "",
                "Return strict JSON only.",
                'Schema: {"risks":[{"title","body","severity"}], "suggestions":[{"title","body","severity"}]}',
                'Use severity values only from: "low", "medium", "high".',
                "Keep each item concise and actionable.",
            ]
        )
        return "\n".join(lines)

    def _format_dtype_counts(self, summary: DatasetSummary) -> str:
        if not summary.dtype_counts:
            return "none"
        ordered = sorted(summary.dtype_counts.items(), key=lambda item: (-item[1], item[0]))
        return ", ".join(f"{name}={count}" for name, count in ordered)
