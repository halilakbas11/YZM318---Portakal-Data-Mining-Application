from __future__ import annotations

from collections import Counter, defaultdict
from math import sqrt

import polars as pl

from portakal_app.data.models import DatasetHandle, RankedFeature
from portakal_app.data.services.generated_dataset_service import GeneratedDatasetService


class FeatureRankingService:
    def __init__(self) -> None:
        self._generated_dataset_service = GeneratedDatasetService()

    def rank(
        self,
        dataset: DatasetHandle,
        target_name: str | None = None,
        feature_filter: str = "all",
        top_n: int | None = None,
    ) -> list[RankedFeature]:
        if target_name == "":
            active_target = None
        elif target_name is None:
            active_target = self._default_target_name(dataset)
        else:
            active_target = target_name
        rows: list[RankedFeature] = []
        for column in dataset.domain.columns:
            if column.role != "feature":
                continue
            if feature_filter == "numeric" and column.logical_type != "numeric":
                continue
            if feature_filter == "categorical" and column.logical_type == "numeric":
                continue
            score, method, details = self._score_feature(dataset, column.name, column.logical_type, active_target)
            rows.append(
                RankedFeature(
                    feature_name=column.name,
                    logical_type=column.logical_type,
                    score=score,
                    method=method,
                    details=details,
                    target_name=active_target,
                )
            )
        rows.sort(key=lambda item: item.score, reverse=True)
        if top_n is not None and top_n > 0:
            return rows[:top_n]
        return rows

    def build_scores_dataset(self, source_dataset: DatasetHandle, ranked_rows: list[RankedFeature]) -> DatasetHandle:
        dataframe = pl.DataFrame(
            {
                "feature": [row.feature_name for row in ranked_rows],
                "type": [row.logical_type for row in ranked_rows],
                "score": [row.score for row in ranked_rows],
                "method": [row.method for row in ranked_rows],
                "details": [row.details for row in ranked_rows],
            }
        )
        return self._generated_dataset_service.build_dataset(
            dataframe,
            dataset_id=f"{source_dataset.dataset_id}-scores",
            display_name=f"{source_dataset.display_name} Scores",
            file_name=f"{source_dataset.dataset_id}-scores.csv",
            annotations={
                "generated_by": "rank",
                "source_dataset_id": source_dataset.dataset_id,
            },
        )

    def _default_target_name(self, dataset: DatasetHandle) -> str | None:
        target = next((column.name for column in dataset.domain.columns if column.role == "target"), None)
        return target

    def _score_feature(
        self,
        dataset: DatasetHandle,
        feature_name: str,
        feature_type: str,
        target_name: str | None,
    ) -> tuple[float, str, str]:
        series = dataset.dataframe.get_column(feature_name)
        completeness = 1.0 - (series.null_count() / max(1, dataset.row_count))
        unique_ratio = min(1.0, series.n_unique() / max(1, dataset.row_count))

        if target_name is None:
            score = (completeness * 0.65) + (unique_ratio * 0.35)
            return score, "Heuristic", f"Completeness {completeness:.2f}, uniqueness {unique_ratio:.2f}"

        target_series = dataset.dataframe.get_column(target_name)
        target_type = next(column.logical_type for column in dataset.domain.columns if column.name == target_name)

        if feature_type == "numeric" and target_type == "numeric":
            corr = self._pearson_abs(series.to_list(), target_series.to_list())
            return corr, "Abs Pearson", f"Auto selected because feature and target are numeric."

        if feature_type == "numeric":
            eta = self._eta_squared_numeric_target(series.to_list(), target_series.to_list())
            return eta, "Eta squared", f"Auto selected because feature is numeric and target is categorical."

        if target_type == "numeric":
            eta = self._eta_squared_categorical_feature(series.to_list(), target_series.to_list())
            return eta, "Eta squared", f"Auto selected because feature is categorical/text and target is numeric."

        cramers_v = self._cramers_v(series.to_list(), target_series.to_list())
        return cramers_v, "Cramer's V", f"Auto selected because both feature and target are categorical/text."

    def _pearson_abs(self, feature_values: list[object], target_values: list[object]) -> float:
        paired = []
        for feature, target in zip(feature_values, target_values):
            try:
                if feature is None or target is None:
                    continue
                paired.append((float(feature), float(target)))
            except (TypeError, ValueError):
                continue
        if len(paired) < 2:
            return 0.0
        xs = [item[0] for item in paired]
        ys = [item[1] for item in paired]
        mean_x = sum(xs) / len(xs)
        mean_y = sum(ys) / len(ys)
        numerator = sum((x - mean_x) * (y - mean_y) for x, y in paired)
        denom_x = sqrt(sum((x - mean_x) ** 2 for x in xs))
        denom_y = sqrt(sum((y - mean_y) ** 2 for y in ys))
        if denom_x == 0.0 or denom_y == 0.0:
            return 0.0
        return abs(numerator / (denom_x * denom_y))

    def _eta_squared_numeric_target(self, feature_values: list[object], target_values: list[object]) -> float:
        groups: dict[str, list[float]] = defaultdict(list)
        for feature, target in zip(feature_values, target_values):
            if feature is None or target is None:
                continue
            try:
                groups[str(target)].append(float(feature))
            except (TypeError, ValueError):
                continue
        all_values = [value for values in groups.values() for value in values]
        if len(groups) < 2 or len(all_values) < 2:
            return 0.0
        grand_mean = sum(all_values) / len(all_values)
        ss_between = sum(len(values) * ((sum(values) / len(values)) - grand_mean) ** 2 for values in groups.values())
        ss_total = sum((value - grand_mean) ** 2 for value in all_values)
        if ss_total == 0:
            return 0.0
        return max(0.0, min(1.0, ss_between / ss_total))

    def _eta_squared_categorical_feature(self, feature_values: list[object], target_values: list[object]) -> float:
        groups: dict[str, list[float]] = defaultdict(list)
        for feature, target in zip(feature_values, target_values):
            if feature is None or target is None:
                continue
            try:
                groups[str(feature)].append(float(target))
            except (TypeError, ValueError):
                continue
        all_values = [value for values in groups.values() for value in values]
        if len(groups) < 2 or len(all_values) < 2:
            return 0.0
        grand_mean = sum(all_values) / len(all_values)
        ss_between = sum(len(values) * ((sum(values) / len(values)) - grand_mean) ** 2 for values in groups.values())
        ss_total = sum((value - grand_mean) ** 2 for value in all_values)
        if ss_total == 0:
            return 0.0
        return max(0.0, min(1.0, ss_between / ss_total))

    def _cramers_v(self, feature_values: list[object], target_values: list[object]) -> float:
        counts: dict[str, Counter[str]] = defaultdict(Counter)
        valid_rows = 0
        for feature, target in zip(feature_values, target_values):
            if feature is None or target is None:
                continue
            counts[str(feature)][str(target)] += 1
            valid_rows += 1
        if valid_rows == 0:
            return 0.0
        row_keys = list(counts.keys())
        col_keys = sorted({column for row in counts.values() for column in row.keys()})
        if len(row_keys) < 2 or len(col_keys) < 2:
            return 0.0
        row_totals = {row_key: sum(counts[row_key].values()) for row_key in row_keys}
        col_totals = Counter[str]()
        for row_key in row_keys:
            col_totals.update(counts[row_key])
        chi2 = 0.0
        for row_key in row_keys:
            for col_key in col_keys:
                observed = counts[row_key][col_key]
                expected = (row_totals[row_key] * col_totals[col_key]) / valid_rows
                if expected > 0:
                    chi2 += ((observed - expected) ** 2) / expected
        phi2 = chi2 / valid_rows
        min_dim = min(len(row_keys) - 1, len(col_keys) - 1)
        if min_dim <= 0:
            return 0.0
        return max(0.0, min(1.0, sqrt(phi2 / min_dim)))
