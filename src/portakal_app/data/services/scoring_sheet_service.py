from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Any

import numpy as np
import polars as pl

from portakal_app.data.models import DatasetHandle
from portakal_app.scoring_sheet_artifacts import ScoringSheetClassifierArtifact, ScoringSheetRuleArtifact


def _safe_unique_values(series: pl.Series) -> list[Any]:
    try:
        values = series.drop_nulls().unique(maintain_order=True).to_list()
    except TypeError:
        values = series.drop_nulls().unique().to_list()
    return list(values)


@dataclass(frozen=True)
class ScoringSheetSettings:
    max_rules: int = 5
    max_points_per_rule: int = 5


@dataclass(frozen=True)
class _RuleCandidate:
    display_name: str
    source_feature: str
    operator: str
    reference_value: Any
    effect: float
    matches: np.ndarray


class ScoringSheetService:
    def fit(
        self,
        dataset: DatasetHandle,
        settings: ScoringSheetSettings | None = None,
    ) -> ScoringSheetClassifierArtifact:
        config = settings or ScoringSheetSettings()
        target_columns = dataset.domain.target_columns
        if len(target_columns) != 1:
            raise ValueError("Scoring Sheet requires exactly one target column.")
        target_column = target_columns[0]
        if target_column.logical_type not in {"categorical", "boolean"}:
            raise ValueError("Scoring Sheet currently supports only binary categorical targets.")

        dataframe = dataset.dataframe
        target_series = dataframe.get_column(target_column.name)
        class_values = [str(value) for value in _safe_unique_values(target_series)]
        if len(class_values) != 2:
            raise ValueError("Scoring Sheet currently supports exactly two target classes.")

        positive_name = class_values[1]
        y = np.asarray(
            [1.0 if str(value) == positive_name else 0.0 for value in target_series.to_list()],
            dtype=float,
        )
        candidates = self._build_candidates(dataset, y)
        if not candidates:
            raise ValueError("No supported rules could be generated for Scoring Sheet.")

        selected = sorted(candidates, key=lambda candidate: abs(candidate.effect), reverse=True)[: config.max_rules]
        max_abs = max(abs(candidate.effect) for candidate in selected) or 1.0

        rules: list[ScoringSheetRuleArtifact] = []
        score_matrix_parts: list[np.ndarray] = []
        for candidate in selected:
            scaled = int(round(candidate.effect / max_abs * config.max_points_per_rule))
            if scaled == 0:
                scaled = 1 if candidate.effect >= 0 else -1
            rules.append(
                ScoringSheetRuleArtifact(
                    display_name=candidate.display_name,
                    source_feature=candidate.source_feature,
                    operator=candidate.operator,
                    reference_value=candidate.reference_value,
                    points=scaled,
                )
            )
            score_matrix_parts.append(candidate.matches.astype(float)[:, None] * scaled)

        score_matrix = np.concatenate(score_matrix_parts, axis=1) if score_matrix_parts else np.zeros((dataset.row_count, 0), dtype=float)
        scores = np.sum(score_matrix, axis=1) if score_matrix.size else np.zeros(dataset.row_count, dtype=float)
        bias, weight = self._fit_probability_mapping(scores, y)
        all_scores = self._enumerate_scores(tuple(rule.points for rule in rules), scores)
        all_risks = [self._probability_for_score(bias, weight, score) * 100.0 for score in all_scores]

        return ScoringSheetClassifierArtifact(
            classifier_id=f"{dataset.dataset_id}-scoring-sheet",
            display_name="Scoring Sheet",
            instances=dataset,
            target_name=target_column.name,
            class_values=(class_values[0], class_values[1]),
            rules=tuple(rules),
            logistic_bias=bias,
            logistic_weight=weight,
            base_scores=tuple(float(score) for score in all_scores),
            base_risks=tuple(float(risk) for risk in all_risks),
            params={
                "max_rules": config.max_rules,
                "max_points_per_rule": config.max_points_per_rule,
            },
        )

    def _build_candidates(self, dataset: DatasetHandle, y: np.ndarray) -> list[_RuleCandidate]:
        candidates: list[_RuleCandidate] = []
        total_pos = float(np.sum(y))
        total_neg = float(len(y) - total_pos)
        baseline_odds = (total_pos + 0.5) / (total_neg + 0.5)

        for column in dataset.domain.feature_columns:
            series = dataset.dataframe.get_column(column.name)
            if column.logical_type in {"categorical", "boolean"}:
                for value in _safe_unique_values(series):
                    matches = np.asarray([cell == value for cell in series.to_list()], dtype=bool)
                    candidate = self._candidate_from_mask(
                        column.name,
                        "==",
                        value,
                        f"{column.name} = {value}",
                        matches,
                        y,
                        baseline_odds,
                    )
                    if candidate is not None:
                        candidates.append(candidate)
            elif column.logical_type == "numeric":
                values = np.asarray(
                    [float(value) if value is not None else np.nan for value in series.to_list()],
                    dtype=float,
                )
                finite = values[np.isfinite(values)]
                if finite.size < 2:
                    continue
                thresholds = self._numeric_thresholds(finite)
                for threshold in thresholds:
                    matches = np.asarray(np.isfinite(values) & (values <= threshold), dtype=bool)
                    candidate = self._candidate_from_mask(
                        column.name,
                        "<=",
                        float(threshold),
                        f"{column.name} <= {threshold:.2f}",
                        matches,
                        y,
                        baseline_odds,
                    )
                    if candidate is not None:
                        candidates.append(candidate)
        return candidates

    def _candidate_from_mask(
        self,
        source_feature: str,
        operator: str,
        reference_value: Any,
        display_name: str,
        matches: np.ndarray,
        y: np.ndarray,
        baseline_odds: float,
    ) -> _RuleCandidate | None:
        coverage = int(np.sum(matches))
        if coverage < 2 or coverage >= len(y):
            return None
        match_pos = float(np.sum(y[matches]))
        match_neg = float(coverage - match_pos)
        odds = (match_pos + 0.5) / (match_neg + 0.5)
        effect = float(np.log(odds / baseline_odds) * np.sqrt(coverage / len(y)))
        if abs(effect) < 1e-9:
            return None
        return _RuleCandidate(
            display_name=display_name,
            source_feature=source_feature,
            operator=operator,
            reference_value=reference_value,
            effect=effect,
            matches=matches,
        )

    def _numeric_thresholds(self, values: np.ndarray) -> list[float]:
        quantiles = np.unique(np.quantile(values, [0.25, 0.5, 0.75], method="midpoint"))
        return [float(value) for value in quantiles]

    def _fit_probability_mapping(self, scores: np.ndarray, y: np.ndarray) -> tuple[float, float]:
        x = np.column_stack((np.ones(len(scores), dtype=float), scores.astype(float)))
        weights = np.zeros(2, dtype=float)
        ridge = np.diag([0.0, 1e-3])
        for _ in range(100):
            logits = np.clip(x @ weights, -40.0, 40.0)
            probabilities = 1.0 / (1.0 + np.exp(-logits))
            diagonal = np.clip(probabilities * (1.0 - probabilities), 1e-6, None)
            hessian = x.T @ (x * diagonal[:, None]) + ridge
            gradient = x.T @ (probabilities - y) + ridge @ weights
            try:
                step = np.linalg.solve(hessian, gradient)
            except np.linalg.LinAlgError:
                step = np.linalg.pinv(hessian) @ gradient
            weights -= step
            if float(np.max(np.abs(step))) <= 1e-6:
                break
        return float(weights[0]), float(weights[1])

    def _enumerate_scores(self, coefficients: tuple[int, ...], training_scores: np.ndarray) -> list[float]:
        if len(coefficients) <= 10:
            all_scores = sorted({float(sum(choice)) for choice in product(*[(0, coef) for coef in coefficients])})
            if 0.0 not in all_scores:
                all_scores.append(0.0)
                all_scores.sort()
            return all_scores

        unique_scores = np.unique(training_scores.astype(float))
        quantile_len = min(20, len(unique_scores))
        quantiles = np.asarray(range(1, 1 + quantile_len), dtype=float) / quantile_len
        selected = np.quantile(unique_scores, quantiles, method="closest_observation")
        selected = sorted({float(value) for value in selected} | {0.0})
        return selected

    def _probability_for_score(self, bias: float, weight: float, score: float) -> float:
        value = float(np.clip(bias + weight * score, -40.0, 40.0))
        return float(1.0 / (1.0 + np.exp(-value)))
