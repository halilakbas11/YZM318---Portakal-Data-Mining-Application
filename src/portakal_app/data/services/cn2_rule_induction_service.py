from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from uuid import uuid4

import numpy as np

from portakal_app.data.models import ColumnSchema, DatasetHandle
from portakal_app.rule_artifacts import CN2RuleArtifact, CN2RuleClassifierArtifact, RuleConditionArtifact


@dataclass(frozen=True)
class CN2InductionSettings:
    max_rules: int = 8
    min_covered_examples: int = 3


class CN2RuleInductionService:
    def induce(self, dataset: DatasetHandle, settings: CN2InductionSettings) -> CN2RuleClassifierArtifact:
        target = self._target_column(dataset)
        if target is None:
            raise ValueError("CN2 Rule Induction requires a categorical target column.")

        target_values = dataset.dataframe.get_column(target.name).to_list()
        valid_pairs = [(index, value) for index, value in enumerate(target_values) if value is not None]
        if not valid_pairs:
            raise ValueError("Target column does not contain any labeled rows.")

        class_values = tuple(str(value) for value in dict.fromkeys(value for _index, value in valid_pairs))
        if len(class_values) < 2:
            raise ValueError("CN2 Rule Induction requires at least two target classes.")

        feature_columns = [column for column in dataset.domain.feature_columns if self._supported_feature(column)]
        if not feature_columns:
            raise ValueError("Input data does not have supported feature columns for rule induction.")

        total_valid = len(valid_pairs)
        target_counter = Counter(str(value) for _index, value in valid_pairs)
        candidates: list[CN2RuleArtifact] = []
        seen_signatures: set[tuple[str, str, str]] = set()

        for column in feature_columns:
            for selector in self._candidate_selectors(dataset, column):
                rule = self._evaluate_selector(
                    dataset,
                    selector,
                    target_name=target.name,
                    class_values=class_values,
                    target_counter=target_counter,
                    total_valid=total_valid,
                    min_covered_examples=max(1, settings.min_covered_examples),
                )
                if rule is None:
                    continue
                signature = (selector.attribute, selector.operator, str(selector.value))
                if signature in seen_signatures:
                    continue
                seen_signatures.add(signature)
                candidates.append(rule)

        candidates.sort(
            key=lambda rule: (
                -rule.quality,
                -rule.covered_count,
                rule.length,
                rule.conditions_text(compact_view=True),
            )
        )

        selected_rules = candidates[: max(1, settings.max_rules)]
        majority_class = max(class_values, key=lambda value: (target_counter[value], value))
        distribution = tuple(float(target_counter.get(value, 0)) for value in class_values)
        total = float(sum(distribution)) or 1.0
        default_rule = CN2RuleArtifact(
            target_name=target.name,
            prediction=majority_class,
            selectors=(),
            curr_class_dist=distribution,
            probabilities=tuple(value / total for value in distribution),
            quality=0.0,
            covered_count=total_valid,
            learning_covered_indices=tuple(index for index, _value in valid_pairs),
            is_default=True,
        )
        selected_rules.append(default_rule)

        return CN2RuleClassifierArtifact(
            classifier_id=f"cn2-{uuid4().hex[:8]}",
            display_name=f"CN2 Rules ({dataset.display_name})",
            instances=dataset,
            original_feature_names=tuple(column.name for column in dataset.domain.feature_columns),
            target_name=target.name,
            class_values=class_values,
            rule_list=selected_rules,
            params={
                "max_rules": settings.max_rules,
                "min_covered_examples": settings.min_covered_examples,
            },
        )

    def _target_column(self, dataset: DatasetHandle) -> ColumnSchema | None:
        for column in dataset.domain.target_columns:
            if column.logical_type in {"categorical", "boolean"}:
                return column
        return None

    def _supported_feature(self, column: ColumnSchema) -> bool:
        return column.logical_type in {"categorical", "boolean", "numeric"}

    def _candidate_selectors(self, dataset: DatasetHandle, column: ColumnSchema) -> list[RuleConditionArtifact]:
        if column.name not in dataset.dataframe.columns:
            return []
        series = dataset.dataframe.get_column(column.name)
        if column.logical_type == "numeric":
            raw_values: list[float] = []
            for value in series.drop_nulls().to_list():
                try:
                    raw_values.append(float(value))
                except (TypeError, ValueError):
                    continue
            finite = [value for value in raw_values if np.isfinite(value)]
            if len(set(finite)) < 2:
                return []
            thresholds = {
                float(np.quantile(finite, q))
                for q in (0.25, 0.5, 0.75)
            }
            return [
                RuleConditionArtifact(column.name, operator, round(threshold, 6))
                for threshold in sorted(thresholds)
                for operator in ("<=", ">")
            ]

        values = []
        for value in series.drop_nulls().unique(maintain_order=True).to_list():
            values.append(value)
            if len(values) >= 8:
                break
        return [RuleConditionArtifact(column.name, "==", value) for value in values]

    def _evaluate_selector(
        self,
        dataset: DatasetHandle,
        selector: RuleConditionArtifact,
        *,
        target_name: str,
        class_values: tuple[str, ...],
        target_counter: Counter[str],
        total_valid: int,
        min_covered_examples: int,
    ) -> CN2RuleArtifact | None:
        target_values = dataset.dataframe.get_column(target_name).to_list()
        selector_mask = selector.evaluate_dataset(dataset)
        covered_indices = [
            index for index, covered in enumerate(selector_mask)
            if covered and target_values[index] is not None
        ]
        if len(covered_indices) < min_covered_examples:
            return None

        covered_classes = [str(target_values[index]) for index in covered_indices]
        covered_counter = Counter(covered_classes)
        prediction = max(class_values, key=lambda value: (covered_counter.get(value, 0), value))
        distribution = tuple(float(covered_counter.get(value, 0)) for value in class_values)
        total_covered = float(sum(distribution))
        if total_covered <= 0:
            return None

        probabilities = tuple(value / total_covered for value in distribution)
        prediction_share = covered_counter.get(prediction, 0) / total_covered
        prior_share = target_counter.get(prediction, 0) / max(1, total_valid)
        wracc = (total_covered / max(1, total_valid)) * (prediction_share - prior_share)

        return CN2RuleArtifact(
            target_name=target_name,
            prediction=prediction,
            selectors=(selector,),
            curr_class_dist=distribution,
            probabilities=probabilities,
            quality=float(wracc),
            covered_count=int(total_covered),
            learning_covered_indices=tuple(covered_indices),
            is_default=False,
        )
