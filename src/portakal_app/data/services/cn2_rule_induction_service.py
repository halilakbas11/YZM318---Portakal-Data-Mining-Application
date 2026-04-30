from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from uuid import uuid4

import numpy as np

from portakal_app.data.models import ColumnSchema, DatasetHandle
from portakal_app.rule_artifacts import CN2RuleArtifact, CN2RuleClassifierArtifact, RuleConditionArtifact

_MAX_RULES = 500  # internal safety limit


@dataclass(frozen=True)
class CN2InductionSettings:
    rule_ordering: str = "ordered"          # "ordered" | "unordered"
    covering_algorithm: str = "exclusive"   # "exclusive" | "weighted"
    weighted_gamma: float = 0.70
    evaluation_measure: str = "entropy"     # "entropy" | "laplace" | "wracc"
    beam_width: int = 5
    min_covered_examples: int = 1
    max_rule_length: int = 5
    statistical_significance: bool = False
    statistical_alpha: float = 1.0
    relative_significance: bool = False
    relative_alpha: float = 1.0
    restrict_categorical_to_equality: bool = False


class CN2RuleInductionService:
    def induce(self, dataset: DatasetHandle, settings: CN2InductionSettings) -> CN2RuleClassifierArtifact:
        target = self._target_column(dataset)
        if target is None:
            raise ValueError("CN2 Rule Induction requires a categorical target column.")

        target_values_raw = dataset.dataframe.get_column(target.name).to_list()
        target_values: list[str | None] = [str(v) if v is not None else None for v in target_values_raw]
        valid_indices = [i for i, v in enumerate(target_values) if v is not None]

        if not valid_indices:
            raise ValueError("Target column does not contain any labeled rows.")

        class_values = tuple(str(v) for v in dict.fromkeys(v for v in target_values if v is not None))
        if len(class_values) < 2:
            raise ValueError("CN2 Rule Induction requires at least two target classes.")

        feature_columns = [c for c in dataset.domain.feature_columns if self._supported_feature(c)]
        if not feature_columns:
            raise ValueError("Input data does not have supported feature columns for rule induction.")

        all_selectors = []
        for col in feature_columns:
            all_selectors.extend(
                self._candidate_selectors(dataset, col, settings.restrict_categorical_to_equality)
            )

        # Precompute each selector's covered set (indices where condition is True and target is labeled)
        sel_masks: dict[int, frozenset[int]] = {}
        for sel in all_selectors:
            mask = sel.evaluate_dataset(dataset)
            sel_masks[id(sel)] = frozenset(
                i for i in range(len(target_values))
                if i < len(mask) and mask[i] and target_values[i] is not None
            )

        total_counter = Counter(target_values[i] for i in valid_indices)
        remaining: set[int] = set(valid_indices)
        weights: dict[int, float] = {i: 1.0 for i in valid_indices}
        rules: list[CN2RuleArtifact] = []

        for _ in range(_MAX_RULES):
            current_valid = [i for i in remaining if target_values[i] is not None]
            if len(current_valid) < settings.min_covered_examples:
                break

            current_counter = Counter(target_values[i] for i in current_valid)

            best_rule = self._beam_search(
                current_valid, all_selectors, sel_masks, settings,
                class_values, target_values, target.name, current_counter, len(current_valid),
            )

            if best_rule is None:
                break

            rules.append(best_rule)
            covered_set = set(best_rule.learning_covered_indices)

            if settings.covering_algorithm == "exclusive":
                remaining -= covered_set
            else:
                for i in covered_set:
                    weights[i] *= settings.weighted_gamma
                remaining = {i for i in remaining if weights.get(i, 0.0) > 0.01}

        # Default (catch-all) rule
        majority = max(class_values, key=lambda v: (total_counter.get(v, 0), v))
        dist = tuple(float(total_counter.get(v, 0)) for v in class_values)
        total = float(sum(dist)) or 1.0
        rules.append(CN2RuleArtifact(
            target_name=target.name,
            prediction=majority,
            selectors=(),
            curr_class_dist=dist,
            probabilities=tuple(d / total for d in dist),
            quality=0.0,
            covered_count=len(valid_indices),
            learning_covered_indices=tuple(valid_indices),
            is_default=True,
        ))

        return CN2RuleClassifierArtifact(
            classifier_id=f"cn2-{uuid4().hex[:8]}",
            display_name=f"CN2 Rules ({dataset.display_name})",
            instances=dataset,
            original_feature_names=tuple(c.name for c in dataset.domain.feature_columns),
            target_name=target.name,
            class_values=class_values,
            rule_list=rules,
            params={
                "evaluation_measure": settings.evaluation_measure,
                "beam_width": settings.beam_width,
                "min_covered_examples": settings.min_covered_examples,
                "max_rule_length": settings.max_rule_length,
                "rule_ordering": settings.rule_ordering,
                "covering_algorithm": settings.covering_algorithm,
            },
        )

    # ── Beam search ───────────────────────────────────────────────────────────

    def _beam_search(
        self,
        valid_indices: list[int],
        all_selectors: list[RuleConditionArtifact],
        sel_masks: dict[int, frozenset[int]],
        settings: CN2InductionSettings,
        class_values: tuple[str, ...],
        target_values: list[str | None],
        target_name: str,
        current_counter: Counter,
        total_current: int,
    ) -> CN2RuleArtifact | None:
        valid_set = frozenset(valid_indices)
        min_cov = settings.min_covered_examples
        best_rule: CN2RuleArtifact | None = None
        best_quality = -math.inf

        # Build initial 1-condition beam
        beam: list[tuple[tuple[RuleConditionArtifact, ...], frozenset[int], float]] = []
        for sel in all_selectors:
            covered = sel_masks[id(sel)] & valid_set
            if len(covered) < min_cov:
                continue
            q = self._quality(covered, class_values, target_values, current_counter, total_current, settings.evaluation_measure)
            if settings.statistical_significance and not self._lrs_significant(
                covered, target_values, class_values, current_counter, total_current, settings.statistical_alpha
            ):
                continue
            beam.append(((sel,), covered, q))
            if q > best_quality:
                best_quality = q
                best_rule = self._make_rule(
                    (sel,), list(covered), class_values, target_values, target_name, q
                )

        if not beam:
            return None

        beam.sort(key=lambda x: -x[2])
        beam = beam[: settings.beam_width]

        # Extend conditions up to max_rule_length
        for _depth in range(2, settings.max_rule_length + 1):
            next_beam: list[tuple[tuple[RuleConditionArtifact, ...], frozenset[int], float]] = []
            seen_sigs: set[tuple] = set()

            for existing_conds, existing_covered, _ in beam:
                existing_ids = {id(s) for s in existing_conds}
                for sel in all_selectors:
                    if id(sel) in existing_ids:
                        continue
                    new_covered = existing_covered & sel_masks[id(sel)]
                    if len(new_covered) < min_cov:
                        continue

                    new_conds = existing_conds + (sel,)
                    sig = tuple(sorted((s.attribute, s.operator, str(s.value)) for s in new_conds))
                    if sig in seen_sigs:
                        continue
                    seen_sigs.add(sig)

                    q = self._quality(new_covered, class_values, target_values, current_counter, total_current, settings.evaluation_measure)

                    if settings.statistical_significance and not self._lrs_significant(
                        new_covered, target_values, class_values, current_counter, total_current, settings.statistical_alpha
                    ):
                        continue
                    if settings.relative_significance and not self._relative_lrs_significant(
                        new_covered, existing_covered, target_values, class_values, settings.relative_alpha
                    ):
                        continue

                    next_beam.append((new_conds, new_covered, q))
                    if q > best_quality:
                        best_quality = q
                        best_rule = self._make_rule(
                            new_conds, list(new_covered), class_values, target_values, target_name, q
                        )

            if not next_beam:
                break
            next_beam.sort(key=lambda x: -x[2])
            beam = next_beam[: settings.beam_width]

        return best_rule

    # ── Quality metrics ───────────────────────────────────────────────────────

    def _quality(
        self,
        covered: frozenset[int],
        class_values: tuple[str, ...],
        target_values: list[str | None],
        current_counter: Counter,
        total_current: int,
        measure: str,
    ) -> float:
        classes = [target_values[i] for i in covered if target_values[i] is not None]
        if not classes:
            return -math.inf
        counter = Counter(classes)
        n = len(classes)
        prediction = max(class_values, key=lambda v: counter.get(v, 0))
        n_pred = counter.get(prediction, 0)

        if measure == "laplace":
            return (n_pred + 1) / (n + len(class_values))
        if measure == "entropy":
            entropy = -sum(
                (counter.get(v, 0) / n) * math.log2(counter.get(v, 0) / n)
                for v in class_values if counter.get(v, 0) > 0
            )
            return -entropy  # lower entropy = higher quality
        # wracc (default)
        prior = current_counter.get(prediction, 0) / max(1, total_current)
        return (n / max(1, total_current)) * (n_pred / n - prior)

    # ── Statistical significance ──────────────────────────────────────────────

    def _lrs_significant(
        self,
        covered: frozenset[int],
        target_values: list[str | None],
        class_values: tuple[str, ...],
        current_counter: Counter,
        total_current: int,
        alpha: float,
    ) -> bool:
        if alpha >= 1.0:
            return True
        classes = [target_values[i] for i in covered if target_values[i] is not None]
        n = len(classes)
        if n == 0:
            return False
        counter = Counter(classes)
        lrs = 0.0
        for cls in class_values:
            n_c = counter.get(cls, 0)
            expected = n * current_counter.get(cls, 0) / max(1, total_current)
            if n_c > 0 and expected > 0:
                lrs += 2 * n_c * math.log(n_c / expected)
        df = len(class_values) - 1
        return df > 0 and lrs > self._chi2_ppf(1.0 - alpha, df)

    def _relative_lrs_significant(
        self,
        child: frozenset[int],
        parent: frozenset[int],
        target_values: list[str | None],
        class_values: tuple[str, ...],
        alpha: float,
    ) -> bool:
        if alpha >= 1.0:
            return True
        child_classes = [target_values[i] for i in child if target_values[i] is not None]
        parent_classes = [target_values[i] for i in parent if target_values[i] is not None]
        n_child = len(child_classes)
        n_parent = len(parent_classes)
        if n_child == 0 or n_parent == 0:
            return False
        child_counter = Counter(child_classes)
        parent_counter = Counter(parent_classes)
        lrs = 0.0
        for cls in class_values:
            n_c = child_counter.get(cls, 0)
            expected = n_child * parent_counter.get(cls, 0) / n_parent
            if n_c > 0 and expected > 0:
                lrs += 2 * n_c * math.log(n_c / expected)
        df = len(class_values) - 1
        return df > 0 and lrs > self._chi2_ppf(1.0 - alpha, df)

    def _chi2_ppf(self, p: float, df: int) -> float:
        try:
            from scipy.stats import chi2
            return float(chi2.ppf(p, df))
        except Exception:
            return 2.0 * df  # conservative fallback

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _make_rule(
        self,
        conds: tuple[RuleConditionArtifact, ...],
        covered_indices: list[int],
        class_values: tuple[str, ...],
        target_values: list[str | None],
        target_name: str,
        quality: float,
    ) -> CN2RuleArtifact:
        classes = [target_values[i] for i in covered_indices if target_values[i] is not None]
        counter = Counter(classes)
        n = len(classes)
        prediction = max(class_values, key=lambda v: counter.get(v, 0))
        dist = tuple(float(counter.get(v, 0)) for v in class_values)
        total = float(sum(dist)) or 1.0
        return CN2RuleArtifact(
            target_name=target_name,
            prediction=prediction,
            selectors=conds,
            curr_class_dist=dist,
            probabilities=tuple(d / total for d in dist),
            quality=quality,
            covered_count=n,
            learning_covered_indices=tuple(covered_indices),
            is_default=False,
        )

    def _target_column(self, dataset: DatasetHandle) -> ColumnSchema | None:
        for column in dataset.domain.target_columns:
            if column.logical_type in {"categorical", "boolean"}:
                return column
        return None

    def _supported_feature(self, column: ColumnSchema) -> bool:
        return column.logical_type in {"categorical", "boolean", "numeric"}

    def _candidate_selectors(
        self,
        dataset: DatasetHandle,
        column: ColumnSchema,
        restrict_categorical_to_equality: bool,
    ) -> list[RuleConditionArtifact]:
        if column.name not in dataset.dataframe.columns:
            return []
        series = dataset.dataframe.get_column(column.name)

        if column.logical_type == "numeric":
            raw: list[float] = []
            for v in series.drop_nulls().to_list():
                try:
                    raw.append(float(v))
                except (TypeError, ValueError):
                    continue
            finite = [v for v in raw if np.isfinite(v)]
            if len(set(finite)) < 2:
                return []
            thresholds = {float(np.quantile(finite, q)) for q in (0.25, 0.5, 0.75)}
            return [
                RuleConditionArtifact(column.name, op, round(t, 6))
                for t in sorted(thresholds)
                for op in ("<=", ">")
            ]

        values: list = []
        for v in series.drop_nulls().unique(maintain_order=True).to_list():
            values.append(v)
            if len(values) >= 8:
                break

        selectors = [RuleConditionArtifact(column.name, "==", v) for v in values]
        if not restrict_categorical_to_equality:
            selectors += [RuleConditionArtifact(column.name, "!=", v) for v in values]
        return selectors
