from __future__ import annotations

import random
from dataclasses import replace

from portakal_app.data.models import DatasetHandle, build_data_domain


TIEBREAKERS = (
    "First instance",
    "Last instance",
    "Middle instance",
    "Random instance",
    "Discard non-unique",
)


class UniqueService:
    """Unique row filtering aligned with Orange Data Mining.

    Uses dictionary-based grouping (same algorithm as Orange's OWUnique)
    instead of Polars' `.unique()` to ensure identical duplicate detection.
    """

    def filter_unique(
        self,
        dataset: DatasetHandle,
        *,
        group_by_columns: list[str],
        tiebreaker: str = "First instance",
    ) -> DatasetHandle:
        df = dataset.dataframe

        # If no columns selected, use ALL columns (Orange default behaviour)
        key_cols = group_by_columns or list(df.columns)

        # ── Build groups dict (Orange's _compute_unique_data approach) ─
        col_values = [df.get_column(c).to_list() for c in key_cols]
        groups: dict[tuple, list[int]] = {}
        for i, key in enumerate(zip(*col_values)):
            # Normalise: convert NaN to None so NaN rows group together
            norm_key = tuple(
                None if isinstance(v, float) and v != v else v for v in key
            )
            groups.setdefault(norm_key, []).append(i)

        # ── Apply tiebreaker ──────────────────────────────────────────
        if tiebreaker == "First instance":
            selection = [inds[0] for inds in groups.values()]
        elif tiebreaker == "Last instance":
            selection = [inds[-1] for inds in groups.values()]
        elif tiebreaker == "Middle instance":
            selection = [inds[len(inds) // 2] for inds in groups.values()]
        elif tiebreaker == "Random instance":
            selection = [random.choice(inds) for inds in groups.values()]
        elif tiebreaker == "Discard non-unique":
            selection = [
                inds[0] for inds in groups.values() if len(inds) == 1
            ]
        else:
            selection = [inds[0] for inds in groups.values()]

        selection.sort()

        if not selection:
            # All rows discarded (Discard non-unique with no unique rows)
            result = df.head(0)
        else:
            result = df[selection]

        # ── Column reorder: Target → Meta → Features ──────────────────
        result = _reorder_columns(result, dataset.domain)

        return replace(
            dataset,
            dataset_id=f"{dataset.dataset_id}-unique",
            display_name=f"{dataset.display_name} (unique)",
            dataframe=result,
            row_count=result.height,
            domain=build_data_domain(result, source_domain=dataset.domain),
        )


def _reorder_columns(df, domain):
    """Reorder columns: Target → Meta → Features."""
    if domain is None:
        return df
    target_names = [c.name for c in domain.target_columns if c.name in df.columns]
    meta_names = [c.name for c in domain.meta_columns if c.name in df.columns]
    feature_names = [c.name for c in domain.feature_columns if c.name in df.columns]
    ordered = target_names + meta_names + feature_names
    seen = set(ordered)
    extras = [c for c in df.columns if c not in seen]
    final_order = ordered + extras
    if final_order == list(df.columns):
        return df
    return df.select(final_order)
