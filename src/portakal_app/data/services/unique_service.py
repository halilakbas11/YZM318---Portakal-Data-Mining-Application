from __future__ import annotations

import random
from dataclasses import replace

import polars as pl

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
    ) -> tuple[DatasetHandle, DatasetHandle | None, DatasetHandle]:
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

        unique_dataset = replace(
            dataset,
            dataset_id=f"{dataset.dataset_id}-unique",
            display_name=f"{dataset.display_name} (unique)",
            dataframe=result,
            row_count=result.height,
            column_count=result.width,
            domain=build_data_domain(result, source_domain=dataset.domain),
        )

        # ── Removed duplicates ─────────────────────────────────────────
        removed_indices = list(set(range(df.height)) - set(selection))
        removed_indices.sort()
        
        removed_dataset = None
        if removed_indices:
            removed_df = df[removed_indices]
            removed_df = _reorder_columns(removed_df, dataset.domain)
            removed_dataset = replace(
                dataset,
                dataset_id=f"{dataset.dataset_id}-removed",
                display_name=f"{dataset.display_name} (removed)",
                dataframe=removed_df,
                row_count=removed_df.height,
                column_count=removed_df.width,
                domain=build_data_domain(removed_df, source_domain=dataset.domain),
            )

        # ── Annotated Data ──────────────────────────────────────────────
        selection_set = set(selection)
        duplicate_col_data = ["No" if i in selection_set else "Yes" for i in range(df.height)]
        annotated_df = df.with_columns(pl.Series("Duplicate", duplicate_col_data))
        
        annotated_domain = build_data_domain(annotated_df, source_domain=dataset.domain)
        new_columns = [replace(c, role="meta") if c.name == "Duplicate" else c for c in annotated_domain.columns]
        annotated_domain = replace(annotated_domain, columns=tuple(new_columns))
        
        annotated_df = _reorder_columns(annotated_df, annotated_domain)

        annotated_dataset = replace(
            dataset,
            dataset_id=f"{dataset.dataset_id}-annotated",
            display_name=f"{dataset.display_name} (annotated)",
            dataframe=annotated_df,
            row_count=annotated_df.height,
            column_count=annotated_df.width,
            domain=annotated_domain,
        )

        return unique_dataset, removed_dataset, annotated_dataset


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
