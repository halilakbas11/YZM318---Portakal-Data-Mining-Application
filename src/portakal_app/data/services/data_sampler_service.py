from __future__ import annotations

from dataclasses import replace

import numpy as np

from portakal_app.data.models import DatasetHandle, build_data_domain


class DataSamplerService:
    """Data sampling service aligned with Orange Data Mining standards.

    Uses numpy.random.RandomState (same as Orange) so identical seeds
    produce identical splits.
    """

    def sample(
        self,
        dataset: DatasetHandle,
        *,
        mode: str = "percentage",
        percentage: int = 70,
        fixed_size: int = 10,
        folds: int = 10,
        selected_fold: int = 1,
        with_replacement: bool = False,
        stratify: bool = False,
        seed: int | None = None,
    ) -> tuple[DatasetHandle, DatasetHandle | None]:
        df = dataset.dataframe
        n = df.height
        rgen = np.random.RandomState(seed)

        # ------------------------------------------------------------------
        # Fixed Proportion
        # ------------------------------------------------------------------
        if mode == "percentage":
            sample_size = max(1, int(np.ceil(n * percentage / 100)))
            if with_replacement:
                sample_idx, remaining_idx = _sample_with_replacement(
                    rgen, n, sample_size
                )
            elif stratify:
                sample_idx, remaining_idx = _stratified_sample(
                    rgen, dataset, sample_size, False
                )
            else:
                sample_idx, remaining_idx = _sample_without_replacement(
                    rgen, n, sample_size
                )

        # ------------------------------------------------------------------
        # Fixed Size
        # ------------------------------------------------------------------
        elif mode == "fixed":
            if with_replacement:
                sample_size = fixed_size
            else:
                sample_size = min(fixed_size, n)

            if with_replacement:
                sample_idx, remaining_idx = _sample_with_replacement(
                    rgen, n, sample_size
                )
            elif stratify:
                sample_idx, remaining_idx = _stratified_sample(
                    rgen, dataset, sample_size, False
                )
            else:
                sample_idx, remaining_idx = _sample_without_replacement(
                    rgen, n, sample_size
                )

        # ------------------------------------------------------------------
        # Cross Validation  (Data Sample = Train, Remaining = Test)
        # ------------------------------------------------------------------
        elif mode == "cross-validation":
            shuffled = np.arange(n)
            rgen.shuffle(shuffled)

            fold_size = n // folds
            fold_buckets: list[np.ndarray] = []
            for f in range(folds):
                start = f * fold_size
                end = (start + fold_size) if f < folds - 1 else n
                fold_buckets.append(shuffled[start:end])

            test_idx = np.sort(fold_buckets[selected_fold - 1])
            train_idx = np.sort(
                np.concatenate(
                    [b for i, b in enumerate(fold_buckets) if i != selected_fold - 1]
                )
            )

            train_df = _reorder_columns(df[train_idx.tolist()], dataset.domain)
            test_df = _reorder_columns(df[test_idx.tolist()], dataset.domain)
            return _build_pair(dataset, train_df, test_df)

        # ------------------------------------------------------------------
        # Bootstrap  (identical to Orange's SampleBootstrap)
        # ------------------------------------------------------------------
        elif mode == "bootstrap":
            sample = rgen.randint(0, n, n)
            sample.sort()

            insample = np.ones(n, dtype=bool)
            insample[sample] = False
            oob_idx = np.flatnonzero(insample)

            sample_df = _reorder_columns(df[sample.tolist()], dataset.domain)
            oob_df = (
                _reorder_columns(df[oob_idx.tolist()], dataset.domain)
                if len(oob_idx) > 0
                else None
            )
            return _build_pair(dataset, sample_df, oob_df)

        else:
            sample_idx = np.arange(n)
            remaining_idx = np.array([], dtype=int)

        sample_df = _reorder_columns(df[sample_idx.tolist()], dataset.domain)
        remaining_df = (
            _reorder_columns(df[remaining_idx.tolist()], dataset.domain)
            if len(remaining_idx) > 0
            else None
        )
        return _build_pair(dataset, sample_df, remaining_df)


# ── helpers ───────────────────────────────────────────────────────────────


def _sample_with_replacement(
    rgen: np.random.RandomState, n: int, size: int
) -> tuple[np.ndarray, np.ndarray]:
    """Orange's SampleRandomN with replace=True."""
    sample = rgen.randint(0, n, size)
    o = np.ones(n)
    o[sample] = 0
    others = np.nonzero(o)[0]
    return sample, others


def _sample_without_replacement(
    rgen: np.random.RandomState, n: int, size: int
) -> tuple[np.ndarray, np.ndarray]:
    """Orange's SampleRandomN with replace=False (via ShuffleSplit logic)."""
    indices = np.arange(n)
    rgen.shuffle(indices)
    sample = np.sort(indices[:size])
    remaining = np.sort(indices[size:])
    return sample, remaining


def _stratified_sample(
    rgen: np.random.RandomState,
    dataset: DatasetHandle,
    sample_size: int,
    with_replacement: bool,
) -> tuple[np.ndarray, np.ndarray]:
    target_cols = dataset.domain.target_columns if dataset.domain else ()
    if not target_cols:
        if with_replacement:
            return _sample_with_replacement(rgen, dataset.dataframe.height, sample_size)
        return _sample_without_replacement(rgen, dataset.dataframe.height, sample_size)

    target_name = target_cols[0].name
    series = dataset.dataframe.get_column(target_name)
    groups: dict[str, list[int]] = {}
    for i, val in enumerate(series.to_list()):
        groups.setdefault(str(val), []).append(i)

    n = dataset.dataframe.height
    sample_indices: list[int] = []
    for group_indices in groups.values():
        group_n = len(group_indices)
        group_size = max(1, round(group_n / n * sample_size))
        arr = np.array(group_indices)
        if with_replacement:
            chosen = rgen.choice(arr, size=group_size, replace=True)
        else:
            chosen = rgen.choice(arr, size=min(group_size, group_n), replace=False)
        sample_indices.extend(chosen.tolist())

    sample = np.array(sample_indices)
    if with_replacement:
        o = np.ones(n)
        o[sample] = 0
        remaining = np.nonzero(o)[0]
    else:
        sample = np.sort(np.unique(sample))
        remaining = np.setdiff1d(np.arange(n), sample)

    return sample, remaining


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


def _build_pair(
    dataset: DatasetHandle,
    sample_df,
    remaining_df,
) -> tuple[DatasetHandle, DatasetHandle | None]:
    sample_ds = replace(
        dataset,
        dataset_id=f"{dataset.dataset_id}-sample",
        display_name=f"{dataset.display_name} (sample)",
        dataframe=sample_df,
        row_count=sample_df.height,
        domain=build_data_domain(sample_df, source_domain=dataset.domain),
    )
    remaining_ds = None
    if remaining_df is not None and remaining_df.height > 0:
        remaining_ds = replace(
            dataset,
            dataset_id=f"{dataset.dataset_id}-remaining",
            display_name=f"{dataset.display_name} (remaining)",
            dataframe=remaining_df,
            row_count=remaining_df.height,
            domain=build_data_domain(remaining_df, source_domain=dataset.domain),
        )
    return sample_ds, remaining_ds
