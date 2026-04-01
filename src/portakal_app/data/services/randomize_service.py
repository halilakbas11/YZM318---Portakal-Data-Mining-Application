from __future__ import annotations

import random
from dataclasses import replace

import polars as pl

from portakal_app.data.models import DatasetHandle, build_data_domain


class RandomizeService:
    def randomize(
        self,
        dataset: DatasetHandle,
        *,
        shuffle_classes: bool = True,
        shuffle_features: bool = False,
        shuffle_metas: bool = False,
        shuffle_ratio: int = 100,
        seed: int | None = None,
    ) -> DatasetHandle:
        df = dataset.dataframe
        rng = random.Random(seed)
        n_rows = df.height
        n_shuffle = max(1, int(n_rows * shuffle_ratio / 100))
        shuffle_indices = rng.sample(range(n_rows), min(n_shuffle, n_rows))

        result_columns: dict[str, pl.Series] = {}
        for col_schema in dataset.domain.columns:
            series = df.get_column(col_schema.name)
            should_shuffle = (
                (col_schema.role == "target" and shuffle_classes)
                or (col_schema.role == "feature" and shuffle_features)
                or (col_schema.role == "meta" and shuffle_metas)
            )
            if should_shuffle and shuffle_indices:
                values = series.to_list()
                subset = [values[i] for i in shuffle_indices]
                rng.shuffle(subset)
                for idx, si in enumerate(shuffle_indices):
                    values[si] = subset[idx]
                series = pl.Series(col_schema.name, values, dtype=series.dtype)
            result_columns[col_schema.name] = series

        new_df = pl.DataFrame(result_columns)
        return replace(
            dataset,
            dataset_id=f"{dataset.dataset_id}-randomized",
            display_name=f"{dataset.display_name} (randomized)",
            dataframe=new_df,
            domain=build_data_domain(new_df, source_domain=dataset.domain),
        )
