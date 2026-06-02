from __future__ import annotations

from dataclasses import replace

import numpy as np
import polars as pl
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

from portakal_app.data.models import DatasetHandle, build_data_domain


class TSNEService:
    """t-Distributed Stochastic Neighbor Embedding (t-SNE) dimensionality reduction."""

    def run(
        self,
        dataset: DatasetHandle,
        *,
        n_components: int = 2,
        perplexity: float = 30.0,
        learning_rate: float | str = "auto",
        max_iter: int = 1000,
        random_state: int = 42,
        normalize: bool = True,
        init: str = "pca",
        metric: str = "euclidean",
        apply_pca: bool = False,
        pca_components: int = 20,
    ) -> DatasetHandle:
        df = dataset.dataframe

        numeric_cols = [
            col.name
            for col in dataset.domain.feature_columns
            if col.logical_type in ("numeric", "continuous", "integer")
        ]
        if not numeric_cols:
            raise ValueError("t-SNE requires at least one numeric feature column.")

        X = df.select(numeric_cols).to_numpy()
        mask = ~np.isnan(X).any(axis=1)
        X = X[mask]

        if normalize:
            X = StandardScaler().fit_transform(X)

        if apply_pca and X.shape[1] > pca_components:
            from sklearn.decomposition import PCA
            n_pca = min(pca_components, X.shape[0] - 1, X.shape[1])
            X = PCA(n_components=n_pca, random_state=random_state).fit_transform(X)

        n_comp = min(n_components, X.shape[1], X.shape[0] - 1)
        perp = min(perplexity, X.shape[0] - 1)

        actual_init = init if metric == "euclidean" else "random"

        tsne = TSNE(
            n_components=n_comp,
            perplexity=perp,
            learning_rate=learning_rate,
            max_iter=max_iter,
            random_state=random_state,
            init=actual_init,
            metric=metric,
        )
        embedding = tsne.fit_transform(X)

        component_names = [f"tsne-{i + 1}" for i in range(n_comp)]
        embed_df = pl.DataFrame(
            {name: embedding[:, i].tolist() for i, name in enumerate(component_names)}
        )

        other_cols = [c for c in df.columns if c not in numeric_cols]
        if other_cols:
            other_df = df.select(other_cols).filter(pl.Series(mask.tolist()))
            result_df = pl.concat([other_df, embed_df], how="horizontal")
        else:
            result_df = embed_df

        new_domain = build_data_domain(result_df, source_domain=dataset.domain)
        return replace(
            dataset,
            dataset_id=f"{dataset.dataset_id}-tsne",
            display_name=f"{dataset.display_name} (t-SNE)",
            dataframe=result_df,
            domain=new_domain,
            row_count=len(result_df),
            column_count=len(result_df.columns),
        )
