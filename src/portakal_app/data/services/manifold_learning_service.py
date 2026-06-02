from __future__ import annotations

from dataclasses import replace

import numpy as np
import polars as pl
from sklearn import manifold
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

from portakal_app.data.models import DatasetHandle, build_data_domain

METHODS: list[str] = [
    "t-SNE",
    "Isomap",
    "LLE",
    "Hessian LLE",
    "LTSA",
    "MDS",
    "Spectral Embedding",
]


class ManifoldLearningService:
    """Non-linear dimensionality reduction via various manifold learning methods."""

    def run(
        self,
        dataset: DatasetHandle,
        *,
        method: str = "t-SNE",
        n_components: int = 2,
        n_neighbors: int = 10,
        random_state: int = 42,
        normalize: bool = True,
        metric: str = "euclidean",
        # t-SNE specific
        perplexity: float = 30.0,
        early_exaggeration: float = 12.0,
        tsne_learning_rate: float = 200.0,
        max_iter: int = 1000,
        tsne_init: str = "pca",
    ) -> DatasetHandle:
        df = dataset.dataframe

        numeric_cols = [
            col.name
            for col in dataset.domain.feature_columns
            if col.logical_type in ("numeric", "continuous", "integer")
        ]
        if not numeric_cols:
            raise ValueError("Manifold learning requires at least one numeric feature column.")

        X = df.select(numeric_cols).to_numpy()
        mask = ~np.isnan(X).any(axis=1)
        X = X[mask]

        if normalize:
            X = StandardScaler().fit_transform(X)

        n_comp = min(n_components, X.shape[1])

        if method == "t-SNE":
            perp = min(perplexity, X.shape[0] - 1)
            actual_init = tsne_init if metric == "euclidean" else "random"
            estimator = TSNE(
                n_components=n_comp,
                perplexity=perp,
                early_exaggeration=early_exaggeration,
                learning_rate=tsne_learning_rate,
                max_iter=max_iter,
                random_state=random_state,
                init=actual_init,
                metric=metric,
            )
        elif method == "Isomap":
            estimator = manifold.Isomap(n_neighbors=n_neighbors, n_components=n_comp)
        elif method == "LLE":
            estimator = manifold.LocallyLinearEmbedding(
                n_neighbors=n_neighbors, n_components=n_comp,
                method="standard", random_state=random_state,
            )
        elif method == "Hessian LLE":
            estimator = manifold.LocallyLinearEmbedding(
                n_neighbors=n_neighbors, n_components=n_comp,
                method="hessian", random_state=random_state,
            )
        elif method == "LTSA":
            estimator = manifold.LocallyLinearEmbedding(
                n_neighbors=n_neighbors, n_components=n_comp,
                method="ltsa", random_state=random_state,
            )
        elif method == "MDS":
            estimator = manifold.MDS(n_components=n_comp, random_state=random_state)
        elif method == "Spectral Embedding":
            estimator = manifold.SpectralEmbedding(
                n_components=n_comp, n_neighbors=n_neighbors, random_state=random_state,
            )
        else:
            raise ValueError(f"Unknown manifold method: {method}")

        embedding = estimator.fit_transform(X)

        prefix = method.lower().replace(" ", "_").replace("-", "")
        component_names = [f"{prefix}-{i + 1}" for i in range(n_comp)]
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
            dataset_id=f"{dataset.dataset_id}-manifold",
            display_name=f"{dataset.display_name} ({method})",
            dataframe=result_df,
            domain=new_domain,
            row_count=len(result_df),
            column_count=len(result_df.columns),
        )
