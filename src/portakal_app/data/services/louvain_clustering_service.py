from __future__ import annotations

from dataclasses import replace

import numpy as np
import polars as pl
from sklearn.decomposition import PCA
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import StandardScaler

from portakal_app.data.models import DatasetHandle, build_data_domain


def _louvain_communities(adjacency: np.ndarray, resolution: float, rng: np.random.Generator) -> list[int]:
    """
    Simplified Louvain-style community detection on a weighted adjacency matrix.
    Uses greedy modularity optimisation (one pass = one phase).
    """
    n = adjacency.shape[0]
    # Degree (sum of edge weights)
    deg = adjacency.sum(axis=1)
    m = deg.sum() / 2.0
    if m == 0:
        return list(range(n))

    labels = list(range(n))

    improved = True
    while improved:
        improved = False
        order = rng.permutation(n)
        for i in order:
            current_community = labels[i]
            # Compute modularity gain for each neighbour community
            neighbour_comms: dict[int, float] = {}
            for j in range(n):
                if adjacency[i, j] > 0 and i != j:
                    c = labels[j]
                    neighbour_comms[c] = neighbour_comms.get(c, 0.0) + adjacency[i, j]

            # Remove i from its community
            labels[i] = -1
            best_comm = current_community
            best_gain = 0.0

            for c, k_ic in neighbour_comms.items():
                # Sum of degrees in community c
                sigma_c = sum(deg[j] for j in range(n) if labels[j] == c)
                gain = (k_ic / m) - resolution * (deg[i] * sigma_c) / (2 * m * m)
                if gain > best_gain:
                    best_gain = gain
                    best_comm = c

            labels[i] = best_comm
            if best_comm != current_community:
                improved = True

    # Remap labels to 0..K-1
    unique = sorted(set(labels))
    remap = {old: new for new, old in enumerate(unique)}
    return [remap[l] for l in labels]


class LouvainClusteringService:
    """Graph-based community detection via the Louvain algorithm."""

    def run(
        self,
        dataset: DatasetHandle,
        *,
        n_neighbors: int = 30,
        resolution: float = 1.0,
        random_state: int = 42,
        normalize: bool = True,
        apply_pca: bool = False,
        pca_components: int = 10,
        metric: str = "euclidean",
    ) -> DatasetHandle:
        df = dataset.dataframe

        numeric_cols = [
            col.name
            for col in dataset.domain.feature_columns
            if col.logical_type in ("numeric", "continuous", "integer")
        ]
        if not numeric_cols:
            raise ValueError("Louvain clustering requires at least one numeric feature column.")

        X = df.select(numeric_cols).to_numpy().astype(float)
        mask = ~np.isnan(X).any(axis=1)
        X = X[mask]

        if normalize:
            X = StandardScaler().fit_transform(X)

        if apply_pca and X.shape[1] > pca_components:
            n_pca = min(pca_components, X.shape[0] - 1, X.shape[1])
            X = PCA(n_components=n_pca, random_state=random_state).fit_transform(X)

        k = min(n_neighbors, len(X) - 1)

        # Build k-NN graph (sparse → dense for small datasets)
        graph = kneighbors_graph(X, n_neighbors=k, mode="distance", metric=metric, include_self=False)
        # Convert distance to similarity: w = exp(-d)
        graph_coo = graph.tocoo()
        weights = np.exp(-graph_coo.data)
        adj = np.zeros((len(X), len(X)), dtype=float)
        for i, j, w in zip(graph_coo.row, graph_coo.col, weights):
            adj[i, j] = w
            adj[j, i] = w  # symmetrize

        rng = np.random.default_rng(random_state)
        labels = _louvain_communities(adj, resolution=resolution, rng=rng)
        n_clusters = len(set(labels))

        result_df = df.filter(pl.Series(mask.tolist())).with_columns(
            pl.Series("louvain_cluster", labels, dtype=pl.Int32)
        )

        new_domain = build_data_domain(result_df, source_domain=dataset.domain)
        return replace(
            dataset,
            dataset_id=f"{dataset.dataset_id}-louvain",
            display_name=f"{dataset.display_name} (Louvain, {n_clusters} clusters)",
            dataframe=result_df,
            domain=new_domain,
            row_count=len(result_df),
            column_count=len(result_df.columns),
        )
