from __future__ import annotations

from dataclasses import replace

import numpy as np
import polars as pl
from sklearn.preprocessing import StandardScaler

from portakal_app.data.models import DatasetHandle, build_data_domain


class SOMService:
    """Self-Organizing Map (Kohonen Map) implemented in pure NumPy."""

    def run(
        self,
        dataset: DatasetHandle,
        *,
        grid_rows: int = 5,
        grid_cols: int = 5,
        sigma: float = 1.0,
        learning_rate: float = 0.5,
        n_iter: int = 1000,
        random_state: int = 42,
        normalize: bool = True,
        init: str = "pca",
    ) -> DatasetHandle:
        df = dataset.dataframe

        numeric_cols = [
            col.name
            for col in dataset.domain.feature_columns
            if col.logical_type in ("numeric", "continuous", "integer")
        ]
        if not numeric_cols:
            raise ValueError("SOM requires at least one numeric feature column.")

        X = df.select(numeric_cols).to_numpy().astype(float)
        mask = ~np.isnan(X).any(axis=1)
        X = X[mask]

        if normalize:
            X = StandardScaler().fit_transform(X)

        rng = np.random.default_rng(random_state)
        n_features = X.shape[1]

        # Initialize weights
        if init == "pca" and X.shape[0] > 2 and n_features >= 2:
            from sklearn.decomposition import PCA
            pca = PCA(n_components=min(2, n_features), random_state=random_state)
            pca.fit(X)
            pc1 = pca.components_[0]
            pc2 = pca.components_[1] if pca.n_components_ > 1 else np.zeros_like(pc1)
            weights = np.zeros((grid_rows, grid_cols, n_features))
            for r in range(grid_rows):
                for c in range(grid_cols):
                    t1 = (r / max(grid_rows - 1, 1) - 0.5) * 2
                    t2 = (c / max(grid_cols - 1, 1) - 0.5) * 2
                    weights[r, c] = t1 * pc1 + t2 * pc2
        else:
            weights = rng.uniform(
                X.min(axis=0), X.max(axis=0),
                size=(grid_rows, grid_cols, n_features)
            )

        # Precompute grid positions for neighbourhood calculation
        grid_pos = np.array(
            [[r, c] for r in range(grid_rows) for c in range(grid_cols)],
            dtype=float
        )

        for t in range(n_iter):
            decay = 1.0 - t / n_iter
            lr = learning_rate * decay
            sig = max(sigma * decay, 0.5)

            # Pick a random sample
            x = X[rng.integers(0, len(X))]

            # Find BMU (best matching unit)
            diffs = weights.reshape(-1, n_features) - x
            dists = np.linalg.norm(diffs, axis=1)
            bmu_idx = int(np.argmin(dists))
            bmu_pos = grid_pos[bmu_idx]

            # Neighbourhood function
            neighbour_dists = np.linalg.norm(grid_pos - bmu_pos, axis=1)
            neighbourhood = np.exp(-(neighbour_dists ** 2) / (2 * sig ** 2))

            # Update weights
            delta = neighbourhood[:, np.newaxis] * (x - weights.reshape(-1, n_features))
            weights += (lr * delta).reshape(grid_rows, grid_cols, n_features)

        # Map each sample to its BMU coordinates
        bmu_rows, bmu_cols = [], []
        for x in X:
            diffs = weights.reshape(-1, n_features) - x
            dists = np.linalg.norm(diffs, axis=1)
            bmu_idx = int(np.argmin(dists))
            bmu_rows.append(bmu_idx // grid_cols)
            bmu_cols.append(bmu_idx % grid_cols)

        embed_df = pl.DataFrame({"som-row": bmu_rows, "som-col": bmu_cols})

        other_cols = [c for c in df.columns if c not in numeric_cols]
        if other_cols:
            other_df = df.select(other_cols).filter(pl.Series(mask.tolist()))
            result_df = pl.concat([other_df, embed_df], how="horizontal")
        else:
            result_df = embed_df

        new_domain = build_data_domain(result_df, source_domain=dataset.domain)
        return replace(
            dataset,
            dataset_id=f"{dataset.dataset_id}-som",
            display_name=f"{dataset.display_name} (SOM {grid_rows}×{grid_cols})",
            dataframe=result_df,
            domain=new_domain,
            row_count=len(result_df),
            column_count=len(result_df.columns),
        )
