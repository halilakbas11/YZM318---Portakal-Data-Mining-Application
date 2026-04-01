from __future__ import annotations

import math
import random as _random
from dataclasses import dataclass, replace
from typing import Any

import numpy as np
import polars as pl
from portakal_app.data.models import DatasetHandle, build_data_domain

@dataclass
class PreprocessStep:
    name: str
    params: dict[str, Any]

class PreprocessService:
    def preprocess(
        self,
        dataset: DatasetHandle,
        *,
        steps: list[PreprocessStep],
        missing_threshold: float = 0.5,
    ) -> DatasetHandle:
        df = dataset.dataframe

        # Track column roles from the original domain throughout the pipeline
        target_names = {c.name for c in dataset.domain.target_columns}
        meta_names = {c.name for c in dataset.domain.meta_columns}

        for step in steps:
            if isinstance(step, str):
                step = PreprocessStep(name=step, params={})

            # Recompute feature columns after every step (columns may have changed)
            feature_cols = [
                c for c in df.columns
                if c not in target_names and c not in meta_names
            ]

            # ── Legacy simple-mode steps ──────────────────────────
            if step.name == "Normalize (0-1)":
                df = _normalize(df, method="Normalize to [0, 1]",
                                columns=feature_cols)
            elif step.name == "Standardize (mean=0, var=1)":
                df = _normalize(df, method="Standardize to μ=0, σ²=1",
                                columns=feature_cols)
            elif step.name == "Remove rows with missing values":
                df = df.drop_nulls()
            elif step.name == "Remove constant features":
                cols = []
                for c in df.columns:
                    valid_series = df.get_column(c).drop_nulls()
                    if valid_series.dtype in (pl.Utf8, pl.Categorical):
                        valid_series = valid_series.filter(valid_series.cast(pl.Utf8) != "")
                    if valid_series.n_unique() <= 1:
                        cols.append(c)
                if cols:
                    df = df.drop(cols)
            elif step.name == "Remove features with too many missing values":
                cols = [c for c in feature_cols
                        if df.get_column(c).null_count()
                        / max(df.height, 1) > missing_threshold]
                if cols:
                    df = df.drop(cols)

            # ── Modern steps ──────────────────────────────────────
            elif step.name == "Normalize Features":
                method = step.params.get("method", "Standardize to μ=0, σ²=1")
                df = _normalize(df, method=method, columns=feature_cols)

            elif step.name == "Impute Missing Values":
                method = step.params.get("method", "Average/Most frequent")
                if method == "Remove rows with missing values":
                    df = df.drop_nulls()
                elif method == "Average/Most frequent":
                    df = _impute_average(df)
                elif method == "Replace with random value":
                    df = df.fill_null(strategy="backward").fill_null(
                        strategy="forward")

            elif step.name == "Continuize Discrete Variables":
                method = step.params.get("method", "One feature per value")
                df = _continuize(df, method=method,
                                 target_names=target_names,
                                 meta_names=meta_names)

            elif step.name == "Select Relevant Features":
                k = step.params.get("k", 10)
                strategy = step.params.get("strategy", "Fixed")
                score_method = step.params.get("score", "Information Gain")
                if strategy != "Fixed":
                    k = int(max(len(feature_cols) * float(k) / 100, 1))
                k = min(int(k), len(feature_cols))
                if k < len(feature_cols):
                    selected = _select_relevant_features(
                        df, feature_cols, target_names, score_method, k)
                    keep = set(selected) | target_names | meta_names
                    df = df.select([c for c in df.columns if c in keep])

            elif step.name == "Select Random Features":
                k = step.params.get("k", 10)
                strategy = step.params.get("strategy", "Fixed")
                if strategy != "Fixed":
                    k = int(max(len(feature_cols) * float(k) / 100, 1))
                k = min(int(k), len(feature_cols))
                if k < len(feature_cols):
                    selected = _random.sample(feature_cols, k)
                    keep = set(selected) | target_names | meta_names
                    df = df.select([c for c in df.columns if c in keep])

            elif step.name == "Remove Sparse Features":
                use_fixed = step.params.get("useFixedThreshold", False)
                if use_fixed:
                    threshold = step.params.get("fixedThresh", 50)
                else:
                    threshold = (df.height
                                 * float(step.params.get("percThresh", 5))
                                 / 100)
                filter0 = step.params.get("filter0", 0)
                cols_to_drop = []
                for c in feature_cols:          # only feature columns
                    col = df.get_column(c)
                    count = (col.null_count() if filter0 == 0
                             else (col == 0).sum())
                    if count >= threshold:
                        cols_to_drop.append(c)
                if cols_to_drop:
                    df = df.drop(cols_to_drop)

            elif step.name == "Randomize":
                df = _randomize(
                    df, target_names, meta_names,
                    shuffle_classes=step.params.get("classes", True),
                    shuffle_features=step.params.get("features", False),
                    shuffle_meta=step.params.get("meta", False),
                )

            elif step.name == "Principal Component Analysis":
                n_components = int(step.params.get("n_components", 5))
                df = _pca(df, feature_cols, n_components,
                          target_names, meta_names)

            elif step.name == "CUR Matrix Decomposition":
                k = step.params.get("k", 10)
                strategy = step.params.get("strategy", "Fixed")
                max_error = float(step.params.get("max_error", 1.0))
                if strategy != "Fixed":
                    k = int(max(len(feature_cols) * float(k) / 100, 1))
                k = min(int(k), len(feature_cols))
                df = _cur_decomposition(df, feature_cols, k, max_error,
                                        target_names, meta_names)

        return replace(
            dataset,
            dataset_id=f"{dataset.dataset_id}-preprocessed",
            display_name=f"{dataset.display_name} (preprocessed)",
            dataframe=df,
            row_count=df.height,
            column_count=df.width,
            domain=build_data_domain(df, source_domain=dataset.domain),
        )

def _continuize(
    df: pl.DataFrame,
    method: str,
    target_names: set[str],
    meta_names: set[str],
) -> pl.DataFrame:
    """Continuize categorical feature columns. Target/meta columns pass through."""
    result_series: list[pl.Series] = []

    for c in df.columns:
        series = df.get_column(c)

        # Always keep target / meta columns untouched
        if c in target_names or c in meta_names:
            result_series.append(series)
            continue

        is_categorical = (series.dtype == pl.Utf8
                          or series.dtype == pl.Categorical)
        if not is_categorical:
            result_series.append(series)
            continue

        # ── categorical feature column ──
        if method == "Remove categorical features":
            continue

        if method == "Remove non-binary features":
            if series.drop_nulls().n_unique() > 2:
                continue
            result_series.append(series)
            continue

        str_series = series.cast(pl.Utf8).fill_null("")
        unique_vals = sorted(set(str_series.to_list()) - {""})
        if not unique_vals:
            result_series.append(series)
            continue

        if method == "One feature per value":
            for val in unique_vals:
                ind = [1.0 if str(v) == val else 0.0
                       for v in str_series.to_list()]
                result_series.append(
                    pl.Series(f"{c}={val}", ind, dtype=pl.Float64))

        elif method == "Treat as ordinal":
            mapping = {v: float(i) for i, v in enumerate(unique_vals)}
            vals = [mapping.get(str(v), None) if str(v) != "" else None
                    for v in str_series.to_list()]
            result_series.append(pl.Series(c, vals, dtype=pl.Float64))

        elif method == "Most frequent is base":
            freq: dict[str, int] = {}
            for v in str_series.to_list():
                if v:
                    freq[v] = freq.get(v, 0) + 1
            if not freq:
                continue
            most_freq = max(freq, key=lambda x: freq[x])
            for val in unique_vals:
                if val == most_freq:
                    continue
                ind = [1.0 if str(v) == val else 0.0
                       for v in str_series.to_list()]
                result_series.append(
                    pl.Series(f"{c}={val}", ind, dtype=pl.Float64))

        elif method == "Divide by number of values":
            n = len(unique_vals)
            if n > 1:
                mapping = {v: float(i) / (n - 1) for i, v in enumerate(unique_vals)}
            elif n == 1:
                mapping = {unique_vals[0]: 0.0}
            else:
                mapping = {}
            vals = [mapping.get(str(v), None) if str(v) != "" else None
                    for v in str_series.to_list()]
            result_series.append(pl.Series(c, vals, dtype=pl.Float64))
        else:
            result_series.append(series)

    return pl.DataFrame(result_series) if result_series else pl.DataFrame()

def _impute_average(df: pl.DataFrame) -> pl.DataFrame:
    res = df
    for col_name in df.columns:
        series = df.get_column(col_name)
        if series.dtype.is_numeric():
            mean = series.mean()
            if mean is not None:
                res = res.with_columns(series.fill_null(mean).alias(col_name))
        else:
            mode_s = series.mode()
            if mode_s.len() > 0:
                res = res.with_columns(series.fill_null(mode_s[0]).alias(col_name))
    return res

def _normalize(
    df: pl.DataFrame,
    method: str,
    columns: list[str] | None = None,
) -> pl.DataFrame:
    result = df
    cols_to_process = columns if columns is not None else df.columns
    for col_name in cols_to_process:
        if col_name not in df.columns:
            continue
        series = df.get_column(col_name)
        if not series.dtype.is_numeric():
            continue
        float_s = series.cast(pl.Float64)
        if "Normalize" in method:
            min_val = float_s.min()
            max_val = float_s.max()
            if min_val is None or max_val is None or min_val == max_val:
                continue
            expr = (pl.col(col_name).cast(pl.Float64) - min_val) / (max_val - min_val)
            if "[-1, 1]" in method:
                expr = expr * 2 - 1
            result = result.with_columns(expr.alias(col_name))
        elif "Standardize" in method or "Scale" in method or "Center" in method:
            mean = float_s.mean() if ("Standardize" in method or "Center" in method) else 0.0
            std = float_s.std() if ("Standardize" in method or "Scale" in method) else 1.0
            if mean is None or std is None or std == 0:
                continue
            expr = (pl.col(col_name).cast(pl.Float64) - mean) / std
            result = result.with_columns(expr.alias(col_name))
    return result


# ── Randomize ─────────────────────────────────────────────────
def _randomize(
    df: pl.DataFrame,
    target_names: set[str],
    meta_names: set[str],
    *,
    shuffle_classes: bool,
    shuffle_features: bool,
    shuffle_meta: bool,
) -> pl.DataFrame:
    """Shuffle individual columns by role (like Orange3 Randomize)."""
    rng = _random.Random()
    result_columns: dict[str, pl.Series] = {}

    for c in df.columns:
        series = df.get_column(c)
        if c in target_names:
            role = "target"
        elif c in meta_names:
            role = "meta"
        else:
            role = "feature"

        should_shuffle = (
            (role == "target" and shuffle_classes)
            or (role == "feature" and shuffle_features)
            or (role == "meta" and shuffle_meta)
        )
        if should_shuffle:
            values = series.to_list()
            rng.shuffle(values)
            series = pl.Series(c, values, dtype=series.dtype)
        result_columns[c] = series

    return pl.DataFrame(result_columns)


# ── Feature scoring helpers ───────────────────────────────────
def _entropy_from_list(values: list[str]) -> float:
    total = len(values)
    if total == 0:
        return 0.0
    counts: dict[str, int] = {}
    for v in values:
        counts[v] = counts.get(v, 0) + 1
    ent = 0.0
    for c in counts.values():
        p = c / total
        if p > 0:
            ent -= p * math.log2(p)
    return ent


def _info_gain_score(feature: pl.Series, target: pl.Series) -> float:
    target_list = target.cast(pl.Utf8).fill_null("__NULL__").to_list()
    target_entropy = _entropy_from_list(target_list)

    if feature.dtype.is_numeric():
        feat_arr = feature.cast(pl.Float64).fill_null(float("nan")).to_list()
        non_null = [x for x in feat_arr if not math.isnan(x)]
        if len(non_null) < 2:
            return 0.0
        min_v, max_v = min(non_null), max(non_null)
        if min_v == max_v:
            return 0.0
        n_bins = min(10, len(set(non_null)))
        if n_bins <= 1:
            return 0.0
        bin_w = (max_v - min_v) / n_bins
        feat_list = [
            str(min(int((x - min_v) / bin_w), n_bins - 1))
            if not math.isnan(x) else "__NULL__"
            for x in feat_arr
        ]
    else:
        feat_list = feature.cast(pl.Utf8).fill_null("__NULL__").to_list()

    total = len(feat_list)
    if total == 0:
        return 0.0

    groups: dict[str, list[str]] = {}
    for f, t in zip(feat_list, target_list):
        groups.setdefault(f, []).append(t)

    cond_entropy = sum(
        len(vals) / total * _entropy_from_list(vals)
        for vals in groups.values()
    )
    return target_entropy - cond_entropy


def _anova_score(feature: pl.Series, target: pl.Series) -> float:
    if not feature.dtype.is_numeric():
        return 0.0
    feat_vals = feature.cast(pl.Float64).fill_null(float("nan")).to_list()
    target_list = target.cast(pl.Utf8).fill_null("__NULL__").to_list()

    groups: dict[str, list[float]] = {}
    for f, t in zip(feat_vals, target_list):
        if not math.isnan(f):
            groups.setdefault(t, []).append(f)
    if len(groups) < 2:
        return 0.0

    all_vals = [v for vals in groups.values() for v in vals]
    grand_mean = sum(all_vals) / len(all_vals)
    ss_between = sum(
        len(vals) * (sum(vals) / len(vals) - grand_mean) ** 2
        for vals in groups.values()
    )
    ss_within = sum(
        sum((v - sum(vals) / len(vals)) ** 2 for v in vals)
        for vals in groups.values()
    )
    df_between = len(groups) - 1
    df_within = len(all_vals) - len(groups)
    if df_within <= 0 or ss_within == 0:
        return float("inf") if ss_between > 0 else 0.0
    return (ss_between / df_between) / (ss_within / df_within)


def _chi2_score(feature: pl.Series, target: pl.Series) -> float:
    feat_list = feature.cast(pl.Utf8).fill_null("__NULL__").to_list()
    target_list = target.cast(pl.Utf8).fill_null("__NULL__").to_list()
    total = len(feat_list)
    if total == 0:
        return 0.0
    observed: dict[tuple[str, str], int] = {}
    feat_totals: dict[str, int] = {}
    target_totals: dict[str, int] = {}
    for f, t in zip(feat_list, target_list):
        observed[(f, t)] = observed.get((f, t), 0) + 1
        feat_totals[f] = feat_totals.get(f, 0) + 1
        target_totals[t] = target_totals.get(t, 0) + 1
    chi2 = 0.0
    for (f, t), obs in observed.items():
        exp = feat_totals[f] * target_totals[t] / total
        if exp > 0:
            chi2 += (obs - exp) ** 2 / exp
    return chi2


def _linear_regression_score(feature: pl.Series, target: pl.Series) -> float:
    if not feature.dtype.is_numeric() or not target.dtype.is_numeric():
        return 0.0
    f_list = feature.cast(pl.Float64).fill_null(float("nan")).to_list()
    t_list = target.cast(pl.Float64).fill_null(float("nan")).to_list()
    pairs = [(f, t) for f, t in zip(f_list, t_list)
             if not math.isnan(f) and not math.isnan(t)]
    if len(pairs) < 2:
        return 0.0
    fs, ts = zip(*pairs)
    n = len(fs)
    mean_f = sum(fs) / n
    mean_t = sum(ts) / n
    cov = sum((f - mean_f) * (t - mean_t) for f, t in zip(fs, ts))
    var_f = sum((f - mean_f) ** 2 for f in fs)
    var_t = sum((t - mean_t) ** 2 for t in ts)
    denom = math.sqrt(var_f * var_t)
    if denom == 0:
        return 0.0
    return abs(cov / denom)


def _select_relevant_features(
    df: pl.DataFrame,
    feature_cols: list[str],
    target_names: set[str],
    score_method: str,
    k: int,
) -> list[str]:
    """Score every feature column against the target and return top-k names."""
    target_col = None
    for t in target_names:
        if t in df.columns:
            target_col = t
            break
    if target_col is None:
        return feature_cols[:k]

    target_series = df.get_column(target_col)
    scores: dict[str, float] = {}
    for c in feature_cols:
        if c not in df.columns:
            continue
        feat = df.get_column(c)
        try:
            if score_method in ("Information Gain", "Gain Ratio", "Gini Index"):
                sc = _info_gain_score(feat, target_series)
                if score_method == "Gain Ratio":
                    intrinsic = _entropy_from_list(
                        feat.cast(pl.Utf8).fill_null("__NULL__").to_list())
                    sc = sc / intrinsic if intrinsic > 0 else 0.0
            elif score_method == "ANOVA":
                sc = _anova_score(feat, target_series)
            elif score_method == "Chi2":
                sc = _chi2_score(feat, target_series)
            elif score_method == "Univariate Linear Regression":
                sc = _linear_regression_score(feat, target_series)
            else:
                sc = _info_gain_score(feat, target_series)
        except Exception:
            sc = 0.0
        scores[c] = sc

    return sorted(scores, key=lambda c: scores[c], reverse=True)[:k]


# ── PCA ───────────────────────────────────────────────────────
def _pca(
    df: pl.DataFrame,
    feature_cols: list[str],
    n_components: int,
    target_names: set[str],
    meta_names: set[str],
) -> pl.DataFrame:
    numeric_features = [
        c for c in feature_cols
        if c in df.columns and df.get_column(c).dtype.is_numeric()
    ]
    if not numeric_features:
        return df

    n_components = min(n_components, len(numeric_features), df.height)
    if n_components <= 0:
        return df

    X = np.column_stack([
        df.get_column(c).cast(pl.Float64).fill_null(0.0).to_list()
        for c in numeric_features
    ]).astype(np.float64)

    col_means = np.nanmean(X, axis=0)
    nan_mask = np.isnan(X)
    for j in range(X.shape[1]):
        X[nan_mask[:, j], j] = col_means[j] if not np.isnan(col_means[j]) else 0.0

    X_centered = X - col_means

    try:
        _U, _s, Vt = np.linalg.svd(X_centered, full_matrices=False)
        components = X_centered @ Vt[:n_components].T
    except np.linalg.LinAlgError:
        return df

    result_series: list[pl.Series] = []
    for i in range(n_components):
        result_series.append(
            pl.Series(f"PC{i + 1}", components[:, i].tolist(),
                      dtype=pl.Float64))
    for c in df.columns:
        if c in target_names or c in meta_names:
            result_series.append(df.get_column(c))

    return pl.DataFrame(result_series) if result_series else df


# ── CUR Matrix Decomposition ─────────────────────────────────
def _cur_decomposition(
    df: pl.DataFrame,
    feature_cols: list[str],
    rank: int,
    max_error: float,
    target_names: set[str],
    meta_names: set[str],
) -> pl.DataFrame:
    numeric_features = [
        c for c in feature_cols
        if c in df.columns and df.get_column(c).dtype.is_numeric()
    ]
    if not numeric_features or rank <= 0:
        return df

    rank = min(rank, len(numeric_features), df.height - 1)
    if rank <= 0:
        return df

    X = np.column_stack([
        df.get_column(c).cast(pl.Float64).fill_null(0.0).to_list()
        for c in numeric_features
    ]).astype(np.float64)

    col_means = np.nanmean(X, axis=0)
    nan_mask = np.isnan(X)
    for j in range(X.shape[1]):
        X[nan_mask[:, j], j] = col_means[j] if not np.isnan(col_means[j]) else 0.0

    try:
        _U, _s, Vt = np.linalg.svd(X, full_matrices=False)
        V = Vt[:rank].T  # (n_features, rank)
        leverage = np.sum(V ** 2, axis=1) / rank
        
        c_factor = rank * math.log(max(rank, 2)) / max(max_error ** 2, 0.0001)
        num_cols = min(int(math.ceil(c_factor)), len(numeric_features))
        
        top_idx = np.argsort(-leverage)[:num_cols]
        selected_cols_info = [(numeric_features[i], leverage[i]) for i in top_idx]
    except np.linalg.LinAlgError:
        return df

    if not selected_cols_info:
        return df

    result_series = []
    for col_name, score in selected_cols_info:
        s = df.get_column(col_name)
        new_name = f"{col_name} ({score:.4f})"
        result_series.append(s.alias(new_name))
        
    for c in df.columns:
        if c in target_names or c in meta_names:
            result_series.append(df.get_column(c))

    return pl.DataFrame(result_series) if result_series else df
