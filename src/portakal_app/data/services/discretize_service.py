from __future__ import annotations

import math
from dataclasses import replace
import polars as pl
import numpy as np

from portakal_app.data.models import DatasetHandle, build_data_domain

METHODS = (
    "Keep numeric",
    "Remove",
    "Natural binning",
    "Fixed width",
    "Time interval",
    "Equal frequency",
    "Equal width",
    "Entropy vs. MDL",
    "Custom",
    "Use default setting"
)

class DiscretizeService:
    def discretize(
        self,
        dataset: DatasetHandle,
        *,
        default_method: str = "Keep numeric",
        default_params: dict[str, object] | None = None,
        column_methods: dict[str, dict[str, object]] | None = None,
    ) -> DatasetHandle:
        df = dataset.dataframe
        new_columns: dict[str, pl.Series] = {}
        col_methods = column_methods or {}
        default_params = default_params or {"n_bins": 4}

        # Target series for Entropy vs. MDL
        target_series = None
        target_names = {c.name for c in dataset.domain.target_columns}
        if target_names:
            target_col = list(target_names)[0]
            if target_col in df.columns:
                target_series = df.get_column(target_col)

        for col_name in df.columns:
            series = df.get_column(col_name)
            
            # Non-numeric columns are just passed through
            if not series.dtype.is_numeric():
                new_columns[col_name] = series
                continue

            method_params = col_methods.get(col_name, {})
            method = method_params.get("method", "Use default setting")
            
            if method == "Use default setting":
                method = default_method
                params = default_params
            else:
                params = method_params

            if method == "Keep numeric":
                new_columns[col_name] = series
            elif method == "Remove":
                continue
            elif method == "Equal width":
                n_bins = int(params.get("n_bins", default_params.get("n_bins", 4)))
                new_columns[col_name] = _equal_width_bin(series, col_name, max(2, n_bins))
            elif method == "Natural binning":
                n_bins = int(params.get("n_bins", default_params.get("n_bins", 4)))
                new_columns[col_name] = _natural_bin_bin(series, col_name, max(2, n_bins))
            elif method == "Equal frequency":
                n_bins = int(params.get("n_bins", default_params.get("n_bins", 4)))
                new_columns[col_name] = _equal_freq_bin(series, col_name, max(2, n_bins))
            elif method == "Fixed width":
                width = float(params.get("width", default_params.get("width", 1.0)))
                new_columns[col_name] = _fixed_width_bin(series, col_name, width)
            elif method == "Custom":
                cuts_str = str(params.get("cuts", default_params.get("cuts", "")))
                try:
                    cuts = [float(x.strip()) for x in cuts_str.split(",") if x.strip()]
                    new_columns[col_name] = _custom_bin(series, col_name, cuts)
                except ValueError:
                    new_columns[col_name] = series
            elif method == "Entropy vs. MDL":
                if target_series is not None and col_name != target_series.name:
                    new_columns[col_name] = _entropy_mdl_bin(series, col_name, target_series)
                else:
                    new_columns[col_name] = series
            else:
                new_columns[col_name] = series

        result_df = pl.DataFrame(new_columns)

        return replace(
            dataset,
            dataset_id=f"{dataset.dataset_id}-discretized",
            display_name=f"{dataset.display_name} (discretized)",
            dataframe=result_df,
            row_count=result_df.height,
            column_count=result_df.width,
            domain=build_data_domain(result_df, source_domain=dataset.domain),
        )


def _equal_width_bin(series: pl.Series, name: str, n_bins: int) -> pl.Series:
    non_null = series.drop_nulls()
    if non_null.len() == 0:
        return pl.Series(name, ["?"] * series.len(), dtype=pl.Utf8)

    min_val = float(non_null.min())
    max_val = float(non_null.max())

    if min_val == max_val:
        return pl.Series(name, [f"[{min_val:.2f}]"] * series.len(), dtype=pl.Utf8)

    step = (max_val - min_val) / n_bins
    labels: list[str | None] = []
    for val in series.to_list():
        if val is None:
            labels.append(None)
        else:
            fval = float(val)
            bin_idx = min(int((fval - min_val) / step), n_bins - 1)
            lo = min_val + bin_idx * step
            hi = lo + step
            labels.append(f"[{lo:.2f}, {hi:.2f})")
    
    # Needs to be dict encoded for Polars categorical/ordinal if we want proper sort,
    # but for Portakal logic, string representation is temporarily fine to mimic Orange's physical discretization.
    return pl.Series(name, labels, dtype=pl.Utf8)


def _equal_freq_bin(series: pl.Series, name: str, n_bins: int) -> pl.Series:
    non_null = series.drop_nulls()
    if non_null.len() == 0:
        return pl.Series(name, ["?"] * series.len(), dtype=pl.Utf8)

    vals = non_null.cast(pl.Float64).to_list()
    quantiles = np.linspace(0, 100, n_bins + 1)[1:-1]
    thresholds = sorted(list(set(np.percentile(vals, quantiles))))

    if not thresholds:
        return pl.Series(name, [f"[{np.min(vals):.2f}]"] * series.len(), dtype=pl.Utf8)

    return _custom_bin(series, name, thresholds)

def _natural_binning_thresholds(min_v: float, max_v: float, n_bins: int) -> list[float]:
    if min_v >= max_v or n_bins <= 1:
        return []
    target_step = (max_v - min_v) / n_bins
    if target_step <= 0:
        return []
    magnitude = 10 ** math.floor(math.log10(target_step))
    residuals = [1, 2, 2.5, 5, 10]
    best_step = magnitude
    best_diff = float('inf')
    
    for r in residuals:
        step = r * magnitude
        diff = abs(step - target_step)
        if diff < best_diff:
            best_diff = diff
            best_step = step
            
    lo = math.floor(min_v / best_step) * best_step
    hi = math.ceil(max_v / best_step) * best_step
    
    thresholds = []
    curr = lo + best_step
    while curr < max_v - 1e-9:
        if curr > min_v + 1e-9:
            thresholds.append(curr)
        curr += best_step
        
    return thresholds

def _natural_bin_bin(series: pl.Series, name: str, n_bins: int) -> pl.Series:
    non_null = series.drop_nulls()
    if non_null.len() == 0:
        return pl.Series(name, ["?"] * series.len(), dtype=pl.Utf8)
    
    vals = non_null.cast(pl.Float64).to_list()
    min_v, max_v = min(vals), max(vals)
    if min_v == max_v:
        return pl.Series(name, [f"[{min_v:.2f}]"] * series.len(), dtype=pl.Utf8)

    thresholds = _natural_binning_thresholds(min_v, max_v, n_bins)
    if not thresholds:
        return _equal_width_bin(series, name, n_bins)
    return _custom_bin(series, name, thresholds)

def _mdl_discretize_recursive(vals: list[float], target: list[str], min_v: float, max_v: float) -> list[float]:
    n = len(vals)
    if n < 2:
        return []
        
    def entropy(targets: list[str]) -> float:
        counts = {}
        for t in targets:
            counts[t] = counts.get(t, 0) + 1
        total = len(targets)
        ent = 0.0
        for c in counts.values():
            p = c / total
            if p > 0:
                ent -= p * math.log2(p)
        return ent

    base_ent = entropy(target)
    
    best_gain = -1.0
    best_idx = -1
    best_cut = None
    
    for i in range(1, n):
        if vals[i] == vals[i-1]:
            continue
            
        cut = (vals[i] + vals[i-1]) / 2.0
        
        left_t = target[:i]
        right_t = target[i:]
        
        ent1 = entropy(left_t)
        ent2 = entropy(right_t)
        
        cond_ent = (len(left_t) / n) * ent1 + (len(right_t) / n) * ent2
        gain = base_ent - cond_ent
        
        if gain > best_gain:
            best_gain = gain
            best_idx = i
            best_cut = cut
            
    if best_idx == -1 or best_cut is None:
        return []
        
    left_t = target[:best_idx]
    right_t = target[best_idx:]
    k = len(set(target))
    k1 = len(set(left_t))
    k2 = len(set(right_t))
    
    delta = math.log2(3**k - 2) - (k * base_ent - k1 * entropy(left_t) - k2 * entropy(right_t))
    if best_gain > (math.log2(n - 1) + delta) / n:
        left_cuts = _mdl_discretize_recursive(vals[:best_idx], left_t, min_v, best_cut)
        right_cuts = _mdl_discretize_recursive(vals[best_idx:], right_t, best_cut, max_v)
        return left_cuts + [best_cut] + right_cuts
    else:
        return []

def _entropy_mdl_bin(series: pl.Series, col_name: str, target_series: pl.Series) -> pl.Series:
    non_null_mask = series.is_not_null() & target_series.is_not_null()
    if non_null_mask.sum() < 2:
        return series
        
    filtered_series = series.filter(non_null_mask).cast(pl.Float64)
    filtered_target = target_series.filter(non_null_mask).cast(pl.Utf8)
    
    f_list = filtered_series.to_list()
    t_list = filtered_target.to_list()
    
    pairs = sorted(zip(f_list, t_list), key=lambda x: x[0])
    vals, targets = zip(*pairs)
    
    cuts = _mdl_discretize_recursive(list(vals), list(targets), vals[0], vals[-1])
    return _custom_bin(series, col_name, cuts)

def _fixed_width_bin(series: pl.Series, name: str, width: float) -> pl.Series:
    if width <= 0:
        return series
    
    labels: list[str | None] = []
    for val in series.to_list():
        if val is None:
            labels.append(None)
        else:
            fval = float(val)
            base = math.floor(fval / width) * width
            labels.append(f"[{base:.2f}, {base + width:.2f})")
    return pl.Series(name, labels, dtype=pl.Utf8)

def _custom_bin(series: pl.Series, name: str, thresholds: list[float]) -> pl.Series:
    thresholds = sorted(list(set(thresholds)))
    if not thresholds:
        return series
        
    labels: list[str | None] = []
    for val in series.to_list():
        if val is None:
            labels.append(None)
        else:
            fval = float(val)
            bin_idx = 0
            for t in thresholds:
                if fval >= t:
                    bin_idx += 1
                    
            if bin_idx == 0:
                labels.append(f"< {thresholds[0]:.2f}")
            elif bin_idx >= len(thresholds):
                labels.append(f">= {thresholds[-1]:.2f}")
            else:
                labels.append(f"[{thresholds[bin_idx - 1]:.2f}, {thresholds[bin_idx]:.2f})")
                
    return pl.Series(name, labels, dtype=pl.Utf8)
