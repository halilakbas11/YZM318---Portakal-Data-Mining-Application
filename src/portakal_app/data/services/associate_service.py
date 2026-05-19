from __future__ import annotations

from typing import Any
import pandas as pd
import polars as pl
from mlxtend.frequent_patterns import apriori, fpgrowth, association_rules

from portakal_app.data.models import DatasetHandle

class AssociateService:
    def preprocess_to_onehot(self, dataset: DatasetHandle) -> pd.DataFrame:
        """
        Convert the given DatasetHandle to a boolean pandas DataFrame suitable for mlxtend.
        Categorical columns are one-hot encoded.
        Boolean columns are kept as is.
        Numeric columns are ignored or binned (for now, ignored or simply booleanized).
        """
        df_pl = dataset.dataframe
        onehot_cols = []
        
        for col in dataset.domain.feature_columns + dataset.domain.target_columns:
            if col.logical_type in {"categorical", "string"}:
                # One-hot encode
                series = df_pl.get_column(col.name)
                unique_vals = series.drop_nulls().unique().to_list()
                for val in unique_vals:
                    col_name = f"{col.name}={val}"
                    bool_series = (series == val).fill_null(False)
                    onehot_cols.append(pd.Series(bool_series.to_list(), name=col_name))
            elif col.logical_type == "boolean":
                series = df_pl.get_column(col.name).fill_null(False)
                onehot_cols.append(pd.Series(series.to_list(), name=col.name).astype(bool))
                
        if not onehot_cols:
            return pd.DataFrame()
            
        return pd.concat(onehot_cols, axis=1)

    def find_frequent_itemsets(self, dataset: DatasetHandle, min_support: float = 0.5, use_fpgrowth: bool = True) -> pd.DataFrame:
        df_onehot = self.preprocess_to_onehot(dataset)
        if df_onehot.empty:
            return pd.DataFrame(columns=["support", "itemsets"])
            
        if use_fpgrowth:
            frequent_itemsets = fpgrowth(df_onehot, min_support=min_support, use_colnames=True)
        else:
            frequent_itemsets = apriori(df_onehot, min_support=min_support, use_colnames=True)
            
        return frequent_itemsets

    def generate_rules(self, frequent_itemsets: pd.DataFrame, metric: str = "confidence", min_threshold: float = 0.5) -> pd.DataFrame:
        if frequent_itemsets.empty:
            return pd.DataFrame()
        try:
            rules = association_rules(frequent_itemsets, metric=metric, min_threshold=min_threshold)
            return rules
        except Exception:
            return pd.DataFrame()
