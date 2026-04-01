from __future__ import annotations

import math
from dataclasses import replace

import polars as pl

from portakal_app.data.models import DatasetHandle, build_data_domain

_SAFE_BUILTINS = {
    "abs": abs,
    "round": round,
    "min": min,
    "max": max,
    "sum": sum,
    "len": len,
    "int": int,
    "float": float,
    "str": str,
    "bool": bool,
}

_SAFE_MATH = {
    "sqrt": math.sqrt,
    "log": math.log,
    "log2": math.log2,
    "log10": math.log10,
    "exp": math.exp,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "pi": math.pi,
    "e": math.e,
    "ceil": math.ceil,
    "floor": math.floor,
    "pow": math.pow,
}


class FormulaService:
    def apply_formulas(
        self,
        dataset: DatasetHandle,
        *,
        formulas: list[dict[str, object]],
    ) -> DatasetHandle:
        df = dataset.dataframe

        if not formulas:
            return dataset

        for formula in formulas:
            if isinstance(formula, (tuple, list)):
                col_name = str(formula[0]).strip() if len(formula) > 0 else ""
                expr_str = str(formula[1]).strip() if len(formula) > 1 else ""
                var_type = "numeric"
            else:
                col_name = str(formula.get("name", "")).strip()
                expr_str = str(formula.get("expr", "")).strip()
                var_type = str(formula.get("type", "numeric"))

            if not col_name or not expr_str:
                continue
            try:
                expr = _parse_formula(expr_str, df.columns)
                df = df.with_columns(expr.alias(col_name))
                # Cast column to the user-chosen type
                if var_type == "numeric":
                    df = df.with_columns(pl.col(col_name).cast(pl.Float64, strict=False))
                elif var_type in ("text", "categorical"):
                    df = df.with_columns(pl.col(col_name).cast(pl.Utf8, strict=False))
                elif var_type == "datetime":
                    try:
                        df = df.with_columns(pl.col(col_name).str.to_datetime(strict=False))
                    except Exception:
                        pass  # keep as-is if datetime parse fails
            except Exception:
                df = df.with_columns(pl.lit(None).alias(col_name))

        return replace(
            dataset,
            dataset_id=f"{dataset.dataset_id}-formula",
            display_name=f"{dataset.display_name} (formula)",
            dataframe=df,
            row_count=df.height,
            column_count=df.width,
            domain=build_data_domain(df, source_domain=dataset.domain),
        )


def _parse_formula(expr_str: str, columns: list[str]) -> pl.Expr:
    namespace: dict = {}
    namespace.update(_SAFE_BUILTINS)
    namespace.update(_SAFE_MATH)

    for col_name in columns:
        safe_name = col_name.replace(" ", "_").replace("-", "_")
        namespace[safe_name] = pl.col(col_name)
        if safe_name != col_name:
            namespace[col_name] = pl.col(col_name)

    namespace["col"] = pl.col
    namespace["lit"] = pl.lit
    namespace["when"] = pl.when

    namespace["__builtins__"] = {}

    result = eval(expr_str, namespace)  # noqa: S307
    if isinstance(result, pl.Expr):
        return result
    return pl.lit(result)
