from __future__ import annotations

import builtins
import io
import contextlib
import traceback as _traceback
import tempfile
from dataclasses import replace
from datetime import datetime
from pathlib import Path

import polars as pl

from portakal_app.data.models import (
    DatasetHandle,
    SourceInfo,
    build_data_domain,
)


class PythonScriptResult:
    """Holds the result of a Python script execution."""

    def __init__(
        self,
        output_dataset: DatasetHandle | None = None,
        stdout: str = "",
        error: str = "",
    ) -> None:
        self.output_dataset = output_dataset
        self.stdout = stdout
        self.error = error


class PythonScriptService:
    def execute(
        self,
        dataset: DatasetHandle | None,
        *,
        code: str,
    ) -> PythonScriptResult:
        """Execute arbitrary Python code.

        Guarantees:
        * ``print()`` and any stdout/stderr always captured.
        * ``in_data`` is a Polars DataFrame by default.
        * If the code uses Orange API (``in_data.X``, ``in_data.domain``…),
          ``in_data`` is automatically promoted to an Orange Table.
        * ``out_data`` can be Polars, Pandas, Orange Table, dict, or list.
        * A ``PythonScriptResult`` is **always** returned – never raises.
        """
        if not code.strip():
            return PythonScriptResult(
                output_dataset=dataset, stdout="", error="No code provided."
            )

        # ── stdout / stderr capture ───────────────────────────────────
        stdout_buf = io.StringIO()
        stderr_buf = io.StringIO()

        # Custom print that is guaranteed to write to our buffer
        def _print(*args, **kwargs):
            kwargs.setdefault("file", stdout_buf)
            builtins.print(*args, **kwargs)

        in_df = dataset.dataframe if dataset else None

        # ── Namespace ─────────────────────────────────────────────────
        namespace: dict = {
            "__builtins__": builtins,
            "print": _print,
            "in_data": in_df,
            "in_datas": [in_df] if in_df is not None else [],
            "out_data": None,
            "pl": pl,
        }

        # numpy
        try:
            import numpy as np
            namespace["np"] = np
            namespace["numpy"] = np
        except ImportError:
            pass

        # pandas
        try:
            import pandas as pd
            namespace["pd"] = pd
            namespace["pandas"] = pd
        except ImportError:
            pass

        # PySide6
        try:
            from PySide6 import QtCore, QtWidgets, QtGui
            from PySide6.QtCore import Qt
            namespace.update(
                {"QtCore": QtCore, "QtWidgets": QtWidgets, "QtGui": QtGui, "Qt": Qt}
            )
        except ImportError:
            pass

        # ── Orange integration ────────────────────────────────────────
        uses_orange = any(
            token in code
            for token in (
                "in_data.X",
                "in_data.Y",
                "in_data.domain",
                "in_data.metas",
                "Orange.data",
                "from Orange",
                "import Orange",
            )
        )

        try:
            import Orange

            namespace["Orange"] = Orange
            if in_df is not None and dataset is not None and uses_orange:
                try:
                    orange_table = _polars_to_orange_table(in_df, dataset.domain)
                    namespace["in_data"] = orange_table
                    namespace["in_datas"] = [orange_table]
                except Exception as conv_err:
                    _print(f"[Warning] Orange conversion failed: {conv_err}")
                    _print("[Warning] in_data remains a Polars DataFrame\n")
        except ImportError:
            if uses_orange:
                _print("[Warning] Orange is not installed. in_data is a Polars DataFrame.\n")

        # ── Step 1: Compile ───────────────────────────────────────────
        try:
            compiled = compile(code, "<script>", "exec")
        except SyntaxError:
            return PythonScriptResult(
                output_dataset=None,
                stdout=stdout_buf.getvalue(),
                error=_traceback.format_exc(),
            )

        # ── Step 2: Execute ───────────────────────────────────────────
        error_msg = ""
        out_df = None

        try:
            with contextlib.redirect_stdout(stdout_buf), \
                 contextlib.redirect_stderr(stderr_buf):
                exec(compiled, namespace)  # noqa: S102

            out_obj = namespace.get("out_data")
            out_df = _normalise_output(out_obj)

            if out_obj is not None and out_df is None:
                error_msg = (
                    f"out_data type '{type(out_obj).__name__}' could not be "
                    f"converted to a DataFrame.\n"
                    f"Supported types: polars.DataFrame, pandas.DataFrame, "
                    f"Orange.data.Table, dict, list[dict]"
                )

        except Exception:
            error_msg = _traceback.format_exc()

        # ── Collect output text ───────────────────────────────────────
        stdout_text = stdout_buf.getvalue()
        stderr_text = stderr_buf.getvalue()
        full_stdout = stdout_text
        if stderr_text:
            full_stdout += "\n--- STDERR ---\n" + stderr_text

        if error_msg:
            return PythonScriptResult(
                output_dataset=None, stdout=full_stdout, error=error_msg
            )

        if out_df is None:
            return PythonScriptResult(
                output_dataset=None, stdout=full_stdout, error=""
            )

        # ── Build output DatasetHandle ────────────────────────────────
        try:
            if dataset is not None:
                output = replace(
                    dataset,
                    dataset_id=f"{dataset.dataset_id}-script",
                    display_name=f"{dataset.display_name} (script)",
                    dataframe=out_df,
                    row_count=out_df.height,
                    column_count=out_df.width,
                    domain=build_data_domain(out_df, source_domain=dataset.domain),
                )
            else:
                dummy_path = (
                    Path(tempfile.gettempdir())
                    / "portakal-app"
                    / "script-output.parquet"
                )
                output = DatasetHandle(
                    dataset_id="script-output",
                    display_name="Script Output",
                    source=SourceInfo(
                        path=dummy_path,
                        format="script",
                        size_bytes=0,
                        modified_at=datetime.now(),
                        cache_path=dummy_path,
                    ),
                    dataframe=out_df,
                    row_count=out_df.height,
                    column_count=out_df.width,
                    domain=build_data_domain(out_df),
                    cache_path=dummy_path,
                )
            return PythonScriptResult(
                output_dataset=output, stdout=full_stdout, error=""
            )
        except Exception as e:
            return PythonScriptResult(
                output_dataset=None,
                stdout=full_stdout,
                error=f"Error building output dataset: {e}",
            )


# ═══════════════════════════════════════════════════════════════════════════
#  Polars ↔ Orange conversion
# ═══════════════════════════════════════════════════════════════════════════


def _polars_to_orange_table(df: pl.DataFrame, domain):
    """Polars DataFrame + Portakal DataDomain → Orange.data.Table."""
    import numpy as np
    from Orange.data import (
        ContinuousVariable,
        DiscreteVariable,
        StringVariable,
        Domain,
        Table,
    )

    attributes: list = []
    class_vars: list = []
    metas_vars: list = []

    for col_schema in domain.columns:
        name = col_schema.name
        ltype = col_schema.logical_type
        role = col_schema.role

        if ltype == "numeric":
            var = ContinuousVariable(name)
        elif ltype in ("categorical", "boolean"):
            unique_vals = (
                df.get_column(name)
                .drop_nulls()
                .cast(pl.Utf8)
                .unique()
                .sort()
                .to_list()
            )
            var = DiscreteVariable(name, values=unique_vals)
        else:
            var = StringVariable(name)

        if role == "target":
            class_vars.append(var)
        elif role == "meta":
            metas_vars.append(var)
        else:
            attributes.append(var)

    orange_domain = Domain(attributes, class_vars, metas_vars)
    n = df.height

    # X (features)
    if attributes:
        X = np.full((n, len(attributes)), np.nan)
        for j, var in enumerate(attributes):
            _fill_column(X, j, var, df.get_column(var.name))
    else:
        X = np.zeros((n, 0))

    # Y (class)
    if class_vars:
        if len(class_vars) == 1:
            Y = np.full(n, np.nan)
            _fill_column_1d(Y, class_vars[0], df.get_column(class_vars[0].name))
        else:
            Y = np.full((n, len(class_vars)), np.nan)
            for j, var in enumerate(class_vars):
                _fill_column(Y, j, var, df.get_column(var.name))
    else:
        Y = None

    # Metas
    if metas_vars:
        metas_arr = np.empty((n, len(metas_vars)), dtype=object)
        for j, var in enumerate(metas_vars):
            col = df.get_column(var.name)
            if isinstance(var, StringVariable):
                metas_arr[:, j] = (
                    col.cast(pl.Utf8, strict=False).fill_null("").to_list()
                )
            elif isinstance(var, ContinuousVariable):
                metas_arr[:, j] = (
                    col.cast(pl.Float64, strict=False)
                    .fill_null(float("nan"))
                    .to_numpy()
                )
            elif isinstance(var, DiscreteVariable):
                str_vals = (
                    col.cast(pl.Utf8, strict=False).fill_null("").to_list()
                )
                metas_arr[:, j] = [
                    var.values.index(v) if v in var.values else np.nan
                    for v in str_vals
                ]
    else:
        metas_arr = None

    return Table.from_numpy(orange_domain, X, Y, metas_arr)


def _fill_column(matrix, j, var, series):
    """Fill column *j* of a 2-D numpy array from a Polars Series."""
    import numpy as np
    from Orange.data import ContinuousVariable, DiscreteVariable

    if isinstance(var, ContinuousVariable):
        matrix[:, j] = (
            series.cast(pl.Float64, strict=False)
            .fill_null(float("nan"))
            .to_numpy()
        )
    elif isinstance(var, DiscreteVariable):
        str_vals = series.cast(pl.Utf8, strict=False).fill_null("").to_list()
        matrix[:, j] = [
            var.values.index(v) if v in var.values else np.nan
            for v in str_vals
        ]


def _fill_column_1d(array, var, series):
    """Fill a 1-D numpy array from a Polars Series."""
    import numpy as np
    from Orange.data import ContinuousVariable, DiscreteVariable

    if isinstance(var, ContinuousVariable):
        array[:] = (
            series.cast(pl.Float64, strict=False)
            .fill_null(float("nan"))
            .to_numpy()
        )
    elif isinstance(var, DiscreteVariable):
        str_vals = series.cast(pl.Utf8, strict=False).fill_null("").to_list()
        for i, v in enumerate(str_vals):
            array[i] = var.values.index(v) if v in var.values else np.nan


# ── Orange Table → Polars ─────────────────────────────────────────────────


def _orange_table_to_polars(table) -> pl.DataFrame:
    """Orange.data.Table → Polars DataFrame."""
    import numpy as np
    from Orange.data import ContinuousVariable, DiscreteVariable, StringVariable

    data: dict[str, list] = {}

    # Attributes
    for j, var in enumerate(table.domain.attributes):
        col = table.X[:, j]
        if isinstance(var, DiscreteVariable):
            data[var.name] = [
                var.values[int(v)] if not np.isnan(v) else None for v in col
            ]
        else:
            data[var.name] = col.tolist()

    # Class
    if table.domain.class_var is not None:
        var = table.domain.class_var
        y = table.Y
        if y.ndim == 1:
            if isinstance(var, DiscreteVariable):
                data[var.name] = [
                    var.values[int(v)] if not np.isnan(v) else None for v in y
                ]
            else:
                data[var.name] = y.tolist()
    elif table.domain.class_vars:
        for j, var in enumerate(table.domain.class_vars):
            col = table.Y[:, j]
            if isinstance(var, DiscreteVariable):
                data[var.name] = [
                    var.values[int(v)] if not np.isnan(v) else None for v in col
                ]
            else:
                data[var.name] = col.tolist()

    # Metas
    if table.domain.metas:
        for j, var in enumerate(table.domain.metas):
            col = table.metas[:, j]
            if isinstance(var, StringVariable):
                data[var.name] = [
                    str(v) if v is not None and str(v) != "nan" else None
                    for v in col
                ]
            elif isinstance(var, DiscreteVariable):
                data[var.name] = _decode_discrete_meta(var, col)
            else:
                data[var.name] = [
                    float(v) if v is not None else None for v in col
                ]

    return pl.DataFrame(data)


def _decode_discrete_meta(var, col):
    """Safely decode discrete meta column."""
    import numpy as np

    result = []
    for v in col:
        try:
            fv = float(v)
            if np.isnan(fv):
                result.append(None)
            else:
                result.append(var.values[int(fv)])
        except (ValueError, TypeError, IndexError):
            result.append(None)
    return result


# ── Normalise any output type → Polars ────────────────────────────────────


def _normalise_output(out_obj):
    """Accept out_data as Polars, Pandas, Orange, dict, or list → Polars."""
    if out_obj is None:
        return None

    # Already Polars
    if isinstance(out_obj, pl.DataFrame):
        return out_obj

    # Orange Table
    try:
        import Orange

        if isinstance(out_obj, Orange.data.Table):
            return _orange_table_to_polars(out_obj)
    except ImportError:
        pass

    # Pandas DataFrame
    try:
        import pandas as pd

        if isinstance(out_obj, pd.DataFrame):
            return pl.from_pandas(out_obj)
    except (ImportError, Exception):
        pass

    # dict → DataFrame
    if isinstance(out_obj, dict):
        try:
            return pl.DataFrame(out_obj)
        except Exception:
            pass

    # list[dict] → DataFrame
    if isinstance(out_obj, list) and out_obj and isinstance(out_obj[0], dict):
        try:
            return pl.DataFrame(out_obj)
        except Exception:
            pass

    # Generic .to_pandas()
    if hasattr(out_obj, "to_pandas"):
        try:
            return pl.from_pandas(out_obj.to_pandas())
        except Exception:
            pass

    return None
