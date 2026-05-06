from __future__ import annotations

import csv
from pathlib import Path

import polars as pl

from portakal_app.data.services.distance_matrix_service import DistanceMatrixHandle, build_distance_matrix


class DistanceFileService:
    def load(self, path: str, *, treat_triangular_as_symmetric: bool = True) -> DistanceMatrixHandle:
        file_path = Path(path).expanduser()
        suffix = file_path.suffix.lower()
        rows = self._read_rows(file_path, suffix)
        matrix_rows, row_labels, column_labels = self._split_labels(rows)
        matrix = self._build_square_matrix(matrix_rows, treat_triangular_as_symmetric=treat_triangular_as_symmetric)
        labels = row_labels or column_labels or tuple(str(index + 1) for index in range(matrix.shape[0]))
        if len(labels) != int(matrix.shape[0]):
            labels = tuple(str(index + 1) for index in range(matrix.shape[0]))
        return build_distance_matrix(
            matrix,
            metric="precomputed",
            metric_label="Precomputed",
            axis="rows",
            axis_label="Distances between rows",
            row_labels=labels,
            feature_names=labels,
            metadata={
                "source_path": str(file_path),
                "treat_triangular_as_symmetric": treat_triangular_as_symmetric,
            },
        )

    def _read_rows(self, path: Path, suffix: str) -> list[list[str | None]]:
        if suffix in {".csv", ".tsv", ".txt"}:
            delimiter = "\t" if suffix in {".tsv", ".txt"} else ","
            with path.open("r", encoding="utf-8-sig", newline="") as handle:
                reader = csv.reader(handle, delimiter=delimiter)
                return [self._normalize_row(row) for row in reader]
        if suffix == ".xlsx":
            frame = pl.read_excel(path, has_header=False)
            return [self._normalize_row(list(row)) for row in frame.rows()]
        raise ValueError(f"Unsupported distance file format: {suffix or 'unknown'}")

    def _normalize_row(self, row: list[object]) -> list[str | None]:
        normalized: list[str | None] = []
        for cell in row:
            if cell is None:
                normalized.append(None)
                continue
            text = str(cell).strip()
            normalized.append(text or None)
        while normalized and normalized[-1] is None:
            normalized.pop()
        return normalized

    def _split_labels(
        self,
        rows: list[list[str | None]],
    ) -> tuple[list[list[str | None]], tuple[str, ...] | None, tuple[str, ...] | None]:
        cleaned = [row for row in rows if any(cell is not None for cell in row)]
        if not cleaned:
            raise ValueError("Distance file is empty.")

        has_header_row = len(cleaned[0]) > 1 and all(not self._is_number(cell) for cell in cleaned[0][1:] if cell is not None)
        has_header_col = len(cleaned) > 1 and all(
            len(row) > 0 and row[0] is not None and not self._is_number(row[0])
            for row in cleaned[1 if has_header_row else 0 :]
        )

        column_labels: tuple[str, ...] | None = None
        if has_header_row:
            column_labels = tuple(str(cell) for cell in cleaned[0][1 if has_header_col else 0 :] if cell is not None)
            cleaned = cleaned[1:]

        row_labels: tuple[str, ...] | None = None
        if has_header_col:
            row_labels = tuple(str(row[0]) for row in cleaned if row and row[0] is not None)
            cleaned = [row[1:] for row in cleaned]

        return cleaned, row_labels, column_labels

    def _build_square_matrix(
        self,
        rows: list[list[str | None]],
        *,
        treat_triangular_as_symmetric: bool,
    ):
        counts = [sum(1 for cell in row if cell is not None) for row in rows]
        if counts == list(range(1, len(rows) + 1)):
            return self._expand_triangular(rows, lower=True, symmetric=treat_triangular_as_symmetric)
        if counts == list(range(len(rows), 0, -1)):
            return self._expand_triangular(rows, lower=False, symmetric=treat_triangular_as_symmetric)

        width = max((len(row) for row in rows), default=0)
        if width != len(rows):
            raise ValueError("Distance Matrix must be square or triangular.")
        matrix = []
        for row in rows:
            padded = list(row) + [None] * (width - len(row))
            matrix.append([0.0 if cell is None else float(cell) for cell in padded])
        values = pl.DataFrame(matrix).to_numpy()
        if treat_triangular_as_symmetric:
            values = self._symmetrize(values)
        return values

    def _expand_triangular(self, rows: list[list[str | None]], *, lower: bool, symmetric: bool):
        size = len(rows)
        matrix = [[0.0 for _ in range(size)] for _ in range(size)]
        for row_index, row in enumerate(rows):
            values = [0.0 if cell is None else float(cell) for cell in row if cell is not None]
            for offset, value in enumerate(values):
                col_index = offset if lower else row_index + offset
                if lower:
                    matrix[row_index][col_index] = value
                    if symmetric:
                        matrix[col_index][row_index] = value
                else:
                    matrix[row_index][col_index] = value
                    if symmetric:
                        matrix[col_index][row_index] = value
        return pl.DataFrame(matrix).to_numpy()

    def _symmetrize(self, values):
        matrix = values.astype(float, copy=True)
        for row_index in range(matrix.shape[0]):
            for col_index in range(matrix.shape[1]):
                if row_index == col_index:
                    matrix[row_index, col_index] = 0.0
                    continue
                left = matrix[row_index, col_index]
                right = matrix[col_index, row_index]
                if left == 0.0 and right != 0.0:
                    matrix[row_index, col_index] = right
                elif right == 0.0 and left != 0.0:
                    matrix[col_index, row_index] = left
        return matrix

    def _is_number(self, value: str | None) -> bool:
        if value is None:
            return False
        try:
            float(value)
        except (TypeError, ValueError):
            return False
        return True

