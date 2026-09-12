from __future__ import annotations

import json

import polars as pl


def csv_safe_frame(frame: pl.DataFrame) -> pl.DataFrame:
    """Serialize nested columns for CSV views while leaving canonical frames untouched."""
    nested = {pl.List, pl.Array, pl.Struct, pl.Object}
    expressions: list[pl.Expr] = []
    for name, dtype in frame.schema.items():
        if dtype.base_type() in nested:
            expressions.append(
                pl.col(name)
                .map_elements(
                    lambda value: json.dumps(value, default=str) if value is not None else None,
                    return_dtype=pl.Utf8,
                )
                .alias(name)
            )
    return frame.with_columns(expressions) if expressions else frame


def write_csv_safe(frame: pl.DataFrame, path: object) -> None:
    """Write a CSV-safe view of a potentially rich/nested dataframe."""
    csv_safe_frame(frame).write_csv(path)
