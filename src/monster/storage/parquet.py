from __future__ import annotations
from pathlib import Path
import duckdb


def compact_parquet(input_parquet: str | Path, output_parquet: str | Path) -> None:
    """Rewrite Parquet with ZSTD; DuckDB reads only needed columns/row groups later."""
    con = duckdb.connect()
    con.execute(
        "COPY (SELECT * FROM read_parquet(?)) TO ? (FORMAT PARQUET, COMPRESSION ZSTD)",
        [str(input_parquet), str(output_parquet)],
    )
    con.close()
