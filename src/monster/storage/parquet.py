from __future__ import annotations

from pathlib import Path

import duckdb


def summarize_world_file(path: str | Path) -> dict:
    path = str(path)
    con = duckdb.connect()
    try:
        row = con.execute(
            "select count(*) as rows, count(*) filter (where 1=1) as observed from read_parquet(?)",
            [path],
        ).fetchone()
        return {"rows": row[0], "observed": row[1]}
    finally:
        con.close()
