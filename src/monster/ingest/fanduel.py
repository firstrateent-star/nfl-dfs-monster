from __future__ import annotations

from pathlib import Path

import polars as pl

# FanDuel data is downstream DFS data. This loader must not be imported by market-blind simulation modules.


def load_fanduel_csv(path: str | Path) -> pl.DataFrame:
    return pl.read_csv(path, infer_schema_length=10_000)
