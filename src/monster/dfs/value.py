from __future__ import annotations

import re
import unicodedata

import polars as pl

REQUIRED_SALARY_COLUMNS = {"Position", "Player", "Team", "Salary", "FanDuel_ID"}
_SUFFIXES = {"jr", "sr", "ii", "iii", "iv", "v"}
_ALIASES = {
    "hollywood brown": "marquise brown",
    "joshua palmer": "josh palmer",
}


def _identity_key(name: str) -> str:
    text = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode().lower()
    text = re.sub(r"[^a-z0-9 ]+", " ", text)
    parts = [part for part in text.split() if part not in _SUFFIXES]
    key = " ".join(parts)
    return _ALIASES.get(key, key)


def normalize_fanduel_pool(pool: pl.DataFrame) -> pl.DataFrame:
    """Normalize FanDuel metadata while keeping it downstream of football simulation."""
    missing = REQUIRED_SALARY_COLUMNS - set(pool.columns)
    if missing:
        raise ValueError(f"FanDuel pool missing required columns: {sorted(missing)}")
    normalized = pool.select(
        pl.col("Position").cast(pl.String).str.to_uppercase().alias("position"),
        pl.col("Player").cast(pl.String).str.strip_chars().alias("fanduel_player"),
        pl.col("Team").cast(pl.String).str.to_uppercase().alias("team_id"),
        pl.col("Salary").cast(pl.Int64).alias("salary"),
        pl.col("FanDuel_ID").cast(pl.String).alias("fanduel_id"),
    ).unique(subset=["fanduel_id"])
    return normalized.with_columns(
        pl.col("fanduel_player")
        .map_elements(_identity_key, return_dtype=pl.String)
        .alias("identity_key")
    )


def attach_salary_value(
    distributions: pl.DataFrame,
    salary_pool: pl.DataFrame,
) -> pl.DataFrame:
    """Attach salary/value only after frozen football and fantasy scoring.

    Identity resolution uses team + position + a conservative normalized name key. FanDuel IDs
    must remain unique at that key; ambiguous identities fail loudly rather than entering an
    optimizer under an arbitrary player match.
    """
    pool = normalize_fanduel_pool(salary_pool)
    ambiguous = (
        pool.group_by(["position", "team_id", "identity_key"])
        .agg(pl.col("fanduel_id").n_unique().alias("ids"))
        .filter(pl.col("ids") > 1)
    )
    if ambiguous.height:
        raise ValueError("Ambiguous FanDuel player identity after normalization")

    keyed = distributions.with_columns(
        pl.col("player").map_elements(_identity_key, return_dtype=pl.String).alias("identity_key")
    )
    joined = keyed.join(pool, on=["position", "team_id", "identity_key"], how="left")
    return joined.with_columns(
        (pl.col("fd_mean") / (pl.col("salary") / 1000.0)).alias("fd_mean_per_1k"),
        (pl.col("fd_p90") / (pl.col("salary") / 1000.0)).alias("fd_p90_per_1k"),
        (pl.col("fd_p95") / (pl.col("salary") / 1000.0)).alias("fd_p95_per_1k"),
        (pl.col("fd_p99") / (pl.col("salary") / 1000.0)).alias("fd_p99_per_1k"),
    )


def salary_join_audit(frame: pl.DataFrame) -> dict[str, int]:
    offense = frame.filter(pl.col("position").is_in(["QB", "RB", "WR", "TE"]))
    matched = offense.filter(pl.col("salary").is_not_null())
    return {
        "simulated_offensive_players": offense.height,
        "salary_matched_offensive_players": matched.height,
        "salary_unmatched_offensive_players": offense.height - matched.height,
        "unique_fanduel_ids": matched.select(pl.col("fanduel_id").n_unique()).item()
        if matched.height
        else 0,
    }
