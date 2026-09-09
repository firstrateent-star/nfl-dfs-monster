from __future__ import annotations

import polars as pl

REQUIRED_SALARY_COLUMNS = {"Position", "Player", "Team", "Salary", "FanDuel_ID"}


def normalize_fanduel_pool(pool: pl.DataFrame) -> pl.DataFrame:
    """Normalize a FanDuel salary pool without allowing it into football simulation state."""
    missing = REQUIRED_SALARY_COLUMNS - set(pool.columns)
    if missing:
        raise ValueError(f"FanDuel pool missing required columns: {sorted(missing)}")
    return pool.select(
        pl.col("Position").cast(pl.String).str.to_uppercase().alias("position"),
        pl.col("Player").cast(pl.String).str.strip_chars().alias("player"),
        pl.col("Team").cast(pl.String).str.to_uppercase().alias("team_id"),
        pl.col("Salary").cast(pl.Int64).alias("salary"),
        pl.col("FanDuel_ID").cast(pl.String).alias("fanduel_id"),
    ).unique(subset=["fanduel_id"])


def attach_salary_value(
    distributions: pl.DataFrame,
    salary_pool: pl.DataFrame,
) -> pl.DataFrame:
    """Join frozen football/fantasy distributions to FanDuel only after simulation.

    Matching is deliberately on display name + team + position. Unmatched rows remain visible so
    slate identity problems cannot silently become optimizer inputs.
    """
    pool = normalize_fanduel_pool(salary_pool)
    joined = distributions.join(pool, on=["position", "player", "team_id"], how="left")
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
