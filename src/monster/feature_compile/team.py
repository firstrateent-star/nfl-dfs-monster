from __future__ import annotations
import polars as pl


def compile_team_policy(pbp: pl.DataFrame) -> pl.DataFrame:
    """Build situation-aware policy priors from football behavior, not market totals."""
    required = {"posteam", "play_type", "score_differential", "game_seconds_remaining"}
    missing = required.difference(pbp.columns)
    if missing:
        raise ValueError(f"PBP missing required columns: {sorted(missing)}")

    neutral = pbp.filter(
        (pl.col("score_differential").abs() <= 7)
        & (pl.col("game_seconds_remaining") > 900)
        & pl.col("play_type").is_in(["pass", "run"])
    )
    return (
        neutral.with_columns((pl.col("play_type") == "pass").cast(pl.Float64).alias("is_pass"))
        .group_by("posteam")
        .agg(
            pl.len().alias("neutral_plays"),
            pl.col("is_pass").mean().alias("neutral_pass_rate"),
        )
        .rename({"posteam": "team_id"})
    )
