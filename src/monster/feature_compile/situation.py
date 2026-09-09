from __future__ import annotations

import polars as pl

from monster.teams import TEAM_ALIASES

SCRIMMAGE_TYPES = ["pass", "run"]
DISTANCE_BUCKETS = ("short", "medium", "long")


def _distance_bucket() -> pl.Expr:
    return (
        pl.when(pl.col("ydstogo") <= 3)
        .then(pl.lit("short"))
        .when(pl.col("ydstogo") <= 7)
        .then(pl.lit("medium"))
        .otherwise(pl.lit("long"))
    )


def compile_situational_pass_context(pbp: pl.DataFrame) -> pl.DataFrame:
    """Compile league pass propensity by down/distance from neutral game states.

    Team identity is intentionally excluded here. The simulator combines this large-sample
    league context with a continuity-conditioned team neutral-pass deviation downstream.
    This prevents normal down/distance behavior from being added twice to a historical
    neutral pass rate that already contains the league's ordinary situational mixture.
    """
    required = {
        "posteam",
        "play_type",
        "down",
        "ydstogo",
        "score_differential",
        "game_seconds_remaining",
    }
    missing = required.difference(pbp.columns)
    if missing:
        raise ValueError(f"PBP missing situational-pass columns: {sorted(missing)}")

    neutral = (
        pbp.with_columns(pl.col("posteam").replace(TEAM_ALIASES))
        .filter(
            pl.col("posteam").is_not_null()
            & pl.col("play_type").is_in(SCRIMMAGE_TYPES)
            & pl.col("down").is_between(1, 4)
            & pl.col("ydstogo").is_not_null()
            & (pl.col("score_differential").abs() <= 7)
            & (pl.col("game_seconds_remaining") > 900)
        )
        .with_columns(
            pl.col("down").cast(pl.Int64),
            (pl.col("play_type") == "pass").cast(pl.Float64).alias("is_pass"),
            _distance_bucket().alias("distance_bucket"),
        )
    )
    if not neutral.height:
        raise ValueError("no neutral scrimmage plays available for situational pass context")

    league_neutral_pass_rate = float(neutral.get_column("is_pass").mean())
    context = (
        neutral.group_by(["down", "distance_bucket"])
        .agg(
            pl.len().alias("samples"),
            pl.col("is_pass").mean().alias("pass_rate"),
        )
        .with_columns(pl.lit(league_neutral_pass_rate).alias("league_neutral_pass_rate"))
    )

    # Guarantee all 12 cells. Sparse historical cells fall back to the large-sample
    # league-neutral rate rather than an invented coefficient.
    scaffold = pl.DataFrame(
        {
            "down": [down for down in range(1, 5) for _ in DISTANCE_BUCKETS],
            "distance_bucket": list(DISTANCE_BUCKETS) * 4,
        }
    )
    return (
        scaffold.join(context, on=["down", "distance_bucket"], how="left")
        .with_columns(
            pl.col("samples").fill_null(0),
            pl.col("pass_rate").fill_null(league_neutral_pass_rate),
            pl.col("league_neutral_pass_rate").fill_null(league_neutral_pass_rate),
        )
        .sort(["down", "distance_bucket"])
    )
