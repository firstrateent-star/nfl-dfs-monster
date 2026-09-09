from __future__ import annotations

import polars as pl

from monster.teams import TEAM_ALIASES

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
    """Compile league dropback-family propensity by down/distance in neutral states.

    nflfastR labels scrambles as ``play_type='run'`` even though they begin as quarterback
    dropbacks. Monster's PASS state is the causal family before the QB resolves pressure,
    so historical pass-family identity must use ``qb_dropback`` (attempt, sack, or scramble)
    rather than the terminal play_type label.

    Team identity is intentionally excluded here. The simulator combines this large-sample
    league context with a continuity-conditioned team neutral-pass deviation downstream.
    """
    required = {
        "posteam",
        "play_type",
        "qb_dropback",
        "down",
        "ydstogo",
        "score_differential",
        "game_seconds_remaining",
    }
    missing = required.difference(pbp.columns)
    if missing:
        raise ValueError(f"PBP missing situational-pass columns: {sorted(missing)}")

    dropback = pl.col("qb_dropback").fill_null(0).cast(pl.Float64) == 1.0
    designed_run = (pl.col("play_type") == "run") & ~dropback
    neutral = (
        pbp.with_columns(pl.col("posteam").replace(TEAM_ALIASES))
        .filter(
            pl.col("posteam").is_not_null()
            & (dropback | designed_run)
            & pl.col("down").is_between(1, 4)
            & pl.col("ydstogo").is_not_null()
            & (pl.col("score_differential").abs() <= 7)
            & (pl.col("game_seconds_remaining") > 900)
        )
        .with_columns(
            pl.col("down").cast(pl.Int64),
            dropback.cast(pl.Float64).alias("is_pass"),
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
