from __future__ import annotations

import polars as pl

from monster.teams import TEAM_ALIASES

_CONTEXT_KEYS = ["down", "distance_bucket", "field_zone", "time_mode", "score_state"]


def distance_bucket_expr() -> pl.Expr:
    return (
        pl.when(pl.col("ydstogo") <= 1)
        .then(pl.lit("1"))
        .when(pl.col("ydstogo") == 2)
        .then(pl.lit("2"))
        .when(pl.col("ydstogo") <= 4)
        .then(pl.lit("3_4"))
        .when(pl.col("ydstogo") <= 7)
        .then(pl.lit("5_7"))
        .when(pl.col("ydstogo") <= 10)
        .then(pl.lit("8_10"))
        .when(pl.col("ydstogo") <= 15)
        .then(pl.lit("11_15"))
        .otherwise(pl.lit("16_plus"))
    )


def field_zone_expr() -> pl.Expr:
    # Historical nflverse `yardline_100` is distance to the opponent goal line.
    return (
        pl.when(pl.col("yardline_100") >= 80)
        .then(pl.lit("backed_up"))
        .when(pl.col("yardline_100") >= 60)
        .then(pl.lit("own_field"))
        .when(pl.col("yardline_100") >= 40)
        .then(pl.lit("midfield"))
        .when(pl.col("yardline_100") > 20)
        .then(pl.lit("opp_territory"))
        .when(pl.col("yardline_100") > 10)
        .then(pl.lit("high_red_zone"))
        .otherwise(pl.lit("low_red_zone"))
    )


def score_state_expr() -> pl.Expr:
    diff = pl.col("score_differential")
    return (
        pl.when(diff <= -9)
        .then(pl.lit("trail_9_plus"))
        .when(diff < 0)
        .then(pl.lit("trail_1_8"))
        .when(diff == 0)
        .then(pl.lit("tied"))
        .when(diff <= 8)
        .then(pl.lit("lead_1_8"))
        .otherwise(pl.lit("lead_9_plus"))
    )


def time_mode_expr() -> pl.Expr:
    qtr = pl.col("qtr")
    game_left = pl.col("game_seconds_remaining")
    half_left = pl.when(qtr <= 2).then(game_left - 1800).otherwise(game_left)
    return (
        pl.when((qtr == 2) & (half_left <= 120))
        .then(pl.lit("two_minute_first_half"))
        .when((qtr == 4) & (game_left <= 120))
        .then(pl.lit("two_minute_game"))
        .when((qtr == 4) & (game_left <= 240))
        .then(pl.lit("four_minute_game"))
        .when(qtr == 1)
        .then(pl.lit("q1_normal"))
        .when(qtr == 2)
        .then(pl.lit("q2_normal"))
        .when(qtr == 3)
        .then(pl.lit("q3_normal"))
        .when(qtr == 4)
        .then(pl.lit("q4_normal"))
        .otherwise(pl.lit("overtime"))
    )


def prepare_game_flow_plays(pbp: pl.DataFrame) -> pl.DataFrame:
    """Return definition-safe run/dropback decisions with fine game-flow context."""

    required = {
        "posteam",
        "qb_dropback",
        "rush_attempt",
        "down",
        "ydstogo",
        "yardline_100",
        "qtr",
        "game_seconds_remaining",
        "score_differential",
    }
    missing = required.difference(pbp.columns)
    if missing:
        raise ValueError(f"PBP missing game-flow policy columns: {sorted(missing)}")

    dropback = pl.col("qb_dropback").fill_null(0).cast(pl.Float64) == 1.0
    designed_run = (pl.col("rush_attempt").fill_null(0).cast(pl.Float64) == 1.0) & ~dropback
    filters = (
        pl.col("posteam").is_not_null()
        & pl.col("down").is_between(1, 4)
        & pl.col("ydstogo").is_not_null()
        & pl.col("yardline_100").is_not_null()
        & pl.col("qtr").is_not_null()
        & pl.col("game_seconds_remaining").is_not_null()
        & pl.col("score_differential").is_not_null()
        & (dropback | designed_run)
    )
    if "qb_kneel" in pbp.columns:
        filters &= pl.col("qb_kneel").fill_null(0) == 0
    if "qb_spike" in pbp.columns:
        filters &= pl.col("qb_spike").fill_null(0) == 0

    return (
        pbp.with_columns(pl.col("posteam").replace(TEAM_ALIASES))
        .filter(filters)
        .with_columns(
            dropback.cast(pl.Float64).alias("is_dropback"),
            distance_bucket_expr().alias("distance_bucket"),
            field_zone_expr().alias("field_zone"),
            time_mode_expr().alias("time_mode"),
            score_state_expr().alias("score_state"),
        )
    )


def compile_game_flow_policy(pbp: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Compile league and team/coach-style game-flow decision evidence.

    No shrinkage is applied here. Raw sample size travels with every cell so the runtime
    can shrink team evidence toward the matching league state instead of mistaking sparse
    history for certainty.
    """

    frame = prepare_game_flow_plays(pbp)
    if not frame.height:
        raise ValueError("no game-flow scrimmage decisions available")

    league = (
        frame.group_by(_CONTEXT_KEYS)
        .agg(
            pl.len().alias("samples"),
            pl.col("is_dropback").mean().alias("dropback_rate"),
        )
        .sort(_CONTEXT_KEYS)
    )
    team = (
        frame.group_by(["posteam", *_CONTEXT_KEYS])
        .agg(
            pl.len().alias("samples"),
            pl.col("is_dropback").mean().alias("dropback_rate"),
        )
        .rename({"posteam": "team_id"})
        .sort(["team_id", *_CONTEXT_KEYS])
    )
    return league, team
