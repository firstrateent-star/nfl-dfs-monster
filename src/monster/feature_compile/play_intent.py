from __future__ import annotations

import polars as pl

from monster.feature_compile.game_flow_policy import (
    distance_bucket_expr,
    field_zone_expr,
    score_state_expr,
    time_mode_expr,
)
from monster.teams import TEAM_ALIASES

_CONTEXT_KEYS = ["down", "distance_bucket", "field_zone", "time_mode", "score_state"]
PASS_DEPTH_CATEGORIES = (
    "behind_los",
    "short_0_5",
    "short_6_9",
    "intermediate_10_19",
    "deep_20_39",
    "bomb_40_plus",
)
RUN_GEOMETRY_CATEGORIES = (
    "interior",
    "left_offtackle",
    "right_offtackle",
    "left_edge",
    "right_edge",
    "qb_sneak",
    "other",
)


def pass_depth_category_expr() -> pl.Expr:
    air = pl.col("air_yards")
    return (
        pl.when(air < 0)
        .then(pl.lit("behind_los"))
        .when(air <= 5)
        .then(pl.lit("short_0_5"))
        .when(air <= 9)
        .then(pl.lit("short_6_9"))
        .when(air <= 19)
        .then(pl.lit("intermediate_10_19"))
        .when(air <= 39)
        .then(pl.lit("deep_20_39"))
        .otherwise(pl.lit("bomb_40_plus"))
    )


def run_geometry_category_expr() -> pl.Expr:
    location = pl.col("run_location").fill_null("unknown").str.to_lowercase()
    gap = pl.col("run_gap").fill_null("unknown").str.to_lowercase()
    sneak = (
        pl.col("qb_sneak").fill_null(0).cast(pl.Float64) == 1.0
        if "qb_sneak" else pl.lit(False)
    )
    return (
        pl.when(sneak)
        .then(pl.lit("qb_sneak"))
        .when(location == "middle")
        .then(pl.lit("interior"))
        .when((location == "left") & gap.is_in(["guard", "unknown"]))
        .then(pl.lit("interior"))
        .when((location == "right") & gap.is_in(["guard", "unknown"]))
        .then(pl.lit("interior"))
        .when((location == "left") & (gap == "tackle"))
        .then(pl.lit("left_offtackle"))
        .when((location == "right") & (gap == "tackle"))
        .then(pl.lit("right_offtackle"))
        .when((location == "left") & (gap == "end"))
        .then(pl.lit("left_edge"))
        .when((location == "right") & (gap == "end"))
        .then(pl.lit("right_edge"))
        .otherwise(pl.lit("other"))
    )


def _context_columns() -> list[pl.Expr]:
    return [
        distance_bucket_expr().alias("distance_bucket"),
        field_zone_expr().alias("field_zone"),
        time_mode_expr().alias("time_mode"),
        score_state_expr().alias("score_state"),
    ]


def prepare_pass_intent_plays(pbp: pl.DataFrame) -> pl.DataFrame:
    required = {
        "posteam",
        "qb_dropback",
        "sack",
        "qb_scramble",
        "air_yards",
        "down",
        "ydstogo",
        "yardline_100",
        "qtr",
        "game_seconds_remaining",
        "score_differential",
    }
    missing = required.difference(pbp.columns)
    if missing:
        raise ValueError(f"PBP missing pass-intent columns: {sorted(missing)}")

    frame = pbp.with_columns(pl.col("posteam").replace(TEAM_ALIASES)).filter(
        pl.col("posteam").is_not_null()
        & (pl.col("qb_dropback").fill_null(0).cast(pl.Float64) == 1.0)
        & (pl.col("sack").fill_null(0).cast(pl.Float64) != 1.0)
        & (pl.col("qb_scramble").fill_null(0).cast(pl.Float64) != 1.0)
        & pl.col("air_yards").is_not_null()
        & pl.col("down").is_between(1, 4)
        & pl.col("ydstogo").is_not_null()
        & pl.col("yardline_100").is_not_null()
        & pl.col("qtr").is_not_null()
        & pl.col("game_seconds_remaining").is_not_null()
        & pl.col("score_differential").is_not_null()
    )
    return frame.with_columns(
        *_context_columns(),
        pass_depth_category_expr().alias("category"),
    )


def prepare_run_intent_plays(pbp: pl.DataFrame) -> pl.DataFrame:
    required = {
        "posteam",
        "rush_attempt",
        "qb_dropback",
        "down",
        "ydstogo",
        "yardline_100",
        "qtr",
        "game_seconds_remaining",
        "score_differential",
    }
    missing = required.difference(pbp.columns)
    if missing:
        raise ValueError(f"PBP missing run-intent columns: {sorted(missing)}")

    frame = pbp.with_columns(pl.col("posteam").replace(TEAM_ALIASES)).filter(
        pl.col("posteam").is_not_null()
        & (pl.col("rush_attempt").fill_null(0).cast(pl.Float64) == 1.0)
        & (pl.col("qb_dropback").fill_null(0).cast(pl.Float64) != 1.0)
        & pl.col("down").is_between(1, 4)
        & pl.col("ydstogo").is_not_null()
        & pl.col("yardline_100").is_not_null()
        & pl.col("qtr").is_not_null()
        & pl.col("game_seconds_remaining").is_not_null()
        & pl.col("score_differential").is_not_null()
    )
    if "qb_kneel" in frame.columns:
        frame = frame.filter(pl.col("qb_kneel").fill_null(0) == 0)
    if "qb_spike" in frame.columns:
        frame = frame.filter(pl.col("qb_spike").fill_null(0) == 0)
    if "run_location" not in frame.columns:
        frame = frame.with_columns(pl.lit(None, dtype=pl.String).alias("run_location"))
    if "run_gap" not in frame.columns:
        frame = frame.with_columns(pl.lit(None, dtype=pl.String).alias("run_gap"))
    if "qb_sneak" not in frame.columns:
        frame = frame.with_columns(pl.lit(0.0).alias("qb_sneak"))
    return frame.with_columns(
        *_context_columns(),
        run_geometry_category_expr().alias("category"),
    )


def _categorical_context_counts(frame: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    league = (
        frame.group_by([*_CONTEXT_KEYS, "category"])
        .agg(pl.len().alias("attempts"))
        .sort([*_CONTEXT_KEYS, "category"])
    )
    team = (
        frame.group_by(["posteam", *_CONTEXT_KEYS, "category"])
        .agg(pl.len().alias("attempts"))
        .rename({"posteam": "team_id"})
        .sort(["team_id", *_CONTEXT_KEYS, "category"])
    )
    return league, team


def compile_pass_intent_policy(
    pbp: pl.DataFrame,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Compile depth choice, QB identity, depth outcomes, and target-by-depth evidence."""

    frame = prepare_pass_intent_plays(pbp)
    league, team = _categorical_context_counts(frame)
    if "passer_player_id" in frame.columns:
        qb = (
            frame.filter(pl.col("passer_player_id").is_not_null())
            .group_by(["passer_player_id", "category"])
            .agg(pl.len().alias("attempts"))
            .rename({"passer_player_id": "actor_id"})
            .sort(["actor_id", "category"])
        )
    else:
        qb = pl.DataFrame(schema={"actor_id": pl.String, "category": pl.String, "attempts": pl.Int64})

    completions = pl.col("complete_pass").fill_null(0).cast(pl.Float64)
    interceptions = pl.col("interception").fill_null(0).cast(pl.Float64)
    pass_tds = pl.col("pass_touchdown").fill_null(0).cast(pl.Float64)
    yards = pl.col("yards_gained").cast(pl.Float64)
    yac = pl.col("yards_after_catch").cast(pl.Float64)
    completion_yards = yards.filter(completions == 1.0)
    forty_plus_completion_yards = yards.filter((completions == 1.0) & (yards >= 40.0))
    outcomes = (
        frame.group_by("category")
        .agg(
            pl.len().alias("attempts"),
            completions.mean().alias("completion_rate"),
            interceptions.mean().alias("interception_rate"),
            pass_tds.mean().alias("touchdown_rate"),
            pl.col("air_yards").mean().alias("air_yards_mean"),
            pl.col("air_yards").std().fill_null(0.0).alias("air_yards_sd"),
            yards.mean().alias("yards_per_attempt"),
            completion_yards.mean().fill_null(0.0).alias("yards_mean_completed"),
            completion_yards.std().fill_null(0.0).alias("yards_sd_completed"),
            (completion_yards >= 5.0).mean().fill_null(0.0).alias("gain_5plus_completion_rate"),
            (completion_yards >= 10.0).mean().fill_null(0.0).alias("gain_10plus_completion_rate"),
            (completion_yards >= 15.0).mean().fill_null(0.0).alias("gain_15plus_completion_rate"),
            (completion_yards >= 20.0).mean().fill_null(0.0).alias("gain_20plus_completion_rate"),
            (completion_yards >= 40.0).mean().fill_null(0.0).alias("gain_40plus_completion_rate"),
            forty_plus_completion_yards.mean().fill_null(0.0).alias("yards_40plus_mean_completed"),
            yac.filter(completions == 1.0).mean().fill_null(0.0).alias("yac_mean_completed"),
            yac.filter(completions == 1.0).std().fill_null(0.0).alias("yac_sd_completed"),
            (completion_yards < 0).mean().fill_null(0.0).alias("negative_completion_rate"),
            (completion_yards == 0).mean().fill_null(0.0).alias("zero_completion_rate"),
        )
        .sort("category")
    )

    if "receiver_player_id" in frame.columns:
        target_depth = (
            frame.filter(pl.col("receiver_player_id").is_not_null())
            .group_by(["posteam", "receiver_player_id", "category"])
            .agg(pl.len().alias("attempts"))
            .rename({"posteam": "team_id", "receiver_player_id": "player_id"})
            .with_columns(
                (
                    pl.col("attempts")
                    / pl.col("attempts").sum().over(["team_id", "category"]).clip(lower_bound=1)
                ).alias("target_share_within_depth")
            )
            .sort(["team_id", "category", "attempts"], descending=[False, False, True])
        )
    else:
        target_depth = pl.DataFrame(
            schema={
                "team_id": pl.String,
                "player_id": pl.String,
                "category": pl.String,
                "attempts": pl.Int64,
                "target_share_within_depth": pl.Float64,
            }
        )
    return league, team, qb, outcomes, target_depth


def compile_run_intent_policy(
    pbp: pl.DataFrame,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Compile observable run geometry, runner identity, and branch outcome anatomy."""

    frame = prepare_run_intent_plays(pbp)
    league, team = _categorical_context_counts(frame)
    if "rusher_player_id" in frame.columns:
        rusher = (
            frame.filter(pl.col("rusher_player_id").is_not_null())
            .group_by(["rusher_player_id", "category"])
            .agg(pl.len().alias("attempts"))
            .rename({"rusher_player_id": "actor_id"})
            .sort(["actor_id", "category"])
        )
    else:
        rusher = pl.DataFrame(schema={"actor_id": pl.String, "category": pl.String, "attempts": pl.Int64})

    yards = pl.col("yards_gained").fill_null(0.0).cast(pl.Float64)
    forty_plus_yards = yards.filter(yards >= 40.0)
    outcomes = (
        frame.group_by("category")
        .agg(
            pl.len().alias("attempts"),
            yards.mean().alias("yards_mean"),
            yards.std().fill_null(0.0).alias("yards_sd"),
            (yards < 0).mean().alias("negative_rate"),
            (yards == 0).mean().alias("zero_rate"),
            (yards <= -2).mean().alias("loss_2_plus_rate"),
            (yards <= -5).mean().alias("loss_5_plus_rate"),
            (yards >= 10).mean().alias("explosive_10_rate"),
            (yards >= 15).mean().alias("explosive_15_rate"),
            (yards >= 20).mean().alias("explosive_20_rate"),
            (yards >= 40).mean().alias("explosive_40_rate"),
            forty_plus_yards.mean().fill_null(0.0).alias("yards_40plus_mean"),
            pl.col("rush_touchdown").fill_null(0).cast(pl.Float64).mean().alias("touchdown_rate"),
            pl.col("fumble_lost").fill_null(0).cast(pl.Float64).mean().alias("fumble_lost_rate"),
            yards.quantile(0.10).alias("yards_p10"),
            yards.quantile(0.50).alias("yards_p50"),
            yards.quantile(0.90).alias("yards_p90"),
            yards.quantile(0.99).alias("yards_p99"),
        )
        .sort("category")
    )
    return league, team, rusher, outcomes
