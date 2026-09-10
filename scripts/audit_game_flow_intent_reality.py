from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import nflreadpy as nfl
import polars as pl

from monster.ingest.nflverse import configure_cache
from monster.teams import TEAM_ALIASES


def _distance_bucket() -> pl.Expr:
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


def _field_zone() -> pl.Expr:
    # nflfastR yardline_100 is distance to the opponent goal line.
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


def _score_state() -> pl.Expr:
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


def _time_mode() -> pl.Expr:
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


def _pass_depth_band() -> pl.Expr:
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


def _special_state() -> pl.Expr:
    down = pl.col("down")
    dist = pl.col("ydstogo")
    qtr = pl.col("qtr")
    game_left = pl.col("game_seconds_remaining")
    diff = pl.col("score_differential")
    yardline = pl.col("yardline_100")
    return (
        pl.when((down == 2) & (dist <= 2))
        .then(pl.lit("second_and_2_or_less"))
        .when((down == 3) & (dist <= 2))
        .then(pl.lit("third_and_short"))
        .when((down == 3) & dist.is_between(3, 6))
        .then(pl.lit("third_and_medium"))
        .when((down == 3) & dist.is_between(7, 10))
        .then(pl.lit("third_and_7_10"))
        .when((down == 3) & (dist >= 11) & (dist <= 17))
        .then(pl.lit("third_and_11_17"))
        .when((down == 3) & (dist >= 18))
        .then(pl.lit("third_and_18_plus"))
        .when((down == 4) & (dist <= 2))
        .then(pl.lit("fourth_and_short"))
        .when((qtr == 2) & (game_left - 1800 <= 120))
        .then(pl.lit("end_first_half"))
        .when((qtr == 4) & (game_left <= 120) & (diff < 0))
        .then(pl.lit("late_trailing"))
        .when((qtr == 4) & (game_left <= 240) & (diff >= 1))
        .then(pl.lit("four_minute_lead"))
        .when((yardline <= 10) & (pl.col("goal_to_go").fill_null(0) == 1))
        .then(pl.lit("low_goal_to_go"))
        .when(yardline >= 90)
        .then(pl.lit("deep_backed_up"))
        .otherwise(pl.lit("ordinary"))
    )


def _prepare(pbp: pl.DataFrame) -> pl.DataFrame:
    required = {
        "posteam",
        "play_type",
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
        raise ValueError(f"PBP missing game-flow columns: {sorted(missing)}")

    dropback = pl.col("qb_dropback").fill_null(0).cast(pl.Float64) == 1.0
    designed_run = (pl.col("rush_attempt").fill_null(0).cast(pl.Float64) == 1.0) & ~dropback
    frame = (
        pbp.with_columns(pl.col("posteam").replace(TEAM_ALIASES))
        .filter(
            pl.col("posteam").is_not_null()
            & pl.col("down").is_between(1, 4)
            & pl.col("ydstogo").is_not_null()
            & pl.col("yardline_100").is_not_null()
            & pl.col("game_seconds_remaining").is_not_null()
            & pl.col("score_differential").is_not_null()
            & (dropback | designed_run)
        )
        .with_columns(
            dropback.alias("is_dropback"),
            designed_run.alias("is_designed_run"),
            _distance_bucket().alias("distance_bucket"),
            _field_zone().alias("field_zone"),
            _score_state().alias("score_state"),
            _time_mode().alias("time_mode"),
            _special_state().alias("special_state"),
        )
    )
    optional = []
    if "shotgun" in frame.columns:
        optional.append(pl.col("shotgun").fill_null(0).cast(pl.Float64).alias("shotgun_flag"))
    if "no_huddle" in frame.columns:
        optional.append(pl.col("no_huddle").fill_null(0).cast(pl.Float64).alias("no_huddle_flag"))
    if optional:
        frame = frame.with_columns(*optional)
    return frame


def _context_summary(frame: pl.DataFrame) -> pl.DataFrame:
    aggs: list[pl.Expr] = [
        pl.len().alias("samples"),
        pl.col("is_dropback").mean().alias("dropback_rate"),
        pl.col("is_designed_run").mean().alias("designed_run_rate"),
    ]
    if "shotgun_flag" in frame.columns:
        aggs.append(pl.col("shotgun_flag").mean().alias("shotgun_rate"))
    if "no_huddle_flag" in frame.columns:
        aggs.append(pl.col("no_huddle_flag").mean().alias("no_huddle_rate"))
    return (
        frame.group_by(["down", "distance_bucket", "field_zone", "time_mode", "score_state"])
        .agg(*aggs)
        .sort(["down", "distance_bucket", "field_zone", "time_mode", "score_state"])
    )


def _special_summary(frame: pl.DataFrame, *, team: bool) -> pl.DataFrame:
    keys = ["special_state"]
    if team:
        keys.insert(0, "posteam")
    aggs: list[pl.Expr] = [
        pl.len().alias("samples"),
        pl.col("is_dropback").mean().alias("dropback_rate"),
        pl.col("is_designed_run").mean().alias("designed_run_rate"),
    ]
    if "shotgun_flag" in frame.columns:
        aggs.append(pl.col("shotgun_flag").mean().alias("shotgun_rate"))
    if "no_huddle_flag" in frame.columns:
        aggs.append(pl.col("no_huddle_flag").mean().alias("no_huddle_rate"))
    return frame.group_by(keys).agg(*aggs).sort(keys)


def _pass_depth_summary(frame: pl.DataFrame) -> pl.DataFrame:
    if "air_yards" not in frame.columns:
        return pl.DataFrame()
    throws = frame.filter(
        pl.col("is_dropback")
        & (pl.col("sack").fill_null(0) != 1)
        & (pl.col("qb_scramble").fill_null(0) != 1)
        & pl.col("air_yards").is_not_null()
    ).with_columns(_pass_depth_band().alias("pass_depth_band"))
    if not throws.height:
        return pl.DataFrame()
    return (
        throws.group_by(["special_state", "pass_depth_band"])
        .agg(
            pl.len().alias("attempts"),
            pl.col("complete_pass").fill_null(0).cast(pl.Float64).mean().alias("completion_rate"),
            pl.col("interception").fill_null(0).cast(pl.Float64).mean().alias("interception_rate"),
            pl.col("air_yards").mean().alias("air_yards_mean"),
            pl.col("yards_gained").mean().alias("yards_gained_mean"),
        )
        .with_columns(
            (pl.col("attempts") / pl.col("attempts").sum().over("special_state")).alias(
                "attempt_share_within_state"
            )
        )
        .sort(["special_state", "pass_depth_band"])
    )


def _run_shape_summary(frame: pl.DataFrame) -> pl.DataFrame:
    if "run_location" not in frame.columns:
        return pl.DataFrame()
    runs = frame.filter(pl.col("is_designed_run")).with_columns(
        pl.col("run_location").fill_null("unknown"),
        (
            pl.col("run_gap").fill_null("unknown")
            if "run_gap" in frame.columns
            else pl.lit("unknown")
        ).alias("run_gap_norm"),
    )
    return (
        runs.group_by(["special_state", "run_location", "run_gap_norm"])
        .agg(
            pl.len().alias("attempts"),
            pl.col("yards_gained").mean().alias("yards_mean"),
            (pl.col("yards_gained") < 0).mean().alias("negative_gain_rate"),
            (pl.col("yards_gained") >= 15).mean().alias("explosive_15_rate"),
        )
        .with_columns(
            (pl.col("attempts") / pl.col("attempts").sum().over("special_state")).alias(
                "attempt_share_within_state"
            )
        )
        .sort(["special_state", "run_location", "run_gap_norm"])
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--history", type=int, default=2025)
    parser.add_argument("--cache-dir", type=Path, default=Path(".cache/monster"))
    parser.add_argument("--out", type=Path, default=Path("artifacts/game-flow-intent-reality"))
    args = parser.parse_args()

    configure_cache(args.cache_dir)
    pbp = nfl.load_pbp([args.history])
    if "season_type" in pbp.columns:
        pbp = pbp.filter(pl.col("season_type") == "REG")
    if "qb_kneel" in pbp.columns:
        pbp = pbp.filter(pl.col("qb_kneel").fill_null(0) == 0)
    if "qb_spike" in pbp.columns:
        pbp = pbp.filter(pl.col("qb_spike").fill_null(0) == 0)

    frame = _prepare(pbp)
    context = _context_summary(frame)
    special = _special_summary(frame, team=False)
    team_special = _special_summary(frame, team=True)
    pass_depth = _pass_depth_summary(frame)
    run_shape = _run_shape_summary(frame)

    args.out.mkdir(parents=True, exist_ok=True)
    context.write_csv(args.out / "game_flow_context.csv")
    special.write_csv(args.out / "special_state_summary.csv")
    team_special.write_csv(args.out / "team_special_state_summary.csv")
    if pass_depth.height:
        pass_depth.write_csv(args.out / "pass_depth_by_special_state.csv")
    if run_shape.height:
        run_shape.write_csv(args.out / "run_shape_by_special_state.csv")

    manifest = {
        "artifact": "Monster Game Flow + Play Intent Reality Audit",
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "historical_season": args.history,
        "market_blind": True,
        "behavior_changed_by_audit": False,
        "scrimmage_samples": frame.height,
        "context_cells": context.height,
        "special_state_rows": special.height,
        "team_special_state_rows": team_special.height,
        "pass_depth_rows": pass_depth.height,
        "run_shape_rows": run_shape.height,
        "principle": (
            "Game state creates a strategic problem; team/coach identity changes the decision "
            "distribution; play resolution remains a separate mechanism. This audit observes "
            "historical intent proxies and does not authorize live behavior changes."
        ),
        "definitions": {
            "pass_family": "qb_dropback == 1, preserving sacks and scrambles as dropback intent",
            "designed_run": "rush_attempt == 1 and qb_dropback != 1",
            "distance_buckets": ["1", "2", "3_4", "5_7", "8_10", "11_15", "16_plus"],
            "field_zones": [
                "backed_up",
                "own_field",
                "midfield",
                "opp_territory",
                "high_red_zone",
                "low_red_zone",
            ],
            "special_states": [
                "second_and_2_or_less",
                "third_and_short",
                "third_and_medium",
                "third_and_7_10",
                "third_and_11_17",
                "third_and_18_plus",
                "fourth_and_short",
                "end_first_half",
                "late_trailing",
                "four_minute_lead",
                "low_goal_to_go",
                "deep_backed_up",
                "ordinary",
            ],
        },
        "files": {
            "context": "game_flow_context.csv",
            "special": "special_state_summary.csv",
            "team_special": "team_special_state_summary.csv",
            "pass_depth": "pass_depth_by_special_state.csv" if pass_depth.height else None,
            "run_shape": "run_shape_by_special_state.csv" if run_shape.height else None,
        },
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
