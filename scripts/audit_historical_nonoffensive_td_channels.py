from __future__ import annotations

import argparse
import json
from pathlib import Path

import nflreadpy as nfl
import polars as pl


def _bool_col(frame: pl.DataFrame, column: str) -> pl.Expr:
    if column not in frame.columns:
        return pl.lit(False)
    return pl.col(column).fill_null(0).cast(pl.Int64) == 1


def _text_col(frame: pl.DataFrame, column: str) -> pl.Expr:
    if column not in frame.columns:
        return pl.lit("")
    return pl.col(column).cast(pl.Utf8).fill_null("").str.to_lowercase()


def _rate(frame: pl.DataFrame, expression: pl.Expr) -> float:
    if not frame.height:
        return 0.0
    return float(frame.select(expression.cast(pl.Float64).mean()).item())


def _return_profile(frame: pl.DataFrame) -> dict[str, float | int]:
    if not frame.height or "return_yards" not in frame.columns:
        return {"rows": 0}
    returns = frame.filter(pl.col("return_yards").is_not_null())
    if not returns.height:
        return {"rows": 0}
    return {
        "rows": returns.height,
        "zero_rate": _rate(returns, pl.col("return_yards") <= 0.0),
        "20_plus_rate": _rate(returns, pl.col("return_yards") >= 20.0),
        "40_plus_rate": _rate(returns, pl.col("return_yards") >= 40.0),
        "60_plus_rate": _rate(returns, pl.col("return_yards") >= 60.0),
        "80_plus_rate": _rate(returns, pl.col("return_yards") >= 80.0),
        "mean": float(returns.select(pl.col("return_yards").mean()).item()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--season", type=int, default=2025)
    parser.add_argument("--out", type=Path, default=Path("artifacts/nonoffensive-td-audit.json"))
    args = parser.parse_args()

    pbp = nfl.load_pbp([args.season])
    if "season_type" in pbp.columns:
        pbp = pbp.filter(pl.col("season_type") == "REG")

    schedules = nfl.load_schedules([args.season])
    games = schedules.filter(
        pl.col("home_score").is_not_null() & pl.col("away_score").is_not_null()
    )
    if "game_type" in games.columns:
        games = games.filter(pl.col("game_type") == "REG")
    game_count = games.height

    touchdown = _bool_col(pbp, "touchdown")
    offensive = _bool_col(pbp, "pass_touchdown") | _bool_col(pbp, "rush_touchdown")
    nonoff = pbp.filter(touchdown & ~offensive)

    interception = _bool_col(nonoff, "interception")
    fumble_lost = _bool_col(nonoff, "fumble_lost")
    punt = _bool_col(nonoff, "punt_attempt")
    kickoff = _bool_col(nonoff, "kickoff_attempt")
    fg_attempt = _bool_col(nonoff, "field_goal_attempt")
    punt_blocked = _bool_col(nonoff, "punt_blocked")
    fg_blocked = fg_attempt & (_text_col(nonoff, "field_goal_result") == "blocked")

    classified = nonoff.with_columns(
        pl.when(interception)
        .then(pl.lit("interception_return"))
        .when(punt & punt_blocked)
        .then(pl.lit("blocked_punt_return"))
        .when(kickoff)
        .then(pl.lit("kickoff_return"))
        .when(punt)
        .then(pl.lit("punt_return"))
        .when(fg_blocked)
        .then(pl.lit("blocked_field_goal_return"))
        .when(fumble_lost)
        .then(pl.lit("fumble_return"))
        .otherwise(pl.lit("other_nonoffensive"))
        .alias("nonoff_channel")
    )
    channel_rows = (
        classified.group_by("nonoff_channel")
        .len()
        .sort("len", descending=True)
        .to_dicts()
    )
    channels = {
        str(row["nonoff_channel"]): {
            "touchdowns": int(row["len"]),
            "per_game": float(row["len"] / game_count),
        }
        for row in channel_rows
    }

    all_interceptions = pbp.filter(_bool_col(pbp, "interception"))
    all_fumbles = pbp.filter(_bool_col(pbp, "fumble_lost"))
    all_punts = pbp.filter(_bool_col(pbp, "punt_attempt"))
    all_kickoffs = pbp.filter(_bool_col(pbp, "kickoff_attempt"))

    return_profiles = {
        "interception": _return_profile(all_interceptions),
        "fumble": _return_profile(all_fumbles),
        "punt": _return_profile(all_punts),
        "kickoff": _return_profile(all_kickoffs),
    }

    interception_geometry: dict[str, object] = {}
    if {"yardline_100", "air_yards", "return_yards"}.issubset(all_interceptions.columns):
        geometry = all_interceptions.filter(
            pl.col("yardline_100").is_not_null()
            & pl.col("air_yards").is_not_null()
            & pl.col("return_yards").is_not_null()
        ).with_columns(
            (
                100.0
                - pl.col("yardline_100").cast(pl.Float64)
                + pl.col("air_yards").cast(pl.Float64)
            )
            .clip(1.0, 99.0)
            .alias("required_return_distance")
        )
        if geometry.height:
            geometry = geometry.with_columns(
                (pl.col("return_yards") - pl.col("required_return_distance")).alias(
                    "distance_minus_required"
                ),
                pl.when(pl.col("required_return_distance") <= 20.0)
                .then(pl.lit("00_20"))
                .when(pl.col("required_return_distance") <= 40.0)
                .then(pl.lit("20_40"))
                .when(pl.col("required_return_distance") <= 60.0)
                .then(pl.lit("40_60"))
                .when(pl.col("required_return_distance") <= 80.0)
                .then(pl.lit("60_80"))
                .otherwise(pl.lit("80_100"))
                .alias("required_bucket"),
            )
            by_bucket = []
            for bucket, group in geometry.group_by("required_bucket"):
                bucket_name = bucket[0] if isinstance(bucket, tuple) else bucket
                td_expression = (
                    _bool_col(group, "return_touchdown")
                    if "return_touchdown" in group.columns
                    else _bool_col(group, "touchdown")
                )
                by_bucket.append(
                    {
                        "bucket": str(bucket_name),
                        "rows": group.height,
                        "actual_return_td_rate": _rate(group, td_expression),
                        "geometry_cover_rate": _rate(
                            group,
                            pl.col("return_yards") >= pl.col("required_return_distance") - 0.5,
                        ),
                        "required_distance_mean": float(
                            group.select(pl.col("required_return_distance").mean()).item()
                        ),
                        "return_yards_mean": float(
                            group.select(pl.col("return_yards").mean()).item()
                        ),
                    }
                )
            interception_geometry = {
                "rows": geometry.height,
                "by_required_distance_bucket": sorted(by_bucket, key=lambda row: row["bucket"]),
            }

    payload = {
        "season": args.season,
        "games": game_count,
        "nonoffensive_touchdowns": classified.height,
        "nonoffensive_touchdowns_per_game": float(classified.height / game_count),
        "channels": channels,
        "return_profiles": return_profiles,
        "interception_geometry": interception_geometry,
        "principle": (
            "This audit diagnoses historical scoring channels and field geometry only. Runtime "
            "authority remains return distance + live field state, never a direct touchdown rate."
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()