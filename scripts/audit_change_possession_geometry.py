from __future__ import annotations

import argparse
import json

import nflreadpy as nfl
import polars as pl


def _flag(frame: pl.DataFrame, column: str) -> pl.Expr:
    if column not in frame.columns:
        return pl.lit(False)
    return pl.col(column).fill_null(0).cast(pl.Int64, strict=False) == 1


def _profile(frame: pl.DataFrame, required: str, returned: str, td: str = "return_touchdown") -> dict:
    if not frame.height:
        return {"rows": 0}
    rows = frame.filter(pl.col(required).is_not_null() & pl.col(returned).is_not_null())
    if not rows.height:
        return {"rows": 0}
    req = rows.get_column(required)
    ret = rows.get_column(returned)
    payload = {
        "rows": rows.height,
        "required_mean": float(req.mean()),
        "required_p25": float(req.quantile(0.25)),
        "required_p50": float(req.quantile(0.50)),
        "required_p75": float(req.quantile(0.75)),
        "return_mean": float(ret.mean()),
        "geometry_cover_rate": float((ret >= req - 0.5).mean()),
    }
    if td in rows.columns:
        payload["actual_td_rate"] = float(
            rows.select(_flag(rows, td).cast(pl.Float64).mean()).item()
        )
    bucketed = rows.with_columns(
        pl.when(pl.col(required) <= 20.0)
        .then(pl.lit("00_20"))
        .when(pl.col(required) <= 40.0)
        .then(pl.lit("20_40"))
        .when(pl.col(required) <= 60.0)
        .then(pl.lit("40_60"))
        .when(pl.col(required) <= 80.0)
        .then(pl.lit("60_80"))
        .otherwise(pl.lit("80_100"))
        .alias("required_bucket")
    )
    buckets = []
    for bucket, group in bucketed.group_by("required_bucket"):
        name = bucket[0] if isinstance(bucket, tuple) else bucket
        item = {
            "bucket": str(name),
            "rows": group.height,
            "required_mean": float(group.get_column(required).mean()),
            "return_mean": float(group.get_column(returned).mean()),
            "cover_rate": float(
                (group.get_column(returned) >= group.get_column(required) - 0.5).mean()
            ),
        }
        if td in group.columns:
            item["actual_td_rate"] = float(
                group.select(_flag(group, td).cast(pl.Float64).mean()).item()
            )
        buckets.append(item)
    payload["by_required_bucket"] = sorted(buckets, key=lambda item: item["bucket"])
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--season", type=int, default=2025)
    args = parser.parse_args()
    pbp = nfl.load_pbp([args.season])
    if "season_type" in pbp.columns:
        pbp = pbp.filter(pl.col("season_type") == "REG")

    payload: dict[str, object] = {"season": args.season}

    # For kicks, nflverse yardline_100 is distance from the kicking team to its target goal.
    # Gross kick distance moves toward that goal, so the receiving team's own-yardline landing
    # coordinate is yardline_100 - kick_distance. The returner must cover 100 - landing to score.
    if {"kick_distance", "yardline_100", "return_yards"}.issubset(pbp.columns):
        punts = pbp.filter(
            _flag(pbp, "punt_attempt")
            & ~_flag(pbp, "punt_blocked")
            & pl.col("return_yards").is_not_null()
            & (pl.col("return_yards") >= 0.0)
        ).with_columns(
            (pl.col("yardline_100") - pl.col("kick_distance"))
            .cast(pl.Float64)
            .clip(0.0, 100.0)
            .alias("landing_from_receiving_goal")
        ).with_columns(
            (100.0 - pl.col("landing_from_receiving_goal")).alias("required_return_distance"),
            (pl.col("landing_from_receiving_goal") + pl.col("return_yards")).alias("resulting_start"),
        )
        live_punts = punts.filter(pl.col("return_yards") > 0.0)
        payload["punt_live_return"] = {
            **_profile(live_punts, "required_return_distance", "return_yards"),
            "landing_mean": float(live_punts.get_column("landing_from_receiving_goal").mean()) if live_punts.height else None,
            "resulting_start_mean": float(live_punts.get_column("resulting_start").mean()) if live_punts.height else None,
        }

        kicks = pbp.filter(
            _flag(pbp, "kickoff_attempt")
            & ~_flag(pbp, "touchback")
            & pl.col("return_yards").is_not_null()
            & (pl.col("return_yards") >= 0.0)
        ).with_columns(
            (pl.col("yardline_100") - pl.col("kick_distance"))
            .cast(pl.Float64)
            .clip(0.0, 100.0)
            .alias("landing_from_receiving_goal")
        ).with_columns(
            (100.0 - pl.col("landing_from_receiving_goal")).alias("required_return_distance"),
            (pl.col("landing_from_receiving_goal") + pl.col("return_yards")).alias("resulting_start"),
        )
        live_kicks = kicks.filter(pl.col("return_yards") > 0.0)
        payload["kickoff_live_return"] = {
            **_profile(live_kicks, "required_return_distance", "return_yards"),
            "landing_mean": float(live_kicks.get_column("landing_from_receiving_goal").mean()) if live_kicks.height else None,
            "landing_p25": float(live_kicks.get_column("landing_from_receiving_goal").quantile(0.25)) if live_kicks.height else None,
            "landing_p50": float(live_kicks.get_column("landing_from_receiving_goal").quantile(0.50)) if live_kicks.height else None,
            "landing_p75": float(live_kicks.get_column("landing_from_receiving_goal").quantile(0.75)) if live_kicks.height else None,
            "resulting_start_mean": float(live_kicks.get_column("resulting_start").mean()) if live_kicks.height else None,
        }
    else:
        payload["kick_geometry_error"] = "kick_distance column unavailable"

    # A lost fumble at the offense's own coordinate x requires the recovering defense to cover x
    # yards to reach that offense's goal line. Approximate the live-ball spot with start coordinate
    # plus scrimmage yards, then verify the formula against actual recovery touchdowns.
    if {"yardline_100", "yards_gained", "fumble_recovery_1_yards"}.issubset(pbp.columns):
        fumbles = pbp.filter(
            _flag(pbp, "fumble_lost")
            & pl.col("fumble_recovery_1_yards").is_not_null()
        ).with_columns(
            (100.0 - pl.col("yardline_100") + pl.col("yards_gained").fill_null(0.0))
            .cast(pl.Float64)
            .clip(0.0, 100.0)
            .alias("required_return_distance"),
            pl.col("fumble_recovery_1_yards").cast(pl.Float64).alias("recovery_return_yards"),
        )
        payload["fumble_recovery"] = _profile(
            fumbles, "required_return_distance", "recovery_return_yards", td="touchdown"
        )

    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
