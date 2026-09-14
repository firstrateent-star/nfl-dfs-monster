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
    upper = rows.filter(pl.col(returned) >= 80.0)
    payload = {
        "rows": rows.height,
        "required_mean": float(req.mean()),
        "required_p25": float(req.quantile(0.25)),
        "required_p50": float(req.quantile(0.50)),
        "required_p75": float(req.quantile(0.75)),
        "return_mean": float(ret.mean()),
        "geometry_cover_rate": float((ret >= req - 0.5).mean()),
        "upper_80_rows": upper.height,
        "upper_80_mean": float(upper.get_column(returned).mean()) if upper.height else None,
        "upper_80_p50": float(upper.get_column(returned).quantile(0.50)) if upper.height else None,
        "upper_80_p90": float(upper.get_column(returned).quantile(0.90)) if upper.height else None,
        "upper_80_max": float(upper.get_column(returned).max()) if upper.height else None,
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
            "zero_rate": float((group.get_column(returned) <= 0.0).mean()),
            "p20": float((group.get_column(returned) >= 20.0).mean()),
            "p40": float((group.get_column(returned) >= 40.0).mean()),
            "p60": float((group.get_column(returned) >= 60.0).mean()),
            "p80": float((group.get_column(returned) >= 80.0).mean()),
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


def _drive_starts(pbp: pl.DataFrame) -> pl.DataFrame:
    scrimmage = _flag(pbp, "pass_attempt") | _flag(pbp, "rush_attempt") | _flag(pbp, "sack")
    sort_cols = [column for column in ["game_id", "fixed_drive", "play_id"] if column in pbp.columns]
    return (
        pbp.filter(
            pl.col("game_id").is_not_null()
            & pl.col("fixed_drive").is_not_null()
            & pl.col("yardline_100").is_not_null()
            & scrimmage
        )
        .sort(sort_cols)
        .group_by(["game_id", "fixed_drive"], maintain_order=True)
        .agg(pl.col("yardline_100").first().alias("first_scrimmage_yardline_100"))
        .with_columns(
            (100.0 - pl.col("first_scrimmage_yardline_100")).alias("drive_start_from_own_goal")
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--season", type=int, default=2025)
    args = parser.parse_args()
    pbp = nfl.load_pbp([args.season])
    if "season_type" in pbp.columns:
        pbp = pbp.filter(pl.col("season_type") == "REG")

    payload: dict[str, object] = {"season": args.season}
    drive_starts = _drive_starts(pbp)

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

        # nflverse's 2025 dynamic-kickoff kick_distance does not map cleanly to the receiving
        # landing coordinate. Reconstruct the receiving drive's first scrimmage position and infer
        # the physical return start from final position - return yards instead.
        kicks = (
            pbp.filter(
                _flag(pbp, "kickoff_attempt")
                & ~_flag(pbp, "touchback")
                & pl.col("return_yards").is_not_null()
                & pl.col("game_id").is_not_null()
                & pl.col("fixed_drive").is_not_null()
            )
            .join(drive_starts, on=["game_id", "fixed_drive"], how="inner")
            .with_columns(
                (
                    pl.col("drive_start_from_own_goal")
                    - pl.col("return_yards").cast(pl.Float64)
                )
                .clip(0.0, 20.0)
                .alias("landing_from_receiving_goal")
            )
            .with_columns(
                (100.0 - pl.col("landing_from_receiving_goal")).alias("required_return_distance")
            )
        )
        live_kicks = kicks.filter(pl.col("return_yards") > 0.0)
        payload["kickoff_live_return"] = {
            **_profile(live_kicks, "required_return_distance", "return_yards"),
            "landing_mean": float(live_kicks.get_column("landing_from_receiving_goal").mean()) if live_kicks.height else None,
            "landing_p25": float(live_kicks.get_column("landing_from_receiving_goal").quantile(0.25)) if live_kicks.height else None,
            "landing_p50": float(live_kicks.get_column("landing_from_receiving_goal").quantile(0.50)) if live_kicks.height else None,
            "landing_p75": float(live_kicks.get_column("landing_from_receiving_goal").quantile(0.75)) if live_kicks.height else None,
            "drive_start_mean": float(live_kicks.get_column("drive_start_from_own_goal").mean()) if live_kicks.height else None,
        }
    else:
        payload["kick_geometry_error"] = "kick_distance column unavailable"

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
