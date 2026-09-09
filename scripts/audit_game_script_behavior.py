from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import nflreadpy as nfl
import numpy as np
import polars as pl

from monster.ingest.nflverse import PBP_COLUMNS

SEASONS = (2022, 2023, 2024, 2025)


def _score_bucket(value: float | None) -> str:
    if value is None or not np.isfinite(value):
        return "unknown"
    if value <= -14:
        return "trail_14plus"
    if value <= -7:
        return "trail_7_13"
    if value <= -1:
        return "trail_1_6"
    if value == 0:
        return "tied"
    if value <= 6:
        return "lead_1_6"
    if value <= 13:
        return "lead_7_13"
    return "lead_14plus"


def _phase(seconds: float | None) -> str:
    if seconds is None or not np.isfinite(seconds):
        return "unknown"
    if seconds > 1800:
        return "first_half"
    if seconds > 900:
        return "q3"
    if seconds > 300:
        return "q4_early"
    return "final_5"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seasons", default=",".join(str(x) for x in SEASONS))
    parser.add_argument("--out", type=Path, default=Path("artifacts/game-script-behavior"))
    args = parser.parse_args()

    seasons = [int(x) for x in args.seasons.split(",")]
    pbp = nfl.load_pbp(seasons)
    keep = [c for c in PBP_COLUMNS if c in pbp.columns]
    pbp = pbp.select(keep)
    if "season_type" in pbp.columns:
        pbp = pbp.filter(pl.col("season_type") == "REG")

    required = {"season", "posteam", "pass_attempt", "rush_attempt", "score_differential", "game_seconds_remaining"}
    missing = required - set(pbp.columns)
    if missing:
        raise ValueError(f"Missing required PBP columns: {sorted(missing)}")

    plays = (
        pbp.filter(pl.col("posteam").is_not_null())
        .filter((pl.col("pass_attempt") == 1) | (pl.col("rush_attempt") == 1))
        .with_columns(
            pl.col("score_differential").map_elements(_score_bucket, return_dtype=pl.String).alias("score_bucket"),
            pl.col("game_seconds_remaining").map_elements(_phase, return_dtype=pl.String).alias("phase"),
            (pl.col("pass_attempt") == 1).cast(pl.Float64).alias("is_pass"),
            (pl.col("rush_attempt") == 1).cast(pl.Float64).alias("is_rush"),
            ((pl.col("interception").fill_null(0) == 1) | (pl.col("fumble_lost").fill_null(0) == 1)).cast(pl.Float64).alias("turnover_play"),
            ((pl.col("pass_touchdown").fill_null(0) == 1) | (pl.col("rush_touchdown").fill_null(0) == 1)).cast(pl.Float64).alias("offensive_td_play"),
        )
    )

    grouped = (
        plays.group_by(["season", "phase", "score_bucket"])
        .agg(
            pl.len().alias("plays"),
            pl.col("is_pass").mean().alias("pass_rate"),
            pl.col("epa").mean().alias("epa_per_play"),
            pl.col("success").mean().alias("success_rate"),
            pl.col("turnover_play").mean().alias("turnover_play_rate"),
            pl.col("offensive_td_play").mean().alias("offensive_td_play_rate"),
            pl.col("sack").fill_null(0).mean().alias("sack_play_rate"),
        )
        .filter(pl.col("plays") >= 100)
        .sort(["season", "phase", "score_bucket"])
    )

    pooled = (
        plays.group_by(["phase", "score_bucket"])
        .agg(
            pl.len().alias("plays"),
            pl.col("is_pass").mean().alias("pass_rate"),
            pl.col("epa").mean().alias("epa_per_play"),
            pl.col("success").mean().alias("success_rate"),
            pl.col("turnover_play").mean().alias("turnover_play_rate"),
            pl.col("offensive_td_play").mean().alias("offensive_td_play_rate"),
            pl.col("sack").fill_null(0).mean().alias("sack_play_rate"),
        )
        .sort(["phase", "score_bucket"])
    )

    final5 = pooled.filter(pl.col("phase") == "final_5")
    def value(bucket: str, col: str) -> float:
        sub = final5.filter(pl.col("score_bucket") == bucket)
        return float(sub[col][0]) if sub.height else float("nan")

    pass_spread = value("trail_7_13", "pass_rate") - value("lead_7_13", "pass_rate")
    turnover_spread = value("trail_7_13", "turnover_play_rate") - value("lead_7_13", "turnover_play_rate")
    sack_spread = value("trail_7_13", "sack_play_rate") - value("lead_7_13", "sack_play_rate")

    seasonal_final = grouped.filter(pl.col("phase") == "final_5")
    season_checks = []
    for season in seasons:
        sub = seasonal_final.filter(pl.col("season") == season)
        tr = sub.filter(pl.col("score_bucket") == "trail_7_13")
        ld = sub.filter(pl.col("score_bucket") == "lead_7_13")
        if tr.height and ld.height:
            season_checks.append({
                "season": season,
                "trailing_minus_leading_pass_rate": float(tr["pass_rate"][0] - ld["pass_rate"][0]),
                "direction_pass": bool(tr["pass_rate"][0] > ld["pass_rate"][0]),
            })
    checks = pl.DataFrame(season_checks)
    stable_direction = bool(checks.height >= 3 and checks["direction_pass"].all())

    manifest = {
        "artifact": "Monster Historical Game-Script Behavior Audit",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "seasons": seasons,
        "pbp_scope": "REG scrimmage pass/rush plays",
        "final5_trail_7_13_minus_lead_7_13_pass_rate": pass_spread,
        "final5_trail_7_13_minus_lead_7_13_turnover_play_rate": turnover_spread,
        "final5_trail_7_13_minus_lead_7_13_sack_play_rate": sack_spread,
        "pass_direction_stable_by_season": stable_direction,
        "structural_script_signal_pass": bool(stable_direction and pass_spread >= 0.15),
        "market_blind": True,
        "principle": "Score-state behavior is measured before sequential policy coefficients are introduced.",
    }

    args.out.mkdir(parents=True, exist_ok=True)
    grouped.write_csv(args.out / "season_phase_score_behavior.csv")
    pooled.write_csv(args.out / "pooled_phase_score_behavior.csv")
    checks.write_csv(args.out / "season_direction_checks.csv")
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))
    print(final5)


if __name__ == "__main__":
    main()
