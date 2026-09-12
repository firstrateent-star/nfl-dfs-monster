from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import nflreadpy as nfl
import polars as pl


def _safe_success(column: str, frame: pl.DataFrame, values: list[str]) -> pl.Expr:
    if column not in frame.columns:
        return pl.lit(False)
    return (
        pl.col(column)
        .cast(pl.Utf8)
        .str.to_lowercase()
        .fill_null("")
        .is_in(values)
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--simulation", type=Path, required=True)
    parser.add_argument("--history", type=int, default=2025)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    sim_dir = args.simulation
    out = args.out or sim_dir
    anatomy = pl.read_csv(sim_dir / "football_anatomy.csv")
    games = pl.read_csv(sim_dir / "game_distributions.csv")

    def model_mean(column: str) -> float:
        return float(anatomy.select(pl.col(column).mean()).item())

    model = {
        "points_per_game": float(games.select(pl.col("total_mean").mean()).item()),
        "touchdowns_per_game": model_mean("touchdowns_mean"),
        "fg_attempts_per_game": model_mean("field_goal_attempts_mean"),
        "fg_made_per_game": model_mean("field_goals_made_mean"),
        "punts_per_game": model_mean("punts_mean"),
        "drives_per_game": model_mean("drives_mean"),
        "scrimmage_plays_per_game": model_mean("scrimmage_plays_mean"),
        "turnovers_per_game": model_mean("interceptions_mean") + model_mean("fumbles_lost_mean"),
    }

    schedules = nfl.load_schedules([args.history])
    regular = schedules.filter(
        pl.col("home_score").is_not_null() & pl.col("away_score").is_not_null()
    )
    if "game_type" in regular.columns:
        regular = regular.filter(pl.col("game_type") == "REG")
    historical_games = regular.height
    if historical_games <= 0:
        raise RuntimeError(f"No completed regular-season games found for {args.history}")

    pbp = nfl.load_pbp([args.history])
    if "season_type" in pbp.columns:
        pbp = pbp.filter(pl.col("season_type") == "REG")

    pass_td = pl.col("pass_touchdown").fill_null(0) == 1
    rush_td = pl.col("rush_touchdown").fill_null(0) == 1
    offensive_td = pass_td | rush_td
    any_td = pl.col("touchdown").fill_null(0) == 1
    non_offensive_td = any_td & ~offensive_td
    fg_attempt = (
        pl.col("field_goal_attempt").fill_null(0) == 1
        if "field_goal_attempt" in pbp.columns
        else pl.col("field_goal_result").is_not_null()
    )
    fg_made = pl.col("field_goal_result").cast(pl.Utf8).str.to_lowercase() == "made"
    xp_made = _safe_success(
        "extra_point_result", pbp, ["good", "made", "success", "successful"]
    )
    two_made = _safe_success(
        "two_point_conv_result", pbp, ["success", "good", "made", "successful"]
    )
    safety = (
        pl.col("safety").fill_null(0) == 1
        if "safety" in pbp.columns
        else pl.lit(False)
    )
    punt = (
        pl.col("punt_attempt").fill_null(0) == 1
        if "punt_attempt" in pbp.columns
        else pl.lit(False)
    )
    interception = (
        pl.col("interception").fill_null(0) == 1
        if "interception" in pbp.columns
        else pl.lit(False)
    )
    fumble_lost = (
        pl.col("fumble_lost").fill_null(0) == 1
        if "fumble_lost" in pbp.columns
        else pl.lit(False)
    )

    totals = pbp.select(
        offensive_td.cast(pl.Int64).sum().alias("offensive_tds"),
        non_offensive_td.cast(pl.Int64).sum().alias("non_offensive_tds"),
        fg_attempt.cast(pl.Int64).sum().alias("fg_attempts"),
        fg_made.cast(pl.Int64).sum().alias("fg_made"),
        xp_made.cast(pl.Int64).sum().alias("xp_made"),
        two_made.cast(pl.Int64).sum().alias("two_made"),
        safety.cast(pl.Int64).sum().alias("safeties"),
        punt.cast(pl.Int64).sum().alias("punts"),
        interception.cast(pl.Int64).sum().alias("interceptions"),
        fumble_lost.cast(pl.Int64).sum().alias("fumbles_lost"),
    ).row(0, named=True)

    actual_scoreboard_points = float(
        regular.get_column("home_score").sum() + regular.get_column("away_score").sum()
    )
    historical = {
        "points_per_game": actual_scoreboard_points / historical_games,
        "offensive_touchdowns_per_game": totals["offensive_tds"] / historical_games,
        "non_offensive_touchdowns_per_game": totals["non_offensive_tds"] / historical_games,
        "all_touchdowns_per_game": (totals["offensive_tds"] + totals["non_offensive_tds"])
        / historical_games,
        "fg_attempts_per_game": totals["fg_attempts"] / historical_games,
        "fg_made_per_game": totals["fg_made"] / historical_games,
        "xp_made_per_game": totals["xp_made"] / historical_games,
        "two_point_conversions_per_game": totals["two_made"] / historical_games,
        "safeties_per_game": totals["safeties"] / historical_games,
        "punts_per_game": totals["punts"] / historical_games,
        "turnovers_per_game": (totals["interceptions"] + totals["fumbles_lost"])
        / historical_games,
    }

    model_td_points = model["touchdowns_per_game"] * 7.0
    model_fg_points = model["fg_made_per_game"] * 3.0
    model_accounted = model_td_points + model_fg_points

    offensive_td_gap = historical["offensive_touchdowns_per_game"] - model["touchdowns_per_game"]
    fg_gap = historical["fg_made_per_game"] - model["fg_made_per_game"]
    non_offensive_points = (
        6.0 * historical["non_offensive_touchdowns_per_game"]
        + 2.0 * historical["safeties_per_game"]
    )
    scoreboard_gap = historical["points_per_game"] - model["points_per_game"]

    manifest = {
        "artifact": "Reality Loop v2 Same-World Score Anatomy",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "historical_season": args.history,
        "historical_games": historical_games,
        "simulated_matchups": anatomy.height,
        "model": model,
        "historical": historical,
        "gaps": {
            "scoreboard_points_per_game": scoreboard_gap,
            "offensive_touchdowns_per_game": offensive_td_gap,
            "approx_offensive_td_points_per_game": 7.0 * offensive_td_gap,
            "fg_made_per_game": fg_gap,
            "approx_fg_points_per_game": 3.0 * fg_gap,
            "historical_non_offensive_points_per_game_approx": non_offensive_points,
            "residual_after_offensive_td_fg_nonoffensive_approx": scoreboard_gap
            - 7.0 * offensive_td_gap
            - 3.0 * fg_gap
            - non_offensive_points,
        },
        "conservation": {
            "model_points_from_7x_td_plus_3x_fg": model_accounted,
            "model_scoreboard_minus_accounted": model["points_per_game"] - model_accounted,
        },
        "principle": "Name the missing scoring channels before changing football coefficients.",
    }

    out.mkdir(parents=True, exist_ok=True)
    (out / "score_anatomy.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
