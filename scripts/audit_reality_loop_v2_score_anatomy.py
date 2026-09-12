from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
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


def _mean(frame: pl.DataFrame, column: str) -> float:
    return float(frame.select(pl.col(column).mean()).item())


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
    weirdness_path = sim_dir / "football_weirdness.csv"
    weirdness = pl.read_csv(weirdness_path) if weirdness_path.exists() else None

    def model_mean(column: str) -> float:
        return _mean(anatomy, column)

    def detailed_mean(metric: str, fallback: float = 0.0) -> float:
        column = f"{metric}_mean"
        if weirdness is None or column not in weirdness.columns:
            return fallback
        return _mean(weirdness, column)

    all_tds = model_mean("touchdowns_mean")
    offensive_tds = detailed_mean("offensive_touchdowns", all_tds)
    defensive_tds = detailed_mean("defensive_touchdowns")
    special_teams_tds = detailed_mean("special_teams_touchdowns")
    safeties = detailed_mean("safeties")
    try_points = detailed_mean("try_points", max(0.0, all_tds - 0.02))

    model = {
        "points_per_game": float(games.select(pl.col("total_mean").mean()).item()),
        "all_touchdowns_per_game": all_tds,
        "offensive_touchdowns_per_game": offensive_tds,
        "defensive_touchdowns_per_game": defensive_tds,
        "special_teams_touchdowns_per_game": special_teams_tds,
        "non_offensive_touchdowns_per_game": defensive_tds + special_teams_tds,
        "safeties_per_game": safeties,
        "try_points_per_game": try_points,
        "fg_attempts_per_game": model_mean("field_goal_attempts_mean"),
        "fg_made_per_game": model_mean("field_goals_made_mean"),
        "punts_per_game": model_mean("punts_mean"),
        "drives_per_game": model_mean("drives_mean"),
        "scrimmage_plays_per_game": model_mean("scrimmage_plays_mean"),
        "turnovers_per_game": model_mean("interceptions_mean") + model_mean("fumbles_lost_mean"),
        "turnovers_on_downs_per_game": detailed_mean("turnovers_on_downs"),
        "short_field_drives_per_game": detailed_mean("short_field_drives"),
        "drive_start_yardline_mean": detailed_mean("drive_start_yardline_mean"),
        "turnover_drive_start_yardline_mean": detailed_mean("turnover_drive_start_yardline_mean"),
        "explosive_15_per_game": detailed_mean("explosive_15"),
        "explosive_20_per_game": detailed_mean("explosive_20"),
        "explosive_40_per_game": detailed_mean("explosive_40"),
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
    explosive_15 = (
        pl.col("yards_gained").fill_null(0.0) >= 15.0
        if "yards_gained" in pbp.columns
        else pl.lit(False)
    )
    explosive_20 = (
        pl.col("yards_gained").fill_null(0.0) >= 20.0
        if "yards_gained" in pbp.columns
        else pl.lit(False)
    )
    explosive_40 = (
        pl.col("yards_gained").fill_null(0.0) >= 40.0
        if "yards_gained" in pbp.columns
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
        explosive_15.cast(pl.Int64).sum().alias("explosive_15"),
        explosive_20.cast(pl.Int64).sum().alias("explosive_20"),
        explosive_40.cast(pl.Int64).sum().alias("explosive_40"),
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
        "try_points_per_game": (totals["xp_made"] + 2 * totals["two_made"])
        / historical_games,
        "safeties_per_game": totals["safeties"] / historical_games,
        "punts_per_game": totals["punts"] / historical_games,
        "turnovers_per_game": (totals["interceptions"] + totals["fumbles_lost"])
        / historical_games,
        "explosive_15_per_game": totals["explosive_15"] / historical_games,
        "explosive_20_per_game": totals["explosive_20"] / historical_games,
        "explosive_40_per_game": totals["explosive_40"] / historical_games,
    }

    score_channels = {
        "offensive_td_points": 6.0 * model["offensive_touchdowns_per_game"],
        "non_offensive_td_points": 6.0 * model["non_offensive_touchdowns_per_game"],
        "field_goal_points": 3.0 * model["fg_made_per_game"],
        "try_points": model["try_points_per_game"],
        "safety_points": 2.0 * model["safeties_per_game"],
    }
    historical_score_channels = {
        "offensive_td_points": 6.0 * historical["offensive_touchdowns_per_game"],
        "non_offensive_td_points": 6.0 * historical["non_offensive_touchdowns_per_game"],
        "field_goal_points": 3.0 * historical["fg_made_per_game"],
        "try_points": historical["try_points_per_game"],
        "safety_points": 2.0 * historical["safeties_per_game"],
    }
    channel_gaps = {
        key: historical_score_channels[key] - score_channels[key] for key in score_channels
    }
    scoreboard_gap = historical["points_per_game"] - model["points_per_game"]

    model_accounted = sum(score_channels.values())
    historical_accounted = sum(historical_score_channels.values())
    manifest = {
        "artifact": "Reality Loop v2 Same-World Score Anatomy",
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "historical_season": args.history,
        "historical_games": historical_games,
        "simulated_matchups": anatomy.height,
        "model": model,
        "historical": historical,
        "score_channels": {
            "model": score_channels,
            "historical": historical_score_channels,
            "gap": channel_gaps,
        },
        "gaps": {
            "scoreboard_points_per_game": scoreboard_gap,
            "punts_per_game": historical["punts_per_game"] - model["punts_per_game"],
            "turnovers_per_game": historical["turnovers_per_game"] - model["turnovers_per_game"],
            "fg_attempts_per_game": historical["fg_attempts_per_game"] - model["fg_attempts_per_game"],
            "offensive_touchdowns_per_game": historical["offensive_touchdowns_per_game"]
            - model["offensive_touchdowns_per_game"],
            "non_offensive_touchdowns_per_game": historical["non_offensive_touchdowns_per_game"]
            - model["non_offensive_touchdowns_per_game"],
            "explosive_15_per_game": historical["explosive_15_per_game"]
            - model["explosive_15_per_game"],
            "explosive_20_per_game": historical["explosive_20_per_game"]
            - model["explosive_20_per_game"],
            "explosive_40_per_game": historical["explosive_40_per_game"]
            - model["explosive_40_per_game"],
            "sum_scoring_channel_gap": sum(channel_gaps.values()),
            "residual_after_scoring_channels": scoreboard_gap - sum(channel_gaps.values()),
        },
        "conservation": {
            "model_event_scoring_reconstruction": model_accounted,
            "model_scoreboard_minus_reconstruction": model["points_per_game"] - model_accounted,
            "historical_event_scoring_reconstruction": historical_accounted,
            "historical_scoreboard_minus_reconstruction": historical["points_per_game"]
            - historical_accounted,
        },
        "principle": "Name the missing scoring and drive-terminal channels before changing football coefficients.",
    }

    out.mkdir(parents=True, exist_ok=True)
    (out / "score_anatomy.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
