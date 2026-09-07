from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import nflreadpy as nfl
import polars as pl


def _as_bool(column: str, frame: pl.DataFrame) -> pl.Expr:
    if column not in frame.columns:
        return pl.lit(False)
    return pl.col(column).fill_null(0).cast(pl.Int8) == 1


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--season", type=int, default=2025)
    parser.add_argument("--out", type=Path, default=Path("artifacts/score-components"))
    args = parser.parse_args()

    pbp = nfl.load_pbp([args.season])
    schedules = nfl.load_schedules([args.season])
    regular = schedules.filter(pl.col("home_score").is_not_null() & pl.col("away_score").is_not_null())
    if "game_type" in regular.columns:
        regular = regular.filter(pl.col("game_type") == "REG")

    games = regular.height
    team_games = games * 2
    scoreboard_points = float((regular.get_column("home_score").sum() + regular.get_column("away_score").sum()))

    pass_td = _as_bool("pass_touchdown", pbp)
    rush_td = _as_bool("rush_touchdown", pbp)
    any_td = _as_bool("touchdown", pbp)
    offensive_td = pass_td | rush_td
    other_td = any_td & ~offensive_td

    fg_made = (
        (pl.col("field_goal_result") == "made")
        if "field_goal_result" in pbp.columns
        else pl.lit(False)
    )
    xp_made = (
        pl.col("extra_point_result").cast(pl.Utf8).str.to_lowercase().is_in(["good", "made", "success"])
        if "extra_point_result" in pbp.columns
        else pl.lit(False)
    )
    two_pt = (
        pl.col("two_point_conv_result").cast(pl.Utf8).str.to_lowercase().is_in(["success", "good", "made"])
        if "two_point_conv_result" in pbp.columns
        else pl.lit(False)
    )
    safety = _as_bool("safety", pbp)

    counts = pbp.select(
        offensive_td.cast(pl.Int64).sum().alias("offensive_tds"),
        other_td.cast(pl.Int64).sum().alias("other_tds"),
        fg_made.cast(pl.Int64).sum().alias("field_goals"),
        xp_made.cast(pl.Int64).sum().alias("extra_points"),
        two_pt.cast(pl.Int64).sum().alias("two_point_conversions"),
        safety.cast(pl.Int64).sum().alias("safeties"),
    ).row(0, named=True)

    reconstructed = (
        6 * counts["offensive_tds"]
        + 6 * counts["other_tds"]
        + 3 * counts["field_goals"]
        + counts["extra_points"]
        + 2 * counts["two_point_conversions"]
        + 2 * counts["safeties"]
    )

    return_td_misattribution = None
    if "td_team" in pbp.columns and "posteam" in pbp.columns:
        return_td_misattribution = int(
            pbp.select(
                (any_td & pl.col("td_team").is_not_null() & (pl.col("td_team") != pl.col("posteam")))
                .cast(pl.Int64)
                .sum()
            ).item()
        )

    result = {
        "artifact": "Monster Historical Score Component Audit",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "season": args.season,
        "games": games,
        "team_games": team_games,
        "scoreboard_points": scoreboard_points,
        "scoreboard_points_per_team_game": scoreboard_points / team_games,
        "offensive_tds": counts["offensive_tds"],
        "other_tds": counts["other_tds"],
        "field_goals": counts["field_goals"],
        "extra_points": counts["extra_points"],
        "two_point_conversions": counts["two_point_conversions"],
        "safeties": counts["safeties"],
        "offensive_td_points_per_team_game": 6 * counts["offensive_tds"] / team_games,
        "other_td_points_per_team_game": 6 * counts["other_tds"] / team_games,
        "field_goal_points_per_team_game": 3 * counts["field_goals"] / team_games,
        "extra_point_points_per_team_game": counts["extra_points"] / team_games,
        "two_point_points_per_team_game": 2 * counts["two_point_conversions"] / team_games,
        "safety_points_per_team_game": 2 * counts["safeties"] / team_games,
        "reconstructed_points_per_team_game": reconstructed / team_games,
        "scoreboard_minus_reconstructed_per_team_game": (scoreboard_points - reconstructed) / team_games,
        "touchdowns_where_td_team_differs_from_posteam": return_td_misattribution,
        "generic_drive_td_prior_risk": bool(return_td_misattribution and return_td_misattribution > 0),
        "principle": "Before calibrating the simulator, verify that historical scoring labels represent the scoring team and causal score anatomy.",
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "manifest.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
