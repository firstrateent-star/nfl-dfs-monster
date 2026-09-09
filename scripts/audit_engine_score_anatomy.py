from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import nflreadpy as nfl
import numpy as np
import polars as pl

from monster.sim.game import simulate_game
from monster.snapshot.league import compile_team_state_map
from monster.snapshot.model import GameState
from monster.teams import normalize_team_id


def _read(path: Path) -> pl.DataFrame:
    return pl.read_parquet(path) if path.suffix == ".parquet" else pl.read_csv(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--season", type=int, default=2025)
    parser.add_argument("--worlds", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=2025090717)
    parser.add_argument("--out", type=Path, default=Path("artifacts/engine-score-anatomy"))
    args = parser.parse_args()

    policy = _read(args.policy)
    schedules = nfl.load_schedules([args.season])
    regular = schedules.filter(
        pl.col("home_score").is_not_null() & pl.col("away_score").is_not_null()
    )
    if "game_type" in regular.columns:
        regular = regular.filter(pl.col("game_type") == "REG")

    model_rows: list[dict] = []
    for idx, row in enumerate(regular.to_dicts()):
        home = normalize_team_id(str(row["home_team"]))
        away = normalize_team_id(str(row["away_team"]))
        states = compile_team_state_map(policy, {away: home, home: away}, prior_uncertainty=0.04)
        result = simulate_game(
            GameState(f"{away}@{home}", states[away], states[home]),
            worlds=args.worlds,
            seed=args.seed + idx * 101,
        )
        for side, team in (("away", away), ("home", home)):
            model_rows.append(
                {
                    "team_id": team,
                    "drives": float(getattr(result, f"{side}_drives").mean()),
                    "touchdowns": float(getattr(result, f"{side}_touchdowns").mean()),
                    "field_goals": float(getattr(result, f"{side}_field_goals").mean()),
                    "turnovers": float(getattr(result, f"{side}_turnovers").mean()),
                    "points": float(getattr(result, f"{side}_points").mean()),
                }
            )

    model = pl.DataFrame(model_rows)
    policy_summary = policy.select(
        pl.col("drives_per_game").mean().alias("historical_policy_drives_per_game"),
        pl.col("td_drive_rate").mean().alias("historical_policy_td_drive_rate_unweighted"),
        pl.col("fg_drive_rate").mean().alias("historical_policy_fg_drive_rate_unweighted"),
        pl.col("turnover_drive_rate").mean().alias("historical_policy_turnover_drive_rate_unweighted"),
        pl.col("red_zone_td_rate").mean().alias("historical_policy_red_zone_td_rate_unweighted"),
        pl.col("offensive_epa_per_play").mean().alias("historical_policy_offensive_epa_unweighted"),
        pl.col("defensive_epa_allowed_per_play").mean().alias("historical_policy_defensive_epa_unweighted"),
    ).row(0, named=True)

    model_means = model.select(
        pl.col("drives").mean().alias("model_drives_per_team_game"),
        pl.col("touchdowns").mean().alias("model_offensive_tds_per_team_game"),
        pl.col("field_goals").mean().alias("model_fgs_per_team_game"),
        pl.col("turnovers").mean().alias("model_turnovers_per_team_game"),
        pl.col("points").mean().alias("model_points_per_team_game"),
    ).row(0, named=True)

    pbp = nfl.load_pbp([args.season])
    pass_td = pl.col("pass_touchdown").fill_null(0) == 1
    rush_td = pl.col("rush_touchdown").fill_null(0) == 1
    offensive_td = pass_td | rush_td
    fg_made = pl.col("field_goal_result") == "made"
    extra_point_made = (
        pl.col("extra_point_result").cast(pl.Utf8).str.to_lowercase().is_in(["good", "made", "success"])
        if "extra_point_result" in pbp.columns
        else pl.lit(False)
    )
    two_pt_made = (
        pl.col("two_point_conv_result").cast(pl.Utf8).str.to_lowercase().is_in(["success", "good", "made"])
        if "two_point_conv_result" in pbp.columns
        else pl.lit(False)
    )
    any_td = pl.col("touchdown").fill_null(0) == 1
    other_td = any_td & ~offensive_td
    team_games = regular.height * 2
    actual = pbp.select(
        offensive_td.cast(pl.Int64).sum().alias("offensive_tds"),
        fg_made.cast(pl.Int64).sum().alias("field_goals"),
        extra_point_made.cast(pl.Int64).sum().alias("extra_points"),
        two_pt_made.cast(pl.Int64).sum().alias("two_points"),
        other_td.cast(pl.Int64).sum().alias("other_tds"),
    ).row(0, named=True)

    actual_points = float(
        regular.get_column("home_score").sum() + regular.get_column("away_score").sum()
    )
    actual_means = {
        "actual_offensive_tds_per_team_game": actual["offensive_tds"] / team_games,
        "actual_fgs_per_team_game": actual["field_goals"] / team_games,
        "actual_extra_points_per_team_game": actual["extra_points"] / team_games,
        "actual_two_points_per_team_game": actual["two_points"] / team_games,
        "actual_other_tds_per_team_game": actual["other_tds"] / team_games,
        "actual_scoreboard_points_per_team_game": actual_points / team_games,
    }

    td_deficit_points = 7.0 * (
        actual_means["actual_offensive_tds_per_team_game"]
        - model_means["model_offensive_tds_per_team_game"]
    )
    fg_deficit_points = 3.0 * (
        actual_means["actual_fgs_per_team_game"] - model_means["model_fgs_per_team_game"]
    )
    non_offensive_score_points = 6.0 * actual_means["actual_other_tds_per_team_game"]

    manifest = {
        "artifact": "Monster Engine Score Anatomy Conservation Audit",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "season": args.season,
        "games": regular.height,
        "team_games": team_games,
        "worlds_per_game": args.worlds,
        **policy_summary,
        **model_means,
        **actual_means,
        "approx_td_deficit_points_per_team_game": td_deficit_points,
        "approx_fg_deficit_points_per_team_game": fg_deficit_points,
        "approx_non_offensive_td_points_per_team_game": non_offensive_score_points,
        "residual_score_gap_after_td_fg_and_other_td_approx": (
            actual_means["actual_scoreboard_points_per_team_game"]
            - model_means["model_points_per_team_game"]
            - td_deficit_points
            - fg_deficit_points
            - non_offensive_score_points
        ),
        "principle": "Name the missing scoring channels before changing simulator coefficients.",
    }
    args.out.mkdir(parents=True, exist_ok=True)
    model.write_csv(args.out / "model_team_game_anatomy.csv")
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
