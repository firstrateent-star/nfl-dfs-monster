from __future__ import annotations

import argparse
import json
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


def _schedule_columns(frame: pl.DataFrame) -> tuple[str, str, str, str]:
    candidates = [
        ("home_team", "away_team", "home_score", "away_score"),
        ("home_team", "away_team", "home_points", "away_points"),
    ]
    for cols in candidates:
        if all(column in frame.columns for column in cols):
            return cols
    raise ValueError(f"Could not identify score columns in schedules: {frame.columns}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--season", type=int, default=2025)
    parser.add_argument("--worlds", type=int, default=2500)
    parser.add_argument("--seed", type=int, default=2025090701)
    parser.add_argument("--out", type=Path, default=Path("artifacts/score-calibration"))
    args = parser.parse_args()

    policy = _read(args.policy)
    schedules = nfl.load_schedules([args.season])
    home_col, away_col, home_score_col, away_score_col = _schedule_columns(schedules)
    regular = schedules.filter(
        pl.col(home_score_col).is_not_null() & pl.col(away_score_col).is_not_null()
    )
    if "game_type" in regular.columns:
        regular = regular.filter(pl.col("game_type") == "REG")
    elif "week" in regular.columns:
        regular = regular.filter(pl.col("week").cast(pl.Utf8).str.contains(r"^\d+$"))

    policy_teams = set(policy.get_column("team_id").to_list())
    rows: list[dict] = []
    skipped: list[dict] = []
    for idx, row in enumerate(regular.to_dicts()):
        home = normalize_team_id(str(row[home_col]))
        away = normalize_team_id(str(row[away_col]))
        if home not in policy_teams or away not in policy_teams:
            skipped.append({"home": home, "away": away, "week": row.get("week")})
            continue
        states = compile_team_state_map(policy, {away: home, home: away}, prior_uncertainty=0.04)
        result = simulate_game(
            GameState(f"{away}@{home}", states[away], states[home]),
            worlds=args.worlds,
            seed=args.seed + idx * 101,
        )
        actual_home = float(row[home_score_col])
        actual_away = float(row[away_score_col])
        actual_total = actual_home + actual_away
        model_home = float(result.home_points.mean())
        model_away = float(result.away_points.mean())
        model_total = model_home + model_away
        rows.append(
            {
                "game_id": str(row.get("game_id") or f"{away}@{home}"),
                "week": row.get("week"),
                "away": away,
                "home": home,
                "actual_away": actual_away,
                "actual_home": actual_home,
                "actual_total": actual_total,
                "model_away": model_away,
                "model_home": model_home,
                "model_total": model_total,
                "total_error": model_total - actual_total,
                "actual_margin": actual_away - actual_home,
                "model_margin": model_away - model_home,
                "actual_points_per_team": actual_total / 2.0,
                "model_points_per_team": model_total / 2.0,
            }
        )

    audit = pl.DataFrame(rows)
    if not audit.height:
        raise RuntimeError("No historical games available for score calibration")
    args.out.mkdir(parents=True, exist_ok=True)
    audit.write_csv(args.out / "game_replay_audit.csv")
    if skipped:
        pl.DataFrame(skipped).write_csv(args.out / "skipped_games.csv")

    team_actual = np.concatenate(
        [audit.get_column("actual_away").to_numpy(), audit.get_column("actual_home").to_numpy()]
    )
    team_model = np.concatenate(
        [audit.get_column("model_away").to_numpy(), audit.get_column("model_home").to_numpy()]
    )
    total_actual = audit.get_column("actual_total").to_numpy()
    total_model = audit.get_column("model_total").to_numpy()
    manifest = {
        "artifact": "Monster Historical Score Anatomy Calibration Audit",
        "season": args.season,
        "regular_games_available": regular.height,
        "games": audit.height,
        "skipped_games": len(skipped),
        "worlds_per_game": args.worlds,
        "actual_points_per_team_mean": float(team_actual.mean()),
        "model_points_per_team_mean": float(team_model.mean()),
        "team_mean_bias": float((team_model - team_actual).mean()),
        "team_mae": float(np.abs(team_model - team_actual).mean()),
        "actual_game_total_mean": float(total_actual.mean()),
        "model_game_total_mean": float(total_model.mean()),
        "game_total_mean_bias": float((total_model - total_actual).mean()),
        "game_total_mae": float(np.abs(total_model - total_actual).mean()),
        "actual_game_total_sd": float(total_actual.std()),
        "model_game_total_sd_of_means": float(total_model.std()),
        "calibration_pass": (
            len(skipped) == 0 and abs(float((total_model - total_actual).mean())) <= 2.5
        ),
        "principle": "Calibrate causal score anatomy against historical football, never against DFS salary or ownership.",
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))
    print(audit.sort("total_error").head(20))
    print(audit.sort("total_error", descending=True).head(20))


if __name__ == "__main__":
    main()
