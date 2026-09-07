from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import nflreadpy as nfl
import numpy as np
import polars as pl

from monster.feature_compile.team import compile_team_policy
from monster.ingest.nflverse import PBP_COLUMNS
from monster.sim.game import simulate_game
from monster.snapshot.league import compile_team_state_map
from monster.snapshot.model import GameState
from monster.teams import NFL_TEAMS, normalize_team_id


def _regular_pbp(season: int) -> pl.DataFrame:
    pbp = nfl.load_pbp([season])
    pbp = pbp.select([c for c in PBP_COLUMNS if c in pbp.columns])
    if "season_type" in pbp.columns:
        pbp = pbp.filter(pl.col("season_type") == "REG")
    return pbp


def _regular_schedule(season: int) -> pl.DataFrame:
    schedules = nfl.load_schedules([season])
    out = schedules.filter(pl.col("home_score").is_not_null() & pl.col("away_score").is_not_null())
    if "game_type" in out.columns:
        out = out.filter(pl.col("game_type") == "REG")
    return out


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 2 or np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _train_league_scoring_mean(season: int) -> float:
    schedule = _regular_schedule(season)
    total = schedule.get_column("home_score").sum() + schedule.get_column("away_score").sum()
    return float(total / schedule.height)


def run_fold(train_season: int, test_season: int, worlds: int, seed: int) -> tuple[dict, pl.DataFrame]:
    pbp = _regular_pbp(train_season)
    policy_raw = compile_team_policy(pbp)
    canonical = pl.DataFrame({"team_id": list(NFL_TEAMS)})
    policy = canonical.join(policy_raw, on="team_id", how="left").sort("team_id")
    schedule = _regular_schedule(test_season)
    train_total_mean = _train_league_scoring_mean(train_season)

    rows: list[dict] = []
    for idx, row in enumerate(schedule.to_dicts()):
        away = normalize_team_id(str(row["away_team"]))
        home = normalize_team_id(str(row["home_team"]))
        states = compile_team_state_map(policy, {away: home, home: away}, prior_uncertainty=0.16)
        result = simulate_game(
            GameState(f"{away}@{home}", states[away], states[home]),
            worlds=worlds,
            seed=seed + idx * 103,
        )
        actual_away = float(row["away_score"])
        actual_home = float(row["home_score"])
        model_away = float(result.away_points.mean())
        model_home = float(result.home_points.mean())
        rows.append(
            {
                "train_season": train_season,
                "test_season": test_season,
                "week": row.get("week"),
                "game_id": str(row.get("game_id") or f"{away}@{home}"),
                "away": away,
                "home": home,
                "actual_away": actual_away,
                "actual_home": actual_home,
                "actual_total": actual_away + actual_home,
                "actual_margin": actual_away - actual_home,
                "model_away": model_away,
                "model_home": model_home,
                "model_total": model_away + model_home,
                "model_margin": model_away - model_home,
                "baseline_total": train_total_mean,
                "baseline_margin": 0.0,
            }
        )

    frame = pl.DataFrame(rows)
    actual_total = frame.get_column("actual_total").to_numpy()
    model_total = frame.get_column("model_total").to_numpy()
    baseline_total = frame.get_column("baseline_total").to_numpy()
    actual_margin = frame.get_column("actual_margin").to_numpy()
    model_margin = frame.get_column("model_margin").to_numpy()
    actual_team = np.concatenate(
        [frame.get_column("actual_away").to_numpy(), frame.get_column("actual_home").to_numpy()]
    )
    model_team = np.concatenate(
        [frame.get_column("model_away").to_numpy(), frame.get_column("model_home").to_numpy()]
    )

    metrics = {
        "train_season": train_season,
        "test_season": test_season,
        "games": frame.height,
        "worlds_per_game": worlds,
        "train_league_total_mean_baseline": train_total_mean,
        "actual_test_total_mean": float(actual_total.mean()),
        "model_total_mean": float(model_total.mean()),
        "model_total_bias": float((model_total - actual_total).mean()),
        "model_total_mae": float(np.abs(model_total - actual_total).mean()),
        "baseline_total_mae": float(np.abs(baseline_total - actual_total).mean()),
        "total_mae_improvement_vs_baseline": float(
            np.abs(baseline_total - actual_total).mean() - np.abs(model_total - actual_total).mean()
        ),
        "game_total_correlation": _corr(model_total, actual_total),
        "team_score_correlation": _corr(model_team, actual_team),
        "margin_correlation": _corr(model_margin, actual_margin),
        "model_margin_mae": float(np.abs(model_margin - actual_margin).mean()),
        "zero_margin_baseline_mae": float(np.abs(actual_margin).mean()),
        "margin_mae_improvement_vs_zero": float(
            np.abs(actual_margin).mean() - np.abs(model_margin - actual_margin).mean()
        ),
    }
    return metrics, frame


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", action="append", default=["2023:2024", "2024:2025"])
    parser.add_argument("--worlds", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=2026090719)
    parser.add_argument("--out", type=Path, default=Path("artifacts/score-oos"))
    args = parser.parse_args()

    fold_metrics: list[dict] = []
    fold_frames: list[pl.DataFrame] = []
    for fold_idx, spec in enumerate(args.fold):
        train, test = (int(x) for x in spec.split(":"))
        metrics, frame = run_fold(train, test, args.worlds, args.seed + fold_idx * 100_003)
        fold_metrics.append(metrics)
        fold_frames.append(frame)

    combined = pl.concat(fold_frames)
    mean_total_gain = float(np.mean([m["total_mae_improvement_vs_baseline"] for m in fold_metrics]))
    mean_margin_gain = float(np.mean([m["margin_mae_improvement_vs_zero"] for m in fold_metrics]))
    mean_total_corr = float(np.mean([m["game_total_correlation"] for m in fold_metrics]))
    mean_margin_corr = float(np.mean([m["margin_correlation"] for m in fold_metrics]))

    manifest = {
        "artifact": "Monster Year-Forward Market-Blind Score Validation",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "folds": fold_metrics,
        "mean_total_mae_improvement_vs_historical_mean_baseline": mean_total_gain,
        "mean_margin_mae_improvement_vs_zero_baseline": mean_margin_gain,
        "mean_game_total_correlation": mean_total_corr,
        "mean_margin_correlation": mean_margin_corr,
        "predictive_structure_pass": (
            mean_total_gain > 0.0
            and mean_margin_gain > 0.0
            and mean_total_corr > 0.10
            and mean_margin_corr > 0.10
        ),
        "market_blind": True,
        "note": "Each test season is simulated using only the prior season's aggregate football policy; no test-season PBP, market totals, salaries, ownership, or fantasy projections enter the predictor.",
    }
    args.out.mkdir(parents=True, exist_ok=True)
    combined.write_csv(args.out / "year_forward_games.csv")
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
