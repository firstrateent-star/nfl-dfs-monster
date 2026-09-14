from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import polars as pl


def _world_totals(path: Path) -> np.ndarray:
    file = path / "football_weirdness_worlds.csv"
    if not file.exists():
        return np.asarray([], dtype=float)
    frame = pl.read_csv(file)
    columns = set(frame.columns)
    if {"away_points", "home_points"}.issubset(columns):
        return (
            frame.get_column("away_points").cast(pl.Float64)
            + frame.get_column("home_points").cast(pl.Float64)
        ).to_numpy()
    for name in ("total_points", "total", "game_total"):
        if name in columns:
            return frame.get_column(name).cast(pl.Float64).to_numpy()
    return np.asarray([], dtype=float)


def _tail(values: np.ndarray) -> dict[str, float]:
    if len(values) == 0:
        return {}
    return {
        "sd": float(values.std(ddof=1)),
        "p05": float(np.quantile(values, 0.05)),
        "p50": float(np.quantile(values, 0.50)),
        "p95": float(np.quantile(values, 0.95)),
        "p99": float(np.quantile(values, 0.99)),
        "max": float(values.max()),
        "rate_70_plus": float(np.mean(values >= 70.0)),
        "rate_80_plus": float(np.mean(values >= 80.0)),
        "rate_30_or_less": float(np.mean(values <= 30.0)),
    }


def _summary(path: Path) -> dict[str, object]:
    score = json.loads((path / "score_anatomy.json").read_text())["model"]
    games = pl.read_csv(path / "game_distributions.csv")
    return {
        "points_per_game": float(score["points_per_game"]),
        "scrimmage_plays_per_game": float(score["scrimmage_plays_per_game"]),
        "drives_per_game": float(score["drives_per_game"]),
        "turnovers_per_game": float(score["turnovers_per_game"]),
        "offensive_touchdowns_per_game": float(score["offensive_touchdowns_per_game"]),
        "defensive_touchdowns_per_game": float(score["defensive_touchdowns_per_game"]),
        "special_teams_touchdowns_per_game": float(score["special_teams_touchdowns_per_game"]),
        "non_offensive_touchdowns_per_game": float(score["non_offensive_touchdowns_per_game"]),
        "explosive_40_per_game": float(score["explosive_40_per_game"]),
        "between_matchup_total_sd": float(games.get_column("total_mean").std(ddof=1)),
        "world_total_tail": _tail(_world_totals(path)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--role-summary", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    baseline = _summary(args.baseline)
    candidate = _summary(args.candidate)
    comparison: dict[str, object] = {
        "baseline": baseline,
        "candidate": candidate,
        "delta": {
            "points_per_game": candidate["points_per_game"] - baseline["points_per_game"],
            "scrimmage_plays_per_game": candidate["scrimmage_plays_per_game"]
            - baseline["scrimmage_plays_per_game"],
            "non_offensive_touchdowns_per_game": candidate[
                "non_offensive_touchdowns_per_game"
            ]
            - baseline["non_offensive_touchdowns_per_game"],
            "between_matchup_total_sd": candidate["between_matchup_total_sd"]
            - baseline["between_matchup_total_sd"],
        },
    }
    if args.role_summary and args.role_summary.exists():
        comparison["role_snap_opportunity"] = json.loads(args.role_summary.read_text())

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(comparison, indent=2) + "\n")
    print(json.dumps(comparison, indent=2))


if __name__ == "__main__":
    main()
