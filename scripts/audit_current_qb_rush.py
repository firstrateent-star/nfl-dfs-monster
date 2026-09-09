from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import polars as pl


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--players", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("artifacts/current-qb-rush-audit"))
    args = parser.parse_args()

    players = pl.read_csv(args.players)
    starters = players.filter(
        (pl.col("position") == "QB") & (pl.col("pass_attempts_mean") >= 15.0)
    ).sort("rush_attempts_mean", descending=True)
    if starters.height < 20:
        raise ValueError(f"Expected at least 20 current starting-QB rows, found {starters.height}")

    attempts = starters.get_column("rush_attempts_mean").to_numpy()
    metrics = {
        "starter_qb_count": int(starters.height),
        "mean_rush_attempts": float(np.mean(attempts)),
        "median_rush_attempts": float(np.median(attempts)),
        "p90_rush_attempts": float(np.quantile(attempts, 0.90)),
        "max_rush_attempts": float(np.max(attempts)),
    }
    metrics["distribution_gate_pass"] = bool(
        1.5 <= metrics["median_rush_attempts"] <= 7.5
        and metrics["p90_rush_attempts"] <= 10.5
        and metrics["max_rush_attempts"] <= 11.0
    )
    metrics["principle"] = (
        "QB rushing must remain a player-tendency channel; current-team RB context cannot "
        "manufacture double-digit QB carry means."
    )

    args.out.mkdir(parents=True, exist_ok=True)
    starters.write_csv(args.out / "starting_qb_rush_distribution.csv")
    (args.out / "manifest.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(starters.select(["game", "team_id", "player", "pass_attempts_mean", "rush_attempts_mean", "rushing_yards_mean", "rushing_tds_mean"]))
    print(json.dumps(metrics, indent=2))

    if not metrics["distribution_gate_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
