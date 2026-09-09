from __future__ import annotations

import argparse
import json
from pathlib import Path

import polars as pl


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--players", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("artifacts/fanduel-scoring-gate"))
    args = parser.parse_args()

    players = pl.read_csv(args.players)
    required = {
        "passing_yards_mean", "passing_tds_mean", "rushing_yards_mean", "rushing_tds_mean",
        "receptions_mean", "receiving_yards_mean", "receiving_tds_mean",
    }
    missing = sorted(required - set(players.columns))
    turnover_cols = [c for c in players.columns if c in {"interceptions_mean", "interceptions_thrown_mean", "fumbles_lost_mean"}]
    partial_fd = "fd_points_before_turnover_penalties_mean" in players.columns
    complete = not missing and partial_fd and len(turnover_cols) >= 2
    manifest = {
        "artifact": "Monster FanDuel Scoring Completeness Gate",
        "player_rows": players.height,
        "partial_fanduel_scoring_present": partial_fd,
        "missing_core_scoring_columns": missing,
        "turnover_attribution_columns_present": turnover_cols,
        "complete_fanduel_scoring_ready": complete,
        "gate_pass": complete,
        "principle": "Salary and optimizer work may consume football outputs only after FanDuel scoring is complete; missing player-level turnover attribution cannot be silently treated as zero.",
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))
    if not complete:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
