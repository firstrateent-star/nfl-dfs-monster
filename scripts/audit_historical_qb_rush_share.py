from __future__ import annotations

import argparse
import json
from pathlib import Path

import nflreadpy as nfl
import numpy as np
import polars as pl


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seasons", default="2022,2023,2024,2025")
    parser.add_argument("--out", type=Path, default=Path("artifacts/historical-qb-rush-share"))
    args = parser.parse_args()
    seasons = [int(x) for x in args.seasons.split(",")]

    pbp = nfl.load_pbp(seasons)
    if "season_type" in pbp.columns:
        pbp = pbp.filter(pl.col("season_type") == "REG")

    required = {"season", "game_id", "posteam", "pass_attempt", "rush_attempt", "passer_player_id", "rusher_player_id"}
    missing = required - set(pbp.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    passers = (
        pbp.filter((pl.col("pass_attempt") == 1) & pl.col("passer_player_id").is_not_null())
        .group_by(["season", "game_id", "posteam", "passer_player_id"])
        .agg(pl.len().alias("pass_attempts"))
        .sort(["season", "game_id", "posteam", "pass_attempts"], descending=[False, False, False, True])
        .group_by(["season", "game_id", "posteam"], maintain_order=True)
        .first()
        .rename({"passer_player_id": "qb_id"})
    )

    rushers = (
        pbp.filter((pl.col("rush_attempt") == 1) & pl.col("rusher_player_id").is_not_null())
        .group_by(["season", "game_id", "posteam", "rusher_player_id"])
        .agg(pl.len().alias("rush_attempts"))
        .rename({"rusher_player_id": "qb_id"})
    )
    team_rush = (
        pbp.filter((pl.col("rush_attempt") == 1) & pl.col("posteam").is_not_null())
        .group_by(["season", "game_id", "posteam"])
        .agg(pl.len().alias("team_rush_attempts"))
    )

    games = (
        passers.join(rushers, on=["season", "game_id", "posteam", "qb_id"], how="left")
        .join(team_rush, on=["season", "game_id", "posteam"], how="left")
        .with_columns(pl.col("rush_attempts").fill_null(0))
        .filter(pl.col("pass_attempts") >= 15)
        .with_columns(
            (pl.col("rush_attempts") / pl.col("team_rush_attempts").clip(lower_bound=1)).alias("qb_team_rush_share")
        )
    )

    shares = games.get_column("qb_team_rush_share").to_numpy()
    rushes = games.get_column("rush_attempts").to_numpy()
    manifest = {
        "seasons": seasons,
        "starter_game_rows": int(games.height),
        "mean_qb_team_rush_share": float(np.mean(shares)),
        "median_qb_team_rush_share": float(np.median(shares)),
        "p75_qb_team_rush_share": float(np.quantile(shares, 0.75)),
        "p90_qb_team_rush_share": float(np.quantile(shares, 0.90)),
        "p95_qb_team_rush_share": float(np.quantile(shares, 0.95)),
        "p99_qb_team_rush_share": float(np.quantile(shares, 0.99)),
        "mean_qb_rush_attempts": float(np.mean(rushes)),
        "median_qb_rush_attempts": float(np.median(rushes)),
        "p90_qb_rush_attempts": float(np.quantile(rushes, 0.90)),
        "p95_qb_rush_attempts": float(np.quantile(rushes, 0.95)),
        "market_blind": True,
        "principle": "Calibrate the QB carry reservoir from observed starting-QB football, not generic skill-position normalization.",
    }
    args.out.mkdir(parents=True, exist_ok=True)
    games.write_csv(args.out / "starter_qb_game_rush_share.csv")
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
