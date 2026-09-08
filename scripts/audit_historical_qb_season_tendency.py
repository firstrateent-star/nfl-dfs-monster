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
    parser.add_argument("--out", type=Path, default=Path("artifacts/historical-qb-season-tendency"))
    args = parser.parse_args()
    seasons = [int(x) for x in args.seasons.split(",")]

    pbp = nfl.load_pbp(seasons)
    if "season_type" in pbp.columns:
        pbp = pbp.filter(pl.col("season_type") == "REG")

    required = {
        "season", "game_id", "posteam", "pass_attempt", "rush_attempt",
        "passer_player_id", "rusher_player_id", "rush_touchdown",
    }
    missing = required - set(pbp.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    lead_passers = (
        pbp.filter((pl.col("pass_attempt") == 1) & pl.col("passer_player_id").is_not_null())
        .group_by(["season", "game_id", "posteam", "passer_player_id"])
        .agg(pl.len().alias("pass_attempts"))
        .sort(["season", "game_id", "posteam", "pass_attempts"], descending=[False, False, False, True])
        .group_by(["season", "game_id", "posteam"], maintain_order=True)
        .first()
        .filter(pl.col("pass_attempts") >= 15)
        .rename({"passer_player_id": "qb_id"})
    )

    qb_rush = (
        pbp.filter((pl.col("rush_attempt") == 1) & pl.col("rusher_player_id").is_not_null())
        .group_by(["season", "game_id", "posteam", "rusher_player_id"])
        .agg(
            pl.len().alias("qb_rush_attempts"),
            pl.col("rush_touchdown").fill_null(0).sum().alias("qb_rush_tds"),
        )
        .rename({"rusher_player_id": "qb_id"})
    )

    team_rush = (
        pbp.filter((pl.col("rush_attempt") == 1) & pl.col("posteam").is_not_null())
        .group_by(["season", "game_id", "posteam"])
        .agg(pl.len().alias("team_rush_attempts"))
    )

    games = (
        lead_passers
        .join(qb_rush, on=["season", "game_id", "posteam", "qb_id"], how="left")
        .join(team_rush, on=["season", "game_id", "posteam"], how="left")
        .with_columns(
            pl.col("qb_rush_attempts").fill_null(0),
            pl.col("qb_rush_tds").fill_null(0),
        )
        .with_columns(
            (pl.col("qb_rush_attempts") / pl.col("team_rush_attempts").clip(lower_bound=1)).alias("qb_team_rush_share"),
            (pl.col("qb_rush_attempts") / pl.col("pass_attempts").clip(lower_bound=1)).alias("rush_per_pass"),
        )
    )

    season_qb = (
        games.group_by(["season", "qb_id"])
        .agg(
            pl.len().alias("lead_passer_games"),
            pl.col("pass_attempts").sum().alias("pass_attempts"),
            pl.col("qb_rush_attempts").sum().alias("qb_rush_attempts"),
            pl.col("qb_rush_tds").sum().alias("qb_rush_tds"),
            pl.col("team_rush_attempts").sum().alias("team_rush_attempts"),
            pl.col("qb_team_rush_share").mean().alias("mean_game_qb_team_rush_share"),
        )
        .filter(pl.col("lead_passer_games") >= 4)
        .with_columns(
            (pl.col("qb_rush_attempts") / pl.col("lead_passer_games")).alias("rush_attempts_per_start"),
            (pl.col("qb_rush_tds") / pl.col("lead_passer_games")).alias("rush_tds_per_start"),
            (pl.col("qb_rush_attempts") / pl.col("team_rush_attempts").clip(lower_bound=1)).alias("season_qb_team_rush_share"),
            (pl.col("qb_rush_attempts") / pl.col("pass_attempts").clip(lower_bound=1)).alias("season_rush_per_pass"),
        )
        .sort(["season", "rush_attempts_per_start"], descending=[False, True])
    )

    rushes = season_qb.get_column("rush_attempts_per_start").to_numpy()
    shares = season_qb.get_column("season_qb_team_rush_share").to_numpy()
    tds = season_qb.get_column("rush_tds_per_start").to_numpy()
    ratios = season_qb.get_column("season_rush_per_pass").to_numpy()

    corr = float(np.corrcoef(ratios, rushes)[0, 1]) if len(ratios) > 1 else float("nan")
    manifest = {
        "seasons": seasons,
        "qb_season_rows": int(season_qb.height),
        "minimum_lead_passer_games": 4,
        "rush_attempts_per_start": {
            "mean": float(np.mean(rushes)),
            "median": float(np.median(rushes)),
            "p75": float(np.quantile(rushes, 0.75)),
            "p90": float(np.quantile(rushes, 0.90)),
            "p95": float(np.quantile(rushes, 0.95)),
        },
        "season_qb_team_rush_share": {
            "mean": float(np.mean(shares)),
            "median": float(np.median(shares)),
            "p75": float(np.quantile(shares, 0.75)),
            "p90": float(np.quantile(shares, 0.90)),
            "p95": float(np.quantile(shares, 0.95)),
        },
        "rush_tds_per_start": {
            "mean": float(np.mean(tds)),
            "median": float(np.median(tds)),
            "p75": float(np.quantile(tds, 0.75)),
            "p90": float(np.quantile(tds, 0.90)),
            "p95": float(np.quantile(tds, 0.95)),
        },
        "corr_season_rush_per_pass_vs_rush_attempts_per_start": corr,
        "market_blind": True,
        "principle": "Calibrate future-world QB means to QB-season starter means; reserve game-level extremes for the simulated tail.",
    }

    args.out.mkdir(parents=True, exist_ok=True)
    season_qb.write_csv(args.out / "qb_season_tendency.csv")
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))
    print(season_qb.sort("rush_attempts_per_start", descending=True).head(30))


if __name__ == "__main__":
    main()
