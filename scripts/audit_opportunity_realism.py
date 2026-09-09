from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import polars as pl

from monster.ingest.nflverse import configure_cache


def _read(path: Path) -> pl.DataFrame:
    return pl.read_parquet(path) if path.suffix == ".parquet" else pl.read_csv(path)


def _q(series: pl.Series, q: float) -> float:
    return float(series.quantile(q, interpolation="linear"))


def _league_summary(frame: pl.DataFrame, columns: list[str]) -> pl.DataFrame:
    rows = []
    for column in columns:
        s = frame.get_column(column).cast(pl.Float64).drop_nulls()
        rows.append({
            "metric": column,
            "historical_mean": float(s.mean()),
            "historical_sd": float(s.std()),
            "historical_p10": _q(s, 0.10),
            "historical_p50": _q(s, 0.50),
            "historical_p90": _q(s, 0.90),
        })
    return pl.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sim-opportunity", type=Path, required=True)
    parser.add_argument("--history", type=int, default=2025)
    parser.add_argument("--cache-dir", type=Path, default=Path(".cache/monster"))
    parser.add_argument("--out", type=Path, default=Path("artifacts/opportunity-realism"))
    args = parser.parse_args()

    configure_cache(args.cache_dir)
    import nflreadpy as nfl

    raw = nfl.load_pbp([args.history])
    if "season_type" in raw.columns:
        raw = raw.filter(pl.col("season_type") == "REG")

    needed = [
        "game_id", "posteam", "fixed_drive", "play_type", "pass_attempt", "rush_attempt",
        "sack", "yards_gained", "qb_kneel", "qb_spike",
    ]
    pbp = raw.select([c for c in needed if c in raw.columns]).filter(pl.col("posteam").is_not_null())
    if "qb_kneel" in pbp.columns:
        pbp = pbp.filter(pl.col("qb_kneel").fill_null(0) == 0)
    if "qb_spike" in pbp.columns:
        pbp = pbp.filter(pl.col("qb_spike").fill_null(0) == 0)

    pass_attempt = pl.col("pass_attempt").fill_null(0).cast(pl.Int16)
    rush_attempt = pl.col("rush_attempt").fill_null(0).cast(pl.Int16)
    sack = pl.col("sack").fill_null(0).cast(pl.Int16)
    offensive_play = ((pass_attempt + rush_attempt + sack) > 0).cast(pl.Int16)
    pbp = pbp.with_columns(
        pass_attempt.alias("pa"),
        rush_attempt.alias("ra"),
        sack.alias("sk"),
        offensive_play.alias("offensive_play"),
        pl.when(offensive_play == 1)
        .then(pl.col("yards_gained").fill_null(0.0))
        .otherwise(0.0)
        .alias("offensive_yards"),
    )

    team_games = (
        pbp.group_by(["game_id", "posteam"])
        .agg(
            pl.col("fixed_drive").drop_nulls().n_unique().alias("drives"),
            pl.sum("offensive_play").alias("plays"),
            pl.sum("pa").alias("pass_attempts"),
            pl.sum("ra").alias("rush_attempts"),
            pl.sum("sk").alias("sacks"),
            pl.sum("offensive_yards").alias("total_yards"),
        )
        .with_columns(
            (pl.col("pass_attempts") + pl.col("sacks")).alias("dropbacks"),
            (pl.col("total_yards") / pl.col("plays").clip(1, None)).alias("yards_per_play"),
        )
    )

    game_shape = (
        team_games.group_by("game_id")
        .agg(
            pl.sum("drives").alias("game_total_drives"),
            pl.max("drives").alias("max_team_drives"),
            pl.min("drives").alias("min_team_drives"),
            pl.sum("plays").alias("game_total_plays"),
            pl.max("plays").alias("max_team_plays"),
            pl.min("plays").alias("min_team_plays"),
        )
        .with_columns(
            (pl.col("max_team_drives") - pl.col("min_team_drives")).alias("drive_imbalance"),
            (pl.col("max_team_plays") - pl.col("min_team_plays")).alias("play_imbalance"),
        )
    )

    x = team_games.get_column("drives").cast(pl.Float64).to_numpy()
    y = team_games.get_column("plays").cast(pl.Float64).to_numpy()
    slope, intercept = np.polyfit(x, y, 1)
    fitted = intercept + slope * x
    residual = y - fitted
    drive_play_corr = float(np.corrcoef(x, y)[0, 1])
    residual_sd = float(np.std(residual, ddof=1))
    residual_p10 = float(np.quantile(residual, 0.10))
    residual_p50 = float(np.quantile(residual, 0.50))
    residual_p90 = float(np.quantile(residual, 0.90))
    conditional_play_model = pl.DataFrame({
        "metric": [
            "intercept", "plays_per_drive_slope", "drive_play_corr",
            "residual_sd", "residual_p10", "residual_p50", "residual_p90",
        ],
        "value": [
            float(intercept), float(slope), drive_play_corr,
            residual_sd, residual_p10, residual_p50, residual_p90,
        ],
    })

    metrics = [
        "drives", "plays", "dropbacks", "pass_attempts", "rush_attempts", "sacks",
        "total_yards", "yards_per_play",
    ]
    league = _league_summary(team_games, metrics)
    game_shape_summary = _league_summary(
        game_shape,
        ["game_total_drives", "drive_imbalance", "game_total_plays", "play_imbalance"],
    )
    team_hist = (
        team_games.group_by("posteam")
        .agg(*[pl.mean(m).alias(f"hist_{m}_mean") for m in metrics])
        .rename({"posteam": "team_id"})
    )

    sim = _read(args.sim_opportunity)
    sim_metric_map = {
        "drives": "drives_mean",
        "plays": "plays_mean",
        "dropbacks": "dropbacks_mean",
        "pass_attempts": "pass_attempts_mean",
        "rush_attempts": "rush_attempts_mean",
        "sacks": "sacks_mean",
        "total_yards": "total_yards_mean",
        "yards_per_play": "yards_per_play_mean",
    }

    league_dict = {row["metric"]: row for row in league.to_dicts()}
    rows = []
    for row in sim.to_dicts():
        out = {"game": row["game"], "team_id": row["team_id"]}
        for metric, sim_col in sim_metric_map.items():
            hist = league_dict[metric]
            value = float(row[sim_col])
            sd = max(float(hist["historical_sd"]), 1e-9)
            out[f"sim_{metric}"] = value
            out[f"league_{metric}_mean"] = float(hist["historical_mean"])
            out[f"league_{metric}_z"] = (value - float(hist["historical_mean"])) / sd
        rows.append(out)
    comparison = pl.DataFrame(rows).join(team_hist, on="team_id", how="left")

    comparison = comparison.with_columns(
        (pl.col("sim_plays") - pl.col("hist_plays_mean")).alias("delta_vs_team_hist_plays"),
        (pl.col("sim_total_yards") - pl.col("hist_total_yards_mean")).alias("delta_vs_team_hist_yards"),
        (pl.col("sim_yards_per_play") - pl.col("hist_yards_per_play_mean")).alias("delta_vs_team_hist_ypp"),
    )

    mean_play_z = float(comparison.get_column("league_plays_z").mean())
    mean_yards_z = float(comparison.get_column("league_total_yards_z").mean())
    mean_ypp_z = float(comparison.get_column("league_yards_per_play_z").mean())
    mean_team_play_delta = float(comparison.get_column("delta_vs_team_hist_plays").mean())
    mean_team_yard_delta = float(comparison.get_column("delta_vs_team_hist_yards").mean())

    sim_p10 = float(sim.get_column("plays_p10").mean())
    sim_p90 = float(sim.get_column("plays_p90").mean())
    historical_play_p10 = float(league_dict["plays"]["historical_p10"])
    historical_play_p90 = float(league_dict["plays"]["historical_p90"])
    play_tail_width_ratio = (sim_p90 - sim_p10) / max(
        historical_play_p90 - historical_play_p10, 1e-9
    )

    if play_tail_width_ratio > 1.50:
        diagnosis = "play_distribution_overdispersed"
    elif mean_play_z > 0.50 and mean_ypp_z > 0.35:
        diagnosis = "both_volume_and_efficiency_high"
    elif mean_play_z > 0.50:
        diagnosis = "play_volume_high"
    elif mean_ypp_z > 0.35:
        diagnosis = "efficiency_high"
    elif mean_yards_z > 0.50:
        diagnosis = "yardage_high_without_single_dominant_driver"
    else:
        diagnosis = "opportunity_supply_within_broad_historical_band"

    args.out.mkdir(parents=True, exist_ok=True)
    team_games.write_csv(args.out / "historical_team_games.csv")
    game_shape.write_csv(args.out / "historical_game_shape.csv")
    league.write_csv(args.out / "historical_league_distribution.csv")
    game_shape_summary.write_csv(args.out / "historical_game_shape_distribution.csv")
    conditional_play_model.write_csv(args.out / "historical_plays_given_drives.csv")
    comparison.write_csv(args.out / "sim_vs_historical.csv")

    game_shape_dict = {row["metric"]: row for row in game_shape_summary.to_dicts()}
    manifest = {
        "artifact": "Monster Opportunity Realism Audit",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "history_season": args.history,
        "historical_team_games": team_games.height,
        "simulated_teams": sim.height,
        "mean_league_play_z": mean_play_z,
        "mean_league_total_yards_z": mean_yards_z,
        "mean_league_ypp_z": mean_ypp_z,
        "mean_delta_vs_same_team_history_plays": mean_team_play_delta,
        "mean_delta_vs_same_team_history_yards": mean_team_yard_delta,
        "mean_sim_play_p10": sim_p10,
        "mean_sim_play_p90": sim_p90,
        "historical_play_p10": historical_play_p10,
        "historical_play_p90": historical_play_p90,
        "play_tail_width_ratio": play_tail_width_ratio,
        "historical_plays_given_drives": {
            "intercept": float(intercept),
            "slope": float(slope),
            "correlation": drive_play_corr,
            "residual_sd": residual_sd,
            "residual_p10": residual_p10,
            "residual_p50": residual_p50,
            "residual_p90": residual_p90,
        },
        "historical_game_play_budget": {
            "mean": float(game_shape_dict["game_total_plays"]["historical_mean"]),
            "sd": float(game_shape_dict["game_total_plays"]["historical_sd"]),
            "p10": float(game_shape_dict["game_total_plays"]["historical_p10"]),
            "p50": float(game_shape_dict["game_total_plays"]["historical_p50"]),
            "p90": float(game_shape_dict["game_total_plays"]["historical_p90"]),
            "mean_team_imbalance": float(game_shape_dict["play_imbalance"]["historical_mean"]),
            "imbalance_p90": float(game_shape_dict["play_imbalance"]["historical_p90"]),
        },
        "diagnosis": diagnosis,
        "market_blind": True,
        "principle": "One game clock governs both possession supply and play supply; diagnose shared budgets before player allocation.",
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(league)
    print(game_shape_summary)
    print(conditional_play_model)
    print(comparison.sort("league_total_yards_z", descending=True))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
