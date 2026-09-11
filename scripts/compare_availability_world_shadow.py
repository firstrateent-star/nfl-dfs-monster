from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import polars as pl


def _dispersion(frame: pl.DataFrame, away: str, home: str, total: str) -> dict[str, float]:
    team_scores = np.concatenate(
        [frame.get_column(away).to_numpy(), frame.get_column(home).to_numpy()]
    ).astype(float)
    margins = (
        frame.get_column(away).to_numpy().astype(float)
        - frame.get_column(home).to_numpy().astype(float)
    )
    totals = frame.get_column(total).to_numpy().astype(float)
    return {
        "team_mean_score_sd": float(np.std(team_scores, ddof=1)),
        "team_mean_score_range": float(np.max(team_scores) - np.min(team_scores)),
        "game_total_mean": float(np.mean(totals)),
        "game_total_sd_across_games": float(np.std(totals, ddof=1)),
        "game_margin_sd_across_games": float(np.std(margins, ddof=1)),
        "mean_absolute_game_margin": float(np.mean(np.abs(margins))),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument("--treatment", type=Path, required=True)
    parser.add_argument("--treatment-manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    control = pl.read_csv(args.control).select(
        "game",
        pl.col("away_points_mean").alias("control_away"),
        pl.col("home_points_mean").alias("control_home"),
        pl.col("total_mean").alias("control_total"),
    )
    treatment = pl.read_csv(args.treatment).select(
        "game",
        pl.col("away_points_mean").alias("treatment_away"),
        pl.col("home_points_mean").alias("treatment_home"),
        pl.col("total_mean").alias("treatment_total"),
    )
    joined = control.join(treatment, on="game", how="inner").with_columns(
        (pl.col("treatment_away") - pl.col("control_away")).alias("away_delta"),
        (pl.col("treatment_home") - pl.col("control_home")).alias("home_delta"),
        (pl.col("treatment_total") - pl.col("control_total")).alias("total_delta"),
    )
    if joined.height != control.height:
        raise ValueError("availability comparison did not match every Stage 3 control game")

    manifest = json.loads(args.treatment_manifest.read_text())
    control_dispersion = _dispersion(joined, "control_away", "control_home", "control_total")
    treatment_dispersion = _dispersion(
        joined, "treatment_away", "treatment_home", "treatment_total"
    )
    report = {
        "artifact": "Monster v1.3 Full-Roster Availability + Environment Paired Comparison",
        "market_blind": bool(manifest["market_blind"]),
        "paired_common_random_numbers": True,
        "games": joined.height,
        "mean_absolute_total_change": float(joined.get_column("total_delta").abs().mean()),
        "mean_total_change": float(joined.get_column("total_delta").mean()),
        "max_absolute_total_change": float(joined.get_column("total_delta").abs().max()),
        "control_dispersion": control_dispersion,
        "treatment_dispersion": treatment_dispersion,
        "team_score_sd_ratio": (
            treatment_dispersion["team_mean_score_sd"]
            / max(control_dispersion["team_mean_score_sd"], 1e-9)
        ),
        "margin_sd_ratio": (
            treatment_dispersion["game_margin_sd_across_games"]
            / max(control_dispersion["game_margin_sd_across_games"], 1e-9)
        ),
        "single_full_roster_world": bool(
            manifest["single_full_roster_availability_draw_per_team_world"]
        ),
        "ol_world_substitution": bool(manifest["ol_world_substitution"]),
        "defensive_world_substitution": bool(manifest["defensive_world_substitution"]),
        "weather_active": bool(manifest["shared_game_weather_active"]),
        "interpretation_rule": "This is a causal personnel/environment experiment, not a score target. Retain only if world coherence and football anatomy remain healthy.",
        "promotion_status": "SHADOW_FULL_ROSTER_ENVIRONMENT_NOT_PROMOTED",
    }
    args.out.mkdir(parents=True, exist_ok=True)
    joined.write_csv(args.out / "availability_world_paired_game_comparison.csv")
    (args.out / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
