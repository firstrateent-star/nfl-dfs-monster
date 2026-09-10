from __future__ import annotations

import argparse
import json
from pathlib import Path

import polars as pl


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument("--treatment", type=Path, required=True)
    parser.add_argument("--treatment-manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    control = pl.read_csv(args.control).select(
        "game", pl.col("away_points_mean").alias("control_away"),
        pl.col("home_points_mean").alias("control_home"),
        pl.col("total_mean").alias("control_total")
    )
    treatment = pl.read_csv(args.treatment).select(
        "game", pl.col("away_points_mean").alias("treatment_away"),
        pl.col("home_points_mean").alias("treatment_home"),
        pl.col("total_mean").alias("treatment_total")
    )
    joined = control.join(treatment, on="game", how="inner").with_columns(
        (pl.col("treatment_away") - pl.col("control_away")).alias("away_delta"),
        (pl.col("treatment_home") - pl.col("control_home")).alias("home_delta"),
        (pl.col("treatment_total") - pl.col("control_total")).alias("total_delta"),
    )
    if joined.height != control.height:
        raise ValueError("availability comparison did not match every Stage 3 control game")

    manifest = json.loads(args.treatment_manifest.read_text())
    report = {
        "artifact": "Monster v1.3 Availability World Paired Comparison",
        "market_blind": bool(manifest["market_blind"]),
        "same_engine_except_availability_semantics": True,
        "games": joined.height,
        "mean_absolute_total_change": float(joined.get_column("total_delta").abs().mean()),
        "mean_total_change": float(joined.get_column("total_delta").mean()),
        "max_absolute_total_change": float(joined.get_column("total_delta").abs().max()),
        "interpretation_rule": "This is a semantic personnel experiment, not a score calibration target. Promote only after role, conservation and player-state audits are healthy.",
        "promotion_status": "SHADOW_AVAILABILITY_EXPERIMENT_ONLY",
    }
    args.out.mkdir(parents=True, exist_ok=True)
    joined.write_csv(args.out / "availability_world_paired_game_comparison.csv")
    (args.out / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
