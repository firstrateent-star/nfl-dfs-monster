from __future__ import annotations

import argparse
import json
from pathlib import Path

import polars as pl


def _read(path: Path) -> pl.DataFrame:
    return pl.read_parquet(path) if path.suffix == ".parquet" else pl.read_csv(path)


def _mean_abs(frame: pl.DataFrame, column: str) -> float:
    return float(frame.get_column(column).abs().mean())


def _weighted_abs(frame: pl.DataFrame, delta: str, weight: str) -> float:
    total = float(frame.get_column(weight).sum())
    if total <= 0.0:
        return 0.0
    return float((frame.get_column(delta).abs() * frame.get_column(weight)).sum() / total)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--stage3", type=Path, required=True)
    parser.add_argument("--pass-historical", type=Path, required=True)
    parser.add_argument("--run-historical", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    baseline = _read(args.baseline / "football_anatomy.csv")
    stage3 = _read(args.stage3 / "football_anatomy.csv")
    stage3_manifest = json.loads((args.stage3 / "manifest.json").read_text())
    required_flags = (
        "hierarchical_game_flow_active",
        "intent_ecology_active",
        "stage3_resolution_ecology_active",
        "signed_pass_geometry_active",
    )
    missing_flags = [name for name in required_flags if stage3_manifest.get(name) is not True]
    if missing_flags:
        raise RuntimeError(f"Stage 3 run missing active flags: {missing_flags}")

    b = baseline.rename({c: f"{c}_legacy" for c in baseline.columns if c != "game"})
    s = stage3.rename({c: f"{c}_stage3" for c in stage3.columns if c != "game"})
    whole = b.join(s, on="game")
    whole_metrics = (
        "scrimmage_plays_mean",
        "dropbacks_mean",
        "run_plays_mean",
        "completion_percentage",
        "sack_rate",
        "scramble_rate",
        "interception_rate",
        "punts_mean",
        "field_goal_attempts_mean",
        "touchdowns_mean",
        "drives_mean",
        "total_mean",
    )
    for metric in whole_metrics:
        whole = whole.with_columns(
            (pl.col(f"{metric}_stage3") - pl.col(f"{metric}_legacy")).alias(f"{metric}_delta")
        )
    whole.write_csv(args.out / "whole_game_paired_anatomy.csv")

    pass_sim = _read(args.stage3 / "stage3_pass_depth_anatomy.csv")
    pass_hist = _read(args.pass_historical)
    total_throws = float(pass_sim.get_column("throws").sum())
    total_hist_pass = float(pass_hist.get_column("attempts").sum())
    pass_sim = pass_sim.with_columns((pl.col("throws") / total_throws).alias("simulated_share"))
    pass_hist = pass_hist.with_columns(
        (pl.col("attempts") / total_hist_pass).alias("historical_share")
    )
    pass_compare = pass_sim.join(pass_hist, on="category", suffix="_historical")
    pass_pairs = {
        "share": ("simulated_share", "historical_share"),
        "completion_rate": ("completion_rate_on_throws", "completion_rate"),
        "interception_rate": ("interception_rate_on_throws", "interception_rate"),
        "air_yards_mean": ("air_yards_mean_on_throws", "air_yards_mean"),
        "negative_completion_rate": ("negative_completion_rate", "negative_completion_rate_historical"),
    }
    for label, (sim_col, hist_col) in pass_pairs.items():
        pass_compare = pass_compare.with_columns(
            (pl.col(sim_col) - pl.col(hist_col)).alias(f"{label}_delta")
        )
    pass_compare.write_csv(args.out / "pass_depth_historical_comparison.csv")

    run_sim = _read(args.stage3 / "stage3_run_geometry_anatomy.csv")
    run_hist = _read(args.run_historical)
    total_runs = float(run_sim.get_column("attempts").sum())
    total_hist_runs = float(run_hist.get_column("attempts").sum())
    run_sim = run_sim.with_columns((pl.col("attempts") / total_runs).alias("simulated_share"))
    run_hist = run_hist.with_columns(
        (pl.col("attempts") / total_hist_runs).alias("historical_share")
    )
    run_compare = run_sim.join(run_hist, on="category", suffix="_historical")
    run_metrics = (
        "yards_mean",
        "yards_p10",
        "yards_p50",
        "yards_p90",
        "yards_p99",
        "negative_rate",
        "zero_rate",
        "loss_2_plus_rate",
        "loss_5_plus_rate",
        "explosive_10_rate",
        "explosive_15_rate",
        "explosive_20_rate",
    )
    run_compare = run_compare.with_columns(
        (pl.col("simulated_share") - pl.col("historical_share")).alias("share_delta")
    )
    for metric in run_metrics:
        run_compare = run_compare.with_columns(
            (pl.col(metric) - pl.col(f"{metric}_historical")).alias(f"{metric}_delta")
        )
    run_compare.write_csv(args.out / "run_geometry_historical_comparison.csv")

    summary = {
        "artifact": "Monster v1.3 Stage 3 Paired Anatomy Experiment",
        "market_blind": True,
        "control": "legacy Week 1 kernel",
        "treatment": "certified Game Flow + certified pass/run intent + Stage 3 shadow resolution",
        "same_seed_and_world_count": True,
        "stage3_attached_team_count": stage3_manifest["stage3_attached_team_count"],
        "whole_game_mean_abs_changes": {
            metric: _mean_abs(whole, f"{metric}_delta") for metric in whole_metrics
        },
        "pass_depth_weighted_absolute_error": {
            label: _weighted_abs(pass_compare, f"{label}_delta", "attempts")
            for label in pass_pairs
        },
        "run_geometry_weighted_absolute_error": {
            "share": _weighted_abs(run_compare, "share_delta", "attempts_historical"),
            **{
                metric: _weighted_abs(
                    run_compare,
                    f"{metric}_delta",
                    "attempts_historical",
                )
                for metric in run_metrics
            },
        },
        "interpretation_rule": (
            "Use branch anatomy against historical football as the tuning target. "
            "Do not tune final score or market agreement directly."
        ),
        "promotion_status": "SHADOW_STAGE3_EVIDENCE_ONLY",
    }
    (args.out / "manifest.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(pass_compare)
    print(run_compare)
    print(whole.select(["game", *[f"{metric}_delta" for metric in whole_metrics]]))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
