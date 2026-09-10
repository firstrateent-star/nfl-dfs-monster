from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import polars as pl

REQUIRED = {
    "terminal",
    "start_seconds_remaining",
    "end_seconds_remaining",
    "scrimmage_plays",
    "first_downs",
    "points",
    "overtime",
}


def _summary(values: list[float]) -> dict[str, float]:
    if not values:
        return {
            "mean": 0.0,
            "p10": 0.0,
            "p50": 0.0,
            "p90": 0.0,
            "short_45s_rate": 0.0,
            "long_300s_rate": 0.0,
        }
    arr = np.asarray(values, dtype=float)
    return {
        "mean": float(arr.mean()),
        "p10": float(np.quantile(arr, 0.10)),
        "p50": float(np.quantile(arr, 0.50)),
        "p90": float(np.quantile(arr, 0.90)),
        "short_45s_rate": float(np.mean(arr <= 45.0)),
        "long_300s_rate": float(np.mean(arr >= 300.0)),
    }


def _condition(frame: pl.DataFrame, label: str) -> tuple[dict[str, object], list[dict[str, object]]]:
    missing = sorted(REQUIRED.difference(frame.columns))
    if missing:
        raise ValueError(f"{label} terminal clock trace missing fields: {missing}")

    regulation = frame.filter(~pl.col("overtime").cast(pl.Boolean))
    rows: list[dict[str, object]] = []
    for terminal in sorted(str(value) for value in regulation.get_column("terminal").drop_nulls().unique()):
        subset = regulation.filter(pl.col("terminal").cast(pl.Utf8) == terminal)
        durations = []
        for row in subset.select(
            ["start_seconds_remaining", "end_seconds_remaining"]
        ).to_dicts():
            duration = max(
                float(row["start_seconds_remaining"]) - float(row["end_seconds_remaining"]),
                0.0,
            )
            if duration <= 900.0:
                durations.append(duration)
        duration_summary = _summary(durations)
        rows.append(
            {
                "scope": label,
                "terminal": terminal,
                "drives": subset.height,
                "drive_share": subset.height / regulation.height if regulation.height else 0.0,
                "mean_duration_seconds": duration_summary["mean"],
                "p10_duration_seconds": duration_summary["p10"],
                "p50_duration_seconds": duration_summary["p50"],
                "p90_duration_seconds": duration_summary["p90"],
                "short_45s_rate": duration_summary["short_45s_rate"],
                "long_300s_rate": duration_summary["long_300s_rate"],
                "mean_scrimmage_plays": float(subset.get_column("scrimmage_plays").mean() or 0.0),
                "mean_first_downs": float(subset.get_column("first_downs").mean() or 0.0),
                "mean_points": float(subset.get_column("points").mean() or 0.0),
            }
        )

    all_durations = []
    for row in regulation.select(
        ["start_seconds_remaining", "end_seconds_remaining"]
    ).to_dicts():
        duration = max(
            float(row["start_seconds_remaining"]) - float(row["end_seconds_remaining"]),
            0.0,
        )
        if duration <= 900.0:
            all_durations.append(duration)
    overall = {
        "scope": label,
        "regulation_drives": regulation.height,
        "duration": _summary(all_durations),
    }
    return overall, rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--simulated", type=Path, required=True)
    parser.add_argument("--historical", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    simulated = pl.read_parquet(args.simulated)
    historical = pl.read_parquet(args.historical)
    sim_overall, sim_rows = _condition(simulated, "week1_stage3")
    hist_overall, hist_rows = _condition(historical, "2025_regular_season")

    hist_by_terminal = {str(row["terminal"]): row for row in hist_rows}
    comparisons: list[dict[str, object]] = []
    for row in sim_rows:
        terminal = str(row["terminal"])
        hist = hist_by_terminal.get(terminal)
        if hist is None:
            continue
        comparisons.append(
            {
                "terminal": terminal,
                "drive_share_delta": float(row["drive_share"]) - float(hist["drive_share"]),
                "mean_duration_seconds_delta": float(row["mean_duration_seconds"])
                - float(hist["mean_duration_seconds"]),
                "p10_duration_seconds_delta": float(row["p10_duration_seconds"])
                - float(hist["p10_duration_seconds"]),
                "p90_duration_seconds_delta": float(row["p90_duration_seconds"])
                - float(hist["p90_duration_seconds"]),
                "short_45s_rate_delta": float(row["short_45s_rate"])
                - float(hist["short_45s_rate"]),
                "long_300s_rate_delta": float(row["long_300s_rate"])
                - float(hist["long_300s_rate"]),
                "mean_scrimmage_plays_delta": float(row["mean_scrimmage_plays"])
                - float(hist["mean_scrimmage_plays"]),
            }
        )

    report = {
        "artifact": "Monster v1.3 Terminal-Conditioned Drive Clock Audit",
        "market_blind": True,
        "behavior_changed_by_audit": False,
        "simulated": sim_overall,
        "historical": hist_overall,
        "terminal_comparison": comparisons,
        "interpretation_rule": (
            "First attribute clock-tail mismatch to drive-terminal mix versus duration conditional on terminal. "
            "Do not alter per-play clock merely to force aggregate drive duration."
        ),
        "promotion_status": "SHADOW_DIAGNOSTIC_ONLY",
    }
    args.out.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(sim_rows + hist_rows).write_csv(args.out / "terminal_conditioned_drive_clock.csv")
    (args.out / "terminal_conditioned_drive_clock.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
