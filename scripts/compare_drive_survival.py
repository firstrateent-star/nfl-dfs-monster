from __future__ import annotations

import argparse
import json
from pathlib import Path

import polars as pl

REQUIRED = {
    "terminal",
    "scrimmage_plays",
    "first_downs",
    "series_started",
    "series_converted",
    "first_down_snaps",
    "second_down_snaps",
    "third_down_snaps",
    "fourth_down_snaps",
    "third_down_conversions",
    "third_and_long_snaps",
    "third_and_long_conversions",
    "third_down_distance_total",
    "early_down_5plus_gains",
}


def _rate(num: float, den: float) -> float:
    return float(num / den) if den else 0.0


def _summarize(frame: pl.DataFrame, label: str) -> dict[str, float | str]:
    missing = sorted(REQUIRED.difference(frame.columns))
    if missing:
        raise ValueError(f"{label} survival trace missing fields: {missing}")

    sums = frame.select(
        [pl.col(name).sum().alias(name) for name in REQUIRED if name != "terminal"]
    ).to_dicts()[0]
    drives = frame.height
    punts = frame.filter(pl.col("terminal") == "punt").height
    scrimmage_three_and_outs = frame.filter(
        (pl.col("terminal") == "punt")
        & (pl.col("first_downs") == 0)
        & (pl.col("scrimmage_plays") == 3)
    ).height
    third_down_snaps = float(sums["third_down_snaps"] or 0)
    third_long_snaps = float(sums["third_and_long_snaps"] or 0)
    early_down_snaps = float(
        (sums["first_down_snaps"] or 0) + (sums["second_down_snaps"] or 0)
    )

    return {
        "scope": label,
        "drives": float(drives),
        "punt_rate": _rate(punts, drives),
        "scrimmage_three_and_out_rate": _rate(scrimmage_three_and_outs, drives),
        "series_conversion_rate": _rate(
            float(sums["series_converted"] or 0), float(sums["series_started"] or 0)
        ),
        "third_down_conversion_rate": _rate(
            float(sums["third_down_conversions"] or 0), third_down_snaps
        ),
        "third_and_long_share_of_third_downs": _rate(third_long_snaps, third_down_snaps),
        "third_and_long_conversion_rate": _rate(
            float(sums["third_and_long_conversions"] or 0), third_long_snaps
        ),
        "mean_third_down_distance": _rate(
            float(sums["third_down_distance_total"] or 0), third_down_snaps
        ),
        "early_down_5plus_rate": _rate(
            float(sums["early_down_5plus_gains"] or 0), early_down_snaps
        ),
        "third_downs_per_drive": _rate(third_down_snaps, drives),
        "fourth_down_snaps_per_drive": _rate(
            float(sums["fourth_down_snaps"] or 0), drives
        ),
    }


def _delta(sim: float, hist: float) -> dict[str, float | None]:
    return {
        "absolute": sim - hist,
        "relative": None if abs(hist) < 1e-12 else sim / hist - 1.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--simulated", type=Path, required=True)
    parser.add_argument("--historical", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("artifacts/drive-survival-comparison"))
    args = parser.parse_args()

    simulated = pl.read_parquet(args.simulated)
    historical = pl.read_parquet(args.historical)
    sim = _summarize(simulated, "week1_stage3")
    hist = _summarize(historical, "2025_regular_season")

    metrics = [key for key in sim if key not in {"scope", "drives"}]
    rows: list[dict[str, object]] = []
    for metric in metrics:
        sim_value = float(sim[metric])
        hist_value = float(hist[metric])
        delta = _delta(sim_value, hist_value)
        rows.append(
            {
                "metric": metric,
                "simulated": sim_value,
                "historical": hist_value,
                "absolute_delta": delta["absolute"],
                "relative_delta": delta["relative"],
            }
        )

    ranked = sorted(
        rows,
        key=lambda row: (
            abs(float(row["relative_delta"])) if row["relative_delta"] is not None else -1.0
        ),
        reverse=True,
    )
    metric_map = {str(row["metric"]): row for row in rows}

    diagnosis = {
        "center": "localize excess possession death before changing any football mechanism",
        "three_and_out": metric_map["scrimmage_three_and_out_rate"],
        "series_survival": metric_map["series_conversion_rate"],
        "third_down_survival": metric_map["third_down_conversion_rate"],
        "third_and_long_creation": metric_map["third_and_long_share_of_third_downs"],
        "third_and_long_survival": metric_map["third_and_long_conversion_rate"],
        "third_down_distance": metric_map["mean_third_down_distance"],
        "early_down_chunk_creation": metric_map["early_down_5plus_rate"],
        "interpretation_rule": (
            "This ledger is observational. A mismatch identifies a candidate causal organ, not a coefficient target. "
            "Any intervention must be owned by the mechanism that creates the mismatch and rerun paired under the same seed/world universe."
        ),
    }

    args.out.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(rows).write_csv(args.out / "drive_survival_comparison.csv")
    report = {
        "artifact": "Monster v1.3 Drive Survival Ledger",
        "market_blind": True,
        "behavior_changed_by_ledger": False,
        "simulated_drives": int(simulated.height),
        "historical_drives": int(historical.height),
        "definitions": {
            "series_started": "definition-safe scrimmage snap on first down",
            "series_conversion": "definition-safe scrimmage play credited with a new first down",
            "scrimmage_three_and_out": "punt drive with exactly three definition-safe scrimmage plays and no scrimmage first down",
            "third_and_long": "third down with distance >= 7 yards",
            "early_down_5plus": "first/second-down scrimmage gain >= 5 yards",
        },
        "simulated": sim,
        "historical": hist,
        "diagnosis": diagnosis,
        "largest_relative_mismatches": ranked,
        "promotion_status": "SHADOW_DIAGNOSTIC_ONLY",
    }
    (args.out / "drive_survival_ledger.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
