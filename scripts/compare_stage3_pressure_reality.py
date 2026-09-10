from __future__ import annotations

import argparse
import json
from pathlib import Path

import polars as pl


def _rate(num: float, den: float) -> float:
    return float(num / den) if den else 0.0


def _stage3_state(frame: pl.DataFrame, pressured: bool) -> dict[str, float]:
    subset = frame.filter(pl.col("pressured") == pressured)
    totals = subset.select(
        [
            pl.col("intents").sum().alias("intents"),
            pl.col("throws").sum().alias("throws"),
            pl.col("completions").sum().alias("completions"),
            pl.col("interceptions").sum().alias("interceptions"),
            pl.col("sacks").sum().alias("sacks"),
            pl.col("scrambles").sum().alias("scrambles"),
        ]
    ).to_dicts()[0]
    intents = float(totals["intents"] or 0)
    throws = float(totals["throws"] or 0)
    return {
        "dropbacks": intents,
        "throws": throws,
        "sacks": float(totals["sacks"] or 0),
        "scrambles": float(totals["scrambles"] or 0),
        "completions": float(totals["completions"] or 0),
        "interceptions": float(totals["interceptions"] or 0),
        "sack_rate": _rate(float(totals["sacks"] or 0), intents),
        "scramble_rate": _rate(float(totals["scrambles"] or 0), intents),
        "throw_rate": _rate(throws, intents),
        "completion_given_throw": _rate(float(totals["completions"] or 0), throws),
        "interception_given_throw": _rate(float(totals["interceptions"] or 0), throws),
    }


def _comparison(sim: dict[str, float], hist: dict[str, object]) -> dict[str, object]:
    metrics = (
        "sack_rate",
        "scramble_rate",
        "throw_rate",
        "completion_given_throw",
        "interception_given_throw",
    )
    return {
        metric: {
            "simulated": sim[metric],
            "historical": float(hist[metric]),
            "absolute_delta": sim[metric] - float(hist[metric]),
            "relative_delta": (
                None
                if abs(float(hist[metric])) < 1e-12
                else sim[metric] / float(hist[metric]) - 1.0
            ),
        }
        for metric in metrics
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage3", type=Path, required=True)
    parser.add_argument("--historical-manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    stage3 = pl.read_csv(args.stage3)
    historical = json.loads(args.historical_manifest.read_text())
    sim_pressured = _stage3_state(stage3, True)
    sim_clean = _stage3_state(stage3, False)
    hist_pressured = historical["pressured_response"]
    hist_clean = historical["unpressured_response"]

    report = {
        "artifact": "Monster v1.3 Pressure-Conditioned Resolution Comparison",
        "market_blind": True,
        "behavior_changed_by_audit": False,
        "historical_source": historical.get("source"),
        "pressure_coverage": historical.get("pressure_coverage"),
        "simulated": {"pressured": sim_pressured, "not_pressured": sim_clean},
        "historical": {"pressured": hist_pressured, "not_pressured": hist_clean},
        "comparison": {
            "pressured": _comparison(sim_pressured, hist_pressured),
            "not_pressured": _comparison(sim_clean, hist_clean),
        },
        "interpretation_rule": (
            "Tune pressure response only from conditional football outcomes. Do not tune final score, total, or market agreement."
        ),
        "promotion_status": "SHADOW_DIAGNOSTIC_ONLY",
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "stage3_pressure_reality.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
