from __future__ import annotations

import argparse
import json
from pathlib import Path

import polars as pl


COMPARE_METRICS = (
    "touchdown_rate",
    "field_goal_rate",
    "punt_rate",
    "turnover_rate",
    "turnover_on_downs_rate",
    "scoring_drive_rate",
    "red_zone_reach_rate",
    "red_zone_snap_rate",
    "goal_to_go_snap_rate",
    "td_per_red_zone_reach",
    "td_per_red_zone_snap",
    "td_per_goal_to_go_snap",
    "td_without_red_zone_snap_share",
    "explosive_drive_rate",
    "td_given_explosive_drive",
    "td_given_no_explosive_drive",
    "short_field_start_rate",
    "td_given_short_field",
    "td_given_long_field",
    "scrimmage_plays_per_drive",
    "net_yards_per_drive",
    "first_downs_per_drive",
    "explosive_plays_per_drive",
    "sacks_per_drive",
    "start_yardline_100_mean",
)


def _relative_delta(simulated: float, historical: float) -> float | None:
    if abs(historical) < 1e-12:
        return None
    return simulated / historical - 1.0


def _signal(simulated: float, historical: float, threshold: float = 0.08) -> str:
    relative = _relative_delta(simulated, historical)
    if relative is None:
        return "no_historical_denominator"
    if relative >= threshold:
        return "high"
    if relative <= -threshold:
        return "low"
    return "near_historical"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--simulated", type=Path, required=True)
    parser.add_argument("--historical", type=Path, required=True)
    parser.add_argument("--sim-manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("artifacts/drive-reality-comparison"))
    args = parser.parse_args()

    simulated = pl.read_csv(args.simulated).to_dicts()[0]
    historical = pl.read_csv(args.historical).to_dicts()[0]
    manifest = json.loads(args.sim_manifest.read_text())

    sim_drives_per_game = float(simulated["drives"]) / (
        int(manifest["games"]) * int(manifest["worlds_per_game"])
    )
    hist_drives_per_game = float(historical.get("drives_per_game", 0.0))

    rows = []
    for metric in COMPARE_METRICS:
        if metric not in simulated or metric not in historical:
            continue
        sim_value = float(simulated[metric])
        hist_value = float(historical[metric])
        rows.append(
            {
                "metric": metric,
                "simulated": sim_value,
                "historical": hist_value,
                "absolute_delta": sim_value - hist_value,
                "relative_delta": _relative_delta(sim_value, hist_value),
                "signal": _signal(sim_value, hist_value),
            }
        )

    metric_map = {row["metric"]: row for row in rows}

    def value(metric: str, side: str) -> float:
        return float(metric_map[metric][side])

    td_sim = value("touchdown_rate", "simulated")
    td_hist = value("touchdown_rate", "historical")
    rz_entry_sim = value("red_zone_snap_rate", "simulated")
    rz_entry_hist = value("red_zone_snap_rate", "historical")
    rz_conv_sim = value("td_per_red_zone_snap", "simulated")
    rz_conv_hist = value("td_per_red_zone_snap", "historical")
    long_td_sim = value("td_without_red_zone_snap_share", "simulated")
    long_td_hist = value("td_without_red_zone_snap_share", "historical")
    explosive_sim = value("explosive_drive_rate", "simulated")
    explosive_hist = value("explosive_drive_rate", "historical")
    explosive_td_sim = value("td_given_explosive_drive", "simulated")
    explosive_td_hist = value("td_given_explosive_drive", "historical")

    diagnosis = {
        "touchdown_rate_excess": td_sim - td_hist,
        "touchdown_rate_relative_excess": _relative_delta(td_sim, td_hist),
        "red_zone_snap_frequency_signal": _signal(rz_entry_sim, rz_entry_hist),
        "red_zone_td_conversion_signal": _signal(rz_conv_sim, rz_conv_hist),
        "td_without_red_zone_snap_signal": _signal(long_td_sim, long_td_hist),
        "explosive_drive_frequency_signal": _signal(explosive_sim, explosive_hist),
        "explosive_drive_td_conversion_signal": _signal(explosive_td_sim, explosive_td_hist),
        "drive_survival_first_down_signal": _signal(
            value("first_downs_per_drive", "simulated"),
            value("first_downs_per_drive", "historical"),
        ),
        "drive_length_signal": _signal(
            value("scrimmage_plays_per_drive", "simulated"),
            value("scrimmage_plays_per_drive", "historical"),
        ),
        "drive_yardage_signal": _signal(
            value("net_yards_per_drive", "simulated"),
            value("net_yards_per_drive", "historical"),
        ),
        "drive_frequency": {
            "simulated_drives_per_game": sim_drives_per_game,
            "historical_drives_per_game": hist_drives_per_game,
            "relative_delta": _relative_delta(sim_drives_per_game, hist_drives_per_game),
        },
        "interpretation_rule": "Signals localize candidate causal organs only. They do not authorize coefficient changes. Inspect raw/by-game traces and mechanism jurisdiction before intervention.",
    }

    ranked = sorted(
        (
            row
            for row in rows
            if row["relative_delta"] is not None
            and row["metric"]
            not in {"td_per_goal_to_go_snap", "td_given_short_field", "td_given_long_field"}
        ),
        key=lambda row: abs(float(row["relative_delta"])),
        reverse=True,
    )

    args.out.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(rows).write_csv(args.out / "drive_reality_comparison.csv")
    report = {
        "artifact": "Monster Drive Reality Causal Comparison",
        "market_blind": True,
        "behavior_changed_by_comparison": False,
        "simulated_source": str(args.simulated),
        "historical_source": str(args.historical),
        "diagnosis": diagnosis,
        "largest_relative_mismatches": ranked[:10],
    }
    (args.out / "drive_reality_comparison.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
