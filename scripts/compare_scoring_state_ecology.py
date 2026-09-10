from __future__ import annotations

import argparse
import json
from pathlib import Path

import polars as pl


def _rate(num: int, den: int) -> float:
    return float(num / den) if den else 0.0


def _summary(frame: pl.DataFrame, label: str) -> dict[str, object]:
    rows = frame.to_dicts()
    drives = len(rows)
    td = [row for row in rows if str(row["terminal"]) == "touchdown"]
    fg = [row for row in rows if str(row["terminal"]) == "field_goal"]
    rz_enter = [row for row in rows if bool(row["red_zone_entered"])]
    rz_snap = [row for row in rows if bool(row["red_zone_snap_seen"])]
    g2g = [row for row in rows if bool(row["goal_to_go_snap_seen"])]
    long_td = [row for row in td if not bool(row["red_zone_snap_seen"])]

    def terminal_rate(subset: list[dict[str, object]], terminal: str) -> float:
        return _rate(sum(str(row["terminal"]) == terminal for row in subset), len(subset))

    zones = {}
    for name, lo, hi in (
        ("own_1_20", 0.0, 20.0),
        ("own_21_40", 20.0, 40.0),
        ("midfield_41_60", 40.0, 60.0),
        ("opp_41_21", 60.0, 80.0),
        ("red_zone_start", 80.0, 101.0),
    ):
        subset = [row for row in rows if lo <= float(row["start_yardline_100"]) < hi]
        zones[name] = {
            "drives": len(subset),
            "td_rate": terminal_rate(subset, "touchdown"),
            "fg_rate": terminal_rate(subset, "field_goal"),
            "punt_rate": terminal_rate(subset, "punt"),
            "points_per_drive": (
                sum(float(row["points"]) for row in subset) / len(subset) if subset else 0.0
            ),
        }

    return {
        "scope": label,
        "drives": drives,
        "red_zone_reach_rate": _rate(len(rz_enter), drives),
        "red_zone_snap_rate": _rate(len(rz_snap), drives),
        "goal_to_go_snap_rate": _rate(len(g2g), drives),
        "td_per_red_zone_reach": terminal_rate(rz_enter, "touchdown"),
        "td_per_red_zone_snap": terminal_rate(rz_snap, "touchdown"),
        "fg_per_red_zone_snap": terminal_rate(rz_snap, "field_goal"),
        "td_per_goal_to_go_snap": terminal_rate(g2g, "touchdown"),
        "long_td_share": _rate(len(long_td), len(td)),
        "start_field_zones": zones,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--simulated", type=Path, required=True)
    parser.add_argument("--historical", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    simulated = pl.read_parquet(args.simulated)
    historical = pl.read_parquet(args.historical)
    sim = _summary(simulated, "stage3")
    hist = _summary(historical, "2025_regular_season")

    core = (
        "red_zone_reach_rate",
        "red_zone_snap_rate",
        "goal_to_go_snap_rate",
        "td_per_red_zone_reach",
        "td_per_red_zone_snap",
        "fg_per_red_zone_snap",
        "td_per_goal_to_go_snap",
        "long_td_share",
    )
    comparison = {
        metric: {
            "simulated": float(sim[metric]),
            "historical": float(hist[metric]),
            "delta": float(sim[metric]) - float(hist[metric]),
        }
        for metric in core
    }
    zone_comparison = {}
    for zone in sim["start_field_zones"]:
        zone_comparison[zone] = {}
        for metric in ("td_rate", "fg_rate", "punt_rate", "points_per_drive"):
            sv = float(sim["start_field_zones"][zone][metric])
            hv = float(hist["start_field_zones"][zone][metric])
            zone_comparison[zone][metric] = {
                "simulated": sv,
                "historical": hv,
                "delta": sv - hv,
            }

    report = {
        "artifact": "Monster v1.3 Scoring-State Ecology Audit",
        "market_blind": True,
        "behavior_changed_by_audit": False,
        "simulated": sim,
        "historical": hist,
        "comparison": comparison,
        "start_field_zone_comparison": zone_comparison,
        "promotion_status": "SHADOW_DIAGNOSTIC_ONLY",
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "scoring_state_ecology.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
