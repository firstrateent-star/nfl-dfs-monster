from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

import numpy as np
import polars as pl

from monster.feature_compile.league_units import unit_player_from_personnel_row
from monster.feature_compile.reality_inputs import compile_player_reality_inputs
from monster.feature_compile.v13_identity_bridge import compile_v13_player_identity


SKILL_FIELDS = {
    "QB": (
        "madden_throw_accuracy",
        "madden_throw_under_pressure",
        "madden_awareness",
        "madden_throw_power",
        "madden_throw_on_run",
        "madden_play_action",
    ),
    "RB": (
        "madden_ball_carrier_vision",
        "madden_agility",
        "madden_change_of_direction",
        "madden_juke",
        "madden_break_tackle",
        "madden_carrying",
        "madden_speed",
    ),
    "WR": (
        "madden_route_running",
        "madden_release",
        "madden_catching",
        "madden_catch_in_traffic",
        "madden_speed",
        "madden_acceleration",
    ),
    "TE": (
        "madden_route_running",
        "madden_release",
        "madden_catching",
        "madden_catch_in_traffic",
        "madden_speed",
        "madden_acceleration",
    ),
}


def _finite(value: object) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def _mean_present(row: dict[str, object], fields: tuple[str, ...]) -> float | None:
    values = [_finite(row.get(field)) for field in fields]
    present = [value for value in values if value is not None]
    return None if not present else float(np.mean(present))


def _summary(frame: pl.DataFrame, column: str) -> dict[str, float | int | None]:
    if column not in frame.columns:
        return {"n": 0, "p10": None, "p50": None, "p90": None, "range_p90_p10": None}
    values = frame.get_column(column).drop_nulls().cast(pl.Float64).to_numpy()
    if len(values) == 0:
        return {"n": 0, "p10": None, "p50": None, "p90": None, "range_p90_p10": None}
    p10, p50, p90 = np.quantile(values, [0.10, 0.50, 0.90])
    return {
        "n": int(len(values)),
        "p10": float(p10),
        "p50": float(p50),
        "p90": float(p90),
        "range_p90_p10": float(p90 - p10),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--personnel", type=Path, required=True)
    parser.add_argument("--game-date", type=date.fromisoformat, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    personnel = pl.read_parquet(args.personnel)
    reality = compile_player_reality_inputs(personnel, game_date=args.game_date)

    rows: list[dict[str, object]] = []
    for raw in personnel.to_dicts():
        position = str(raw.get("position") or "").upper()
        if position not in SKILL_FIELDS:
            continue
        player_id = str(raw.get("gsis_id") or raw.get("pfr_id") or "")
        if not player_id or player_id not in reality:
            continue
        capability = unit_player_from_personnel_row(raw)
        identity, trace = compile_v13_player_identity(
            player_id=player_id,
            name=str(raw.get("display_name") or player_id),
            position=position,
            usage_weight=1.0,
            inputs=reality[player_id],
            capability_inputs=capability,
        )
        rows.append(
            {
                "player_id": player_id,
                "player": str(raw.get("display_name") or player_id),
                "team_id": str(raw.get("team_id") or ""),
                "position": position,
                "raw_madden_primary_mean": _mean_present(raw, SKILL_FIELDS[position]),
                "madden_speed": _finite(raw.get("madden_speed")),
                "madden_awareness": _finite(raw.get("madden_awareness")),
                "madden_pass_block": _finite(raw.get("madden_pass_block")),
                "madden_run_block": _finite(raw.get("madden_run_block")),
                "madden_pass_rush": _finite(raw.get("madden_pass_rush")),
                "madden_coverage": _finite(raw.get("madden_coverage")),
                "madden_tackle": _finite(raw.get("madden_tackle")),
                "compiled_efficiency": float(identity.efficiency),
                "compiled_explosive": float(identity.explosive),
                "primary_skill": float(trace.primary_skill),
                "mobility": float(trace.mobility),
                "route_separation": float(trace.route_separation),
                "rush_creation": float(trace.rush_creation),
                "ball_security": float(trace.ball_security),
                "evidence_fields": int(trace.evidence_fields),
            }
        )

    frame = pl.DataFrame(rows)
    frame.write_csv(args.out / "v635_skill_identity_dynamic_range.csv")

    by_position: dict[str, object] = {}
    for position in ("QB", "RB", "WR", "TE"):
        sub = frame.filter(pl.col("position") == position)
        by_position[position] = {
            "players": sub.height,
            "raw_primary_madden": _summary(sub, "raw_madden_primary_mean"),
            "compiled_efficiency": _summary(sub, "compiled_efficiency"),
            "compiled_primary_skill": _summary(sub, "primary_skill"),
            "compiled_explosive": _summary(sub, "compiled_explosive"),
            "route_separation": _summary(sub, "route_separation"),
            "rush_creation": _summary(sub, "rush_creation"),
            "mobility": _summary(sub, "mobility"),
        }

    # Unit-level raw Madden ranges show whether the input universe itself carries meaningful
    # differentiation before any compiler or 1v1 transformation can compress it.
    unit_groups = {
        "OL": personnel.filter(
            pl.col("position").cast(pl.Utf8).str.to_uppercase().is_in(
                ["LT", "LG", "C", "RG", "RT", "OL", "OT", "G"]
            )
        ),
        "FRONT": personnel.filter(
            pl.col("position").cast(pl.Utf8).str.to_uppercase().is_in(
                ["DE", "DT", "NT", "DL", "EDGE", "LB", "ILB", "OLB", "MLB"]
            )
        ),
        "COVER": personnel.filter(
            pl.col("position").cast(pl.Utf8).str.to_uppercase().is_in(
                ["CB", "DB", "S", "FS", "SS", "LB", "ILB", "OLB", "MLB"]
            )
        ),
    }
    unit_summary = {
        "OL": {
            "madden_pass_block": _summary(unit_groups["OL"], "madden_pass_block"),
            "madden_run_block": _summary(unit_groups["OL"], "madden_run_block"),
        },
        "FRONT": {
            "madden_pass_rush": _summary(unit_groups["FRONT"], "madden_pass_rush"),
            "madden_tackle": _summary(unit_groups["FRONT"], "madden_tackle"),
        },
        "COVER": {
            "madden_coverage": _summary(unit_groups["COVER"], "madden_coverage"),
            "madden_tackle": _summary(unit_groups["COVER"], "madden_tackle"),
        },
    }

    report = {
        "experiment": "v6.3.5-raw-to-identity-dynamic-range",
        "game_date": args.game_date.isoformat(),
        "skill_players": frame.height,
        "by_position": by_position,
        "unit_raw_ranges": unit_summary,
        "interpretation": (
            "This is an authority-compression audit, not a calibration target. Wide raw Madden "
            "ranges with narrow compiled ranges identify a compiler bottleneck; narrow raw ranges "
            "mean the pregame evidence itself does not distinguish players strongly."
        ),
        "direct_points_used": False,
        "fantasy_outcomes_used": False,
    }
    (args.out / "v635_identity_dynamic_range.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
