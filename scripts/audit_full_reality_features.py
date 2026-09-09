from __future__ import annotations

import argparse
import json
from pathlib import Path

import polars as pl

from monster.feature_compile.trait_inputs import trait_coverage


def _count(frame: pl.DataFrame, column: str) -> int:
    return int(frame.select(pl.col(column).is_not_null().sum()).item()) if column in frame.columns else 0


def _relevant(frame: pl.DataFrame) -> pl.DataFrame:
    return frame.filter(
        pl.col("participation_tier").is_in(["core", "rotation"])
        | (pl.col("depth_rank").fill_null(99) <= 3)
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--personnel", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    personnel = pl.read_parquet(args.personnel) if args.personnel.suffix == ".parquet" else pl.read_csv(args.personnel)
    pos = pl.col("position").cast(pl.Utf8).str.to_uppercase()
    skill = personnel.filter(pos.is_in(["QB", "RB", "WR", "TE"]))
    defense = personnel.filter(pos.is_in(["DL", "DE", "DT", "NT", "LB", "ILB", "OLB", "DB", "CB", "S", "FS", "SS"]))
    front = defense.filter(pos.is_in(["DL", "DE", "DT", "NT", "LB", "ILB", "OLB"]))
    coverage_unit = defense.filter(pos.is_in(["DB", "CB", "S", "FS", "SS", "LB", "ILB", "OLB"]))
    ol = personnel.filter(pos.is_in(["OL", "OT", "T", "G", "OG", "C"]))
    specialists = personnel.filter(pos.is_in(["K", "P", "KR", "PR"]))

    relevant_skill = _relevant(skill)
    relevant_defense = _relevant(defense)
    relevant_front = _relevant(front)
    relevant_coverage = _relevant(coverage_unit)
    relevant_ol = _relevant(ol)
    relevant_specialists = _relevant(specialists)

    coverage = trait_coverage(relevant_skill)
    n = max(coverage["skill_players"], 1)
    ratios = {key: value / n for key, value in coverage.items() if key != "skill_players"}

    def ratio(frame: pl.DataFrame, column: str) -> float:
        return _count(frame, column) / max(frame.height, 1)

    unit_coverage = {
        "relevant_defenders": relevant_defense.height,
        "defenders_with_madden_tackle": _count(relevant_defense, "madden_tackle"),
        "front_with_madden_pass_rush": _count(relevant_front, "madden_pass_rush"),
        "coverage_players_with_madden_coverage": _count(relevant_coverage, "madden_coverage"),
        "relevant_ol": relevant_ol.height,
        "ol_with_madden_pass_block": _count(relevant_ol, "madden_pass_block"),
        "ol_with_madden_run_block": _count(relevant_ol, "madden_run_block"),
        "relevant_specialists": relevant_specialists.height,
        "specialists_with_kick_power": _count(relevant_specialists, "madden_kick_power"),
    }
    gates = {
        "height_weight": ratios.get("height", 0) >= 0.95 and ratios.get("weight", 0) >= 0.95,
        "age": ratios.get("birth_date", 0) >= 0.95,
        "athletic_testing_present": coverage.get("forty", 0) > 0,
        "madden_speed": ratios.get("madden_speed", 0) >= 0.70,
        "madden_acceleration": ratios.get("madden_acceleration", 0) >= 0.70,
        "madden_receiving_traits": coverage.get("madden_route_running", 0) > 0 and coverage.get("madden_catching", 0) > 0,
        "madden_defensive_front": ratio(relevant_front, "madden_pass_rush") >= 0.60,
        "madden_coverage_unit": ratio(relevant_coverage, "madden_coverage") >= 0.60,
        "madden_tackling": ratio(relevant_defense, "madden_tackle") >= 0.60,
        "madden_offensive_line": ratio(relevant_ol, "madden_pass_block") >= 0.60 and ratio(relevant_ol, "madden_run_block") >= 0.60,
        "specialist_evidence_present": relevant_specialists.height == 0 or _count(relevant_specialists, "madden_kick_power") > 0,
    }
    payload = {
        "status": "PASS" if all(gates.values()) else "BLOCKED",
        "promotion_population": "current core/rotation OR depth rank <= 3",
        "skill_coverage": coverage,
        "all_skill_roster_coverage": trait_coverage(skill),
        "skill_ratios": ratios,
        "unit_coverage": unit_coverage,
        "gates": gates,
        "market_data_used": False,
        "dfs_data_used": False,
        "note": "Coverage certifies source presence only; causal mechanism and ablation gates remain mandatory.",
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "manifest.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))
    if payload["status"] != "PASS":
        raise SystemExit("Full-Reality feature coverage gate is blocked")


if __name__ == "__main__":
    main()
