from __future__ import annotations

import argparse
import json
from pathlib import Path

import polars as pl

from monster.feature_compile.trait_inputs import trait_coverage


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--personnel", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    personnel = pl.read_parquet(args.personnel) if args.personnel.suffix == ".parquet" else pl.read_csv(args.personnel)
    skill = personnel.filter(pl.col("position").cast(pl.Utf8).str.to_uppercase().is_in(["QB", "RB", "WR", "TE"]))
    # Promotion coverage is measured on players who can plausibly enter current football worlds:
    # projected core/rotation players plus top-three depth. Camp/deep-reserve bodies remain reported
    # in all-skill coverage but cannot veto a game model they do not enter.
    relevant = skill.filter(
        pl.col("participation_tier").is_in(["core", "rotation"])
        | (pl.col("depth_rank").fill_null(99) <= 3)
    )
    coverage = trait_coverage(relevant)
    all_skill_coverage = trait_coverage(skill)
    n = max(coverage["skill_players"], 1)
    ratios = {key: value / n for key, value in coverage.items() if key != "skill_players"}
    gates = {
        "height_weight": ratios.get("height", 0) >= 0.95 and ratios.get("weight", 0) >= 0.95,
        "age": ratios.get("birth_date", 0) >= 0.95,
        "athletic_testing_present": coverage.get("forty", 0) > 0,
        "madden_speed": ratios.get("madden_speed", 0) >= 0.70,
        "madden_acceleration": ratios.get("madden_acceleration", 0) >= 0.70,
        "madden_receiving_traits": coverage.get("madden_route_running", 0) > 0 and coverage.get("madden_catching", 0) > 0,
    }
    payload = {
        "status": "PASS" if all(gates.values()) else "BLOCKED",
        "promotion_population": "skill players projected core/rotation OR current depth rank <= 3",
        "coverage": coverage,
        "all_skill_roster_coverage": all_skill_coverage,
        "ratios": ratios,
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
