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
    coverage = trait_coverage(personnel)
    n = max(coverage["skill_players"], 1)
    ratios = {key: value / n for key, value in coverage.items() if key != "skill_players"}
    gates = {
        "height_weight": ratios.get("height", 0) >= 0.90 and ratios.get("weight", 0) >= 0.90,
        "age": ratios.get("birth_date", 0) >= 0.85,
        "athletic_testing_present": coverage.get("forty", 0) > 0,
        "madden_speed": ratios.get("madden_speed", 0) >= 0.70,
        "madden_acceleration": ratios.get("madden_acceleration", 0) >= 0.70,
        "madden_receiving_traits": coverage.get("madden_route_running", 0) > 0 and coverage.get("madden_catching", 0) > 0,
    }
    payload = {
        "status": "PASS" if all(gates.values()) else "BLOCKED",
        "coverage": coverage,
        "ratios": ratios,
        "gates": gates,
        "market_data_used": False,
        "dfs_data_used": False,
        "note": "Coverage gate certifies data presence only; mechanism-direction and ablation evidence are separate promotion requirements.",
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "manifest.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))
    if payload["status"] != "PASS":
        raise SystemExit("Full-Reality feature coverage gate is blocked")


if __name__ == "__main__":
    main()
