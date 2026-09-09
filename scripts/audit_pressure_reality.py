from __future__ import annotations

import argparse
import json
from pathlib import Path

import nflreadpy as nfl
import polars as pl

from monster.ingest.nflverse import configure_cache


def _safe_ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator > 0 else 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--season", type=int, default=2025)
    parser.add_argument("--cache-dir", type=Path, default=Path(".cache/monster"))
    parser.add_argument("--out", type=Path, default=Path("artifacts/pressure-reality"))
    args = parser.parse_args()

    configure_cache(args.cache_dir)
    passing = nfl.load_pfr_advstats([args.season], stat_type="pass", summary_level="season")
    pbp = nfl.load_pbp([args.season])
    if "season_type" in pbp.columns:
        pbp = pbp.filter(pl.col("season_type") == "REG")
    dropbacks = pbp.filter(pl.col("qb_dropback") == 1)

    required = {"times_pressured", "pressure_pct", "season"}
    missing = required.difference(passing.columns)
    if missing:
        raise ValueError(f"PFR passing data missing pressure columns: {sorted(missing)}")

    pressure = passing.filter(
        (pl.col("season") == args.season)
        & pl.col("times_pressured").is_not_null()
        & pl.col("pressure_pct").is_not_null()
        & (pl.col("pressure_pct") > 0)
    ).with_columns(
        (
            pl.col("times_pressured")
            / (pl.col("pressure_pct") / 100.0)
        ).alias("inferred_pressure_opportunities")
    )

    pfr_pressures = float(pressure.get_column("times_pressured").sum())
    pfr_opportunities = float(pressure.get_column("inferred_pressure_opportunities").sum())
    pbp_dropbacks = float(dropbacks.height)
    sacks = float(dropbacks.get_column("sack").fill_null(0).sum())
    scrambles = float(dropbacks.get_column("qb_scramble").fill_null(0).sum())
    qb_hits = float(dropbacks.get_column("qb_hit").fill_null(0).sum())

    result = {
        "season": args.season,
        "source": "nflverse PFR advanced passing + nflverse regular-season PBP",
        "pfr_pressure_count": pfr_pressures,
        "pfr_inferred_pressure_opportunities": pfr_opportunities,
        "pfr_weighted_pressure_rate": _safe_ratio(pfr_pressures, pfr_opportunities),
        "pbp_dropbacks": pbp_dropbacks,
        "pbp_sacks": sacks,
        "pbp_sack_rate": _safe_ratio(sacks, pbp_dropbacks),
        "pbp_scrambles": scrambles,
        "pbp_scramble_rate": _safe_ratio(scrambles, pbp_dropbacks),
        "pbp_qb_hits": qb_hits,
        "pbp_qb_hit_rate": _safe_ratio(qb_hits, pbp_dropbacks),
        "sacks_per_pfr_pressure": _safe_ratio(sacks, pfr_pressures),
        "note": (
            "sacks_per_pfr_pressure is a league-level cross-source calibration ratio, not "
            "a claim that every PBP scramble or QB hit is a PFR pressure event."
        ),
    }

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "pressure_reality.json").write_text(json.dumps(result, indent=2) + "\n")
    pressure.select(
        [column for column in ("player", "team", "pfr_id", "times_pressured", "pressure_pct") if column in pressure.columns]
    ).write_csv(args.out / "pfr_pressure_rows.csv")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
