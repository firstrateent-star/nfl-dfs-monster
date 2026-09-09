from __future__ import annotations

import argparse
import json
from pathlib import Path

import nflreadpy as nfl
import polars as pl

from monster.ingest.nflverse import configure_cache
from monster.teams import TEAM_ALIASES


def _ratio(numerator: int | float, denominator: int | float) -> float:
    return float(numerator) / float(denominator) if denominator else 0.0


def _response_summary(frame: pl.DataFrame, label: str) -> dict[str, float | int | str]:
    n = frame.height
    sacks = int(frame.get_column("sack").fill_null(0).sum())
    scrambles = int(frame.get_column("qb_scramble").fill_null(0).sum())
    throws = n - sacks - scrambles
    return {
        "state": label,
        "dropbacks": n,
        "sacks": sacks,
        "scrambles": scrambles,
        "throws": throws,
        "sack_rate": _ratio(sacks, n),
        "scramble_rate": _ratio(scrambles, n),
        "throw_rate": _ratio(throws, n),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--season", type=int, default=2025)
    parser.add_argument("--cache-dir", type=Path, default=Path(".cache/monster"))
    parser.add_argument("--out", type=Path, default=Path("artifacts/pressure-participation"))
    args = parser.parse_args()

    configure_cache(args.cache_dir)
    participation = nfl.load_participation([args.season])
    pbp = nfl.load_pbp([args.season])
    if "season_type" in pbp.columns:
        pbp = pbp.filter(pl.col("season_type") == "REG")

    participation_required = {"nflverse_game_id", "play_id", "was_pressure"}
    missing_participation = participation_required.difference(participation.columns)
    if missing_participation:
        raise ValueError(
            f"participation missing pressure columns: {sorted(missing_participation)}; "
            f"available={sorted(participation.columns)}"
        )
    pbp_required = {
        "game_id",
        "play_id",
        "qb_dropback",
        "qb_scramble",
        "sack",
        "qb_hit",
        "passer_player_id",
        "posteam",
        "defteam",
    }
    missing_pbp = pbp_required.difference(pbp.columns)
    if missing_pbp:
        raise ValueError(f"PBP missing pressure-audit columns: {sorted(missing_pbp)}")

    pressure = (
        participation.select(["nflverse_game_id", "play_id", "was_pressure"])
        .rename({"nflverse_game_id": "game_id"})
        .unique(subset=["game_id", "play_id"], keep="last")
    )
    dropbacks = (
        pbp.filter(pl.col("qb_dropback") == 1)
        .select(list(pbp_required))
        .with_columns(
            pl.col("posteam").replace(TEAM_ALIASES),
            pl.col("defteam").replace(TEAM_ALIASES),
        )
    )
    joined = dropbacks.join(pressure, on=["game_id", "play_id"], how="inner")
    observed = joined.filter(pl.col("was_pressure").is_not_null())
    pressured = observed.filter(pl.col("was_pressure") == True)  # noqa: E712
    clean = observed.filter(pl.col("was_pressure") == False)  # noqa: E712

    overall = _response_summary(observed, "all_observed_dropbacks")
    pressured_summary = _response_summary(pressured, "pressured")
    clean_summary = _response_summary(clean, "not_pressured")

    defense = (
        observed.group_by("defteam")
        .agg(
            pl.len().alias("dropbacks"),
            pl.col("was_pressure").cast(pl.Float64).mean().alias("pressure_rate"),
            pl.col("sack").fill_null(0).mean().alias("sack_rate"),
            pl.col("qb_scramble").fill_null(0).mean().alias("scramble_rate"),
        )
        .rename({"defteam": "team_id"})
        .sort("team_id")
    )
    offense = (
        observed.group_by("posteam")
        .agg(
            pl.len().alias("dropbacks"),
            pl.col("was_pressure").cast(pl.Float64).mean().alias("pressure_rate_allowed"),
            pl.col("sack").fill_null(0).mean().alias("sack_rate_allowed"),
            pl.col("qb_scramble").fill_null(0).mean().alias("scramble_rate"),
        )
        .rename({"posteam": "team_id"})
        .sort("team_id")
    )
    quarterbacks = (
        observed.filter(pl.col("passer_player_id").is_not_null())
        .group_by("passer_player_id")
        .agg(
            pl.len().alias("dropbacks"),
            pl.col("was_pressure").cast(pl.Float64).mean().alias("pressure_rate"),
            pl.col("sack").fill_null(0).mean().alias("sack_rate"),
            pl.col("qb_scramble").fill_null(0).mean().alias("scramble_rate"),
            pl.col("sack")
            .fill_null(0)
            .filter(pl.col("was_pressure") == True)  # noqa: E712
            .mean()
            .alias("sack_given_pressure"),
            pl.col("qb_scramble")
            .fill_null(0)
            .filter(pl.col("was_pressure") == True)  # noqa: E712
            .mean()
            .alias("scramble_given_pressure"),
            pl.col("qb_scramble")
            .fill_null(0)
            .filter(pl.col("was_pressure") == False)  # noqa: E712
            .mean()
            .alias("scramble_without_pressure"),
        )
        .filter(pl.col("dropbacks") >= 50)
        .sort("dropbacks", descending=True)
    )

    manifest = {
        "artifact": "Monster Play-Level Pressure Reality Audit",
        "season": args.season,
        "source": "FTN Data via nflverse participation + nflverse regular-season PBP",
        "license_note": "Participation data from 2023 onward is FTN Data via nflverse (CC-BY-SA 4.0).",
        "pbp_dropbacks": dropbacks.height,
        "joined_dropbacks": joined.height,
        "observed_pressure_dropbacks": observed.height,
        "pressure_coverage": _ratio(observed.height, dropbacks.height),
        "pressure_rate": _ratio(pressured.height, observed.height),
        "overall_response": overall,
        "pressured_response": pressured_summary,
        "unpressured_response": clean_summary,
        "principle": (
            "Pressure frequency and QB response are measured as separate causal stages; "
            "Monster should not tune sacks by pretending sack rate is pressure rate."
        ),
    }

    args.out.mkdir(parents=True, exist_ok=True)
    defense.write_csv(args.out / "defense_pressure.csv")
    offense.write_csv(args.out / "offense_pressure_allowed.csv")
    quarterbacks.write_csv(args.out / "quarterback_pressure_response.csv")
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
