from __future__ import annotations

import argparse
import json
from pathlib import Path

import nflreadpy as nfl

from monster.ingest.nflverse import configure_cache

TACTICAL_TOKENS = (
    "coverage",
    "man_zone",
    "defense",
    "defenders",
    "box",
    "blitz",
    "pressure",
    "rusher",
    "rushers",
    "personnel",
)


def _candidate_columns(columns: list[str]) -> list[str]:
    return sorted(
        column
        for column in columns
        if any(token in column.lower() for token in TACTICAL_TOKENS)
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--season", type=int, default=2025)
    parser.add_argument("--cache-dir", type=Path, default=Path(".cache/monster"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    configure_cache(args.cache_dir)
    pbp = nfl.load_pbp([args.season])
    participation = nfl.load_participation([args.season])

    pbp_candidates = _candidate_columns(pbp.columns)
    participation_candidates = _candidate_columns(participation.columns)

    coverage = {}
    for source_name, frame, candidates in (
        ("pbp", pbp, pbp_candidates),
        ("participation", participation, participation_candidates),
    ):
        coverage[source_name] = {}
        for column in candidates:
            series = frame.get_column(column)
            non_null = series.len() - series.null_count()
            coverage[source_name][column] = {
                "rows": frame.height,
                "non_null": non_null,
                "coverage": non_null / frame.height if frame.height else 0.0,
                "dtype": str(series.dtype),
            }

    report = {
        "artifact": "Monster Defensive Tactical Evidence Field Audit",
        "season": args.season,
        "market_blind": True,
        "behavior_changed_by_audit": False,
        "pbp_candidate_columns": pbp_candidates,
        "participation_candidate_columns": participation_candidates,
        "coverage": coverage,
        "production_authority": 0.0,
        "promotion_rule": "Do not activate defensive tactical intent until a definition-safe source has adequate coverage and an OOS behavior audit.",
        "promotion_status": "SHADOW_SOURCE_DISCOVERY_ONLY",
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "defensive_tactical_field_coverage.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
