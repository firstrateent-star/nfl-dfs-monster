from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import nflreadpy as nfl
import polars as pl

from monster.feature_compile.gadget_rush_incidence_v637 import (
    POSITIONS,
    correlation,
    normalize_weekly_stats,
    summarize_incidence,
    transition_table,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--seasons",
        type=int,
        nargs="+",
        default=[2022, 2023, 2024, 2025],
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("artifacts/v637-gadget-incidence-oos"),
    )
    args = parser.parse_args()

    raw = nfl.load_player_stats(args.seasons)
    stats = normalize_weekly_stats(raw)
    summaries = summarize_incidence(stats)
    transitions, transition_summary = transition_table(
        summaries["player_season"]
    )

    persistence = []
    for position in POSITIONS:
        sub = transitions.filter(pl.col("position") == position)
        persistence.append(
            {
                "position": position,
                "player_season_transitions": sub.height,
                "entry_rate_year_to_year_corr": correlation(
                    sub["prior_entry_rate"].to_numpy(),
                    sub["next_entry_rate"].to_numpy(),
                ),
            }
        )
    persistence_frame = pl.DataFrame(persistence)

    manifest = {
        "artifact": "Monster v6.3.7 Gadget Rush Incidence OOS Audit",
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "seasons": args.seasons,
        "positions": list(POSITIONS),
        "weekly_player_rows": stats.height,
        "market_blind": True,
        "week1_2026_truth_used": False,
        "purpose": (
            "Measure actual WR/TE designed-rush incidence and year-to-year "
            "persistence before granting production authority to gadget-entry "
            "probabilities."
        ),
        "position_summary": summaries["position_summary"].to_dicts(),
        "team_week_summary": summaries["team_week_summary"].to_dicts(),
        "transition_summary": transition_summary.to_dicts(),
        "persistence": persistence,
    }

    args.out.mkdir(parents=True, exist_ok=True)
    stats.filter(pl.col("position").is_in(POSITIONS)).write_csv(
        args.out / "gadget_player_games.csv"
    )
    summaries["position_summary"].write_csv(
        args.out / "position_summary.csv"
    )
    summaries["team_week"].write_csv(
        args.out / "team_week_gadget_rushing.csv"
    )
    summaries["team_week_summary"].write_csv(
        args.out / "team_week_summary.csv"
    )
    summaries["player_season"].write_csv(
        args.out / "player_season_incidence.csv"
    )
    transitions.write_csv(args.out / "player_season_transitions.csv")
    transition_summary.write_csv(args.out / "transition_summary.csv")
    persistence_frame.write_csv(args.out / "persistence.csv")
    (args.out / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
