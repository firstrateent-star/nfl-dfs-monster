from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import polars as pl

from monster.ingest.league import build_league_personnel_snapshot, league_coverage_report
from monster.ingest.nflverse import load_league_personnel_inputs
from monster.teams import NFL_TEAMS


def _write_if_present(frame: pl.DataFrame, path: Path) -> None:
    if frame.height:
        frame.write_parquet(path, compression="zstd")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--season", type=int, default=2026)
    parser.add_argument("--history", type=int, nargs="+", default=[2025])
    parser.add_argument("--recent-games", type=int, default=6)
    parser.add_argument("--cache-dir", type=Path, default=Path(".cache/monster"))
    parser.add_argument("--out", type=Path, default=Path("artifacts/league-baseline"))
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    inputs = load_league_personnel_inputs(args.history, args.season, args.cache_dir)
    snapshot = build_league_personnel_snapshot(
        inputs["current_rosters"],
        inputs["players"],
        inputs["historical_snap_counts"],
        inputs["current_snap_counts"],
        recent_games=args.recent_games,
    )
    coverage = league_coverage_report(snapshot)

    snapshot.write_parquet(args.out / "league_personnel.parquet", compression="zstd")
    snapshot.write_csv(args.out / "league_personnel.csv")
    coverage.write_csv(args.out / "coverage_by_team.csv")

    # Keep changing/complex provider tables raw and auditable until their joins are promoted.
    for key in [
        "injuries",
        "depth_charts",
        "combine",
        "pfr_defense_weekly",
        "current_snap_counts",
    ]:
        _write_if_present(inputs[key], args.out / f"raw_{key}.parquet")

    manifest = {
        "artifact": "Monster 2026 League Personnel Baseline",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "current_season": args.season,
        "history_seasons": args.history,
        "recent_snap_games": args.recent_games,
        "canonical_team_count": len(NFL_TEAMS),
        "snapshot_team_count": snapshot.get_column("team_id").n_unique(),
        "roster_rows": snapshot.height,
        "players_with_snap_prior": int(
            snapshot.select((pl.col("snap_games_observed") > 0).sum()).item()
        ),
        "principle": "League baseline is canonical; weekly DFS slates are downstream filters.",
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
