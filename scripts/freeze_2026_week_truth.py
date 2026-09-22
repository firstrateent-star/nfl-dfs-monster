from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import nflreadpy as nfl
import polars as pl

from monster.ingest.nflverse import configure_cache


def _week_filter(frame: pl.DataFrame, *, season: int, week: int) -> pl.DataFrame:
    if frame.is_empty():
        return frame
    if "season" in frame.columns:
        frame = frame.filter(
            pl.col("season").cast(pl.Int64, strict=False) == season
        )
    if "week" in frame.columns:
        frame = frame.filter(
            pl.col("week").cast(pl.Int64, strict=False) == week
        )
    elif "nflverse_game_id" in frame.columns:
        frame = frame.filter(
            pl.col("nflverse_game_id")
            .cast(pl.String)
            .str.starts_with(f"{season}_{week:02d}_")
        )
    for column in ("season_type", "game_type"):
        if column in frame.columns:
            frame = frame.filter(
                pl.col(column).cast(pl.String).str.to_uppercase() == "REG"
            )
    return frame


def _write(frame: pl.DataFrame, path: Path) -> None:
    if path.suffix == ".parquet":
        frame.write_parquet(path, compression="zstd")
    else:
        frame.write_csv(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--season", type=int, default=2026)
    parser.add_argument("--week", type=int, required=True)
    parser.add_argument("--cache-dir", type=Path, default=Path(".cache/monster"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    configure_cache(args.cache_dir)
    args.out.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, object] = {
        "artifact": "MONSTER post-simulation football truth freeze",
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "season": args.season,
        "week": args.week,
        "truth_loaded_after_simulation": True,
        "sources": {},
        "principle": (
            "Postgame truth is observational evidence only. It is loaded after the "
            "frozen simulation and cannot alter simulation inputs or behavior."
        ),
    }

    loaders = {
        "player_stats": (
            lambda: nfl.load_player_stats([args.season]),
            "player_stats.csv",
        ),
        "team_stats": (
            lambda: nfl.load_team_stats([args.season]),
            "team_stats.csv",
        ),
        "snap_counts": (
            lambda: nfl.load_snap_counts([args.season]),
            "snap_counts.csv",
        ),
        "schedules": (
            lambda: nfl.load_schedules([args.season]),
            "schedules.csv",
        ),
        "pbp": (
            lambda: nfl.load_pbp([args.season]),
            "pbp.parquet",
        ),
        "ftn_charting": (
            lambda: nfl.load_ftn_charting([args.season]),
            "ftn_charting.parquet",
        ),
        "pfr_pass": (
            lambda: nfl.load_pfr_advstats(
                [args.season], stat_type="pass", summary_level="week"
            ),
            "pfr_pass.csv",
        ),
        "pfr_rush": (
            lambda: nfl.load_pfr_advstats(
                [args.season], stat_type="rush", summary_level="week"
            ),
            "pfr_rush.csv",
        ),
        "pfr_rec": (
            lambda: nfl.load_pfr_advstats(
                [args.season], stat_type="rec", summary_level="week"
            ),
            "pfr_rec.csv",
        ),
        "pfr_def": (
            lambda: nfl.load_pfr_advstats(
                [args.season], stat_type="def", summary_level="week"
            ),
            "pfr_def.csv",
        ),
        "ngs_passing": (
            lambda: nfl.load_nextgen_stats(
                [args.season], stat_type="passing"
            ),
            "ngs_passing.csv",
        ),
        "ngs_receiving": (
            lambda: nfl.load_nextgen_stats(
                [args.season], stat_type="receiving"
            ),
            "ngs_receiving.csv",
        ),
        "ngs_rushing": (
            lambda: nfl.load_nextgen_stats(
                [args.season], stat_type="rushing"
            ),
            "ngs_rushing.csv",
        ),
    }

    for name, (loader, filename) in loaders.items():
        try:
            frame = _week_filter(
                loader(), season=args.season, week=args.week
            )
            _write(frame, args.out / filename)
            manifest["sources"][name] = {
                "available": True,
                "rows": frame.height,
                "columns": frame.columns,
                "file": filename,
            }
        except Exception as exc:
            manifest["sources"][name] = {
                "available": False,
                "error": str(exc),
            }

    participation_note = (
        "nflverse participation data from 2023 onward is not an in-season source; "
        "the audit therefore uses current FTN charting for rush-count/blitz evidence, "
        "PFR advanced passing for pressure/protection outcomes, NGS passing for timing, "
        "and PBP/box-score outcomes for coverage effectiveness."
    )
    manifest["current_season_participation_policy"] = participation_note
    (args.out / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
