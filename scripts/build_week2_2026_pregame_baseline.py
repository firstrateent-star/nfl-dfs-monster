from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

import polars as pl

BASELINE_ROOT = Path.cwd()
sys.path.insert(0, str(BASELINE_ROOT / "scripts"))

import build_league_baseline as baseline  # noqa: E402

from monster.ingest.madden_canonical import canonicalize_madden_attribute_mirror  # noqa: E402
from monster.ingest.madden_players import load_madden27_player_ratings  # noqa: E402

CUTOFF_DATE = date(2026, 9, 16)
ROSTER_CUTOFF_WEEK = 2
CURRENT_SNAP_CUTOFF_WEEK = 1
MADDEN_MIRROR_COMMIT = "ad0350f3b47b5559f2b2580dab9d9403acd4f4d0"
MADDEN_FROZEN_URL = (
    "https://raw.githubusercontent.com/zachxwalton/madden-ratings-breakdown/"
    f"{MADDEN_MIRROR_COMMIT}/scraper/output/madden27_ratings.csv"
)


def _week_filter(frame: pl.DataFrame, cutoff_week: int) -> pl.DataFrame:
    if frame.height and "week" in frame.columns:
        return frame.filter(
            pl.col("week").cast(pl.Int64, strict=False) <= cutoff_week
        )
    return frame


def _date_filter(frame: pl.DataFrame, column: str) -> pl.DataFrame:
    if not frame.height or column not in frame.columns:
        return frame
    parsed = pl.col(column).cast(pl.Datetime, strict=False).dt.date()
    return frame.filter(parsed.is_null() | (parsed <= pl.lit(CUTOFF_DATE)))


def freeze_inputs(inputs: dict[str, pl.DataFrame]) -> dict[str, pl.DataFrame]:
    frozen = dict(inputs)
    frozen["current_rosters"] = _week_filter(
        frozen["current_rosters"], ROSTER_CUTOFF_WEEK
    )
    frozen["current_snap_counts"] = _week_filter(
        frozen.get("current_snap_counts", pl.DataFrame()),
        CURRENT_SNAP_CUTOFF_WEEK,
    )

    depth = frozen.get("depth_charts", pl.DataFrame())
    depth = _date_filter(depth, "dt")
    frozen["depth_charts"] = depth

    injuries = _week_filter(
        frozen.get("injuries", pl.DataFrame()),
        ROSTER_CUTOFF_WEEK,
    )
    for column in ("date_modified", "report_date", "date"):
        injuries = _date_filter(injuries, column)
    frozen["injuries"] = injuries
    return frozen


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build a leakage-resistant 2026 Week 2 personnel snapshot. "
            "Week 1 current-season snaps are allowed; Week 2 snaps/results are not."
        )
    )
    parser.add_argument("--season", type=int, default=2026)
    parser.add_argument("--history", type=int, nargs="+", default=[2025])
    parser.add_argument("--recent-games", type=int, default=6)
    parser.add_argument("--cache-dir", type=Path, default=Path(".cache/monster"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.season != 2026:
        raise ValueError("Week 2 freezer is intentionally specific to 2026")

    original_loader = baseline.load_league_personnel_inputs

    def _frozen_loader(
        history_seasons: list[int],
        current_season: int,
        cache_dir: Path,
    ):
        return freeze_inputs(
            original_loader(history_seasons, current_season, cache_dir)
        )

    def _frozen_madden():
        return canonicalize_madden_attribute_mirror(
            load_madden27_player_ratings(MADDEN_FROZEN_URL)
        )

    baseline.load_league_personnel_inputs = _frozen_loader
    baseline.load_official_madden27_player_ratings = _frozen_madden

    original_argv = sys.argv[:]
    sys.argv = [
        original_argv[0],
        "--season",
        str(args.season),
        "--history",
        *[str(x) for x in args.history],
        "--recent-games",
        str(args.recent_games),
        "--cache-dir",
        str(args.cache_dir),
        "--out",
        str(args.out),
    ]
    try:
        baseline.main()
    finally:
        sys.argv = original_argv
        baseline.load_league_personnel_inputs = original_loader

    manifest_path = args.out / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["benchmark_input_freeze"] = {
        "name": "week2_2026_pregame_v1",
        "cutoff_date": CUTOFF_DATE.isoformat(),
        "roster_cutoff_week": ROSTER_CUTOFF_WEEK,
        "current_snap_cutoff_week": CURRENT_SNAP_CUTOFF_WEEK,
        "week1_current_snaps_allowed": True,
        "week2_current_snaps_allowed": False,
        "depth_charts_restricted_to_date_lte": CUTOFF_DATE.isoformat(),
        "injuries_restricted_to_week_and_date_lte": True,
        "madden_mirror_commit": MADDEN_MIRROR_COMMIT,
        "madden_frozen_url": MADDEN_FROZEN_URL,
        "week2_truth_available_to_builder": False,
        "market_blind": True,
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest["benchmark_input_freeze"], indent=2))


if __name__ == "__main__":
    main()
