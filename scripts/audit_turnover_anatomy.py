from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import nflreadpy as nfl
import polars as pl


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seasons", nargs="+", type=int, default=[2022, 2023, 2024, 2025])
    parser.add_argument("--out", type=Path, default=Path("artifacts/turnover-anatomy"))
    args = parser.parse_args()

    pbp = nfl.load_pbp(args.seasons)
    if "season_type" in pbp.columns:
        pbp = pbp.filter(pl.col("season_type") == "REG")
    interceptions = int(pbp.select(pl.col("interception").fill_null(0).sum()).item())
    fumbles_lost = int(pbp.select(pl.col("fumble_lost").fill_null(0).sum()).item())
    total = interceptions + fumbles_lost
    if total <= 0:
        raise RuntimeError("Historical turnover audit found no turnovers")

    by_season = (
        pbp.group_by("season")
        .agg(
            pl.col("interception").fill_null(0).sum().alias("interceptions"),
            pl.col("fumble_lost").fill_null(0).sum().alias("fumbles_lost"),
        )
        .with_columns(
            (
                pl.col("interceptions")
                / (pl.col("interceptions") + pl.col("fumbles_lost")).clip(lower_bound=1)
            ).alias("interception_fraction")
        )
        .sort("season")
    )
    result = {
        "artifact": "Monster Historical Turnover Anatomy Audit",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "seasons": args.seasons,
        "interceptions": interceptions,
        "fumbles_lost": fumbles_lost,
        "turnovers": total,
        "interception_fraction": interceptions / total,
        "fumble_lost_fraction": fumbles_lost / total,
        "market_blind": True,
        "principle": "Use football history to decompose the frozen team-turnover reservoir; never use DFS scoring or market information to set turnover anatomy.",
    }
    args.out.mkdir(parents=True, exist_ok=True)
    by_season.write_csv(args.out / "by_season.csv")
    (args.out / "manifest.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
