from __future__ import annotations

import argparse
import json

import nflreadpy as nfl
import polars as pl


def _flag(frame: pl.DataFrame, column: str) -> pl.Expr:
    if column not in frame.columns:
        return pl.lit(False)
    return pl.col(column).fill_null(0).cast(pl.Int64, strict=False) == 1


def season_drive_starts(season: int) -> dict[str, float | int]:
    pbp = nfl.load_pbp([season])
    if "season_type" in pbp.columns:
        pbp = pbp.filter(pl.col("season_type") == "REG")
    required = {"game_id", "fixed_drive", "posteam", "yardline_100"}
    if not required.issubset(pbp.columns):
        raise RuntimeError(f"Missing drive-start columns: {sorted(required - set(pbp.columns))}")

    # Kickoff/punt rows share drive identifiers in nflverse and cannot be used as the receiving
    # offense's starting field position. The first true scrimmage snap is the observable state we
    # want: pass attempt, rush attempt or sack. This also excludes administrative/no-play rows.
    scrimmage = _flag(pbp, "pass_attempt") | _flag(pbp, "rush_attempt") | _flag(pbp, "sack")
    sort_cols = [column for column in ["game_id", "fixed_drive", "play_id"] if column in pbp.columns]
    starts = (
        pbp.filter(
            pl.col("posteam").is_not_null()
            & pl.col("fixed_drive").is_not_null()
            & pl.col("yardline_100").is_not_null()
            & scrimmage
        )
        .sort(sort_cols)
        .group_by(["game_id", "fixed_drive"], maintain_order=True)
        .agg(pl.col("yardline_100").first().alias("yardline_100"))
        .with_columns((100.0 - pl.col("yardline_100")).alias("start_from_own_goal"))
    )
    values = starts.get_column("start_from_own_goal")
    return {
        "season": season,
        "drives": starts.height,
        "mean": float(values.mean()),
        "p10": float(values.quantile(0.10)),
        "p25": float(values.quantile(0.25)),
        "p50": float(values.quantile(0.50)),
        "p75": float(values.quantile(0.75)),
        "p90": float(values.quantile(0.90)),
        "short_field_40_plus_rate": float((values >= 40.0).mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seasons", nargs="+", type=int, default=[2024, 2025])
    args = parser.parse_args()
    rows = [season_drive_starts(season) for season in args.seasons]
    print(json.dumps({"drive_starts": rows}, indent=2))


if __name__ == "__main__":
    main()
