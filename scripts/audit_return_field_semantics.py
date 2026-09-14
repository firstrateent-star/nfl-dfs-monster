from __future__ import annotations

import argparse
import json
from pathlib import Path

import nflreadpy as nfl
import polars as pl


TOKENS = (
    "fumble",
    "recovery",
    "return",
    "punt",
    "kickoff",
    "touchback",
    "fair_catch",
    "blocked",
)


def _bool(frame: pl.DataFrame, column: str) -> pl.Expr:
    if column not in frame.columns:
        return pl.lit(False)
    return pl.col(column).fill_null(0).cast(pl.Int64, strict=False) == 1


def _not_null(frame: pl.DataFrame, column: str) -> pl.Expr:
    if column not in frame.columns:
        return pl.lit(False)
    return pl.col(column).is_not_null()


def _profile(frame: pl.DataFrame, *, yard_col: str = "return_yards") -> dict[str, float | int]:
    if not frame.height or yard_col not in frame.columns:
        return {"rows": 0}
    usable = frame.filter(pl.col(yard_col).is_not_null())
    if not usable.height:
        return {"rows": 0}
    yards = pl.col(yard_col).cast(pl.Float64, strict=False)
    return {
        "rows": usable.height,
        "mean": float(usable.select(yards.mean()).item()),
        "zero_rate": float(usable.select((yards <= 0).cast(pl.Float64).mean()).item()),
        "20_plus_rate": float(usable.select((yards >= 20).cast(pl.Float64).mean()).item()),
        "40_plus_rate": float(usable.select((yards >= 40).cast(pl.Float64).mean()).item()),
        "60_plus_rate": float(usable.select((yards >= 60).cast(pl.Float64).mean()).item()),
        "80_plus_rate": float(usable.select((yards >= 80).cast(pl.Float64).mean()).item()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--season", type=int, default=2025)
    parser.add_argument("--out", type=Path, default=Path("artifacts/return-field-semantics.json"))
    args = parser.parse_args()

    pbp = nfl.load_pbp([args.season])
    if "season_type" in pbp.columns:
        pbp = pbp.filter(pl.col("season_type") == "REG")

    relevant_columns = sorted(
        column for column in pbp.columns if any(token in column.lower() for token in TOKENS)
    )

    fumbles = pbp.filter(_bool(pbp, "fumble_lost"))
    fumble_tds = fumbles.filter(_bool(fumbles, "touchdown"))
    fumble_candidate_cols = [
        column
        for column in relevant_columns
        if "fumble" in column.lower() or "recovery" in column.lower()
    ]
    fumble_field_coverage = {}
    for column in fumble_candidate_cols:
        series = fumbles.get_column(column)
        fumble_field_coverage[column] = {
            "dtype": str(series.dtype),
            "non_null_rows": int(series.is_not_null().sum()),
            "td_non_null_rows": int(fumble_tds.get_column(column).is_not_null().sum())
            if column in fumble_tds.columns
            else 0,
        }
        if series.dtype.is_numeric():
            fumble_field_coverage[column]["all_profile"] = _profile(fumbles, yard_col=column)
            fumble_field_coverage[column]["td_profile"] = _profile(fumble_tds, yard_col=column)

    punts = pbp.filter(_bool(pbp, "punt_attempt"))
    kickoffs = pbp.filter(_bool(pbp, "kickoff_attempt"))

    punt_returner = _not_null(punts, "punt_returner_player_id")
    punt_fair = _bool(punts, "punt_fair_catch") | _bool(punts, "fair_catch")
    punt_touchback = _bool(punts, "touchback")
    punt_blocked = _bool(punts, "punt_blocked")
    punt_live = punts.filter(punt_returner & ~punt_fair & ~punt_touchback & ~punt_blocked)

    kickoff_returner = _not_null(kickoffs, "kickoff_returner_player_id")
    kickoff_touchback = _bool(kickoffs, "touchback")
    kickoff_live = kickoffs.filter(kickoff_returner & ~kickoff_touchback)

    payload = {
        "season": args.season,
        "relevant_columns": relevant_columns,
        "fumble": {
            "lost_rows": fumbles.height,
            "touchdown_rows": fumble_tds.height,
            "field_coverage": fumble_field_coverage,
        },
        "punt": {
            "attempt_rows": punts.height,
            "returner_id_rows": int(punts.select(punt_returner.cast(pl.Int64).sum()).item()),
            "fair_catch_rows": int(punts.select(punt_fair.cast(pl.Int64).sum()).item()),
            "touchback_rows": int(punts.select(punt_touchback.cast(pl.Int64).sum()).item()),
            "blocked_rows": int(punts.select(punt_blocked.cast(pl.Int64).sum()).item()),
            "live_return_rows": punt_live.height,
            "all_return_yards_profile": _profile(punts),
            "live_return_yards_profile": _profile(punt_live),
            "return_td_rows": int(punts.select(_bool(punts, "return_touchdown").cast(pl.Int64).sum()).item()),
        },
        "kickoff": {
            "attempt_rows": kickoffs.height,
            "returner_id_rows": int(kickoffs.select(kickoff_returner.cast(pl.Int64).sum()).item()),
            "touchback_rows": int(kickoffs.select(kickoff_touchback.cast(pl.Int64).sum()).item()),
            "live_return_rows": kickoff_live.height,
            "all_return_yards_profile": _profile(kickoffs),
            "live_return_yards_profile": _profile(kickoff_live),
            "return_td_rows": int(
                kickoffs.select(_bool(kickoffs, "return_touchdown").cast(pl.Int64).sum()).item()
            ),
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()