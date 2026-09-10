from __future__ import annotations

import argparse
import json
from pathlib import Path

import nflreadpy as nfl
import numpy as np
import polars as pl

from monster.ingest.nflverse import configure_cache


def _num(frame: pl.DataFrame, column: str) -> np.ndarray:
    if column not in frame.columns:
        return np.asarray([], dtype=float)
    return np.asarray(frame.get_column(column).drop_nulls().cast(pl.Float64).to_list(), dtype=float)


def _summary(values: np.ndarray) -> dict[str, float]:
    if values.size == 0:
        return {"mean": 0.0, "p10": 0.0, "p50": 0.0, "p90": 0.0}
    return {
        "mean": float(values.mean()),
        "p10": float(np.quantile(values, 0.10)),
        "p50": float(np.quantile(values, 0.50)),
        "p90": float(np.quantile(values, 0.90)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--season", type=int, default=2025)
    parser.add_argument("--cache-dir", type=Path, default=Path(".cache/monster"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    configure_cache(args.cache_dir)
    pbp = nfl.load_pbp([args.season])
    if "season_type" in pbp.columns:
        pbp = pbp.filter(pl.col("season_type") == "REG")

    punts = pbp.filter(pl.col("play_type") == "punt") if "play_type" in pbp.columns else pl.DataFrame()
    field_goals = (
        pbp.filter(pl.col("field_goal_attempt").fill_null(0) == 1)
        if "field_goal_attempt" in pbp.columns
        else pl.DataFrame()
    )
    kickoffs = (
        pbp.filter(pl.col("kickoff_attempt").fill_null(0) == 1)
        if "kickoff_attempt" in pbp.columns
        else pl.DataFrame()
    )

    punt_distance = _num(punts, "kick_distance")
    punt_return = _num(punts, "return_yards")
    fg_distance = _num(field_goals, "kick_distance")
    kickoff_return = _num(kickoffs, "return_yards")

    fg_rows = field_goals.to_dicts() if field_goals.height else []
    made = sum(str(row.get("field_goal_result", "")).lower() == "made" for row in fg_rows)
    punt_rows = punts.to_dicts() if punts.height else []
    kickoff_rows = kickoffs.to_dicts() if kickoffs.height else []

    def flag_rate(rows: list[dict[str, object]], column: str) -> float | None:
        if not rows or column not in pbp.columns:
            return None
        vals = [row.get(column) for row in rows if row.get(column) is not None]
        if not vals:
            return None
        return float(np.mean([float(value) == 1.0 for value in vals]))

    report = {
        "artifact": "Monster Historical Special-Teams Ecology Audit",
        "season": args.season,
        "market_blind": True,
        "behavior_changed_by_audit": False,
        "punts": {
            "attempts": punts.height,
            "kick_distance": _summary(punt_distance),
            "return_yards": _summary(punt_return),
            "touchback_rate": flag_rate(punt_rows, "punt_in_endzone"),
            "blocked_rate": flag_rate(punt_rows, "punt_blocked"),
        },
        "field_goals": {
            "attempts": field_goals.height,
            "make_rate": made / len(fg_rows) if fg_rows else 0.0,
            "kick_distance": _summary(fg_distance),
            "blocked_rate": flag_rate(fg_rows, "field_goal_blocked"),
        },
        "kickoffs": {
            "attempts": kickoffs.height,
            "return_yards": _summary(kickoff_return),
            "touchback_rate": flag_rate(kickoff_rows, "touchback"),
        },
        "available_columns": sorted(
            column
            for column in pbp.columns
            if any(token in column.lower() for token in ("kick", "punt", "field_goal", "return"))
        ),
        "promotion_status": "SHADOW_DIAGNOSTIC_ONLY",
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "special_teams_ecology.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
