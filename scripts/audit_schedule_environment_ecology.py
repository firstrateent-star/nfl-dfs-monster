from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import date
from itertools import pairwise
from pathlib import Path
from typing import Any

import nflreadpy as nfl
import numpy as np
import polars as pl

from monster.ingest.nflverse import configure_cache


def _date_value(value: Any) -> date | None:
    if value is None:
        return None
    if isinstance(value, date):
        return value
    try:
        return date.fromisoformat(str(value)[:10])
    except ValueError:
        return None


def _summary(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "p10": 0.0, "p50": 0.0, "p90": 0.0}
    arr = np.asarray(values, dtype=float)
    return {
        "mean": float(arr.mean()),
        "p10": float(np.quantile(arr, 0.10)),
        "p50": float(np.quantile(arr, 0.50)),
        "p90": float(np.quantile(arr, 0.90)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--season", type=int, default=2025)
    parser.add_argument("--cache-dir", type=Path, default=Path(".cache/monster"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    configure_cache(args.cache_dir)
    schedules = nfl.load_schedules([args.season])
    if "game_type" in schedules.columns:
        schedules = schedules.filter(pl.col("game_type") == "REG")

    date_col = next((name for name in ("gameday", "game_date", "date") if name in schedules.columns), None)
    home_col = next((name for name in ("home_team", "home") if name in schedules.columns), None)
    away_col = next((name for name in ("away_team", "away") if name in schedules.columns), None)
    if date_col is None or home_col is None or away_col is None:
        raise ValueError("schedule environment audit cannot identify date/home/away columns")

    team_games: dict[str, list[date]] = defaultdict(list)
    rows = []
    for row in schedules.to_dicts():
        game_date = _date_value(row.get(date_col))
        if game_date is None:
            continue
        home = str(row[home_col])
        away = str(row[away_col])
        team_games[home].append(game_date)
        team_games[away].append(game_date)
        rows.append(row)

    rest_days = []
    short_rest = 0
    long_rest = 0
    rest_observations = 0
    for dates in team_games.values():
        ordered = sorted(set(dates))
        for previous, current in pairwise(ordered):
            rest = (current - previous).days
            rest_days.append(float(rest))
            rest_observations += 1
            short_rest += int(rest <= 6)
            long_rest += int(rest >= 10)

    categorical = {}
    for column in ("roof", "surface", "stadium", "location"):
        if column not in schedules.columns:
            continue
        counts = schedules.group_by(column).len().sort("len", descending=True)
        categorical[column] = counts.head(20).to_dicts()

    report = {
        "artifact": "Monster Schedule / Rest / Venue Evidence Audit",
        "season": args.season,
        "market_blind": True,
        "behavior_changed_by_audit": False,
        "teams": len(team_games),
        "games_with_dates": len(rows),
        "rest_observations": rest_observations,
        "rest_days": _summary(rest_days),
        "short_rest_rate_6_or_less": short_rest / rest_observations if rest_observations else 0.0,
        "long_rest_rate_10_or_more": long_rest / rest_observations if rest_observations else 0.0,
        "categorical_coverage": categorical,
        "available_environment_columns": sorted(
            column
            for column in schedules.columns
            if any(
                token in column.lower()
                for token in ("roof", "surface", "stadium", "weather", "temp", "wind", "location")
            )
        ),
        "travel_distance_status": "not_authorized_without_definition-safe team-location/stadium coordinates",
        "promotion_status": "SHADOW_DIAGNOSTIC_ONLY",
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "schedule_environment_ecology.json").write_text(json.dumps(report, indent=2, default=str) + "\n")
    print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()
