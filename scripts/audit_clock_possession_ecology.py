from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import nflreadpy as nfl
import numpy as np
import polars as pl

from monster.ingest.nflverse import configure_cache


def _num(row: dict[str, Any], key: str) -> float | None:
    value = row.get(key)
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
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


def _simulated(frame: pl.DataFrame, *, games: int, worlds: int) -> dict[str, object]:
    rows = frame.to_dicts()
    regulation = [row for row in rows if not bool(row.get("overtime"))]
    durations = []
    for row in regulation:
        start = float(row["start_seconds_remaining"])
        end = float(row["end_seconds_remaining"])
        duration = max(start - end, 0.0)
        # halftime/end-game administrative traces can span reset boundaries; keep only
        # definition-safe possession durations inside a regulation half.
        if 0.0 <= duration <= 900.0:
            durations.append(duration)
    return {
        "drives_per_game": len(regulation) / max(games * worlds, 1),
        "drive_duration_seconds": _summary(durations),
        "usable_drive_durations": len(durations),
    }


def _historical(season: int, cache_dir: Path) -> dict[str, object]:
    configure_cache(cache_dir)
    pbp = nfl.load_pbp([season])
    if "season_type" in pbp.columns:
        pbp = pbp.filter(pl.col("season_type") == "REG")
    if "qb_kneel" in pbp.columns:
        pbp = pbp.filter(pl.col("qb_kneel").fill_null(0) == 0)
    if "qb_spike" in pbp.columns:
        pbp = pbp.filter(pl.col("qb_spike").fill_null(0) == 0)

    required = {"game_id", "fixed_drive", "posteam", "play_id", "game_seconds_remaining"}
    missing = sorted(required.difference(pbp.columns))
    if missing:
        raise ValueError(f"clock audit missing nflverse fields: {missing}")

    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    usable = pbp.filter(
        pl.col("game_id").is_not_null()
        & pl.col("fixed_drive").is_not_null()
        & pl.col("posteam").is_not_null()
        & pl.col("game_seconds_remaining").is_not_null()
    ).sort(["game_id", "fixed_drive", "play_id"])
    for row in usable.to_dicts():
        grouped[(str(row["game_id"]), str(row["fixed_drive"]), str(row["posteam"]))].append(row)

    durations = []
    drive_count = 0
    for rows in grouped.values():
        scrimmage = [
            row
            for row in rows
            if _num(row, "qb_dropback") == 1.0
            or (_num(row, "rush_attempt") == 1.0 and _num(row, "qb_dropback") != 1.0)
        ]
        if not scrimmage:
            continue
        start = _num(scrimmage[0], "game_seconds_remaining")
        end = _num(rows[-1], "game_seconds_remaining")
        if start is None or end is None:
            continue
        duration = max(start - end, 0.0)
        if duration <= 900.0:
            durations.append(duration)
        drive_count += 1

    schedules = nfl.load_schedules([season])
    games_frame = schedules.filter(pl.col("home_score").is_not_null() & pl.col("away_score").is_not_null())
    if "game_type" in games_frame.columns:
        games_frame = games_frame.filter(pl.col("game_type") == "REG")
    games = games_frame.height
    if games <= 0:
        raise ValueError("clock audit found no completed regular-season games")

    return {
        "drives_per_game": drive_count / games,
        "drive_duration_seconds": _summary(durations),
        "usable_drive_durations": len(durations),
        "games": games,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--simulated", type=Path, required=True)
    parser.add_argument("--sim-manifest", type=Path, required=True)
    parser.add_argument("--season", type=int, default=2025)
    parser.add_argument("--cache-dir", type=Path, default=Path(".cache/monster"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    frame = pl.read_parquet(args.simulated)
    manifest = json.loads(args.sim_manifest.read_text())
    sim = _simulated(frame, games=int(manifest["games"]), worlds=int(manifest["worlds_per_game"]))
    hist = _historical(args.season, args.cache_dir)

    report = {
        "artifact": "Monster v1.3 Clock and Possession Ecology Audit",
        "market_blind": True,
        "behavior_changed_by_audit": False,
        "simulated": sim,
        "historical": hist,
        "comparison": {
            "drives_per_game_delta": float(sim["drives_per_game"]) - float(hist["drives_per_game"]),
            "mean_drive_duration_seconds_delta": float(sim["drive_duration_seconds"]["mean"])
            - float(hist["drive_duration_seconds"]["mean"]),
        },
        "interpretation_rule": "Clock mismatches localize opportunity-supply mechanics only. Do not tune game totals directly.",
        "promotion_status": "SHADOW_DIAGNOSTIC_ONLY",
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "clock_possession_ecology.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
