from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import nflreadpy as nfl
import polars as pl
from audit_week1_drive_reality import _summarize

from monster.ingest.nflverse import configure_cache
from monster.sim.football_state import PossessionTerminal


def _number(row: dict[str, Any], name: str, default: float = 0.0) -> float:
    value = row.get(name)
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _flag(row: dict[str, Any], name: str) -> bool:
    return _number(row, name) == 1.0


def _is_scrimmage(row: dict[str, Any]) -> bool:
    dropback = _flag(row, "qb_dropback")
    designed_run = _flag(row, "rush_attempt") and not dropback
    return dropback or designed_run


def _first_number(rows: list[dict[str, Any]], name: str) -> float | None:
    for row in rows:
        value = row.get(name)
        if value is not None:
            try:
                return float(value)
            except (TypeError, ValueError):
                continue
    return None


def _first_scrimmage_number(rows: list[dict[str, Any]], name: str) -> float | None:
    return _first_number([row for row in rows if _is_scrimmage(row)], name)


def _last_number(rows: list[dict[str, Any]], name: str) -> float | None:
    for row in reversed(rows):
        value = row.get(name)
        if value is not None:
            try:
                return float(value)
            except (TypeError, ValueError):
                continue
    return None


def _terminal(rows: list[dict[str, Any]]) -> str:
    offensive_td = any(_flag(row, "pass_touchdown") or _flag(row, "rush_touchdown") for row in rows)
    if offensive_td:
        return PossessionTerminal.TOUCHDOWN.value
    if any(_flag(row, "safety") for row in rows):
        return PossessionTerminal.SAFETY.value

    fg_rows = [row for row in rows if _flag(row, "field_goal_attempt")]
    if fg_rows:
        made = any(str(row.get("field_goal_result", "")).lower() == "made" for row in fg_rows)
        return PossessionTerminal.FIELD_GOAL.value if made else PossessionTerminal.MISSED_FIELD_GOAL.value
    if any(str(row.get("play_type", "")).lower() == "punt" for row in rows):
        return PossessionTerminal.PUNT.value
    if any(_flag(row, "interception") or _flag(row, "fumble_lost") for row in rows):
        return PossessionTerminal.TURNOVER.value
    if any(_flag(row, "fourth_down_failed") for row in rows):
        return PossessionTerminal.TURNOVER_ON_DOWNS.value
    return PossessionTerminal.END_GAME.value


def _drive_points(rows: list[dict[str, Any]], terminal: str) -> int:
    start_score = _first_scrimmage_number(rows, "posteam_score")
    end_score = _last_number(rows, "posteam_score_post")
    if start_score is not None and end_score is not None and end_score >= start_score:
        delta = round(end_score - start_score)
        if 0 <= delta <= 8:
            return delta
    if terminal == PossessionTerminal.FIELD_GOAL.value:
        return 3
    if terminal == PossessionTerminal.TOUCHDOWN.value:
        return 7
    return 0


def _historical_first_down(row: dict[str, Any]) -> bool:
    if "first_down_pass" in row or "first_down_rush" in row:
        return _flag(row, "first_down_pass") or _flag(row, "first_down_rush")
    return _flag(row, "first_down")


def _drive_row(
    game_id: str,
    fixed_drive: str,
    posteam: str,
    rows: list[dict[str, Any]],
) -> dict[str, object] | None:
    first_yardline_to_goal = _first_scrimmage_number(rows, "yardline_100")
    if first_yardline_to_goal is None:
        return None
    start_yardline_from_own = 100.0 - first_yardline_to_goal

    start_seconds_remaining = _first_scrimmage_number(rows, "game_seconds_remaining")
    end_seconds_remaining = _last_number(rows, "game_seconds_remaining")
    start_quarter = _first_scrimmage_number(rows, "qtr")
    if start_seconds_remaining is None or end_seconds_remaining is None:
        return None

    scrimmage_plays = 0
    net_scrimmage_yards = 0.0
    first_downs = 0
    explosive_plays = 0
    red_zone_reached = False
    red_zone_snap_seen = False
    goal_to_go_snap_seen = False
    sacks = 0
    turnovers = 0
    series_started = 0
    series_converted = 0
    first_down_snaps = 0
    second_down_snaps = 0
    third_down_snaps = 0
    fourth_down_snaps = 0
    third_down_conversions = 0
    third_and_long_snaps = 0
    third_and_long_conversions = 0
    third_down_distance_total = 0.0
    early_down_5plus_gains = 0
    early_down_run_snaps = 0
    early_down_run_yards_total = 0.0
    early_down_run_negative_gains = 0
    early_down_run_3plus_gains = 0
    early_down_run_5plus_gains = 0
    early_down_run_10plus_gains = 0
    early_down_pass_snaps = 0
    early_down_pass_yards_total = 0.0
    early_down_pass_negative_gains = 0
    early_down_pass_3plus_gains = 0
    early_down_pass_5plus_gains = 0
    early_down_pass_10plus_gains = 0

    for row in rows:
        if not _is_scrimmage(row):
            continue

        dropback = _flag(row, "qb_dropback")
        designed_run = _flag(row, "rush_attempt") and not dropback
        yardline_to_goal = row.get("yardline_100")
        yards = _number(row, "yards_gained")
        down = int(_number(row, "down", 0.0))
        distance = _number(row, "ydstogo", 0.0)
        if yardline_to_goal is not None:
            yardline_to_goal_f = float(yardline_to_goal)
            red_zone_snap_seen = red_zone_snap_seen or yardline_to_goal_f <= 20.0
            if "goal_to_go" in row and row.get("goal_to_go") is not None:
                goal_to_go_snap_seen = goal_to_go_snap_seen or _flag(row, "goal_to_go")
            else:
                goal_to_go_snap_seen = goal_to_go_snap_seen or (
                    yardline_to_goal_f <= 10.0 and distance >= yardline_to_goal_f
                )
            end_yardline_to_goal = yardline_to_goal_f - yards
            red_zone_reached = red_zone_reached or yardline_to_goal_f <= 20.0 or end_yardline_to_goal <= 20.0

        scrimmage_plays += 1
        net_scrimmage_yards += yards
        explosive_plays += int(yards >= 15.0)
        sacks += int(_flag(row, "sack"))
        turnovers += int(_flag(row, "interception") or _flag(row, "fumble_lost"))

        converted = _historical_first_down(row)
        first_downs += int(converted)
        if down == 1:
            series_started += 1
            first_down_snaps += 1
        elif down == 2:
            second_down_snaps += 1
        elif down == 3:
            third_down_snaps += 1
            third_down_distance_total += distance
            if distance >= 7.0:
                third_and_long_snaps += 1
        elif down == 4:
            fourth_down_snaps += 1

        if converted:
            series_converted += 1
            if down == 3:
                third_down_conversions += 1
                if distance >= 7.0:
                    third_and_long_conversions += 1

        if down in {1, 2}:
            early_down_5plus_gains += int(yards >= 5.0)
            if designed_run:
                early_down_run_snaps += 1
                early_down_run_yards_total += yards
                early_down_run_negative_gains += int(yards < 0.0)
                early_down_run_3plus_gains += int(yards >= 3.0)
                early_down_run_5plus_gains += int(yards >= 5.0)
                early_down_run_10plus_gains += int(yards >= 10.0)
            elif dropback:
                early_down_pass_snaps += 1
                early_down_pass_yards_total += yards
                early_down_pass_negative_gains += int(yards < 0.0)
                early_down_pass_3plus_gains += int(yards >= 3.0)
                early_down_pass_5plus_gains += int(yards >= 5.0)
                early_down_pass_10plus_gains += int(yards >= 10.0)

    terminal = _terminal(rows)
    return {
        "game_id": game_id,
        "fixed_drive": fixed_drive,
        "offense_team_id": posteam,
        "defense_team_id": "historical_opponent",
        "start_seconds_remaining": float(start_seconds_remaining),
        "end_seconds_remaining": float(end_seconds_remaining),
        "start_yardline_100": float(start_yardline_from_own),
        "terminal": terminal,
        "points": _drive_points(rows, terminal),
        "scrimmage_plays": scrimmage_plays,
        "net_scrimmage_yards": float(net_scrimmage_yards),
        "first_downs": first_downs,
        "explosive_plays": explosive_plays,
        "red_zone_entered": red_zone_reached,
        "red_zone_snap_seen": red_zone_snap_seen,
        "goal_to_go_reached": goal_to_go_snap_seen,
        "goal_to_go_snap_seen": goal_to_go_snap_seen,
        "pressured_dropbacks": 0,
        "sacks": sacks,
        "turnovers": turnovers,
        "overtime": bool(start_quarter is not None and start_quarter >= 5.0),
        "series_started": series_started,
        "series_converted": series_converted,
        "first_down_snaps": first_down_snaps,
        "second_down_snaps": second_down_snaps,
        "third_down_snaps": third_down_snaps,
        "fourth_down_snaps": fourth_down_snaps,
        "third_down_conversions": third_down_conversions,
        "third_and_long_snaps": third_and_long_snaps,
        "third_and_long_conversions": third_and_long_conversions,
        "third_down_distance_total": float(third_down_distance_total),
        "early_down_5plus_gains": early_down_5plus_gains,
        "early_down_run_snaps": early_down_run_snaps,
        "early_down_run_yards_total": float(early_down_run_yards_total),
        "early_down_run_negative_gains": early_down_run_negative_gains,
        "early_down_run_3plus_gains": early_down_run_3plus_gains,
        "early_down_run_5plus_gains": early_down_run_5plus_gains,
        "early_down_run_10plus_gains": early_down_run_10plus_gains,
        "early_down_pass_snaps": early_down_pass_snaps,
        "early_down_pass_yards_total": float(early_down_pass_yards_total),
        "early_down_pass_negative_gains": early_down_pass_negative_gains,
        "early_down_pass_3plus_gains": early_down_pass_3plus_gains,
        "early_down_pass_5plus_gains": early_down_pass_5plus_gains,
        "early_down_pass_10plus_gains": early_down_pass_10plus_gains,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--season", type=int, default=2025)
    parser.add_argument("--cache-dir", type=Path, default=Path(".cache/monster"))
    parser.add_argument("--out", type=Path, default=Path("artifacts/historical-drive-reality"))
    args = parser.parse_args()

    configure_cache(args.cache_dir)
    pbp = nfl.load_pbp([args.season])
    if "season_type" in pbp.columns:
        pbp = pbp.filter(pl.col("season_type") == "REG")
    if "qb_kneel" in pbp.columns:
        pbp = pbp.filter(pl.col("qb_kneel").fill_null(0) == 0)
    if "qb_spike" in pbp.columns:
        pbp = pbp.filter(pl.col("qb_spike").fill_null(0) == 0)
    required = {"game_id", "fixed_drive", "posteam", "play_id", "game_seconds_remaining"}
    missing = sorted(required.difference(pbp.columns))
    if missing:
        raise ValueError(f"historical drive audit missing required nflverse fields: {missing}")

    usable = pbp.filter(
        pl.col("game_id").is_not_null()
        & pl.col("fixed_drive").is_not_null()
        & pl.col("posteam").is_not_null()
        & pl.col("game_seconds_remaining").is_not_null()
    ).sort(["game_id", "fixed_drive", "play_id"])

    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in usable.to_dicts():
        key = (str(row["game_id"]), str(row["fixed_drive"]), str(row["posteam"]))
        grouped[key].append(row)

    groups_first_event_non_scrimmage = sum(bool(rows) and not _is_scrimmage(rows[0]) for rows in grouped.values())
    candidate_rows = [
        _drive_row(game_id, fixed_drive, posteam, rows)
        for (game_id, fixed_drive, posteam), rows in grouped.items()
    ]
    drive_rows = [row for row in candidate_rows if row is not None]
    groups_without_scrimmage = len(candidate_rows) - len(drive_rows)
    if not drive_rows:
        raise ValueError("historical audit produced no definition-safe offensive drives")

    schedules = nfl.load_schedules([args.season])
    regular = schedules.filter(pl.col("home_score").is_not_null() & pl.col("away_score").is_not_null())
    if "game_type" in regular.columns:
        regular = regular.filter(pl.col("game_type") == "REG")
    games = regular.height
    if games <= 0:
        raise ValueError("no completed regular-season games found")

    overall = _summarize(drive_rows, scope=f"{args.season}_regular_season")
    overall["drives_per_game"] = len(drive_rows) / games

    args.out.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(drive_rows).write_parquet(args.out / "historical_drive_traces.parquet")
    pl.DataFrame([overall]).write_csv(args.out / "historical_drive_reality_overall.csv")

    manifest = {
        "artifact": "Monster Historical Drive Reality Audit",
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "season": args.season,
        "games": games,
        "drive_groups_seen": len(grouped),
        "definition_safe_drives": len(drive_rows),
        "groups_without_scrimmage": groups_without_scrimmage,
        "groups_first_event_non_scrimmage": groups_first_event_non_scrimmage,
        "market_blind": True,
        "scope": "regular season; kneels/spikes excluded where provider fields exist",
        "definitions": {
            "drive_population": "unique game_id + fixed_drive + posteam groups containing at least one definition-safe scrimmage play",
            "drive_start": "first definition-safe scrimmage play in play_id order; kickoff and other non-scrimmage events cannot define field position",
            "start_yardline_100": "100 - nflverse yardline_100 from the first definition-safe scrimmage play",
            "drive_clock": "start uses game_seconds_remaining at the first definition-safe scrimmage snap; end uses game_seconds_remaining on the last retained event in the fixed-drive group",
            "scrimmage_play": "qb_dropback == 1 OR (rush_attempt == 1 AND not qb_dropback)",
            "explosive_play": "scrimmage yards_gained >= 15",
            "red_zone_reach": "definition-safe scrimmage snap starts at nflverse yardline_100 <= 20 OR its endpoint reaches <= 20",
            "red_zone_snap": "definition-safe scrimmage play begins with nflverse yardline_100 <= 20",
            "touchdown_terminal": "pass_touchdown or rush_touchdown; return scores remain turnover drives",
            "survival_series": "scrimmage series begins on down == 1; conversion uses nflverse first_down_pass/first_down_rush or first_down fallback",
            "third_and_long": "definition-safe third down with ydstogo >= 7",
            "early_down_5plus": "definition-safe first/second-down scrimmage gain >= 5 yards",
            "early_down_ownership": "first/second-down scrimmage events split into designed runs versus qb_dropbacks with mean, negative, 3+, 5+, and 10+ gain anatomy",
            "pressure": "not compared here; play-level pressure requires nflverse participation join",
        },
        "terminal_precedence": [
            "offensive_touchdown",
            "safety",
            "field_goal_attempt",
            "punt",
            "interception_or_fumble_lost",
            "fourth_down_failed",
            "period_or_other_end",
        ],
        "files": {
            "raw": "historical_drive_traces.parquet",
            "overall": "historical_drive_reality_overall.csv",
        },
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"manifest": manifest, "overall": overall}, indent=2))


if __name__ == "__main__":
    main()
