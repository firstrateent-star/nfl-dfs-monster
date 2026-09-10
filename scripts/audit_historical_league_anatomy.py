from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import nflreadpy as nfl
import polars as pl

from monster.ingest.nflverse import configure_cache


def _count(frame: pl.DataFrame, expr: pl.Expr) -> int:
    return int(frame.select(expr.cast(pl.Int64).sum()).item() or 0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--season", type=int, default=2025)
    parser.add_argument("--cache-dir", type=Path, default=Path(".cache/monster"))
    parser.add_argument("--out", type=Path, default=Path("artifacts/historical-league-anatomy"))
    args = parser.parse_args()

    configure_cache(args.cache_dir)
    pbp = nfl.load_pbp([args.season])
    if "season_type" in pbp.columns:
        pbp = pbp.filter(pl.col("season_type") == "REG")
    if "qb_kneel" in pbp.columns:
        pbp = pbp.filter(pl.col("qb_kneel").fill_null(0) == 0)
    if "qb_spike" in pbp.columns:
        pbp = pbp.filter(pl.col("qb_spike").fill_null(0) == 0)

    schedules = nfl.load_schedules([args.season])
    regular = schedules.filter(
        pl.col("home_score").is_not_null() & pl.col("away_score").is_not_null()
    )
    if "game_type" in regular.columns:
        regular = regular.filter(pl.col("game_type") == "REG")
    games = regular.height
    if games <= 0:
        raise ValueError("no completed regular-season games found")

    dropback = pl.col("qb_dropback").fill_null(0) == 1
    scramble = pl.col("qb_scramble").fill_null(0) == 1
    sack = pl.col("sack").fill_null(0) == 1
    designed_run = (pl.col("rush_attempt").fill_null(0) == 1) & ~dropback
    throw = dropback & ~sack & ~scramble
    completion = pl.col("complete_pass").fill_null(0) == 1
    interception = pl.col("interception").fill_null(0) == 1
    fumble_lost = pl.col("fumble_lost").fill_null(0) == 1
    field_goal_attempt = pl.col("field_goal_attempt").fill_null(0) == 1
    field_goal_made = pl.col("field_goal_result").cast(pl.Utf8).str.to_lowercase() == "made"
    offensive_td = (pl.col("pass_touchdown").fill_null(0) == 1) | (
        pl.col("rush_touchdown").fill_null(0) == 1
    )
    any_td = pl.col("touchdown").fill_null(0) == 1
    punt = pl.col("play_type").cast(pl.Utf8).str.to_lowercase() == "punt"

    counts = {
        "scrimmage_plays": _count(pbp, dropback | designed_run),
        "dropbacks": _count(pbp, dropback),
        "pass_attempts": _count(pbp, throw),
        "designed_runs": _count(pbp, designed_run),
        "rush_attempts": _count(pbp, designed_run | scramble),
        "scrambles": _count(pbp, scramble),
        "sacks": _count(pbp, sack),
        "completions": _count(pbp, completion),
        "interceptions": _count(pbp, interception),
        "fumbles_lost": _count(pbp, fumble_lost),
        "punts": _count(pbp, punt),
        "field_goal_attempts": _count(pbp, field_goal_attempt),
        "field_goals_made": _count(pbp, field_goal_made),
        "offensive_touchdowns": _count(pbp, offensive_td),
        "all_touchdowns": _count(pbp, any_td),
    }

    drives = 0
    if "fixed_drive" in pbp.columns:
        drives = (
            pbp.filter(pl.col("posteam").is_not_null() & pl.col("fixed_drive").is_not_null())
            .select(["game_id", "posteam", "fixed_drive"])
            .unique()
            .height
        )

    scoreboard_points = float(
        regular.get_column("home_score").sum() + regular.get_column("away_score").sum()
    )
    per_game = {key: value / games for key, value in counts.items()}
    per_game["drives"] = drives / games if drives else 0.0
    per_game["scoreboard_points"] = scoreboard_points / games
    per_game["non_offensive_touchdowns"] = (
        counts["all_touchdowns"] - counts["offensive_touchdowns"]
    ) / games

    rates = {
        "completion_percentage": counts["completions"] / max(counts["pass_attempts"], 1),
        "sack_rate": counts["sacks"] / max(counts["dropbacks"], 1),
        "scramble_rate": counts["scrambles"] / max(counts["dropbacks"], 1),
        "interception_rate": counts["interceptions"] / max(counts["pass_attempts"], 1),
        "offensive_td_per_drive": counts["offensive_touchdowns"] / max(drives, 1),
        "fg_made_per_drive": counts["field_goals_made"] / max(drives, 1),
        "turnovers_per_drive": (counts["interceptions"] + counts["fumbles_lost"])
        / max(drives, 1),
    }

    manifest = {
        "artifact": "Monster Historical League Anatomy Audit",
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "season": args.season,
        "games": games,
        "scope": "regular season; kneels/spikes excluded where provider fields exist",
        "definitions": {
            "dropback": "qb_dropback == 1",
            "pass_attempt": "dropback excluding sacks and scrambles",
            "designed_run": "rush_attempt == 1 and not dropback",
            "rush_attempt": "designed run plus scramble",
            "offensive_touchdown": "pass_touchdown or rush_touchdown",
        },
        "counts": counts,
        "per_game": per_game,
        "rates": rates,
        "market_blind": True,
        "principle": "Use historical football anatomy to identify the causal scoring channel before changing v1.3 mechanics; do not tune to sportsbook totals.",
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "historical_league_anatomy.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    pl.DataFrame([{**per_game, **rates}]).write_csv(args.out / "historical_league_anatomy.csv")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
