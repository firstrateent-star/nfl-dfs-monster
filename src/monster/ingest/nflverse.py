from __future__ import annotations

from io import BytesIO
from pathlib import Path

import nflreadpy as nfl
import polars as pl
import requests

PBP_COLUMNS = [
    "game_id",
    "play_id",
    "season",
    "season_type",
    "week",
    "home_team",
    "away_team",
    "posteam",
    "defteam",
    "posteam_type",
    "play_type",
    "qtr",
    "down",
    "goal_to_go",
    "ydstogo",
    "yardline_100",
    "game_seconds_remaining",
    "score_differential",
    "fixed_drive",
    "epa",
    "success",
    "pass_attempt",
    "rush_attempt",
    "qb_dropback",
    "complete_pass",
    "interception",
    "interception_player_id",
    "fumble_lost",
    "fumble_recovery_1_team",
    "fumble_recovery_1_player_id",
    "fumble_recovery_1_yards",
    "fumble_recovery_2_team",
    "fumble_recovery_2_player_id",
    "fumble_recovery_2_yards",
    "sack",
    "qb_hit",
    "qb_scramble",
    "qb_sneak",
    "air_yards",
    "yards_after_catch",
    "yards_gained",
    "kick_distance",
    "return_yards",
    "return_team",
    "touchdown",
    "td_team",
    "return_touchdown",
    "pass_touchdown",
    "rush_touchdown",
    "safety",
    "field_goal_attempt",
    "field_goal_result",
    "punt_attempt",
    "punt_blocked",
    "punt_fair_catch",
    "punt_downed",
    "punt_out_of_bounds",
    "punt_in_endzone",
    "kickoff_attempt",
    "kickoff_fair_catch",
    "kickoff_downed",
    "kickoff_out_of_bounds",
    "kickoff_in_endzone",
    "touchback",
    "kickoff_returner_player_id",
    "punt_returner_player_id",
    "receiver_player_id",
    "rusher_player_id",
    "passer_player_id",
    "run_location",
    "run_gap",
    "pass_location",
    "pass_length",
    "shotgun",
    "no_huddle",
]


def configure_cache(cache_dir: Path) -> None:
    from nflreadpy.config import update_config

    update_config(
        cache_mode="filesystem",
        cache_dir=cache_dir,
        cache_duration=21_600,
        verbose=False,
        user_agent="nfl-dfs-monster/0.1",
    )


def _read_public_parquet(url: str) -> pl.DataFrame:
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    return pl.read_parquet(BytesIO(response.content))


def _load_weekly_rosters(season: int) -> pl.DataFrame:
    """Use nflreadpy when current; otherwise read nflverse's release asset directly."""
    try:
        return nfl.load_rosters_weekly([season])
    except ValueError:
        url = (
            "https://github.com/nflverse/nflverse-data/releases/download/weekly_rosters/"
            f"roster_weekly_{season}.parquet"
        )
        return _read_public_parquet(url)


def _load_optional_current(loader, current_season: int) -> pl.DataFrame:
    try:
        return loader([current_season])
    except (OSError, RuntimeError, ValueError, requests.RequestException):
        return pl.DataFrame()


def _load_current_snap_counts(current_season: int) -> pl.DataFrame:
    return _load_optional_current(nfl.load_snap_counts, current_season)


def _supplement_player_ids(players: pl.DataFrame, ff_ids: pl.DataFrame) -> pl.DataFrame:
    """Fill missing cross-provider IDs from the GSIS-keyed ffverse identity table."""
    if players.is_empty() or ff_ids.is_empty() or "gsis_id" not in players.columns:
        return players

    mapping_columns = [
        column
        for column in ["gsis_id", "espn_id", "yahoo_id", "sleeper_id", "pfr_id"]
        if column in ff_ids.columns
    ]
    if len(mapping_columns) <= 1:
        return players

    mapping = ff_ids.select(mapping_columns).unique(subset=["gsis_id"], keep="first")
    renamed = mapping.rename({column: f"ff_{column}" for column in mapping_columns if column != "gsis_id"})
    enriched = players.join(renamed, on="gsis_id", how="left")
    expressions = []
    for column in ["espn_id", "yahoo_id", "sleeper_id", "pfr_id"]:
        ff_column = f"ff_{column}"
        if ff_column not in enriched.columns:
            continue
        if column in enriched.columns:
            expressions.append(pl.coalesce([pl.col(column), pl.col(ff_column)]).alias(column))
        else:
            expressions.append(pl.col(ff_column).alias(column))
    if expressions:
        enriched = enriched.with_columns(expressions)
    drop_columns = [column for column in enriched.columns if column.startswith("ff_")]
    return enriched.drop(drop_columns) if drop_columns else enriched
