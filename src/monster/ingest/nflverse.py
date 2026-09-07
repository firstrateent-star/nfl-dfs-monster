from __future__ import annotations

from io import BytesIO
from pathlib import Path

import nflreadpy as nfl
import polars as pl
import requests

PBP_COLUMNS = [
    "game_id",
    "season",
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
    "fumble_lost",
    "sack",
    "qb_hit",
    "qb_scramble",
    "air_yards",
    "yards_after_catch",
    "yards_gained",
    "touchdown",
    "pass_touchdown",
    "rush_touchdown",
    "field_goal_attempt",
    "field_goal_result",
    "receiver_player_id",
    "rusher_player_id",
    "passer_player_id",
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


def load_league_personnel_inputs(
    history_seasons: list[int], current_season: int, cache_dir: Path
) -> dict:
    """Load the reusable all-32-team personnel/capability layer without PBP.

    Historical player stats are small enough to belong here and provide free tackling,
    coverage and pressure evidence for defenders. The expensive play-by-play table stays
    in the separate heavy football-history path.
    """
    configure_cache(cache_dir)
    return {
        "players": nfl.load_players(),
        "teams": nfl.load_teams(),
        "current_rosters": _load_weekly_rosters(current_season),
        "injuries": _load_optional_current(nfl.load_injuries, current_season),
        "historical_snap_counts": nfl.load_snap_counts(history_seasons),
        "current_snap_counts": _load_current_snap_counts(current_season),
        "player_stats_history": nfl.load_player_stats(history_seasons),
        "combine": nfl.load_combine(),
        "depth_charts": _load_optional_current(nfl.load_depth_charts, current_season),
        "pfr_defense_weekly": nfl.load_pfr_advstats(
            history_seasons,
            stat_type="def",
            summary_level="week",
        ),
    }


def load_reference_inputs(
    history_seasons: list[int], current_season: int, cache_dir: Path
) -> dict:
    """Load heavy football-history inputs plus the reusable league personnel layer."""
    personnel = load_league_personnel_inputs(history_seasons, current_season, cache_dir)
    pbp = nfl.load_pbp(history_seasons)
    pbp = pbp.select([c for c in PBP_COLUMNS if c in pbp.columns])
    return {
        **personnel,
        "pbp": pbp,
        "player_stats": personnel["player_stats_history"],
        "team_stats": nfl.load_team_stats(history_seasons),
        "schedules": nfl.load_schedules(sorted(set(history_seasons + [current_season]))),
        "historical_rosters": nfl.load_rosters_weekly(history_seasons),
    }


def load_week_inputs(season: int, cache_dir: Path) -> dict:
    return load_reference_inputs([season], season, cache_dir)
