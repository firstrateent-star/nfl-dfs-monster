from __future__ import annotations

from pathlib import Path

import nflreadpy as nfl
import polars as pl

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


def _load_current_snap_counts(current_season: int) -> pl.DataFrame:
    """Current-season snap files may not exist before Week 1; fail neutral, not hard."""
    try:
        return nfl.load_snap_counts([current_season])
    except (OSError, RuntimeError, ValueError):
        return pl.DataFrame()


def load_league_personnel_inputs(
    history_seasons: list[int], current_season: int, cache_dir: Path
) -> dict:
    """Load the cheap, all-32-team personnel layer without downloading PBP.

    This is suitable for frequent free-tier refreshes. Static player identity/capability
    facts are separated from weekly roster/depth/injury/snap state.
    """
    configure_cache(cache_dir)
    return {
        "players": nfl.load_players(),
        "teams": nfl.load_teams(),
        "current_rosters": nfl.load_rosters_weekly([current_season]),
        "injuries": nfl.load_injuries([current_season]),
        "historical_snap_counts": nfl.load_snap_counts(history_seasons),
        "current_snap_counts": _load_current_snap_counts(current_season),
        "combine": nfl.load_combine(),
        "depth_charts": nfl.load_depth_charts([current_season]),
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
        "player_stats": nfl.load_player_stats(history_seasons),
        "team_stats": nfl.load_team_stats(history_seasons),
        "schedules": nfl.load_schedules(sorted(set(history_seasons + [current_season]))),
        "historical_rosters": nfl.load_rosters_weekly(history_seasons),
    }


def load_week_inputs(season: int, cache_dir: Path) -> dict:
    return load_reference_inputs([season], season, cache_dir)
