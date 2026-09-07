from __future__ import annotations
from pathlib import Path
import nflreadpy as nfl

PBP_COLUMNS = [
    "game_id", "season", "week", "posteam", "defteam", "play_type", "down", "ydstogo",
    "yardline_100", "game_seconds_remaining", "score_differential", "epa", "success",
    "pass_attempt", "rush_attempt", "complete_pass", "interception", "sack", "qb_scramble",
    "air_yards", "yards_after_catch", "yards_gained", "touchdown", "pass_touchdown",
    "rush_touchdown", "receiver_player_id", "rusher_player_id", "passer_player_id",
]


def configure_cache(cache_dir: Path) -> None:
    from nflreadpy.config import update_config
    update_config(cache_mode="filesystem", cache_dir=str(cache_dir), cache_duration=21_600, verbose=False)


def load_week_inputs(season: int, cache_dir: Path):
    configure_cache(cache_dir)
    pbp = nfl.load_pbp([season])
    available = [c for c in PBP_COLUMNS if c in pbp.columns]
    pbp = pbp.select(available)
    return {
        "pbp": pbp,
        "player_stats": nfl.load_player_stats([season]),
        "team_stats": nfl.load_team_stats([season]),
        "schedules": nfl.load_schedules([season]),
        "weekly_rosters": nfl.load_rosters_weekly([season]),
        "injuries": nfl.load_injuries([season]),
        "snap_counts": nfl.load_snap_counts([season]),
        "combine": nfl.load_combine(),
    }
