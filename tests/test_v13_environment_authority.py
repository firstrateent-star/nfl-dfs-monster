from __future__ import annotations

import polars as pl

from monster.feature_compile.v13_environment_authority import apply_v13_game_environment
from monster.snapshot.model import TeamState
from monster.snapshot.player import PlayerState, TeamPlayerPool


def _state(team: str, opponent: str) -> TeamState:
    return TeamState(team_id=team, opponent_id=opponent)


def _pool(team: str) -> TeamPlayerPool:
    return TeamPlayerPool(
        team_id=team,
        players=(PlayerState(f"{team}-wr", "WR", "WR", team),),
        neutral_pass_rate=0.56,
        targetable_dropback_rate=0.94,
    )


def test_shared_open_air_environment_affects_both_teams_equally() -> None:
    environment = pl.DataFrame(
        {
            "team_id": ["H"],
            "venue": ["Test Field"],
            "dome": [False],
            "surface": ["grass"],
            "temperature_f": [20.0],
            "wind_mph": [25.0],
            "precipitation_probability": [0.8],
            "source_note": ["test"],
        }
    )
    away_state, home_state, away_pool, home_pool, trace = apply_v13_game_environment(
        away_state=_state("A", "H"),
        home_state=_state("H", "A"),
        away_pool=_pool("A"),
        home_pool=_pool("H"),
        environment=environment,
        home_team="H",
    )
    assert away_state.weather_effect == home_state.weather_effect < 0.0
    assert away_pool.neutral_pass_rate == home_pool.neutral_pass_rate < 0.56
    assert trace.home_team == "H"
    assert trace.surface == "grass"


def test_dome_environment_is_exact_null() -> None:
    environment = pl.DataFrame(
        {
            "team_id": ["H"],
            "venue": ["Dome"],
            "dome": [True],
            "surface": ["turf"],
            "temperature_f": [5.0],
            "wind_mph": [40.0],
            "precipitation_probability": [1.0],
            "source_note": ["test"],
        }
    )
    away_state, home_state, away_pool, home_pool, trace = apply_v13_game_environment(
        away_state=_state("A", "H"),
        home_state=_state("H", "A"),
        away_pool=_pool("A"),
        home_pool=_pool("H"),
        environment=environment,
        home_team="H",
    )
    assert away_state.weather_effect == home_state.weather_effect == 0.0
    assert away_pool.neutral_pass_rate == home_pool.neutral_pass_rate == 0.56
    assert trace.dome is True
