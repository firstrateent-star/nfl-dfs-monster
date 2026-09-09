from monster.feature_compile.environment import apply_environment
from monster.feature_compile.mechanisms import TeamMechanismInputs
from monster.snapshot.model import TeamState
from monster.snapshot.player import PlayerState, TeamPlayerPool


def _state():
    return TeamState(team_id="A", opponent_id="B")


def _pool():
    return TeamPlayerPool(
        team_id="A",
        players=(PlayerState("p", "Player", "WR", "A"),),
        neutral_pass_rate=0.56,
        targetable_dropback_rate=0.94,
    )


def test_dome_is_exact_environment_null_across_both_jurisdictions():
    team, pool = apply_environment(
        _state(),
        _pool(),
        TeamMechanismInputs(
            wind_mph=35.0,
            precipitation_probability=1.0,
            temperature_f=5.0,
            dome=True,
        ),
    )
    assert team.weather_effect == 0.0
    assert pool.neutral_pass_rate == 0.56
    assert pool.targetable_dropback_rate == 0.94


def test_open_air_bad_weather_reduces_scoring_and_pass_opportunity_state():
    team, pool = apply_environment(
        _state(),
        _pool(),
        TeamMechanismInputs(
            wind_mph=25.0,
            precipitation_probability=0.8,
            temperature_f=20.0,
            dome=False,
        ),
    )
    assert team.weather_effect < 0.0
    assert pool.neutral_pass_rate < 0.56
    assert pool.targetable_dropback_rate < 0.94
