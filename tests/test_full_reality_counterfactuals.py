from __future__ import annotations

from monster.feature_compile.mechanisms import (
    PlayerMechanismInputs,
    TeamMechanismInputs,
    _weather_effect,
    compile_player_mechanisms,
    compile_team_mechanisms,
)
from monster.snapshot.model import TeamState
from monster.snapshot.player import PlayerState, TeamPlayerPool


def _player(position: str = "WR") -> PlayerState:
    return PlayerState(player_id="p1", display_name="Test Player", position=position, team_id="T")


def test_elite_speed_evidence_moves_explosiveness_in_expected_direction():
    base = _player("WR")
    fast, fast_trace = compile_player_mechanisms(
        base,
        PlayerMechanismInputs(forty_time=4.30, madden_speed=96.0, madden_acceleration=95.0),
    )
    slow, slow_trace = compile_player_mechanisms(
        base,
        PlayerMechanismInputs(forty_time=4.70, madden_speed=72.0, madden_acceleration=74.0),
    )
    assert fast_trace.speed_signal > slow_trace.speed_signal
    assert fast.explosive_modifier > slow.explosive_modifier


def test_catchpoint_traits_move_catchpoint_not_direct_fantasy_state():
    base = _player("TE")
    strong, trace = compile_player_mechanisms(
        base,
        PlayerMechanismInputs(
            height_in=79.0,
            wingspan_in=84.0,
            madden_catching=94.0,
            madden_route_running=90.0,
        ),
    )
    assert trace.catchpoint_signal > 0.0
    assert strong.catchpoint_modifier > base.catchpoint_modifier
    assert strong.target_share == base.target_share
    assert strong.receiving_td_share == base.receiving_td_share


def test_age_alone_has_zero_production_penalty():
    base = _player("RB")
    compiled, trace = compile_player_mechanisms(base, PlayerMechanismInputs(age_years=35.0))
    assert trace.biology_uncertainty == 0.0
    assert compiled.explosive_modifier == base.explosive_modifier
    assert compiled.rushing_efficiency_modifier == base.rushing_efficiency_modifier
    assert compiled.role_uncertainty == base.role_uncertainty


def test_age_plus_high_workload_changes_uncertainty_not_mean_skill():
    base = _player("RB")
    compiled, trace = compile_player_mechanisms(
        base,
        PlayerMechanismInputs(age_years=35.0, career_workload=2500.0),
    )
    assert trace.biology_uncertainty > 0.0
    assert compiled.role_uncertainty > base.role_uncertainty
    assert compiled.explosive_modifier == base.explosive_modifier
    assert compiled.rushing_efficiency_modifier == base.rushing_efficiency_modifier


def test_astrology_is_trace_only_and_has_zero_production_authority():
    base = _player("WR")
    neutral, _ = compile_player_mechanisms(base, PlayerMechanismInputs())
    shadow, trace = compile_player_mechanisms(base, PlayerMechanismInputs(astrology_shadow_signal=1.0))
    assert trace.astrology_shadow_signal == 1.0
    assert shadow == neutral


def test_dome_nullifies_weather_and_open_air_bad_weather_reduces_pass_environment():
    assert _weather_effect(TeamMechanismInputs(dome=True, wind_mph=35.0, precipitation_probability=1.0)) == 0.0
    team = TeamState(team_id="T", opponent_id="O")
    pool = TeamPlayerPool(team_id="T", players=(_player("WR"),))
    dome_team, dome_pool, _ = compile_team_mechanisms(
        team, pool, TeamMechanismInputs(dome=True, wind_mph=35.0, precipitation_probability=1.0)
    )
    bad_team, bad_pool, trace = compile_team_mechanisms(
        team, pool, TeamMechanismInputs(dome=False, wind_mph=35.0, precipitation_probability=1.0, temperature_f=15.0)
    )
    assert trace.weather_effect < 0.0
    assert bad_team.weather_effect < dome_team.weather_effect
    assert bad_pool.neutral_pass_rate < dome_pool.neutral_pass_rate
    assert bad_pool.targetable_dropback_rate < dome_pool.targetable_dropback_rate


def test_missing_team_trait_evidence_is_neutral():
    team = TeamState(team_id="T", opponent_id="O")
    pool = TeamPlayerPool(team_id="T", players=(_player("WR"),))
    compiled_team, compiled_pool, trace = compile_team_mechanisms(team, pool, TeamMechanismInputs())
    assert trace.personnel_signal == 0.0
    assert compiled_team.physical_madden_effect == 0.0
    assert compiled_team.weather_effect == 0.0
    assert compiled_pool.neutral_pass_rate == pool.neutral_pass_rate
