from monster.feature_compile.mechanisms import (
    PlayerMechanismInputs,
    TeamMechanismInputs,
    compile_player_mechanisms,
    compile_team_mechanisms,
)
from monster.snapshot.model import TeamState
from monster.snapshot.player import PlayerState, TeamPlayerPool


def _player(position: str = "WR") -> PlayerState:
    return PlayerState(
        player_id="p1",
        display_name="Test Player",
        position=position,
        team_id="A",
        target_share=0.24,
        rush_share=0.03,
        role_uncertainty=0.10,
    )


def test_missing_extra_metrics_are_neutral():
    base = _player()
    compiled, trace = compile_player_mechanisms(base, PlayerMechanismInputs())
    assert compiled == base
    assert trace.speed_signal == 0.0
    assert trace.catchpoint_signal == 0.0
    assert trace.rushing_signal == 0.0


def test_physical_and_madden_effects_are_directional_and_bounded():
    base = _player()
    compiled, _ = compile_player_mechanisms(
        base,
        PlayerMechanismInputs(
            height_in=77.0,
            weight_lbs=220.0,
            wingspan_in=82.0,
            forty_time=4.28,
            madden_speed=99.0,
            madden_acceleration=98.0,
            madden_route_running=94.0,
            madden_catching=95.0,
        ),
    )
    assert 1.0 < compiled.explosive_modifier <= 1.05
    assert 1.0 < compiled.catchpoint_modifier <= 1.05
    assert 0.95 <= compiled.rushing_efficiency_modifier <= 1.05


def test_age_alone_cannot_penalize_expected_player_mechanisms():
    base = _player("RB")
    compiled, trace = compile_player_mechanisms(
        base,
        PlayerMechanismInputs(age_years=34.0),
    )
    assert compiled.effectiveness_if_active == base.effectiveness_if_active
    assert compiled.explosive_modifier == base.explosive_modifier
    assert compiled.rushing_efficiency_modifier == base.rushing_efficiency_modifier
    assert compiled.role_uncertainty == base.role_uncertainty
    assert trace.biology_uncertainty == 0.0


def test_age_plus_heavy_workload_changes_uncertainty_not_mean():
    base = _player("RB")
    compiled, trace = compile_player_mechanisms(
        base,
        PlayerMechanismInputs(age_years=33.0, career_workload=2400.0),
    )
    assert compiled.effectiveness_if_active == base.effectiveness_if_active
    assert compiled.role_uncertainty > base.role_uncertainty
    assert trace.biology_uncertainty > 0.0


def test_astrology_is_shadow_only():
    base = _player()
    first, trace_first = compile_player_mechanisms(
        base,
        PlayerMechanismInputs(astrology_shadow_signal=-0.95),
    )
    second, trace_second = compile_player_mechanisms(
        base,
        PlayerMechanismInputs(astrology_shadow_signal=0.95),
    )
    assert first == second
    assert trace_first.astrology_shadow_signal == -0.95
    assert trace_second.astrology_shadow_signal == 0.95


def test_continuity_reduces_role_uncertainty():
    base = _player()
    high, _ = compile_player_mechanisms(base, PlayerMechanismInputs(unit_continuity=1.0))
    low, _ = compile_player_mechanisms(base, PlayerMechanismInputs(unit_continuity=0.0))
    assert high.role_uncertainty < base.role_uncertainty < low.role_uncertainty


def test_bad_weather_is_bounded_and_dome_nullifies_it():
    team = TeamState("A", "H")
    pool = TeamPlayerPool("A", (_player(),))
    outdoor_team, outdoor_pool, outdoor_trace = compile_team_mechanisms(
        team,
        pool,
        TeamMechanismInputs(
            wind_mph=28.0,
            precipitation_probability=0.9,
            temperature_f=18.0,
            dome=False,
        ),
    )
    dome_team, dome_pool, dome_trace = compile_team_mechanisms(
        team,
        pool,
        TeamMechanismInputs(
            wind_mph=28.0,
            precipitation_probability=0.9,
            temperature_f=18.0,
            dome=True,
        ),
    )
    assert -0.06 <= outdoor_team.weather_effect < 0.0
    assert outdoor_pool.neutral_pass_rate < pool.neutral_pass_rate
    assert outdoor_pool.targetable_dropback_rate < pool.targetable_dropback_rate
    assert outdoor_trace.weather_effect < 0.0
    assert dome_team.weather_effect == 0.0
    assert dome_pool.neutral_pass_rate == pool.neutral_pass_rate
    assert dome_trace.weather_effect == 0.0


def test_team_personnel_proxy_effect_is_small_and_bounded():
    team = TeamState("A", "H")
    pool = TeamPlayerPool("A", (_player(),))
    compiled_team, _, trace = compile_team_mechanisms(
        team,
        pool,
        TeamMechanismInputs(
            team_madden_ovr=99.0,
            offensive_line_index=5.0,
            opponent_front_index=-4.0,
        ),
    )
    assert 0.0 < compiled_team.physical_madden_effect <= 0.035
    assert 0.0 < trace.personnel_signal <= 1.0
