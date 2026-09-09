from monster.feature_compile.units import UnitPlayerInputs, apply_team_unit_effects, compile_team_unit_effects
from monster.snapshot.model import TeamState


def _team():
    return TeamState(team_id="AAA", opponent_id="BBB")


def test_elite_qb_skill_moves_game_quality_state_upward():
    elite = UnitPlayerInputs(
        player_id="qb", position="QB", offense_snap_share=1.0,
        madden_throw_accuracy=96, madden_throw_power=96,
        madden_throw_under_pressure=96, madden_awareness=96,
    )
    poor = UnitPlayerInputs(
        player_id="qb", position="QB", offense_snap_share=1.0,
        madden_throw_accuracy=60, madden_throw_power=60,
        madden_throw_under_pressure=60, madden_awareness=60,
    )
    elite_effects, _ = compile_team_unit_effects((elite,))
    poor_effects, _ = compile_team_unit_effects((poor,))
    assert elite_effects.skill_talent_effect > 0
    assert poor_effects.skill_talent_effect < 0
    assert apply_team_unit_effects(_team(), elite_effects).physical_madden_effect > 0
    assert apply_team_unit_effects(_team(), poor_effects).physical_madden_effect < 0


def test_receiver_and_rb_traits_are_snap_weighted_not_direct_fantasy_points():
    wr = UnitPlayerInputs(
        player_id="wr", position="WR", offense_snap_share=0.8,
        madden_speed=95, madden_acceleration=94, madden_route_running=92,
        madden_catching=90, madden_release=91,
    )
    rb = UnitPlayerInputs(
        player_id="rb", position="RB", offense_snap_share=0.7,
        madden_speed=91, madden_acceleration=93, madden_carrying=89,
        madden_break_tackle=90, madden_ball_carrier_vision=92,
    )
    effects, _ = compile_team_unit_effects((wr, rb))
    assert 0 < effects.skill_talent_effect <= 0.045
