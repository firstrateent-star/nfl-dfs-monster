from datetime import date

import polars as pl

from monster.feature_compile.reality_inputs import compile_player_reality_inputs


def test_full_reality_inputs_carry_physical_and_madden_evidence():
    personnel = pl.DataFrame(
        {
            "gsis_id": ["p1"], "pfr_id": [None], "position": ["WR"],
            "height": [74.0], "weight": [205.0], "wingspan": [80.0], "forty": [4.36],
            "birth_date": ["2000-01-01"], "madden_speed": [95.0],
            "madden_acceleration": [94.0], "madden_route_running": [91.0],
            "madden_catching": [90.0],
        }
    )
    inputs = compile_player_reality_inputs(personnel, game_date=date(2026, 9, 13))["p1"]
    assert inputs.height_in == 74.0
    assert inputs.weight_lbs == 205.0
    assert inputs.wingspan_in == 80.0
    assert inputs.forty_time == 4.36
    assert inputs.madden_speed == 95.0
    assert inputs.madden_acceleration == 94.0
    assert inputs.madden_route_running == 91.0
    assert inputs.madden_catching == 90.0
    assert 26.6 < inputs.age_years < 26.8


def test_full_reality_inputs_wire_health_and_experience_without_scoring_bonus():
    personnel = pl.DataFrame(
        {
            "gsis_id": ["p1"], "position": ["RB"], "birth_date": ["1996-01-01"],
            "years_of_experience": [8.0], "health_availability_probability": [0.72],
            "health_effectiveness_if_active": [0.81],
        }
    )
    inputs = compile_player_reality_inputs(personnel, game_date=date(2026, 9, 13))["p1"]
    assert inputs.active_probability == 0.72
    assert inputs.effectiveness_if_active == 0.81
    assert inputs.career_workload == 8.0 * 190.0


def test_full_reality_inputs_never_use_fantasy_or_market_columns():
    personnel = pl.DataFrame(
        {
            "gsis_id": ["p1"], "pfr_id": [None], "position": ["RB"],
            "height": [70.0], "weight": [215.0], "birth_date": ["2001-01-01"],
            "salary": [10000], "ownership": [0.99], "vegas_total": [60.0],
            "fantasy_projection": [40.0],
        }
    )
    inputs = compile_player_reality_inputs(personnel, game_date=date(2026, 9, 13))["p1"]
    assert inputs.height_in == 70.0
    assert inputs.weight_lbs == 215.0
    assert inputs.madden_speed is None
