from datetime import date

import polars as pl

from monster.feature_compile.trait_inputs import compile_player_trait_inputs
from monster.ingest.madden_skill import attach_madden_skill_traits, compile_madden_skill_traits


def _ratings():
    return pl.DataFrame({
        "full_name": ["Test Receiver", "Test Quarterback"],
        "position": ["WR", "QB"],
        "speed_rating": [96, 82],
        "acceleration_rating": [95, 84],
        "short_route_running_rating": [90, None],
        "medium_route_running_rating": [88, None],
        "deep_route_running_rating": [92, None],
        "catching_rating": [91, 60],
        "catch_in_traffic_rating": [87, 55],
        "spectacular_catch_rating": [89, 50],
        "release_rating": [93, 60],
        "carrying_rating": [70, 65],
        "break_tackle_rating": [72, 68],
        "strength_rating": [65, 70],
        "agility_rating": [94, 80],
        "change_of_direction_rating": [93, 78],
        "awareness_rating": [88, 92],
        "throw_power_rating": [None, 97],
        "throw_accuracy_short_rating": [None, 94],
        "throw_accuracy_mid_rating": [None, 91],
        "throw_accuracy_deep_rating": [None, 89],
        "throw_under_pressure_rating": [None, 90],
        "play_action_rating": [None, 88],
        "break_sack_rating": [None, 75],
    })


def _personnel():
    return pl.DataFrame({
        "display_name": ["Test Receiver", "Test Quarterback"],
        "position": ["WR", "QB"],
        "team_id": ["AAA", "AAA"],
        "gsis_id": ["wr1", "qb1"],
        "pfr_id": ["WR1", "QB1"],
        "height": [73.0, 75.0],
        "weight": [200.0, 225.0],
        "forty": [4.35, 4.75],
        "birth_date": [date(2000, 1, 1), date(1997, 1, 1)],
    })


def test_madden_skill_adapter_extracts_route_and_qb_accuracy():
    traits = compile_madden_skill_traits(_ratings())
    wr = traits.filter(pl.col("_madden_position") == "WR").row(0, named=True)
    qb = traits.filter(pl.col("_madden_position") == "QB").row(0, named=True)
    assert wr["madden_speed"] == 96
    assert wr["madden_route_running"] == 90
    assert qb["madden_throw_power"] == 97
    assert qb["madden_throw_accuracy"] > 90


def test_madden_traits_cross_into_player_mechanism_inputs():
    personnel = attach_madden_skill_traits(_personnel(), _ratings())
    inputs = compile_player_trait_inputs(personnel, game_date=date(2026, 9, 13))
    assert inputs["wr1"].height_in == 73
    assert inputs["wr1"].forty_time == 4.35
    assert inputs["wr1"].madden_speed == 96
    assert inputs["wr1"].madden_acceleration == 95
    assert inputs["wr1"].madden_route_running == 90
    assert inputs["wr1"].madden_catching == 91
    assert inputs["qb1"].age_years is not None
