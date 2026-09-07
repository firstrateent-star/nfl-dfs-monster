from __future__ import annotations

import polars as pl

from monster.snapshot.league import compile_team_state_map, team_state_from_policy_row


def test_policy_row_compiles_market_blind_team_state():
    row = {
        "team_id": "A",
        "neutral_pass_rate": 0.61,
        "neutral_seconds_per_play": 25.0,
        "drives_per_game": 11.2,
        "td_drive_rate": 0.27,
        "fg_drive_rate": 0.12,
        "turnover_drive_rate": 0.09,
        "red_zone_td_rate": 0.63,
        "offensive_epa_per_play": 0.10,
        "offensive_success_rate": 0.49,
        "offensive_explosive_rate": 0.13,
        "defensive_td_drive_rate_allowed": 0.19,
        "defensive_fg_drive_rate_allowed": 0.13,
        "defensive_takeaway_drive_rate": 0.12,
        "defensive_epa_allowed_per_play": -0.03,
        "defensive_explosive_rate_allowed": 0.09,
        "defensive_sack_rate": 0.08,
        "defensive_qb_hit_rate": 0.21,
    }
    team = team_state_from_policy_row(row, opponent_id="B", prior_uncertainty=0.14)
    assert team.team_id == "A"
    assert team.opponent_id == "B"
    assert team.neutral_pass_rate == 0.61
    assert team.pace_factor > 1.0
    assert team.uncertainty == 0.14


def test_missing_policy_team_falls_back_neutral_with_uncertainty():
    policy = pl.DataFrame({"team_id": ["A"], "neutral_pass_rate": [0.60]})
    states = compile_team_state_map(
        policy,
        {"A": "B", "B": "A"},
        prior_uncertainty=0.15,
    )
    assert states["A"].neutral_pass_rate == 0.60
    assert states["B"].neutral_pass_rate == 0.56
    assert states["B"].uncertainty == 0.15
