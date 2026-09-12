from __future__ import annotations

from monster.sim.decision_policy import FourthDownDecision, fourth_down_decision, situation_policy
from monster.sim.football_state import FootballState


def _state(**kwargs) -> FootballState:
    base = {
        "possession": "away",
        "defense": "home",
        "quarter": 1,
        "seconds_remaining": 3300,
        "yardline_100": 40.0,
        "down": 1,
        "distance": 10.0,
        "away_score": 0,
        "home_score": 0,
    }
    base.update(kwargs)
    return FootballState(**base)


def test_trailing_late_increases_pass_and_hurry_pressure() -> None:
    neutral = situation_policy(_state(), 0.56)
    late = situation_policy(
        _state(quarter=4, seconds_remaining=240, away_score=17, home_score=27), 0.56
    )
    assert late.pass_probability > neutral.pass_probability
    assert late.hurry_probability > neutral.hurry_probability


def test_late_lead_reduces_pass_pressure() -> None:
    tied = situation_policy(_state(quarter=4, seconds_remaining=240), 0.56)
    leading = situation_policy(
        _state(quarter=4, seconds_remaining=240, away_score=27, home_score=17), 0.56
    )
    assert leading.pass_probability < tied.pass_probability


def test_empirical_context_replaces_additive_down_distance_pressure() -> None:
    state = _state(down=1, distance=10.0)
    empirical = situation_policy(state, 0.56, contextual_pass_rate=0.44)
    assert empirical.pass_probability == 0.44


def test_empirical_context_still_responds_to_game_script() -> None:
    tied = situation_policy(
        _state(quarter=3, seconds_remaining=1200),
        0.56,
        contextual_pass_rate=0.50,
    )
    trailing = situation_policy(
        _state(quarter=3, seconds_remaining=1200, away_score=10, home_score=20),
        0.56,
        contextual_pass_rate=0.50,
    )
    assert trailing.pass_probability > tied.pass_probability


def test_short_fourth_down_in_opponent_territory_can_go() -> None:
    state = _state(yardline_100=68.0, down=4, distance=1.0)
    assert fourth_down_decision(state) == FourthDownDecision.GO


def test_makeable_fourth_down_field_goal_state_selects_kick() -> None:
    state = _state(yardline_100=65.0, down=4, distance=7.0)
    assert fourth_down_decision(state) == FourthDownDecision.FIELD_GOAL


def test_own_territory_fourth_down_selects_punt() -> None:
    state = _state(yardline_100=35.0, down=4, distance=6.0)
    assert fourth_down_decision(state) == FourthDownDecision.PUNT


def test_late_trailing_four_to_eight_points_preserves_possession() -> None:
    state = _state(
        quarter=4,
        seconds_remaining=180,
        yardline_100=48.0,
        down=4,
        distance=5.0,
        away_score=20,
        home_score=24,
    )
    assert fourth_down_decision(state) == FourthDownDecision.GO


def test_final_seconds_trailing_by_field_goal_prefers_tying_kick_when_available() -> None:
    state = _state(
        quarter=4,
        seconds_remaining=20,
        yardline_100=75.0,
        down=4,
        distance=5.0,
        away_score=20,
        home_score=23,
    )
    assert fourth_down_decision(state) == FourthDownDecision.FIELD_GOAL


def test_final_seconds_trailing_by_field_goal_must_go_when_kick_is_not_available() -> None:
    state = _state(
        quarter=4,
        seconds_remaining=20,
        yardline_100=35.0,
        down=4,
        distance=2.0,
        away_score=20,
        home_score=23,
    )
    assert fourth_down_decision(state) == FourthDownDecision.GO


def test_end_of_half_scoring_range_prefers_points_on_fourth_and_medium() -> None:
    state = _state(
        quarter=2,
        seconds_remaining=1818,
        yardline_100=75.0,
        down=4,
        distance=5.0,
    )
    assert fourth_down_decision(state) == FourthDownDecision.FIELD_GOAL


def test_end_of_half_own_territory_still_allows_punt() -> None:
    state = _state(
        quarter=2,
        seconds_remaining=1818,
        yardline_100=30.0,
        down=4,
        distance=7.0,
    )
    assert fourth_down_decision(state) == FourthDownDecision.PUNT
