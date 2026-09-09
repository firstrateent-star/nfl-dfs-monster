from __future__ import annotations

from monster.sim.clock import (
    advance_game_clock,
    halftime_crossed,
    quarter_from_clock,
    regulation_complete,
    seconds_remaining_in_quarter,
    two_minute_warning_crossed,
)
from monster.sim.football_state import FootballState


def _state(seconds: int, quarter: int = 1) -> FootballState:
    return FootballState(
        possession="away",
        defense="home",
        quarter=quarter,
        seconds_remaining=seconds,
        yardline_100=25.0,
        down=1,
        distance=10.0,
    )


def test_quarter_is_derived_from_one_regulation_clock() -> None:
    assert quarter_from_clock(3600) == 1
    assert quarter_from_clock(2701) == 1
    assert quarter_from_clock(2700) == 2
    assert quarter_from_clock(1800) == 3
    assert quarter_from_clock(900) == 4
    assert quarter_from_clock(0) == 4


def test_clock_consumption_crosses_quarter_boundary() -> None:
    after = advance_game_clock(_state(2710), 20)
    assert after.seconds_remaining == 2690
    assert after.quarter == 2
    assert seconds_remaining_in_quarter(after.seconds_remaining) == 890


def test_halftime_is_a_detectable_state_transition() -> None:
    before = _state(1812, quarter=2)
    after = advance_game_clock(before, 20)
    assert halftime_crossed(before, after)
    assert after.quarter == 3


def test_two_minute_warning_boundaries_are_detectable() -> None:
    q2_before = _state(1930, quarter=2)
    q2_after = advance_game_clock(q2_before, 20)
    assert two_minute_warning_crossed(q2_before, q2_after)

    q4_before = _state(130, quarter=4)
    q4_after = advance_game_clock(q4_before, 20)
    assert two_minute_warning_crossed(q4_before, q4_after)


def test_regulation_ends_at_zero_and_never_goes_negative() -> None:
    after = advance_game_clock(_state(8, quarter=4), 35)
    assert after.seconds_remaining == 0
    assert after.quarter == 4
    assert regulation_complete(after)
