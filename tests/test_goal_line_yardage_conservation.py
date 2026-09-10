from __future__ import annotations

from monster.sim.football_state import FootballState
from monster.sim.play_kernel import _credit_scrimmage_yards


def _state(yardline_100: float) -> FootballState:
    return FootballState(
        possession="A",
        defense="B",
        yardline_100=yardline_100,
        away_team_id="A",
        home_team_id="B",
    )


def test_positive_gain_cannot_be_credited_beyond_goal_line() -> None:
    assert _credit_scrimmage_yards(_state(90.0), 30.0) == 10.0
    assert _credit_scrimmage_yards(_state(99.0), 8.5) == 1.0


def test_non_scoring_positive_gain_is_unchanged() -> None:
    assert _credit_scrimmage_yards(_state(25.0), 18.25) == 18.25


def test_negative_gain_is_not_clipped_by_goal_line() -> None:
    assert _credit_scrimmage_yards(_state(95.0), -7.0) == -7.0
