from __future__ import annotations

from monster.sim.football_state import (
    FootballState,
    apply_scrimmage_yards,
    kickoff_transition,
    missed_field_goal_transition,
    punt_transition,
    turnover_at_spot,
    turnover_on_downs,
)


def _state(**kwargs) -> FootballState:
    base = {
        "possession": "away",
        "defense": "home",
        "quarter": 2,
        "seconds_remaining": 1400,
        "yardline_100": 42.0,
        "down": 2,
        "distance": 7.0,
        "away_score": 10,
        "home_score": 14,
    }
    base.update(kwargs)
    return FootballState(**base)


def test_score_margin_uses_explicit_team_orientation_for_real_ids() -> None:
    away = FootballState(
        possession="CHI",
        defense="CAR",
        away_score=10,
        home_score=17,
        away_team_id="CHI",
        home_team_id="CAR",
    )
    assert away.score_margin_for_offense == -7

    home = kickoff_transition(away, receiving_yardline_100=30.0, elapsed_seconds=0)
    assert home.possession == "CAR"
    assert home.score_margin_for_offense == 7
    assert home.away_team_id == "CHI"
    assert home.home_team_id == "CAR"


def test_first_down_preserves_possession_and_updates_clock_field() -> None:
    after = apply_scrimmage_yards(_state(), yards=9.0, elapsed_seconds=31)
    assert after.possession == "away"
    assert after.seconds_remaining == 1369
    assert after.yardline_100 == 51.0
    assert after.down == 1
    assert after.distance == 10.0


def test_failed_play_advances_down_and_distance() -> None:
    after = apply_scrimmage_yards(_state(), yards=3.0, elapsed_seconds=27)
    assert after.down == 3
    assert after.distance == 4.0
    assert after.yardline_100 == 45.0


def test_turnover_mirrors_same_physical_field_spot() -> None:
    after = turnover_at_spot(_state(), spot_yardline_100=68.0, elapsed_seconds=6)
    assert after.possession == "home"
    assert after.defense == "away"
    assert after.yardline_100 == 32.0
    assert after.down == 1
    assert after.distance == 10.0
    assert after.seconds_remaining == 1394


def test_turnover_on_downs_does_not_reset_to_25() -> None:
    after = turnover_on_downs(_state(yardline_100=73.0), elapsed_seconds=5)
    assert after.yardline_100 == 27.0
    assert after.possession == "home"


def test_punt_creates_opponent_starting_field_position() -> None:
    after = punt_transition(
        _state(yardline_100=35.0),
        gross_yards=46.0,
        return_yards=8.0,
        elapsed_seconds=9,
    )
    # Punt lands at physical 81; opponent owns the mirrored 19 plus an 8-yard return.
    assert after.yardline_100 == 27.0
    assert after.possession == "home"
    assert after.seconds_remaining == 1391


def test_punt_touchback_starts_at_twenty_in_kernel() -> None:
    after = punt_transition(_state(yardline_100=70.0), gross_yards=45.0, touchback=True)
    assert after.yardline_100 == 20.0


def test_kickoff_is_explicit_change_of_possession() -> None:
    after = kickoff_transition(_state(), receiving_yardline_100=30.0, elapsed_seconds=6)
    assert after.possession == "home"
    assert after.yardline_100 == 30.0


def test_missed_field_goal_uses_kick_spot_not_drive_reset() -> None:
    after = missed_field_goal_transition(
        _state(yardline_100=64.0), kick_spot_yardline_100=57.0, elapsed_seconds=5
    )
    assert after.possession == "home"
    assert after.yardline_100 == 43.0
