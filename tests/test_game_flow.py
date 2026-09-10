from monster.sim.football_state import FootballState
from monster.sim.game_flow import FlowTag, derive_game_flow_state


def _state(**kwargs) -> FootballState:
    base = {
        "possession": "away",
        "defense": "home",
        "quarter": 1,
        "seconds_remaining": 3600,
        "yardline_100": 25.0,
        "down": 1,
        "distance": 10.0,
        "away_score": 0,
        "home_score": 0,
    }
    base.update(kwargs)
    return FootballState(**base)


def test_second_and_short_is_compositional_not_a_play_call() -> None:
    flow = derive_game_flow_state(_state(down=2, distance=2.0, yardline_100=50.0))
    assert FlowTag.SECOND_AND_SHORT in flow.tags
    assert FlowTag.MIDFIELD in flow.tags
    assert flow.distance == 2.0


def test_third_and_extreme_can_also_be_late_trailing_and_backed_up() -> None:
    flow = derive_game_flow_state(
        _state(
            quarter=4,
            seconds_remaining=75,
            down=3,
            distance=18.0,
            yardline_100=15.0,
            away_score=17,
            home_score=24,
        )
    )
    assert FlowTag.THIRD_AND_EXTREME in flow.tags
    assert FlowTag.TWO_MINUTE_GAME in flow.tags
    assert FlowTag.LATE_TRAILING in flow.tags
    assert FlowTag.BACKED_UP in flow.tags


def test_four_minute_lead_overlaps_field_position_state() -> None:
    flow = derive_game_flow_state(
        _state(
            quarter=4,
            seconds_remaining=220,
            down=1,
            distance=10.0,
            yardline_100=65.0,
            away_score=27,
            home_score=20,
        )
    )
    assert FlowTag.FOUR_MINUTE_LEAD in flow.tags
    assert FlowTag.OPPONENT_TERRITORY in flow.tags


def test_end_first_half_is_derived_from_period_clock() -> None:
    flow = derive_game_flow_state(
        _state(
            quarter=2,
            seconds_remaining=1875,
            down=1,
            distance=10.0,
            yardline_100=35.0,
        )
    )
    assert flow.seconds_remaining_in_period == 75
    assert FlowTag.END_FIRST_HALF in flow.tags


def test_low_red_zone_and_goal_to_go_like_can_overlap() -> None:
    flow = derive_game_flow_state(
        _state(down=2, distance=5.0, yardline_100=95.0)
    )
    assert flow.yards_to_goal == 5.0
    assert FlowTag.HIGH_RED_ZONE in flow.tags
    assert FlowTag.LOW_RED_ZONE in flow.tags
    assert FlowTag.GOAL_TO_GO_LIKE in flow.tags


def test_overtime_is_preserved_as_own_game_flow_tag() -> None:
    flow = derive_game_flow_state(
        _state(quarter=5, seconds_remaining=500, down=1, distance=10.0)
    )
    assert FlowTag.OVERTIME in flow.tags
