from __future__ import annotations

import numpy as np

from monster.feature_compile.mechanisms import PlayerMechanismInputs
from monster.feature_compile.v13_identity_bridge import compile_v13_player_identity
from monster.sim.football_state import FootballState
from monster.sim.rules_v13 import (
    PenaltyEvent,
    PenaltySide,
    TryResult,
    choose_two_point,
    enforce_penalty,
    is_safety,
    overtime_required,
    simulate_penalty,
    simulate_try,
)
from monster.sim.special_teams_v13 import simulate_field_goal, simulate_kickoff, simulate_punt


def test_penalties_have_both_offensive_and_defensive_paths() -> None:
    rng = np.random.default_rng(101)
    events = [simulate_penalty(rng, base_rate=1.0) for _ in range(300)]
    sides = {event.side for event in events if event is not None}
    assert PenaltySide.OFFENSE in sides
    assert PenaltySide.DEFENSE in sides


def test_penalties_change_field_and_series_state() -> None:
    state = FootballState("away", "home", yardline_100=40.0, down=2, distance=7.0)
    offensive = enforce_penalty(state, PenaltyEvent(PenaltySide.OFFENSE, 10))
    assert offensive.yardline_100 == 30.0
    assert offensive.down == 2
    assert offensive.distance == 17.0
    defensive = enforce_penalty(state, PenaltyEvent(PenaltySide.DEFENSE, 5, automatic_first_down=True))
    assert defensive.yardline_100 == 45.0
    assert defensive.down == 1
    assert defensive.distance == 10.0


def test_late_score_state_can_choose_two_point_try() -> None:
    assert choose_two_point(quarter=4, seconds_remaining=240, score_margin_after_td=-8)
    assert not choose_two_point(quarter=2, seconds_remaining=240, score_margin_after_td=-8)


def test_try_anatomy_has_pat_and_two_point_outcomes() -> None:
    pat_rng = np.random.default_rng(102)
    two_rng = np.random.default_rng(103)
    pats = {simulate_try(pat_rng, go_for_two=False).result for _ in range(500)}
    twos = {simulate_try(two_rng, go_for_two=True).result for _ in range(500)}
    assert TryResult.PAT_GOOD in pats and TryResult.PAT_MISS in pats
    assert TryResult.TWO_POINT_GOOD in twos and TryResult.TWO_POINT_FAIL in twos


def test_safety_and_overtime_are_explicit_rules_states() -> None:
    assert is_safety(yardline_100=3.0, yards=-4.0)
    assert overtime_required(24, 24)
    assert not overtime_required(24, 21)


def test_special_teams_generate_explicit_event_anatomy() -> None:
    rng = np.random.default_rng(104)
    kickoff = simulate_kickoff(rng, returner_id="kr", kicker_id="k")
    punt = simulate_punt(rng, returner_id="pr", punter_id="p")
    fg = simulate_field_goal(rng, distance=47.0, kicker_id="k")
    assert kickoff.event_type == "kickoff"
    assert punt.event_type == "punt"
    assert fg.event_type == "field_goal"
    assert fg.made is not None


def test_full_reality_bridge_is_neutral_when_evidence_missing() -> None:
    identity, trace = compile_v13_player_identity(
        player_id="p1",
        name="Neutral Player",
        position="WR",
        usage_weight=0.2,
        inputs=PlayerMechanismInputs(),
    )
    assert identity.efficiency == 1.0
    assert identity.explosive == 1.0
    assert identity.turnover_security == 1.0
    assert trace.evidence_fields == 0


def test_full_reality_bridge_moves_play_traits_with_bounded_evidence() -> None:
    identity, trace = compile_v13_player_identity(
        player_id="p2",
        name="Evidence Player",
        position="WR",
        usage_weight=0.3,
        inputs=PlayerMechanismInputs(
            height_in=75.0,
            weight_lbs=215.0,
            forty_time=4.35,
            madden_speed=94.0,
            madden_acceleration=93.0,
            madden_route_running=91.0,
            madden_catching=92.0,
            unit_continuity=0.8,
            active_probability=1.0,
            effectiveness_if_active=1.0,
        ),
    )
    assert identity.explosive > 1.0
    assert identity.efficiency > 1.0
    assert 0.35 <= identity.efficiency <= 1.10
    assert trace.evidence_fields >= 8
