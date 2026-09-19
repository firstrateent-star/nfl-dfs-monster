from __future__ import annotations

import polars as pl
import pytest

from monster.reality import game_script_v721 as script
from monster.reality.participation_authority_v72 import (
    configure_participation_authority_v72,
)
from monster.sim.decision_policy import SituationPolicy
from monster.sim.defensive_intent import CoverageShell, DefensiveIntent, RushPlan
from monster.sim.football_state import FootballState
from monster.sim.play_kernel import (
    PassResult,
    PlayerIdentity,
    PlayEvent,
    PlayType,
    TeamIdentity,
)
from monster.sim.reality_snap_v5 import _EVENT_META, event_metadata


def _offense(team: str = "A") -> TeamIdentity:
    qb = PlayerIdentity("qb1", "QB1", "QB", usage_weight=1.0)
    rb = PlayerIdentity("rb1", "RB1", "RB", usage_weight=0.70)
    wr = PlayerIdentity("wr1", "WR1", "WR", usage_weight=0.50)
    return TeamIdentity(team, qb, (rb,), (wr,))


def _state(
    *,
    quarter: int,
    seconds_remaining: int,
    away_score: int,
    home_score: int,
    possession: str = "A",
) -> FootballState:
    return FootballState(
        possession=possession,
        defense="B" if possession == "A" else "A",
        quarter=quarter,
        seconds_remaining=seconds_remaining,
        yardline_100=50.0,
        down=1,
        distance=10.0,
        away_score=away_score,
        home_score=home_score,
        away_team_id="A",
        home_team_id="B",
    )


def _base_policy(state, offense):
    return SituationPolicy(pass_probability=0.55, hurry_probability=0.05)


def test_v721_first_half_two_minute_raises_clock_urgency_without_changing_pass_call() -> None:
    script.configure_v721_hooks(
        policy=_base_policy,
        scrimmage=lambda *args, **kwargs: None,
        defensive_intent=lambda *args, **kwargs: None,
        team_identity=lambda *args, **kwargs: None,
    )
    state = _state(
        quarter=2,
        seconds_remaining=1860,
        away_score=10,
        home_score=10,
    )
    policy = script.policy_for_state_v721(state, _offense())
    assert policy.pass_probability == pytest.approx(0.55)
    assert policy.hurry_probability >= 0.70


def test_v721_four_minute_lead_drains_clock() -> None:
    script.configure_v721_hooks(
        policy=_base_policy,
        scrimmage=lambda *args, **kwargs: None,
        defensive_intent=lambda *args, **kwargs: None,
        team_identity=lambda *args, **kwargs: None,
    )
    state = _state(
        quarter=4,
        seconds_remaining=180,
        away_score=27,
        home_score=17,
    )
    policy = script.policy_for_state_v721(state, _offense())
    assert policy.hurry_probability <= 0.04


def test_v721_timeout_bank_stops_live_clock_at_most_three_times(monkeypatch) -> None:
    script.reset_v721_state()
    monkeypatch.setattr(script, "_stable_uniform", lambda *parts: 0.0)

    state = _state(
        quarter=4,
        seconds_remaining=45,
        away_score=20,
        home_score=24,
    )
    offense = _offense()
    event = PlayEvent(
        play_type=PlayType.PASS,
        elapsed_seconds=35,
        passer_id="qb1",
        target_id="wr1",
        pass_result=PassResult.COMPLETE,
        yards=8.0,
    )
    _EVENT_META[id(event)] = {"snap_key": "timeout-test"}

    elapsed = []
    chosen = []
    rng = __import__("numpy").random.default_rng(72101)
    for _ in range(4):
        adjusted, team = script._apply_timeout_clock(
            state,
            offense,
            event,
            rng,
        )
        elapsed.append(adjusted.elapsed_seconds)
        chosen.append(team)
        if team is not None:
            assert event_metadata(adjusted)["snap_key"] == "timeout-test"

    assert elapsed[:3] == [6, 6, 6]
    assert chosen[:3] == ["A", "A", "A"]
    assert elapsed[3] == 35
    assert chosen[3] is None
    _EVENT_META.pop(id(event), None)


def test_v721_multiscore_late_defense_earns_more_script_authority() -> None:
    base = DefensiveIntent(
        coverage_shell=CoverageShell.MAN,
        rush_plan=RushPlan.BLITZ,
        box_aggression=0.6,
        authority=0.24,
    )
    script.configure_v721_hooks(
        policy=_base_policy,
        scrimmage=lambda *args, **kwargs: None,
        defensive_intent=lambda *args, **kwargs: base,
        team_identity=lambda *args, **kwargs: None,
    )
    state = _state(
        quarter=4,
        seconds_remaining=420,
        away_score=10,
        home_score=27,
    )
    out = script.defensive_intent_v721(
        "late-defense",
        state=state,
        offense=_offense(),
    )
    assert out.authority >= 0.28
    assert out.authority <= 0.42


def test_v721_terminal_blowout_can_preserve_qb_and_reduce_star_usage() -> None:
    personnel = pl.DataFrame(
        [
            {
                "gsis_id": "qb1",
                "team_id": "A",
                "position": "QB",
                "position_group": "QB",
                "depth_position": "QB",
                "depth_rank": 1,
                "conditional_offense_snap_share": 1.0,
                "conditional_defense_snap_share": 0.0,
                "conditional_special_teams_snap_share": 0.0,
                "game_day_active_probability": 1.0,
                "status": "ACT",
            },
            {
                "gsis_id": "qb2",
                "team_id": "A",
                "position": "QB",
                "position_group": "QB",
                "depth_position": "QB",
                "depth_rank": 2,
                "conditional_offense_snap_share": 0.02,
                "conditional_defense_snap_share": 0.0,
                "conditional_special_teams_snap_share": 0.0,
                "game_day_active_probability": 1.0,
                "status": "ACT",
            },
        ]
    )
    configure_participation_authority_v72(personnel)
    qb1 = PlayerIdentity("qb1", "QB1", "QB", usage_weight=1.0)
    qb2 = PlayerIdentity("qb2", "QB2", "QB", usage_weight=0.01)
    wr1 = PlayerIdentity("wr1", "WR1", "WR", usage_weight=0.60)
    wr2 = PlayerIdentity("wr2", "WR2", "WR", usage_weight=0.25)
    wr3 = PlayerIdentity("wr3", "WR3", "WR", usage_weight=0.15)
    rb1 = PlayerIdentity("rb1", "RB1", "RB", usage_weight=0.75)
    rb2 = PlayerIdentity("rb2", "RB2", "RB", usage_weight=0.25)
    script._QB_DEPTH["A"] = (qb1, qb2)

    offense = TeamIdentity(
        "A",
        qb1,
        (rb1, rb2),
        (wr1, wr2, wr3),
    )
    state = _state(
        quarter=4,
        seconds_remaining=300,
        away_score=38,
        home_score=14,
    )
    adjusted = script.apply_preservation_v721(offense, state)

    assert adjusted.quarterback.player_id == "qb2"
    wr = {player.player_id: player.usage_weight for player in adjusted.receivers}
    rb = {player.player_id: player.usage_weight for player in adjusted.rushers}
    assert wr["wr1"] < wr1.usage_weight
    assert rb["rb1"] < rb1.usage_weight
    assert wr["wr3"] > wr3.usage_weight
    assert rb["rb2"] > rb2.usage_weight
