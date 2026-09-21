from __future__ import annotations

import pytest

from monster.reality import failure_paths_v722 as failure
from monster.sim.football_state import FootballState
from monster.sim.play_kernel import (
    PassResult,
    PlayerIdentity,
    PlayEvent,
    PlayType,
    TeamIdentity,
)


def _offense() -> TeamIdentity:
    qb = PlayerIdentity(
        "qb1",
        "QB1",
        "QB",
        efficiency=1.0,
        turnover_security=1.0,
    )
    rb = PlayerIdentity("rb1", "RB1", "RB", usage_weight=0.7)
    wr = PlayerIdentity("wr1", "WR1", "WR", usage_weight=0.8)
    return TeamIdentity("A", qb, (rb, qb), (wr, rb))


def _state(yardline: float) -> FootballState:
    return FootballState(
        possession="A",
        defense="B",
        quarter=3,
        seconds_remaining=1500,
        yardline_100=yardline,
        down=1,
        distance=10.0,
        away_score=10,
        home_score=20,
        away_team_id="A",
        home_team_id="B",
    )


def test_failure_world_is_deterministic_for_seed() -> None:
    first = failure.begin_failure_path_world_v722(
        seed=722001,
        game="A@B",
        team_ids=("A", "B"),
    )
    snapshot = (
        first.game_drag,
        first.teams["A"].collapse_mode,
        first.teams["A"].collapse_strength,
        first.teams["A"].finishing_friction,
    )
    failure.reset_failure_paths_v722()
    second = failure.begin_failure_path_world_v722(
        seed=722001,
        game="A@B",
        team_ids=("A", "B"),
    )
    assert snapshot == (
        second.game_drag,
        second.teams["A"].collapse_mode,
        second.teams["A"].collapse_strength,
        second.teams["A"].finishing_friction,
    )


def test_finishing_friction_only_bites_after_scoring_territory() -> None:
    failure.reset_failure_paths_v722()
    failure.begin_failure_path_world_v722(
        seed=722002,
        game="A@B",
        team_ids=("A", "B"),
    )
    team = failure.team_failure_state_v722("A")
    assert team is not None
    team.collapse_strength = 0.0
    team.finishing_friction = 0.24
    world = failure.current_failure_world_v722()
    assert world is not None
    world.game_drag = 0.0

    offense = _offense()
    midfield = failure._apply_failure_identity_v722(
        offense,
        _state(45.0),
    )
    red_zone = failure._apply_failure_identity_v722(
        offense,
        _state(88.0),
    )

    assert midfield.pass_efficiency == pytest.approx(
        offense.pass_efficiency
    )
    assert red_zone.pass_efficiency < midfield.pass_efficiency
    assert red_zone.field_goal_skill < midfield.field_goal_skill
    assert (
        red_zone.quarterback.turnover_security
        < midfield.quarterback.turnover_security
    )


def test_collapse_state_depresses_offense_without_direct_score_edit() -> None:
    failure.reset_failure_paths_v722()
    failure.begin_failure_path_world_v722(
        seed=722003,
        game="A@B",
        team_ids=("A", "B"),
    )
    team = failure.team_failure_state_v722("A")
    assert team is not None
    team.collapse_strength = 0.24
    team.finishing_friction = 0.0
    world = failure.current_failure_world_v722()
    assert world is not None
    world.game_drag = 0.0

    offense = _offense()
    adjusted = failure._apply_failure_identity_v722(
        offense,
        _state(40.0),
    )
    assert adjusted.pass_efficiency < offense.pass_efficiency
    assert adjusted.rush_efficiency < offense.rush_efficiency
    assert adjusted.pass_protection < offense.pass_protection
    assert (
        adjusted.quarterback.turnover_security
        < offense.quarterback.turnover_security
    )


def test_severe_second_half_qb_failure_can_trigger_backup(monkeypatch) -> None:
    from monster.reality import game_script_v721

    failure.reset_failure_paths_v722()
    failure.begin_failure_path_world_v722(
        seed=722004,
        game="A@B",
        team_ids=("A", "B"),
    )
    team = failure.team_failure_state_v722("A")
    assert team is not None
    team.qb_dropbacks = 13
    team.qb_failures = 8
    team.qb_turnovers = 1

    backup = PlayerIdentity("qb2", "QB2", "QB", usage_weight=0.01)
    monkeypatch.setattr(
        game_script_v721,
        "_backup_qb",
        lambda team_id, starter_id: backup,
    )
    state = FootballState(
        possession="A",
        defense="B",
        quarter=3,
        seconds_remaining=1200,
        yardline_100=50.0,
        down=2,
        distance=9.0,
        away_score=3,
        home_score=24,
        away_team_id="A",
        home_team_id="B",
    )
    offense = _offense()
    event = PlayEvent(
        play_type=PlayType.PASS,
        elapsed_seconds=6,
        passer_id="qb1",
        pass_result=PassResult.INTERCEPTION,
        turnover=True,
    )

    failure._update_qb_performance_v722(
        state=state,
        offense=offense,
        event=event,
    )

    assert team.benched_qb_id == "qb1"
    assert team.replacement_qb_id == "qb2"
    assert "turnovers=2" in team.bench_reason
