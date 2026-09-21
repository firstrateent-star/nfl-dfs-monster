from __future__ import annotations

from dataclasses import replace

from monster.reality import failure_paths_v722 as v722
from monster.reality import failure_paths_v723 as v723
from monster.sim.football_state import FootballState
from monster.sim.play_kernel import (
    PassResult,
    PlayerIdentity,
    PlayEvent,
    PlayType,
    TeamIdentity,
)


def _offense() -> TeamIdentity:
    qb = PlayerIdentity("qb1", "QB1", "QB", efficiency=1.0, turnover_security=1.0)
    rb = PlayerIdentity("rb1", "RB1", "RB", usage_weight=0.7)
    wr = PlayerIdentity("wr1", "WR1", "WR", usage_weight=0.8)
    return TeamIdentity("A", qb, (rb, qb), (wr, rb))


def _state(
    *,
    quarter: int = 4,
    seconds_remaining: int = 600,
    yardline: float = 82.0,
    down: int = 3,
    distance: float = 7.0,
    away_score: int = 10,
    home_score: int = 31,
) -> FootballState:
    return FootballState(
        possession="A",
        defense="B",
        quarter=quarter,
        seconds_remaining=seconds_remaining,
        yardline_100=yardline,
        down=down,
        distance=distance,
        away_score=away_score,
        home_score=home_score,
        away_team_id="A",
        home_team_id="B",
    )


def _world(seed: int = 723001):
    v722.reset_failure_paths_v722()
    v723.reset_v723_state()
    v722.begin_failure_path_world_v722(
        seed=seed,
        game="A@B",
        team_ids=("A", "B"),
    )
    v723.recalibrate_current_failure_world_v723(
        seed=seed,
        game="A@B",
        team_ids=("A", "B"),
    )
    return v722.current_failure_world_v722()


def test_v723_moves_material_mass_into_drag_or_hard_tail() -> None:
    hard = 0
    affected = 0
    samples = 2000
    for seed in range(723000, 723000 + samples):
        world = _world(seed)
        assert world is not None
        mode = world.teams["A"].collapse_mode
        hard += int(mode == "hard")
        affected += int(mode != "none")
    hard_rate = hard / samples
    affected_rate = affected / samples
    assert 0.055 <= hard_rate <= 0.095
    assert 0.20 <= affected_rate <= 0.28


def test_v723_drive_stall_can_nullify_third_down_touchdown(monkeypatch) -> None:
    world = _world()
    assert world is not None
    team = world.teams["A"]
    team.finishing_friction = 0.38
    monkeypatch.setattr(v723, "_stable_uniform", lambda *args: 0.0)

    event = PlayEvent(
        play_type=PlayType.PASS,
        elapsed_seconds=6,
        passer_id="qb1",
        target_id="wr1",
        pass_result=PassResult.COMPLETE,
        yards=18.0,
        touchdown=True,
    )
    adjusted = v723.apply_drive_finishing_v723(
        state=_state(),
        offense=_offense(),
        event=event,
    )
    assert adjusted.touchdown is False
    assert adjusted.pass_result == PassResult.SACK
    assert adjusted.yards < 0.0


def test_v723_fourth_down_run_stall_fails_conversion(monkeypatch) -> None:
    world = _world()
    assert world is not None
    world.teams["A"].finishing_friction = 0.38
    monkeypatch.setattr(v723, "_stable_uniform", lambda *args: 0.0)
    state = _state(down=4, distance=3.0)
    event = PlayEvent(
        play_type=PlayType.RUN,
        elapsed_seconds=7,
        rusher_id="rb1",
        yards=6.0,
    )
    adjusted = v723.apply_drive_finishing_v723(
        state=state,
        offense=_offense(),
        event=event,
    )
    assert adjusted.yards < state.distance
    assert adjusted.touchdown is False
    assert adjusted.stuffed is True


def test_v723_old_two_turnover_q3_bench_case_no_longer_benches(monkeypatch) -> None:
    world = _world()
    assert world is not None
    team = world.teams["A"]
    team.qb_dropbacks = 23
    team.qb_failures = 15
    team.qb_turnovers = 1

    from monster.reality import game_script_v721

    backup = PlayerIdentity("qb2", "QB2", "QB", usage_weight=0.01)
    monkeypatch.setattr(game_script_v721, "_backup_qb", lambda *args: backup)
    monkeypatch.setattr(v723, "_stable_uniform", lambda *args: 0.0)

    event = PlayEvent(
        play_type=PlayType.PASS,
        elapsed_seconds=6,
        passer_id="qb1",
        pass_result=PassResult.INTERCEPTION,
        turnover=True,
    )
    v723.update_qb_performance_v723(
        state=_state(
            quarter=3,
            seconds_remaining=1000,
            away_score=7,
            home_score=28,
        ),
        offense=_offense(),
        event=event,
    )
    assert team.replacement_qb_id == ""


def test_v723_catastrophic_q4_can_still_bench(monkeypatch) -> None:
    world = _world()
    assert world is not None
    team = world.teams["A"]
    team.qb_dropbacks = 27
    team.qb_failures = 18
    team.qb_turnovers = 2

    from monster.reality import game_script_v721

    backup = PlayerIdentity("qb2", "QB2", "QB", usage_weight=0.01)
    monkeypatch.setattr(game_script_v721, "_backup_qb", lambda *args: backup)
    monkeypatch.setattr(v723, "_stable_uniform", lambda *args: 0.0)

    event = PlayEvent(
        play_type=PlayType.PASS,
        elapsed_seconds=6,
        passer_id="qb1",
        pass_result=PassResult.INTERCEPTION,
        turnover=True,
    )
    v723.update_qb_performance_v723(
        state=_state(
            quarter=4,
            seconds_remaining=700,
            away_score=7,
            home_score=31,
        ),
        offense=_offense(),
        event=event,
    )
    assert team.benched_qb_id == "qb1"
    assert team.replacement_qb_id == "qb2"


def test_v723_non_scoring_territory_preserves_event() -> None:
    _world()
    event = PlayEvent(
        play_type=PlayType.RUN,
        elapsed_seconds=7,
        rusher_id="rb1",
        yards=12.0,
        touchdown=False,
    )
    state = replace(_state(), yardline_100=50.0)
    adjusted = v723.apply_drive_finishing_v723(
        state=state,
        offense=_offense(),
        event=event,
    )
    assert adjusted == event


def test_v723_failed_qb_scramble_preserves_rush_ownership(monkeypatch) -> None:
    world = _world()
    assert world is not None
    world.teams["A"].finishing_friction = 0.38
    monkeypatch.setattr(v723, "_stable_uniform", lambda *args: 0.0)
    state = _state(down=3, distance=5.0)
    event = PlayEvent(
        play_type=PlayType.PASS,
        elapsed_seconds=7,
        passer_id="qb1",
        rusher_id="qb1",
        pass_result=PassResult.SCRAMBLE,
        yards=9.0,
        touchdown=False,
    )
    adjusted = v723.apply_drive_finishing_v723(
        state=state,
        offense=_offense(),
        event=event,
    )
    assert adjusted.pass_result == PassResult.SCRAMBLE
    assert adjusted.rusher_id == "qb1"
    assert adjusted.target_id is None
    assert adjusted.yards < state.distance
    assert adjusted.touchdown is False
