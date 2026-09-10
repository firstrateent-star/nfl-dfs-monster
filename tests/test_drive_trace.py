from __future__ import annotations

from dataclasses import replace

from monster.sim.drive_trace import DriveTraceRecorder
from monster.sim.football_state import FootballState, PossessionTerminal, apply_scrimmage_yards
from monster.sim.play_kernel import PassResult, PlayEvent, PlayType


def _state(**changes) -> FootballState:
    base = FootballState(
        possession="A",
        defense="B",
        yardline_100=25.0,
        away_team_id="A",
        home_team_id="B",
    )
    return replace(base, **changes)


def test_drive_trace_observes_progress_without_changing_state() -> None:
    start = _state()
    recorder = DriveTraceRecorder(start)
    first = PlayEvent(play_type=PlayType.RUN, elapsed_seconds=25, yards=6.0, rusher_id="rb")
    recorder.observe(start, first)
    after_first = apply_scrimmage_yards(start, first.yards, first.elapsed_seconds)
    second = PlayEvent(
        play_type=PlayType.PASS,
        elapsed_seconds=24,
        yards=18.0,
        passer_id="qb",
        target_id="wr",
        pass_result=PassResult.COMPLETE,
        pressured=True,
    )
    recorder.observe(after_first, second)
    end = apply_scrimmage_yards(after_first, second.yards, second.elapsed_seconds)
    trace = recorder.finish(end, PossessionTerminal.PUNT, points=0)

    assert start.yardline_100 == 25.0
    assert trace.scrimmage_plays == 2
    assert trace.net_scrimmage_yards == 24.0
    assert trace.first_downs == 1
    assert trace.explosive_plays == 1
    assert trace.pressured_dropbacks == 1
    assert trace.points == 0
    assert trace.terminal == PossessionTerminal.PUNT


def test_drive_trace_detects_red_zone_and_goal_to_go() -> None:
    start = _state(yardline_100=75.0, distance=10.0)
    recorder = DriveTraceRecorder(start)
    event = PlayEvent(play_type=PlayType.RUN, elapsed_seconds=20, yards=17.0, rusher_id="rb")
    recorder.observe(start, event)
    goal_state = _state(yardline_100=92.0, down=1, distance=8.0)
    next_event = PlayEvent(play_type=PlayType.RUN, elapsed_seconds=18, yards=3.0, rusher_id="rb")
    recorder.observe(goal_state, next_event)
    end = apply_scrimmage_yards(goal_state, next_event.yards, next_event.elapsed_seconds)
    trace = recorder.finish(end, PossessionTerminal.FIELD_GOAL, points=3)

    assert trace.red_zone_entered is True
    assert trace.goal_to_go_reached is True
    assert trace.end_yardline_100 == 95.0


def test_drive_trace_uses_scoreboard_delta_for_points() -> None:
    start = _state(yardline_100=95.0)
    recorder = DriveTraceRecorder(start)
    td = PlayEvent(
        play_type=PlayType.PASS,
        elapsed_seconds=8,
        yards=5.0,
        passer_id="qb",
        target_id="wr",
        pass_result=PassResult.COMPLETE,
        touchdown=True,
    )
    recorder.observe(start, td)
    scored = replace(start, seconds_remaining=start.seconds_remaining - 8, away_score=7)
    trace = recorder.finish(scored, PossessionTerminal.TOUCHDOWN)

    assert trace.points == 7
    assert trace.end_yardline_100 == 100.0


def test_drive_trace_marks_overtime_from_start_state() -> None:
    start = _state(quarter=5, seconds_remaining=600)
    trace = DriveTraceRecorder(start).finish(start, PossessionTerminal.END_GAME, points=0)
    assert trace.overtime is True
