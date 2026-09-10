from __future__ import annotations

from monster.sim.play_kernel import PassResult, PlayType, _event_elapsed_seconds


def test_in_bounds_scrimmage_keeps_snap_to_snap_cadence() -> None:
    assert _event_elapsed_seconds(36, play_type=PlayType.RUN) == 36
    assert (
        _event_elapsed_seconds(
            36,
            play_type=PlayType.PASS,
            pass_result=PassResult.COMPLETE,
        )
        == 36
    )
    assert (
        _event_elapsed_seconds(
            36,
            play_type=PlayType.PASS,
            pass_result=PassResult.SACK,
        )
        == 36
    )


def test_clock_stopping_results_charge_only_live_ball_duration() -> None:
    incomplete = _event_elapsed_seconds(
        36,
        play_type=PlayType.PASS,
        pass_result=PassResult.INCOMPLETE,
    )
    interception = _event_elapsed_seconds(
        36,
        play_type=PlayType.PASS,
        pass_result=PassResult.INTERCEPTION,
        turnover=True,
    )
    touchdown = _event_elapsed_seconds(36, play_type=PlayType.RUN, touchdown=True)
    fumble = _event_elapsed_seconds(36, play_type=PlayType.RUN, turnover=True)

    assert incomplete == interception == touchdown == fumble == 7
    assert incomplete < 36


def test_clock_stop_duration_is_bounded() -> None:
    assert (
        _event_elapsed_seconds(
            12,
            play_type=PlayType.PASS,
            pass_result=PassResult.INCOMPLETE,
        )
        == 3
    )
    assert (
        _event_elapsed_seconds(
            48,
            play_type=PlayType.PASS,
            pass_result=PassResult.INCOMPLETE,
        )
        == 10
    )
