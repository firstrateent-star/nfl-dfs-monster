from __future__ import annotations

import numpy as np

from monster.sim.play_kernel import (
    PassResult,
    PlayType,
    _event_elapsed_seconds,
    _sample_snap_cadence,
)


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


def test_rebalanced_snap_cadence_has_slow_and_fast_in_bounds_tail() -> None:
    rng = np.random.default_rng(2026091042)
    ordinary = np.asarray(
        [_sample_snap_cadence(hurry=0.0, rng=rng) for _ in range(20000)],
        dtype=float,
    )
    hurry = np.asarray(
        [_sample_snap_cadence(hurry=1.0, rng=rng) for _ in range(20000)],
        dtype=float,
    )

    assert 37.0 < float(ordinary.mean()) < 41.0
    assert float(np.quantile(ordinary, 0.10)) < 27.0
    assert float(np.quantile(ordinary, 0.90)) > 50.0
    assert 15.0 < float(hurry.mean()) < 21.0
    assert float(hurry.mean()) < float(ordinary.mean()) - 15.0
