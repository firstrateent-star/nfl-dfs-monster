from __future__ import annotations

from monster.reality.qb import (
    QBRushFamily,
    classify_qb_rush_event,
    summarize_qb_rush_families,
)
from monster.sim.play_kernel import PassResult, PlayEvent, PlayType


def test_qb_rush_family_separates_designed_and_reactive_runs() -> None:
    qb = "qb"
    events = (
        PlayEvent(
            play_type=PlayType.RUN,
            elapsed_seconds=20,
            yards=1.0,
            rusher_id=qb,
            run_geometry_category="qb_sneak",
        ),
        PlayEvent(
            play_type=PlayType.RUN,
            elapsed_seconds=25,
            yards=8.0,
            rusher_id=qb,
            run_geometry_category="left_edge",
        ),
        PlayEvent(
            play_type=PlayType.PASS,
            elapsed_seconds=12,
            yards=7.0,
            passer_id=qb,
            rusher_id=qb,
            pass_result=PassResult.SCRAMBLE,
            pressured=True,
        ),
        PlayEvent(
            play_type=PlayType.PASS,
            elapsed_seconds=13,
            yards=11.0,
            passer_id=qb,
            rusher_id=qb,
            pass_result=PassResult.SCRAMBLE,
            pressured=False,
        ),
    )

    assert classify_qb_rush_event(events[0], quarterback_id=qb) == QBRushFamily.SNEAK
    assert classify_qb_rush_event(events[1], quarterback_id=qb) == QBRushFamily.DESIGNED_KEEPER
    assert classify_qb_rush_event(events[2], quarterback_id=qb) == QBRushFamily.PRESSURE_SCRAMBLE
    assert classify_qb_rush_event(events[3], quarterback_id=qb) == QBRushFamily.COVERAGE_SCRAMBLE


def test_qb_rush_summary_conserves_family_attempts_and_yards() -> None:
    qb = "qb"
    events = [
        PlayEvent(
            play_type=PlayType.PASS,
            elapsed_seconds=10,
            yards=6.0,
            passer_id=qb,
            rusher_id=qb,
            pass_result=PassResult.SCRAMBLE,
            pressured=True,
        ),
        PlayEvent(
            play_type=PlayType.PASS,
            elapsed_seconds=10,
            yards=4.0,
            passer_id=qb,
            rusher_id=qb,
            pass_result=PassResult.SCRAMBLE,
            pressured=True,
        ),
        PlayEvent(
            play_type=PlayType.RUN,
            elapsed_seconds=20,
            yards=2.0,
            rusher_id=qb,
            run_geometry_category="qb_sneak",
            touchdown=True,
        ),
    ]
    summary = summarize_qb_rush_families(events, quarterback_id=qb)
    by_family = {row.family: row for row in summary}

    assert by_family[QBRushFamily.PRESSURE_SCRAMBLE].attempts == 2
    assert by_family[QBRushFamily.PRESSURE_SCRAMBLE].yards == 10.0
    assert by_family[QBRushFamily.SNEAK].attempts == 1
    assert by_family[QBRushFamily.SNEAK].touchdowns == 1
    assert sum(row.attempts for row in summary) == 3


def test_non_qb_run_is_not_misclassified() -> None:
    event = PlayEvent(
        play_type=PlayType.RUN,
        elapsed_seconds=20,
        yards=5.0,
        rusher_id="rb",
    )
    assert classify_qb_rush_event(event, quarterback_id="qb") is None
