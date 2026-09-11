from __future__ import annotations

from monster.sim.defensive_attribution import attribute_defensive_box_score
from monster.sim.matchup_kernel import DefensiveIdentity, DefensiveUnit
from monster.sim.play_kernel import PassResult, PlayEvent, PlayType, RunLane


def _defender(player_id: str, position: str, *, rush=1.0, coverage=1.0, run=1.0, tackle=1.0):
    return DefensiveIdentity(
        player_id=player_id,
        name=player_id,
        position=position,
        pass_rush=rush,
        coverage=coverage,
        run_defense=run,
        tackling=tackle,
        ball_hawk=coverage,
        snap_weight=0.8,
    )


def _unit() -> DefensiveUnit:
    return DefensiveUnit(
        front=(
            _defender("edge", "EDGE", rush=1.4),
            _defender("dt", "DT", rush=1.1, run=1.3),
            _defender("lb", "LB", run=1.2, tackle=1.3),
        ),
        coverage=(
            _defender("cb", "CB", coverage=1.3),
            _defender("s", "S", coverage=1.2, tackle=1.2),
            _defender("lb", "LB", coverage=0.9, tackle=1.3),
        ),
    )


def test_defensive_events_conserve_without_single_representative_collapse() -> None:
    plays = []
    for _ in range(12):
        plays.append(
            PlayEvent(
                play_type=PlayType.PASS,
                elapsed_seconds=25,
                passer_id="qb",
                target_id="wr",
                pass_result=PassResult.COMPLETE,
                yards=8.0,
                pressured=True,
            )
        )
    for _ in range(4):
        plays.append(
            PlayEvent(
                play_type=PlayType.PASS,
                elapsed_seconds=25,
                passer_id="qb",
                pass_result=PassResult.SACK,
                yards=-6.0,
                pressured=True,
            )
        )
    for _ in range(14):
        plays.append(
            PlayEvent(
                play_type=PlayType.RUN,
                elapsed_seconds=35,
                rusher_id="rb",
                run_lane=RunLane.INSIDE,
                yards=4.0,
            )
        )
    plays.append(
        PlayEvent(
            play_type=PlayType.PASS,
            elapsed_seconds=8,
            passer_id="qb",
            target_id="wr",
            pass_result=PassResult.INTERCEPTION,
            turnover=True,
        )
    )

    stats = attribute_defensive_box_score(tuple(plays), _unit(), seed=17)

    assert sum(x.pressures for x in stats.values()) == 16
    assert sum(x.sacks for x in stats.values()) == 4
    assert sum(x.interceptions for x in stats.values()) == 1
    assert sum(x.tackles for x in stats.values()) == 26
    assert max(x.pressures for x in stats.values()) < 16
    assert max(x.tackles for x in stats.values()) < 26
    assert stats["cb"].sacks == 0
    assert stats["s"].sacks == 0


def test_defensive_snap_estimates_are_bounded_by_opponent_scrimmage_plays() -> None:
    plays = tuple(
        PlayEvent(
            play_type=PlayType.RUN,
            elapsed_seconds=30,
            rusher_id="rb",
            yards=3.0,
        )
        for _ in range(20)
    )
    stats = attribute_defensive_box_score(plays, _unit(), seed=5)
    assert all(0 <= box.defensive_snaps <= 20 for box in stats.values())
    assert all(box.defensive_snaps > 0 for box in stats.values())
