from __future__ import annotations

from dataclasses import replace

from monster.sim.interaction_topology_v4 import effective_participant_count
from monster.sim.interaction_topology_v4b import (
    choose_coverage_participants,
    choose_pass_rushers,
    choose_run_blockers,
    choose_run_participants,
    snap_responsibility_key,
)
from monster.sim.matchup_kernel import DefensiveIdentity
from monster.sim.play_kernel import PlayerIdentity
from monster.sim.snap_ecology import BlockerProfile, ProtectorProfile


def _coverage() -> tuple[DefensiveIdentity, ...]:
    return (
        DefensiveIdentity("cb1", "CB1", "CB", coverage=0.82, snap_weight=0.94),
        DefensiveIdentity("cb2", "CB2", "CB", coverage=1.18, snap_weight=0.88),
        DefensiveIdentity("slot", "Slot", "DB", coverage=1.02, snap_weight=0.74),
        DefensiveIdentity("fs", "FS", "FS", coverage=1.12, snap_weight=0.91),
        DefensiveIdentity("ss", "SS", "SS", coverage=0.96, snap_weight=0.84),
        DefensiveIdentity("lb", "LB", "LB", coverage=0.92, snap_weight=0.79),
    )


def _front() -> tuple[DefensiveIdentity, ...]:
    return (
        DefensiveIdentity(
            "edge1",
            "EDGE1",
            "EDGE",
            pass_rush=0.78,
            run_defense=0.88,
            snap_weight=0.88,
        ),
        DefensiveIdentity(
            "edge2",
            "EDGE2",
            "EDGE",
            pass_rush=1.24,
            run_defense=1.08,
            snap_weight=0.83,
        ),
        DefensiveIdentity(
            "dt1",
            "DT1",
            "DT",
            pass_rush=1.20,
            run_defense=1.24,
            snap_weight=0.77,
        ),
        DefensiveIdentity(
            "dt2",
            "DT2",
            "DT",
            pass_rush=0.84,
            run_defense=0.92,
            snap_weight=0.69,
        ),
        DefensiveIdentity(
            "lb1",
            "LB1",
            "LB",
            pass_rush=1.06,
            run_defense=1.16,
            snap_weight=0.75,
        ),
        DefensiveIdentity(
            "lb2",
            "LB2",
            "OLB",
            pass_rush=0.94,
            run_defense=1.02,
            snap_weight=0.66,
        ),
    )


def _key(second: int) -> str:
    return snap_responsibility_key(
        offense_team_id="OFF",
        defense_team_id="DEF",
        quarter=2,
        seconds_remaining=second,
        down=2,
        distance=7.0,
        yardline_100=43.0,
        offense_score=10,
        defense_score=7,
    )


def test_snap_responsibility_key_is_stable_and_state_sensitive() -> None:
    assert _key(1200) == _key(1200)
    assert _key(1200) != _key(1199)


def test_same_snap_gives_same_coverage_world() -> None:
    target = PlayerIdentity("wr1", "WR1", "WR", usage_weight=0.34, efficiency=1.05)
    first = choose_coverage_participants(
        target=target,
        defenders=_coverage(),
        responsibility_key=_key(1200),
    )
    second = choose_coverage_participants(
        target=target,
        defenders=_coverage(),
        responsibility_key=_key(1200),
    )
    assert first == second


def test_coverage_rotates_across_snaps_without_using_defender_skill() -> None:
    target = PlayerIdentity("wr1", "WR1", "WR", usage_weight=0.34, efficiency=1.05)
    base = _coverage()
    inverted = tuple(
        replace(defender, coverage=1.30 if defender.coverage < 1.0 else 0.72)
        for defender in base
    )

    base_ids = []
    inverted_ids = []
    for second in range(900, 1000):
        key = _key(second)
        base_ids.append(
            choose_coverage_participants(
                target=target,
                defenders=base,
                responsibility_key=key,
            ).primary_defender_id
        )
        inverted_ids.append(
            choose_coverage_participants(
                target=target,
                defenders=inverted,
                responsibility_key=key,
            ).primary_defender_id
        )

    assert base_ids == inverted_ids
    assert len({player_id for player_id in base_ids if player_id}) >= 3
    assert effective_participant_count(base_ids) >= 2.0


def test_pass_rush_rotates_across_snaps_without_using_pass_rush_skill() -> None:
    base = _front()
    inverted = tuple(
        replace(defender, pass_rush=1.30 if defender.pass_rush < 1.0 else 0.70)
        for defender in base
    )

    base_sets = []
    inverted_sets = []
    participants = []
    for second in range(700, 760):
        key = _key(second)
        selected = choose_pass_rushers(base, count=4, responsibility_key=key)
        inverted_selected = choose_pass_rushers(inverted, count=4, responsibility_key=key)
        base_sets.append(tuple(player.player_id for player in selected))
        inverted_sets.append(tuple(player.player_id for player in inverted_selected))
        participants.extend(player.player_id for player in selected)

    assert base_sets == inverted_sets
    assert len(set(base_sets)) > 1
    assert len(set(participants)) >= 5


def test_run_geometry_changes_blocker_jurisdiction() -> None:
    blockers = (
        BlockerProfile("lt", "LT", run_block=1.10, snap_weight=0.99),
        BlockerProfile("lg", "LG", run_block=0.96, snap_weight=0.98),
        BlockerProfile("c", "C", run_block=1.02, snap_weight=1.0),
        BlockerProfile("rg", "RG", run_block=1.08, snap_weight=0.97),
        BlockerProfile("rt", "RT", run_block=0.94, snap_weight=0.99),
    )
    protectors = (
        ProtectorProfile("te", "TE", run_block=1.14, snap_weight=0.82),
        ProtectorProfile("fb", "FB", run_block=1.06, snap_weight=0.31),
        ProtectorProfile("rb", "RB", run_block=0.86, snap_weight=0.61),
    )

    interior = choose_run_blockers(blockers, protectors, run_geometry="interior")
    left_edge = choose_run_blockers(blockers, protectors, run_geometry="left_edge")
    right_edge = choose_run_blockers(blockers, protectors, run_geometry="right_edge")

    assert {player.player_id for player in interior} == {"lg", "c", "rg"}
    assert {player.player_id for player in left_edge} == {"lt", "lg", "te", "fb"}
    assert {player.player_id for player in right_edge} == {"rt", "rg", "te", "fb"}
    assert "rb" not in {player.player_id for player in left_edge}


def test_run_responsibility_rotates_and_never_reads_defender_skill() -> None:
    rusher = PlayerIdentity("rb1", "RB1", "RB", usage_weight=0.60, efficiency=1.05)
    front = _front()
    coverage = _coverage()
    altered_front = tuple(
        replace(defender, run_defense=1.30 if defender.run_defense < 1.0 else 0.72)
        for defender in front
    )
    altered_coverage = tuple(
        replace(defender, tackling=1.30 if defender.tackling < 1.0 else 0.72)
        for defender in coverage
    )

    base_box = []
    altered_box = []
    base_pursuit = []
    altered_pursuit = []
    for second in range(500, 620):
        key = _key(second)
        base = choose_run_participants(
            rusher=rusher,
            front=front,
            coverage=coverage,
            responsibility_key=key,
            run_geometry="left_edge",
        )
        altered = choose_run_participants(
            rusher=rusher,
            front=altered_front,
            coverage=altered_coverage,
            responsibility_key=key,
            run_geometry="left_edge",
        )
        base_box.append(base.box_defender_id)
        altered_box.append(altered.box_defender_id)
        base_pursuit.append(base.pursuit_defender_id)
        altered_pursuit.append(altered.pursuit_defender_id)

    assert base_box == altered_box
    assert base_pursuit == altered_pursuit
    assert len({player_id for player_id in base_box if player_id}) >= 3
    assert len({player_id for player_id in base_pursuit if player_id}) >= 3
    assert effective_participant_count(base_box) >= 2.0
    assert effective_participant_count(base_pursuit) >= 2.0
