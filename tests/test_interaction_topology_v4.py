from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

from monster.sim.interaction_topology_v4 import (
    choose_coverage_participants,
    choose_pass_rushers,
    choose_run_participants,
    effective_participant_count,
    pair_pass_rushers_to_blockers,
)
from monster.sim.matchup_kernel import DefensiveIdentity, DefensiveUnit, resolve_pass_matchup
from monster.sim.play_kernel import PlayerIdentity
from monster.sim.snap_ecology import BlockerProfile, resolve_coverage_assignment, resolve_run_snap


def _coverage_unit(*, safety_coverage: float = 1.0) -> tuple[DefensiveIdentity, ...]:
    return (
        DefensiveIdentity(
            "cb1",
            "CB1",
            "CB",
            coverage=0.92,
            tackling=0.94,
            snap_weight=0.92,
        ),
        DefensiveIdentity(
            "cb2",
            "CB2",
            "CB",
            coverage=1.18,
            tackling=1.02,
            snap_weight=0.86,
        ),
        DefensiveIdentity(
            "slot",
            "Slot",
            "DB",
            coverage=1.02,
            tackling=0.98,
            snap_weight=0.68,
        ),
        DefensiveIdentity(
            "s1",
            "Safety",
            "S",
            coverage=safety_coverage,
            tackling=safety_coverage,
            snap_weight=0.88,
        ),
        DefensiveIdentity(
            "lb1",
            "LB1",
            "LB",
            coverage=0.96,
            tackling=1.08,
            snap_weight=0.78,
        ),
    )


def _front() -> tuple[DefensiveIdentity, ...]:
    return (
        DefensiveIdentity(
            "edge1",
            "EDGE1",
            "EDGE",
            pass_rush=0.84,
            run_defense=0.94,
            tackling=0.98,
            snap_weight=0.86,
        ),
        DefensiveIdentity(
            "edge2",
            "EDGE2",
            "EDGE",
            pass_rush=1.24,
            run_defense=1.08,
            tackling=1.02,
            snap_weight=0.82,
        ),
        DefensiveIdentity(
            "dt1",
            "DT1",
            "DT",
            pass_rush=1.16,
            run_defense=1.20,
            tackling=1.08,
            snap_weight=0.74,
        ),
        DefensiveIdentity(
            "dt2",
            "DT2",
            "DT",
            pass_rush=0.90,
            run_defense=0.90,
            tackling=0.94,
            snap_weight=0.66,
        ),
        DefensiveIdentity(
            "lb2",
            "LB2",
            "LB",
            pass_rush=1.10,
            run_defense=1.10,
            tackling=1.12,
            snap_weight=0.72,
        ),
    )


def test_coverage_responsibility_does_not_use_coverage_skill() -> None:
    target = PlayerIdentity("wr1", "WR1", "WR", efficiency=1.05, explosive=1.08)
    base = _coverage_unit()
    altered = tuple(
        replace(
            defender,
            coverage=1.30 if defender.coverage < 1.0 else 0.72,
            ball_hawk=1.25 if defender.ball_hawk < 1.0 else 0.75,
        )
        for defender in base
    )
    base_participants = choose_coverage_participants(target=target, defenders=base)
    altered_participants = choose_coverage_participants(target=target, defenders=altered)
    assert base_participants == altered_participants


def test_selected_corner_skill_changes_duel_after_selection() -> None:
    target = PlayerIdentity("wr1", "WR1", "WR", efficiency=1.08, explosive=1.08)
    defenders = _coverage_unit()
    selected = choose_coverage_participants(target=target, defenders=defenders)
    assert selected.primary_defender_id is not None

    weak = tuple(
        replace(defender, coverage=0.78)
        if defender.player_id == selected.primary_defender_id
        else defender
        for defender in defenders
    )
    elite = tuple(
        replace(defender, coverage=1.24)
        if defender.player_id == selected.primary_defender_id
        else defender
        for defender in defenders
    )
    weak_assignment = resolve_coverage_assignment(target=target, defenders=weak)
    elite_assignment = resolve_coverage_assignment(target=target, defenders=elite)
    assert weak_assignment.defender_id == elite_assignment.defender_id
    assert elite_assignment.local_coverage > weak_assignment.local_coverage
    assert elite_assignment.separation_edge < weak_assignment.separation_edge


def test_safety_skill_suppresses_same_receiver_without_becoming_primary() -> None:
    target = PlayerIdentity("wr1", "WR1", "WR", efficiency=1.08, explosive=1.12)
    front = _front()
    weak_defense = DefensiveUnit(
        front=front,
        coverage=_coverage_unit(safety_coverage=0.82),
    )
    elite_defense = DefensiveUnit(
        front=front,
        coverage=_coverage_unit(safety_coverage=1.24),
    )
    weak = resolve_pass_matchup(
        target,
        weak_defense,
        pass_protection=1.0,
        quarterback_efficiency=1.0,
    )
    elite = resolve_pass_matchup(
        target,
        elite_defense,
        pass_protection=1.0,
        quarterback_efficiency=1.0,
    )
    assert weak.primary_defender_id == elite.primary_defender_id
    assert elite.safety_help >= weak.safety_help
    assert elite.yards_multiplier <= weak.yards_multiplier


def test_pass_rush_population_does_not_use_pass_rush_skill() -> None:
    base = _front()
    inverted = tuple(
        replace(defender, pass_rush=1.30 if defender.pass_rush < 1.0 else 0.70)
        for defender in base
    )
    assert [player.player_id for player in choose_pass_rushers(base, count=4)] == [
        player.player_id for player in choose_pass_rushers(inverted, count=4)
    ]


def test_trench_pairing_uses_position_not_blocker_or_rusher_skill() -> None:
    rushers = _front()[:4]
    blockers = (
        BlockerProfile("lt", "LT", pass_block=0.72, snap_weight=0.98),
        BlockerProfile("lg", "LG", pass_block=1.25, snap_weight=0.96),
        BlockerProfile("c", "C", pass_block=0.78, snap_weight=0.99),
        BlockerProfile("rg", "RG", pass_block=1.22, snap_weight=0.94),
        BlockerProfile("rt", "RT", pass_block=0.80, snap_weight=0.97),
    )
    pairs = pair_pass_rushers_to_blockers(rushers, blockers)
    assert pairs
    for rusher, blocker in pairs:
        if rusher.position in {"EDGE", "DE", "OLB"}:
            assert blocker.position in {"LT", "RT", "T", "OT"}
        if rusher.position in {"DT", "DL", "NT"}:
            assert blocker.position in {"LG", "RG", "G", "OG", "C"}


def test_run_responsibility_is_skill_independent_but_skill_changes_result() -> None:
    rusher = PlayerIdentity("rb1", "RB1", "RB", efficiency=1.04)
    front = _front()
    coverage = _coverage_unit()
    base_participants = choose_run_participants(rusher=rusher, front=front, coverage=coverage)
    altered_front = tuple(
        replace(defender, run_defense=1.28 if defender.run_defense < 1.0 else 0.74)
        for defender in front
    )
    altered_coverage = tuple(
        replace(defender, tackling=1.26 if defender.tackling < 1.0 else 0.76)
        for defender in coverage
    )
    altered_participants = choose_run_participants(
        rusher=rusher,
        front=altered_front,
        coverage=altered_coverage,
    )
    assert base_participants == altered_participants

    base_snap = resolve_run_snap(
        rusher=rusher,
        defense=DefensiveUnit(front=front, coverage=coverage),
        run_blocking=1.0,
    )
    altered_snap = resolve_run_snap(
        rusher=rusher,
        defense=DefensiveUnit(front=altered_front, coverage=altered_coverage),
        run_blocking=1.0,
    )
    assert base_snap.primary_defender_id == altered_snap.primary_defender_id
    assert base_snap.pursuit_defender_id == altered_snap.pursuit_defender_id
    assert (
        base_snap.front_fit != altered_snap.front_fit
        or base_snap.second_level_fit != altered_snap.second_level_fit
    )


def test_coverage_topology_uses_multiple_effective_defenders() -> None:
    defenders = _coverage_unit()
    targets = (
        PlayerIdentity("wr1", "WR1", "WR", usage_weight=0.35, efficiency=1.0),
        PlayerIdentity("wr2", "WR2", "WR", usage_weight=0.27, efficiency=1.0),
        PlayerIdentity("wr3", "WR3", "WR", usage_weight=0.18, efficiency=1.0),
        PlayerIdentity("wr4", "WR4", "WR", usage_weight=0.10, efficiency=1.0),
        PlayerIdentity("wr5", "WR5", "WR", usage_weight=0.05, efficiency=1.0),
        PlayerIdentity("te1", "TE1", "TE", usage_weight=0.20, efficiency=1.0),
        PlayerIdentity("te2", "TE2", "TE", usage_weight=0.09, efficiency=1.0),
        PlayerIdentity("rb1", "RB1", "RB", usage_weight=0.16, efficiency=1.0),
        PlayerIdentity("rb2", "RB2", "RB", usage_weight=0.07, efficiency=1.0),
    )
    assignments = [
        choose_coverage_participants(target=target, defenders=defenders).primary_defender_id
        for target in targets
    ]
    assert len({player_id for player_id in assignments if player_id}) >= 3
    assert effective_participant_count(assignments) >= 2.0


def test_topology_api_has_no_realized_outcome_inputs() -> None:
    rusher = SimpleNamespace(player_id="rb", position="RB")
    participants = choose_run_participants(
        rusher=rusher,
        front=_front(),
        coverage=_coverage_unit(),
    )
    assert participants.box_defender_id is not None
    assert participants.pursuit_defender_id is not None
