from __future__ import annotations

from monster.sim.assignment_matchups import (
    build_shadow_coverage_assignments,
    build_shadow_rush_assignments,
)
from monster.sim.matchup_kernel import DefensiveIdentity
from monster.sim.play_kernel import PlayerIdentity


def test_coverage_topology_addresses_every_receiver_without_authority() -> None:
    receivers = (
        PlayerIdentity("wr1", "WR1", "WR", usage_weight=0.6),
        PlayerIdentity("te1", "TE1", "TE", usage_weight=0.4),
    )
    defenders = (
        DefensiveIdentity("cb1", "CB1", "CB", snap_weight=0.7),
        DefensiveIdentity("s1", "S1", "S", snap_weight=0.3),
    )
    assignments = build_shadow_coverage_assignments(receivers, defenders)
    assert {assignment.receiver_id for assignment in assignments} == {"wr1", "te1"}
    assert abs(sum(assignment.receiver_share for assignment in assignments) - 1.0) < 1e-12
    assert all(assignment.authority == 0.0 for assignment in assignments)


def test_trench_topology_validates_aligned_blocker_inputs() -> None:
    rushers = (DefensiveIdentity("edge", "EDGE", "EDGE", snap_weight=1.0),)
    try:
        build_shadow_rush_assignments(rushers, ("lt", "lg"), (1.0,))
    except ValueError as exc:
        assert "aligned" in str(exc)
    else:
        raise AssertionError("misaligned blocker inputs must fail")


def test_missing_coverage_population_remains_explicit() -> None:
    receiver = PlayerIdentity("wr1", "WR1", "WR", usage_weight=1.0)
    assignments = build_shadow_coverage_assignments((receiver,), ())
    assert assignments[0].defender_id is None
    assert assignments[0].authority == 0.0
