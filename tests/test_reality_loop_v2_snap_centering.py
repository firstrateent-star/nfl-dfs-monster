from __future__ import annotations

from monster.sim.matchup_kernel import DefensiveIdentity, DefensiveUnit
from monster.sim.play_kernel import PlayerIdentity
from monster.sim.snap_ecology import resolve_run_snap
from monster.sim.snap_ecology_v2 import (
    resolve_coverage_assignment_v2,
    resolve_run_snap_v2,
)


def test_wr_coverage_assignment_does_not_assign_linebacker_just_for_best_rating() -> None:
    target = PlayerIdentity("wr", "WR", "WR", efficiency=1.0, explosive=1.0)
    defenders = (
        DefensiveIdentity(
            "lb",
            "Elite LB",
            "LB",
            coverage=1.35,
            snap_weight=0.98,
        ),
        DefensiveIdentity(
            "cb",
            "Starting CB",
            "CB",
            coverage=1.02,
            snap_weight=0.94,
        ),
        DefensiveIdentity(
            "s",
            "Safety",
            "S",
            coverage=1.00,
            snap_weight=0.92,
        ),
    )

    assignment = resolve_coverage_assignment_v2(target=target, defenders=defenders)

    assert assignment.defender_id == "cb"
    assert assignment.local_coverage < 1.10


def test_run_front_uses_participation_weighted_fit_not_best_four_every_snap() -> None:
    runner = PlayerIdentity("rb", "RB", "RB", efficiency=1.0, explosive=1.0)
    defense = DefensiveUnit(
        front=(
            DefensiveIdentity("star", "Star", "EDGE", run_defense=1.35, snap_weight=0.08),
            DefensiveIdentity("d1", "D1", "DT", run_defense=1.02, snap_weight=0.90),
            DefensiveIdentity("d2", "D2", "DT", run_defense=1.00, snap_weight=0.88),
            DefensiveIdentity("d3", "D3", "EDGE", run_defense=0.98, snap_weight=0.85),
            DefensiveIdentity("rot", "Rot", "LB", run_defense=0.92, snap_weight=0.45),
        ),
        coverage=(),
        run_stuff_rate=0.18,
    )

    old = resolve_run_snap(rusher=runner, defense=defense, run_blocking=1.0)
    new = resolve_run_snap_v2(rusher=runner, defense=defense, run_blocking=1.0)

    assert new.front_fit < old.front_fit
    assert new.stuff_probability < old.stuff_probability
    assert 0.95 < new.front_fit < 1.08
