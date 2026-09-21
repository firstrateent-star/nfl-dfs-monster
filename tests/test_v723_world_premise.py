from __future__ import annotations

from monster.reality.world_premise_v723 import (
    materialize_world_defense_v723,
    materialize_world_team_v723,
    premise_rows_v723,
    reset_world_premise_v723,
)
from monster.sim.matchup_kernel import DefensiveIdentity, DefensiveUnit
from monster.sim.play_kernel import PlayerIdentity, TeamIdentity


def _team(team_id: str) -> TeamIdentity:
    qb = PlayerIdentity(f"{team_id}-qb", "QB", "QB", usage_weight=1.0)
    rb = PlayerIdentity(f"{team_id}-rb", "RB", "RB", usage_weight=0.70)
    wr1 = PlayerIdentity(f"{team_id}-wr1", "WR1", "WR", usage_weight=0.60)
    wr2 = PlayerIdentity(f"{team_id}-wr2", "WR2", "WR", usage_weight=0.40)
    return TeamIdentity(
        team_id=team_id,
        quarterback=qb,
        rushers=(rb, qb),
        receivers=(wr1, wr2, rb),
        neutral_pass_rate=0.57,
        pass_efficiency=1.03,
        rush_efficiency=1.01,
        pass_protection=1.02,
        run_blocking=0.99,
        field_goal_skill=1.01,
        punt_skill=1.00,
    )


def _defense(team_id: str) -> DefensiveUnit:
    edge = DefensiveIdentity(
        f"{team_id}-edge",
        "EDGE",
        "EDGE",
        pass_rush=1.08,
        run_defense=1.02,
        tackling=1.03,
    )
    lb = DefensiveIdentity(
        f"{team_id}-lb",
        "LB",
        "LB",
        pass_rush=0.98,
        coverage=1.01,
        run_defense=1.04,
        tackling=1.05,
        ball_hawk=0.99,
    )
    cb = DefensiveIdentity(
        f"{team_id}-cb",
        "CB",
        "CB",
        coverage=1.07,
        tackling=0.98,
        ball_hawk=1.05,
    )
    return DefensiveUnit(front=(edge, lb), coverage=(lb, cb))


def test_world_team_premise_is_deterministic_and_preserves_role_weights() -> None:
    reset_world_premise_v723()
    base = _team("AWY")

    first = materialize_world_team_v723(
        base, seed=723001, game="AWY@HME", world=4
    )
    second = materialize_world_team_v723(
        base, seed=723001, game="AWY@HME", world=4
    )

    assert first == second
    assert first != base
    assert first.neutral_pass_rate == base.neutral_pass_rate
    assert [p.usage_weight for p in first.receivers] == [
        p.usage_weight for p in base.receivers
    ]
    assert [p.usage_weight for p in first.rushers] == [
        p.usage_weight for p in base.rushers
    ]


def test_distinct_worlds_materialize_distinct_execution_premises() -> None:
    reset_world_premise_v723()
    base = _team("AWY")

    first = materialize_world_team_v723(
        base, seed=723101, game="AWY@HME", world=0
    )
    second = materialize_world_team_v723(
        base, seed=723102, game="AWY@HME", world=1
    )

    assert first != second


def test_world_defense_premise_is_bounded_and_deterministic() -> None:
    reset_world_premise_v723()
    base = _defense("HME")

    first = materialize_world_defense_v723(
        base,
        team_id="HME",
        seed=723201,
        game="AWY@HME",
        world=3,
    )
    second = materialize_world_defense_v723(
        base,
        team_id="HME",
        seed=723201,
        game="AWY@HME",
        world=3,
    )

    assert first == second
    assert 0.12 <= first.pressure_rate <= 0.50
    assert 0.08 <= first.run_stuff_rate <= 0.32
    assert all(0.50 <= p.pass_rush <= 1.72 for p in first.front)
    assert all(0.50 <= p.coverage <= 1.72 for p in first.coverage)


def test_premise_telemetry_records_one_row_per_team_world() -> None:
    reset_world_premise_v723()

    materialize_world_team_v723(
        _team("AWY"), seed=723301, game="AWY@HME", world=8
    )
    materialize_world_defense_v723(
        _defense("AWY"),
        team_id="AWY",
        seed=723301,
        game="AWY@HME",
        world=8,
    )
    materialize_world_team_v723(
        _team("HME"), seed=723301, game="AWY@HME", world=8
    )

    rows = premise_rows_v723()
    assert len(rows) == 2
    assert {row["team"] for row in rows} == {"AWY", "HME"}
    assert {row["regime"] for row in rows} <= {
        "collapse",
        "fragile",
        "normal",
        "surge",
    }
    assert all("offensive_cohesion" in row for row in rows)
    assert all("qb_execution" in row for row in rows)
    assert all("coverage_execution" in row for row in rows)
