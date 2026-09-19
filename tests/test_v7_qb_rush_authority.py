from __future__ import annotations

import pytest

from monster.reality.qb_rush_authority import (
    QB_SENTINEL_USAGE,
    apply_qb_family_role_world,
    separate_qb_from_rush_role_plan,
)
from monster.sim.play_kernel import PlayerIdentity, TeamIdentity
from monster.sim.reality_v62 import RoleWorldPlan
from monster.snapshot.player import PlayerState, TeamPlayerPool


def _state(player_id: str, position: str, rush_share: float) -> PlayerState:
    return PlayerState(
        player_id=player_id,
        display_name=player_id,
        team_id="T",
        position=position,
        rush_share=rush_share,
    )


def test_qb_total_rush_share_is_removed_from_designed_workload_simplex() -> None:
    pool = TeamPlayerPool(
        team_id="T",
        neutral_pass_rate=0.55,
        players=(
            _state("qb", "QB", 0.20),
            _state("rb1", "RB", 0.56),
            _state("rb2", "RB", 0.24),
        ),
    )
    plan = RoleWorldPlan(
        {"qb": 0.20, "rb1": 0.56, "rb2": 0.24},
        {"wr": 0.60, "te": 0.40},
    )

    separated = separate_qb_from_rush_role_plan(pool, plan)

    assert "qb" not in separated
    assert separated["rb1"] == pytest.approx(0.70)
    assert separated["rb2"] == pytest.approx(0.30)
    assert separated.target_plan == plan.target_plan


def test_qb_remains_concept_eligible_without_owning_generic_rush_share() -> None:
    qb = PlayerIdentity("qb", "QB", "QB", usage_weight=0.30)
    rb1 = PlayerIdentity("rb1", "RB1", "RB", usage_weight=0.50)
    rb2 = PlayerIdentity("rb2", "RB2", "RB", usage_weight=0.20)
    wr = PlayerIdentity("wr", "WR", "WR", usage_weight=1.0)
    team = TeamIdentity("T", qb, (rb1, rb2, qb), (wr,))
    plan = RoleWorldPlan({"rb1": 0.70, "rb2": 0.30}, {"wr": 1.0})

    live = apply_qb_family_role_world(team, plan)

    assert [player.player_id for player in live.rushers] == ["rb1", "rb2", "qb"]
    assert live.rushers[0].usage_weight == pytest.approx(0.70)
    assert live.rushers[1].usage_weight == pytest.approx(0.30)
    assert live.rushers[2].usage_weight == pytest.approx(QB_SENTINEL_USAGE)
    assert [player.player_id for player in live.receivers] == ["wr"]
    assert live.receivers[0].usage_weight == pytest.approx(1.0)
