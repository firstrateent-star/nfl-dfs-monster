from __future__ import annotations

import numpy as np
import pytest

import runtime_v701_composer as v701
from monster.sim.play_kernel import PlayerIdentity, TeamIdentity
from monster.sim.reality_v62 import RoleWorldPlan
from monster.snapshot.player import PlayerState, TeamPlayerPool


def _pool() -> TeamPlayerPool:
    return TeamPlayerPool(
        team_id="T",
        players=(
            PlayerState("qb", "QB", "QB", "T", rush_share=0.20),
            PlayerState("rb1", "RB1", "RB", "T", rush_share=0.56),
            PlayerState("rb2", "RB2", "RB", "T", rush_share=0.24),
        ),
    )


def test_v701_sampler_removes_qb_mass_without_new_randomness(monkeypatch) -> None:
    plan = RoleWorldPlan(
        {"qb": 0.20, "rb1": 0.56, "rb2": 0.24},
        {"wr": 0.65, "te": 0.35},
    )

    def base(pool, *, rng, **kwargs):
        assert pool.team_id == "T"
        return plan

    monkeypatch.setattr(v701, "_BASE_ROLE_SAMPLER", base)
    rng = np.random.default_rng(701)
    before = rng.bit_generator.state
    separated = v701.sample_role_world_v701(_pool(), rng=rng)
    after = rng.bit_generator.state

    assert before == after
    assert "qb" not in separated
    assert separated["rb1"] == pytest.approx(0.70)
    assert separated["rb2"] == pytest.approx(0.30)
    assert separated.target_plan == plan.target_plan


def test_v701_role_applier_preserves_target_world_and_qb_identity() -> None:
    qb = PlayerIdentity("qb", "QB", "QB", usage_weight=0.2)
    rb1 = PlayerIdentity("rb1", "RB1", "RB", usage_weight=0.7)
    rb2 = PlayerIdentity("rb2", "RB2", "RB", usage_weight=0.3)
    wr = PlayerIdentity("wr", "WR", "WR", usage_weight=0.8)
    te = PlayerIdentity("te", "TE", "TE", usage_weight=0.2)
    team = TeamIdentity("T", qb, (rb1, rb2, qb), (wr, te))
    plan = RoleWorldPlan(
        {"rb1": 0.70, "rb2": 0.30},
        {"wr": 0.60, "te": 0.40},
    )

    live = v701.apply_role_world_v701(team, plan)

    assert [player.player_id for player in live.rushers] == ["rb1", "rb2", "qb"]
    assert [player.player_id for player in live.receivers] == ["wr", "te"]
    assert live.receivers[0].usage_weight == pytest.approx(0.60)
    assert live.receivers[1].usage_weight == pytest.approx(0.40)
    assert live.quarterback.player_id == "qb"
