from __future__ import annotations

import numpy as np

from monster.sim.rushing_roles import sample_event_rush_share_plan
from monster.snapshot.player import PlayerState, TeamPlayerPool


def _pool() -> TeamPlayerPool:
    return TeamPlayerPool(
        team_id="TST",
        neutral_pass_rate=0.56,
        players=(
            PlayerState(
                "qb",
                "QB",
                "QB",
                "TST",
                rush_share=0.12,
                rush_role_probability=0.99,
                role_uncertainty=0.05,
            ),
            PlayerState(
                "rb1",
                "RB1",
                "RB",
                "TST",
                rush_share=0.46,
                rush_role_probability=0.995,
                role_uncertainty=0.05,
            ),
            PlayerState(
                "rb2",
                "RB2",
                "RB",
                "TST",
                rush_share=0.24,
                rush_role_probability=0.90,
                role_uncertainty=0.08,
            ),
            PlayerState(
                "rb3",
                "RB3",
                "RB",
                "TST",
                rush_share=0.10,
                rush_role_probability=0.35,
                role_uncertainty=0.20,
            ),
            PlayerState(
                "wr",
                "WR",
                "WR",
                "TST",
                rush_share=0.08,
                rush_role_probability=0.10,
                role_uncertainty=0.15,
            ),
        ),
    )


def test_event_role_plan_is_reproducible_and_conserves_share() -> None:
    a = sample_event_rush_share_plan(_pool(), rng=np.random.default_rng(101))
    b = sample_event_rush_share_plan(_pool(), rng=np.random.default_rng(101))
    assert a == b
    assert abs(sum(a.values()) - 1.0) < 1e-12


def test_event_role_plan_preserves_qb_designed_run_reservoir() -> None:
    pool = _pool()
    raw_qb_share = next(player.rush_share for player in pool.players if player.player_id == "qb")
    plan = sample_event_rush_share_plan(pool, rng=np.random.default_rng(102))
    assert abs(plan["qb"] - raw_qb_share) < 1e-12


def test_event_role_plan_concentrates_non_qb_work_into_stable_core() -> None:
    plan = sample_event_rush_share_plan(_pool(), rng=np.random.default_rng(103))
    non_qb = {player_id: share for player_id, share in plan.items() if player_id != "qb"}
    ordered = sorted(non_qb.values(), reverse=True)
    # At least 92% of the non-QB reservoir belongs to no more than three sampled core roles.
    assert sum(ordered[:3]) >= 0.92 * sum(ordered) - 1e-12
    assert max(non_qb, key=non_qb.get) in {"rb1", "rb2"}


def test_unavailable_non_qb_cannot_enter_event_role_plan() -> None:
    pool = _pool()
    players = tuple(
        PlayerState(
            **{
                **player.__dict__,
                "active_probability": 0.0 if player.player_id == "rb3" else player.active_probability,
            }
        )
        for player in pool.players
    )
    unavailable_pool = TeamPlayerPool(
        team_id=pool.team_id,
        players=players,
        neutral_pass_rate=pool.neutral_pass_rate,
    )
    plan = sample_event_rush_share_plan(unavailable_pool, rng=np.random.default_rng(104))
    assert "rb3" not in plan
