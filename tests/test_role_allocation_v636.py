from __future__ import annotations

import numpy as np
import polars as pl

from monster.sim import current_role_guard_v635 as v635
from monster.sim import reality_v62
from monster.sim.current_role_guard_v636 import (
    install_current_role_guard_v636,
    sample_rush_share_plan_v636,
)
from monster.sim.rushing_roles import sample_event_rush_share_plan
from monster.snapshot.player import PlayerState, TeamPlayerPool


def _personnel(rows: list[dict[str, object]]) -> pl.DataFrame:
    return pl.DataFrame(rows)


def _group_mass(plan: dict[str, float], ids: set[str]) -> float:
    return sum(float(plan.get(player_id, 0.0)) for player_id in ids)


def test_v636_preserves_native_qb_and_gadget_mass_exactly() -> None:
    pool = TeamPlayerPool(
        "A",
        (
            PlayerState(
                "rb1",
                "RB1",
                "RB",
                "A",
                rush_share=0.58,
                rush_role_probability=1.0,
                role_uncertainty=0.06,
            ),
            PlayerState(
                "rb2",
                "RB2",
                "RB",
                "A",
                rush_share=0.22,
                rush_role_probability=0.92,
                role_uncertainty=0.10,
            ),
            PlayerState(
                "qb",
                "QB",
                "QB",
                "A",
                rush_share=0.12,
                rush_role_probability=1.0,
            ),
            PlayerState(
                "wr",
                "WR",
                "WR",
                "A",
                rush_share=0.06,
                rush_role_probability=0.55,
            ),
            PlayerState(
                "te",
                "TE",
                "TE",
                "A",
                rush_share=0.02,
                rush_role_probability=0.30,
            ),
        ),
    )
    v635.configure_current_skill_roles_v635(
        _personnel(
            [
                {
                    "gsis_id": "rb1",
                    "position": "RB",
                    "depth_rank": 1,
                    "conditional_offense_snap_share": 0.72,
                    "game_day_active_probability": 1.0,
                    "status": "ACT",
                },
                {
                    "gsis_id": "rb2",
                    "position": "RB",
                    "depth_rank": 2,
                    "conditional_offense_snap_share": 0.34,
                    "game_day_active_probability": 1.0,
                    "status": "ACT",
                },
                {
                    "gsis_id": "qb",
                    "position": "QB",
                    "depth_rank": 1,
                    "conditional_offense_snap_share": 1.0,
                    "game_day_active_probability": 1.0,
                    "status": "ACT",
                },
                {
                    "gsis_id": "wr",
                    "position": "WR",
                    "depth_rank": 1,
                    "conditional_offense_snap_share": 0.94,
                    "game_day_active_probability": 1.0,
                    "status": "ACT",
                },
                {
                    "gsis_id": "te",
                    "position": "TE",
                    "depth_rank": 1,
                    "conditional_offense_snap_share": 0.88,
                    "game_day_active_probability": 1.0,
                    "status": "ACT",
                },
            ]
        )
    )

    for seed in range(80):
        native = sample_event_rush_share_plan(pool, rng=np.random.default_rng(seed))
        challenger = sample_rush_share_plan_v636(pool, rng=np.random.default_rng(seed))
        assert abs(sum(challenger.values()) - 1.0) < 1e-9
        assert abs(
            _group_mass(challenger, {"qb"}) - _group_mass(native, {"qb"})
        ) < 1e-9
        assert abs(
            _group_mass(challenger, {"wr", "te"})
            - _group_mass(native, {"wr", "te"})
        ) < 1e-9


def test_v636_does_not_turn_receiver_starter_status_into_carry_mass() -> None:
    pool = TeamPlayerPool(
        "A",
        (
            PlayerState("rb", "RB", "RB", "A", rush_share=0.80, rush_role_probability=1.0),
            PlayerState("qb", "QB", "QB", "A", rush_share=0.20, rush_role_probability=1.0),
            PlayerState("wr", "WR", "WR", "A", rush_share=0.0, rush_role_probability=0.05),
            PlayerState("te", "TE", "TE", "A", rush_share=0.0, rush_role_probability=0.02),
        ),
    )
    v635.configure_current_skill_roles_v635(
        _personnel(
            [
                {
                    "gsis_id": "rb",
                    "position": "RB",
                    "depth_rank": 1,
                    "conditional_offense_snap_share": 0.75,
                    "game_day_active_probability": 1.0,
                    "status": "ACT",
                },
                {
                    "gsis_id": "qb",
                    "position": "QB",
                    "depth_rank": 1,
                    "conditional_offense_snap_share": 1.0,
                    "game_day_active_probability": 1.0,
                    "status": "ACT",
                },
                {
                    "gsis_id": "wr",
                    "position": "WR",
                    "depth_rank": 1,
                    "conditional_offense_snap_share": 0.95,
                    "game_day_active_probability": 1.0,
                    "status": "ACT",
                },
                {
                    "gsis_id": "te",
                    "position": "TE",
                    "depth_rank": 1,
                    "conditional_offense_snap_share": 0.92,
                    "game_day_active_probability": 1.0,
                    "status": "ACT",
                },
            ]
        )
    )

    old_gadget = []
    new_gadget = []
    for seed in range(100):
        old = v635.sample_rush_share_plan_v635(pool, rng=np.random.default_rng(seed))
        new = sample_rush_share_plan_v636(pool, rng=np.random.default_rng(seed))
        old_gadget.append(_group_mass(old, {"wr", "te"}))
        new_gadget.append(_group_mass(new, {"wr", "te"}))

    assert float(np.mean(old_gadget)) > 0.05
    assert float(np.mean(new_gadget)) == 0.0


def test_v636_current_rb_starter_can_enter_despite_zero_old_share() -> None:
    pool = TeamPlayerPool(
        "A",
        (
            PlayerState("old", "Old", "RB", "A", rush_share=0.80, rush_role_probability=1.0),
            PlayerState(
                "starter",
                "Starter",
                "RB",
                "A",
                rush_share=0.0,
                rush_role_probability=1.0,
            ),
            PlayerState("qb", "QB", "QB", "A", rush_share=0.20, rush_role_probability=1.0),
        ),
    )
    v635.configure_current_skill_roles_v635(
        _personnel(
            [
                {
                    "gsis_id": "old",
                    "position": "RB",
                    "depth_rank": 2,
                    "conditional_offense_snap_share": 0.30,
                    "game_day_active_probability": 1.0,
                    "status": "ACT",
                },
                {
                    "gsis_id": "starter",
                    "position": "RB",
                    "depth_rank": 1,
                    "conditional_offense_snap_share": 0.78,
                    "game_day_active_probability": 1.0,
                    "status": "ACT",
                },
                {
                    "gsis_id": "qb",
                    "position": "QB",
                    "depth_rank": 1,
                    "conditional_offense_snap_share": 1.0,
                    "game_day_active_probability": 1.0,
                    "status": "ACT",
                },
            ]
        )
    )

    plan = sample_rush_share_plan_v636(pool, rng=np.random.default_rng(636))
    assert plan["starter"] > 0.0
    assert abs(sum(plan.values()) - 1.0) < 1e-9


def test_v636_install_keeps_v635_target_sampler() -> None:
    personnel = _personnel(
        [
            {
                "gsis_id": "rb",
                "position": "RB",
                "depth_rank": 1,
                "conditional_offense_snap_share": 0.70,
                "game_day_active_probability": 1.0,
                "status": "ACT",
            }
        ]
    )
    install_current_role_guard_v636(personnel)
    assert reality_v62.sample_target_share_plan is v635.sample_target_share_plan_v635
    assert reality_v62.sample_event_rush_share_plan is sample_rush_share_plan_v636
