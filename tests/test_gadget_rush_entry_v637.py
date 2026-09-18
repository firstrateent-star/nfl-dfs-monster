from __future__ import annotations

import numpy as np
import polars as pl

from monster.sim import current_role_guard_v635 as v635
from monster.sim import current_role_guard_v636 as v636
from monster.sim import reality_v62
from monster.sim.current_role_guard_v637 import (
    install_current_role_guard_v637,
    sample_rush_share_plan_v637,
)
from monster.snapshot.player import PlayerState, TeamPlayerPool


def _personnel() -> pl.DataFrame:
    return pl.DataFrame(
        [
            {
                "gsis_id": "rb1",
                "position": "RB",
                "depth_rank": 1,
                "conditional_offense_snap_share": 0.70,
                "game_day_active_probability": 1.0,
                "status": "ACT",
            },
            {
                "gsis_id": "rb2",
                "position": "RB",
                "depth_rank": 2,
                "conditional_offense_snap_share": 0.35,
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
                "conditional_offense_snap_share": 0.96,
                "game_day_active_probability": 1.0,
                "status": "ACT",
            },
            {
                "gsis_id": "te",
                "position": "TE",
                "depth_rank": 1,
                "conditional_offense_snap_share": 0.90,
                "game_day_active_probability": 1.0,
                "status": "ACT",
            },
        ]
    )


def _pool(*, wr_probability: float, te_probability: float = 0.0) -> TeamPlayerPool:
    return TeamPlayerPool(
        "A",
        (
            PlayerState(
                "rb1",
                "RB1",
                "RB",
                "A",
                target_share=0.08,
                rush_share=0.52,
                rush_role_probability=1.0,
                role_uncertainty=0.07,
            ),
            PlayerState(
                "rb2",
                "RB2",
                "RB",
                "A",
                target_share=0.05,
                rush_share=0.22,
                rush_role_probability=0.90,
                role_uncertainty=0.11,
            ),
            PlayerState(
                "qb",
                "QB",
                "QB",
                "A",
                rush_share=0.14,
                rush_role_probability=1.0,
            ),
            PlayerState(
                "wr",
                "WR",
                "WR",
                "A",
                target_share=0.58,
                rush_share=0.09,
                rush_role_probability=wr_probability,
                role_uncertainty=0.07,
            ),
            PlayerState(
                "te",
                "TE",
                "TE",
                "A",
                target_share=0.29,
                rush_share=0.03,
                rush_role_probability=te_probability,
                role_uncertainty=0.08,
            ),
        ),
    )


def _configure() -> None:
    v635.configure_current_skill_roles_v635(_personnel())


def test_v637_zero_entry_probability_blocks_receiver_carries_despite_starter_status() -> None:
    _configure()
    pool = _pool(wr_probability=0.0, te_probability=0.0)

    control_mass = []
    for seed in range(120):
        control = v636.sample_rush_share_plan_v636(
            pool, rng=np.random.default_rng(seed)
        )
        challenger = sample_rush_share_plan_v637(
            pool, rng=np.random.default_rng(seed)
        )
        control_mass.append(
            float(control.get("wr", 0.0)) + float(control.get("te", 0.0))
        )
        assert challenger.get("wr", 0.0) == 0.0
        assert challenger.get("te", 0.0) == 0.0
        assert abs(sum(challenger.values()) - 1.0) < 1e-9

    assert float(np.mean(control_mass)) > 0.03


def test_v637_probability_one_is_exact_v636_rush_plan() -> None:
    _configure()
    pool = _pool(wr_probability=1.0, te_probability=1.0)

    for seed in range(80):
        control = v636.sample_rush_share_plan_v636(
            pool, rng=np.random.default_rng(seed)
        )
        challenger = sample_rush_share_plan_v637(
            pool, rng=np.random.default_rng(seed)
        )
        assert challenger.keys() == control.keys()
        for player_id in control:
            assert abs(challenger[player_id] - control[player_id]) < 1e-12


def test_v637_gadget_entry_frequency_tracks_existing_role_probability() -> None:
    _configure()
    pool = _pool(wr_probability=0.25, te_probability=0.0)

    admitted = 0
    worlds = 600
    for seed in range(worlds):
        challenger = sample_rush_share_plan_v637(
            pool, rng=np.random.default_rng(seed)
        )
        admitted += int(challenger.get("wr", 0.0) > 0.0)

    rate = admitted / worlds
    assert 0.19 <= rate <= 0.31


def test_v637_preserves_qb_mass_and_rb_internal_hierarchy() -> None:
    _configure()
    pool = _pool(wr_probability=0.0, te_probability=0.0)

    for seed in range(100):
        control = v636.sample_rush_share_plan_v636(
            pool, rng=np.random.default_rng(seed)
        )
        challenger = sample_rush_share_plan_v637(
            pool, rng=np.random.default_rng(seed)
        )
        assert abs(challenger.get("qb", 0.0) - control.get("qb", 0.0)) < 1e-12

        control_rb1 = float(control.get("rb1", 0.0))
        control_rb2 = float(control.get("rb2", 0.0))
        challenger_rb1 = float(challenger.get("rb1", 0.0))
        challenger_rb2 = float(challenger.get("rb2", 0.0))
        if control_rb1 > 0.0 and control_rb2 > 0.0:
            assert abs(
                challenger_rb1 / challenger_rb2 - control_rb1 / control_rb2
            ) < 1e-10


def test_v637_preserves_v636_downstream_target_random_world() -> None:
    _configure()
    pool = _pool(wr_probability=0.25, te_probability=0.10)

    for seed in range(50):
        control_rng = np.random.default_rng(seed)
        challenger_rng = np.random.default_rng(seed)

        v636.sample_rush_share_plan_v636(pool, rng=control_rng)
        control_targets = v635.sample_target_share_plan_v635(pool, rng=control_rng)
        control_next = float(control_rng.random())

        sample_rush_share_plan_v637(pool, rng=challenger_rng)
        challenger_targets = v635.sample_target_share_plan_v635(
            pool, rng=challenger_rng
        )
        challenger_next = float(challenger_rng.random())

        assert challenger_targets == control_targets
        assert challenger_next == control_next


def test_v637_installer_keeps_v635_targets_and_installs_only_new_rush_sampler() -> None:
    install_current_role_guard_v637(_personnel())
    assert reality_v62.sample_target_share_plan is v635.sample_target_share_plan_v635
    assert reality_v62.sample_event_rush_share_plan is sample_rush_share_plan_v637
