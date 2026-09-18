from __future__ import annotations

import numpy as np
import polars as pl

from monster.sim import current_role_guard_v635 as v635
from monster.sim import current_role_guard_v636 as v636
from monster.sim import reality_v62
from monster.sim.current_role_guard_v639 import (
    _entry_probability_v639,
    install_current_role_guard_v639,
    sample_rush_share_plan_v639,
)
from monster.sim.gadget_rush_entry_priors_v639 import (
    carry_bin,
    empirical_gadget_entry_prior,
)
from monster.snapshot.player import PlayerState, TeamPlayerPool


def _personnel(*, wr_depth: int = 1, wr_snap: float = 1.0) -> pl.DataFrame:
    return pl.DataFrame(
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
                "depth_rank": wr_depth,
                "conditional_offense_snap_share": wr_snap,
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


def _pool(
    *,
    wr_rushes: float = 0.0,
    te_rushes: float = 0.0,
) -> TeamPlayerPool:
    return TeamPlayerPool(
        "A",
        (
            PlayerState(
                "rb1",
                "RB1",
                "RB",
                "A",
                rush_share=0.52,
                rush_role_probability=1.0,
            ),
            PlayerState(
                "rb2",
                "RB2",
                "RB",
                "A",
                rush_share=0.22,
                rush_role_probability=0.90,
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
                target_share=0.60,
                rush_share=0.09,
                rush_role_probability=0.75,
                historical_rushes=wr_rushes,
            ),
            PlayerState(
                "te",
                "TE",
                "TE",
                "A",
                target_share=0.27,
                rush_share=0.03,
                rush_role_probability=0.30,
                historical_rushes=te_rushes,
            ),
        ),
    )


def _configure(*, wr_depth: int = 1, wr_snap: float = 1.0) -> None:
    v635.configure_current_skill_roles_v635(
        _personnel(wr_depth=wr_depth, wr_snap=wr_snap)
    )


def test_v639_empirical_entry_bins_are_frozen_to_oos_recurrence() -> None:
    assert carry_bin(0) == "0"
    assert carry_bin(2) == "1-3"
    assert carry_bin(6) == "4-8"
    assert carry_bin(12) == "9+"

    assert abs(empirical_gadget_entry_prior("WR", 0) - 82 / 2592) < 1e-12
    assert abs(empirical_gadget_entry_prior("WR", 10) - 248 / 610) < 1e-12
    assert abs(empirical_gadget_entry_prior("TE", 0) - 29 / 2489) < 1e-12
    assert abs(empirical_gadget_entry_prior("TE", 10) - 36 / 37) < 1e-12


def test_v639_specialist_tail_is_not_flattened_by_position_average() -> None:
    _configure()
    ordinary = _pool(te_rushes=0.0)
    specialist = _pool(te_rushes=12.0)
    ordinary_te = next(p for p in ordinary.players if p.player_id == "te")
    specialist_te = next(p for p in specialist.players if p.player_id == "te")

    ordinary_p = _entry_probability_v639(ordinary_te)
    specialist_p = _entry_probability_v639(specialist_te)

    assert ordinary_p < 0.02
    assert specialist_p > 0.80
    assert specialist_p > ordinary_p * 40.0


def test_v639_current_rotation_modulates_same_gadget_identity() -> None:
    pool = _pool(wr_rushes=5.0)
    wr = next(p for p in pool.players if p.player_id == "wr")

    _configure(wr_depth=1, wr_snap=1.0)
    high = _entry_probability_v639(wr)

    _configure(wr_depth=4, wr_snap=0.0)
    low = _entry_probability_v639(wr)

    assert high > low > 0.0


def test_v639_preserves_qb_mass_and_rb_internal_ratio() -> None:
    _configure()
    pool = _pool(wr_rushes=10.0, te_rushes=3.0)

    for seed in range(100):
        control = v636.sample_rush_share_plan_v636(
            pool,
            rng=np.random.default_rng(seed),
        )
        challenger = sample_rush_share_plan_v639(
            pool,
            rng=np.random.default_rng(seed),
        )
        assert abs(
            challenger.get("qb", 0.0) - control.get("qb", 0.0)
        ) < 1e-12

        c1 = float(control.get("rb1", 0.0))
        c2 = float(control.get("rb2", 0.0))
        n1 = float(challenger.get("rb1", 0.0))
        n2 = float(challenger.get("rb2", 0.0))
        if c1 > 0.0 and c2 > 0.0 and n1 > 0.0 and n2 > 0.0:
            assert abs(n1 / n2 - c1 / c2) < 1e-10


def test_v639_preserves_v636_downstream_target_random_world() -> None:
    _configure()
    pool = _pool(wr_rushes=5.0, te_rushes=2.0)

    for seed in range(50):
        control_rng = np.random.default_rng(seed)
        challenger_rng = np.random.default_rng(seed)

        v636.sample_rush_share_plan_v636(pool, rng=control_rng)
        control_targets = v635.sample_target_share_plan_v635(
            pool,
            rng=control_rng,
        )
        control_next = float(control_rng.random())

        sample_rush_share_plan_v639(pool, rng=challenger_rng)
        challenger_targets = v635.sample_target_share_plan_v635(
            pool,
            rng=challenger_rng,
        )
        challenger_next = float(challenger_rng.random())

        assert challenger_targets == control_targets
        assert challenger_next == control_next


def test_v639_installer_keeps_target_sampler_and_installs_calibrated_rush() -> None:
    install_current_role_guard_v639(_personnel())
    assert reality_v62.sample_target_share_plan is v635.sample_target_share_plan_v635
    assert reality_v62.sample_event_rush_share_plan is sample_rush_share_plan_v639
