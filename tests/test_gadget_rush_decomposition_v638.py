from __future__ import annotations

import numpy as np
import polars as pl

from monster.sim import current_role_guard_v635 as v635
from monster.sim import current_role_guard_v636 as v636
from monster.sim import reality_v62
from monster.sim.current_role_guard_v638 import (
    _entry_probability_v638,
    install_current_role_guard_v638,
    sample_rush_share_plan_v638,
)
from monster.sim.gadget_rush_priors_v638 import conditional_carry_mean
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
                "conditional_offense_snap_share": 0.85,
                "game_day_active_probability": 1.0,
                "status": "ACT",
            },
        ]
    )


def _pool(
    *,
    wr_probability: float = 0.0,
    te_probability: float = 0.0,
) -> TeamPlayerPool:
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


def _configure(*, wr_depth: int = 1, wr_snap: float = 1.0) -> None:
    v635.configure_current_skill_roles_v635(
        _personnel(wr_depth=wr_depth, wr_snap=wr_snap)
    )


def test_v638_empirical_conditional_carry_means_are_frozen() -> None:
    assert abs(conditional_carry_mean("WR") - 1.3609604957397365) < 1e-12
    assert abs(conditional_carry_mean("TE") - 2.32) < 1e-12


def test_v638_zero_role_probability_means_no_gadget_work() -> None:
    _configure()
    pool = _pool(wr_probability=0.0, te_probability=0.0)
    for seed in range(80):
        plan = sample_rush_share_plan_v638(
            pool,
            rng=np.random.default_rng(seed),
        )
        assert plan.get("wr", 0.0) == 0.0
        assert plan.get("te", 0.0) == 0.0
        assert abs(sum(plan.values()) - 1.0) < 1e-9


def test_v638_entry_probability_separates_rotation_from_rush_role() -> None:
    pool = _pool(wr_probability=0.60)
    wr = next(player for player in pool.players if player.player_id == "wr")

    _configure(wr_depth=1, wr_snap=1.0)
    high = _entry_probability_v638(wr)

    _configure(wr_depth=4, wr_snap=0.0)
    low = _entry_probability_v638(wr)

    assert abs(high - 0.60) < 1e-12
    assert 0.0 < low < high


def test_v638_preserves_qb_mass_and_rb_internal_ratio() -> None:
    _configure()
    pool = _pool(wr_probability=1.0, te_probability=1.0)

    for seed in range(100):
        control = v636.sample_rush_share_plan_v636(
            pool,
            rng=np.random.default_rng(seed),
        )
        challenger = sample_rush_share_plan_v638(
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


def test_v638_full_exposure_gadget_share_is_discrete_carry_mass() -> None:
    _configure()
    pool = _pool(wr_probability=1.0, te_probability=0.0)
    expected_runs = round(62.0 * (1.0 - pool.neutral_pass_rate))

    observed = set()
    for seed in range(120):
        plan = sample_rush_share_plan_v638(
            pool,
            rng=np.random.default_rng(seed),
        )
        if "wr" in plan:
            implied = plan["wr"] * expected_runs
            observed.add(round(implied, 8))

    assert observed
    assert min(observed) >= 1.0 - 1e-8
    assert max(observed) <= 8.0 + 1e-8


def test_v638_preserves_v636_downstream_target_random_world() -> None:
    _configure()
    pool = _pool(wr_probability=0.35, te_probability=0.10)

    for seed in range(50):
        control_rng = np.random.default_rng(seed)
        challenger_rng = np.random.default_rng(seed)

        v636.sample_rush_share_plan_v636(pool, rng=control_rng)
        control_targets = v635.sample_target_share_plan_v635(
            pool,
            rng=control_rng,
        )
        control_next = float(control_rng.random())

        sample_rush_share_plan_v638(pool, rng=challenger_rng)
        challenger_targets = v635.sample_target_share_plan_v635(
            pool,
            rng=challenger_rng,
        )
        challenger_next = float(challenger_rng.random())

        assert challenger_targets == control_targets
        assert challenger_next == control_next


def test_v638_installer_keeps_v635_targets_and_installs_new_rush_sampler() -> None:
    install_current_role_guard_v638(_personnel())
    assert reality_v62.sample_target_share_plan is v635.sample_target_share_plan_v635
    assert reality_v62.sample_event_rush_share_plan is sample_rush_share_plan_v638
