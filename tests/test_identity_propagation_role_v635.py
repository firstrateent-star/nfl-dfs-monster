from __future__ import annotations

import numpy as np
import polars as pl

from monster.sim import resolution_ecology
from monster.sim.current_role_guard_v635 import (
    configure_current_skill_roles_v635,
    sample_rush_share_plan_v635,
    sample_target_share_plan_v635,
)
from monster.sim.intent_ecology import PassDepthOutcome
from monster.sim.matchup_kernel import DefensiveIdentity, DefensiveUnit
from monster.sim.play_kernel import PlayerIdentity, TeamIdentity, _field_read_target
from monster.sim.rich_identity import RichPlayerIdentity
from monster.sim.rushing_roles import sample_event_rush_share_plan
from monster.snapshot.player import PlayerState, TeamPlayerPool


def _personnel(rows: list[dict[str, object]]) -> pl.DataFrame:
    return pl.DataFrame(rows)


def test_target_participation_truth_does_not_flatten_historical_alpha() -> None:
    pool = TeamPlayerPool(
        "A",
        (
            PlayerState("alpha", "Alpha", "WR", "A", target_share=0.48, role_uncertainty=0.05),
            PlayerState("new", "New", "WR", "A", target_share=0.01, role_uncertainty=0.15),
        ),
    )
    configure_current_skill_roles_v635(
        _personnel(
            [
                {
                    "gsis_id": "alpha",
                    "position": "WR",
                    "depth_rank": 1,
                    "conditional_offense_snap_share": 0.92,
                    "game_day_active_probability": 1.0,
                    "status": "ACT",
                },
                {
                    "gsis_id": "new",
                    "position": "WR",
                    "depth_rank": 1,
                    "conditional_offense_snap_share": 0.72,
                    "game_day_active_probability": 1.0,
                    "status": "ACT",
                },
            ]
        )
    )

    alpha = []
    newcomer = []
    rng = np.random.default_rng(63501)
    for _ in range(250):
        plan = sample_target_share_plan_v635(pool, rng=rng)
        alpha.append(plan["alpha"])
        newcomer.append(plan["new"])

    assert float(np.mean(alpha)) > 0.72
    assert float(np.mean(alpha)) > float(np.mean(newcomer)) * 2.5
    assert min(newcomer) > 0.0


def test_current_rushing_starter_can_enter_despite_zero_old_share() -> None:
    pool = TeamPlayerPool(
        "A",
        (
            PlayerState("old", "Old", "RB", "A", rush_share=0.85, rush_role_probability=1.0),
            PlayerState("starter", "Starter", "RB", "A", rush_share=0.0, rush_role_probability=1.0),
            PlayerState("qb", "QB", "QB", "A", rush_share=0.15, qb_pass_share=1.0),
        ),
    )
    configure_current_skill_roles_v635(
        _personnel(
            [
                {
                    "gsis_id": "old",
                    "position": "RB",
                    "depth_rank": 2,
                    "conditional_offense_snap_share": 0.35,
                    "game_day_active_probability": 1.0,
                    "status": "ACT",
                },
                {
                    "gsis_id": "starter",
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
            ]
        )
    )
    plan = sample_rush_share_plan_v635(pool, rng=np.random.default_rng(77))
    assert plan["starter"] > 0.0
    assert abs(sum(plan.values()) - 1.0) < 1e-9


def test_current_rushing_depth_reweights_conditional_share_without_replacing_prior() -> None:
    pool = TeamPlayerPool(
        "A",
        (
            PlayerState("a", "A", "RB", "A", rush_share=0.70, rush_role_probability=1.0),
            PlayerState("b", "B", "RB", "A", rush_share=0.30, rush_role_probability=1.0),
        ),
    )
    configure_current_skill_roles_v635(
        _personnel(
            [
                {
                    "gsis_id": "a",
                    "position": "RB",
                    "depth_rank": 2,
                    "conditional_offense_snap_share": 0.38,
                    "game_day_active_probability": 1.0,
                    "status": "ACT",
                },
                {
                    "gsis_id": "b",
                    "position": "RB",
                    "depth_rank": 1,
                    "conditional_offense_snap_share": 0.78,
                    "game_day_active_probability": 1.0,
                    "status": "ACT",
                },
            ]
        )
    )

    baseline_b = []
    current_b = []
    for seed in range(120):
        baseline_b.append(
            sample_event_rush_share_plan(pool, rng=np.random.default_rng(seed)).get("b", 0.0)
        )
        current_b.append(
            sample_rush_share_plan_v635(pool, rng=np.random.default_rng(seed)).get("b", 0.0)
        )
    assert float(np.mean(current_b)) > float(np.mean(baseline_b)) + 0.04
    # History is still meaningful: the 70% historical back is not automatically erased.
    assert float(np.mean(current_b)) < 0.62


def _defense() -> DefensiveUnit:
    return DefensiveUnit(
        front=(
            DefensiveIdentity(
                "edge", "Edge", "EDGE", pass_rush=1.0, run_defense=1.0, tackling=1.0
            ),
        ),
        coverage=(
            DefensiveIdentity(
                "cb", "CB", "CB", coverage=1.0, ball_hawk=1.0, tackling=1.0
            ),
        ),
    )


def test_rich_qb_execution_reaches_live_field_read() -> None:
    receiver = PlayerIdentity("wr", "WR", "WR", usage_weight=1.0)
    low_qb = RichPlayerIdentity(
        "qb-low",
        "Low",
        "QB",
        efficiency=1.0,
        explosive=1.0,
        qb_execution_skill=-1.0,
        mobility_skill=0.0,
    )
    high_qb = RichPlayerIdentity(
        "qb-high",
        "High",
        "QB",
        efficiency=1.0,
        explosive=1.0,
        qb_execution_skill=1.0,
        mobility_skill=0.0,
    )
    low_team = TeamIdentity("A", low_qb, (low_qb,), (receiver,), pass_protection=1.0)
    high_team = TeamIdentity("A", high_qb, (high_qb,), (receiver,), pass_protection=1.0)
    fatigue: dict[str, float] = {}

    _, low = _field_read_target(
        low_team,
        _defense(),
        np.random.default_rng(1),
        fatigue=fatigue,
        responsibility_key="v635-qb",
    )
    _, high = _field_read_target(
        high_team,
        _defense(),
        np.random.default_rng(1),
        fatigue=fatigue,
        responsibility_key="v635-qb",
    )
    assert high.qb_read_quality > low.qb_read_quality + 0.20
    assert high.completion_probability > low.completion_probability


def test_live_snap_authority_is_behavioral_and_coarse_knob_is_not() -> None:
    profile = PassDepthOutcome(
        category="short_0_5",
        attempts=1000,
        completion_rate=0.70,
        interception_rate=0.02,
        touchdown_rate=0.03,
        air_yards_mean=3.0,
        air_yards_sd=2.0,
        yac_mean_completed=5.0,
        yac_sd_completed=3.0,
        negative_completion_rate=0.05,
        zero_completion_rate=0.03,
    )
    native_live = resolution_ecology._SNAP_MATCHUP_AUTHORITY
    had_coarse = hasattr(resolution_ecology, "_COARSE_MATCHUP_AUTHORITY")
    coarse = getattr(resolution_ecology, "_COARSE_MATCHUP_AUTHORITY", None)
    try:
        resolution_ecology._SNAP_MATCHUP_AUTHORITY = 0.78
        baseline = resolution_ecology.depth_throw_probabilities(
            profile,
            matchup_completion_probability=0.82,
            matchup_interception_probability=0.012,
            pressured=False,
        ).completion

        resolution_ecology._COARSE_MATCHUP_AUTHORITY = 0.01
        stale_changed = resolution_ecology.depth_throw_probabilities(
            profile,
            matchup_completion_probability=0.82,
            matchup_interception_probability=0.012,
            pressured=False,
        ).completion
        assert stale_changed == baseline

        resolution_ecology._SNAP_MATCHUP_AUTHORITY = 0.10
        low_live = resolution_ecology.depth_throw_probabilities(
            profile,
            matchup_completion_probability=0.82,
            matchup_interception_probability=0.012,
            pressured=False,
        ).completion
        resolution_ecology._SNAP_MATCHUP_AUTHORITY = 0.95
        high_live = resolution_ecology.depth_throw_probabilities(
            profile,
            matchup_completion_probability=0.82,
            matchup_interception_probability=0.012,
            pressured=False,
        ).completion
        assert high_live > low_live
    finally:
        resolution_ecology._SNAP_MATCHUP_AUTHORITY = native_live
        if had_coarse:
            resolution_ecology._COARSE_MATCHUP_AUTHORITY = coarse
        else:
            delattr(resolution_ecology, "_COARSE_MATCHUP_AUTHORITY")
