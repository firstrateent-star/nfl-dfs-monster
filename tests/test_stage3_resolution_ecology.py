from __future__ import annotations

import numpy as np

from monster.sim.intent_ecology import PassDepthOutcome, RunGeometryOutcome, sample_air_yards
from monster.sim.play_anatomy import CatchpointResult, resolve_catchpoint
from monster.sim.resolution_ecology import (
    completed_pass_yards,
    depth_throw_probabilities,
    resolve_run_ecology,
    sample_yac,
)


class _FixedRandom:
    def __init__(self, values: list[float]):
        self._values = iter(values)

    def random(self) -> float:
        return next(self._values)


def _pass_profile(category: str = "deep_20_39") -> PassDepthOutcome:
    return PassDepthOutcome(
        category=category,
        attempts=1000,
        completion_rate=0.60,
        interception_rate=0.02,
        touchdown_rate=0.05,
        air_yards_mean=28.0,
        air_yards_sd=4.0,
        yac_mean_completed=4.0,
        yac_sd_completed=3.0,
        negative_completion_rate=0.0,
        zero_completion_rate=0.0,
    )


def _run_profile() -> RunGeometryOutcome:
    return RunGeometryOutcome(
        category="interior",
        attempts=10000,
        yards_mean=4.4,
        yards_sd=5.1,
        negative_rate=0.12,
        zero_rate=0.07,
        loss_2_plus_rate=0.055,
        loss_5_plus_rate=0.012,
        explosive_10_rate=0.11,
        explosive_15_rate=0.055,
        explosive_20_rate=0.03,
        touchdown_rate=0.025,
        fumble_lost_rate=0.006,
        yards_p10=-1.0,
        yards_p50=4.0,
        yards_p90=10.0,
        yards_p99=24.0,
    )


def test_depth_aware_catchpoint_does_not_apply_depth_twice() -> None:
    depth_aware = resolve_catchpoint(
        catch_skill=1.0,
        coverage_strength=1.0,
        ball_hawk=1.0,
        air_yards=30.0,
        rng=_FixedRandom([0.50]),
        completion_probability=0.60,
        interception_probability=0.01,
        completion_probability_includes_depth=True,
    )
    legacy = resolve_catchpoint(
        catch_skill=1.0,
        coverage_strength=1.0,
        ball_hawk=1.0,
        air_yards=30.0,
        rng=_FixedRandom([0.50, 0.90]),
        completion_probability=0.60,
        interception_probability=0.01,
        completion_probability_includes_depth=False,
    )
    assert depth_aware == CatchpointResult.CATCH
    assert legacy != CatchpointResult.CATCH


def test_completed_pass_yards_preserve_negative_air_yards() -> None:
    assert completed_pass_yards(-4.0, 2.0) == -2.0
    assert completed_pass_yards(8.0, 5.0) == 13.0


def test_air_yards_respect_depth_family_and_field_geometry() -> None:
    rng = np.random.default_rng(20260910)
    behind = _pass_profile("behind_los")
    behind = PassDepthOutcome(
        **{
            **behind.__dict__,
            "air_yards_mean": -3.0,
            "air_yards_sd": 2.0,
        }
    )
    samples = [
        sample_air_yards(
            "behind_los",
            behind,
            yards_to_goal=60.0,
            rng=rng,
        )
        for _ in range(500)
    ]
    assert max(samples) < 0.0
    assert min(samples) >= -12.0

    deep = _pass_profile()
    goal_limited = [
        sample_air_yards(
            "deep_20_39",
            deep,
            yards_to_goal=14.0,
            rng=rng,
        )
        for _ in range(500)
    ]
    assert min(goal_limited) >= 20.0
    assert max(goal_limited) <= 23.0


def test_coarse_matchup_only_perturbs_depth_prior_partially() -> None:
    profile = _pass_profile("short_0_5")
    neutral = depth_throw_probabilities(
        profile,
        matchup_completion_probability=0.64,
        matchup_interception_probability=0.022,
        pressured=False,
    )
    favorable = depth_throw_probabilities(
        profile,
        matchup_completion_probability=0.84,
        matchup_interception_probability=0.012,
        pressured=False,
    )
    raw_relative = 0.84 / 0.64
    modeled_relative = favorable.completion / neutral.completion
    assert 1.0 < modeled_relative < raw_relative
    assert favorable.interception < neutral.interception


def test_behind_los_yac_uses_actual_air_depth_to_match_total_gain_branches() -> None:
    profile = PassDepthOutcome(
        category="behind_los",
        attempts=10000,
        completion_rate=0.77,
        interception_rate=0.008,
        touchdown_rate=0.01,
        air_yards_mean=-3.7,
        air_yards_sd=2.1,
        yac_mean_completed=9.1,
        yac_sd_completed=9.0,
        negative_completion_rate=0.143,
        zero_completion_rate=0.025,
    )
    rng = np.random.default_rng(20260912)
    air_yards = -4.0
    total = np.asarray(
        [
            completed_pass_yards(
                air_yards,
                sample_yac(
                    profile,
                    receiver_explosiveness=1.0,
                    coverage_strength=1.0,
                    air_yards=air_yards,
                    rng=rng,
                ),
            )
            for _ in range(30000)
        ],
        dtype=float,
    )
    assert abs(float((total < 0).mean()) - profile.negative_completion_rate) < 0.012
    assert abs(float(np.isclose(total, 0.0).mean()) - profile.zero_completion_rate) < 0.010


def test_neutral_run_ecology_reproduces_branch_priors_mean_and_tails() -> None:
    profile = _run_profile()
    rng = np.random.default_rng(20260911)
    yards = np.asarray(
        [
            resolve_run_ecology(
                profile,
                matchup_stuff_probability=0.18,
                matchup_yards_multiplier=1.0,
                runner_power=1.0,
                tackling=1.0,
                explosiveness=1.0,
                rng=rng,
            ).total_yards
            for _ in range(30000)
        ],
        dtype=float,
    )

    assert abs(float(yards.mean()) - profile.yards_mean) < 0.15
    assert abs(float((yards < 0).mean()) - profile.negative_rate) < 0.012
    assert abs(float((yards == 0).mean()) - profile.zero_rate) < 0.010
    assert abs(float((yards >= 10).mean()) - profile.explosive_10_rate) < 0.012
    assert abs(float((yards >= 15).mean()) - profile.explosive_15_rate) < 0.010
    assert abs(float((yards >= 20).mean()) - profile.explosive_20_rate) < 0.008
    assert float((yards <= -5).mean()) > 0.005
    assert float((yards >= 20).mean()) > 0.015


def test_other_geometry_stays_empirical_instead_of_inventing_a_lane() -> None:
    # 2025 regular-season unknown/unrecognized run-geometry anatomy. Kneels and spikes
    # are excluded upstream, so this fixture intentionally mirrors the observed class.
    profile = RunGeometryOutcome(
        category="other",
        attempts=4071,
        yards_mean=-0.850095,
        yards_sd=1.4,
        negative_rate=0.781784,
        zero_rate=0.203036,
        loss_2_plus_rate=0.081594,
        loss_5_plus_rate=0.009488,
        explosive_10_rate=0.001898,
        explosive_15_rate=0.001898,
        explosive_20_rate=0.0,
        touchdown_rate=0.0,
        fumble_lost_rate=0.003,
        yards_p10=-1.0,
        yards_p50=-1.0,
        yards_p90=0.0,
        yards_p99=1.0,
    )
    rng = np.random.default_rng(20260913)
    yards = np.asarray(
        [
            resolve_run_ecology(
                profile,
                matchup_stuff_probability=0.35,
                matchup_yards_multiplier=1.30,
                runner_power=1.20,
                tackling=0.80,
                explosiveness=1.20,
                rng=rng,
            ).total_yards
            for _ in range(30000)
        ],
        dtype=float,
    )
    assert abs(float(yards.mean()) - profile.yards_mean) < 0.08
    assert abs(float((yards < 0).mean()) - profile.negative_rate) < 0.012
    assert abs(float((yards == 0).mean()) - profile.zero_rate) < 0.012
    assert float((yards >= 10).mean()) == 0.0
