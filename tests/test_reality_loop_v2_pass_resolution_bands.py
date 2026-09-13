from __future__ import annotations

from dataclasses import replace

import numpy as np

from monster.sim.intent_ecology import PassDepthOutcome
from monster.sim.pass_resolution_bands_v2 import sample_yac_v2
from monster.sim.resolution_ecology import sample_yac


def _profile(category: str, *, air_mean: float) -> PassDepthOutcome:
    return PassDepthOutcome(
        category=category,
        attempts=6000,
        completion_rate=0.74,
        interception_rate=0.013,
        touchdown_rate=0.04,
        air_yards_mean=air_mean,
        air_yards_sd=2.0,
        yac_mean_completed=5.2,
        yac_sd_completed=6.1,
        negative_completion_rate=0.0,
        zero_completion_rate=0.0,
        yards_mean_completed=8.8,
        yards_sd_completed=10.5,
        gain_5plus_completion_rate=0.7068,
        gain_10plus_completion_rate=0.2331,
        gain_15plus_completion_rate=0.105,
        gain_20plus_completion_rate=0.040,
        gain_40plus_completion_rate=0.008,
        yards_40plus_mean_completed=50.0,
    )


def test_short_completion_v2_restores_empirical_gain_bands_and_conserves_yac() -> None:
    profile = _profile("short_0_5", air_mean=3.0)
    rng = np.random.default_rng(2026091221)
    totals = []
    yacs = []
    for _ in range(45_000):
        air = float(rng.uniform(0.0, 5.0))
        yac = sample_yac_v2(
            profile,
            receiver_explosiveness=1.0,
            coverage_strength=1.0,
            rng=rng,
            air_yards=air,
        )
        totals.append(air + yac)
        yacs.append(yac)
    arr = np.asarray(totals)
    assert abs(float(np.mean(arr >= 5.0)) - profile.gain_5plus_completion_rate) < 0.020
    assert abs(float(np.mean(arr >= 10.0)) - profile.gain_10plus_completion_rate) < 0.017
    assert abs(float(np.mean(arr >= 15.0)) - profile.gain_15plus_completion_rate) < 0.014
    assert abs(float(np.mean(arr >= 20.0)) - profile.gain_20plus_completion_rate) < 0.012
    assert abs(float(np.mean(arr >= 40.0)) - profile.gain_40plus_completion_rate) < 0.0045
    assert abs(float(np.mean(yacs)) - profile.yac_mean_completed) < 0.65


def test_behind_los_v2_preserves_signed_negative_completion_branch() -> None:
    profile = PassDepthOutcome(
        category="behind_los",
        attempts=3000,
        completion_rate=0.77,
        interception_rate=0.008,
        touchdown_rate=0.03,
        air_yards_mean=-3.7,
        air_yards_sd=2.0,
        yac_mean_completed=9.9,
        yac_sd_completed=7.4,
        negative_completion_rate=0.143,
        zero_completion_rate=0.018,
        yards_mean_completed=7.3,
        yards_sd_completed=10.1,
        gain_5plus_completion_rate=0.4975,
        gain_10plus_completion_rate=0.1934,
        gain_15plus_completion_rate=0.082,
        gain_20plus_completion_rate=0.037,
        gain_40plus_completion_rate=0.006,
        yards_40plus_mean_completed=48.0,
    )
    rng = np.random.default_rng(2026091222)
    totals = []
    for _ in range(45_000):
        air = -float(rng.uniform(0.1, 7.5))
        yac = sample_yac_v2(
            profile,
            receiver_explosiveness=1.0,
            coverage_strength=1.0,
            rng=rng,
            air_yards=air,
        )
        totals.append(air + yac)
    arr = np.asarray(totals)
    assert abs(float(np.mean(arr < 0.0)) - profile.negative_completion_rate) < 0.016
    assert abs(float(np.mean(arr >= 5.0)) - profile.gain_5plus_completion_rate) < 0.020
    assert abs(float(np.mean(arr >= 10.0)) - profile.gain_10plus_completion_rate) < 0.017
    assert abs(float(np.mean(arr >= 20.0)) - profile.gain_20plus_completion_rate) < 0.012
    assert abs(float(np.mean(arr >= 40.0)) - profile.gain_40plus_completion_rate) < 0.0045


def test_shallow_completion_v2_keeps_receiver_open_field_authority() -> None:
    profile = _profile("short_0_5", air_mean=3.0)

    def sample(explosive: float, coverage: float, seed: int) -> np.ndarray:
        rng = np.random.default_rng(seed)
        values = []
        for _ in range(30_000):
            air = float(rng.uniform(0.0, 5.0))
            yac = sample_yac_v2(
                profile,
                receiver_explosiveness=explosive,
                coverage_strength=coverage,
                rng=rng,
                air_yards=air,
            )
            values.append(air + yac)
        return np.asarray(values)

    constrained = sample(0.86, 1.12, 2026091223)
    open_field = sample(1.18, 0.92, 2026091224)
    assert float(np.mean(open_field >= 10.0)) > float(np.mean(constrained >= 10.0))
    assert float(np.mean(open_field >= 20.0)) > float(np.mean(constrained >= 20.0))
    assert float(np.mean(open_field >= 40.0)) > float(np.mean(constrained >= 40.0))
    assert float(open_field.mean()) > float(constrained.mean())


def test_deeper_completion_uses_empirical_40plus_tail_and_skill_authority() -> None:
    profile = replace(
        _profile("intermediate_10_19", air_mean=14.0),
        gain_20plus_completion_rate=0.18,
        gain_40plus_completion_rate=0.025,
        yards_40plus_mean_completed=51.0,
    )

    def sample(explosive: float, coverage: float, seed: int) -> np.ndarray:
        rng = np.random.default_rng(seed)
        values = []
        for _ in range(30_000):
            air = float(rng.uniform(10.0, 19.0))
            yac = sample_yac_v2(
                profile,
                receiver_explosiveness=explosive,
                coverage_strength=coverage,
                rng=rng,
                air_yards=air,
            )
            values.append(air + yac)
        return np.asarray(values)

    neutral = sample(1.0, 1.0, 2026091225)
    elite = sample(1.22, 0.90, 2026091226)
    constrained = sample(0.82, 1.12, 2026091227)
    assert abs(float(np.mean(neutral >= 40.0)) - profile.gain_40plus_completion_rate) < 0.0045
    assert float(np.mean(elite >= 40.0)) > float(np.mean(neutral >= 40.0))
    assert float(np.mean(neutral >= 40.0)) > float(np.mean(constrained >= 40.0))


def test_deeper_completion_without_40plus_evidence_delegates_to_existing_yac_model() -> None:
    profile = replace(
        _profile("intermediate_10_19", air_mean=14.0),
        gain_40plus_completion_rate=0.0,
        yards_40plus_mean_completed=0.0,
    )
    rng_a = np.random.default_rng(2026091228)
    rng_b = np.random.default_rng(2026091228)
    base = sample_yac(
        profile,
        receiver_explosiveness=1.05,
        coverage_strength=0.98,
        rng=rng_a,
        air_yards=14.0,
    )
    v2 = sample_yac_v2(
        profile,
        receiver_explosiveness=1.05,
        coverage_strength=0.98,
        rng=rng_b,
        air_yards=14.0,
    )
    assert v2 == base
