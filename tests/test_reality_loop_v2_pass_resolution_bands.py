from __future__ import annotations

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
        yards_mean_completed=7.8,
        yards_sd_completed=8.5,
        gain_5plus_completion_rate=0.7068,
        gain_10plus_completion_rate=0.2331,
        gain_15plus_completion_rate=0.105,
        gain_20plus_completion_rate=0.040,
    )


def test_short_completion_v2_restores_empirical_gain_bands_and_conserves_yac() -> None:
    profile = _profile("short_0_5", air_mean=3.0)
    rng = np.random.default_rng(2026091221)
    totals = []
    yacs = []
    for _ in range(35_000):
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
    assert abs(float(np.mean(arr >= 5.0)) - profile.gain_5plus_completion_rate) < 0.018
    assert abs(float(np.mean(arr >= 10.0)) - profile.gain_10plus_completion_rate) < 0.015
    assert abs(float(np.mean(arr >= 15.0)) - profile.gain_15plus_completion_rate) < 0.012
    assert abs(float(np.mean(arr >= 20.0)) - profile.gain_20plus_completion_rate) < 0.010
    assert abs(float(np.mean(yacs)) - profile.yac_mean_completed) < 0.35


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
        yards_mean_completed=6.2,
        yards_sd_completed=9.1,
        gain_5plus_completion_rate=0.4975,
        gain_10plus_completion_rate=0.1934,
        gain_15plus_completion_rate=0.082,
        gain_20plus_completion_rate=0.037,
    )
    rng = np.random.default_rng(2026091222)
    totals = []
    for _ in range(35_000):
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
    assert abs(float(np.mean(arr < 0.0)) - profile.negative_completion_rate) < 0.015
    assert abs(float(np.mean(arr >= 5.0)) - profile.gain_5plus_completion_rate) < 0.018
    assert abs(float(np.mean(arr >= 10.0)) - profile.gain_10plus_completion_rate) < 0.015
    assert abs(float(np.mean(arr >= 20.0)) - profile.gain_20plus_completion_rate) < 0.010


def test_shallow_completion_v2_keeps_receiver_open_field_authority() -> None:
    profile = _profile("short_0_5", air_mean=3.0)

    def sample(explosive: float, coverage: float, seed: int) -> np.ndarray:
        rng = np.random.default_rng(seed)
        values = []
        for _ in range(15_000):
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
    assert float(open_field.mean()) > float(constrained.mean())


def test_deeper_completions_delegate_to_existing_yac_model() -> None:
    profile = _profile("intermediate_10_19", air_mean=14.0)
    rng_a = np.random.default_rng(2026091225)
    rng_b = np.random.default_rng(2026091225)
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
