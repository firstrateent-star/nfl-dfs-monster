from __future__ import annotations

import numpy as np

from monster.sim.intent_ecology import PassDepthOutcome, RunGeometryOutcome
from monster.sim.progressive_skill_tail_v2 import (
    resolve_run_ecology_progressive_skill_v2,
    resolve_scramble_contact_progressive_skill_v2,
    sample_yac_progressive_skill_v2,
)


def _pass_profile() -> PassDepthOutcome:
    return PassDepthOutcome(
        category="short_0_5",
        attempts=6000,
        completion_rate=0.74,
        interception_rate=0.013,
        touchdown_rate=0.04,
        air_yards_mean=3.0,
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


def _run_profile() -> RunGeometryOutcome:
    return RunGeometryOutcome(
        category="interior",
        attempts=7000,
        yards_mean=3.89,
        yards_sd=5.3,
        negative_rate=0.077012,
        zero_rate=0.027,
        loss_2_plus_rate=0.034,
        loss_5_plus_rate=0.006,
        explosive_10_rate=0.073845,
        explosive_15_rate=0.028933,
        explosive_20_rate=0.014,
        touchdown_rate=0.026,
        fumble_lost_rate=0.007,
        yards_p10=0.0,
        yards_p50=3.0,
        yards_p90=8.0,
        yards_p99=24.0,
        explosive_40_rate=0.0032,
        yards_40plus_mean=47.5,
    )


def _tail_rates(values: np.ndarray) -> tuple[float, float, float]:
    return (
        float(np.mean(values >= 15.0)),
        float(np.mean(values >= 20.0)),
        float(np.mean(values >= 40.0)),
    )


def _relative_lifts(elite: tuple[float, ...], constrained: tuple[float, ...]) -> tuple[float, ...]:
    return tuple(e / max(c, 1e-9) for e, c in zip(elite, constrained, strict=True))


def test_receiver_skill_authority_grows_deeper_into_tail_and_neutral_stays_centered() -> None:
    profile = _pass_profile()

    def sample(explosive: float, coverage: float, seed: int) -> np.ndarray:
        rng = np.random.default_rng(seed)
        totals = []
        for _ in range(55_000):
            air = float(rng.uniform(0.0, 5.0))
            yac = sample_yac_progressive_skill_v2(
                profile,
                receiver_explosiveness=explosive,
                coverage_strength=coverage,
                rng=rng,
                air_yards=air,
            )
            totals.append(air + yac)
        return np.asarray(totals)

    neutral = _tail_rates(sample(1.0, 1.0, 2026091302))
    elite = _tail_rates(sample(1.24, 0.90, 2026091303))
    constrained = _tail_rates(sample(0.82, 1.12, 2026091304))

    assert abs(neutral[0] - profile.gain_15plus_completion_rate) < 0.012
    assert abs(neutral[1] - profile.gain_20plus_completion_rate) < 0.009
    assert abs(neutral[2] - profile.gain_40plus_completion_rate) < 0.0035
    assert all(e > n > c for e, n, c in zip(elite, neutral, constrained, strict=True))
    lifts = _relative_lifts(elite, constrained)
    assert lifts[2] > lifts[1] > lifts[0]


def test_designed_run_skill_authority_grows_deeper_into_tail_and_neutral_stays_centered() -> None:
    profile = _run_profile()

    def sample(explosive: float, power: float, tackling: float, seed: int) -> np.ndarray:
        rng = np.random.default_rng(seed)
        return np.asarray(
            [
                resolve_run_ecology_progressive_skill_v2(
                    profile,
                    matchup_stuff_probability=0.18,
                    matchup_yards_multiplier=1.0,
                    runner_power=power,
                    tackling=tackling,
                    explosiveness=explosive,
                    rng=rng,
                ).total_yards
                for _ in range(65_000)
            ]
        )

    neutral = _tail_rates(sample(1.0, 1.0, 1.0, 2026091305))
    elite = _tail_rates(sample(1.24, 1.12, 0.90, 2026091306))
    constrained = _tail_rates(sample(0.82, 0.90, 1.12, 2026091307))

    assert abs(neutral[0] - profile.explosive_15_rate) < 0.010
    assert abs(neutral[1] - profile.explosive_20_rate) < 0.007
    assert abs(neutral[2] - profile.explosive_40_rate) < 0.0022
    assert all(e > n > c for e, n, c in zip(elite, neutral, constrained, strict=True))
    lifts = _relative_lifts(elite, constrained)
    assert lifts[2] > lifts[1] > lifts[0]


def test_scramble_skill_authority_is_progressive_without_moving_neutral_nfl_center() -> None:
    def sample(explosive: float, tackling: float, seed: int) -> np.ndarray:
        rng = np.random.default_rng(seed)
        return np.asarray(
            [
                resolve_scramble_contact_progressive_skill_v2(
                    penetration_probability=0.08,
                    runner_power=explosive,
                    tackling=tackling,
                    explosiveness=explosive,
                    rng=rng,
                ).total_yards
                for _ in range(85_000)
            ]
        )

    neutral_values = sample(1.0, 1.0, 2026091308)
    elite_values = sample(1.24, 0.90, 2026091309)
    constrained_values = sample(0.84, 1.12, 2026091310)
    neutral = _tail_rates(neutral_values)
    elite = _tail_rates(elite_values)
    constrained = _tail_rates(constrained_values)

    assert abs(neutral[0] - 0.10192837465564739) < 0.009
    assert abs(neutral[2] - 0.004591) < 0.0015
    assert all(e > n > c for e, n, c in zip(elite, neutral, constrained, strict=True))
    lifts = _relative_lifts(elite, constrained)
    assert lifts[2] > lifts[0]
