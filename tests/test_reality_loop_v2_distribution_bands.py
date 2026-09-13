from __future__ import annotations

import numpy as np

from monster.sim.intent_ecology import RunGeometryOutcome
from monster.sim.resolution_bands_v2 import (
    resolve_run_ecology_v2,
    resolve_scramble_contact_v2,
)


def _profile(category: str = "interior") -> RunGeometryOutcome:
    return RunGeometryOutcome(
        category=category,
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


def _rates(values: np.ndarray) -> dict[str, float]:
    return {
        "negative": float(np.mean(values < 0.0)),
        "gain3": float(np.mean(values >= 3.0)),
        "gain5": float(np.mean(values >= 5.0)),
        "gain10": float(np.mean(values >= 10.0)),
        "gain15": float(np.mean(values >= 15.0)),
        "gain20": float(np.mean(values >= 20.0)),
        "gain40": float(np.mean(values >= 40.0)),
    }


def test_designed_run_v2_restores_middle_band_and_empirical_far_tail() -> None:
    rng = np.random.default_rng(2026091211)
    profile = _profile()
    values = np.asarray(
        [
            resolve_run_ecology_v2(
                profile,
                matchup_stuff_probability=0.18,
                matchup_yards_multiplier=1.0,
                runner_power=1.0,
                tackling=1.0,
                explosiveness=1.0,
                rng=rng,
            ).total_yards
            for _ in range(55_000)
        ],
        dtype=float,
    )
    rates = _rates(values)
    assert abs(rates["negative"] - profile.negative_rate) < 0.015
    assert abs(rates["gain10"] - profile.explosive_10_rate) < 0.015
    assert abs(rates["gain15"] - profile.explosive_15_rate) < 0.012
    assert abs(rates["gain20"] - profile.explosive_20_rate) < 0.009
    assert abs(rates["gain40"] - profile.explosive_40_rate) < 0.0025
    assert abs(rates["gain5"] - 0.309198) < 0.020
    assert abs(rates["gain3"] - 0.565136) < 0.022


def test_designed_run_far_tail_preserves_player_explosive_authority() -> None:
    profile = _profile()

    def sample(explosive: float, power: float, tackling: float, seed: int) -> np.ndarray:
        rng = np.random.default_rng(seed)
        return np.asarray(
            [
                resolve_run_ecology_v2(
                    profile,
                    matchup_stuff_probability=0.18,
                    matchup_yards_multiplier=1.0,
                    runner_power=power,
                    tackling=tackling,
                    explosiveness=explosive,
                    rng=rng,
                ).total_yards
                for _ in range(45_000)
            ],
            dtype=float,
        )

    constrained = sample(0.82, 0.90, 1.12, 2026091218)
    elite = sample(1.24, 1.12, 0.90, 2026091219)
    assert float(np.mean(elite >= 40.0)) > float(np.mean(constrained >= 40.0))


def test_unknown_run_geometry_no_longer_becomes_an_artificial_stuff_bucket() -> None:
    rng = np.random.default_rng(2026091212)
    profile = RunGeometryOutcome(
        category="other",
        attempts=100,
        yards_mean=-0.8,
        yards_sd=2.0,
        negative_rate=0.80,
        zero_rate=0.04,
        loss_2_plus_rate=0.35,
        loss_5_plus_rate=0.08,
        explosive_10_rate=0.0,
        explosive_15_rate=0.0,
        explosive_20_rate=0.0,
        touchdown_rate=0.0,
        fumble_lost_rate=0.0,
        yards_p10=-2.0,
        yards_p50=-1.0,
        yards_p90=1.0,
        yards_p99=3.0,
    )
    values = np.asarray(
        [
            resolve_run_ecology_v2(
                profile,
                matchup_stuff_probability=0.30,
                matchup_yards_multiplier=0.80,
                runner_power=0.90,
                tackling=1.10,
                explosiveness=0.90,
                rng=rng,
            ).total_yards
            for _ in range(20_000)
        ],
        dtype=float,
    )
    rates = _rates(values)
    assert 0.03 < rates["negative"] < 0.07
    assert 0.008 < rates["gain5"] < 0.025


def test_scramble_v2_fills_10_to_14_and_40plus_bands_without_inflating_15_plus() -> None:
    rng = np.random.default_rng(2026091213)
    values = np.asarray(
        [
            resolve_scramble_contact_v2(
                penetration_probability=0.08,
                runner_power=1.0,
                tackling=1.0,
                explosiveness=1.0,
                rng=rng,
            ).total_yards
            for _ in range(80_000)
        ],
        dtype=float,
    )
    rates = _rates(values)
    assert rates["negative"] == 0.0
    assert abs(rates["gain3"] - 0.835629) < 0.016
    assert abs(rates["gain5"] - 0.629936) < 0.016
    assert abs(rates["gain10"] - 0.258953) < 0.015
    assert abs(rates["gain15"] - 0.101928) < 0.012
    assert abs(rates["gain40"] - 0.004591) < 0.0018


def test_scramble_v2_preserves_player_mobility_authority() -> None:
    def sample(explosiveness: float, tackling: float, seed: int) -> np.ndarray:
        rng = np.random.default_rng(seed)
        return np.asarray(
            [
                resolve_scramble_contact_v2(
                    penetration_probability=0.08,
                    runner_power=explosiveness,
                    tackling=tackling,
                    explosiveness=explosiveness,
                    rng=rng,
                ).total_yards
                for _ in range(45_000)
            ],
            dtype=float,
        )

    ordinary = sample(0.86, 1.10, 2026091214)
    elite = sample(1.22, 0.90, 2026091215)
    assert float(elite.mean()) > float(ordinary.mean())
    assert float(np.mean(elite >= 10.0)) > float(np.mean(ordinary >= 10.0))
    assert float(np.mean(elite >= 40.0)) > float(np.mean(ordinary >= 40.0))
