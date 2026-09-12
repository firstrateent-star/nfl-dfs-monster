from __future__ import annotations

import numpy as np

from monster.sim.intent_ecology import PassDepthOutcome
from monster.sim.resolution_ecology import sample_yac as _base_sample_yac

_SHALLOW_CATEGORIES = {"behind_los", "short_0_5"}


def _tilt_positive_bands(probabilities: np.ndarray, *, interaction: float) -> np.ndarray:
    probabilities = np.clip(np.asarray(probabilities, dtype=float), 0.0, None)
    total = float(probabilities.sum())
    if total <= 0.0:
        return np.full(len(probabilities), 1.0 / len(probabilities), dtype=float)
    ranks = np.linspace(-1.0, 1.0, len(probabilities))
    edge = float(np.clip(interaction, 0.72, 1.35))
    weights = probabilities * np.power(edge, 0.34 * ranks)
    return weights / weights.sum()


def _sample_between(low: float, high: float, rng: np.random.Generator) -> float:
    if high <= low + 1e-8:
        return float(low)
    # NFL threshold bands are not uniform: most qualifying gains live nearer the threshold
    # than the next boundary. A lower-skewed beta preserves membership without inflating mean.
    fraction = float(rng.beta(1.70, 3.00))
    return float(low + (high - low) * fraction)


def _twenty_plus_mean(profile: PassDepthOutcome, *, low_mean: float) -> float:
    """Solve the 20+ conditional center while conserving historical mean catch yards."""
    negative_p = float(np.clip(profile.negative_completion_rate, 0.0, 0.80))
    zero_p = float(np.clip(profile.zero_completion_rate, 0.0, 0.40))
    p5 = float(np.clip(profile.gain_5plus_completion_rate, 0.0, 1.0))
    p10 = float(np.clip(profile.gain_10plus_completion_rate, 0.0, p5))
    p15 = float(np.clip(profile.gain_15plus_completion_rate, 0.0, p10))
    p20 = float(np.clip(profile.gain_20plus_completion_rate, 0.0, p15))
    p0_5 = max(1.0 - negative_p - zero_p - p5, 0.0)
    p5_10 = max(p5 - p10, 0.0)
    p10_15 = max(p10 - p15, 0.0)
    p15_20 = max(p15 - p20, 0.0)
    target = max(float(profile.yards_mean_completed), 0.0)
    if p20 <= 1e-6:
        return 25.0
    # Fixed conditional centers are only used to solve the tail center. Actual samples stay
    # continuous. This turns historical threshold mass into shape rather than adding yardage.
    non_tail = (
        negative_p * -1.5
        + p0_5 * low_mean
        + p5_10 * 6.80
        + p10_15 * 11.80
        + p15_20 * 16.80
    )
    solved = (target - non_tail) / p20
    return float(np.clip(solved, 20.75, 52.0))


def sample_yac_v2(
    profile: PassDepthOutcome,
    *,
    receiver_explosiveness: float,
    coverage_strength: float,
    rng: np.random.Generator,
    air_yards: float | None = None,
) -> float:
    """Restore shallow-completion gain topology without globally adding passing yards.

    Historical completion probability remains owned by the throw resolver. Once a shallow
    pass is caught, this function samples total-gain bands measured directly from completed
    NFL passes, including the 20+ YAC tail. Receiver open-field skill and defensive pursuit
    receive bounded relative authority over adjacent positive bands. The historical completed
    catch mean is conserved by solving the 20+ conditional center instead of applying a boost.
    Deeper throws retain the existing YAC model.
    """
    if (
        profile.category not in _SHALLOW_CATEGORIES
        or air_yards is None
        or profile.gain_5plus_completion_rate <= 0.0
    ):
        return _base_sample_yac(
            profile,
            receiver_explosiveness=receiver_explosiveness,
            coverage_strength=coverage_strength,
            rng=rng,
            air_yards=air_yards,
        )

    air = float(air_yards)
    negative_p = float(np.clip(profile.negative_completion_rate, 0.0, 0.80))
    zero_p = float(np.clip(profile.zero_completion_rate, 0.0, max(0.0, 0.90 - negative_p)))
    p5 = float(np.clip(profile.gain_5plus_completion_rate, 0.0, 1.0))
    p10 = float(np.clip(profile.gain_10plus_completion_rate, 0.0, p5))
    p15 = float(np.clip(profile.gain_15plus_completion_rate, 0.0, p10))
    p20 = float(np.clip(profile.gain_20plus_completion_rate, 0.0, p15))
    p0_5 = max(1.0 - negative_p - zero_p - p5, 0.0)
    positive = np.asarray(
        [
            p0_5,
            max(p5 - p10, 0.0),
            max(p10 - p15, 0.0),
            max(p15 - p20, 0.0),
            p20,
        ],
        dtype=float,
    )
    interaction = float(
        np.clip(
            receiver_explosiveness / max(coverage_strength**0.30, 0.80),
            0.72,
            1.35,
        )
    )
    positive = _tilt_positive_bands(positive, interaction=interaction)
    positive_mass = max(1.0 - negative_p - zero_p, 0.0)
    positive *= positive_mass
    probabilities = np.asarray([negative_p, zero_p, *positive.tolist()], dtype=float)
    probabilities = np.clip(probabilities, 0.0, None)
    probabilities /= probabilities.sum()

    # Current event geometry does not represent negative YAC after a positive-air catch.
    # Keep those physically infeasible negative/zero historical branches in the <5 bucket.
    if air >= 0.0:
        probabilities[2] += probabilities[0] + probabilities[1]
        probabilities[0] = 0.0
        probabilities[1] = 0.0
        probabilities /= probabilities.sum()

    band = int(rng.choice(len(probabilities), p=probabilities))
    if band == 0:  # negative total gain; feasible on behind-LOS catches
        high = min(-0.01, max(air + 0.01, -0.01))
        low = min(air, high - 0.01)
        total_yards = _sample_between(low, high, rng)
    elif band == 1:
        total_yards = 0.0
    elif band == 2:
        low = max(air, 0.05)
        total_yards = low if low >= 5.0 else _sample_between(low, 4.999, rng)
    elif band == 3:
        total_yards = _sample_between(max(air, 5.0), 9.999, rng)
    elif band == 4:
        total_yards = _sample_between(max(air, 10.0), 14.999, rng)
    elif band == 5:
        total_yards = _sample_between(max(air, 15.0), 19.999, rng)
    else:
        low_mean = 2.10 if profile.category == "behind_los" else 3.10
        mean = _twenty_plus_mean(profile, low_mean=low_mean)
        scale = max(mean - 20.0, 0.75)
        total_yards = float(np.clip(20.0 + rng.exponential(scale), 20.0, 70.0))

    return float(max(total_yards - air, 0.0))
