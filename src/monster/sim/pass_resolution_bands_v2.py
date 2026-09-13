from __future__ import annotations

import numpy as np

from monster.sim.intent_ecology import PassDepthOutcome
from monster.sim.resolution_ecology import sample_yac as _base_sample_yac

_SHALLOW_CATEGORIES = {"behind_los", "short_0_5"}


def _tilt_positive_bands(
    probabilities: np.ndarray,
    *,
    interaction: float,
    authority: float = 0.42,
) -> np.ndarray:
    probabilities = np.clip(np.asarray(probabilities, dtype=float), 0.0, None)
    total = float(probabilities.sum())
    if total <= 0.0:
        return np.full(len(probabilities), 1.0 / len(probabilities), dtype=float)
    ranks = np.linspace(-1.0, 1.0, len(probabilities))
    edge = float(np.clip(interaction, 0.72, 1.35))
    weights = probabilities * np.power(edge, authority * ranks)
    return weights / weights.sum()


def _sample_between(low: float, high: float, rng: np.random.Generator) -> float:
    if high <= low + 1e-8:
        return float(low)
    # NFL threshold bands are not uniform: most qualifying gains live nearer the threshold
    # than the next boundary. A lower-skewed beta preserves membership without inflating mean.
    fraction = float(rng.beta(1.70, 3.00))
    return float(low + (high - low) * fraction)


def _forty_plus_mean(profile: PassDepthOutcome, *, minimum: float = 40.0) -> float:
    observed = float(profile.yards_40plus_mean_completed)
    if observed >= minimum:
        return float(np.clip(observed, minimum, 75.0))
    return float(max(minimum + 7.0, profile.yards_mean_completed + 31.0))


def _twenty_to_forty_mean(profile: PassDepthOutcome, *, low_mean: float) -> float:
    """Solve the 20-39 center while preserving the historical completed-catch mean."""
    negative_p = float(np.clip(profile.negative_completion_rate, 0.0, 0.80))
    zero_p = float(np.clip(profile.zero_completion_rate, 0.0, 0.40))
    p5 = float(np.clip(profile.gain_5plus_completion_rate, 0.0, 1.0))
    p10 = float(np.clip(profile.gain_10plus_completion_rate, 0.0, p5))
    p15 = float(np.clip(profile.gain_15plus_completion_rate, 0.0, p10))
    p20 = float(np.clip(profile.gain_20plus_completion_rate, 0.0, p15))
    p40 = float(np.clip(profile.gain_40plus_completion_rate, 0.0, p20))
    p20_40 = max(p20 - p40, 0.0)
    p0_5 = max(1.0 - negative_p - zero_p - p5, 0.0)
    p5_10 = max(p5 - p10, 0.0)
    p10_15 = max(p10 - p15, 0.0)
    p15_20 = max(p15 - p20, 0.0)
    target = max(float(profile.yards_mean_completed), 0.0)
    if p20_40 <= 1e-6:
        return 27.0
    known = (
        negative_p * -1.5
        + p0_5 * low_mean
        + p5_10 * 6.80
        + p10_15 * 11.80
        + p15_20 * 16.80
        + p40 * _forty_plus_mean(profile)
    )
    solved = (target - known) / p20_40
    return float(np.clip(solved, 20.75, 37.5))


def _sample_40plus_total(
    profile: PassDepthOutcome,
    *,
    air_yards: float,
    rng: np.random.Generator,
) -> float:
    mean = max(_forty_plus_mean(profile), air_yards, 40.0)
    low = max(40.0, air_yards)
    residual_mean = max(mean - low, 1.25)
    # A shifted gamma keeps 40+ severity continuous while centering on the empirical
    # conditional mean rather than using a generic fantasy-style ceiling boost.
    residual = float(rng.gamma(shape=2.0, scale=residual_mean / 2.0))
    return float(np.clip(low + residual, low, 80.0))


def _far_tail_probability(
    profile: PassDepthOutcome,
    *,
    receiver_explosiveness: float,
    coverage_strength: float,
) -> float:
    baseline = float(
        np.clip(
            profile.gain_40plus_completion_rate,
            0.0,
            max(profile.gain_20plus_completion_rate, 0.0),
        )
    )
    if baseline <= 0.0:
        return 0.0
    interaction = float(
        np.clip(
            receiver_explosiveness / max(coverage_strength**0.28, 0.80),
            0.70,
            1.42,
        )
    )
    # Usage determines how many chances a featured player receives. This bounded rate tilt
    # determines how often an individual chance turns into a true breakaway. Neutral players
    # retain the historical rate exactly; elite open-field players earn more of the tail.
    probability = baseline * interaction**0.90
    upper = min(max(profile.gain_20plus_completion_rate, baseline), baseline * 1.45)
    lower = baseline * 0.62
    return float(np.clip(probability, lower, upper))


def _sample_deeper_yac_with_far_tail(
    profile: PassDepthOutcome,
    *,
    receiver_explosiveness: float,
    coverage_strength: float,
    rng: np.random.Generator,
    air_yards: float,
) -> float:
    # Any completed pass thrown 40+ yards downfield is already a 40+ gain under the current
    # non-negative YAC geometry. Preserve its existing depth-specific resolution.
    if air_yards >= 40.0:
        return _base_sample_yac(
            profile,
            receiver_explosiveness=receiver_explosiveness,
            coverage_strength=coverage_strength,
            rng=rng,
            air_yards=air_yards,
        )

    p40 = _far_tail_probability(
        profile,
        receiver_explosiveness=receiver_explosiveness,
        coverage_strength=coverage_strength,
    )
    if p40 <= 0.0:
        return _base_sample_yac(
            profile,
            receiver_explosiveness=receiver_explosiveness,
            coverage_strength=coverage_strength,
            rng=rng,
            air_yards=air_yards,
        )

    if rng.random() < p40:
        total_yards = _sample_40plus_total(profile, air_yards=air_yards, rng=rng)
        return float(max(total_yards - air_yards, 0.0))

    # The historical p40 now owns the far-tail frequency. Condition the ordinary YAC draw
    # below 40 so the legacy gamma tail cannot independently add a second 40+ mechanism.
    for _ in range(8):
        yac = _base_sample_yac(
            profile,
            receiver_explosiveness=receiver_explosiveness,
            coverage_strength=coverage_strength,
            rng=rng,
            air_yards=air_yards,
        )
        if air_yards + yac < 40.0:
            return float(yac)
    return float(max(39.999 - air_yards, 0.0))


def sample_yac_v2(
    profile: PassDepthOutcome,
    *,
    receiver_explosiveness: float,
    coverage_strength: float,
    rng: np.random.Generator,
    air_yards: float | None = None,
) -> float:
    """Restore empirical completion topology and preserve a skill-owned 40+ tail.

    Historical completion probability remains owned by the throw resolver. Historical gain
    bands own the neutral shape of a completed catch. Player speed/open-field skill and the
    opposing coverage/pursuit state receive bounded relative authority over which players
    occupy the rare 40+ tail; usage still controls how many chances featured players receive.
    """
    if air_yards is None:
        return _base_sample_yac(
            profile,
            receiver_explosiveness=receiver_explosiveness,
            coverage_strength=coverage_strength,
            rng=rng,
            air_yards=air_yards,
        )

    air = float(air_yards)
    if profile.category not in _SHALLOW_CATEGORIES or profile.gain_5plus_completion_rate <= 0.0:
        return _sample_deeper_yac_with_far_tail(
            profile,
            receiver_explosiveness=receiver_explosiveness,
            coverage_strength=coverage_strength,
            rng=rng,
            air_yards=air,
        )

    negative_p = float(np.clip(profile.negative_completion_rate, 0.0, 0.80))
    zero_p = float(np.clip(profile.zero_completion_rate, 0.0, max(0.0, 0.90 - negative_p)))
    p5 = float(np.clip(profile.gain_5plus_completion_rate, 0.0, 1.0))
    p10 = float(np.clip(profile.gain_10plus_completion_rate, 0.0, p5))
    p15 = float(np.clip(profile.gain_15plus_completion_rate, 0.0, p10))
    p20 = float(np.clip(profile.gain_20plus_completion_rate, 0.0, p15))
    baseline_p40 = float(np.clip(profile.gain_40plus_completion_rate, 0.0, p20))
    interaction = float(
        np.clip(
            receiver_explosiveness / max(coverage_strength**0.30, 0.80),
            0.72,
            1.35,
        )
    )
    p40 = _far_tail_probability(
        profile,
        receiver_explosiveness=receiver_explosiveness,
        coverage_strength=coverage_strength,
    )
    # Keep total 20+ mass close to its measured center while skill redistributes the far end.
    p20_40 = max(p20 - p40, 0.0)
    p0_5 = max(1.0 - negative_p - zero_p - p5, 0.0)
    positive = np.asarray(
        [
            p0_5,
            max(p5 - p10, 0.0),
            max(p10 - p15, 0.0),
            max(p15 - p20, 0.0),
            p20_40,
            p40,
        ],
        dtype=float,
    )
    # Moderate authority across routine bands, with the explicit p40 adjustment above giving
    # the far tail stronger skill ownership without moving the league center for neutral skill.
    positive = _tilt_positive_bands(positive, interaction=interaction, authority=0.30)
    positive_mass = max(1.0 - negative_p - zero_p, 0.0)
    positive *= positive_mass
    probabilities = np.asarray([negative_p, zero_p, *positive.tolist()], dtype=float)
    probabilities = np.clip(probabilities, 0.0, None)
    probabilities /= probabilities.sum()

    if air >= 0.0:
        probabilities[2] += probabilities[0] + probabilities[1]
        probabilities[0] = 0.0
        probabilities[1] = 0.0
        probabilities /= probabilities.sum()

    band = int(rng.choice(len(probabilities), p=probabilities))
    if band == 0:
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
    elif band == 6:
        low_mean = 2.10 if profile.category == "behind_los" else 3.10
        mean = _twenty_to_forty_mean(profile, low_mean=low_mean)
        low = max(air, 20.0)
        high = 39.999
        if low >= high:
            total_yards = low
        else:
            # Gamma-like variation around the solved 20-39 center while respecting the band.
            scale = max(mean - 20.0, 0.75)
            total_yards = float(np.clip(20.0 + rng.exponential(scale), low, high))
    else:
        total_yards = _sample_40plus_total(profile, air_yards=air, rng=rng)

    # baseline_p40 is intentionally referenced in this implementation contract: when both
    # players are neutral, p40 remains the empirical baseline before ordinary band tilt.
    _ = baseline_p40
    return float(max(total_yards - air, 0.0))
