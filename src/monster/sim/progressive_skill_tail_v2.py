from __future__ import annotations

from dataclasses import replace

import numpy as np

from monster.sim.intent_ecology import PassDepthOutcome, RunGeometryOutcome
from monster.sim.pass_resolution_bands_v2 import (
    _sample_40plus_total,
    _sample_between,
    _twenty_to_forty_mean,
)
from monster.sim.play_anatomy import ContactResult, RunAnatomy, resolve_run_contact as _base_run_contact
from monster.sim.resolution_bands_v2 import (
    _anatomy_from_yards,
    _resolve_other_run_v2,
    _reshape_run_20plus_tail,
    _sample_routine_run_band_v2,
)
from monster.sim.resolution_ecology import (
    resolve_run_ecology as _historical_run_ecology,
    sample_yac as _historical_sample_yac,
)

# Progressive authority is intentionally tied to yardage severity.  A neutral interaction
# (edge == 1.0) leaves historical NFL band probabilities exactly unchanged.  Better/worse
# player-vs-defender interactions redistribute probability mass; they do not add plays,
# catches, drives, scores, or fantasy points directly.
_PASS_POSITIVE_BAND_AUTHORITY = np.asarray([-0.12, -0.05, 0.20, 0.48, 0.82, 1.25])
_SCRAMBLE_BAND_AUTHORITY = np.asarray([-0.10, -0.04, 0.08, 0.28, 0.65, 1.20])

_SCRAMBLE_GAIN3 = 0.8356290174471993
_SCRAMBLE_GAIN5 = 0.6299357208448118
_SCRAMBLE_GAIN10 = 0.25895316804407714
_SCRAMBLE_GAIN15 = 0.10192837465564739
_SCRAMBLE_GAIN40 = 0.004591


def _normalize_weighted(
    probabilities: np.ndarray,
    *,
    edge: float,
    authorities: np.ndarray,
) -> np.ndarray:
    probs = np.clip(np.asarray(probabilities, dtype=float), 0.0, None)
    total = float(probs.sum())
    if total <= 0.0:
        return np.full(len(probs), 1.0 / len(probs), dtype=float)
    edge = float(np.clip(edge, 0.62, 1.62))
    weights = probs * np.power(edge, authorities)
    weighted_total = float(weights.sum())
    return weights / weighted_total if weighted_total > 0.0 else probs / total


def _pass_interaction(receiver_explosiveness: float, coverage_strength: float) -> float:
    return float(
        np.clip(
            max(receiver_explosiveness, 0.55) / max(coverage_strength, 0.60) ** 0.30,
            0.62,
            1.62,
        )
    )


def _deeper_pass_yac(
    profile: PassDepthOutcome,
    *,
    receiver_explosiveness: float,
    coverage_strength: float,
    rng: np.random.Generator,
    air_yards: float,
) -> float:
    if air_yards >= 40.0 or profile.gain_40plus_completion_rate <= 0.0:
        return _historical_sample_yac(
            profile,
            receiver_explosiveness=receiver_explosiveness,
            coverage_strength=coverage_strength,
            rng=rng,
            air_yards=air_yards,
        )

    edge = _pass_interaction(receiver_explosiveness, coverage_strength)
    baseline = float(
        np.clip(
            profile.gain_40plus_completion_rate,
            0.0,
            max(profile.gain_20plus_completion_rate, 0.0),
        )
    )
    p40 = float(
        np.clip(
            baseline * edge**1.25,
            baseline * 0.42,
            min(max(profile.gain_20plus_completion_rate, baseline), baseline * 2.15),
        )
    )
    if rng.random() < p40:
        total_yards = _sample_40plus_total(profile, air_yards=air_yards, rng=rng)
        return float(max(total_yards - air_yards, 0.0))

    # p40 owns the far tail, so condition the ordinary depth/YAC draw below 40 and avoid a
    # second hidden 40+ mechanism.  This preserves the existing depth-specific YAC ecology.
    for _ in range(10):
        yac = _historical_sample_yac(
            profile,
            receiver_explosiveness=receiver_explosiveness,
            coverage_strength=coverage_strength,
            rng=rng,
            air_yards=air_yards,
        )
        if air_yards + yac < 40.0:
            return float(yac)
    return float(max(39.999 - air_yards, 0.0))


def sample_yac_progressive_skill_v2(
    profile: PassDepthOutcome,
    *,
    receiver_explosiveness: float,
    coverage_strength: float,
    rng: np.random.Generator,
    air_yards: float | None = None,
) -> float:
    """Give receiver-vs-pursuit skill progressively more authority deeper into the gain tail."""
    if air_yards is None:
        return _historical_sample_yac(
            profile,
            receiver_explosiveness=receiver_explosiveness,
            coverage_strength=coverage_strength,
            rng=rng,
            air_yards=air_yards,
        )

    air = float(air_yards)
    if profile.category not in {"behind_los", "short_0_5"} or profile.gain_5plus_completion_rate <= 0.0:
        return _deeper_pass_yac(
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
    p40 = float(np.clip(profile.gain_40plus_completion_rate, 0.0, p20))

    positive = np.asarray(
        [
            max(1.0 - negative_p - zero_p - p5, 0.0),
            max(p5 - p10, 0.0),
            max(p10 - p15, 0.0),
            max(p15 - p20, 0.0),
            max(p20 - p40, 0.0),
            p40,
        ],
        dtype=float,
    )
    positive = _normalize_weighted(
        positive,
        edge=_pass_interaction(receiver_explosiveness, coverage_strength),
        authorities=_PASS_POSITIVE_BAND_AUTHORITY,
    )
    positive *= max(1.0 - negative_p - zero_p, 0.0)
    probabilities = np.asarray([negative_p, zero_p, *positive.tolist()], dtype=float)
    probabilities = np.clip(probabilities, 0.0, None)
    probabilities /= probabilities.sum()

    # A non-negative air-yard completion cannot finish negative or at zero.  Preserve the
    # measured mass by moving those branches into the routine 0-5 gain bucket.
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
        center = _twenty_to_forty_mean(profile, low_mean=low_mean)
        low = max(air, 20.0)
        if low >= 39.999:
            total_yards = low
        else:
            scale = max(center - 20.0, 0.75)
            total_yards = float(np.clip(20.0 + rng.exponential(scale), low, 39.999))
    else:
        total_yards = _sample_40plus_total(profile, air_yards=air, rng=rng)
    return float(max(total_yards - air, 0.0))


def _run_skill_edge(*, explosiveness: float, runner_power: float, tackling: float) -> float:
    return float(
        np.clip(
            max(explosiveness, 0.55)
            * max(runner_power, 0.55) ** 0.20
            / max(tackling, 0.60) ** 0.28,
            0.62,
            1.62,
        )
    )


def _progressive_run_profile(
    profile: RunGeometryOutcome,
    *,
    explosiveness: float,
    runner_power: float,
    tackling: float,
) -> RunGeometryOutcome:
    edge = _run_skill_edge(
        explosiveness=explosiveness,
        runner_power=runner_power,
        tackling=tackling,
    )
    p10 = float(np.clip(profile.explosive_10_rate * edge**0.22, 0.0, 0.24))
    p15 = float(np.clip(profile.explosive_15_rate * edge**0.52, 0.0, p10))
    p20 = float(np.clip(profile.explosive_20_rate * edge**0.86, 0.0, p15))
    p40 = float(
        np.clip(
            profile.explosive_40_rate * edge**1.30,
            0.0,
            min(p20, max(profile.explosive_40_rate * 2.25, profile.explosive_40_rate)),
        )
    )
    return replace(
        profile,
        explosive_10_rate=p10,
        explosive_15_rate=p15,
        explosive_20_rate=p20,
        explosive_40_rate=p40,
    )


def resolve_run_ecology_progressive_skill_v2(
    profile: RunGeometryOutcome,
    *,
    matchup_stuff_probability: float,
    matchup_yards_multiplier: float,
    runner_power: float,
    tackling: float,
    explosiveness: float,
    rng: np.random.Generator,
) -> RunAnatomy:
    """Resolve designed runs with increasing player authority at 10+, 15+, 20+ and 40+."""
    if profile.category == "other":
        return _resolve_other_run_v2(rng=rng)

    shaped = _progressive_run_profile(
        profile,
        explosiveness=explosiveness,
        runner_power=runner_power,
        tackling=tackling,
    )
    # The shaped historical profile now owns explosive-frequency authority.  Passing neutral
    # values into the legacy explosive factor prevents the same player/defender evidence from
    # being charged a second time.  Stuff/efficiency still retain their existing mechanisms.
    anatomy = _historical_run_ecology(
        shaped,
        matchup_stuff_probability=matchup_stuff_probability,
        matchup_yards_multiplier=matchup_yards_multiplier,
        runner_power=runner_power,
        tackling=1.0,
        explosiveness=1.0,
        rng=rng,
    )
    if anatomy.total_yards >= 20.0:
        return _reshape_run_20plus_tail(
            anatomy,
            shaped,
            runner_power=1.0,
            tackling=1.0,
            explosiveness=1.0,
            rng=rng,
        )
    if anatomy.total_yards <= 0.0 or anatomy.total_yards >= 10.0:
        return anatomy
    replacement = _sample_routine_run_band_v2(
        shaped,
        matchup_yards_multiplier=matchup_yards_multiplier,
        runner_power=runner_power,
        tackling=tackling,
        rng=rng,
    )
    return replacement if replacement is not None else anatomy


def resolve_scramble_contact_progressive_skill_v2(
    *,
    penetration_probability: float,
    runner_power: float,
    tackling: float,
    explosiveness: float,
    rng: np.random.Generator,
) -> RunAnatomy:
    """Give QB mobility/pursuit interaction progressively more authority in open space."""
    burst = float(np.clip(explosiveness, 0.55, 1.60))
    runner = float(np.clip(runner_power, 0.55, 1.50))
    pursuit = float(np.clip(tackling, 0.60, 1.60))
    pressure_escape = float(((1.0 + 0.08) / (1.0 + max(penetration_probability, 0.0))) ** 0.25)
    edge = float(
        np.clip(
            burst**0.62 * runner**0.22 * pressure_escape / pursuit**0.36,
            0.62,
            1.62,
        )
    )
    base = np.asarray(
        [
            1.0 - _SCRAMBLE_GAIN3,
            _SCRAMBLE_GAIN3 - _SCRAMBLE_GAIN5,
            _SCRAMBLE_GAIN5 - _SCRAMBLE_GAIN10,
            _SCRAMBLE_GAIN10 - _SCRAMBLE_GAIN15,
            _SCRAMBLE_GAIN15 - _SCRAMBLE_GAIN40,
            _SCRAMBLE_GAIN40,
        ],
        dtype=float,
    )
    probabilities = _normalize_weighted(
        base,
        edge=edge,
        authorities=_SCRAMBLE_BAND_AUTHORITY,
    )
    band = int(rng.choice(6, p=probabilities))

    if band == 0:
        yards = 0.05 + 2.949 * float(rng.beta(1.8, 2.2))
    elif band == 1:
        yards = 3.0 + 1.999 * float(rng.beta(2.0, 2.0))
    elif band == 2:
        yards = 5.0 + 4.999 * float(rng.beta(2.0, 2.0))
    elif band == 3:
        yards = 10.0 + 4.999 * float(rng.beta(2.1, 2.0))
    elif band == 4:
        fraction = float(rng.beta(1.6, 2.5))
        # Within the 15-39 branch, open-field skill also changes how often the runner reaches
        # 20+ without inventing a new league-wide 20+ target.  Neutral edge is identical.
        fraction = fraction ** (1.0 / max(edge**0.70, 0.55))
        yards = 15.0 + 24.999 * fraction
    else:
        yards = float(np.clip(40.0 + rng.gamma(2.0, 4.0 * burst), 40.0, 70.0))

    return _anatomy_from_yards(
        yards,
        rng=rng,
        broken_tackle_bias=0.12 * max(edge - 1.0, 0.0) + (0.12 if band >= 4 else 0.0),
    )


def resolve_run_contact_progressive_skill_v2(
    *,
    penetration_probability: float,
    runner_power: float,
    tackling: float,
    explosiveness: float,
    rng: np.random.Generator,
) -> RunAnatomy:
    if abs(float(penetration_probability) - 0.08) <= 1e-9:
        return resolve_scramble_contact_progressive_skill_v2(
            penetration_probability=penetration_probability,
            runner_power=runner_power,
            tackling=tackling,
            explosiveness=explosiveness,
            rng=rng,
        )
    return _base_run_contact(
        penetration_probability=penetration_probability,
        runner_power=runner_power,
        tackling=tackling,
        explosiveness=explosiveness,
        rng=rng,
    )
