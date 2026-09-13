from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from monster.sim.intent_ecology import RunGeometryOutcome
from monster.sim.play_anatomy import (
    ContactResult,
    RunAnatomy,
    resolve_run_contact as _base_resolve_run_contact,
)
from monster.sim.resolution_ecology import resolve_run_ecology as _base_resolve_run_ecology


@dataclass(frozen=True)
class RunBandTarget:
    gain_3plus: float
    gain_5plus: float


_RUN_BAND_TARGETS: dict[str, RunBandTarget] = {
    "interior": RunBandTarget(0.565136, 0.309198),
    "left_edge": RunBandTarget(0.597662, 0.423659),
    "left_offtackle": RunBandTarget(0.561069, 0.331425),
    "right_edge": RunBandTarget(0.608567, 0.426883),
    "right_offtackle": RunBandTarget(0.592883, 0.361882),
}

_OTHER_RUN = {
    "negative": 0.047619,
    "gain_3plus": 0.023810,
    "gain_5plus": 0.015873,
    "gain_10plus": 0.007937,
    "gain_15plus": 0.007937,
    "yards_mean": 0.357143,
}

# 2025 regular-season QB scramble topology from the same nflverse audit used by Reality Loop.
_SCRAMBLE_15PLUS = 0.10192837465564739
_SCRAMBLE_40PLUS = 0.004591
_SCRAMBLE_BANDS = np.asarray(
    [
        1.0 - 0.8356290174471993,
        0.8356290174471993 - 0.6299357208448118,
        0.6299357208448118 - 0.25895316804407714,
        0.25895316804407714 - _SCRAMBLE_15PLUS,
        _SCRAMBLE_15PLUS,
    ],
    dtype=float,
)


def _tilt_ordered(probabilities: np.ndarray, *, edge: float, authority: float) -> np.ndarray:
    """Tilt ordered outcome bands while keeping a neutral edge exactly neutral."""
    probs = np.asarray(probabilities, dtype=float)
    probs = np.clip(probs, 0.0, None)
    if probs.sum() <= 0.0:
        return np.full(len(probs), 1.0 / len(probs), dtype=float)
    ranks = np.linspace(-1.0, 1.0, len(probs))
    clipped_edge = float(np.clip(edge, 0.65, 1.55))
    weights = probs * np.power(clipped_edge, authority * ranks)
    total = float(weights.sum())
    return weights / total if total > 0.0 else probs / probs.sum()


def _anatomy_from_yards(
    yards: float,
    *,
    rng: np.random.Generator,
    broken_tackle_bias: float = 0.0,
) -> RunAnatomy:
    yards = float(max(yards, 0.0))
    if yards <= 0.0:
        return RunAnatomy(False, ContactResult.TACKLED, 0.0, 0.0, 0.0)
    before_center = min(max(0.62 * yards, 0.6), 5.2)
    before = float(np.clip(rng.normal(before_center, 0.85), 0.0, yards))
    after = float(max(yards - before, 0.0))
    break_probability = float(
        np.clip(0.08 + 0.07 * max(yards - 5.0, 0.0) + broken_tackle_bias, 0.04, 0.72)
    )
    broken = rng.random() < break_probability
    return RunAnatomy(
        False,
        ContactResult.BROKEN_TACKLE if broken else ContactResult.TACKLED,
        before,
        after,
        yards,
    )


def _sample_routine_run_band_v2(
    profile: RunGeometryOutcome,
    *,
    matchup_yards_multiplier: float,
    runner_power: float,
    tackling: float,
    rng: np.random.Generator,
) -> RunAnatomy | None:
    target = _RUN_BAND_TARGETS.get(profile.category)
    if target is None:
        return None

    p0_3 = max(1.0 - profile.negative_rate - profile.zero_rate - target.gain_3plus, 0.0)
    p3_5 = max(target.gain_3plus - target.gain_5plus, 0.0)
    p5_10 = max(target.gain_5plus - profile.explosive_10_rate, 0.0)
    probabilities = np.asarray([p0_3, p3_5, p5_10], dtype=float)

    edge = float(
        np.clip(
            matchup_yards_multiplier
            * max(runner_power, 0.65) ** 0.12
            / max(tackling, 0.65) ** 0.10,
            0.72,
            1.38,
        )
    )
    probabilities = _tilt_ordered(probabilities, edge=edge, authority=0.85)
    band = int(rng.choice(3, p=probabilities))

    if band == 0:
        yards = 0.10 + 2.899 * float(rng.beta(2.0, 2.7))
    elif band == 1:
        yards = 3.0 + 1.999 * float(rng.beta(2.2, 2.0))
    else:
        yards = 5.0 + 4.999 * float(rng.beta(2.15, 1.95))

    return _anatomy_from_yards(
        yards,
        rng=rng,
        broken_tackle_bias=0.05 * max(edge - 1.0, 0.0),
    )


def _resolve_other_run_v2(*, rng: np.random.Generator) -> RunAnatomy:
    negative = _OTHER_RUN["negative"]
    p15 = _OTHER_RUN["gain_15plus"]
    p10_15 = max(_OTHER_RUN["gain_10plus"] - p15, 0.0)
    p5_10 = max(_OTHER_RUN["gain_5plus"] - _OTHER_RUN["gain_10plus"], 0.0)
    p3_5 = max(_OTHER_RUN["gain_3plus"] - _OTHER_RUN["gain_5plus"], 0.0)
    p0_3 = max(1.0 - negative - _OTHER_RUN["gain_3plus"], 0.0)
    probabilities = np.asarray([negative, p0_3, p3_5, p5_10, p10_15, p15], dtype=float)
    probabilities /= probabilities.sum()
    band = int(rng.choice(len(probabilities), p=probabilities))

    if band == 0:
        yards = -float(np.clip(rng.normal(1.0, 0.40), 0.1, 2.5))
        return RunAnatomy(True, ContactResult.STUFF, yards, 0.0, yards)
    if band == 1:
        yards = float(np.clip(rng.exponential(0.22), 0.0, 2.999))
        return _anatomy_from_yards(yards, rng=rng)
    if band == 2:
        return _anatomy_from_yards(float(rng.uniform(3.0, 5.0)), rng=rng)
    if band == 3:
        return _anatomy_from_yards(float(rng.uniform(5.0, 10.0)), rng=rng)
    if band == 4:
        return _anatomy_from_yards(float(rng.uniform(10.0, 15.0)), rng=rng)
    return _anatomy_from_yards(float(np.clip(15.0 + rng.exponential(3.0), 15.0, 35.0)), rng=rng)


def _run_far_tail_probability(
    profile: RunGeometryOutcome,
    *,
    runner_power: float,
    tackling: float,
    explosiveness: float,
) -> float:
    if profile.explosive_20_rate <= 1e-9 or profile.explosive_40_rate <= 0.0:
        return 0.0
    baseline_conditional = float(
        np.clip(profile.explosive_40_rate / profile.explosive_20_rate, 0.0, 0.90)
    )
    edge = float(
        np.clip(
            max(explosiveness, 0.60) ** 0.75
            * max(runner_power, 0.60) ** 0.18
            / max(tackling, 0.60) ** 0.30,
            0.68,
            1.50,
        )
    )
    probability = baseline_conditional * edge**0.90
    return float(
        np.clip(
            probability,
            baseline_conditional * 0.58,
            min(0.95, baseline_conditional * 1.55),
        )
    )


def _reshape_run_20plus_tail(
    anatomy: RunAnatomy,
    profile: RunGeometryOutcome,
    *,
    runner_power: float,
    tackling: float,
    explosiveness: float,
    rng: np.random.Generator,
) -> RunAnatomy:
    if anatomy.total_yards < 20.0 or profile.explosive_40_rate <= 0.0:
        return anatomy
    p40 = _run_far_tail_probability(
        profile,
        runner_power=runner_power,
        tackling=tackling,
        explosiveness=explosiveness,
    )
    edge = float(
        np.clip(
            max(explosiveness, 0.60) / max(tackling, 0.60) ** 0.28,
            0.70,
            1.48,
        )
    )
    if rng.random() < p40:
        observed_mean = float(profile.yards_40plus_mean)
        mean = observed_mean if observed_mean >= 40.0 else 48.0
        residual_mean = max(mean - 40.0, 1.5)
        yards = float(np.clip(40.0 + rng.gamma(2.0, residual_mean / 2.0), 40.0, 90.0))
        return _anatomy_from_yards(
            yards,
            rng=rng,
            broken_tackle_bias=0.14 * max(edge - 1.0, 0.0) + 0.10,
        )
    # The empirical p40 owns the far-tail frequency. Keep the remaining 20+ branch below 40.
    if anatomy.total_yards >= 40.0:
        yards = 20.0 + 19.999 * float(rng.beta(1.8, 2.4))
        return _anatomy_from_yards(yards, rng=rng)
    return anatomy


def resolve_run_ecology_v2(
    profile: RunGeometryOutcome,
    *,
    matchup_stuff_probability: float,
    matchup_yards_multiplier: float,
    runner_power: float,
    tackling: float,
    explosiveness: float,
    rng: np.random.Generator,
) -> RunAnatomy:
    """Preserve measured run branches while restoring routine and true-breakaway topology."""
    if profile.category == "other":
        return _resolve_other_run_v2(rng=rng)

    anatomy = _base_resolve_run_ecology(
        profile,
        matchup_stuff_probability=matchup_stuff_probability,
        matchup_yards_multiplier=matchup_yards_multiplier,
        runner_power=runner_power,
        tackling=tackling,
        explosiveness=explosiveness,
        rng=rng,
    )
    if anatomy.total_yards >= 20.0:
        return _reshape_run_20plus_tail(
            anatomy,
            profile,
            runner_power=runner_power,
            tackling=tackling,
            explosiveness=explosiveness,
            rng=rng,
        )
    if anatomy.total_yards <= 0.0 or anatomy.total_yards >= 10.0:
        return anatomy
    replacement = _sample_routine_run_band_v2(
        profile,
        matchup_yards_multiplier=matchup_yards_multiplier,
        runner_power=runner_power,
        tackling=tackling,
        rng=rng,
    )
    return replacement if replacement is not None else anatomy


def resolve_scramble_contact_v2(
    *,
    penetration_probability: float,
    runner_power: float,
    tackling: float,
    explosiveness: float,
    rng: np.random.Generator,
) -> RunAnatomy:
    """Resolve QB scramble topology, including the observed rare 40+ escape tail."""
    burst = float(np.clip(explosiveness, 0.60, 1.55))
    runner = float(np.clip(runner_power, 0.60, 1.45))
    pursuit = float(np.clip(tackling, 0.60, 1.55))
    pressure_escape = float(((1.0 + 0.08) / (1.0 + max(penetration_probability, 0.0))) ** 0.25)
    edge = float(
        np.clip(
            burst**0.55 * runner**0.20 * pressure_escape / max(pursuit, 0.60) ** 0.35,
            0.68,
            1.48,
        )
    )
    probabilities = _tilt_ordered(_SCRAMBLE_BANDS, edge=edge, authority=0.70)
    band = int(rng.choice(5, p=probabilities))

    if band == 0:
        yards = 0.05 + 2.949 * float(rng.beta(1.8, 2.2))
    elif band == 1:
        yards = 3.0 + 1.999 * float(rng.beta(2.0, 2.0))
    elif band == 2:
        yards = 5.0 + 4.999 * float(rng.beta(2.0, 2.0))
    elif band == 3:
        yards = 10.0 + 4.999 * float(rng.beta(2.1, 2.0))
    else:
        baseline_conditional = _SCRAMBLE_40PLUS / _SCRAMBLE_15PLUS
        p40 = float(
            np.clip(
                baseline_conditional * edge**1.05,
                baseline_conditional * 0.52,
                min(0.30, baseline_conditional * 1.75),
            )
        )
        if rng.random() < p40:
            # Long QB escapes are rare but materially longer than the ordinary 15+ branch.
            yards = float(np.clip(40.0 + rng.gamma(2.0, 4.0 * burst), 40.0, 70.0))
        else:
            yards = 15.0 + 24.999 * float(rng.beta(1.6, 2.5))

    return _anatomy_from_yards(
        yards,
        rng=rng,
        broken_tackle_bias=0.10 * max(edge - 1.0, 0.0) + (0.12 if band >= 3 else 0.0),
    )


def resolve_run_contact_v2(
    *,
    penetration_probability: float,
    runner_power: float,
    tackling: float,
    explosiveness: float,
    rng: np.random.Generator,
) -> RunAnatomy:
    """Shadow seam: route the dedicated scramble signature into scramble topology v2."""
    if abs(float(penetration_probability) - 0.08) <= 1e-9:
        return resolve_scramble_contact_v2(
            penetration_probability=penetration_probability,
            runner_power=runner_power,
            tackling=tackling,
            explosiveness=explosiveness,
            rng=rng,
        )
    return _base_resolve_run_contact(
        penetration_probability=penetration_probability,
        runner_power=runner_power,
        tackling=tackling,
        explosiveness=explosiveness,
        rng=rng,
    )
