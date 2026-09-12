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


# 2025 regular-season NFL designed-run topology, measured from the same nflverse definition
# used by the Reality Loop audit. These are branch priors, not fantasy/stat targets. Existing
# negative/zero/10+/15+/20+ ecology stays owned by RunGeometryOutcome; this table supplies the
# missing 3+ and 5+ boundaries so the routine branch cannot collapse too much mass into 2-4 yd.
_RUN_BAND_TARGETS: dict[str, RunBandTarget] = {
    "interior": RunBandTarget(0.565136, 0.309198),
    "left_edge": RunBandTarget(0.597662, 0.423659),
    "left_offtackle": RunBandTarget(0.561069, 0.331425),
    "right_edge": RunBandTarget(0.608567, 0.426883),
    "right_offtackle": RunBandTarget(0.592883, 0.361882),
}

# Unknown/missing run geometry is deliberately low-information. The old fallback accidentally
# turned this small class into an 80% negative-gain bucket. Preserve its observed marginal
# topology instead of pretending an unknown lane has known front/blocking mechanics.
_OTHER_RUN = {
    "negative": 0.047619,
    "gain_3plus": 0.023810,
    "gain_5plus": 0.015873,
    "gain_10plus": 0.007937,
    "gain_15plus": 0.007937,
    "yards_mean": 0.357143,
}

# 2025 regular-season QB scramble topology from the same play-family audit. Once the QB has
# resolved to SCRAMBLE, the old generic contact model created too many small/negative gains and
# skipped the 10-14 yard escape band. These probabilities describe open-field topology only;
# mobility, runner power, pursuit and tackling still tilt probability between adjacent bands.
_SCRAMBLE_BANDS = np.asarray(
    [
        1.0 - 0.8356290174471993,  # 0-2
        0.8356290174471993 - 0.6299357208448118,  # 3-4
        0.6299357208448118 - 0.25895316804407714,  # 5-9
        0.25895316804407714 - 0.10192837465564739,  # 10-14
        0.10192837465564739,  # 15+
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

    # This function is called only after the base resolver chose a positive sub-10 routine
    # outcome. Reconstruct the empirical conditional topology of that same reservoir.
    p0_3 = max(1.0 - profile.negative_rate - profile.zero_rate - target.gain_3plus, 0.0)
    p3_5 = max(target.gain_3plus - target.gain_5plus, 0.0)
    p5_10 = max(target.gain_5plus - profile.explosive_10_rate, 0.0)
    probabilities = np.asarray([p0_3, p3_5, p5_10], dtype=float)

    # Matchup quality moves mass between adjacent routine bands. It does not alter the
    # negative or 10+ branches already chosen by the base causal ecology.
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
        # Dense low-success cluster without piling probability at exactly 3 yards.
        yards = 0.10 + 2.899 * float(rng.beta(2.0, 2.7))
    elif band == 1:
        yards = 3.0 + 1.999 * float(rng.beta(2.2, 2.0))
    else:
        # The missing Monster mass lives mainly here. Use a continuous 5-9.999 branch,
        # preserving the existing 10+ tail as a separate mechanism.
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
        # Solve the low branch around the observed marginal mean instead of assigning normal
        # run-lane efficiency to a class whose geometry is explicitly unknown.
        yards = float(np.clip(rng.exponential(0.22), 0.0, 2.999))
        return _anatomy_from_yards(yards, rng=rng)
    if band == 2:
        return _anatomy_from_yards(float(rng.uniform(3.0, 5.0)), rng=rng)
    if band == 3:
        return _anatomy_from_yards(float(rng.uniform(5.0, 10.0)), rng=rng)
    if band == 4:
        return _anatomy_from_yards(float(rng.uniform(10.0, 15.0)), rng=rng)
    return _anatomy_from_yards(float(np.clip(15.0 + rng.exponential(3.0), 15.0, 35.0)), rng=rng)


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
    """Preserve negative/explosive branches while restoring ordinary NFL run-success bands."""
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
    # Do not disturb failure, zero, or explosive outcomes; those were already near the NFL
    # target. Only re-express the positive sub-10 routine reservoir.
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
    """Resolve a QB scramble through measured 0-2/3-4/5-9/10-14/15+ open-field bands."""
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
        # Explicitly fill the missing chunk-escape band without stealing from 15+ plays.
        yards = 10.0 + 4.999 * float(rng.beta(2.1, 2.0))
    else:
        yards = float(np.clip(15.0 + rng.exponential(4.8 * burst), 15.0, 48.0))

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
