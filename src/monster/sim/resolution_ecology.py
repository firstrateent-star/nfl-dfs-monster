from __future__ import annotations

from dataclasses import dataclass
from math import erf, exp, log, pi, sqrt
from statistics import NormalDist

import numpy as np

from monster.sim.intent_ecology import PassDepthOutcome, RunGeometryOutcome
from monster.sim.play_anatomy import (
    ContactResult,
    RunAnatomy,
    condition_throw_probabilities,
)

# Stage 3 historical outcome priors already contain league-wide difficulty. The current
# matchup kernel is still a coarse unit/player bridge rather than a snap-specific assignment,
# so it receives only partial relative authority. Run blocking/front evidence is somewhat
# more direct than pass coverage assignment, hence the modestly higher run authority.
_COARSE_MATCHUP_AUTHORITY = 0.35
_COARSE_RUN_MATCHUP_AUTHORITY = 0.50
_STANDARD_NORMAL = NormalDist()


@dataclass(frozen=True)
class DepthThrowProbabilities:
    completion: float
    interception: float


def _shrink_relative(relative: float, *, low: float, high: float) -> float:
    clipped = float(np.clip(relative, low, high))
    return float(exp(_COARSE_MATCHUP_AUTHORITY * log(max(clipped, 1e-9))))


def _shrink_run_relative(relative: float, *, low: float, high: float) -> float:
    """Grant coarse run matchup evidence bounded authority around a neutral 1.0 prior."""
    clipped = float(np.clip(relative, low, high))
    return float(exp(_COARSE_RUN_MATCHUP_AUTHORITY * log(max(clipped, 1e-9))))


def depth_throw_probabilities(
    profile: PassDepthOutcome,
    *,
    matchup_completion_probability: float | None,
    matchup_interception_probability: float | None,
    pressured: bool,
) -> DepthThrowProbabilities:
    """Resolve depth-aware throw quality without reintroducing a second league baseline.

    Historical depth outcome is the unconditional causal prior. The existing matchup path
    contributes only a shrunk relative player/QB/unit perturbation because it is still a
    coarse interaction layer. Pressure then conditions the already depth-aware probability
    through the independently measured clean-vs-pressure split.
    """

    completion = profile.completion_rate
    interception = profile.interception_rate
    if matchup_completion_probability is not None:
        completion *= _shrink_relative(
            matchup_completion_probability / 0.64,
            low=0.72,
            high=1.30,
        )
    if matchup_interception_probability is not None:
        interception *= _shrink_relative(
            matchup_interception_probability / 0.022,
            low=0.55,
            high=1.70,
        )
    completion, interception = condition_throw_probabilities(
        completion_probability=completion,
        interception_probability=interception,
        pressured=pressured,
    )
    return DepthThrowProbabilities(
        completion=completion,
        interception=interception,
    )


def completed_pass_yards(air_yards: float, yards_after_catch: float) -> float:
    """Preserve signed throw geometry when converting a catch into scrimmage yards."""

    return float(air_yards + max(yards_after_catch, 0.0))


def _gamma_draw(mean: float, sd: float, rng: np.random.Generator) -> float:
    mean = max(mean, 0.05)
    sd = max(sd, 0.35)
    shape = max((mean / sd) ** 2, 0.20)
    scale = max((sd**2) / mean, 0.05)
    return float(rng.gamma(shape, scale))


def sample_yac(
    profile: PassDepthOutcome,
    *,
    receiver_explosiveness: float,
    coverage_strength: float,
    rng: np.random.Generator,
    air_yards: float | None = None,
) -> float:
    """Sample YAC while preserving the measured relationship to signed throw geometry.

    Ordinary depths retain the historical depth-specific marginal YAC distribution. For
    behind-LOS completions, air yards and YAC cannot be independent: whether the receiver
    gets back to the line of scrimmage is itself a football outcome. We therefore sample
    negative/zero/non-negative total-gain branches from the measured completion anatomy,
    then solve the positive branch mean so the overall YAC mean remains coherent.
    """

    interaction = float(
        np.clip(
            receiver_explosiveness / max(coverage_strength**0.30, 0.80),
            0.72,
            1.35,
        )
    )
    target_mean = max(profile.yac_mean_completed * interaction, 0.05)
    sd = max(profile.yac_sd_completed, 0.35)

    if profile.category != "behind_los" or air_yards is None or air_yards >= 0.0:
        return float(np.clip(_gamma_draw(target_mean, sd, rng), 0.0, 65.0))

    threshold = max(-float(air_yards), 0.01)
    negative_p = float(np.clip(profile.negative_completion_rate, 0.0, 0.85))
    zero_p = float(
        np.clip(profile.zero_completion_rate, 0.0, max(0.0, 0.90 - negative_p))
    )
    positive_p = max(1.0 - negative_p - zero_p, 1e-6)

    # A negative completed screen must finish short of the line of scrimmage. A beta
    # fraction gives a smooth distribution within that physically constrained interval.
    negative_mean = threshold * (2.2 / (2.2 + 1.8))
    positive_mean = (
        target_mean - negative_p * negative_mean - zero_p * threshold
    ) / positive_p
    positive_mean = max(positive_mean, threshold + 0.20)

    draw = rng.random()
    if draw < negative_p:
        return float(threshold * rng.beta(2.2, 1.8) * (1.0 - 1e-6))
    if draw < negative_p + zero_p:
        return float(threshold)

    residual_mean = max(positive_mean - threshold, 0.20)
    residual_sd = max(min(sd, residual_mean * 1.25), 0.35)
    return float(
        np.clip(threshold + _gamma_draw(residual_mean, residual_sd, rng), 0.0, 65.0)
    )


def _normal_pdf(value: float) -> float:
    return exp(-0.5 * value * value) / sqrt(2.0 * pi)


def _normal_cdf(value: float) -> float:
    return 0.5 * (1.0 + erf(value / sqrt(2.0)))


def _truncated_normal_mean(
    center: float,
    sd: float,
    *,
    low: float = 0.1,
    high: float = 9.999,
) -> float:
    """Mean of a normal draw conditioned to remain inside the routine-run branch."""
    alpha = (low - center) / sd
    beta = (high - center) / sd
    mass = _normal_cdf(beta) - _normal_cdf(alpha)
    if mass <= 1e-12:
        return float(np.clip(center, low, high))
    return float(center + sd * (_normal_pdf(alpha) - _normal_pdf(beta)) / mass)


def _center_for_truncated_mean(
    target: float,
    sd: float,
    *,
    low: float = 0.1,
    high: float = 9.999,
) -> float:
    target = float(np.clip(target, low + 1e-4, high - 1e-4))
    lower, upper = -20.0, 30.0
    for _ in range(56):
        middle = (lower + upper) / 2.0
        if _truncated_normal_mean(middle, sd, low=low, high=high) < target:
            lower = middle
        else:
            upper = middle
    return float((lower + upper) / 2.0)


def _sample_truncated_normal(
    center: float,
    sd: float,
    rng: np.random.Generator,
    *,
    low: float = 0.1,
    high: float = 9.999,
) -> float:
    """Sample continuously inside a branch instead of winsorizing mass onto its edges."""
    alpha = (low - center) / sd
    beta = (high - center) / sd
    cdf_low = _normal_cdf(alpha)
    cdf_high = _normal_cdf(beta)
    mass = cdf_high - cdf_low
    if mass <= 1e-12:
        return float(np.clip(center, low + 1e-6, high - 1e-6))
    u = float(rng.uniform(cdf_low, cdf_high))
    u = float(np.clip(u, 1e-12, 1.0 - 1e-12))
    value = center + sd * _STANDARD_NORMAL.inv_cdf(u)
    return float(np.clip(value, low + 1e-9, high - 1e-9))


def _conditional_loss_mean(profile: RunGeometryOutcome) -> float:
    negative = max(profile.negative_rate, 1e-9)
    loss_5 = float(np.clip(profile.loss_5_plus_rate / negative, 0.0, 1.0))
    loss_2 = float(np.clip(profile.loss_2_plus_rate / negative, 0.0, 1.0))
    moderate = max(loss_2 - loss_5, 0.0)
    shallow = max(1.0 - loss_2, 0.0)
    # These are the expected severities of the explicit negative-yard samplers below.
    return -(loss_5 * 5.85 + moderate * 3.0 + shallow * 1.0)


def _breakaway_scale(profile: RunGeometryOutcome) -> float:
    """Derive conditional 20+ severity from the global p99 and 20+ frequency.

    If p20 > 1%, the league-wide 99th percentile lies inside the conditional 20+ tail.
    For a shifted exponential Y=20+Exp(scale), solve the scale so that corresponding
    conditional quantile lands on the observed global p99. This keeps breakaway frequency
    and breakaway severity as separate mechanisms.
    """

    p20 = max(profile.explosive_20_rate, 0.0)
    if p20 > 0.010001 and profile.yards_p99 > 20.0:
        denominator = log(p20 / 0.01)
        if denominator > 1e-6:
            return float(np.clip((profile.yards_p99 - 20.0) / denominator, 0.50, 24.0))
    return float(np.clip(max(profile.yards_p99, 22.0) - 20.0, 0.50, 8.0))


def _neutral_routine_mean(profile: RunGeometryOutcome) -> float:
    p_negative = max(profile.negative_rate, 0.0)
    p_zero = max(profile.zero_rate, 0.0)
    p20 = max(profile.explosive_20_rate, 0.0)
    p15 = max(profile.explosive_15_rate - p20, 0.0)
    p10 = max(profile.explosive_10_rate - profile.explosive_15_rate, 0.0)
    routine_p = max(1.0 - p_negative - p_zero - p10 - p15 - p20, 1e-6)
    known_mean = p_negative * _conditional_loss_mean(profile)
    known_mean += p10 * 12.5 + p15 * 17.5
    known_mean += p20 * (20.0 + _breakaway_scale(profile))
    return float((profile.yards_mean - known_mean) / routine_p)


def _sample_negative_run(
    profile: RunGeometryOutcome,
    rng: np.random.Generator,
) -> float:
    negative = max(profile.negative_rate, 1e-9)
    conditional_loss_5 = float(
        np.clip(profile.loss_5_plus_rate / negative, 0.0, 1.0)
    )
    conditional_loss_2 = float(
        np.clip(profile.loss_2_plus_rate / negative, 0.0, 1.0)
    )
    severity_draw = rng.random()
    if severity_draw < conditional_loss_5:
        return -float(np.clip(rng.lognormal(1.70, 0.35), 5.0, 14.0))
    if severity_draw < conditional_loss_2:
        return -float(np.clip(rng.normal(3.0, 0.8), 2.0, 5.0))
    return -float(np.clip(rng.normal(1.0, 0.45), 0.1, 2.0))


def _resolve_other_run(
    profile: RunGeometryOutcome,
    *,
    rng: np.random.Generator,
) -> RunAnatomy:
    """Resolve missing/unrecognized run geometry without inventing a known blocking lane.

    Kneels/spikes are excluded upstream. The remaining `other` class is overwhelmingly
    missing/unknown run location or gap, so it receives low-information empirical anatomy
    rather than ordinary interior/edge matchup assumptions.
    """

    negative_p = float(np.clip(profile.negative_rate, 0.0, 0.95))
    zero_p = float(np.clip(profile.zero_rate, 0.0, max(0.0, 0.98 - negative_p)))
    positive_p = max(1.0 - negative_p - zero_p, 0.0)

    negative = max(profile.negative_rate, 1e-9)
    loss_5 = float(np.clip(profile.loss_5_plus_rate / negative, 0.0, 1.0))
    loss_2 = float(np.clip(profile.loss_2_plus_rate / negative, 0.0, 1.0))
    moderate = max(loss_2 - loss_5, 0.0)
    shallow = max(1.0 - loss_2, 1e-6)
    positive_mean = max(min(profile.yards_p99, 2.0) * 0.65, 0.35)
    required_negative_mean = (
        positive_p * positive_mean - profile.yards_mean
    ) / max(negative_p, 1e-9)
    shallow_mean = (
        required_negative_mean - loss_5 * 5.5 - moderate * 2.25
    ) / shallow
    shallow_mean = float(np.clip(shallow_mean, 0.20, 1.80))

    draw = rng.random()
    if draw < negative_p:
        severity_draw = rng.random()
        if severity_draw < loss_5:
            yards = -float(rng.uniform(5.0, 6.0))
        elif severity_draw < loss_2:
            yards = -float(rng.uniform(2.0, 2.5))
        else:
            yards = -float(np.clip(rng.normal(shallow_mean, 0.16), 0.10, 1.99))
        return RunAnatomy(True, ContactResult.STUFF, yards, 0.0, yards)
    if draw < negative_p + zero_p:
        return RunAnatomy(True, ContactResult.STUFF, 0.0, 0.0, 0.0)

    high = max(min(profile.yards_p99 * 2.0 + 0.5, 4.0), 0.75)
    yards = float(rng.uniform(0.10, high))
    return RunAnatomy(False, ContactResult.TACKLED, yards, 0.0, yards)


def resolve_run_ecology(
    profile: RunGeometryOutcome,
    *,
    matchup_stuff_probability: float,
    matchup_yards_multiplier: float,
    runner_power: float,
    tackling: float,
    explosiveness: float,
    rng: np.random.Generator,
) -> RunAnatomy:
    """Resolve a run through explicit failure/routine/crease/breakaway branches.

    Frequency and severity are separate. Historical geometry provides the branch prior.
    Coarse front/player matchup evidence receives bounded relative authority: it can move
    a world above or below the historical prior without replacing the prior itself. Front
    penetration controls loss frequency; runner power can resist that penetration; the
    matchup yards multiplier owns ordinary efficiency; explosiveness and pursuit own
    breakaway frequency.
    """

    if profile.category == "other":
        return _resolve_other_run(profile, rng=rng)

    stuff_factor = _shrink_run_relative(
        matchup_stuff_probability / 0.18,
        low=0.55,
        high=1.80,
    )
    power_factor = _shrink_run_relative(runner_power, low=0.70, high=1.30)
    efficiency_factor = _shrink_run_relative(
        matchup_yards_multiplier,
        low=0.72,
        high=1.38,
    )
    explosive_factor = _shrink_run_relative(
        explosiveness / max(tackling**0.30, 0.80),
        low=0.65,
        high=1.45,
    )

    negative_p = float(
        np.clip(
            profile.negative_rate * stuff_factor / np.sqrt(power_factor),
            0.025,
            0.34,
        )
    )
    zero_p = float(
        np.clip(
            profile.zero_rate * stuff_factor**0.35 / power_factor**0.20,
            0.015,
            0.20,
        )
    )
    explosive_20_p = float(
        np.clip(profile.explosive_20_rate * explosive_factor, 0.0, 0.12)
    )
    explosive_15_p = float(
        np.clip(
            max(profile.explosive_15_rate - profile.explosive_20_rate, 0.0)
            * explosive_factor,
            0.0,
            0.12,
        )
    )
    explosive_10_p = float(
        np.clip(
            max(profile.explosive_10_rate - profile.explosive_15_rate, 0.0)
            * explosive_factor,
            0.0,
            0.18,
        )
    )
    branch_total = negative_p + zero_p + explosive_10_p + explosive_15_p + explosive_20_p
    if branch_total > 0.88:
        scale = 0.88 / branch_total
        negative_p *= scale
        zero_p *= scale
        explosive_10_p *= scale
        explosive_15_p *= scale
        explosive_20_p *= scale

    draw = rng.random()
    if draw < negative_p:
        yards = _sample_negative_run(profile, rng)
        return RunAnatomy(True, ContactResult.STUFF, yards, 0.0, yards)

    draw -= negative_p
    if draw < zero_p:
        return RunAnatomy(True, ContactResult.STUFF, 0.0, 0.0, 0.0)

    draw -= zero_p
    if draw < explosive_20_p:
        yards = float(
            np.clip(20.0 + rng.exponential(_breakaway_scale(profile)), 20.0, 75.0)
        )
        before = float(np.clip(rng.normal(5.0, 1.7), 2.0, min(12.0, yards)))
        return RunAnatomy(False, ContactResult.BROKEN_TACKLE, before, yards - before, yards)

    draw -= explosive_20_p
    if draw < explosive_15_p:
        yards = float(rng.uniform(15.0, 20.0))
        before = float(np.clip(rng.normal(4.8, 1.5), 1.5, min(10.0, yards)))
        return RunAnatomy(False, ContactResult.BROKEN_TACKLE, before, yards - before, yards)

    draw -= explosive_15_p
    if draw < explosive_10_p:
        yards = float(rng.uniform(10.0, 15.0))
        before = float(np.clip(rng.normal(4.5, 1.4), 1.0, min(9.0, yards)))
        return RunAnatomy(False, ContactResult.BROKEN_TACKLE, before, yards - before, yards)

    routine_sd = float(np.clip(profile.yards_sd * 0.45, 1.1, 3.0))
    neutral_routine_mean = _neutral_routine_mean(profile)
    live_routine_mean = float(
        np.clip(neutral_routine_mean * efficiency_factor, 0.11, 9.90)
    )
    routine_center = _center_for_truncated_mean(live_routine_mean, routine_sd)
    yards = _sample_truncated_normal(routine_center, routine_sd, rng)
    before = float(np.clip(rng.normal(min(3.2, yards), 1.0), 0.0, yards))
    after = yards - before
    contact = ContactResult.TACKLED if after <= 2.5 else ContactResult.BROKEN_TACKLE
    return RunAnatomy(False, contact, before, after, yards)
