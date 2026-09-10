from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from monster.sim.intent_ecology import PassDepthOutcome, RunGeometryOutcome
from monster.sim.play_anatomy import (
    ContactResult,
    RunAnatomy,
    condition_throw_probabilities,
)


@dataclass(frozen=True)
class DepthThrowProbabilities:
    completion: float
    interception: float


def depth_throw_probabilities(
    profile: PassDepthOutcome,
    *,
    matchup_completion_probability: float | None,
    matchup_interception_probability: float | None,
    pressured: bool,
) -> DepthThrowProbabilities:
    """Combine depth difficulty with bounded QB/target/coverage and pressure interaction.

    Historical depth outcome is the causal prior. The existing matchup path contributes
    only a relative skill/coverage modifier, avoiding a second global completion model.
    Pressure then conditions the already depth-aware probability through the measured
    clean-vs-pressure split.
    """

    completion = profile.completion_rate
    interception = profile.interception_rate
    if matchup_completion_probability is not None:
        relative = float(
            np.clip(matchup_completion_probability / 0.64, 0.72, 1.30)
        )
        completion *= relative
    if matchup_interception_probability is not None:
        relative_int = float(
            np.clip(matchup_interception_probability / 0.022, 0.55, 1.70)
        )
        interception *= relative_int
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


def sample_yac(
    profile: PassDepthOutcome,
    *,
    receiver_explosiveness: float,
    coverage_strength: float,
    rng: np.random.Generator,
) -> float:
    """Sample non-negative YAC with depth-specific shape and bounded pursuit interaction."""

    mean = max(profile.yac_mean_completed, 0.05)
    sd = max(profile.yac_sd_completed, 0.35)
    shape = max((mean / sd) ** 2, 0.20)
    scale = max((sd**2) / mean, 0.05)
    base = float(rng.gamma(shape, scale))
    interaction = float(
        np.clip(
            receiver_explosiveness / max(coverage_strength**0.30, 0.80),
            0.72,
            1.35,
        )
    )
    return float(np.clip(base * interaction, 0.0, 65.0))


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
    Front penetration controls loss frequency; runner power can resist that penetration;
    the matchup yards multiplier owns ordinary efficiency; explosiveness and pursuit own
    the upper tail. Each trait therefore has one primary causal jurisdiction instead of
    being multiplied repeatedly across the same outcome path.
    """

    stuff_factor = float(
        np.clip(matchup_stuff_probability / 0.18, 0.55, 1.80)
    )
    power_factor = float(np.clip(runner_power, 0.70, 1.30))
    efficiency_factor = float(
        np.clip(matchup_yards_multiplier, 0.72, 1.38)
    )
    explosive_factor = float(
        np.clip(
            explosiveness / max(tackling**0.30, 0.80),
            0.65,
            1.45,
        )
    )

    negative_p = float(
        np.clip(
            profile.negative_rate * stuff_factor / np.sqrt(power_factor),
            0.025,
            0.24,
        )
    )
    zero_p = float(
        np.clip(
            profile.zero_rate
            * stuff_factor**0.35
            / power_factor**0.20,
            0.015,
            0.16,
        )
    )
    explosive_20_p = float(
        np.clip(profile.explosive_20_rate * explosive_factor, 0.001, 0.12)
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
    branch_total = (
        negative_p
        + zero_p
        + explosive_10_p
        + explosive_15_p
        + explosive_20_p
    )
    if branch_total > 0.82:
        scale = 0.82 / branch_total
        negative_p *= scale
        zero_p *= scale
        explosive_10_p *= scale
        explosive_15_p *= scale
        explosive_20_p *= scale

    draw = rng.random()
    if draw < negative_p:
        conditional_loss_5 = (
            profile.loss_5_plus_rate / max(profile.negative_rate, 1e-9)
        )
        conditional_loss_2 = (
            profile.loss_2_plus_rate / max(profile.negative_rate, 1e-9)
        )
        severity_draw = rng.random()
        if severity_draw < conditional_loss_5:
            yards = -float(
                np.clip(rng.lognormal(1.70, 0.35), 5.0, 14.0)
            )
        elif severity_draw < conditional_loss_2:
            yards = -float(np.clip(rng.normal(3.0, 0.8), 2.0, 5.0))
        else:
            yards = -float(np.clip(rng.normal(1.0, 0.45), 0.1, 2.0))
        return RunAnatomy(True, ContactResult.STUFF, yards, 0.0, yards)

    draw -= negative_p
    if draw < zero_p:
        return RunAnatomy(True, ContactResult.STUFF, 0.0, 0.0, 0.0)

    draw -= zero_p
    if draw < explosive_20_p:
        center = max(profile.yards_p99, 24.0)
        yards = float(
            np.clip(rng.lognormal(np.log(center), 0.38), 20.0, 75.0)
        )
        before = float(
            np.clip(rng.normal(5.0, 1.7), 2.0, min(12.0, yards))
        )
        return RunAnatomy(
            False,
            ContactResult.BROKEN_TACKLE,
            before,
            yards - before,
            yards,
        )

    draw -= explosive_20_p
    if draw < explosive_15_p:
        yards = float(rng.uniform(15.0, 20.0))
        before = float(
            np.clip(rng.normal(4.8, 1.5), 1.5, min(10.0, yards))
        )
        return RunAnatomy(
            False,
            ContactResult.BROKEN_TACKLE,
            before,
            yards - before,
            yards,
        )

    draw -= explosive_15_p
    if draw < explosive_10_p:
        yards = float(rng.uniform(10.0, 15.0))
        before = float(
            np.clip(rng.normal(4.5, 1.4), 1.0, min(9.0, yards))
        )
        return RunAnatomy(
            False,
            ContactResult.BROKEN_TACKLE,
            before,
            yards - before,
            yards,
        )

    routine_center = float(
        np.clip(profile.yards_p50 * efficiency_factor, 1.5, 6.5)
    )
    routine_sd = float(np.clip(profile.yards_sd * 0.45, 1.1, 3.0))
    yards = float(np.clip(rng.normal(routine_center, routine_sd), 0.1, 9.999))
    before = float(
        np.clip(rng.normal(min(3.2, yards), 1.0), 0.0, yards)
    )
    after = yards - before
    contact = (
        ContactResult.TACKLED
        if after <= 2.5
        else ContactResult.BROKEN_TACKLE
    )
    return RunAnatomy(False, contact, before, after, yards)
