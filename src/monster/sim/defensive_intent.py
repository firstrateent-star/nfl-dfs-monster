from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import numpy as np


class CoverageShell(StrEnum):
    MAN = "man"
    SINGLE_HIGH = "single_high"
    TWO_HIGH = "two_high"
    SOFT_ZONE = "soft_zone"


class RushPlan(StrEnum):
    FOUR = "rush_four"
    BLITZ = "blitz"
    CONTAIN = "contain"
    SIMULATED_PRESSURE = "simulated_pressure"


@dataclass(frozen=True)
class DefensiveTacticalPrior:
    """Market-blind tactical evidence for one defensive unit.

    This is an evidence container, not production authority. Values should be estimated from
    historical defensive behavior or validated scouting evidence before promotion.
    """

    man_rate: float = 0.25
    single_high_rate: float = 0.25
    two_high_rate: float = 0.30
    soft_zone_rate: float = 0.20
    blitz_rate: float = 0.20
    simulated_pressure_rate: float = 0.08
    contain_rate: float = 0.08


@dataclass(frozen=True)
class DefensiveIntent:
    coverage_shell: CoverageShell
    rush_plan: RushPlan
    box_aggression: float
    authority: float = 0.0


def _normalized(values: tuple[float, ...]) -> np.ndarray:
    arr = np.clip(np.asarray(values, dtype=float), 0.0, None)
    total = float(arr.sum())
    if total <= 0.0:
        return np.full(len(arr), 1.0 / len(arr))
    return arr / total


def sample_defensive_intent(
    prior: DefensiveTacticalPrior,
    *,
    rng: np.random.Generator,
    short_yardage: bool = False,
    late_lead: bool = False,
    qb_run_threat: float = 0.0,
) -> DefensiveIntent:
    """Sample a shadow defensive call without reading the offense's realized play call.

    Context may alter tactical probabilities, but this function intentionally has no access to
    the offense's sampled RUN/PASS decision. That preserves simultaneous strategic uncertainty.
    Authority remains zero until historical tactical fields and OOS behavior are certified.
    """
    shells = tuple(CoverageShell)
    shell_weights = [
        prior.man_rate,
        prior.single_high_rate,
        prior.two_high_rate,
        prior.soft_zone_rate,
    ]
    if late_lead:
        shell_weights[shells.index(CoverageShell.TWO_HIGH)] *= 1.25
        shell_weights[shells.index(CoverageShell.SOFT_ZONE)] *= 1.35
        shell_weights[shells.index(CoverageShell.MAN)] *= 0.80
    shell = shells[int(rng.choice(len(shells), p=_normalized(tuple(shell_weights))))]

    blitz = max(prior.blitz_rate * (0.85 if late_lead else 1.0), 0.0)
    simulated = max(prior.simulated_pressure_rate, 0.0)
    contain = max(prior.contain_rate * (1.0 + 0.8 * np.clip(qb_run_threat, 0.0, 1.0)), 0.0)
    rush_four = max(1.0 - blitz - simulated - contain, 0.0)
    plans = tuple(RushPlan)
    plan_map = {
        RushPlan.FOUR: rush_four,
        RushPlan.BLITZ: blitz,
        RushPlan.CONTAIN: contain,
        RushPlan.SIMULATED_PRESSURE: simulated,
    }
    rush_plan = plans[int(rng.choice(len(plans), p=_normalized(tuple(plan_map[p] for p in plans))))]

    box_aggression = 0.50
    if short_yardage:
        box_aggression += 0.30
    if shell in {CoverageShell.TWO_HIGH, CoverageShell.SOFT_ZONE}:
        box_aggression -= 0.10
    if rush_plan == RushPlan.BLITZ:
        box_aggression += 0.12
    if rush_plan == RushPlan.CONTAIN:
        box_aggression -= 0.04

    return DefensiveIntent(
        coverage_shell=shell,
        rush_plan=rush_plan,
        box_aggression=float(np.clip(box_aggression, 0.0, 1.0)),
        authority=0.0,
    )
