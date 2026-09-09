from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import numpy as np

# 2025 regular-season FTN participation via nflverse, measured on qb_dropback plays.
# These are causal branch baselines, not desired final-stat targets.
PRESSURED_SACK_RATE = 0.217914
PRESSURED_SCRAMBLE_RATE = 0.069760
CLEAN_SCRAMBLE_RATE = 0.048621


class QBResponse(StrEnum):
    THROW = "throw"
    SCRAMBLE = "scramble"
    SACK = "sack"


class CatchpointResult(StrEnum):
    CATCH = "catch"
    DROP = "drop"
    BREAKUP = "breakup"
    INTERCEPTION = "interception"


class ContactResult(StrEnum):
    CLEAN = "clean"
    STUFF = "stuff"
    TACKLED = "tackled"
    BROKEN_TACKLE = "broken_tackle"


@dataclass(frozen=True)
class PassAnatomy:
    pressured: bool
    qb_response: QBResponse
    air_yards: float
    catchpoint: CatchpointResult | None
    yards_after_catch: float
    total_yards: float


@dataclass(frozen=True)
class RunAnatomy:
    penetration: bool
    contact: ContactResult
    yards_before_contact: float
    yards_after_contact: float
    total_yards: float


def resolve_qb_response(
    *,
    pressured: bool,
    mobility: float,
    pocket_skill: float,
    rng: np.random.Generator,
) -> QBResponse:
    """Resolve a dropback after the pressure state is known.

    The league branch probabilities come from 2025 play-level pressure participation.
    Existing bounded QB/team traits only perturb those baselines; they do not replace
    the empirical causal anatomy. Scrambles can occur from both pressured and clean
    dropbacks, matching the historical definition of a dropback family.
    """
    mobility_factor = float(np.clip(mobility, 0.70, 1.30))
    draw = rng.random()

    if not pressured:
        scramble = float(np.clip(CLEAN_SCRAMBLE_RATE * mobility_factor, 0.01, 0.15))
        return QBResponse.SCRAMBLE if draw < scramble else QBResponse.THROW

    pocket_factor = float(np.clip(1.0 / max(pocket_skill, 0.60), 0.75, 1.35))
    sack = float(np.clip(PRESSURED_SACK_RATE * pocket_factor, 0.10, 0.40))
    scramble = float(np.clip(PRESSURED_SCRAMBLE_RATE * mobility_factor, 0.02, 0.18))
    if draw < sack:
        return QBResponse.SACK
    if draw < sack + scramble:
        return QBResponse.SCRAMBLE
    return QBResponse.THROW


def resolve_catchpoint(
    *,
    catch_skill: float,
    coverage_strength: float,
    ball_hawk: float,
    air_yards: float,
    rng: np.random.Generator,
) -> CatchpointResult:
    depth_penalty = max(air_yards - 10.0, 0.0) * 0.008
    interception_p = float(np.clip(0.018 * ball_hawk * coverage_strength, 0.004, 0.08))
    catch_p = float(
        np.clip(0.70 * catch_skill / max(coverage_strength, 0.60) - depth_penalty, 0.28, 0.86)
    )
    draw = rng.random()
    if draw < interception_p:
        return CatchpointResult.INTERCEPTION
    if draw < interception_p + catch_p:
        return CatchpointResult.CATCH
    drop_share = float(np.clip(0.18 / max(catch_skill, 0.65), 0.08, 0.30))
    return CatchpointResult.DROP if rng.random() < drop_share else CatchpointResult.BREAKUP


def resolve_run_contact(
    *,
    penetration_probability: float,
    runner_power: float,
    tackling: float,
    explosiveness: float,
    rng: np.random.Generator,
) -> RunAnatomy:
    penetration = rng.random() < penetration_probability
    if penetration:
        before = float(np.clip(rng.normal(0.0, 1.1), -4.0, 2.0))
        break_p = float(np.clip(0.12 * runner_power / max(tackling, 0.60), 0.03, 0.32))
        if rng.random() >= break_p:
            return RunAnatomy(penetration, ContactResult.STUFF, before, 0.0, before)
    else:
        before = float(np.clip(rng.normal(3.1, 1.8), 0.0, 12.0))

    break_p = float(np.clip(0.22 * runner_power / max(tackling, 0.60), 0.06, 0.46))
    broken = rng.random() < break_p
    if broken:
        after = float(np.clip(rng.lognormal(1.25, 0.65) * explosiveness, 0.5, 55.0))
        contact = ContactResult.BROKEN_TACKLE
    else:
        after = float(np.clip(rng.normal(1.0, 0.8), 0.0, 4.0))
        contact = ContactResult.TACKLED
    return RunAnatomy(penetration, contact, before, after, before + after)