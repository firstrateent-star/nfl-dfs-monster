from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import numpy as np

# 2025 regular-season FTN participation via nflverse, measured on qb_dropback plays.
# These are causal branch baselines, not desired final-stat targets.
PRESSURED_SACK_RATE = 0.217914
PRESSURED_SCRAMBLE_RATE = 0.069760
CLEAN_SCRAMBLE_RATE = 0.048621

# Among actual throws in the same 2025 play-level pressure sample:
# overall completion=64.2661%, pressured=45.9472%, clean=70.0838%;
# overall interception=2.1772%, pressured=2.8999%, clean=1.9476%.
# Ratios to the overall throw baseline let pressure condition a matchup-owned probability
# without changing that matchup's unconditional calibration target.
PRESSURED_COMPLETION_MULTIPLIER = 0.45947230805799855 / 0.6426607081471296
CLEAN_COMPLETION_MULTIPLIER = 0.7008379255680531 / 0.6426607081471296
PRESSURED_INTERCEPTION_MULTIPLIER = 0.028999286902781078 / 0.021771513693136242
CLEAN_INTERCEPTION_MULTIPLIER = 0.019476107797992 / 0.021771513693136242


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


def condition_throw_probabilities(
    *,
    completion_probability: float,
    interception_probability: float,
    pressured: bool,
) -> tuple[float, float]:
    """Condition matchup-owned throw quality on the already-resolved pressure state.

    The multipliers are ratios to the 2025 overall throw baseline. With the observed
    pressure mix they preserve the unconditional baseline while creating the large,
    measured clean-vs-pressure split. This keeps pressure causal rather than applying
    a second global completion calibration downstream.
    """
    if pressured:
        completion_multiplier = PRESSURED_COMPLETION_MULTIPLIER
        interception_multiplier = PRESSURED_INTERCEPTION_MULTIPLIER
    else:
        completion_multiplier = CLEAN_COMPLETION_MULTIPLIER
        interception_multiplier = CLEAN_INTERCEPTION_MULTIPLIER
    completion = float(
        np.clip(completion_probability * completion_multiplier, 0.12, 0.92)
    )
    interception = float(
        np.clip(interception_probability * interception_multiplier, 0.001, 0.12)
    )
    return completion, interception


def resolve_catchpoint(
    *,
    catch_skill: float,
    coverage_strength: float,
    ball_hawk: float,
    air_yards: float,
    rng: np.random.Generator,
    completion_probability: float | None = None,
    interception_probability: float | None = None,
    completion_probability_includes_depth: bool = False,
) -> CatchpointResult:
    """Resolve a targeted throw from one coherent matchup probability path.

    When the matchup kernel supplies completion/interception probabilities, those
    probabilities already contain QB, target and defensive-unit interaction. Stage 3
    depth-aware callers may additionally declare that completion probability already
    contains throw-depth difficulty; in that path the legacy generic depth penalty is
    disabled so depth enters exactly once. Catch skill still governs the drop/breakup
    split after an incompletion reservoir is established.
    """
    depth_penalty = (
        0.0
        if completion_probability_includes_depth
        else max(air_yards - 10.0, 0.0) * 0.008
    )
    if interception_probability is None:
        interception_p = float(
            np.clip(0.018 * ball_hawk * coverage_strength, 0.004, 0.08)
        )
    else:
        interception_p = float(np.clip(interception_probability, 0.001, 0.10))

    if completion_probability is None:
        catch_p = float(
            np.clip(
                0.70 * catch_skill / max(coverage_strength, 0.60) - depth_penalty,
                0.28,
                0.86,
            )
        )
    else:
        catch_p = float(
            np.clip(completion_probability - depth_penalty, 0.20, 0.88)
        )

    # Interception and completion are mutually exclusive outcomes from the same throw.
    # Preserve at least a small incompletion reservoir even under extreme inputs.
    catch_p = min(catch_p, max(0.0, 0.98 - interception_p))
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
    """Resolve contact while preserving a real open-field positive tail.

    This helper is used by the non-geometry fallback and, importantly, by QB scrambles.
    The former implementation produced the correct probability of a useful scramble but
    compressed nearly every successful escape into 3-6 yards because every non-penetrated
    run was forced into immediate contact around three yards. NFL scrambles have a distinct
    open-field topology: once the QB clears the rush there can be meaningful space before the
    first tackle attempt. We therefore separate penetration, ordinary pursuit, clean-lane and
    breakaway branches. ``explosiveness`` (which now carries Madden speed/mobility evidence)
    and defender tackling govern movement among those branches rather than directly adding
    fantasy yards.
    """
    runner = float(np.clip(runner_power, 0.60, 1.45))
    pursuit = float(np.clip(tackling, 0.60, 1.55))
    burst = float(np.clip(explosiveness, 0.60, 1.55))

    penetration = rng.random() < penetration_probability
    if penetration:
        before = float(np.clip(rng.normal(-0.10, 1.20), -5.0, 2.0))
        escape_p = float(np.clip(0.14 * runner * burst / pursuit, 0.035, 0.38))
        if rng.random() >= escape_p:
            return RunAnatomy(True, ContactResult.STUFF, before, 0.0, before)
        # Escaping a penetrated pocket should not immediately guarantee an explosive play.
        before = float(np.clip(rng.normal(2.8 * burst, 1.35), 0.0, 7.0))

    else:
        # Roughly one quarter of league scrambles reach 10+ yards and about one tenth reach
        # 15+. These are causal open-space branches, not final-stat targets: player burst and
        # pursuit shift their mass while the broad branch structure supplies the missing NFL
        # topology. The remaining mass resolves through ordinary pursuit around 3-8 yards.
        open_field_factor = float(np.clip(burst / pursuit, 0.55, 1.75))
        breakaway_p = float(np.clip(0.105 * open_field_factor, 0.035, 0.22))
        clear_lane_p = float(np.clip(0.165 * open_field_factor, 0.06, 0.30))
        lane_draw = rng.random()
        if lane_draw < breakaway_p:
            before = float(np.clip(14.0 + rng.exponential(5.5 * burst), 11.0, 45.0))
            after = float(np.clip(rng.lognormal(0.65, 0.50) * burst / pursuit, 0.0, 12.0))
            return RunAnatomy(
                False,
                ContactResult.BROKEN_TACKLE,
                before,
                after,
                before + after,
            )
        if lane_draw < breakaway_p + clear_lane_p:
            before = float(np.clip(rng.normal(8.0 * burst, 2.2), 4.5, 15.0))
        else:
            before = float(np.clip(rng.normal(4.0 * burst, 1.55), 0.0, 9.0))

    break_p = float(np.clip(0.20 * runner * burst / pursuit, 0.055, 0.46))
    broken = rng.random() < break_p
    if broken:
        after = float(
            np.clip(rng.lognormal(1.10, 0.68) * burst / max(pursuit**0.35, 0.80), 0.5, 42.0)
        )
        contact = ContactResult.BROKEN_TACKLE
    else:
        after = float(np.clip(rng.normal(0.85, 0.72), 0.0, 3.5))
        contact = ContactResult.TACKLED
    return RunAnatomy(penetration, contact, before, after, before + after)
