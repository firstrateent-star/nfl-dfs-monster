from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import numpy as np

from monster.sim.chaos_ecology import (
    DEFAULT_CHAOS_ECOLOGY,
    ChaosEcology,
    sample_return_yards,
)


class SpecialTeamsType(StrEnum):
    KICKOFF = "kickoff"
    PUNT = "punt"
    FIELD_GOAL = "field_goal"
    PAT = "pat"


@dataclass(frozen=True)
class SpecialTeamsEvent:
    event_type: SpecialTeamsType
    kick_distance: float
    return_yards: float = 0.0
    touchback: bool = False
    blocked: bool = False
    made: bool | None = None
    returner_id: str | None = None
    kicker_id: str | None = None
    punter_id: str | None = None
    return_start_yardline_100: float | None = None
    fair_catch: bool = False
    muffed: bool = False
    kicking_team_recovery: bool = False
    return_touchdown: bool = False


def simulate_kickoff(
    rng: np.random.Generator,
    *,
    returner_id: str | None = None,
    kicker_id: str | None = None,
    return_skill: float = 1.0,
    ecology: ChaosEcology = DEFAULT_CHAOS_ECOLOGY,
) -> SpecialTeamsEvent:
    """Resolve the 2026 dynamic kickoff into touchback/landing/return branches."""
    touchback = rng.random() < ecology.kickoff_touchback_rate
    if touchback:
        return SpecialTeamsEvent(
            SpecialTeamsType.KICKOFF,
            kick_distance=65.0,
            touchback=True,
            returner_id=returner_id,
            kicker_id=kicker_id,
        )

    # Dynamic-kickoff returns originate in the 0-20 yard landing zone. A beta draw preserves
    # ordinary deep kicks while allowing shorter strategic placements without inventing a
    # fixed receiving spot.
    landing = float(np.clip(20.0 * rng.beta(2.0, 3.0), 0.0, 20.0))
    muffed = rng.random() < ecology.kickoff_muff_rate
    kicking_recovery = muffed and rng.random() < ecology.kickoff_muff_kicking_recovery_rate
    return_yards = 0.0
    if not muffed:
        return_yards = sample_return_yards(
            mean=ecology.kickoff_return_mean,
            sd=ecology.kickoff_return_sd,
            zero_rate=0.0,
            forty_plus_rate=ecology.kickoff_40_plus_rate,
            return_skill=return_skill,
            rng=rng,
            maximum=100.0 - landing,
        )
    return SpecialTeamsEvent(
        SpecialTeamsType.KICKOFF,
        kick_distance=65.0,
        return_yards=return_yards,
        touchback=False,
        returner_id=returner_id,
        kicker_id=kicker_id,
        return_start_yardline_100=landing,
        muffed=muffed,
        kicking_team_recovery=kicking_recovery,
    )


def simulate_punt(
    rng: np.random.Generator,
    *,
    punter_skill: float = 1.0,
    returner_id: str | None = None,
    punter_id: str | None = None,
    return_skill: float = 1.0,
    ecology: ChaosEcology = DEFAULT_CHAOS_ECOLOGY,
) -> SpecialTeamsEvent:
    blocked = rng.random() < ecology.blocked_punt_rate
    gross = 0.0 if blocked else float(np.clip(rng.normal(45.0 * punter_skill, 6.0), 20.0, 70.0))
    touchback = not blocked and rng.random() < 0.08
    if blocked or touchback:
        return SpecialTeamsEvent(
            SpecialTeamsType.PUNT,
            kick_distance=gross,
            touchback=touchback,
            blocked=blocked,
            returner_id=returner_id,
            punter_id=punter_id,
        )

    muffed = rng.random() < ecology.punt_muff_rate
    kicking_recovery = muffed and rng.random() < ecology.punt_muff_kicking_recovery_rate
    ret = 0.0
    if not muffed:
        ret = sample_return_yards(
            mean=ecology.punt_return_mean,
            sd=ecology.punt_return_sd,
            zero_rate=ecology.punt_zero_return_rate,
            forty_plus_rate=ecology.punt_40_plus_rate,
            return_skill=return_skill,
            rng=rng,
            maximum=100.0,
        )
    return SpecialTeamsEvent(
        SpecialTeamsType.PUNT,
        kick_distance=gross,
        return_yards=ret,
        touchback=False,
        blocked=False,
        returner_id=returner_id,
        punter_id=punter_id,
        fair_catch=not muffed and ret <= 1e-9,
        muffed=muffed,
        kicking_team_recovery=kicking_recovery,
    )


def simulate_field_goal(
    rng: np.random.Generator,
    *,
    distance: float,
    kicking_skill: float = 1.0,
    kicker_id: str | None = None,
    ecology: ChaosEcology = DEFAULT_CHAOS_ECOLOGY,
) -> SpecialTeamsEvent:
    # Modern NFL kickers convert the ordinary attempt mix at a high rate. Keep
    # distance as the primary mechanism and let certified specialist evidence
    # make only a bounded multiplicative adjustment around that curve.
    blocked = rng.random() < ecology.blocked_field_goal_rate
    if distance <= 29.0:
        base_make = 0.985
    elif distance <= 39.0:
        base_make = 0.955
    elif distance <= 49.0:
        base_make = 0.885
    elif distance <= 59.0:
        base_make = 0.735
    else:
        base_make = 0.48
    skill_adjustment = float(np.clip(kicking_skill, 0.92, 1.08))
    made = False if blocked else rng.random() < float(np.clip(base_make * skill_adjustment, 0.05, 0.995))
    return SpecialTeamsEvent(
        SpecialTeamsType.FIELD_GOAL,
        kick_distance=distance,
        blocked=blocked,
        made=made,
        kicker_id=kicker_id,
    )
