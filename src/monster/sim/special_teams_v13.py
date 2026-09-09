from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import numpy as np


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


def simulate_kickoff(
    rng: np.random.Generator,
    *,
    returner_id: str | None = None,
    kicker_id: str | None = None,
) -> SpecialTeamsEvent:
    touchback = rng.random() < 0.67
    return_yards = 0.0 if touchback else float(np.clip(rng.normal(24.0, 7.0), 0.0, 75.0))
    return SpecialTeamsEvent(
        SpecialTeamsType.KICKOFF,
        kick_distance=65.0,
        return_yards=return_yards,
        touchback=touchback,
        returner_id=returner_id,
        kicker_id=kicker_id,
    )


def simulate_punt(
    rng: np.random.Generator,
    *,
    punter_skill: float = 1.0,
    returner_id: str | None = None,
    punter_id: str | None = None,
) -> SpecialTeamsEvent:
    blocked = rng.random() < 0.012
    gross = 0.0 if blocked else float(np.clip(rng.normal(45.0 * punter_skill, 6.0), 20.0, 70.0))
    touchback = not blocked and rng.random() < 0.08
    ret = 0.0 if blocked or touchback else float(np.clip(rng.normal(8.5, 7.0), 0.0, 80.0))
    return SpecialTeamsEvent(
        SpecialTeamsType.PUNT,
        kick_distance=gross,
        return_yards=ret,
        touchback=touchback,
        blocked=blocked,
        returner_id=returner_id,
        punter_id=punter_id,
    )


def simulate_field_goal(
    rng: np.random.Generator,
    *,
    distance: float,
    kicking_skill: float = 1.0,
    kicker_id: str | None = None,
) -> SpecialTeamsEvent:
    blocked = rng.random() < 0.015
    make_p = float(np.clip(0.99 - max(distance - 25.0, 0.0) * 0.011, 0.10, 0.99))
    made = False if blocked else rng.random() < float(np.clip(make_p * kicking_skill, 0.05, 0.995))
    return SpecialTeamsEvent(
        SpecialTeamsType.FIELD_GOAL,
        kick_distance=distance,
        blocked=blocked,
        made=made,
        kicker_id=kicker_id,
    )
