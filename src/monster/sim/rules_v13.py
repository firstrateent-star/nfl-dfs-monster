from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import numpy as np


class PenaltySide(StrEnum):
    OFFENSE = "offense"
    DEFENSE = "defense"


class TryResult(StrEnum):
    PAT_GOOD = "pat_good"
    PAT_MISS = "pat_miss"
    TWO_POINT_GOOD = "two_point_good"
    TWO_POINT_FAIL = "two_point_fail"


@dataclass(frozen=True)
class PenaltyEvent:
    side: PenaltySide
    yards: int
    automatic_first_down: bool = False
    loss_of_down: bool = False


@dataclass(frozen=True)
class TryEvent:
    result: TryResult
    points: int


def simulate_penalty(rng: np.random.Generator, *, base_rate: float = 0.055) -> PenaltyEvent | None:
    if rng.random() >= base_rate:
        return None
    offense = rng.random() < 0.52
    if offense:
        yards = 5 if rng.random() < 0.62 else 10
        return PenaltyEvent(PenaltySide.OFFENSE, yards, loss_of_down=False)
    yards = 5 if rng.random() < 0.55 else 10
    automatic = rng.random() < 0.18
    return PenaltyEvent(PenaltySide.DEFENSE, yards, automatic_first_down=automatic)


def choose_two_point(*, quarter: int, seconds_remaining: int, score_margin_after_td: int) -> bool:
    if quarter < 4:
        return False
    if seconds_remaining > 600:
        return False
    return score_margin_after_td in {-8, -5, -2, 1, 5}


def simulate_try(
    rng: np.random.Generator,
    *,
    go_for_two: bool,
    kicking_skill: float = 1.0,
    offense_skill: float = 1.0,
    defense_skill: float = 1.0,
) -> TryEvent:
    if go_for_two:
        success = float(np.clip(0.48 * offense_skill / max(defense_skill, 0.65), 0.25, 0.70))
        good = rng.random() < success
        return TryEvent(TryResult.TWO_POINT_GOOD if good else TryResult.TWO_POINT_FAIL, 2 if good else 0)
    pat = float(np.clip(0.94 * kicking_skill, 0.78, 0.995))
    good = rng.random() < pat
    return TryEvent(TryResult.PAT_GOOD if good else TryResult.PAT_MISS, 1 if good else 0)


def is_safety(*, yardline_100: float, yards: float) -> bool:
    return yardline_100 + yards <= 0.0


def overtime_required(away_score: int, home_score: int) -> bool:
    return away_score == home_score
