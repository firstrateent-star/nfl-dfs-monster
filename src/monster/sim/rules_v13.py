from __future__ import annotations

from dataclasses import dataclass, replace
from enum import StrEnum

import numpy as np

from monster.sim.football_state import FootballState, next_series_distance


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


def enforce_penalty(state: FootballState, penalty: PenaltyEvent, elapsed_seconds: int = 0) -> FootballState:
    """Apply a simplified accepted live-ball penalty to pre-snap state.

    The v1.3 first penalty layer intentionally models aggregate accepted penalties rather than
    pretending to know a foul subtype. Offensive penalties move the offense backward and replay
    the down unless a future evidence-backed subtype carries loss of down. Defensive penalties
    move the offense forward; automatic-first-down events and penalties reaching the line to gain
    start a new series. Clock runoff is explicit and bounded by the event caller.
    """
    clock = max(state.seconds_remaining - max(int(elapsed_seconds), 0), 0)
    if penalty.side == PenaltySide.OFFENSE:
        enforced = min(float(penalty.yards), max(state.yardline_100 - 1.0, 0.0))
        yardline = max(state.yardline_100 - enforced, 1.0)
        down = min(state.down + int(penalty.loss_of_down), 4)
        distance = max(state.distance + enforced, 1.0)
    else:
        enforced = min(float(penalty.yards), max(99.0 - state.yardline_100, 0.0))
        yardline = min(state.yardline_100 + enforced, 99.0)
        first_down = penalty.automatic_first_down or enforced >= state.distance
        down = 1 if first_down else state.down
        distance = next_series_distance(yardline) if first_down else max(state.distance - enforced, 1.0)
    return replace(state, seconds_remaining=clock, yardline_100=yardline, down=down, distance=distance)


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
