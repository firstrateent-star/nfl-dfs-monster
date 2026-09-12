from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import numpy as np

from monster.sim.clock import seconds_remaining_in_quarter
from monster.sim.football_state import FootballState


class FourthDownDecision(StrEnum):
    GO = "go"
    FIELD_GOAL = "field_goal"
    PUNT = "punt"


@dataclass(frozen=True)
class FourthDownProbabilities:
    go: float
    field_goal: float
    punt: float


@dataclass(frozen=True)
class SituationPolicy:
    pass_probability: float
    hurry_probability: float
    fourth_down: FourthDownDecision | None = None


def _late_game_pressure(state: FootballState) -> float:
    q_clock = seconds_remaining_in_quarter(state.seconds_remaining)
    if state.quarter < 4 or q_clock > 300:
        return 0.0
    margin = state.score_margin_for_offense
    if margin >= 0:
        return 0.0
    return float(np.clip((-margin) / 14.0, 0.0, 1.0))


def _game_script_pass_shift(state: FootballState) -> float:
    """Bounded score/time play-calling response, independent of market data."""
    margin = state.score_margin_for_offense
    if margin == 0 or state.quarter <= 1:
        return 0.0
    game_progress = float(np.clip((state.quarter - 1) / 3.0, 0.0, 1.0))
    magnitude = float(np.clip(abs(margin) / 14.0, 0.0, 1.0))
    direction = -1.0 if margin > 0 else 1.0
    return direction * 0.10 * magnitude * game_progress


def _fourth_down_zone(state: FootballState) -> str:
    if state.yardline_100 < 40.0:
        return "own_1_39"
    if state.yardline_100 < 55.0:
        return "own_40_to_midfield"
    if state.yardline_100 < 70.0:
        return "plus_45_to_31"
    if state.yardline_100 < 80.0:
        return "plus_30_to_21"
    return "red_zone"


def _fourth_down_distance(state: FootballState) -> str:
    if state.distance <= 1.0:
        return "1"
    if state.distance <= 3.0:
        return "2_3"
    if state.distance <= 6.0:
        return "4_6"
    return "7_plus"


# 2025 regular-season decision anatomy from definition-matched nflverse fourth-down states.
# These are football-decision priors only: no score, spread, total, salary or ownership data.
_FOURTH_DOWN_PRIORS: dict[tuple[str, str], FourthDownProbabilities] = {
    ("own_1_39", "1"): FourthDownProbabilities(0.386, 0.000, 0.614),
    ("own_1_39", "2_3"): FourthDownProbabilities(0.084, 0.000, 0.916),
    ("own_1_39", "4_6"): FourthDownProbabilities(0.064, 0.000, 0.936),
    ("own_1_39", "7_plus"): FourthDownProbabilities(0.045, 0.000, 0.955),
    ("own_40_to_midfield", "1"): FourthDownProbabilities(0.850, 0.009, 0.141),
    ("own_40_to_midfield", "2_3"): FourthDownProbabilities(0.356, 0.000, 0.644),
    ("own_40_to_midfield", "4_6"): FourthDownProbabilities(0.190, 0.005, 0.805),
    ("own_40_to_midfield", "7_plus"): FourthDownProbabilities(0.093, 0.003, 0.904),
    ("plus_45_to_31", "1"): FourthDownProbabilities(0.944, 0.056, 0.000),
    ("plus_45_to_31", "2_3"): FourthDownProbabilities(0.708, 0.270, 0.022),
    ("plus_45_to_31", "4_6"): FourthDownProbabilities(0.384, 0.403, 0.213),
    ("plus_45_to_31", "7_plus"): FourthDownProbabilities(0.122, 0.519, 0.359),
    ("plus_30_to_21", "1"): FourthDownProbabilities(0.842, 0.158, 0.000),
    ("plus_30_to_21", "2_3"): FourthDownProbabilities(0.536, 0.449, 0.015),
    ("plus_30_to_21", "4_6"): FourthDownProbabilities(0.168, 0.832, 0.000),
    ("plus_30_to_21", "7_plus"): FourthDownProbabilities(0.057, 0.943, 0.000),
    ("red_zone", "1"): FourthDownProbabilities(0.943, 0.057, 0.000),
    ("red_zone", "2_3"): FourthDownProbabilities(0.523, 0.477, 0.000),
    ("red_zone", "4_6"): FourthDownProbabilities(0.175, 0.825, 0.000),
    ("red_zone", "7_plus"): FourthDownProbabilities(0.097, 0.903, 0.000),
}


def fourth_down_probabilities(state: FootballState) -> FourthDownProbabilities:
    """Return market-blind fourth-down decision probabilities from football state.

    Ordinary states use coarse 2025 league decision anatomy. Late trailing and overtime
    response possessions remain governed by explicit game-theory constraints rather than the
    pooled historical prior.
    """
    if state.down != 4:
        raise ValueError("fourth_down_probabilities requires fourth down")

    yards_to_goal = 100.0 - state.yardline_100
    margin = state.score_margin_for_offense
    q_clock = seconds_remaining_in_quarter(state.seconds_remaining)

    if state.quarter == 5 and margin < 0:
        if margin >= -3 and state.yardline_100 >= 58.0 and yards_to_goal <= 42.0:
            return FourthDownProbabilities(0.0, 1.0, 0.0)
        return FourthDownProbabilities(1.0, 0.0, 0.0)

    desperate = state.quarter >= 4 and q_clock <= 360 and margin < 0
    if desperate:
        if state.distance <= 8.0:
            return FourthDownProbabilities(1.0, 0.0, 0.0)
        # Very long desperation downs should still overwhelmingly preserve possession.
        return FourthDownProbabilities(0.92, 0.04, 0.04)

    return _FOURTH_DOWN_PRIORS[(_fourth_down_zone(state), _fourth_down_distance(state))]


def sample_fourth_down_decision(
    state: FootballState,
    rng: np.random.Generator,
) -> FourthDownDecision:
    """Sample the empirical fourth-down choice using the smallest RNG contract possible."""
    probabilities = fourth_down_probabilities(state)
    total = probabilities.go + probabilities.field_goal + probabilities.punt
    if total <= 0.0:
        raise ValueError("fourth-down probabilities must contain positive mass")
    go = probabilities.go / total
    field_goal = probabilities.field_goal / total
    draw = float(rng.random())
    if draw < go:
        return FourthDownDecision.GO
    if draw < go + field_goal:
        return FourthDownDecision.FIELD_GOAL
    return FourthDownDecision.PUNT


def fourth_down_decision(state: FootballState) -> FourthDownDecision:
    """Return the modal decision for deterministic diagnostics and compatibility."""
    probabilities = fourth_down_probabilities(state)
    choices = (
        FourthDownDecision.GO,
        FourthDownDecision.FIELD_GOAL,
        FourthDownDecision.PUNT,
    )
    values = (probabilities.go, probabilities.field_goal, probabilities.punt)
    return choices[int(np.argmax(values))]


def situation_policy(
    state: FootballState,
    neutral_pass_rate: float,
    *,
    contextual_pass_rate: float | None = None,
) -> SituationPolicy:
    """Translate football state into pass/hurry pressure without double counting context.

    When an empirical down/distance pass rate is supplied, it is already conditioned on
    ordinary football situation and therefore replaces the old additive down/distance
    heuristic. Team neutral identity, score/time script and late-game behavior remain
    separate causal layers.
    """
    if contextual_pass_rate is None:
        neutral = float(np.clip(neutral_pass_rate, 0.30, 0.75))
        distance_pressure = float(np.clip((state.distance - 6.0) * 0.025, -0.08, 0.18))
        down_pressure = {1: -0.02, 2: 0.01, 3: 0.10, 4: 0.12}[state.down]
    else:
        neutral = float(np.clip(contextual_pass_rate, 0.18, 0.90))
        distance_pressure = 0.0
        down_pressure = 0.0

    late = _late_game_pressure(state)
    script_shift = _game_script_pass_shift(state)
    lead_drain = 0.0
    if (
        state.quarter == 4
        and seconds_remaining_in_quarter(state.seconds_remaining) <= 360
        and state.score_margin_for_offense >= 7
    ):
        lead_drain = -0.12

    red_zone = 0.02 if state.yardline_100 >= 80.0 else 0.0
    pass_probability = float(
        np.clip(
            neutral
            + distance_pressure
            + down_pressure
            + script_shift
            + 0.18 * late
            + lead_drain
            + red_zone,
            0.18,
            0.90,
        )
    )
    hurry_probability = float(np.clip(0.05 + 0.75 * late, 0.02, 0.90))
    fourth = fourth_down_decision(state) if state.down == 4 else None
    return SituationPolicy(
        pass_probability=pass_probability,
        hurry_probability=hurry_probability,
        fourth_down=fourth,
    )
