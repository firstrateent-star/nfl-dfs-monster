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


def fourth_down_decision(state: FootballState) -> FourthDownDecision:
    """Bounded football decision scaffold based only on game state."""
    if state.down != 4:
        raise ValueError("fourth_down_decision requires fourth down")

    yards_to_goal = 100.0 - state.yardline_100
    margin = state.score_margin_for_offense
    q_clock = seconds_remaining_in_quarter(state.seconds_remaining)

    # In regular-season overtime a trailing offense is necessarily on the response
    # possession. Ending that possession with a punt loses the game, so it must either
    # kick a field goal that can tie/win (down 1-3) or keep the possession alive.
    if state.quarter == 5 and margin < 0:
        if margin >= -3 and state.yardline_100 >= 58.0 and yards_to_goal <= 42.0:
            return FourthDownDecision.FIELD_GOAL
        return FourthDownDecision.GO

    desperate = state.quarter >= 4 and q_clock <= 360 and margin < 0
    if desperate and state.distance <= 8.0:
        return FourthDownDecision.GO
    if state.yardline_100 >= 60.0 and state.distance <= 2.0:
        return FourthDownDecision.GO
    if state.yardline_100 >= 58.0 and yards_to_goal <= 42.0:
        return FourthDownDecision.FIELD_GOAL
    if state.yardline_100 < 55.0:
        return FourthDownDecision.PUNT
    if state.distance <= 1.0:
        return FourthDownDecision.GO
    return FourthDownDecision.PUNT


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
