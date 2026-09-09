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


def fourth_down_decision(state: FootballState) -> FourthDownDecision:
    """Bounded football decision scaffold based only on game state.

    This is intentionally policy-shaped rather than optimizer-shaped. Historical NFL decision
    calibration will replace/refine thresholds before promotion. No market or DFS information is
    permitted here.
    """
    if state.down != 4:
        raise ValueError("fourth_down_decision requires fourth down")

    yards_to_goal = 100.0 - state.yardline_100
    margin = state.score_margin_for_offense
    q_clock = seconds_remaining_in_quarter(state.seconds_remaining)
    desperate = state.quarter == 4 and q_clock <= 360 and margin < 0

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


def situation_policy(state: FootballState, neutral_pass_rate: float) -> SituationPolicy:
    """Translate football state into bounded play-selection pressure."""
    neutral = float(np.clip(neutral_pass_rate, 0.30, 0.75))
    distance_pressure = float(np.clip((state.distance - 6.0) * 0.025, -0.08, 0.18))
    down_pressure = {1: -0.02, 2: 0.01, 3: 0.10, 4: 0.12}[state.down]
    late = _late_game_pressure(state)
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
            neutral + distance_pressure + down_pressure + 0.18 * late + lead_drain + red_zone,
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
