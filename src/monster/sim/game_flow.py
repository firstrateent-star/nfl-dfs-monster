from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from monster.sim.clock import seconds_remaining_in_quarter
from monster.sim.football_state import FootballState


class FlowTag(StrEnum):
    SECOND_AND_SHORT = "second_and_short"
    THIRD_AND_SHORT = "third_and_short"
    THIRD_AND_MEDIUM = "third_and_medium"
    THIRD_AND_LONG = "third_and_long"
    THIRD_AND_EXTREME = "third_and_extreme"
    FOURTH_AND_SHORT = "fourth_and_short"
    BACKED_UP = "backed_up"
    MIDFIELD = "midfield"
    OPPONENT_TERRITORY = "opponent_territory"
    HIGH_RED_ZONE = "high_red_zone"
    LOW_RED_ZONE = "low_red_zone"
    GOAL_TO_GO_LIKE = "goal_to_go_like"
    END_FIRST_HALF = "end_first_half"
    TWO_MINUTE_GAME = "two_minute_game"
    FOUR_MINUTE_LEAD = "four_minute_lead"
    LATE_TRAILING = "late_trailing"
    MULTI_SCORE_TRAILING = "multi_score_trailing"
    MULTI_SCORE_LEAD = "multi_score_lead"
    OVERTIME = "overtime"


class StrategicObjective(StrEnum):
    STAY_ON_SCHEDULE = "stay_on_schedule"
    IMPROVE_NEXT_DOWN = "improve_next_down"
    CONVERT_NOW = "convert_now"
    CREATE_EXPLOSIVE = "create_explosive"
    PROTECT_FIELD_POSITION = "protect_field_position"
    REACH_SCORING_RANGE = "reach_scoring_range"
    SCORE_NOW = "score_now"
    STOP_CLOCK = "stop_clock"
    DRAIN_CLOCK = "drain_clock"
    PUNISH_PRESSURE = "punish_pressure"
    SETUP_CONSTRAINT = "setup_constraint"


class PlayFamilyIntent(StrEnum):
    DESIGNED_RUN = "designed_run"
    DROPBACK_PASS = "dropback_pass"
    PLAY_ACTION_PASS = "play_action_pass"
    SCREEN = "screen"
    RPO = "rpo"
    DESIGNED_QB_RUN = "designed_qb_run"
    ROLLOUT_PASS = "rollout_pass"
    SNEAK = "sneak"
    DRAW = "draw"
    TRICK = "trick"
    CLOCK_MANAGEMENT = "clock_management"
    PUNT = "punt"
    FIELD_GOAL = "field_goal"


class PassConceptIntent(StrEnum):
    BEHIND_LOS = "behind_los"
    QUICK = "quick"
    SHORT = "short"
    STICKS = "sticks"
    INTERMEDIATE = "intermediate"
    DEEP = "deep"
    BOMB = "bomb"
    HAIL_MARY = "hail_mary"
    CHECKDOWN_ACCESS = "checkdown_access"
    MAX_PROTECT_VERTICAL = "max_protect_vertical"


class RunConceptIntent(StrEnum):
    INSIDE_ZONE = "inside_zone"
    OUTSIDE_ZONE = "outside_zone"
    POWER_GAP = "power_gap"
    COUNTER = "counter"
    DRAW = "draw"
    SWEEP_TOSS = "sweep_toss"
    SHORT_YARDAGE = "short_yardage"
    QB_SNEAK = "qb_sneak"
    QB_KEEPER = "qb_keeper"
    CONSTRAINT_RUN = "constraint_run"


@dataclass(frozen=True)
class GameFlowState:
    """Non-authoritative strategic context derived before a snap.

    This object intentionally does not choose a play or assign success probability. It
    captures the overlapping football constraints that a future empirical policy can use
    to choose strategic objective and play intent.
    """

    down: int
    distance: float
    yardline: float
    quarter: int
    seconds_remaining: int
    seconds_remaining_in_period: int
    score_margin: int
    yards_to_goal: float
    tags: frozenset[FlowTag]


def derive_game_flow_state(state: FootballState) -> GameFlowState:
    tags: set[FlowTag] = set()
    distance = float(state.distance)
    yardline = float(state.yardline_100)
    yards_to_goal = max(100.0 - yardline, 0.0)
    margin = state.score_margin_for_offense
    period_clock = seconds_remaining_in_quarter(state.seconds_remaining)

    if state.down == 2 and distance <= 2.0:
        tags.add(FlowTag.SECOND_AND_SHORT)
    if state.down == 3:
        if distance <= 2.0:
            tags.add(FlowTag.THIRD_AND_SHORT)
        elif distance <= 6.0:
            tags.add(FlowTag.THIRD_AND_MEDIUM)
        elif distance <= 17.0:
            tags.add(FlowTag.THIRD_AND_LONG)
        else:
            tags.add(FlowTag.THIRD_AND_EXTREME)
    if state.down == 4 and distance <= 2.0:
        tags.add(FlowTag.FOURTH_AND_SHORT)

    if yardline <= 20.0:
        tags.add(FlowTag.BACKED_UP)
    elif 40.0 <= yardline <= 60.0:
        tags.add(FlowTag.MIDFIELD)
    elif yardline >= 60.0:
        tags.add(FlowTag.OPPONENT_TERRITORY)

    if yardline >= 80.0:
        tags.add(FlowTag.HIGH_RED_ZONE)
    if yardline >= 90.0:
        tags.add(FlowTag.LOW_RED_ZONE)
    if distance >= yards_to_goal - 1e-9:
        tags.add(FlowTag.GOAL_TO_GO_LIKE)

    if margin <= -9:
        tags.add(FlowTag.MULTI_SCORE_TRAILING)
    elif margin >= 9:
        tags.add(FlowTag.MULTI_SCORE_LEAD)

    if state.quarter == 2 and period_clock <= 120:
        tags.add(FlowTag.END_FIRST_HALF)
    if state.quarter == 4 and period_clock <= 120:
        tags.add(FlowTag.TWO_MINUTE_GAME)
        if margin < 0:
            tags.add(FlowTag.LATE_TRAILING)
    if state.quarter == 4 and period_clock <= 240 and margin > 0:
        tags.add(FlowTag.FOUR_MINUTE_LEAD)
    if state.quarter == 5:
        tags.add(FlowTag.OVERTIME)

    return GameFlowState(
        down=state.down,
        distance=distance,
        yardline=yardline,
        quarter=state.quarter,
        seconds_remaining=state.seconds_remaining,
        seconds_remaining_in_period=period_clock,
        score_margin=margin,
        yards_to_goal=yards_to_goal,
        tags=frozenset(tags),
    )
