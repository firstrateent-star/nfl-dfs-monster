from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import numpy as np

from monster.sim.decision_policy import FourthDownDecision, situation_policy
from monster.sim.football_state import FootballState


class PlayType(StrEnum):
    RUN = "run"
    PASS = "pass"
    PUNT = "punt"
    FIELD_GOAL = "field_goal"


class PassResult(StrEnum):
    COMPLETE = "complete"
    INCOMPLETE = "incomplete"
    SACK = "sack"
    INTERCEPTION = "interception"


class RunLane(StrEnum):
    INSIDE = "inside"
    OUTSIDE = "outside"
    QB = "qb"


@dataclass(frozen=True)
class PlayerIdentity:
    player_id: str
    name: str
    position: str
    usage_weight: float = 1.0
    efficiency: float = 1.0
    explosive: float = 1.0
    turnover_security: float = 1.0


@dataclass(frozen=True)
class TeamIdentity:
    team_id: str
    quarterback: PlayerIdentity
    rushers: tuple[PlayerIdentity, ...]
    receivers: tuple[PlayerIdentity, ...]
    neutral_pass_rate: float = 0.56
    pass_efficiency: float = 1.0
    rush_efficiency: float = 1.0
    pass_protection: float = 1.0
    field_goal_skill: float = 1.0
    punt_skill: float = 1.0


@dataclass(frozen=True)
class PlayEvent:
    play_type: PlayType
    elapsed_seconds: int
    yards: float = 0.0
    passer_id: str | None = None
    target_id: str | None = None
    rusher_id: str | None = None
    pass_result: PassResult | None = None
    run_lane: RunLane | None = None
    touchdown: bool = False
    turnover: bool = False
    field_goal_made: bool = False


def _weighted_player(players: tuple[PlayerIdentity, ...], rng: np.random.Generator) -> PlayerIdentity:
    if not players:
        raise ValueError("player pool cannot be empty")
    weights = np.asarray([max(p.usage_weight, 0.001) for p in players], dtype=float)
    weights /= weights.sum()
    return players[int(rng.choice(len(players), p=weights))]


def choose_play_type(state: FootballState, offense: TeamIdentity, rng: np.random.Generator) -> PlayType:
    policy = situation_policy(state, offense.neutral_pass_rate)
    if state.down == 4:
        if policy.fourth_down == FourthDownDecision.PUNT:
            return PlayType.PUNT
        if policy.fourth_down == FourthDownDecision.FIELD_GOAL:
            return PlayType.FIELD_GOAL
    return PlayType.PASS if rng.random() < policy.pass_probability else PlayType.RUN


def simulate_scrimmage_play(
    state: FootballState,
    offense: TeamIdentity,
    defense_strength: float,
    rng: np.random.Generator,
) -> PlayEvent:
    play_type = choose_play_type(state, offense, rng)
    hurry = situation_policy(state, offense.neutral_pass_rate).hurry_probability
    elapsed = int(np.clip(rng.normal(29.0 - 13.0 * hurry, 7.0), 5.0, 45.0))

    if play_type == PlayType.PUNT:
        return PlayEvent(play_type=play_type, elapsed_seconds=8)
    if play_type == PlayType.FIELD_GOAL:
        distance = 117.0 - state.yardline_100
        make_p = float(np.clip(0.98 - max(distance - 32.0, 0.0) * 0.012, 0.18, 0.98))
        make_p = float(np.clip(make_p * offense.field_goal_skill, 0.05, 0.995))
        return PlayEvent(play_type=play_type, elapsed_seconds=5, field_goal_made=rng.random() < make_p)

    if play_type == PlayType.RUN:
        rusher = _weighted_player(offense.rushers, rng)
        if rusher.position == "QB":
            lane = RunLane.QB
        else:
            lane = RunLane.INSIDE if rng.random() < 0.62 else RunLane.OUTSIDE
        mean = 4.2 * offense.rush_efficiency * rusher.efficiency / max(defense_strength, 0.55)
        yards = float(np.clip(rng.normal(mean, 4.8 * rusher.explosive), -8.0, 60.0))
        fumble_p = float(np.clip(0.012 / max(rusher.turnover_security, 0.5), 0.003, 0.04))
        turnover = rng.random() < fumble_p
        return PlayEvent(
            play_type=play_type,
            elapsed_seconds=elapsed,
            yards=yards,
            rusher_id=rusher.player_id,
            run_lane=lane,
            touchdown=state.yardline_100 + yards >= 100.0,
            turnover=turnover,
        )

    target = _weighted_player(offense.receivers, rng)
    pressure = float(np.clip(0.065 * defense_strength / max(offense.pass_protection, 0.55), 0.025, 0.18))
    if rng.random() < pressure:
        yards = -float(np.clip(rng.normal(6.5, 2.5), 1.0, 15.0))
        return PlayEvent(
            play_type=play_type,
            elapsed_seconds=elapsed,
            yards=yards,
            passer_id=offense.quarterback.player_id,
            pass_result=PassResult.SACK,
        )
    interception_p = float(np.clip(0.024 * defense_strength / max(offense.pass_efficiency, 0.55), 0.008, 0.065))
    if rng.random() < interception_p:
        return PlayEvent(
            play_type=play_type,
            elapsed_seconds=elapsed,
            passer_id=offense.quarterback.player_id,
            target_id=target.player_id,
            pass_result=PassResult.INTERCEPTION,
            turnover=True,
        )
    completion_p = float(np.clip(0.64 * offense.pass_efficiency * target.efficiency / max(defense_strength, 0.65), 0.38, 0.82))
    if rng.random() >= completion_p:
        return PlayEvent(
            play_type=play_type,
            elapsed_seconds=elapsed,
            passer_id=offense.quarterback.player_id,
            target_id=target.player_id,
            pass_result=PassResult.INCOMPLETE,
        )
    yards = float(np.clip(rng.lognormal(mean=2.25, sigma=0.55) * target.explosive, 0.0, 75.0))
    return PlayEvent(
        play_type=play_type,
        elapsed_seconds=elapsed,
        yards=yards,
        passer_id=offense.quarterback.player_id,
        target_id=target.player_id,
        pass_result=PassResult.COMPLETE,
        touchdown=state.yardline_100 + yards >= 100.0,
    )
