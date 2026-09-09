from __future__ import annotations

from dataclasses import dataclass, replace
from enum import StrEnum
from typing import TYPE_CHECKING

import numpy as np

from monster.sim.decision_policy import FourthDownDecision, situation_policy
from monster.sim.football_state import FootballState
from monster.sim.play_anatomy import (
    CatchpointResult,
    ContactResult,
    QBResponse,
    resolve_catchpoint,
    resolve_qb_response,
    resolve_run_contact,
)

if TYPE_CHECKING:
    from monster.sim.matchup_kernel import DefensiveUnit


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
    SCRAMBLE = "scramble"


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
    run_blocking: float = 1.0
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
    fumbler_id: str | None = None
    primary_defender_id: str | None = None
    pass_result: PassResult | None = None
    run_lane: RunLane | None = None
    touchdown: bool = False
    turnover: bool = False
    field_goal_made: bool = False
    pressured: bool = False
    stuffed: bool = False
    qb_response: QBResponse | None = None
    catchpoint_result: CatchpointResult | None = None
    contact_result: ContactResult | None = None
    air_yards: float = 0.0
    yards_after_catch: float = 0.0
    yards_before_contact: float = 0.0
    yards_after_contact: float = 0.0


def _weighted_player(players, rng):
    weights = np.asarray([max(p.usage_weight, 0.001) for p in players], dtype=float)
    weights /= weights.sum()
    return players[int(rng.choice(len(players), p=weights))]


def _lost_fumble_probability(*, security, contact=None, base_rate):
    mult = {
        ContactResult.STUFF: 1.18,
        ContactResult.TACKLED: 1.0,
        ContactResult.BROKEN_TACKLE: 0.82,
        ContactResult.CLEAN: 0.55,
        None: 1.0,
    }[contact]
    return float(np.clip(base_rate * mult / max(security, 0.5), 0.001, 0.04))


def choose_play_type(state, offense, rng):
    policy = situation_policy(state, offense.neutral_pass_rate)
    if state.down == 4:
        if policy.fourth_down == FourthDownDecision.PUNT:
            return PlayType.PUNT
        if policy.fourth_down == FourthDownDecision.FIELD_GOAL:
            return PlayType.FIELD_GOAL
    return PlayType.PASS if rng.random() < policy.pass_probability else PlayType.RUN


def _outcome_runoff(event: PlayEvent, baseline: int, hurry: float) -> int:
    """Nudge the calibrated whole-snap clock using actual play anatomy."""
    delta = 0
    if event.pass_result == PassResult.INCOMPLETE:
        delta = -4
    elif event.turnover or event.touchdown:
        delta = -2
    elif event.pass_result == PassResult.SACK:
        delta = 3
    elif event.play_type == PlayType.RUN or event.pass_result == PassResult.SCRAMBLE:
        delta = 2
    elif event.pass_result == PassResult.COMPLETE:
        delta = 1
    delta = round(delta * (1.0 - 0.35 * hurry))
    return int(np.clip(baseline + delta, 5, 45))


def _clocked(event, baseline, hurry):
    return replace(event, elapsed_seconds=_outcome_runoff(event, baseline, hurry))


def simulate_scrimmage_play(
    state: FootballState,
    offense: TeamIdentity,
    defense_strength: float,
    rng: np.random.Generator,
    defense: DefensiveUnit | None = None,
) -> PlayEvent:
    play_type = choose_play_type(state, offense, rng)
    policy = situation_policy(state, offense.neutral_pass_rate)
    hurry = policy.hurry_probability
    elapsed = int(np.clip(rng.normal(29.0 - 13.0 * hurry, 7.0), 5.0, 45.0))
    if play_type == PlayType.PUNT:
        return PlayEvent(play_type, 8)
    if play_type == PlayType.FIELD_GOAL:
        distance = 117.0 - state.yardline_100
        make_p = float(
            np.clip(
                (0.98 - max(distance - 32.0, 0.0) * 0.012) * offense.field_goal_skill,
                0.05,
                0.995,
            )
        )
        return PlayEvent(play_type, 5, field_goal_made=rng.random() < make_p)
    if play_type == PlayType.RUN:
        rusher = _weighted_player(offense.rushers, rng)
        lane = RunLane.QB if rusher.position == "QB" else (RunLane.INSIDE if rng.random() < 0.62 else RunLane.OUTSIDE)
        primary = None
        tackling = max(defense_strength, 0.65)
        penetration = float(np.clip(0.18 * defense_strength / max(offense.run_blocking, 0.55), 0.06, 0.42))
        if defense is not None:
            from monster.sim.matchup_kernel import resolve_run_matchup

            m = resolve_run_matchup(rusher, defense, run_blocking=offense.run_blocking)
            primary = m.primary_defender_id
            penetration = m.stuff_probability
            if primary is not None:
                d = next((x for x in defense.front + defense.coverage if x.player_id == primary), None)
                tackling = d.tackling if d else tackling
        a = resolve_run_contact(penetration_probability=penetration, runner_power=max(rusher.efficiency, 0.6), tackling=tackling, explosiveness=rusher.explosive * offense.rush_efficiency, rng=rng)
        yards = a.total_yards
        turnover = rng.random() < _lost_fumble_probability(security=rusher.turnover_security, contact=a.contact, base_rate=0.012)
        td = state.yardline_100 + yards >= 100.0 and not turnover
        e = PlayEvent(play_type, elapsed, yards, rusher_id=rusher.player_id, fumbler_id=rusher.player_id if turnover else None, primary_defender_id=primary, run_lane=lane, touchdown=td, turnover=turnover, stuffed=a.contact == ContactResult.STUFF, contact_result=a.contact, yards_before_contact=a.yards_before_contact, yards_after_contact=a.yards_after_contact)
        return _clocked(e, elapsed, hurry)
    target = _weighted_player(offense.receivers, rng)
    primary = None
    coverage = max(defense_strength, 0.65)
    ball_hawk = max(defense_strength, 0.65)
    if defense is not None:
        from monster.sim.matchup_kernel import resolve_pass_matchup

        m = resolve_pass_matchup(target, defense, pass_protection=offense.pass_protection, quarterback_efficiency=offense.pass_efficiency)
        pressure = m.pressure_probability
        primary = m.primary_defender_id
        if primary is not None:
            d = next((x for x in defense.coverage if x.player_id == primary), None)
            if d:
                coverage = d.coverage
                ball_hawk = d.ball_hawk
    else:
        pressure = float(np.clip(0.065 * defense_strength / max(offense.pass_protection, 0.55), 0.025, 0.18))
    pressured = rng.random() < pressure
    response = resolve_qb_response(pressured=pressured, mobility=offense.quarterback.explosive, pocket_skill=offense.pass_efficiency, rng=rng)
    if response == QBResponse.SACK:
        yards = -float(np.clip(rng.normal(6.5, 2.5), 1.0, 15.0))
        turnover = rng.random() < _lost_fumble_probability(security=offense.quarterback.turnover_security, base_rate=0.010)
        return _clocked(PlayEvent(play_type, elapsed, yards, passer_id=offense.quarterback.player_id, fumbler_id=offense.quarterback.player_id if turnover else None, primary_defender_id=primary, pass_result=PassResult.SACK, turnover=turnover, pressured=pressured, qb_response=response), elapsed, hurry)
    if response == QBResponse.SCRAMBLE:
        a = resolve_run_contact(penetration_probability=0.08, runner_power=max(offense.quarterback.efficiency, 0.6), tackling=coverage, explosiveness=offense.quarterback.explosive, rng=rng)
        turnover = rng.random() < _lost_fumble_probability(security=offense.quarterback.turnover_security, contact=a.contact, base_rate=0.012)
        td = state.yardline_100 + a.total_yards >= 100.0 and not turnover
        return _clocked(PlayEvent(play_type, elapsed, a.total_yards, passer_id=offense.quarterback.player_id, rusher_id=offense.quarterback.player_id, fumbler_id=offense.quarterback.player_id if turnover else None, primary_defender_id=primary, pass_result=PassResult.SCRAMBLE, run_lane=RunLane.QB, touchdown=td, turnover=turnover, pressured=pressured, qb_response=response, contact_result=a.contact, yards_before_contact=a.yards_before_contact, yards_after_contact=a.yards_after_contact), elapsed, hurry)
    air = float(np.clip(rng.normal(8.5 * target.explosive, 6.5), -3.0, 45.0))
    cp = resolve_catchpoint(catch_skill=max(target.efficiency, 0.55), coverage_strength=coverage, ball_hawk=ball_hawk, air_yards=air, rng=rng)
    if cp == CatchpointResult.INTERCEPTION:
        return _clocked(PlayEvent(play_type, elapsed, passer_id=offense.quarterback.player_id, target_id=target.player_id, primary_defender_id=primary, pass_result=PassResult.INTERCEPTION, turnover=True, pressured=pressured, qb_response=response, catchpoint_result=cp, air_yards=air), elapsed, hurry)
    if cp != CatchpointResult.CATCH:
        return _clocked(PlayEvent(play_type, elapsed, passer_id=offense.quarterback.player_id, target_id=target.player_id, primary_defender_id=primary, pass_result=PassResult.INCOMPLETE, pressured=pressured, qb_response=response, catchpoint_result=cp, air_yards=air), elapsed, hurry)
    yac = float(np.clip(rng.lognormal(1.25, 0.65) * target.explosive / max(coverage**0.25, 0.75), 0.0, 55.0))
    yards = max(air, 0.0) + yac
    turnover = rng.random() < _lost_fumble_probability(security=target.turnover_security, contact=ContactResult.TACKLED, base_rate=0.008)
    td = state.yardline_100 + yards >= 100.0 and not turnover
    return _clocked(PlayEvent(play_type, elapsed, yards, passer_id=offense.quarterback.player_id, target_id=target.player_id, fumbler_id=target.player_id if turnover else None, primary_defender_id=primary, pass_result=PassResult.COMPLETE, touchdown=td, turnover=turnover, pressured=pressured, qb_response=response, catchpoint_result=cp, air_yards=air, yards_after_catch=yac), elapsed, hurry)
