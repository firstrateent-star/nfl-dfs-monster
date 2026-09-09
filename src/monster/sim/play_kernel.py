from __future__ import annotations

from dataclasses import dataclass
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
    league_neutral_pass_rate: float = 0.56
    situational_pass_rates: tuple[float, ...] | None = None


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


def _weighted_player(players: tuple[PlayerIdentity, ...], rng: np.random.Generator) -> PlayerIdentity:
    if not players:
        raise ValueError("player pool cannot be empty")
    weights = np.asarray([max(p.usage_weight, 0.001) for p in players], dtype=float)
    weights /= weights.sum()
    return players[int(rng.choice(len(players), p=weights))]


def _lost_fumble_probability(
    *,
    security: float,
    contact: ContactResult | None = None,
    base_rate: float,
) -> float:
    contact_multiplier = {
        ContactResult.STUFF: 1.18,
        ContactResult.TACKLED: 1.0,
        ContactResult.BROKEN_TACKLE: 0.82,
        ContactResult.CLEAN: 0.55,
        None: 1.0,
    }[contact]
    return float(
        np.clip(
            base_rate * contact_multiplier / max(security, 0.5),
            0.001,
            0.04,
        )
    )


def _contextual_pass_rate(state: FootballState, offense: TeamIdentity) -> float | None:
    rates = offense.situational_pass_rates
    if rates is None or len(rates) != 12:
        return None
    bucket = 0 if state.distance <= 3.0 else (1 if state.distance <= 7.0 else 2)
    down = min(max(int(state.down), 1), 4)
    league_context_rate = rates[(down - 1) * 3 + bucket]
    team_identity_delta = offense.neutral_pass_rate - offense.league_neutral_pass_rate
    return float(np.clip(league_context_rate + team_identity_delta, 0.18, 0.90))


def _policy_for_state(state: FootballState, offense: TeamIdentity):
    return situation_policy(
        state,
        offense.neutral_pass_rate,
        contextual_pass_rate=_contextual_pass_rate(state, offense),
    )


def choose_play_type(state: FootballState, offense: TeamIdentity, rng: np.random.Generator) -> PlayType:
    policy = _policy_for_state(state, offense)
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
    defense: DefensiveUnit | None = None,
) -> PlayEvent:
    play_type = choose_play_type(state, offense, rng)
    hurry = _policy_for_state(state, offense).hurry_probability
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
        lane = RunLane.QB if rusher.position == "QB" else (RunLane.INSIDE if rng.random() < 0.62 else RunLane.OUTSIDE)
        primary_defender_id = None
        tackling = max(defense_strength, 0.65)
        penetration_p = float(np.clip(0.18 * defense_strength / max(offense.run_blocking, 0.55), 0.06, 0.42))
        if defense is not None:
            from monster.sim.matchup_kernel import resolve_run_matchup

            matchup = resolve_run_matchup(rusher, defense, run_blocking=offense.run_blocking)
            primary_defender_id = matchup.primary_defender_id
            penetration_p = matchup.stuff_probability
            if primary_defender_id is not None:
                defenders = defense.front + defense.coverage
                defender = next((d for d in defenders if d.player_id == primary_defender_id), None)
                if defender is not None:
                    tackling = defender.tackling
        anatomy = resolve_run_contact(
            penetration_probability=penetration_p,
            runner_power=max(rusher.efficiency, 0.6),
            tackling=tackling,
            explosiveness=rusher.explosive * offense.rush_efficiency,
            rng=rng,
        )
        yards = anatomy.total_yards
        fumble_p = _lost_fumble_probability(
            security=rusher.turnover_security,
            contact=anatomy.contact,
            base_rate=0.012,
        )
        turnover = rng.random() < fumble_p
        touchdown = state.yardline_100 + yards >= 100.0 and not turnover
        return PlayEvent(
            play_type=play_type,
            elapsed_seconds=elapsed,
            yards=yards,
            rusher_id=rusher.player_id,
            fumbler_id=rusher.player_id if turnover else None,
            primary_defender_id=primary_defender_id,
            run_lane=lane,
            touchdown=touchdown,
            turnover=turnover,
            stuffed=anatomy.contact == ContactResult.STUFF,
            contact_result=anatomy.contact,
            yards_before_contact=anatomy.yards_before_contact,
            yards_after_contact=anatomy.yards_after_contact,
        )

    target = _weighted_player(offense.receivers, rng)
    primary_defender_id = None
    coverage_strength = max(defense_strength, 0.65)
    ball_hawk = max(defense_strength, 0.65)
    if defense is not None:
        from monster.sim.matchup_kernel import resolve_pass_matchup

        matchup = resolve_pass_matchup(
            target,
            defense,
            pass_protection=offense.pass_protection,
            quarterback_efficiency=offense.pass_efficiency,
        )
        pressure = matchup.pressure_probability
        primary_defender_id = matchup.primary_defender_id
        if primary_defender_id is not None:
            defender = next((d for d in defense.coverage if d.player_id == primary_defender_id), None)
            if defender is not None:
                coverage_strength = defender.coverage
                ball_hawk = defender.ball_hawk
    else:
        pressure = float(np.clip(0.065 * defense_strength / max(offense.pass_protection, 0.55), 0.025, 0.18))

    pressured = rng.random() < pressure
    response = resolve_qb_response(
        pressured=pressured,
        mobility=offense.quarterback.explosive,
        pocket_skill=offense.pass_efficiency,
        rng=rng,
    )
    if response == QBResponse.SACK:
        yards = -float(np.clip(rng.normal(6.5, 2.5), 1.0, 15.0))
        turnover = rng.random() < _lost_fumble_probability(
            security=offense.quarterback.turnover_security,
            base_rate=0.010,
        )
        return PlayEvent(
            play_type=play_type,
            elapsed_seconds=elapsed,
            yards=yards,
            passer_id=offense.quarterback.player_id,
            fumbler_id=offense.quarterback.player_id if turnover else None,
            primary_defender_id=primary_defender_id,
            pass_result=PassResult.SACK,
            turnover=turnover,
            pressured=pressured,
            qb_response=response,
        )
    if response == QBResponse.SCRAMBLE:
        anatomy = resolve_run_contact(
            penetration_probability=0.08,
            runner_power=max(offense.quarterback.efficiency, 0.6),
            tackling=coverage_strength,
            explosiveness=offense.quarterback.explosive,
            rng=rng,
        )
        turnover = rng.random() < _lost_fumble_probability(
            security=offense.quarterback.turnover_security,
            contact=anatomy.contact,
            base_rate=0.012,
        )
        touchdown = state.yardline_100 + anatomy.total_yards >= 100.0 and not turnover
        return PlayEvent(
            play_type=play_type,
            elapsed_seconds=elapsed,
            yards=anatomy.total_yards,
            passer_id=offense.quarterback.player_id,
            rusher_id=offense.quarterback.player_id,
            fumbler_id=offense.quarterback.player_id if turnover else None,
            primary_defender_id=primary_defender_id,
            pass_result=PassResult.SCRAMBLE,
            run_lane=RunLane.QB,
            touchdown=touchdown,
            turnover=turnover,
            pressured=pressured,
            qb_response=response,
            contact_result=anatomy.contact,
            yards_before_contact=anatomy.yards_before_contact,
            yards_after_contact=anatomy.yards_after_contact,
        )

    air_yards = float(np.clip(rng.normal(8.5 * target.explosive, 6.5), -3.0, 45.0))
    catchpoint = resolve_catchpoint(
        catch_skill=max(target.efficiency, 0.55),
        coverage_strength=coverage_strength,
        ball_hawk=ball_hawk,
        air_yards=air_yards,
        rng=rng,
    )
    if catchpoint == CatchpointResult.INTERCEPTION:
        return PlayEvent(
            play_type=play_type,
            elapsed_seconds=elapsed,
            passer_id=offense.quarterback.player_id,
            target_id=target.player_id,
            primary_defender_id=primary_defender_id,
            pass_result=PassResult.INTERCEPTION,
            turnover=True,
            pressured=pressured,
            qb_response=response,
            catchpoint_result=catchpoint,
            air_yards=air_yards,
        )
    if catchpoint != CatchpointResult.CATCH:
        return PlayEvent(
            play_type=play_type,
            elapsed_seconds=elapsed,
            passer_id=offense.quarterback.player_id,
            target_id=target.player_id,
            primary_defender_id=primary_defender_id,
            pass_result=PassResult.INCOMPLETE,
            pressured=pressured,
            qb_response=response,
            catchpoint_result=catchpoint,
            air_yards=air_yards,
        )
    yac = float(np.clip(rng.lognormal(1.25, 0.65) * target.explosive / max(coverage_strength**0.25, 0.75), 0.0, 55.0))
    yards = max(air_yards, 0.0) + yac
    turnover = rng.random() < _lost_fumble_probability(
        security=target.turnover_security,
        contact=ContactResult.TACKLED,
        base_rate=0.008,
    )
    touchdown = state.yardline_100 + yards >= 100.0 and not turnover
    return PlayEvent(
        play_type=play_type,
        elapsed_seconds=elapsed,
        yards=yards,
        passer_id=offense.quarterback.player_id,
        target_id=target.player_id,
        fumbler_id=target.player_id if turnover else None,
        primary_defender_id=primary_defender_id,
        pass_result=PassResult.COMPLETE,
        touchdown=touchdown,
        turnover=turnover,
        pressured=pressured,
        qb_response=response,
        catchpoint_result=catchpoint,
        air_yards=air_yards,
        yards_after_catch=yac,
    )
