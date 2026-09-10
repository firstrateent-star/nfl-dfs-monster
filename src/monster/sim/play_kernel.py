from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING

import numpy as np

from monster.sim.decision_policy import FourthDownDecision, situation_policy
from monster.sim.football_state import FootballState
from monster.sim.game_flow import derive_game_flow_state
from monster.sim.game_flow_brain import decide_game_flow
from monster.sim.play_anatomy import (
    CatchpointResult,
    ContactResult,
    QBResponse,
    condition_throw_probabilities,
    resolve_catchpoint,
    resolve_qb_response,
    resolve_run_contact,
)

if TYPE_CHECKING:
    from monster.sim.game_flow_lookup import TeamGameFlowPolicy
    from monster.sim.intent_ecology import IntentEcology
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
    game_flow_policy: TeamGameFlowPolicy | None = None
    intent_ecology: IntentEcology | None = None


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
    pass_depth_category: str | None = None
    run_geometry_category: str | None = None


def _credit_scrimmage_yards(state: FootballState, raw_yards: float) -> float:
    """Return official gain/loss without allowing positive credit beyond the goal line."""
    if raw_yards <= 0.0:
        return float(raw_yards)
    yards_to_goal = max(100.0 - state.yardline_100, 0.0)
    return float(min(raw_yards, yards_to_goal))


def _weighted_player(
    players: tuple[PlayerIdentity, ...], rng: np.random.Generator
) -> PlayerIdentity:
    if not players:
        raise ValueError("player pool cannot be empty")
    weights = np.asarray(
        [max(player.usage_weight, 0.001) for player in players],
        dtype=float,
    )
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


def _contextual_pass_rate(
    state: FootballState,
    offense: TeamIdentity,
) -> float | None:
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


def _dropback_probability(state: FootballState, offense: TeamIdentity) -> float:
    """Choose only top-level run/dropback intent; execution remains downstream."""
    if offense.game_flow_policy is None:
        return _policy_for_state(state, offense).pass_probability
    flow = derive_game_flow_state(state)
    evidence = offense.game_flow_policy.evidence_for(flow)
    return decide_game_flow(flow, evidence).dropback_probability


def choose_play_type(
    state: FootballState,
    offense: TeamIdentity,
    rng: np.random.Generator,
) -> PlayType:
    policy = _policy_for_state(state, offense)
    if state.down == 4:
        if policy.fourth_down == FourthDownDecision.PUNT:
            return PlayType.PUNT
        if policy.fourth_down == FourthDownDecision.FIELD_GOAL:
            return PlayType.FIELD_GOAL
    return (
        PlayType.PASS
        if rng.random() < _dropback_probability(state, offense)
        else PlayType.RUN
    )


def _lane_for_geometry(
    category: str,
    rusher: PlayerIdentity,
    rng: np.random.Generator,
) -> RunLane:
    if category == "qb_sneak" or rusher.position == "QB":
        return RunLane.QB
    if category == "interior":
        return RunLane.INSIDE
    if category in {
        "left_offtackle",
        "right_offtackle",
        "left_edge",
        "right_edge",
    }:
        return RunLane.OUTSIDE
    return RunLane.INSIDE if rng.random() < 0.62 else RunLane.OUTSIDE


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
        make_p = float(
            np.clip(
                0.98 - max(distance - 32.0, 0.0) * 0.012,
                0.18,
                0.98,
            )
        )
        make_p = float(np.clip(make_p * offense.field_goal_skill, 0.05, 0.995))
        return PlayEvent(
            play_type=play_type,
            elapsed_seconds=5,
            field_goal_made=rng.random() < make_p,
        )

    if play_type == PlayType.RUN:
        geometry = None
        if offense.intent_ecology is None:
            rusher = _weighted_player(offense.rushers, rng)
            lane = (
                RunLane.QB
                if rusher.position == "QB"
                else (RunLane.INSIDE if rng.random() < 0.62 else RunLane.OUTSIDE)
            )
        else:
            from monster.sim.intent_ecology import (
                choose_rusher_for_geometry,
                sample_run_geometry_intent,
            )

            flow = derive_game_flow_state(state)
            geometry = sample_run_geometry_intent(
                offense.intent_ecology,
                flow,
                rng=rng,
            )
            if geometry == "qb_sneak":
                rusher = offense.quarterback
            else:
                rusher = choose_rusher_for_geometry(
                    offense.rushers,
                    offense.intent_ecology,
                    geometry,
                    rng,
                )
            lane = _lane_for_geometry(geometry, rusher, rng)

        primary_defender_id = None
        tackling = max(defense_strength, 0.65)
        penetration_p = float(
            np.clip(
                0.18 * defense_strength / max(offense.run_blocking, 0.55),
                0.06,
                0.42,
            )
        )
        matchup_yards_multiplier = float(
            np.clip(
                rusher.efficiency
                * offense.run_blocking
                * offense.rush_efficiency
                / max(defense_strength, 0.60),
                0.50,
                1.65,
            )
        )
        if defense is not None:
            from monster.sim.matchup_kernel import resolve_run_matchup

            matchup = resolve_run_matchup(
                rusher,
                defense,
                run_blocking=offense.run_blocking,
            )
            primary_defender_id = matchup.primary_defender_id
            penetration_p = matchup.stuff_probability
            matchup_yards_multiplier = float(
                np.clip(
                    matchup.yards_multiplier * offense.rush_efficiency,
                    0.50,
                    1.65,
                )
            )
            if primary_defender_id is not None:
                defenders = defense.front + defense.coverage
                defender = next(
                    (
                        item
                        for item in defenders
                        if item.player_id == primary_defender_id
                    ),
                    None,
                )
                if defender is not None:
                    tackling = defender.tackling

        if offense.intent_ecology is None:
            anatomy = resolve_run_contact(
                penetration_probability=penetration_p,
                runner_power=max(rusher.efficiency, 0.6),
                tackling=tackling,
                explosiveness=rusher.explosive * offense.rush_efficiency,
                rng=rng,
            )
            fumble_base = 0.012
        else:
            from monster.sim.resolution_ecology import resolve_run_ecology

            profile = offense.intent_ecology.run_outcomes[geometry]
            anatomy = resolve_run_ecology(
                profile,
                matchup_stuff_probability=penetration_p,
                matchup_yards_multiplier=matchup_yards_multiplier,
                runner_power=max(rusher.efficiency, 0.6),
                tackling=tackling,
                explosiveness=rusher.explosive,
                rng=rng,
            )
            fumble_base = profile.fumble_lost_rate

        raw_yards = anatomy.total_yards
        yards = _credit_scrimmage_yards(state, raw_yards)
        fumble_p = _lost_fumble_probability(
            security=rusher.turnover_security,
            contact=anatomy.contact,
            base_rate=fumble_base,
        )
        turnover = rng.random() < fumble_p
        touchdown = state.yardline_100 + raw_yards >= 100.0 and not turnover
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
            run_geometry_category=geometry,
        )

    depth_category = None
    flow = None
    if offense.intent_ecology is None:
        target = _weighted_player(offense.receivers, rng)
    else:
        from monster.sim.intent_ecology import (
            choose_target_for_depth,
            sample_pass_depth_intent,
        )

        flow = derive_game_flow_state(state)
        depth_category = sample_pass_depth_intent(
            offense.intent_ecology,
            flow,
            quarterback_id=offense.quarterback.player_id,
            rng=rng,
        )
        target = choose_target_for_depth(
            offense.receivers,
            offense.intent_ecology,
            depth_category,
            rng,
        )

    primary_defender_id = None
    coverage_strength = max(defense_strength, 0.65)
    ball_hawk = max(defense_strength, 0.65)
    completion_probability = None
    interception_probability = None
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
        coverage_strength = matchup.coverage_strength
        ball_hawk = matchup.ball_hawk_strength
        completion_probability = matchup.completion_probability
        interception_probability = matchup.interception_probability
    else:
        pressure = float(
            np.clip(
                0.297832 * defense_strength / max(offense.pass_protection, 0.55),
                0.12,
                0.50,
            )
        )

    pressured = rng.random() < pressure
    if (
        offense.intent_ecology is None
        and completion_probability is not None
        and interception_probability is not None
    ):
        completion_probability, interception_probability = condition_throw_probabilities(
            completion_probability=completion_probability,
            interception_probability=interception_probability,
            pressured=pressured,
        )
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
            pass_depth_category=depth_category,
        )
    if response == QBResponse.SCRAMBLE:
        anatomy = resolve_run_contact(
            penetration_probability=0.08,
            runner_power=max(offense.quarterback.efficiency, 0.6),
            tackling=coverage_strength,
            explosiveness=offense.quarterback.explosive,
            rng=rng,
        )
        raw_yards = anatomy.total_yards
        yards = _credit_scrimmage_yards(state, raw_yards)
        turnover = rng.random() < _lost_fumble_probability(
            security=offense.quarterback.turnover_security,
            contact=anatomy.contact,
            base_rate=0.012,
        )
        touchdown = state.yardline_100 + raw_yards >= 100.0 and not turnover
        return PlayEvent(
            play_type=play_type,
            elapsed_seconds=elapsed,
            yards=yards,
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
            pass_depth_category=depth_category,
        )

    if offense.intent_ecology is None:
        air_yards = float(
            np.clip(
                rng.normal(8.5 * target.explosive, 6.5),
                -3.0,
                45.0,
            )
        )
        catchpoint = resolve_catchpoint(
            catch_skill=max(target.efficiency, 0.55),
            coverage_strength=coverage_strength,
            ball_hawk=ball_hawk,
            air_yards=air_yards,
            rng=rng,
            completion_probability=completion_probability,
            interception_probability=interception_probability,
        )
        profile = None
    else:
        from monster.sim.intent_ecology import sample_air_yards
        from monster.sim.resolution_ecology import depth_throw_probabilities

        profile = offense.intent_ecology.pass_outcomes[depth_category]
        air_yards = sample_air_yards(
            depth_category,
            profile,
            yards_to_goal=flow.yards_to_goal,
            rng=rng,
        )
        throw_probabilities = depth_throw_probabilities(
            profile,
            matchup_completion_probability=completion_probability,
            matchup_interception_probability=interception_probability,
            pressured=pressured,
        )
        catchpoint = resolve_catchpoint(
            catch_skill=max(target.efficiency, 0.55),
            coverage_strength=coverage_strength,
            ball_hawk=ball_hawk,
            air_yards=air_yards,
            rng=rng,
            completion_probability=throw_probabilities.completion,
            interception_probability=throw_probabilities.interception,
            completion_probability_includes_depth=True,
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
            pass_depth_category=depth_category,
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
            pass_depth_category=depth_category,
        )

    if profile is None:
        yac = float(
            np.clip(
                rng.lognormal(1.25, 0.65)
                * target.explosive
                / max(coverage_strength**0.25, 0.75),
                0.0,
                55.0,
            )
        )
        raw_yards = max(air_yards, 0.0) + yac
    else:
        from monster.sim.resolution_ecology import (
            completed_pass_yards,
            sample_yac,
        )

        yac = (
            0.0
            if air_yards >= flow.yards_to_goal
            else sample_yac(
                profile,
                receiver_explosiveness=target.explosive,
                coverage_strength=coverage_strength,
                rng=rng,
                air_yards=air_yards,
            )
        )
        raw_yards = completed_pass_yards(air_yards, yac)

    yards = _credit_scrimmage_yards(state, raw_yards)
    turnover = rng.random() < _lost_fumble_probability(
        security=target.turnover_security,
        contact=ContactResult.TACKLED,
        base_rate=0.008,
    )
    touchdown = state.yardline_100 + raw_yards >= 100.0 and not turnover
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
        pass_depth_category=depth_category,
    )
