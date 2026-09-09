from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

import numpy as np

from monster.sim.clock import advance_game_clock, regulation_complete
from monster.sim.football_state import (
    FootballState,
    apply_scrimmage_yards,
    kickoff_transition,
    missed_field_goal_transition,
    next_series_distance,
    punt_transition,
    turnover_at_spot,
    turnover_on_downs,
)
from monster.sim.play_kernel import (
    PassResult,
    PlayEvent,
    PlayType,
    TeamIdentity,
    simulate_scrimmage_play,
)
from monster.sim.rules_v13 import (
    PenaltyEvent,
    PenaltySide,
    TryEvent,
    choose_two_point,
    is_safety,
    simulate_penalty,
    simulate_try,
)
from monster.sim.special_teams_v13 import (
    SpecialTeamsEvent,
    simulate_field_goal,
    simulate_kickoff,
    simulate_punt,
)

if TYPE_CHECKING:
    from monster.sim.matchup_kernel import DefensiveUnit


@dataclass
class PlayerBoxScore:
    pass_attempts: int = 0
    completions: int = 0
    passing_yards: float = 0.0
    passing_tds: int = 0
    interceptions: int = 0
    targets: int = 0
    receptions: int = 0
    receiving_yards: float = 0.0
    receiving_tds: int = 0
    rush_attempts: int = 0
    rushing_yards: float = 0.0
    rushing_tds: int = 0
    fumbles_lost: int = 0


@dataclass
class DefensiveBoxScore:
    pressures: int = 0
    sacks: int = 0
    interceptions: int = 0
    tackles: int = 0
    stuffs: int = 0
    forced_fumbles: int = 0


@dataclass(frozen=True)
class GameResultV13:
    final_state: FootballState
    plays: tuple[PlayEvent, ...]
    player_stats: dict[str, PlayerBoxScore]
    drives: int
    defensive_stats: dict[str, DefensiveBoxScore] | None = None
    special_teams_events: tuple[SpecialTeamsEvent, ...] = ()
    try_events: tuple[TryEvent, ...] = ()
    safeties: int = 0
    penalty_events: tuple[PenaltyEvent, ...] = ()


def _add_score(state: FootballState, points: int, *, away_team_id: str) -> FootballState:
    if state.possession == away_team_id:
        return replace(state, away_score=state.away_score + points)
    return replace(state, home_score=state.home_score + points)


def _box(stats: dict[str, PlayerBoxScore], player_id: str | None) -> PlayerBoxScore | None:
    return None if player_id is None else stats.setdefault(player_id, PlayerBoxScore())


def _dbox(stats: dict[str, DefensiveBoxScore], player_id: str | None) -> DefensiveBoxScore | None:
    return None if player_id is None else stats.setdefault(player_id, DefensiveBoxScore())


def _record(
    event: PlayEvent,
    stats: dict[str, PlayerBoxScore],
    defensive_stats: dict[str, DefensiveBoxScore],
) -> None:
    passer = _box(stats, event.passer_id)
    target = _box(stats, event.target_id)
    rusher = _box(stats, event.rusher_id)
    defender = _dbox(defensive_stats, event.primary_defender_id)
    is_throw = event.play_type == PlayType.PASS and event.pass_result not in (
        PassResult.SACK,
        PassResult.SCRAMBLE,
    )
    if is_throw and passer is not None:
        passer.pass_attempts += 1
        if event.pass_result == PassResult.COMPLETE:
            passer.completions += 1
            passer.passing_yards += event.yards
            passer.passing_tds += int(event.touchdown)
        elif event.pass_result == PassResult.INTERCEPTION:
            passer.interceptions += 1
    if target is not None:
        target.targets += 1
        if event.pass_result == PassResult.COMPLETE:
            target.receptions += 1
            target.receiving_yards += event.yards
            target.receiving_tds += int(event.touchdown)
    if rusher is not None:
        rusher.rush_attempts += 1
        rusher.rushing_yards += event.yards
        rusher.rushing_tds += int(event.touchdown)
        rusher.fumbles_lost += int(event.turnover)
    if defender is not None:
        defender.pressures += int(event.pressured)
        defender.sacks += int(event.pass_result == PassResult.SACK)
        defender.interceptions += int(event.pass_result == PassResult.INTERCEPTION)
        defender.stuffs += int(event.stuffed)
        defender.forced_fumbles += int(event.turnover and event.rusher_id is not None)
        defender.tackles += int(event.rusher_id is not None or event.pass_result == PassResult.COMPLETE)


def _kickoff(
    state: FootballState,
    rng: np.random.Generator,
    events: list[SpecialTeamsEvent],
) -> FootballState:
    kick = simulate_kickoff(rng)
    events.append(kick)
    receiving = 30.0 if kick.touchback else float(np.clip(kick.return_yards, 5.0, 45.0))
    return kickoff_transition(state, receiving_yardline_100=receiving, elapsed_seconds=0)


def _apply_penalty(state: FootballState, penalty: PenaltyEvent, elapsed_seconds: int) -> FootballState:
    clock = max(state.seconds_remaining - max(elapsed_seconds, 0), 0)
    if penalty.side == PenaltySide.OFFENSE:
        yardline = max(state.yardline_100 - penalty.yards, 1.0)
        distance = max(state.distance + penalty.yards, 1.0)
        down = min(state.down + int(penalty.loss_of_down), 4)
    else:
        yardline = min(state.yardline_100 + penalty.yards, 99.0)
        gained = penalty.automatic_first_down or penalty.yards >= state.distance
        down = 1 if gained else state.down
        distance = (
            next_series_distance(yardline)
            if gained
            else max(state.distance - penalty.yards, 1.0)
        )
    return replace(
        state,
        seconds_remaining=clock,
        yardline_100=yardline,
        down=down,
        distance=distance,
    )


def simulate_regulation_game(
    away: TeamIdentity,
    home: TeamIdentity,
    *,
    away_defense_strength: float = 1.0,
    home_defense_strength: float = 1.0,
    away_defense: DefensiveUnit | None = None,
    home_defense: DefensiveUnit | None = None,
    seed: int = 1,
    max_plays: int = 260,
    penalty_rate: float = 0.055,
) -> GameResultV13:
    rng = np.random.default_rng(seed)
    state = FootballState(possession=away.team_id, defense=home.team_id)
    stats: dict[str, PlayerBoxScore] = {}
    defensive_stats: dict[str, DefensiveBoxScore] = {}
    plays: list[PlayEvent] = []
    special: list[SpecialTeamsEvent] = []
    tries: list[TryEvent] = []
    penalties: list[PenaltyEvent] = []
    drives = 1
    safeties = 0
    second_half_receiver = home.team_id
    halftime_done = False
    for _ in range(max_plays):
        if regulation_complete(state):
            break
        offense = away if state.possession == away.team_id else home
        if offense.team_id == away.team_id:
            defense_strength, defense = home_defense_strength, home_defense
        else:
            defense_strength, defense = away_defense_strength, away_defense
        before = state
        event = simulate_scrimmage_play(state, offense, defense_strength, rng, defense=defense)
        penalty = simulate_penalty(rng, base_rate=penalty_rate)
        if penalty is not None and event.play_type in (PlayType.RUN, PlayType.PASS):
            penalties.append(penalty)
            state = _apply_penalty(state, penalty, 0)
        else:
            plays.append(event)
            _record(event, stats, defensive_stats)
            if event.play_type == PlayType.PUNT:
                punt = simulate_punt(rng, punter_skill=offense.punt_skill)
                special.append(punt)
                if punt.blocked:
                    state = turnover_at_spot(
                        state,
                        max(state.yardline_100 - 5.0, 1.0),
                        elapsed_seconds=event.elapsed_seconds,
                    )
                else:
                    state = punt_transition(
                        state,
                        gross_yards=punt.kick_distance,
                        return_yards=punt.return_yards,
                        elapsed_seconds=event.elapsed_seconds,
                    )
                drives += 1
            elif event.play_type == PlayType.FIELD_GOAL:
                fg = simulate_field_goal(
                    rng,
                    distance=117.0 - state.yardline_100,
                    kicking_skill=offense.field_goal_skill,
                )
                special.append(fg)
                state = advance_game_clock(state, event.elapsed_seconds)
                if fg.made:
                    state = _kickoff(
                        _add_score(state, 3, away_team_id=away.team_id), rng, special
                    )
                else:
                    state = missed_field_goal_transition(state, elapsed_seconds=0)
                drives += 1
            elif is_safety(yardline_100=state.yardline_100, yards=event.yards):
                state = advance_game_clock(state, event.elapsed_seconds)
                state = replace(state, possession=state.defense, defense=state.possession)
                state = _add_score(state, 2, away_team_id=away.team_id)
                safeties += 1
                state = _kickoff(state, rng, special)
                drives += 1
            elif event.turnover:
                spot = min(max(state.yardline_100 + event.yards, 1.0), 99.0)
                state = turnover_at_spot(state, spot, elapsed_seconds=event.elapsed_seconds)
                drives += 1
            elif event.touchdown:
                state = advance_game_clock(state, event.elapsed_seconds)
                state = _add_score(state, 6, away_team_id=away.team_id)
                if state.possession == away.team_id:
                    margin = state.away_score - state.home_score
                else:
                    margin = state.home_score - state.away_score
                trial = simulate_try(
                    rng,
                    go_for_two=choose_two_point(
                        quarter=state.quarter,
                        seconds_remaining=state.seconds_remaining,
                        score_margin_after_td=margin,
                    ),
                    kicking_skill=offense.field_goal_skill,
                    offense_skill=offense.pass_efficiency,
                    defense_skill=defense_strength,
                )
                tries.append(trial)
                state = _add_score(state, trial.points, away_team_id=away.team_id)
                state = _kickoff(state, rng, special)
                drives += 1
            else:
                state = apply_scrimmage_yards(state, event.yards, event.elapsed_seconds)
                if before.down == 4 and event.yards < before.distance:
                    state = turnover_on_downs(state)
                    drives += 1
        if not halftime_done and before.seconds_remaining > 1800 >= state.seconds_remaining:
            halftime_done = True
            state = FootballState(
                possession=second_half_receiver,
                defense=away.team_id if second_half_receiver == home.team_id else home.team_id,
                quarter=3,
                seconds_remaining=1800,
                yardline_100=30.0,
                away_score=state.away_score,
                home_score=state.home_score,
            )
            drives += 1
    if state.seconds_remaining > 0:
        state = replace(state, seconds_remaining=0, quarter=4)
    return GameResultV13(
        state,
        tuple(plays),
        stats,
        drives,
        defensive_stats,
        tuple(special),
        tuple(tries),
        safeties,
        tuple(penalties),
    )
