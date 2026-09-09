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


def _add_score(state: FootballState, points: int, *, away_team_id: str) -> FootballState:
    """Credit points to the actual offensive team, not a hard-coded side label."""
    if state.possession == away_team_id:
        return replace(state, away_score=state.away_score + points)
    return replace(state, home_score=state.home_score + points)


def _box(stats: dict[str, PlayerBoxScore], player_id: str | None) -> PlayerBoxScore | None:
    if player_id is None:
        return None
    return stats.setdefault(player_id, PlayerBoxScore())


def _dbox(stats: dict[str, DefensiveBoxScore], player_id: str | None) -> DefensiveBoxScore | None:
    if player_id is None:
        return None
    return stats.setdefault(player_id, DefensiveBoxScore())


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
            if event.touchdown:
                passer.passing_tds += 1
        elif event.pass_result == PassResult.INTERCEPTION:
            passer.interceptions += 1
    if target is not None:
        target.targets += 1
        if event.pass_result == PassResult.COMPLETE:
            target.receptions += 1
            target.receiving_yards += event.yards
            if event.touchdown:
                target.receiving_tds += 1
    if rusher is not None:
        rusher.rush_attempts += 1
        rusher.rushing_yards += event.yards
        if event.touchdown:
            rusher.rushing_tds += 1
        if event.turnover:
            rusher.fumbles_lost += 1

    if defender is not None:
        if event.pressured:
            defender.pressures += 1
        if event.pass_result == PassResult.SACK:
            defender.sacks += 1
        if event.pass_result == PassResult.INTERCEPTION:
            defender.interceptions += 1
        if event.stuffed:
            defender.stuffs += 1
        if event.turnover and event.rusher_id is not None:
            defender.forced_fumbles += 1
        if event.rusher_id is not None or event.pass_result == PassResult.COMPLETE:
            defender.tackles += 1


def _post_score_kickoff(state: FootballState) -> FootballState:
    return kickoff_transition(state, receiving_yardline_100=30.0, elapsed_seconds=0)


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
) -> GameResultV13:
    """Continuous regulation game: one clock, one field, one possession, one causal event stream."""
    rng = np.random.default_rng(seed)
    state = FootballState(possession=away.team_id, defense=home.team_id)
    stats: dict[str, PlayerBoxScore] = {}
    defensive_stats: dict[str, DefensiveBoxScore] = {}
    plays: list[PlayEvent] = []
    drives = 1
    second_half_receiver = home.team_id
    halftime_done = False

    for _ in range(max_plays):
        if regulation_complete(state):
            break
        offense = away if state.possession == away.team_id else home
        if offense.team_id == away.team_id:
            defense_strength = home_defense_strength
            defense = home_defense
        else:
            defense_strength = away_defense_strength
            defense = away_defense
        before = state
        event = simulate_scrimmage_play(state, offense, defense_strength, rng, defense=defense)
        plays.append(event)
        _record(event, stats, defensive_stats)

        if event.play_type == PlayType.PUNT:
            gross = float(np.clip(rng.normal(45.0 * offense.punt_skill, 6.0), 25.0, 65.0))
            state = punt_transition(
                state,
                gross_yards=gross,
                return_yards=max(rng.normal(8.0, 6.0), 0.0),
                elapsed_seconds=event.elapsed_seconds,
            )
            drives += 1
        elif event.play_type == PlayType.FIELD_GOAL:
            state = advance_game_clock(state, event.elapsed_seconds)
            if event.field_goal_made:
                state = _post_score_kickoff(_add_score(state, 3, away_team_id=away.team_id))
            else:
                state = missed_field_goal_transition(state, elapsed_seconds=0)
            drives += 1
        elif event.turnover:
            spot = min(max(state.yardline_100 + event.yards, 1.0), 99.0)
            state = turnover_at_spot(state, spot, elapsed_seconds=event.elapsed_seconds)
            drives += 1
        elif event.touchdown:
            state = advance_game_clock(state, event.elapsed_seconds)
            state = _post_score_kickoff(_add_score(state, 7, away_team_id=away.team_id))
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
        final_state=state,
        plays=tuple(plays),
        player_stats=stats,
        drives=drives,
        defensive_stats=defensive_stats,
    )
