from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

import numpy as np

from monster.sim.chaos_ecology import (
    DEFAULT_CHAOS_ECOLOGY,
    ChaosEcology,
    ReturnEvent,
    ReturnKind,
    resolve_turnover_return,
    sample_return_yards,
)
from monster.sim.clock import (
    OVERTIME_SECONDS,
    advance_game_clock,
    overtime_complete,
    regulation_complete,
)
from monster.sim.drive_trace import DriveTrace, DriveTraceRecorder
from monster.sim.football_state import (
    FootballState,
    PossessionTerminal,
    apply_scrimmage_yards,
    change_possession,
    missed_field_goal_transition,
    mirror_field,
    next_series_distance,
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
    TryEvent,
    choose_two_point,
    enforce_penalty,
    is_safety,
    overtime_required,
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
    return_yards: float = 0.0
    defensive_tds: int = 0


@dataclass(frozen=True)
class GameResultV13:
    final_state: FootballState
    plays: tuple[PlayEvent, ...]
    player_stats: dict[str, PlayerBoxScore]
    drives: int
    drive_traces: tuple[DriveTrace, ...] = ()
    defensive_stats: dict[str, DefensiveBoxScore] | None = None
    special_teams_events: tuple[SpecialTeamsEvent, ...] = ()
    return_events: tuple[ReturnEvent, ...] = ()
    try_events: tuple[TryEvent, ...] = ()
    safeties: int = 0
    penalty_events: tuple[PenaltyEvent, ...] = ()
    went_to_overtime: bool = False
    overtime_touchdowns_without_try: int = 0


def _add_score_for_team(
    state: FootballState,
    team_id: str,
    points: int,
    *,
    away_team_id: str,
    home_team_id: str,
) -> FootballState:
    if team_id == away_team_id:
        return replace(state, away_score=state.away_score + points)
    if team_id == home_team_id:
        return replace(state, home_score=state.home_score + points)
    raise ValueError("scoring team must match away or home team")


def _add_score(state: FootballState, points: int, *, away_team_id: str) -> FootballState:
    home_team_id = (
        state.home_team_id
        if state.home_team_id is not None
        else (state.defense if state.possession == away_team_id else state.possession)
    )
    return _add_score_for_team(
        state,
        state.possession,
        points,
        away_team_id=away_team_id,
        home_team_id=home_team_id,
    )


def _team_score(state: FootballState, team_id: str, *, away_team_id: str) -> int:
    return state.away_score if team_id == away_team_id else state.home_score


def _opponent_score(state: FootballState, team_id: str, *, away_team_id: str) -> int:
    return state.home_score if team_id == away_team_id else state.away_score


def _team_for_id(team_id: str, away: TeamIdentity, home: TeamIdentity) -> TeamIdentity:
    return away if team_id == away.team_id else home


def _defense_strength_against(
    team_id: str,
    away: TeamIdentity,
    *,
    away_defense_strength: float,
    home_defense_strength: float,
) -> float:
    return home_defense_strength if team_id == away.team_id else away_defense_strength


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
    fumbler = _box(stats, event.fumbler_id)
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
    if fumbler is not None:
        fumbler.fumbles_lost += 1
    if defender is not None:
        defender.pressures += int(event.pressured)
        defender.sacks += int(event.pass_result == PassResult.SACK)
        defender.interceptions += int(event.pass_result == PassResult.INTERCEPTION)
        defender.stuffs += int(event.stuffed)
        defender.forced_fumbles += int(event.fumbler_id is not None)
        defender.tackles += int(
            event.rusher_id is not None or event.pass_result == PassResult.COMPLETE
        )


def _record_return(
    event: ReturnEvent,
    defensive_stats: dict[str, DefensiveBoxScore],
) -> None:
    if event.returner_id is None:
        return
    box = _dbox(defensive_stats, event.returner_id)
    if box is None:
        return
    box.return_yards += event.return_yards
    box.defensive_tds += int(
        event.touchdown and event.kind in {ReturnKind.INTERCEPTION, ReturnKind.FUMBLE}
    )


def _same_possession_first_down(
    state: FootballState,
    *,
    yardline_100: float,
    elapsed_seconds: int = 0,
) -> FootballState:
    spot = float(np.clip(yardline_100, 1.0, 99.0))
    return FootballState(
        possession=state.possession,
        defense=state.defense,
        quarter=state.quarter,
        seconds_remaining=max(state.seconds_remaining - max(int(elapsed_seconds), 0), 0),
        yardline_100=spot,
        down=1,
        distance=next_series_distance(spot),
        away_score=state.away_score,
        home_score=state.home_score,
        away_team_id=state.away_team_id,
        home_team_id=state.home_team_id,
    )


def _kick_state_after_score(state: FootballState, scoring_team: str, other_team: str) -> FootballState:
    return FootballState(
        possession=scoring_team,
        defense=other_team,
        quarter=state.quarter,
        seconds_remaining=state.seconds_remaining,
        yardline_100=35.0,
        away_score=state.away_score,
        home_score=state.home_score,
        away_team_id=state.away_team_id,
        home_team_id=state.home_team_id,
    )


def _score_touchdown(
    state: FootballState,
    *,
    scoring_team: str,
    rng: np.random.Generator,
    tries: list[TryEvent],
    away: TeamIdentity,
    home: TeamIdentity,
    away_defense_strength: float,
    home_defense_strength: float,
    omit_try: bool = False,
) -> FootballState:
    state = _add_score_for_team(
        state,
        scoring_team,
        6,
        away_team_id=away.team_id,
        home_team_id=home.team_id,
    )
    if omit_try:
        return state
    scoring_identity = _team_for_id(scoring_team, away, home)
    defense_strength = _defense_strength_against(
        scoring_team,
        away,
        away_defense_strength=away_defense_strength,
        home_defense_strength=home_defense_strength,
    )
    margin = (
        state.away_score - state.home_score
        if scoring_team == away.team_id
        else state.home_score - state.away_score
    )
    trial = simulate_try(
        rng,
        go_for_two=choose_two_point(
            quarter=state.quarter,
            seconds_remaining=state.seconds_remaining,
            score_margin_after_td=margin,
        ),
        kicking_skill=scoring_identity.field_goal_skill,
        offense_skill=scoring_identity.pass_efficiency,
        defense_skill=defense_strength,
    )
    tries.append(trial)
    return _add_score_for_team(
        state,
        scoring_team,
        trial.points,
        away_team_id=away.team_id,
        home_team_id=home.team_id,
    )


def _kickoff(
    state: FootballState,
    rng: np.random.Generator,
    events: list[SpecialTeamsEvent],
    return_events: list[ReturnEvent],
    tries: list[TryEvent],
    *,
    away: TeamIdentity,
    home: TeamIdentity,
    away_defense_strength: float,
    home_defense_strength: float,
    ecology: ChaosEcology,
    max_return_tds: int = 3,
) -> FootballState:
    """Resolve a kickoff through landing-zone geometry, muffs and rare return scores."""
    current = state
    for _ in range(max_return_tds):
        kicking_team = current.possession
        receiving_team = current.defense
        kick = simulate_kickoff(rng, ecology=ecology)
        if kick.touchback:
            events.append(kick)
            return change_possession(
                current,
                receiving_yardline_100=ecology.kickoff_touchback_yardline,
                elapsed_seconds=0,
            )

        landing = float(kick.return_start_yardline_100 or 1.0)
        if kick.muffed:
            events.append(kick)
            if kick.kicking_team_recovery:
                recovery = ReturnEvent(
                    kind=ReturnKind.KICKOFF,
                    original_offense_team_id=kicking_team,
                    return_team_id=kicking_team,
                    returner_id=None,
                    change_spot_yardline_100=100.0 - landing,
                    return_yards=0.0,
                    receiving_yardline_100=100.0 - landing,
                    touchdown=False,
                    muffed=True,
                    kicking_team_recovery=True,
                )
                return_events.append(recovery)
                return _same_possession_first_down(
                    current,
                    yardline_100=100.0 - landing,
                    elapsed_seconds=4,
                )
            recovery = ReturnEvent(
                kind=ReturnKind.KICKOFF,
                original_offense_team_id=kicking_team,
                return_team_id=receiving_team,
                returner_id=None,
                change_spot_yardline_100=landing,
                return_yards=0.0,
                receiving_yardline_100=landing,
                touchdown=False,
                muffed=True,
            )
            return_events.append(recovery)
            return change_possession(
                current,
                receiving_yardline_100=max(landing, 1.0),
                elapsed_seconds=4,
            )

        distance_to_goal = 100.0 - landing
        return_yards = min(float(kick.return_yards), distance_to_goal)
        receiving = landing + return_yards
        touchdown = receiving >= 100.0 - 1e-9
        events.append(
            replace(kick, return_yards=return_yards, return_touchdown=touchdown)
        )
        return_events.append(
            ReturnEvent(
                kind=ReturnKind.KICKOFF,
                original_offense_team_id=kicking_team,
                return_team_id=receiving_team,
                returner_id=kick.returner_id,
                change_spot_yardline_100=landing,
                return_yards=return_yards,
                receiving_yardline_100=100.0 if touchdown else receiving,
                touchdown=touchdown,
            )
        )
        if not touchdown:
            return change_possession(
                current,
                receiving_yardline_100=receiving,
                elapsed_seconds=6,
            )

        scored = advance_game_clock(current, 6)
        scored = _score_touchdown(
            scored,
            scoring_team=receiving_team,
            rng=rng,
            tries=tries,
            away=away,
            home=home,
            away_defense_strength=away_defense_strength,
            home_defense_strength=home_defense_strength,
        )
        current = _kick_state_after_score(scored, receiving_team, kicking_team)

    # Back-to-back-to-back return TDs are physically possible but so rare that we terminate
    # the recursive branch with a normal touchback rather than risking an unbounded loop.
    forced = SpecialTeamsEvent(
        event_type="kickoff",
        kick_distance=65.0,
        touchback=True,
    )
    events.append(forced)
    return change_possession(
        current,
        receiving_yardline_100=ecology.kickoff_touchback_yardline,
        elapsed_seconds=0,
    )


def _special_return_from_spot(
    before: FootballState,
    *,
    kind: ReturnKind,
    spot_yardline_100: float,
    elapsed_seconds: int,
    rng: np.random.Generator,
    ecology: ChaosEcology,
) -> tuple[FootballState, ReturnEvent]:
    spot = float(np.clip(spot_yardline_100, 1.0, 99.0))
    start = mirror_field(spot)
    distance_to_goal = 100.0 - start
    yards = sample_return_yards(
        mean=ecology.fumble_return_mean,
        sd=ecology.fumble_return_sd,
        zero_rate=min(ecology.fumble_zero_return_rate, 0.40),
        forty_plus_rate=ecology.fumble_40_plus_rate,
        return_skill=1.0,
        rng=rng,
        maximum=distance_to_goal,
    )
    receiving = start + yards
    touchdown = receiving >= 100.0 - 1e-9
    event = ReturnEvent(
        kind=kind,
        original_offense_team_id=before.possession,
        return_team_id=before.defense,
        returner_id=None,
        change_spot_yardline_100=spot,
        return_yards=distance_to_goal if touchdown else yards,
        receiving_yardline_100=100.0 if touchdown else receiving,
        touchdown=touchdown,
    )
    state = change_possession(
        before,
        receiving_yardline_100=min(receiving, 99.0),
        elapsed_seconds=elapsed_seconds,
    )
    return state, event


def _resolve_punt(
    before: FootballState,
    *,
    punt: SpecialTeamsEvent,
    elapsed_seconds: int,
    rng: np.random.Generator,
    events: list[SpecialTeamsEvent],
    return_events: list[ReturnEvent],
    tries: list[TryEvent],
    away: TeamIdentity,
    home: TeamIdentity,
    away_defense_strength: float,
    home_defense_strength: float,
    ecology: ChaosEcology,
    omit_try_on_return_td: bool = False,
) -> tuple[FootballState, PossessionTerminal, bool]:
    if punt.blocked:
        state, ret = _special_return_from_spot(
            before,
            kind=ReturnKind.BLOCKED_PUNT,
            spot_yardline_100=max(before.yardline_100 - 5.0, 1.0),
            elapsed_seconds=elapsed_seconds,
            rng=rng,
            ecology=ecology,
        )
        return_events.append(ret)
        if not ret.touchdown:
            return state, PossessionTerminal.SPECIAL_TEAMS_TURNOVER, False
        scored = advance_game_clock(before, elapsed_seconds)
        scored = _score_touchdown(
            scored,
            scoring_team=ret.return_team_id,
            rng=rng,
            tries=tries,
            away=away,
            home=home,
            away_defense_strength=away_defense_strength,
            home_defense_strength=home_defense_strength,
            omit_try=omit_try_on_return_td,
        )
        if omit_try_on_return_td:
            return scored, PossessionTerminal.SPECIAL_TEAMS_TOUCHDOWN, True
        kick_state = _kick_state_after_score(
            scored, ret.return_team_id, ret.original_offense_team_id
        )
        return (
            _kickoff(
                kick_state,
                rng,
                events,
                return_events,
                tries,
                away=away,
                home=home,
                away_defense_strength=away_defense_strength,
                home_defense_strength=home_defense_strength,
                ecology=ecology,
            ),
            PossessionTerminal.SPECIAL_TEAMS_TOUCHDOWN,
            True,
        )

    if punt.touchback or before.yardline_100 + punt.kick_distance >= 100.0:
        return (
            change_possession(before, receiving_yardline_100=20.0, elapsed_seconds=elapsed_seconds),
            PossessionTerminal.PUNT,
            False,
        )

    physical_end = float(
        np.clip(before.yardline_100 + max(punt.kick_distance, 0.0), 1.0, 99.0)
    )
    receiving_start = mirror_field(physical_end)
    if punt.muffed:
        ret = ReturnEvent(
            kind=ReturnKind.PUNT,
            original_offense_team_id=before.possession,
            return_team_id=before.possession if punt.kicking_team_recovery else before.defense,
            returner_id=punt.returner_id,
            change_spot_yardline_100=physical_end,
            return_yards=0.0,
            receiving_yardline_100=physical_end if punt.kicking_team_recovery else receiving_start,
            touchdown=False,
            muffed=True,
            kicking_team_recovery=punt.kicking_team_recovery,
        )
        return_events.append(ret)
        if punt.kicking_team_recovery:
            return (
                _same_possession_first_down(
                    before,
                    yardline_100=physical_end,
                    elapsed_seconds=elapsed_seconds,
                ),
                PossessionTerminal.SPECIAL_TEAMS_TURNOVER,
                False,
            )
        return (
            change_possession(
                before,
                receiving_yardline_100=receiving_start,
                elapsed_seconds=elapsed_seconds,
            ),
            PossessionTerminal.PUNT,
            False,
        )

    distance_to_goal = 100.0 - receiving_start
    yards = min(float(punt.return_yards), distance_to_goal)
    receiving = receiving_start + yards
    touchdown = receiving >= 100.0 - 1e-9
    ret = ReturnEvent(
        kind=ReturnKind.PUNT,
        original_offense_team_id=before.possession,
        return_team_id=before.defense,
        returner_id=punt.returner_id,
        change_spot_yardline_100=physical_end,
        return_yards=distance_to_goal if touchdown else yards,
        receiving_yardline_100=100.0 if touchdown else receiving,
        touchdown=touchdown,
    )
    return_events.append(ret)
    if not touchdown:
        return (
            change_possession(
                before,
                receiving_yardline_100=receiving,
                elapsed_seconds=elapsed_seconds,
            ),
            PossessionTerminal.PUNT,
            False,
        )

    scored = advance_game_clock(before, elapsed_seconds)
    scored = _score_touchdown(
        scored,
        scoring_team=ret.return_team_id,
        rng=rng,
        tries=tries,
        away=away,
        home=home,
        away_defense_strength=away_defense_strength,
        home_defense_strength=home_defense_strength,
        omit_try=omit_try_on_return_td,
    )
    if omit_try_on_return_td:
        return scored, PossessionTerminal.SPECIAL_TEAMS_TOUCHDOWN, True
    kick_state = _kick_state_after_score(
        scored, ret.return_team_id, ret.original_offense_team_id
    )
    return (
        _kickoff(
            kick_state,
            rng,
            events,
            return_events,
            tries,
            away=away,
            home=home,
            away_defense_strength=away_defense_strength,
            home_defense_strength=home_defense_strength,
            ecology=ecology,
        ),
        PossessionTerminal.SPECIAL_TEAMS_TOUCHDOWN,
        True,
    )


def _resolve_blocked_field_goal(
    before: FootballState,
    *,
    elapsed_seconds: int,
    rng: np.random.Generator,
    events: list[SpecialTeamsEvent],
    return_events: list[ReturnEvent],
    tries: list[TryEvent],
    away: TeamIdentity,
    home: TeamIdentity,
    away_defense_strength: float,
    home_defense_strength: float,
    ecology: ChaosEcology,
    omit_try_on_return_td: bool = False,
) -> tuple[FootballState, PossessionTerminal, bool]:
    state, ret = _special_return_from_spot(
        before,
        kind=ReturnKind.BLOCKED_FIELD_GOAL,
        spot_yardline_100=max(before.yardline_100 - 7.0, 1.0),
        elapsed_seconds=elapsed_seconds,
        rng=rng,
        ecology=ecology,
    )
    return_events.append(ret)
    if not ret.touchdown:
        return state, PossessionTerminal.SPECIAL_TEAMS_TURNOVER, False
    scored = advance_game_clock(before, elapsed_seconds)
    scored = _score_touchdown(
        scored,
        scoring_team=ret.return_team_id,
        rng=rng,
        tries=tries,
        away=away,
        home=home,
        away_defense_strength=away_defense_strength,
        home_defense_strength=home_defense_strength,
        omit_try=omit_try_on_return_td,
    )
    if omit_try_on_return_td:
        return scored, PossessionTerminal.SPECIAL_TEAMS_TOUCHDOWN, True
    kick_state = _kick_state_after_score(scored, ret.return_team_id, ret.original_offense_team_id)
    return (
        _kickoff(
            kick_state,
            rng,
            events,
            return_events,
            tries,
            away=away,
            home=home,
            away_defense_strength=away_defense_strength,
            home_defense_strength=home_defense_strength,
            ecology=ecology,
        ),
        PossessionTerminal.SPECIAL_TEAMS_TOUCHDOWN,
        True,
    )


def _resolve_scrimmage_turnover(
    before: FootballState,
    event: PlayEvent,
    *,
    defense: DefensiveUnit | None,
    rng: np.random.Generator,
    defensive_stats: dict[str, DefensiveBoxScore],
    events: list[SpecialTeamsEvent],
    return_events: list[ReturnEvent],
    tries: list[TryEvent],
    away: TeamIdentity,
    home: TeamIdentity,
    away_defense_strength: float,
    home_defense_strength: float,
    ecology: ChaosEcology,
    omit_try_on_return_td: bool = False,
) -> tuple[FootballState, PossessionTerminal, bool]:
    ret = resolve_turnover_return(
        before,
        event,
        defense=defense,
        rng=rng,
        ecology=ecology,
    )
    return_events.append(ret)
    _record_return(ret, defensive_stats)
    if not ret.touchdown:
        return (
            change_possession(
                before,
                receiving_yardline_100=ret.receiving_yardline_100,
                elapsed_seconds=event.elapsed_seconds,
            ),
            PossessionTerminal.TURNOVER,
            False,
        )

    scored = advance_game_clock(before, event.elapsed_seconds)
    scored = _score_touchdown(
        scored,
        scoring_team=ret.return_team_id,
        rng=rng,
        tries=tries,
        away=away,
        home=home,
        away_defense_strength=away_defense_strength,
        home_defense_strength=home_defense_strength,
        omit_try=omit_try_on_return_td,
    )
    if omit_try_on_return_td:
        return scored, PossessionTerminal.DEFENSIVE_TOUCHDOWN, True
    kick_state = _kick_state_after_score(scored, ret.return_team_id, ret.original_offense_team_id)
    return (
        _kickoff(
            kick_state,
            rng,
            events,
            return_events,
            tries,
            away=away,
            home=home,
            away_defense_strength=away_defense_strength,
            home_defense_strength=home_defense_strength,
            ecology=ecology,
        ),
        PossessionTerminal.DEFENSIVE_TOUCHDOWN,
        True,
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
    chaos_ecology: ChaosEcology = DEFAULT_CHAOS_ECOLOGY,
) -> GameResultV13:
    rng = np.random.default_rng(seed)
    stats: dict[str, PlayerBoxScore] = {}
    defensive_stats: dict[str, DefensiveBoxScore] = {}
    plays: list[PlayEvent] = []
    special: list[SpecialTeamsEvent] = []
    return_events: list[ReturnEvent] = []
    tries: list[TryEvent] = []
    penalties: list[PenaltyEvent] = []
    drive_traces: list[DriveTrace] = []
    safeties = 0

    opening_receiver = away.team_id if rng.random() < 0.5 else home.team_id
    opening_kicker = home.team_id if opening_receiver == away.team_id else away.team_id
    second_half_receiver = opening_kicker
    state = FootballState(
        possession=opening_kicker,
        defense=opening_receiver,
        away_team_id=away.team_id,
        home_team_id=home.team_id,
        yardline_100=35.0,
    )
    state = _kickoff(
        state,
        rng,
        special,
        return_events,
        tries,
        away=away,
        home=home,
        away_defense_strength=away_defense_strength,
        home_defense_strength=home_defense_strength,
        ecology=chaos_ecology,
    )
    drives = 1
    drive_recorder = DriveTraceRecorder(state)
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
            state = enforce_penalty(state, penalty, elapsed_seconds=event.elapsed_seconds)
            drive_recorder.observe_penalty(before, state)
        else:
            plays.append(event)
            _record(event, stats, defensive_stats)
            drive_recorder.observe(before, event)
            if event.play_type == PlayType.PUNT:
                punt = simulate_punt(
                    rng,
                    punter_skill=offense.punt_skill,
                    ecology=chaos_ecology,
                )
                special.append(punt)
                state, terminal, _ = _resolve_punt(
                    before,
                    punt=punt,
                    elapsed_seconds=event.elapsed_seconds,
                    rng=rng,
                    events=special,
                    return_events=return_events,
                    tries=tries,
                    away=away,
                    home=home,
                    away_defense_strength=away_defense_strength,
                    home_defense_strength=home_defense_strength,
                    ecology=chaos_ecology,
                )
                drive_traces.append(drive_recorder.finish(state, terminal, points=0))
                drives += 1
                drive_recorder = DriveTraceRecorder(state)
            elif event.play_type == PlayType.FIELD_GOAL:
                fg = simulate_field_goal(
                    rng,
                    distance=117.0 - state.yardline_100,
                    kicking_skill=offense.field_goal_skill,
                    ecology=chaos_ecology,
                )
                special.append(fg)
                if fg.made:
                    state = advance_game_clock(state, event.elapsed_seconds)
                    scored_state = _add_score(state, 3, away_team_id=away.team_id)
                    drive_traces.append(
                        drive_recorder.finish(scored_state, PossessionTerminal.FIELD_GOAL)
                    )
                    kick_state = _kick_state_after_score(
                        scored_state,
                        offense.team_id,
                        state.defense,
                    )
                    state = _kickoff(
                        kick_state,
                        rng,
                        special,
                        return_events,
                        tries,
                        away=away,
                        home=home,
                        away_defense_strength=away_defense_strength,
                        home_defense_strength=home_defense_strength,
                        ecology=chaos_ecology,
                    )
                elif fg.blocked:
                    state, terminal, _ = _resolve_blocked_field_goal(
                        before,
                        elapsed_seconds=event.elapsed_seconds,
                        rng=rng,
                        events=special,
                        return_events=return_events,
                        tries=tries,
                        away=away,
                        home=home,
                        away_defense_strength=away_defense_strength,
                        home_defense_strength=home_defense_strength,
                        ecology=chaos_ecology,
                    )
                    drive_traces.append(drive_recorder.finish(state, terminal, points=0))
                else:
                    state = missed_field_goal_transition(
                        state,
                        elapsed_seconds=event.elapsed_seconds,
                    )
                    drive_traces.append(
                        drive_recorder.finish(state, PossessionTerminal.MISSED_FIELD_GOAL, points=0)
                    )
                drives += 1
                drive_recorder = DriveTraceRecorder(state)
            elif is_safety(yardline_100=state.yardline_100, yards=event.yards):
                scoring_team = state.defense
                state = advance_game_clock(state, event.elapsed_seconds)
                state = _add_score_for_team(
                    state,
                    scoring_team,
                    2,
                    away_team_id=away.team_id,
                    home_team_id=home.team_id,
                )
                safeties += 1
                drive_traces.append(
                    drive_recorder.finish(state, PossessionTerminal.SAFETY, points=0)
                )
                # After a safety, the team that conceded the safety free-kicks to the scorer.
                state = _kickoff(
                    state,
                    rng,
                    special,
                    return_events,
                    tries,
                    away=away,
                    home=home,
                    away_defense_strength=away_defense_strength,
                    home_defense_strength=home_defense_strength,
                    ecology=chaos_ecology,
                )
                drives += 1
                drive_recorder = DriveTraceRecorder(state)
            elif event.turnover:
                state, terminal, _ = _resolve_scrimmage_turnover(
                    before,
                    event,
                    defense=defense,
                    rng=rng,
                    defensive_stats=defensive_stats,
                    events=special,
                    return_events=return_events,
                    tries=tries,
                    away=away,
                    home=home,
                    away_defense_strength=away_defense_strength,
                    home_defense_strength=home_defense_strength,
                    ecology=chaos_ecology,
                )
                drive_traces.append(drive_recorder.finish(state, terminal, points=0))
                drives += 1
                drive_recorder = DriveTraceRecorder(state)
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
                drive_traces.append(
                    drive_recorder.finish(state, PossessionTerminal.TOUCHDOWN)
                )
                kick_state = _kick_state_after_score(state, offense.team_id, before.defense)
                state = _kickoff(
                    kick_state,
                    rng,
                    special,
                    return_events,
                    tries,
                    away=away,
                    home=home,
                    away_defense_strength=away_defense_strength,
                    home_defense_strength=home_defense_strength,
                    ecology=chaos_ecology,
                )
                drives += 1
                drive_recorder = DriveTraceRecorder(state)
            else:
                state = apply_scrimmage_yards(state, event.yards, event.elapsed_seconds)
                if before.down == 4 and event.yards < before.distance:
                    state = turnover_on_downs(state)
                    drive_traces.append(
                        drive_recorder.finish(state, PossessionTerminal.TURNOVER_ON_DOWNS, points=0)
                    )
                    drives += 1
                    drive_recorder = DriveTraceRecorder(state)
        if not halftime_done and before.seconds_remaining > 1800 >= state.seconds_remaining:
            if drive_recorder.has_activity and drive_recorder.start.possession == before.possession:
                drive_traces.append(
                    drive_recorder.finish(state, PossessionTerminal.HALFTIME)
                )
            halftime_done = True
            halftime_kicker = (
                home.team_id if second_half_receiver == away.team_id else away.team_id
            )
            state = FootballState(
                possession=halftime_kicker,
                defense=second_half_receiver,
                quarter=3,
                seconds_remaining=1800,
                yardline_100=35.0,
                away_score=state.away_score,
                home_score=state.home_score,
                away_team_id=away.team_id,
                home_team_id=home.team_id,
            )
            state = _kickoff(
                state,
                rng,
                special,
                return_events,
                tries,
                away=away,
                home=home,
                away_defense_strength=away_defense_strength,
                home_defense_strength=home_defense_strength,
                ecology=chaos_ecology,
            )
            drives += 1
            drive_recorder = DriveTraceRecorder(state)
    if state.seconds_remaining > 0:
        state = replace(state, seconds_remaining=0, quarter=4)
    if drive_recorder.has_activity:
        drive_traces.append(drive_recorder.finish(state, PossessionTerminal.END_GAME))
    return GameResultV13(
        final_state=state,
        plays=tuple(plays),
        player_stats=stats,
        drives=drives,
        drive_traces=tuple(drive_traces),
        defensive_stats=defensive_stats,
        special_teams_events=tuple(special),
        return_events=tuple(return_events),
        try_events=tuple(tries),
        safeties=safeties,
        penalty_events=tuple(penalties),
    )


def _simulate_regular_season_overtime(
    regulation: GameResultV13,
    away: TeamIdentity,
    home: TeamIdentity,
    *,
    away_defense_strength: float,
    home_defense_strength: float,
    away_defense: DefensiveUnit | None,
    home_defense: DefensiveUnit | None,
    seed: int,
    max_plays: int,
    penalty_rate: float,
    chaos_ecology: ChaosEcology,
) -> GameResultV13:
    """Continue one tied regulation world through the 2026 regular-season OT rules."""
    if not overtime_required(
        regulation.final_state.away_score, regulation.final_state.home_score
    ):
        return regulation

    rng = np.random.default_rng(seed)
    stats = {player_id: replace(box) for player_id, box in regulation.player_stats.items()}
    defensive_stats = {
        player_id: replace(box)
        for player_id, box in (regulation.defensive_stats or {}).items()
    }
    plays = list(regulation.plays)
    special = list(regulation.special_teams_events)
    return_events = list(regulation.return_events)
    tries = list(regulation.try_events)
    penalties = list(regulation.penalty_events)
    drives = regulation.drives
    drive_traces = list(regulation.drive_traces)
    safeties = regulation.safeties
    overtime_touchdowns_without_try = 0

    opening_receiver = away.team_id if rng.random() < 0.5 else home.team_id
    opening_kicker = home.team_id if opening_receiver == away.team_id else away.team_id
    first_team = opening_receiver

    state = FootballState(
        possession=opening_kicker,
        defense=opening_receiver,
        quarter=5,
        seconds_remaining=OVERTIME_SECONDS,
        yardline_100=35.0,
        away_score=regulation.final_state.away_score,
        home_score=regulation.final_state.home_score,
        away_team_id=away.team_id,
        home_team_id=home.team_id,
    )
    returns_before = len(return_events)
    state = _kickoff(
        state,
        rng,
        special,
        return_events,
        tries,
        away=away,
        home=home,
        away_defense_strength=away_defense_strength,
        home_defense_strength=home_defense_strength,
        ecology=chaos_ecology,
    )
    kickoff_td_teams = {
        event.return_team_id
        for event in return_events[returns_before:]
        if event.kind == ReturnKind.KICKOFF and event.touchdown
    }
    drives += 1
    drive_recorder = DriveTraceRecorder(state)
    initial_completed: set[str] = set(kickoff_td_teams)

    for _ in range(max_plays):
        if overtime_complete(state):
            break

        offense = away if state.possession == away.team_id else home
        if offense.team_id == away.team_id:
            defense_strength, defense = home_defense_strength, home_defense
        else:
            defense_strength, defense = away_defense_strength, away_defense

        before = state
        sudden_death = len(initial_completed) >= 2
        event = simulate_scrimmage_play(state, offense, defense_strength, rng, defense=defense)
        penalty = simulate_penalty(rng, base_rate=penalty_rate)

        if penalty is not None and event.play_type in (PlayType.RUN, PlayType.PASS):
            penalties.append(penalty)
            state = enforce_penalty(state, penalty, elapsed_seconds=event.elapsed_seconds)
            drive_recorder.observe_penalty(before, state)
            continue

        plays.append(event)
        _record(event, stats, defensive_stats)
        drive_recorder.observe(before, event)

        if event.play_type == PlayType.PUNT:
            punt = simulate_punt(
                rng,
                punter_skill=offense.punt_skill,
                ecology=chaos_ecology,
            )
            special.append(punt)
            state, terminal, return_td = _resolve_punt(
                before,
                punt=punt,
                elapsed_seconds=event.elapsed_seconds,
                rng=rng,
                events=special,
                return_events=return_events,
                tries=tries,
                away=away,
                home=home,
                away_defense_strength=away_defense_strength,
                home_defense_strength=home_defense_strength,
                ecology=chaos_ecology,
                omit_try_on_return_td=sudden_death,
            )
            drive_traces.append(drive_recorder.finish(state, terminal, points=0))
            drives += 1
            initial_completed.add(offense.team_id)
            if return_td and (sudden_death or len(initial_completed) >= 2):
                overtime_touchdowns_without_try += int(sudden_death)
                break
            drive_recorder = DriveTraceRecorder(state)
            if len(initial_completed) >= 2 and state.away_score != state.home_score:
                break
            continue

        if event.play_type == PlayType.FIELD_GOAL:
            fg = simulate_field_goal(
                rng,
                distance=117.0 - state.yardline_100,
                kicking_skill=offense.field_goal_skill,
                ecology=chaos_ecology,
            )
            special.append(fg)
            if fg.made:
                state = advance_game_clock(state, event.elapsed_seconds)
                state = _add_score(state, 3, away_team_id=away.team_id)
                drive_traces.append(drive_recorder.finish(state, PossessionTerminal.FIELD_GOAL))
                initial_completed.add(offense.team_id)
                if sudden_death or (
                    len(initial_completed) >= 2 and state.away_score != state.home_score
                ):
                    break
                if overtime_complete(state):
                    break
                kick_state = _kick_state_after_score(state, offense.team_id, before.defense)
                state = _kickoff(
                    kick_state,
                    rng,
                    special,
                    return_events,
                    tries,
                    away=away,
                    home=home,
                    away_defense_strength=away_defense_strength,
                    home_defense_strength=home_defense_strength,
                    ecology=chaos_ecology,
                )
                drives += 1
                drive_recorder = DriveTraceRecorder(state)
            elif fg.blocked:
                state, terminal, return_td = _resolve_blocked_field_goal(
                    before,
                    elapsed_seconds=event.elapsed_seconds,
                    rng=rng,
                    events=special,
                    return_events=return_events,
                    tries=tries,
                    away=away,
                    home=home,
                    away_defense_strength=away_defense_strength,
                    home_defense_strength=home_defense_strength,
                    ecology=chaos_ecology,
                    omit_try_on_return_td=True,
                )
                drive_traces.append(drive_recorder.finish(state, terminal, points=0))
                initial_completed.add(offense.team_id)
                if return_td:
                    overtime_touchdowns_without_try += 1
                    break
                drives += 1
                drive_recorder = DriveTraceRecorder(state)
            else:
                state = missed_field_goal_transition(
                    state,
                    elapsed_seconds=event.elapsed_seconds,
                )
                drive_traces.append(
                    drive_recorder.finish(state, PossessionTerminal.MISSED_FIELD_GOAL, points=0)
                )
                drives += 1
                initial_completed.add(offense.team_id)
                drive_recorder = DriveTraceRecorder(state)
                if len(initial_completed) >= 2 and state.away_score != state.home_score:
                    break
            continue

        if is_safety(yardline_100=state.yardline_100, yards=event.yards):
            scoring_team = state.defense
            state = advance_game_clock(state, event.elapsed_seconds)
            state = _add_score_for_team(
                state,
                scoring_team,
                2,
                away_team_id=away.team_id,
                home_team_id=home.team_id,
            )
            safeties += 1
            drive_traces.append(drive_recorder.finish(state, PossessionTerminal.SAFETY, points=0))
            initial_completed.add(offense.team_id)
            if offense.team_id == first_team and len(initial_completed) == 1:
                break
            if sudden_death or (
                len(initial_completed) >= 2 and state.away_score != state.home_score
            ):
                break
            if overtime_complete(state):
                break
            state = _kickoff(
                state,
                rng,
                special,
                return_events,
                tries,
                away=away,
                home=home,
                away_defense_strength=away_defense_strength,
                home_defense_strength=home_defense_strength,
                ecology=chaos_ecology,
            )
            drives += 1
            drive_recorder = DriveTraceRecorder(state)
            continue

        if event.turnover:
            state, terminal, return_td = _resolve_scrimmage_turnover(
                before,
                event,
                defense=defense,
                rng=rng,
                defensive_stats=defensive_stats,
                events=special,
                return_events=return_events,
                tries=tries,
                away=away,
                home=home,
                away_defense_strength=away_defense_strength,
                home_defense_strength=home_defense_strength,
                ecology=chaos_ecology,
                omit_try_on_return_td=True,
            )
            drive_traces.append(drive_recorder.finish(state, terminal, points=0))
            initial_completed.add(offense.team_id)
            if return_td:
                overtime_touchdowns_without_try += 1
                break
            drives += 1
            drive_recorder = DriveTraceRecorder(state)
            if len(initial_completed) >= 2 and state.away_score != state.home_score:
                break
            continue

        if event.touchdown:
            state = advance_game_clock(state, event.elapsed_seconds)
            state = _add_score(state, 6, away_team_id=away.team_id)
            game_ending_td = sudden_death or (
                first_team in initial_completed
                and offense.team_id != first_team
                and _team_score(state, offense.team_id, away_team_id=away.team_id)
                > _opponent_score(state, offense.team_id, away_team_id=away.team_id)
            )
            if game_ending_td:
                drive_traces.append(drive_recorder.finish(state, PossessionTerminal.TOUCHDOWN))
                initial_completed.add(offense.team_id)
                overtime_touchdowns_without_try += 1
                break

            margin = (
                state.away_score - state.home_score
                if offense.team_id == away.team_id
                else state.home_score - state.away_score
            )
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
            drive_traces.append(drive_recorder.finish(state, PossessionTerminal.TOUCHDOWN))
            initial_completed.add(offense.team_id)
            if len(initial_completed) >= 2 and state.away_score != state.home_score:
                break
            if overtime_complete(state):
                break
            kick_state = _kick_state_after_score(state, offense.team_id, before.defense)
            state = _kickoff(
                kick_state,
                rng,
                special,
                return_events,
                tries,
                away=away,
                home=home,
                away_defense_strength=away_defense_strength,
                home_defense_strength=home_defense_strength,
                ecology=chaos_ecology,
            )
            drives += 1
            drive_recorder = DriveTraceRecorder(state)
            continue

        state = apply_scrimmage_yards(state, event.yards, event.elapsed_seconds)
        if before.down == 4 and event.yards < before.distance:
            state = turnover_on_downs(state)
            drive_traces.append(
                drive_recorder.finish(state, PossessionTerminal.TURNOVER_ON_DOWNS, points=0)
            )
            drives += 1
            initial_completed.add(offense.team_id)
            drive_recorder = DriveTraceRecorder(state)
            if len(initial_completed) >= 2 and state.away_score != state.home_score:
                break

    if state.seconds_remaining > 0 and len(plays) - len(regulation.plays) >= max_plays:
        state = replace(state, seconds_remaining=0, quarter=5)
    if drive_recorder.has_activity:
        drive_traces.append(drive_recorder.finish(state, PossessionTerminal.END_GAME))

    return GameResultV13(
        final_state=state,
        plays=tuple(plays),
        player_stats=stats,
        drives=drives,
        drive_traces=tuple(drive_traces),
        defensive_stats=defensive_stats,
        special_teams_events=tuple(special),
        return_events=tuple(return_events),
        try_events=tuple(tries),
        safeties=safeties,
        penalty_events=tuple(penalties),
        went_to_overtime=True,
        overtime_touchdowns_without_try=overtime_touchdowns_without_try,
    )


def simulate_game(
    away: TeamIdentity,
    home: TeamIdentity,
    *,
    away_defense_strength: float = 1.0,
    home_defense_strength: float = 1.0,
    away_defense: DefensiveUnit | None = None,
    home_defense: DefensiveUnit | None = None,
    seed: int = 1,
    max_plays: int = 260,
    max_overtime_plays: int = 80,
    penalty_rate: float = 0.055,
    chaos_ecology: ChaosEcology = DEFAULT_CHAOS_ECOLOGY,
) -> GameResultV13:
    """Simulate one complete 2026 regular-season NFL game, including football oddities."""
    regulation = simulate_regulation_game(
        away,
        home,
        away_defense_strength=away_defense_strength,
        home_defense_strength=home_defense_strength,
        away_defense=away_defense,
        home_defense=home_defense,
        seed=seed,
        max_plays=max_plays,
        penalty_rate=penalty_rate,
        chaos_ecology=chaos_ecology,
    )
    if not overtime_required(
        regulation.final_state.away_score, regulation.final_state.home_score
    ):
        return regulation
    return _simulate_regular_season_overtime(
        regulation,
        away,
        home,
        away_defense_strength=away_defense_strength,
        home_defense_strength=home_defense_strength,
        away_defense=away_defense,
        home_defense=home_defense,
        seed=seed + 2_000_003,
        max_plays=max_overtime_plays,
        penalty_rate=penalty_rate,
        chaos_ecology=chaos_ecology,
    )
