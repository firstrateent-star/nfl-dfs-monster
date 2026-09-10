from __future__ import annotations

from dataclasses import dataclass

from monster.sim.game_loop_v13 import GameResultV13
from monster.sim.play_kernel import PassResult, PlayType


@dataclass(frozen=True)
class GameAnatomySummary:
    snaps: int
    scrimmage_plays: int
    pass_plays: int
    pass_attempts: int
    dropbacks: int
    run_plays: int
    rush_attempts: int
    scrambles: int
    sacks: int
    completions: int
    incompletions: int
    interceptions: int
    fumbles_lost: int
    punts: int
    field_goal_attempts: int
    field_goals_made: int
    touchdowns: int
    pressures: int
    stuffs: int
    drives: int
    away_points: int
    home_points: int
    try_attempts: int
    try_points: int
    safeties: int
    went_to_overtime: int
    overtime_touchdowns_without_try: int

    @property
    def completion_percentage(self) -> float:
        return self.completions / self.pass_attempts if self.pass_attempts else 0.0

    @property
    def sack_rate(self) -> float:
        return self.sacks / self.dropbacks if self.dropbacks else 0.0

    @property
    def scramble_rate(self) -> float:
        return self.scrambles / self.dropbacks if self.dropbacks else 0.0

    @property
    def interception_rate(self) -> float:
        return self.interceptions / self.pass_attempts if self.pass_attempts else 0.0


def summarize_game(result: GameResultV13) -> GameAnatomySummary:
    p = result.plays
    pass_plays = sum(x.play_type == PlayType.PASS for x in p)
    run_plays = sum(x.play_type == PlayType.RUN for x in p)
    scrambles = sum(x.pass_result == PassResult.SCRAMBLE for x in p)
    sacks = sum(x.pass_result == PassResult.SACK for x in p)
    pass_attempts = sum(
        x.pass_result in {PassResult.COMPLETE, PassResult.INCOMPLETE, PassResult.INTERCEPTION}
        for x in p
    )
    return GameAnatomySummary(
        snaps=len(p),
        scrimmage_plays=pass_plays + run_plays,
        pass_plays=pass_plays,
        pass_attempts=pass_attempts,
        dropbacks=pass_plays,
        run_plays=run_plays,
        rush_attempts=run_plays + scrambles,
        scrambles=scrambles,
        sacks=sacks,
        completions=sum(x.pass_result == PassResult.COMPLETE for x in p),
        incompletions=sum(x.pass_result == PassResult.INCOMPLETE for x in p),
        interceptions=sum(x.pass_result == PassResult.INTERCEPTION for x in p),
        fumbles_lost=sum(x.fumbler_id is not None and x.turnover for x in p),
        punts=sum(x.play_type == PlayType.PUNT for x in p),
        field_goal_attempts=sum(x.play_type == PlayType.FIELD_GOAL for x in p),
        field_goals_made=sum(
            1
            for x in result.special_teams_events
            if x.event_type == "field_goal" and x.made
        ),
        touchdowns=sum(x.touchdown for x in p),
        pressures=sum(x.pressured for x in p),
        stuffs=sum(x.stuffed for x in p),
        drives=result.drives,
        away_points=result.final_state.away_score,
        home_points=result.final_state.home_score,
        try_attempts=len(result.try_events),
        try_points=sum(x.points for x in result.try_events),
        safeties=result.safeties,
        went_to_overtime=int(result.went_to_overtime),
        overtime_touchdowns_without_try=result.overtime_touchdowns_without_try,
    )


def assert_event_conservation(result: GameResultV13) -> None:
    s = summarize_game(result)
    stats = result.player_stats
    pass_attempts = sum(x.pass_attempts for x in stats.values())
    completions = sum(x.completions for x in stats.values())
    targets = sum(x.targets for x in stats.values())
    receptions = sum(x.receptions for x in stats.values())
    rush_attempts = sum(x.rush_attempts for x in stats.values())
    passing_tds = sum(x.passing_tds for x in stats.values())
    receiving_tds = sum(x.receiving_tds for x in stats.values())
    rushing_tds = sum(x.rushing_tds for x in stats.values())
    interceptions = sum(x.interceptions for x in stats.values())
    fumbles = sum(x.fumbles_lost for x in stats.values())

    if s.snaps != s.scrimmage_plays + s.punts + s.field_goal_attempts:
        raise AssertionError("recorded events must partition into scrimmage, punts, and field goals")
    if s.dropbacks != s.pass_attempts + s.sacks + s.scrambles:
        raise AssertionError("dropbacks must partition into throws, sacks, and scrambles")
    if pass_attempts != s.pass_attempts:
        raise AssertionError("player pass attempts must equal event-derived throws")
    if completions != s.completions or receptions != s.completions:
        raise AssertionError("completion/reception conservation failed")
    if targets != pass_attempts:
        raise AssertionError("every modeled throw must own exactly one target")
    if rush_attempts != s.rush_attempts:
        raise AssertionError("rush attempts must equal designed runs plus QB scrambles")
    if passing_tds != receiving_tds:
        raise AssertionError("passing and receiving TDs must be the same events")
    if receiving_tds + rushing_tds != s.touchdowns:
        raise AssertionError("player TD ownership must reconstruct touchdown events")
    if interceptions != s.interceptions:
        raise AssertionError("interception conservation failed")
    if fumbles != s.fumbles_lost:
        raise AssertionError("fumble conservation failed")

    expected = 6 * s.touchdowns + 3 * s.field_goals_made + s.try_points + 2 * s.safeties
    if s.away_points + s.home_points != expected:
        raise AssertionError("scoreboard must equal TD, kick, try, and safety events")
    if s.overtime_touchdowns_without_try > 1:
        raise AssertionError("at most one game-ending overtime touchdown can omit a try")
    if s.overtime_touchdowns_without_try and not result.went_to_overtime:
        raise AssertionError("a touchdown can omit its try only in an overtime game")
    if s.try_attempts + s.overtime_touchdowns_without_try != s.touchdowns:
        raise AssertionError("every touchdown must own a try unless it ended overtime")
