from __future__ import annotations

from dataclasses import dataclass

from monster.sim.game_loop_v13 import GameResultV13
from monster.sim.play_kernel import PassResult, PlayType


@dataclass(frozen=True)
class GameAnatomySummary:
    snaps: int
    pass_plays: int
    run_plays: int
    scrambles: int
    sacks: int
    completions: int
    incompletions: int
    interceptions: int
    rush_fumbles_lost: int
    punts: int
    field_goal_attempts: int
    field_goals_made: int
    touchdowns: int
    pressures: int
    stuffs: int
    drives: int
    away_points: int
    home_points: int


def summarize_game(result: GameResultV13) -> GameAnatomySummary:
    plays = result.plays
    return GameAnatomySummary(
        snaps=len(plays),
        pass_plays=sum(play.play_type == PlayType.PASS for play in plays),
        run_plays=sum(play.play_type == PlayType.RUN for play in plays),
        scrambles=sum(play.pass_result == PassResult.SCRAMBLE for play in plays),
        sacks=sum(play.pass_result == PassResult.SACK for play in plays),
        completions=sum(play.pass_result == PassResult.COMPLETE for play in plays),
        incompletions=sum(play.pass_result == PassResult.INCOMPLETE for play in plays),
        interceptions=sum(play.pass_result == PassResult.INTERCEPTION for play in plays),
        rush_fumbles_lost=sum(play.play_type == PlayType.RUN and play.turnover for play in plays),
        punts=sum(play.play_type == PlayType.PUNT for play in plays),
        field_goal_attempts=sum(play.play_type == PlayType.FIELD_GOAL for play in plays),
        field_goals_made=sum(play.field_goal_made for play in plays),
        touchdowns=sum(play.touchdown for play in plays),
        pressures=sum(play.pressured for play in plays),
        stuffs=sum(play.stuffed for play in plays),
        drives=result.drives,
        away_points=result.final_state.away_score,
        home_points=result.final_state.home_score,
    )


def assert_event_conservation(result: GameResultV13) -> None:
    summary = summarize_game(result)
    stats = result.player_stats
    pass_attempts = sum(box.pass_attempts for box in stats.values())
    completions = sum(box.completions for box in stats.values())
    targets = sum(box.targets for box in stats.values())
    receptions = sum(box.receptions for box in stats.values())
    rush_attempts = sum(box.rush_attempts for box in stats.values())
    passing_tds = sum(box.passing_tds for box in stats.values())
    receiving_tds = sum(box.receiving_tds for box in stats.values())
    rushing_tds = sum(box.rushing_tds for box in stats.values())
    interceptions = sum(box.interceptions for box in stats.values())
    fumbles = sum(box.fumbles_lost for box in stats.values())

    throws = summary.pass_plays - summary.sacks - summary.scrambles
    if pass_attempts != throws:
        raise AssertionError("pass attempts must equal throws, excluding sacks and scrambles")
    if completions != summary.completions or receptions != summary.completions:
        raise AssertionError("completion/reception conservation failed")
    if targets != pass_attempts:
        raise AssertionError("every modeled throw must own exactly one target")
    if rush_attempts != summary.run_plays + summary.scrambles:
        raise AssertionError("rush attempts must equal designed runs plus QB scrambles")
    if passing_tds != receiving_tds:
        raise AssertionError("passing and receiving TDs must be the same events")
    if receiving_tds + rushing_tds != summary.touchdowns:
        raise AssertionError("player TD ownership must reconstruct touchdown events")
    if interceptions != summary.interceptions:
        raise AssertionError("interception conservation failed")
    if fumbles != summary.rush_fumbles_lost:
        raise AssertionError("fumble conservation failed")
    if summary.away_points + summary.home_points != 7 * summary.touchdowns + 3 * summary.field_goals_made:
        raise AssertionError("scoreboard must be accounting over scoring events")
