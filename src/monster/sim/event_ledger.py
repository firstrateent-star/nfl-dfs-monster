from __future__ import annotations

from dataclasses import dataclass

from monster.sim.game_loop_v13 import GameResultV13
from monster.sim.play_kernel import PassResult, PlayType


@dataclass(frozen=True)
class GameAnatomySummary:
    snaps: int; pass_plays: int; run_plays: int; scrambles: int; sacks: int; completions: int; incompletions: int; interceptions: int; fumbles_lost: int; punts: int; field_goal_attempts: int; field_goals_made: int; touchdowns: int; pressures: int; stuffs: int; drives: int; away_points: int; home_points: int; try_attempts: int; try_points: int; safeties: int


def summarize_game(result: GameResultV13) -> GameAnatomySummary:
    p = result.plays
    return GameAnatomySummary(len(p), sum(x.play_type == PlayType.PASS for x in p), sum(x.play_type == PlayType.RUN for x in p), sum(x.pass_result == PassResult.SCRAMBLE for x in p), sum(x.pass_result == PassResult.SACK for x in p), sum(x.pass_result == PassResult.COMPLETE for x in p), sum(x.pass_result == PassResult.INCOMPLETE for x in p), sum(x.pass_result == PassResult.INTERCEPTION for x in p), sum(x.fumbler_id is not None and x.turnover for x in p), sum(x.play_type == PlayType.PUNT for x in p), sum(x.play_type == PlayType.FIELD_GOAL for x in p), sum(1 for x in result.special_teams_events if x.event_type == "field_goal" and x.made), sum(x.touchdown for x in p), sum(x.pressured for x in p), sum(x.stuffed for x in p), result.drives, result.final_state.away_score, result.final_state.home_score, len(result.try_events), sum(x.points for x in result.try_events), result.safeties)


def assert_event_conservation(result: GameResultV13) -> None:
    s = summarize_game(result); stats = result.player_stats
    pass_attempts=sum(x.pass_attempts for x in stats.values()); completions=sum(x.completions for x in stats.values()); targets=sum(x.targets for x in stats.values()); receptions=sum(x.receptions for x in stats.values()); rush_attempts=sum(x.rush_attempts for x in stats.values()); passing_tds=sum(x.passing_tds for x in stats.values()); receiving_tds=sum(x.receiving_tds for x in stats.values()); rushing_tds=sum(x.rushing_tds for x in stats.values()); interceptions=sum(x.interceptions for x in stats.values()); fumbles=sum(x.fumbles_lost for x in stats.values())
    throws=s.pass_plays-s.sacks-s.scrambles
    if pass_attempts != throws: raise AssertionError("pass attempts must equal throws, excluding sacks and scrambles")
    if completions != s.completions or receptions != s.completions: raise AssertionError("completion/reception conservation failed")
    if targets != pass_attempts: raise AssertionError("every modeled throw must own exactly one target")
    if rush_attempts != s.run_plays+s.scrambles: raise AssertionError("rush attempts must equal designed runs plus QB scrambles")
    if passing_tds != receiving_tds: raise AssertionError("passing and receiving TDs must be the same events")
    if receiving_tds+rushing_tds != s.touchdowns: raise AssertionError("player TD ownership must reconstruct touchdown events")
    if interceptions != s.interceptions: raise AssertionError("interception conservation failed")
    if fumbles != s.fumbles_lost: raise AssertionError("fumble conservation failed")
    expected=6*s.touchdowns+3*s.field_goals_made+s.try_points+2*s.safeties
    if s.away_points+s.home_points != expected: raise AssertionError("scoreboard must equal TD, kick, try, and safety events")
    if s.try_attempts != s.touchdowns: raise AssertionError("every regulation offensive touchdown must own one try event")
