from __future__ import annotations

from monster.sim.game_loop_v13 import simulate_regulation_game
from monster.sim.play_kernel import PlayerIdentity, TeamIdentity


def _team(team: str) -> TeamIdentity:
    qb = PlayerIdentity(f"{team}-qb", f"{team} QB", "QB", usage_weight=0.12)
    rb = PlayerIdentity(f"{team}-rb", f"{team} RB", "RB", usage_weight=0.75)
    wr1 = PlayerIdentity(f"{team}-wr1", f"{team} WR1", "WR", usage_weight=0.45)
    wr2 = PlayerIdentity(f"{team}-wr2", f"{team} WR2", "WR", usage_weight=0.30)
    te = PlayerIdentity(f"{team}-te", f"{team} TE", "TE", usage_weight=0.25)
    return TeamIdentity(team, qb, (rb, qb), (wr1, wr2, te))


def test_game_loop_terminates_with_finite_regulation_clock() -> None:
    result = simulate_regulation_game(_team("away"), _team("home"), seed=11)
    assert result.final_state.seconds_remaining == 0
    assert result.final_state.quarter == 4
    assert 20 < len(result.plays) <= 260


def test_player_stats_are_created_by_events_not_posthoc_allocation() -> None:
    result = simulate_regulation_game(_team("away"), _team("home"), seed=22)
    pass_attempts = sum(box.pass_attempts for box in result.player_stats.values())
    targets = sum(box.targets for box in result.player_stats.values())
    completions = sum(box.completions for box in result.player_stats.values())
    receptions = sum(box.receptions for box in result.player_stats.values())
    rush_attempts = sum(box.rush_attempts for box in result.player_stats.values())
    run_plays = sum(play.play_type == "run" for play in result.plays)
    pass_plays = sum(play.play_type == "pass" for play in result.plays)
    assert pass_attempts == pass_plays
    assert targets <= pass_attempts
    assert completions == receptions
    assert rush_attempts == run_plays


def test_touchdowns_are_event_consistent_with_player_stats() -> None:
    result = simulate_regulation_game(_team("away"), _team("home"), seed=33)
    receiving_tds = sum(box.receiving_tds for box in result.player_stats.values())
    rushing_tds = sum(box.rushing_tds for box in result.player_stats.values())
    passing_tds = sum(box.passing_tds for box in result.player_stats.values())
    td_events = sum(play.touchdown for play in result.plays)
    assert receiving_tds + rushing_tds == td_events
    assert passing_tds == receiving_tds


def test_same_seed_reproduces_same_game_world() -> None:
    a = simulate_regulation_game(_team("away"), _team("home"), seed=44)
    b = simulate_regulation_game(_team("away"), _team("home"), seed=44)
    assert a.final_state == b.final_state
    assert a.plays == b.plays
    assert a.player_stats == b.player_stats


def test_score_changes_emerge_from_play_events() -> None:
    result = simulate_regulation_game(_team("away"), _team("home"), seed=55)
    points = result.final_state.away_score + result.final_state.home_score
    touchdowns = sum(play.touchdown for play in result.plays)
    made_fgs = sum(play.field_goal_made for play in result.plays)
    assert points == 7 * touchdowns + 3 * made_fgs
